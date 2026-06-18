"""
MODWT-ANN Hybrid Model  —  Multivariate + All PyWavelets Filters
=================================================================
Workflow:
    y  →  MODWT  →  [W1, W2, …, WJ, VJ]
                         ↓  each coefficient
                    Grid Search best ANN
                    (features: coeff lags + raw exog lags)
                         ↓
                    Fit ANN per coefficient
                         ↓
                    Delete first LAGS rows → iMODWT → prepend Y[:LAGS]
                         ↓
                    Train / Test metrics
                         ↓
                    Autoregressive forecast per coefficient
                    (inject future exog at each step)
                         ↓
                    Sum forecasts → ŷ_future  +  MC Dropout intervals

Multivariate approach
---------------------
  Only the TARGET column is decomposed via MODWT.
  Each coefficient's ANN receives:
      [coeff_lag_1 … coeff_lag_L,
       exog0_lag_1 … exog0_lag_L,
       exog1_lag_1 … exog1_lag_L, …]
  as input features, keeping raw (undecomposed) exogenous values.

PyWavelets integration
----------------------
  All 106+ discrete wavelets available in PyWavelets are supported.
  Pass any valid name(s) in  params["wann_wavelets"].
  Run  list_available_wavelets()  to see every option.
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as itertools_product

warnings.filterwarnings("ignore")

# ── Windows DLL fix for PyTorch ───────────────────────────────
try:
    _dll = os.path.join(sys.exec_prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(_dll):
        os.add_dll_directory(_dll)
except Exception:
    pass

import torch
import torch.nn as nn
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════
# SECTION 1 — WAVELET UTILITIES
# ══════════════════════════════════════════════════════════════

def list_available_wavelets():
    """
    Return all discrete wavelet names supported by PyWavelets (106+).

    Families include:
        Haar      : haar
        Daubechies: db1 … db38
        Symlets   : sym2 … sym20
        Coiflets  : coif1 … coif17
        Biorth.   : bior1.1, bior2.2, … bior6.8
        Rev.Bior. : rbio1.1, … rbio6.8
        Dmey      : dmey
    """
    return pywt.wavelist(kind="discrete")


def get_filter(wavelet: str):
    """
    Return (h, g) for ANY discrete wavelet supported by PyWavelets.

    h = high-pass decomposition filter  (detail / wavelet)
    g = low-pass  decomposition filter  (scaling)

    Parameters
    ----------
    wavelet : str
        Any name from  pywt.wavelist(kind='discrete').
        Examples: 'haar', 'db4', 'sym8', 'coif3', 'bior3.3', 'dmey'
    """
    wav = pywt.Wavelet(wavelet)
    g   = np.array(wav.dec_lo, dtype=float)   # low-pass  / scaling
    h   = np.array(wav.dec_hi, dtype=float)   # high-pass / wavelet
    return h, g


def modwt(x: np.ndarray, wavelet: str, level: int):
    """
    Forward Maximal Overlap Discrete Wavelet Transform (MODWT).

    Uses circular convolution — no edge padding, no length restrictions.
    Any PyWavelets discrete wavelet is supported via get_filter().

    Returns
    -------
    details : list of np.ndarray  [W1, W2, …, WJ]
    smooth  : np.ndarray          VJ  (final scaling coefficients)
    """
    h, g = get_filter(wavelet)
    h = h / np.sqrt(2)
    g = g / np.sqrt(2)

    N       = len(x)
    v       = x.copy().astype(float)
    details = []

    for j in range(1, level + 1):
        L  = len(h)
        up = 2 ** (j - 1)
        W  = np.zeros(N)
        V  = np.zeros(N)
        for t in range(N):
            for k in range(L):
                idx   = (t - up * k) % N
                W[t] += h[k] * v[idx]
                V[t] += g[k] * v[idx]
        details.append(W)
        v = V

    return details, v


def imodwt(details: list, smooth: np.ndarray, wavelet: str):
    """
    Inverse MODWT.
    Guarantees:  imodwt(modwt(y, w, J), w) ≈ y   (max error < 1e-10).
    """
    h, g = get_filter(wavelet)
    h = h / np.sqrt(2)
    g = g / np.sqrt(2)

    N = len(smooth)
    v = smooth.copy().astype(float)

    for j in range(len(details), 0, -1):
        W     = details[j - 1]
        L     = len(h)
        up    = 2 ** (j - 1)
        x_rec = np.zeros(N)
        for t in range(N):
            for k in range(L):
                idx       = (t + up * k) % N
                x_rec[t] += h[k] * W[idx] + g[k] * v[idx]
        v = x_rec

    return v


def reconstruct_train_with_original_prefix(fitted_list, wavelet, lags, y_original_train):
    """
    1. Drop first `lags` from each coefficient's fitted array.
    2. Apply iMODWT on trimmed arrays → length (N_train − lags).
    3. Prepend original Y[:lags]       → full length N_train.
    """
    trimmed = [np.delete(f, range(lags)) for f in fitted_list]
    recon   = imodwt(trimmed[:-1], trimmed[-1], wavelet)
    return np.concatenate([y_original_train[:lags], recon])


# ══════════════════════════════════════════════════════════════
# SECTION 2 — ANN ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class ANNModel(nn.Module):
    """
    Two-hidden-layer ANN with MC Dropout for uncertainty estimation.
    Input  : (batch, input_dim)  — flattened lags  [+ exog lags]
    Output : (batch,)            — single-step prediction (scaled)
    """
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2,
                 dropout_1, dropout_2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_layer_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — LAG FEATURE BUILDER  (coeff + optional exog)
# ══════════════════════════════════════════════════════════════

def _build_lag_features(series: np.ndarray,
                         exog:   np.ndarray,
                         lags:   int):
    """
    Build (X, y) lag matrix for one coefficient series.

    Parameters
    ----------
    series : (N,)         — coefficient values
    exog   : (N, n_exog)  — raw exogenous values aligned to series, or None
    lags   : int

    Feature layout per sample i  (oldest lag first):
        [coeff[i-L], …, coeff[i-1],
         exog0[i-L], …, exog0[i-1],
         exog1[i-L], …, exog1[i-1], …]

    Returns
    -------
    X : (N-lags, input_dim)  float32
    y : (N-lags,)            float32
    """
    n_exog = exog.shape[1] if exog is not None else 0
    X, y   = [], []

    for i in range(lags, len(series)):
        coeff_part = series[i - lags: i]                    # (lags,)
        if n_exog:
            # Stack each exog column's lags sequentially
            exog_part = exog[i - lags: i].flatten()         # (lags * n_exog,)
            row = np.concatenate([coeff_part, exog_part])
        else:
            row = coeff_part
        X.append(row)
        y.append(series[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — GRID SEARCH  (per coefficient)
# ══════════════════════════════════════════════════════════════

def _grid_search_ann(series, exog, params, device, lags):
    """
    Grid-search ANN hyperparameters for one wavelet coefficient series.

    Parameters
    ----------
    series : (N,)         training coefficient values
    exog   : (N, n_exog)  aligned exogenous values, or None
    params : dict         UI hyperparameter dict
    device : torch.device
    lags   : int

    Returns
    -------
    best_p : dict of best hyperparameters
    """

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, dec=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, dec))
            v += step
        return vals

    # Default fallback in case series is too short or search fails
    default = {
        "hidden_layer_1": params.get("hidden_layer_1", 16),
        "hidden_layer_2": params.get("hidden_layer_2", 8),
        "dropout_1"     : params.get("dropout_1",      0.1),
        "dropout_2"     : params.get("dropout_2",      0.1),
        "learning_rate" : params.get("learning_rate",  0.001),
        "weight_decay"  : params.get("weight_decay",   1e-5),
        "huber_delta"   : params.get("huber_delta",    1.0),
        "patience"      : params.get("patience",       30),
        "epochs"        : params.get("epochs",         200),
    }

    X, y = _build_lag_features(series, exog, lags)
    if len(X) < 4:
        return default

    input_dim = X.shape[1]

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    X_s  = x_sc.fit_transform(X)
    y_s  = y_sc.fit_transform(y.reshape(-1, 1)).flatten()

    val_sp          = max(1, int(0.8 * len(X_s)))
    Xtr_s, Xva_s   = X_s[:val_sp], X_s[val_sp:]
    ytr_s, yva_s   = y_s[:val_sp], y_s[val_sp:]

    grid = {
        "hidden_layer_1": irange(params.get("ann_hidden_layer_1_min",  8),
                                  params.get("ann_hidden_layer_1_max",  32),
                                  params.get("ann_hidden_layer_1_step", 8)),
        "hidden_layer_2": irange(params.get("ann_hidden_layer_2_min",  4),
                                  params.get("ann_hidden_layer_2_max",  16),
                                  params.get("ann_hidden_layer_2_step", 4)),
        "dropout_1"     : frange(params.get("ann_dropout_1_min",  0.1),
                                  params.get("ann_dropout_1_max",  0.3),
                                  params.get("ann_dropout_1_step", 0.1), 3),
        "dropout_2"     : frange(params.get("ann_dropout_2_min",  0.1),
                                  params.get("ann_dropout_2_max",  0.4),
                                  params.get("ann_dropout_2_step", 0.1), 3),
        "learning_rate" : frange(params.get("ann_learning_rate_min",  0.001),
                                  params.get("ann_learning_rate_max",  0.005),
                                  params.get("ann_learning_rate_step", 0.002), 6),
        "weight_decay"  : frange(params.get("ann_weight_decay_min",  1e-5),
                                  params.get("ann_weight_decay_max",  5e-4),
                                  params.get("ann_weight_decay_step", 1e-4), 7),
        "huber_delta"   : frange(params.get("ann_huber_delta_min",  0.5),
                                  params.get("ann_huber_delta_max",  1.5),
                                  params.get("ann_huber_delta_step", 0.5), 2),
        "patience"      : irange(params.get("ann_patience_min",  20),
                                  params.get("ann_patience_max",  50),
                                  params.get("ann_patience_step", 10)),
        "epochs"        : irange(params.get("ann_epochs_min",  100),
                                  params.get("ann_epochs_max",  300),
                                  params.get("ann_epochs_step", 100)),
    }

    combos = [dict(zip(grid, v)) for v in itertools_product(*grid.values())]

    Xtr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_s, dtype=torch.float32, device=device)
    Xva = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    yva = torch.tensor(yva_s, dtype=torch.float32, device=device)

    best_val, best_p = float("inf"), None

    for p in combos:
        try:
            torch.manual_seed(SEED)
            m    = ANNModel(input_dim, p["hidden_layer_1"], p["hidden_layer_2"],
                            p["dropout_1"], p["dropout_2"]).to(device)
            crit = nn.HuberLoss(delta=p["huber_delta"])
            opt  = torch.optim.Adam(m.parameters(), lr=p["learning_rate"],
                                     weight_decay=p["weight_decay"])
            best_ep, pat = float("inf"), 0

            for _ in range(p["epochs"]):
                m.train()
                opt.zero_grad()
                crit(m(Xtr), ytr).backward()
                opt.step()

                m.eval()
                with torch.no_grad():
                    vl = crit(m(Xva), yva).item()
                if vl < best_ep:
                    best_ep, pat = vl, 0
                else:
                    pat += 1
                    if pat >= p["patience"]:
                        break

            if best_ep < best_val:
                best_val, best_p = best_ep, p.copy()

        except Exception:
            continue

    return best_p if best_p is not None else default


# ══════════════════════════════════════════════════════════════
# SECTION 5 — FIT ANN PER COEFFICIENT
# ══════════════════════════════════════════════════════════════

def _fit_ann_on_coefficient(train_series, exog_train, params, device,
                             lags, best_p, coef_name, log):
    """
    Fit one ANN on a single coefficient training series.

    Parameters
    ----------
    train_series : (N_train,)
    exog_train   : (N_train, n_exog)  or  None
    lags         : int
    best_p       : dict  — hyperparameters from grid search

    Returns
    -------
    fitted_full    : (N_train,)  — first `lags` = actual coeff; rest = ANN preds
    x_sc           : fitted StandardScaler for X
    y_sc           : fitted StandardScaler for y
    model          : trained ANNModel
    last_coeff_win : (lags,)          — last `lags` coeff values (for forecasting)
    last_exog_win  : (lags, n_exog)   — last `lags` exog rows   (for forecasting), or None
    """
    X, y = _build_lag_features(train_series, exog_train, lags)

    if len(X) < 4:
        log(f"  {coef_name} → SKIPPED (too short)")
        zeros = np.zeros(len(train_series))
        return zeros, None, None, None, None, None

    input_dim = X.shape[1]

    x_sc = StandardScaler()
    y_sc = StandardScaler()
    X_s  = x_sc.fit_transform(X)
    y_s  = y_sc.fit_transform(y.reshape(-1, 1)).flatten()

    torch.manual_seed(SEED)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_s, dtype=torch.float32, device=device)

    model = ANNModel(input_dim,
                     best_p["hidden_layer_1"], best_p["hidden_layer_2"],
                     best_p["dropout_1"],      best_p["dropout_2"]).to(device)

    opt = torch.optim.Adam(model.parameters(),
                            lr=best_p["learning_rate"],
                            weight_decay=best_p["weight_decay"])
    lf  = nn.HuberLoss(delta=best_p["huber_delta"])

    best_loss, best_state, counter = float("inf"), None, 0

    for _ in range(best_p["epochs"]):
        model.train()
        opt.zero_grad()
        loss = lf(model(X_t), y_t)
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= best_p["patience"]:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    log(f"  {coef_name} → ANN({best_p['hidden_layer_1']},{best_p['hidden_layer_2']}) "
        f"| input_dim={input_dim} | Loss: {best_loss:.4f}")

    with torch.no_grad():
        fitted_s = model(X_t).cpu().numpy()

    fitted      = y_sc.inverse_transform(fitted_s.reshape(-1, 1)).flatten()
    fitted_full = np.concatenate([train_series[:lags], fitted])   # (N_train,)

    last_coeff_win = train_series[-lags:].copy()                  # (lags,)
    last_exog_win  = (exog_train[-lags:].copy()
                      if exog_train is not None else None)        # (lags, n_exog) or None

    return fitted_full, x_sc, y_sc, model, last_coeff_win, last_exog_win


# ══════════════════════════════════════════════════════════════
# SECTION 6 — AUTOREGRESSIVE FORECAST PER COEFFICIENT
# ══════════════════════════════════════════════════════════════

def _build_input_row(coeff_win, exog_win):
    """Concatenate coefficient and exog windows into one feature row."""
    if exog_win is not None:
        return np.concatenate([coeff_win, exog_win.flatten()])
    return coeff_win.copy()


def _forecast_coefficient(model, x_sc, y_sc,
                           last_coeff_win, last_exog_win,
                           future_exog, steps, device):
    """
    Deterministic autoregressive forecast for one wavelet coefficient.

    Parameters
    ----------
    last_coeff_win : (lags,)
    last_exog_win  : (lags, n_exog)  or  None
    future_exog    : (steps, n_exog) or  None

    Returns
    -------
    preds : (steps,)
    """
    model.eval()
    coeff_win = last_coeff_win.copy()
    exog_win  = last_exog_win.copy() if last_exog_win is not None else None
    n_exog    = exog_win.shape[1] if exog_win is not None else 0
    preds     = []

    for t in range(steps):
        row = _build_input_row(coeff_win, exog_win)
        x_s = x_sc.transform(row.reshape(1, -1).astype(np.float32))
        x_t = torch.tensor(x_s, dtype=torch.float32, device=device)

        with torch.no_grad():
            p_s = model(x_t).cpu().numpy()

        pred = float(y_sc.inverse_transform(p_s.reshape(-1, 1))[0, 0])
        preds.append(pred)

        # Roll coefficient window
        coeff_win = np.roll(coeff_win, -1)
        coeff_win[-1] = pred

        # Roll exog window — inject known future exog
        if n_exog and future_exog is not None:
            exog_win = np.roll(exog_win, -1, axis=0)
            exog_win[-1] = future_exog[t]   # (n_exog,)

    return np.array(preds)


def _mc_forecast_coefficient(model, x_sc, y_sc,
                               last_coeff_win, last_exog_win,
                               future_exog, steps, device,
                               n_samples=300):
    """
    Monte Carlo Dropout forecast for one wavelet coefficient.

    Keeps model.train() active to sample different dropout masks each run.

    Returns
    -------
    mean : (steps,)
    std  : (steps,)
    """
    model.train()   # MC Dropout — keep dropout active
    all_runs = []

    with torch.no_grad():
        for _ in range(n_samples):
            coeff_win = last_coeff_win.copy()
            exog_win  = last_exog_win.copy() if last_exog_win is not None else None
            n_exog    = exog_win.shape[1] if exog_win is not None else 0
            run       = []

            for t in range(steps):
                row = _build_input_row(coeff_win, exog_win)
                x_s = x_sc.transform(row.reshape(1, -1).astype(np.float32))
                x_t = torch.tensor(x_s, dtype=torch.float32, device=device)
                p_s = model(x_t).cpu().numpy()
                pred = float(y_sc.inverse_transform(p_s.reshape(-1, 1))[0, 0])
                run.append(pred)

                coeff_win = np.roll(coeff_win, -1)
                coeff_win[-1] = pred

                if n_exog and future_exog is not None:
                    exog_win = np.roll(exog_win, -1, axis=0)
                    exog_win[-1] = future_exog[t]

            all_runs.append(run)

    model.eval()
    arr = np.array(all_runs)   # (n_samples, steps)
    return arr.mean(axis=0), arr.std(axis=0)


# ══════════════════════════════════════════════════════════════
# SECTION 7 — SINGLE (wavelet, level) PIPELINE
# ══════════════════════════════════════════════════════════════

def _run_one_combination(y_values, y_original, time_labels,
                          wavelet, level, test_size,
                          exog_full,
                          params, device, log):
    """
    Complete MODWT-ANN pipeline for one (wavelet, level) pair.

    Parameters
    ----------
    y_values   : (N,)         target series
    y_original : (N,)         copy of target (before any transforms)
    time_labels: list (N,)    time index labels
    exog_full  : (N, n_exog)  exogenous data aligned to y, or None
    test_size  : int

    Returns
    -------
    dict with metrics and all stored model objects, or None on failure.
    """
    LAGS = params.get("ann_lags") or params.get("lags", 3)

    try:
        # ── MODWT ──────────────────────────────────────────────
        details, smooth = modwt(y_values, wavelet, level)
        coef_names      = [f"W{j+1}" for j in range(level)] + [f"V{level}"]

        # Verify perfect reconstruction before proceeding
        y_rec   = imodwt(details, smooth, wavelet)
        rec_err = np.max(np.abs(y_values - y_rec))
        if rec_err > 1e-4:
            log(f"  ⚠ Reconstruction error {rec_err:.2e} for {wavelet} L={level} — skipping")
            return None

        # ── Train / Test split ──────────────────────────────────
        all_coefs     = details + [smooth]
        train_coefs   = [c[:-test_size] for c in all_coefs]
        y_train_orig  = y_original[:-test_size]
        y_test_orig   = y_original[-test_size:]
        exog_train    = exog_full[:-test_size] if exog_full is not None else None
        exog_test     = exog_full[-test_size:] if exog_full is not None else None

        # ── Fit one ANN per coefficient ─────────────────────────
        fitted_list  = []
        models_store = []
        ann_configs  = {}

        for name, train in zip(coef_names, train_coefs):
            best_p = _grid_search_ann(train, exog_train, params, device, LAGS)
            ann_configs[name] = best_p

            out = _fit_ann_on_coefficient(
                train, exog_train, params, device, LAGS, best_p, name, log
            )
            fitted_full, x_sc, y_sc, model, last_coeff_win, last_exog_win = out
            fitted_list.append(fitted_full)
            models_store.append(
                (x_sc, y_sc, model, last_coeff_win, last_exog_win, name)
            )

        # ── Train-set reconstruction ────────────────────────────
        train_pred = reconstruct_train_with_original_prefix(
            fitted_list, wavelet, LAGS, y_train_orig
        )

        # ── Test prediction  (sum of per-coeff forecasts) ───────
        test_preds_per_coef = []
        for (x_sc, y_sc, model, last_coeff_win, last_exog_win, name) in models_store:
            if model is None:
                test_preds_per_coef.append(np.zeros(test_size))
                continue
            fc = _forecast_coefficient(
                model, x_sc, y_sc,
                last_coeff_win, last_exog_win,
                exog_test, test_size, device,
            )
            test_preds_per_coef.append(fc)

        test_pred = np.sum(test_preds_per_coef, axis=0)

        # ── Metrics ─────────────────────────────────────────────
        n_tr      = min(len(y_train_orig), len(train_pred))
        actual_tr = y_train_orig[:n_tr]
        pred_tr   = train_pred[:n_tr]

        def mape_fn(a, p):
            return float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

        train_times = time_labels[:-test_size][:n_tr]
        test_times  = time_labels[-test_size:]

        return {
            "wavelet"     : wavelet,
            "level"       : level,
            "rmse_train"  : float(root_mean_squared_error(actual_tr, pred_tr)),
            "mae_train"   : float(mean_absolute_error(actual_tr, pred_tr)),
            "mape_train"  : mape_fn(actual_tr, pred_tr),
            "rmse_test"   : float(root_mean_squared_error(y_test_orig, test_pred)),
            "mae_test"    : float(mean_absolute_error(y_test_orig, test_pred)),
            "mape_test"   : mape_fn(y_test_orig, test_pred),
            "ann_configs" : ann_configs,
            "models_store": models_store,
            "fitted_list" : fitted_list,
            "train_times" : train_times,
            "test_times"  : test_times,
            "actual_tr"   : actual_tr,
            "pred_tr"     : pred_tr,
            "actual_te"   : y_test_orig,
            "pred_te"     : test_pred,
            "y_train_orig": y_train_orig,
        }

    except Exception as e:
        log(f"  ✗ ({wavelet}, L={level}) failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# SECTION 8 — FORECAST-ONLY  (loads saved bundle)
# ══════════════════════════════════════════════════════════════

def _forecast_only_wavelet_ann(params, horizon, n_samples=300):
    """
    Load the saved bundle and generate an MC Dropout interval forecast.

    Multivariate note
    -----------------
    If the model was trained with exogenous variables, you must supply:
        params["future_exog"] = {
            "col1": [v1, v2, …, v_horizon],
            "col2": [v1, v2, …, v_horizon],
        }
    Values are raw (unscaled) — the saved scalers handle normalisation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in ("static/wavelet_ann_bundle.pt", "static/wavelet_ann_weights.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train the model first.")

    bundle      = torch.load("static/wavelet_ann_bundle.pt",
                              weights_only=False, map_location="cpu")
    all_weights = torch.load("static/wavelet_ann_weights.pt",
                              weights_only=False, map_location="cpu")

    wavelet         = bundle["wavelet"]
    level           = bundle["level"]
    coef_names      = bundle["coef_names"]
    lags            = bundle["lags"]
    arch_list       = bundle["arch_list"]
    scaler_list     = bundle["scaler_list"]
    last_coeff_wins = bundle["last_coeff_windows"]   # list of (lags,)
    last_exog_wins  = bundle["last_exog_windows"]    # list of (lags, n_exog) or None
    time_labels     = bundle["future_time_labels"]
    exog_cols       = bundle.get("exog_cols", [])

    # ── Resolve future exog ─────────────────────────────────────
    future_exog_mat = None
    if exog_cols:
        future_exog_dict = params.get("future_exog", {})
        if not future_exog_dict:
            raise ValueError(
                "Model was trained with exogenous variables. "
                "Provide params['future_exog'] = {'col': [v1, v2, ...]}."
            )
        missing = [c for c in exog_cols if c not in future_exog_dict]
        if missing:
            raise ValueError(f"Missing future values for: {missing}")

        future_exog_mat = np.column_stack([
            np.array(future_exog_dict[col][:horizon], dtype=float)
            for col in exog_cols
        ])   # (horizon, n_exog)

    # ── MC forecast per coefficient ─────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    means_all, stds_all = [], []

    for (name, arch, scalers,
         last_coeff_win, last_exog_win, wts) in zip(
        coef_names, arch_list, scaler_list,
        last_coeff_wins, last_exog_wins, all_weights
    ):
        x_sc, y_sc = scalers
        model = ANNModel(
            input_dim      = arch["input_dim"],
            hidden_layer_1 = arch["hidden_layer_1"],
            hidden_layer_2 = arch["hidden_layer_2"],
            dropout_1      = arch["dropout_1"],
            dropout_2      = arch["dropout_2"],
        ).to(device)
        model.load_state_dict({k: v.to(device) for k, v in wts.items()})

        mean, std = _mc_forecast_coefficient(
            model, x_sc, y_sc,
            last_coeff_win, last_exog_win,
            future_exog_mat, horizon, device, n_samples,
        )
        means_all.append(mean)
        stds_all.append(std)

    # ── Aggregate: sum means, root-sum-of-squares for std ───────
    total_mean = np.sum(means_all, axis=0)
    total_std  = np.sqrt(np.sum(np.array(stds_all) ** 2, axis=0))

    lower_95 = total_mean - 1.960 * total_std
    upper_95 = total_mean + 1.960 * total_std
    lower_80 = total_mean - 1.282 * total_std
    upper_80 = total_mean + 1.282 * total_std
    lower_50 = total_mean - 0.674 * total_std
    upper_50 = total_mean + 0.674 * total_std

    future_labels = (time_labels[:horizon]
                     if time_labels else list(range(1, horizon + 1)))

    forecast_df = pd.DataFrame({
        "Period"        : future_labels,
        "Forecast"      : np.round(total_mean, 3),
        "Lower 80%"     : np.round(lower_80,   3),
        "Upper 80%"     : np.round(upper_80,   3),
        "Lower 95%"     : np.round(lower_95,   3),
        "Upper 95%"     : np.round(upper_95,   3),
        "Interval (95%)": [f"[{l:.2f}, {u:.2f}]"
                           for l, u in zip(lower_95, upper_95)],
    })

    return {
        "forecast_table": forecast_df,
        "model_name"    : f"Wavelet-ANN ({wavelet.upper()}, L={level})",
    }


# ══════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_wavelet_ann(
    data,
    params,
    horizon,
    frequency,
    mode         : str = "train",
    log_callback       = None,
):
    """
    MODWT-ANN Hybrid  —  Multivariate + All PyWavelets Filters.

    Parameters
    ----------
    data         : pd.DataFrame  (required in train mode)
    params       : dict  — all hyperparameters from the UI
    horizon      : int   — number of forecast steps
    frequency    : str   — "annual" | "monthly" etc.
    mode         : "train" | "forecast"
    log_callback : callable(str)  — optional progress logger (Flask/SSE)

    Key params entries
    ------------------
    target_col     : str         column to forecast
    time_col       : str         time index column
    exog_cols      : list[str]   exogenous column names ([] = univariate)
    split          : float       train fraction (default 0.85)
    ann_lags / lags: int         number of lags
    wann_wavelets  : list[str]   any pywt discrete wavelet names
                                  e.g. ["db4","sym8","coif3"]
                                  defaults to ["haar","db2","db4","la8"]
    wann_min_level : int         minimum decomposition level (default 1)
    wann_max_level : int         maximum decomposition level
    future_exog    : dict        FORECAST MODE with exog — required
                                  {"col1":[v1,…], "col2":[v1,…]}

    Grid-search params (same names as plain ANN):
        ann_hidden_layer_1_{min,max,step}
        ann_hidden_layer_2_{min,max,step}
        ann_dropout_{1,2}_{min,max,step}
        ann_learning_rate_{min,max,step}
        ann_weight_decay_{min,max,step}
        ann_huber_delta_{min,max,step}
        ann_patience_{min,max,step}
        ann_epochs_{min,max,step}

    Available wavelets
    ------------------
    Call  list_available_wavelets()  to see all 106+ options, including:
        Haar      : haar
        Daubechies: db1 … db38
        Symlets   : sym2 … sym20
        Coiflets  : coif1 … coif17
        Biorth.   : bior1.1 … bior6.8
        Rev.Bior. : rbio1.1 … rbio6.8
        Dmey      : dmey
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST-ONLY ──────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_wavelet_ann(params, horizon)

    # ── VALIDATE ───────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    target_col    = params.get("target_col", "Yield")
    time_col      = params.get("time_col",   "Year")
    exog_cols     = params.get("exog_cols",  []) or []
    has_exogenous = len(exog_cols) > 0

    LAGS      = params.get("ann_lags") or params.get("lags", 3)
    test_size = max(1, int(round((1 - params.get("split", 0.85)) * len(data))))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Wavelet search space ────────────────────────────────────
    # Preferred: pass any pywt discrete wavelet names in wann_wavelets
    wavelets = params.get("wann_wavelets", None)
    if not wavelets:
        # Legacy boolean-flag fallback (backwards compatibility)
        all_filters  = ["haar", "db2", "db4", "la8"]
        filter_flags = [
            params.get("wann_haar", True),
            params.get("wann_db2",  True),
            params.get("wann_db4",  True),
            params.get("wann_la8",  True),
        ]
        wavelets = [f for f, flag in zip(all_filters, filter_flags) if flag]
        if not wavelets:
            wavelets = ["db4"]

    # Validate every requested wavelet name against pywt
    valid_set = set(pywt.wavelist(kind="discrete"))
    bad = [w for w in wavelets if w not in valid_set]
    if bad:
        raise ValueError(
            f"Unknown wavelet(s): {bad}. "
            f"Call list_available_wavelets() to see all valid names."
        )

    N         = len(data)
    max_level = int(np.floor(np.log2(N)))
    min_level = max(1, params.get("wann_min_level", 1))
    max_level = min(max_level, params.get("wann_max_level", max_level))
    levels    = list(range(min_level, max_level + 1))

    log("=" * 62)
    log(f"  Wavelet-ANN  |  Mode: TRAIN")
    log(f"  Target    : {target_col}  |  Time : {time_col}")
    log(f"  Data type : {'Multivariate' if has_exogenous else 'Univariate'}")
    log(f"  Exog cols : {exog_cols if exog_cols else 'None'}")
    log(f"  Wavelets  : {wavelets}")
    log(f"  Levels    : {min_level} → {max_level}")
    log(f"  Lags      : {LAGS}  |  Test size : {test_size}")
    log("=" * 62)

    # ── Prepare series & exog ──────────────────────────────────
    cols_needed = [time_col, target_col] + exog_cols
    df = data[cols_needed].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    for col in [target_col] + exog_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    y_values    = df[target_col].values.astype(float)
    y_original  = y_values.copy()
    time_labels = df[time_col].tolist()
    exog_full   = (df[exog_cols].values.astype(float)
                   if has_exogenous else None)   # (N, n_exog) or None

    # ── Grid search over all (wavelet, level) pairs ─────────────
    all_results = []

    for wv in wavelets:
        for lv in levels:
            log(f"\n>>> Wavelet={wv.upper()}  Level={lv}")
            res = _run_one_combination(
                y_values, y_original, time_labels,
                wv, lv, test_size, exog_full,
                params, device, log,
            )
            if res is not None:
                all_results.append(res)
                log(f"    Test  RMSE={res['rmse_test']:.4f} "
                    f"MAE={res['mae_test']:.4f} "
                    f"MAPE={res['mape_test']:.2f}%")

    if not all_results:
        raise RuntimeError("All (wavelet, level) combinations failed.")

    # ── Select best combination by Test RMSE ───────────────────
    best = min(all_results, key=lambda r: r["rmse_test"])
    log(f"\n{'=' * 62}")
    log(f"  Best → Wavelet={best['wavelet'].upper()}  Level={best['level']}")
    log(f"  Train RMSE={best['rmse_train']:.4f} | Test RMSE={best['rmse_test']:.4f}")
    log(f"{'=' * 62}")

    wavelet  = best["wavelet"]
    level    = best["level"]
    ann_cfgs = best["ann_configs"]

    # ── Re-fit on full train data with best (wavelet, level) ───
    #    (ensures bundle contains models trained on all train data)
    os.makedirs("static", exist_ok=True)

    details_b, smooth_b = modwt(y_values, wavelet, level)
    all_coefs_b   = details_b + [smooth_b]
    train_coefs_b = [c[:-test_size] for c in all_coefs_b]
    coef_names    = [f"W{j+1}" for j in range(level)] + [f"V{level}"]
    exog_train_b  = (exog_full[:-test_size]
                     if exog_full is not None else None)

    n_exog     = len(exog_cols)
    input_dim_per_coef = LAGS * (1 + n_exog)   # coeff lags + exog lags

    arch_list       = []
    scaler_list     = []
    last_coeff_wins = []
    last_exog_wins  = []
    all_weights     = []
    fitted_list_b   = []

    for name, train in zip(coef_names, train_coefs_b):
        best_p = ann_cfgs[name]

        out = _fit_ann_on_coefficient(
            train, exog_train_b, params, device,
            LAGS, best_p, name, log,
        )
        fitted_full, x_sc, y_sc, model, last_coeff_win, last_exog_win = out

        fitted_list_b.append(fitted_full)

        arch_list.append({
            "input_dim"     : input_dim_per_coef,
            "hidden_layer_1": best_p["hidden_layer_1"],
            "hidden_layer_2": best_p["hidden_layer_2"],
            "dropout_1"     : best_p["dropout_1"],
            "dropout_2"     : best_p["dropout_2"],
        })
        scaler_list.append((x_sc, y_sc))
        last_coeff_wins.append(last_coeff_win)
        last_exog_wins.append(last_exog_win)
        all_weights.append(
            {k: v.cpu() for k, v in model.state_dict().items()}
            if model is not None else {}
        )

    # ── Future time labels ──────────────────────────────────────
    last_t = df[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        try:
            last_date = pd.to_datetime(last_t)
            freq_map = {"annual": "YS", "quarterly": "QS", "monthly": "MS",
                        "weekly": "W", "daily": "D"}
            freq_str = freq_map.get(frequency, "MS")
            future_dates = pd.date_range(
                start=last_date, periods=horizon + 1, freq=freq_str
            )[1:]
            future_time_labels = [d.strftime("%b-%y") for d in future_dates]
        except Exception:
            future_time_labels = list(range(1, horizon + 1))

    # ── Save bundle ─────────────────────────────────────────────
    torch.save(all_weights, "static/wavelet_ann_weights.pt")
    torch.save({
        # Decomposition
        "wavelet"           : wavelet,
        "level"             : level,
        "coef_names"        : coef_names,
        "lags"              : LAGS,
        # Per-coefficient model info
        "arch_list"         : arch_list,
        "scaler_list"       : scaler_list,
        "last_coeff_windows": last_coeff_wins,   # list of (lags,)
        "last_exog_windows" : last_exog_wins,    # list of (lags,n_exog) or None
        # Column metadata
        "target_col"        : target_col,
        "time_col"          : time_col,
        "exog_cols"         : exog_cols,
        "has_exogenous"     : has_exogenous,
        # Forecast helpers
        "future_time_labels": future_time_labels,
        # Full params for reference
        "params"            : params,
    }, "static/wavelet_ann_bundle.pt")

    # ── Plot ───────────────────────────────────────────────────
    actual_tr   = best["actual_tr"]
    pred_tr     = best["pred_tr"]
    actual_te   = best["actual_te"]
    pred_te     = best["pred_te"]
    train_times = best["train_times"]
    test_times  = best["test_times"]

    fig, ax = plt.subplots(figsize=(14, 5))
    tr_x = list(range(len(actual_tr)))
    te_x = list(range(len(actual_tr), len(actual_tr) + len(actual_te)))

    ax.plot(tr_x, actual_tr, color="steelblue",  label="Actual (Train)")
    ax.plot(tr_x, pred_tr,   color="steelblue",  linestyle="--",
            marker="o", ms=4, alpha=0.8, label="Predicted (Train)")
    ax.plot(te_x, actual_te, color="darkorange", label="Actual (Test)")
    ax.plot(te_x, pred_te,   color="darkorange", linestyle="--",
            marker="o", ms=4, alpha=0.8, label="Predicted (Test)")
    ax.axvline(x=len(actual_tr) - 1, color="black",
               linestyle=":", linewidth=2, label="Train/Test Split")

    all_x   = tr_x + te_x
    all_lbl = list(train_times) + list(test_times)
    step    = max(1, len(all_x) // 12)
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_lbl[::step], rotation=45, ha="right")
    mode_tag = "Multivariate" if has_exogenous else "Univariate"
    ax.set_title(
        f"Wavelet-ANN ({wavelet.upper()}, L={level}) — {mode_tag} — Train & Test"
    )
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/wavelet_ann_train.png", dpi=120)
    plt.savefig("static/wavelet_ann_test.png",  dpi=120)
    plt.close()

    # ── Comparison table (all wavelet × level combos) ──────────
    comparison_rows = [
        {
            "Wavelet"   : r["wavelet"].upper(),
            "Level"     : r["level"],
            "RMSE_Train": round(r["rmse_train"], 4),
            "MAE_Train" : round(r["mae_train"],  4),
            "MAPE_Train": round(r["mape_train"], 2),
            "RMSE_Test" : round(r["rmse_test"],  4),
            "MAE_Test"  : round(r["mae_test"],   4),
            "MAPE_Test" : round(r["mape_test"],  2),
        }
        for r in sorted(all_results, key=lambda x: x["rmse_test"])
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

    # ── Model config for display ───────────────────────────────
    model_config = {
        "wavelet"          : wavelet.upper(),
        "level"            : level,
        "lags"             : LAGS,
        "data_type"        : "Multivariate" if has_exogenous else "Univariate",
        "exog_cols"        : exog_cols,
        "coefficients"     : coef_names,
        "ann_configs"      : {
            name: {
                "hidden_layer_1": cfg["hidden_layer_1"],
                "hidden_layer_2": cfg["hidden_layer_2"],
                "dropout_1"     : round(cfg["dropout_1"], 3),
                "dropout_2"     : round(cfg["dropout_2"], 3),
                "learning_rate" : cfg["learning_rate"],
                "epochs"        : cfg["epochs"],
                "patience"      : cfg["patience"],
            }
            for name, cfg in ann_cfgs.items()
        },
        "all_combinations" : comparison_df.to_dict(orient="records"),
    }

    return {
        "model_key"       : "wavelet_ann",
        "model_name"      : f"Wavelet-ANN ({wavelet.upper()}, L={level})",
        "frequency"       : frequency,
        "horizon"         : horizon,
        "data_type"       : "Multivariate" if has_exogenous else "Univariate",
        "model_config"    : model_config,
        "auto_tuned"      : bool(params.get("auto_tune_wann", False)),

        "rmse_train"      : round(best["rmse_train"], 4),
        "mae_train"       : round(best["mae_train"],  4),
        "mape_train"      : round(best["mape_train"], 2),
        "rmse_test"       : round(best["rmse_test"],  4),
        "mae_test"        : round(best["mae_test"],   4),
        "mape_test"       : round(best["mape_test"],  2),

        "residuals"       : (actual_tr - pred_tr).tolist(),

        "train_table"     : pd.DataFrame({
            time_col    : train_times,
            "Actual"    : np.round(actual_tr, 4),
            "Predicted" : np.round(pred_tr,   4),
        }),
        "test_table"      : pd.DataFrame({
            time_col    : test_times,
            "Actual"    : np.round(actual_te, 4),
            "Predicted" : np.round(pred_te,   4),
        }),

        "forecast_table"  : None,          # filled on forecast mode
        "comparison_table": comparison_df,
    }