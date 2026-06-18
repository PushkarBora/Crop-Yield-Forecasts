"""
MODWT-Transformer Hybrid Model
================================
Architecture mirrors wavelet_lstm_model.py exactly, but replaces the
per-coefficient LSTM with a small Transformer encoder.

Workflow:
    y  →  MODWT  →  [W1, W2, …, WJ, VJ]
                         ↓  each coefficient
                    Grid Search best Transformer hyperparams
                    (features: coeff lags + raw exog lags as sequences)
                         ↓
                    Fit Transformer per coefficient
                         ↓
                    Delete first LAGS rows → iMODWT → prepend Y[:LAGS]
                         ↓
                    Train / Test metrics
                         ↓
                    Autoregressive MC Dropout forecast per coefficient
                    (inject future exog at each step)
                         ↓
                    Sum forecasts → ŷ_future  +  50/80/95% intervals

Key design decisions
--------------------
  Uses the SAME custom circular-convolution modwt / imodwt as
  wavelet_lstm_model.py — NO pywt.swt, NO level restrictions.

New param keys (collect in collect_hyperparameters):
    auto_tune_wtransformer
    wtransformer_nhead_{min,max,step}       → nhead per coeff Transformer
    wtransformer_num_layers_{min,max,step}  → encoder depth

Reused param keys (shared with wavelet_ann / wavelet_lstm):
    wann_wavelets, wann_min_level, wann_max_level
    wann_lags_{min,max,step}
    ann_hidden_layer_1_{min,max,step}  → d_model
    ann_dropout_1_{min,max,step}
    ann_learning_rate_{min,max,step}
    ann_weight_decay_{min,max,step}
    ann_huber_delta_{min,max,step}
    ann_epochs_{min,max,step}
    ann_patience_{min,max,step}

Bundle saved to: static/wavelet_transformer_bundle.pt
"""

import os
import sys
import math
import random
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — WAVELET UTILITIES  (identical to wavelet_lstm_model.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_filter(wavelet: str):
    wav = pywt.Wavelet(wavelet)
    g   = np.array(wav.dec_lo, dtype=float)
    h   = np.array(wav.dec_hi, dtype=float)
    return h, g


def modwt(x: np.ndarray, wavelet: str, level: int):
    """
    Forward MODWT using circular convolution.
    No edge padding, no length restrictions.
    Returns (details=[W1,…,WJ], smooth=VJ).
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
    """Inverse MODWT. imodwt(modwt(y)) ≈ y  (max error < 1e-10)."""
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
    trimmed = [np.delete(f, range(lags)) for f in fitted_list]
    recon   = imodwt(trimmed[:-1], trimmed[-1], wavelet)
    return np.concatenate([y_original_train[:lags], recon])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRANSFORMER ARCHITECTURE  (per-coefficient)
# ══════════════════════════════════════════════════════════════════════════════

class CoeffTransformer(nn.Module):
    """
    Small Transformer encoder for one wavelet coefficient series.
    Input : (batch, seq_len, n_features)
    Output: (batch,)
    """
    def __init__(self, n_features, d_model, nhead, num_layers,
                 dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.drop       = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.fc      = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.drop(self.input_proj(x))   # (B, L, d_model)
        x = self.encoder(x)
        x = self.norm(x[:, -1, :])          # take last position
        return self.fc(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LAG FEATURE BUILDER  (3-D for Transformer)
# ══════════════════════════════════════════════════════════════════════════════

def _build_lag_features(series: np.ndarray, exog: np.ndarray, lags: int):
    """
    Returns X (N-lags, lags, 1+n_exog) and y (N-lags,).
    Each row of the sequence is [coeff_t, exog0_t, exog1_t, …].
    """
    n_exog = exog.shape[1] if exog is not None else 0
    X, y   = [], []

    for i in range(lags, len(series)):
        seq = []
        for lag in range(lags, 0, -1):          # oldest first
            row = [series[i - lag]]
            if exog is not None:
                row += list(exog[i - lag])
            seq.append(row)
        X.append(seq)
        y.append(series[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GRID SEARCH  (per coefficient)
# ══════════════════════════════════════════════════════════════════════════════

def _grid_search_transformer(series, exog, params, device, lags):
    """Grid-search Transformer hyperparameters for one wavelet coefficient."""

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, dec=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, dec))
            v += step
        return vals

    default = {
        "d_model"        : 8,
        "nhead"          : 2,
        "num_layers"     : 1,
        "dim_feedforward": 16,
        "dropout"        : 0.1,
        "learning_rate"  : 0.001,
        "weight_decay"   : 1e-4,
        "huber_delta"    : 1.0,
        "epochs"         : 200,
        "patience"       : 30,
    }

    X, y = _build_lag_features(series, exog, lags)
    if len(X) < 4:
        return default

    N, L, F = X.shape
    x_sc    = StandardScaler()
    X_s     = x_sc.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    y_sc    = StandardScaler()
    y_s     = y_sc.fit_transform(y.reshape(-1, 1)).flatten()

    val_sp        = max(1, int(0.8 * len(X_s)))
    Xtr_s, Xva_s  = X_s[:val_sp], X_s[val_sp:]
    ytr_s, yva_s  = y_s[:val_sp], y_s[val_sp:]

    grid = {
        "d_model"      : irange(params.get("ann_hidden_layer_1_min",  8),
                                 params.get("ann_hidden_layer_1_max",  16),
                                 params.get("ann_hidden_layer_1_step", 8)),
        "nhead"        : irange(params.get("wtransformer_nhead_min",  1),
                                 params.get("wtransformer_nhead_max",  2),
                                 params.get("wtransformer_nhead_step", 1)),
        "num_layers"   : irange(params.get("wtransformer_num_layers_min",  1),
                                 params.get("wtransformer_num_layers_max",  2),
                                 params.get("wtransformer_num_layers_step", 1)),
        "dropout"      : frange(params.get("ann_dropout_1_min",  0.1),
                                 params.get("ann_dropout_1_max",  0.2),
                                 params.get("ann_dropout_1_step", 0.1), 2),
        "learning_rate": frange(params.get("ann_learning_rate_min",  0.001),
                                 params.get("ann_learning_rate_max",  0.003),
                                 params.get("ann_learning_rate_step", 0.002), 6),
        "weight_decay" : frange(params.get("ann_weight_decay_min",  1e-4),
                                 params.get("ann_weight_decay_max",  1e-4),
                                 params.get("ann_weight_decay_step", 1e-4), 7),
        "huber_delta"  : frange(params.get("ann_huber_delta_min",  1.0),
                                 params.get("ann_huber_delta_max",  1.0),
                                 params.get("ann_huber_delta_step", 0.5), 2),
        "epochs"       : irange(params.get("ann_epochs_min",  200),
                                 params.get("ann_epochs_max",  200),
                                 params.get("ann_epochs_step", 100)),
        "patience"     : irange(params.get("ann_patience_min",  30),
                                 params.get("ann_patience_max",  30),
                                 params.get("ann_patience_step", 10)),
    }

    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]

    Xtr = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_s, dtype=torch.float32, device=device)
    Xva = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    yva = torch.tensor(yva_s, dtype=torch.float32, device=device)

    best_val, best_p = float("inf"), None

    for p in combos:
        # d_model must be divisible by nhead
        if p["d_model"] % p["nhead"] != 0:
            continue
        dim_ff = max(p["d_model"] * 2, 16)
        try:
            torch.manual_seed(SEED)
            m    = CoeffTransformer(
                n_features      = F,
                d_model         = p["d_model"],
                nhead           = p["nhead"],
                num_layers      = p["num_layers"],
                dim_feedforward = dim_ff,
                dropout         = p["dropout"],
            ).to(device)
            crit = nn.HuberLoss(delta=p["huber_delta"])
            opt  = torch.optim.Adam(m.parameters(),
                                    lr=p["learning_rate"],
                                    weight_decay=p["weight_decay"])
            best_ep, pat = float("inf"), 0
            for _ in range(p["epochs"]):
                m.train(); opt.zero_grad()
                crit(m(Xtr), ytr).backward(); opt.step()
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
                best_val = best_ep
                best_p   = {**p, "dim_feedforward": dim_ff}
        except Exception:
            continue

    if best_p is None:
        best_p = {**default}
    if "dim_feedforward" not in best_p:
        best_p["dim_feedforward"] = max(best_p["d_model"] * 2, 16)
    return best_p


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIT TRANSFORMER PER COEFFICIENT
# ══════════════════════════════════════════════════════════════════════════════

def _fit_transformer_on_coefficient(train_series, exog_train, params, device,
                                     lags, best_p, coef_name, log):
    """
    Fit one Transformer on a single coefficient training series.

    Returns
    -------
    fitted_full : (N_train,)       first `lags` = actual coeff; rest = preds
    x_sc        : StandardScaler for X (2-D)
    y_sc        : StandardScaler for y
    model       : trained CoeffTransformer
    last_win    : (lags, n_feat)  last window in X-scaled space
    """
    X, y = _build_lag_features(train_series, exog_train, lags)

    if len(X) < 4:
        log(f"  {coef_name} -> SKIPPED (too short)")
        return np.zeros(len(train_series)), None, None, None, None

    N, L, F = X.shape
    x_sc    = StandardScaler()
    X_s     = x_sc.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    y_sc    = StandardScaler()
    y_s     = y_sc.fit_transform(y.reshape(-1, 1)).flatten()

    torch.manual_seed(SEED)
    X_t = torch.tensor(X_s, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_s, dtype=torch.float32, device=device)

    model = CoeffTransformer(
        n_features      = F,
        d_model         = best_p["d_model"],
        nhead           = best_p["nhead"],
        num_layers      = best_p["num_layers"],
        dim_feedforward = best_p.get("dim_feedforward",
                                      max(best_p["d_model"] * 2, 16)),
        dropout         = best_p["dropout"],
    ).to(device)

    opt  = torch.optim.Adam(model.parameters(),
                             lr=best_p["learning_rate"],
                             weight_decay=best_p["weight_decay"])
    crit = nn.HuberLoss(delta=best_p["huber_delta"])

    best_loss, best_state, counter = float("inf"), None, 0
    for _ in range(best_p["epochs"]):
        model.train(); opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward(); opt.step()
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

    log(f"  {coef_name} -> Transformer(d={best_p['d_model']}, "
        f"h={best_p['nhead']}, nl={best_p['num_layers']}) | Loss: {best_loss:.4f}")

    with torch.no_grad():
        fitted_s = model(X_t).cpu().numpy()

    fitted      = y_sc.inverse_transform(fitted_s.reshape(-1, 1)).flatten()
    fitted_full = np.concatenate([train_series[:lags], fitted])
    last_win    = X_s[-1].copy()   # (lags, F)

    return fitted_full, x_sc, y_sc, model, last_win


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — AUTOREGRESSIVE FORECAST PER COEFFICIENT
# ══════════════════════════════════════════════════════════════════════════════

def _forecast_coefficient(model, x_sc, y_sc, last_win,
                           future_exog, steps, device):
    """Deterministic autoregressive forecast."""
    model.eval()
    F      = last_win.shape[1]
    n_exog = F - 1
    win    = last_win.copy()
    preds  = []

    for t in range(steps):
        x_t = torch.tensor(win[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            p_s = model(x_t).cpu().numpy()
        pred = float(y_sc.inverse_transform(p_s.reshape(-1, 1))[0, 0])
        preds.append(pred)

        new_row    = win[-1].copy()
        new_row[0] = (pred - x_sc.mean_[0]) / x_sc.scale_[0]
        if n_exog > 0 and future_exog is not None:
            for j in range(n_exog):
                raw = future_exog[t, j]
                new_row[1 + j] = (raw - x_sc.mean_[1 + j]) / x_sc.scale_[1 + j]
        win = np.vstack([win[1:], new_row])

    return np.array(preds)


def _mc_forecast_coefficient(model, x_sc, y_sc, last_win,
                              future_exog, steps, device, n_samples=300):
    """MC Dropout forecast. Returns mean (steps,) and std (steps,)."""
    F      = last_win.shape[1]
    n_exog = F - 1
    model.train()   # enable dropout for MC sampling
    all_runs = []

    with torch.no_grad():
        for _ in range(n_samples):
            win = last_win.copy()
            run = []
            for t in range(steps):
                x_t = torch.tensor(win[np.newaxis], dtype=torch.float32, device=device)
                p_s = model(x_t).cpu().numpy()
                pred = float(y_sc.inverse_transform(p_s.reshape(-1, 1))[0, 0])
                run.append(pred)

                new_row    = win[-1].copy()
                new_row[0] = (pred - x_sc.mean_[0]) / x_sc.scale_[0]
                if n_exog > 0 and future_exog is not None:
                    for j in range(n_exog):
                        raw = future_exog[t, j]
                        new_row[1 + j] = (raw - x_sc.mean_[1 + j]) / x_sc.scale_[1 + j]
                win = np.vstack([win[1:], new_row])
            all_runs.append(run)

    model.eval()
    arr = np.array(all_runs)
    return arr.mean(axis=0), arr.std(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SINGLE (wavelet, level, lags) PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_combination(y_values, y_original, time_labels,
                          wavelet, level, test_size,
                          exog_full, params, device, log, lags):
    """Full MODWT-Transformer pipeline for one (wavelet, level, lags) combo."""
    try:
        details, smooth = modwt(y_values, wavelet, level)
        coef_names      = [f"W{j+1}" for j in range(level)] + [f"V{level}"]

        # Verify perfect reconstruction
        y_rec   = imodwt(details, smooth, wavelet)
        rec_err = np.max(np.abs(y_values - y_rec))
        if rec_err > 1e-4:
            log(f"  Reconstruction error {rec_err:.2e} — skipping")
            return None

        all_coefs    = details + [smooth]
        train_coefs  = [c[:-test_size] for c in all_coefs]
        y_train_orig = y_original[:-test_size]
        y_test_orig  = y_original[-test_size:]
        exog_train   = exog_full[:-test_size] if exog_full is not None else None
        exog_test    = exog_full[-test_size:]  if exog_full is not None else None

        fitted_list      = []
        models_store     = []
        transformer_cfgs = {}

        for name, train in zip(coef_names, train_coefs):
            best_p = _grid_search_transformer(train, exog_train, params, device, lags)
            transformer_cfgs[name] = best_p
            out = _fit_transformer_on_coefficient(
                train, exog_train, params, device, lags, best_p, name, log
            )
            fitted_full, x_sc, y_sc, model, last_win = out
            fitted_list.append(fitted_full)
            models_store.append((x_sc, y_sc, model, last_win, name))

        train_pred = reconstruct_train_with_original_prefix(
            fitted_list, wavelet, lags, y_train_orig
        )

        test_preds = []
        for (x_sc, y_sc, model, last_win, name) in models_store:
            if model is None:
                test_preds.append(np.zeros(test_size))
                continue
            fc = _forecast_coefficient(
                model, x_sc, y_sc, last_win, exog_test, test_size, device
            )
            test_preds.append(fc)
        test_pred = np.sum(test_preds, axis=0)

        n_tr      = min(len(y_train_orig), len(train_pred))
        actual_tr = y_train_orig[:n_tr]
        pred_tr   = train_pred[:n_tr]

        def mape_fn(a, p):
            return float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

        train_times = time_labels[:-test_size][:n_tr]
        test_times  = time_labels[-test_size:]

        return {
            "wavelet"          : wavelet,
            "level"            : level,
            "lags"             : lags,
            "rmse_train"       : float(root_mean_squared_error(actual_tr, pred_tr)),
            "mae_train"        : float(mean_absolute_error(actual_tr, pred_tr)),
            "mape_train"       : mape_fn(actual_tr, pred_tr),
            "rmse_test"        : float(root_mean_squared_error(y_test_orig, test_pred)),
            "mae_test"         : float(mean_absolute_error(y_test_orig, test_pred)),
            "mape_test"        : mape_fn(y_test_orig, test_pred),
            "transformer_cfgs" : transformer_cfgs,
            "models_store"     : models_store,
            "fitted_list"      : fitted_list,
            "train_times"      : train_times,
            "test_times"       : test_times,
            "actual_tr"        : actual_tr,
            "pred_tr"          : pred_tr,
            "actual_te"        : y_test_orig,
            "pred_te"          : test_pred,
            "y_train_orig"     : y_train_orig,
        }

    except Exception as e:
        log(f"  ({wavelet}, L={level}) failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FORECAST-ONLY  (loads saved bundle)
# ══════════════════════════════════════════════════════════════════════════════

def _forecast_only_wavelet_transformer(params, horizon, n_samples=300):
    """Load saved bundle and generate MC Dropout interval forecast."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle_path = "static/wavelet_transformer_bundle.pt"
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"{bundle_path} not found. Train the model first.")

    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")

    wavelet     = bundle["wavelet"]
    level       = bundle["level"]
    coef_names  = bundle["coef_names"]
    arch_list   = bundle["arch_list"]
    scaler_list = bundle["scaler_list"]
    last_wins   = bundle["last_wins"]
    time_labels = bundle["future_time_labels"]
    exog_cols   = bundle.get("exog_cols", [])

    future_exog_mat = None
    if exog_cols:
        future_exog_dict = params.get("future_exog", {})
        if not future_exog_dict:
            raise ValueError("Supply params['future_exog'] for multivariate forecast.")
        future_exog_mat = np.column_stack([
            np.array(future_exog_dict[col][:horizon], dtype=float)
            for col in exog_cols
        ])

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    means_all, stds_all = [], []
    for ci, cname in enumerate(coef_names):
        arch        = arch_list[ci]
        x_sc, y_sc  = scaler_list[ci]
        last_win    = last_wins[ci]

        model = CoeffTransformer(
            n_features      = arch["n_features"],
            d_model         = arch["d_model"],
            nhead           = arch["nhead"],
            num_layers      = arch["num_layers"],
            dim_feedforward = arch["dim_feedforward"],
            dropout         = arch["dropout"],
        ).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in bundle[f"weights_{cname}"].items()}
        )

        mean, std = _mc_forecast_coefficient(
            model, x_sc, y_sc, last_win,
            future_exog_mat, horizon, device, n_samples,
        )
        means_all.append(mean)
        stds_all.append(std)

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
        "model_name"    : f"Wavelet-Transformer ({wavelet.upper()}, L={level})",
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_wavelet_transformer(
    data,
    params,
    horizon,
    frequency,
    mode         : str = "train",
    log_callback       = None,
):
    """
    MODWT-Transformer Hybrid Model.

    Param keys (collect in collect_hyperparameters):
        target_col, time_col, exog_cols, split
        wann_wavelets, wann_min_level, wann_max_level
        wann_lags_min, wann_lags_max, wann_lags_step
        ann_hidden_layer_1_{min,max,step}       → d_model
        wtransformer_nhead_{min,max,step}        → nhead
        wtransformer_num_layers_{min,max,step}   → encoder depth
        ann_dropout_1_{min,max,step}
        ann_learning_rate_{min,max,step}
        ann_weight_decay_{min,max,step}
        ann_huber_delta_{min,max,step}
        ann_epochs_{min,max,step}
        ann_patience_{min,max,step}
        auto_tune_wtransformer
        future_exog  (forecast mode with exog)
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    if mode == "forecast":
        return _forecast_only_wavelet_transformer(params, horizon)

    if data is None:
        raise ValueError("data must be provided in train mode.")

    target_col    = params.get("target_col", "Yield")
    time_col      = params.get("time_col",   "Year")
    exog_cols     = params.get("exog_cols",  []) or []
    has_exogenous = len(exog_cols) > 0
    auto_tune     = params.get("auto_tune_wtransformer", True)
    split         = params.get("split", 0.85)
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Wavelet search space ───────────────────────────────────────────────
    ALIAS        = {"la8": "sym8", "la16": "sym16", "la20": "sym20"}
    raw_wavelets = params.get("wann_wavelets", ["haar", "db2", "db4", "sym8"])
    if isinstance(raw_wavelets, str):
        raw_wavelets = [w.strip() for w in raw_wavelets.split(",") if w.strip()]
    wavelets = [ALIAS.get(w.lower(), w) for w in raw_wavelets]

    valid_set = set(pywt.wavelist(kind="discrete"))
    bad = [w for w in wavelets if w not in valid_set]
    if bad:
        raise ValueError(f"Unknown wavelet(s): {bad}.")

    min_level_p = max(1, params.get("wann_min_level", 1))
    max_level_p = params.get("wann_max_level", 3)

    lags_min  = params.get("wann_lags_min",  2)
    lags_max  = params.get("wann_lags_max",  4)
    lags_step = params.get("wann_lags_step", 2)
    lags_list = list(range(lags_min, lags_max + 1, lags_step))
    if not lags_list:
        lags_list = [lags_min]

    # ── Prepare data ──────────────────────────────────────────────────────
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
    N           = len(y_values)
    time_labels = df[time_col].tolist()
    exog_full   = (df[exog_cols].values.astype(float)
                   if has_exogenous else None)
    test_size   = max(1, int(round((1 - split) * N)))

    log(f"Wavelet-Transformer | {'Multivariate' if has_exogenous else 'Univariate'}")
    log(f"   Target: {target_col}  |  Exog: {exog_cols if exog_cols else 'None'}")
    log(f"   Dataset size: {N} samples  |  Train: {N - test_size}  Test: {test_size}")

    # ── Build outer grid — no pywt.swt level restriction ──────────────────
    data_max_level = max(1, int(math.floor(math.log2(N))))
    effective_max  = min(max_level_p, data_max_level)

    outer_combos = []
    for wv in wavelets:
        for lv in range(min_level_p, effective_max + 1):
            for lg in lags_list:
                outer_combos.append((wv, lv, lg))

    seen = set()
    outer_combos = [c for c in outer_combos if not (c in seen or seen.add(c))]

    if not outer_combos:
        outer_combos = [(wavelets[0], 1, lags_list[0])]
        log("No valid combos after filtering — using fallback (level=1).")

    log(f"Outer grid: {len(outer_combos)} combos "
        f"({len(wavelets)} wavelets x levels x lags)")

    # ── Grid search ───────────────────────────────────────────────────────
    all_results = []
    for combo_idx, (wv, lv, lg) in enumerate(outer_combos, 1):
        log(f"\n>>> Combo {combo_idx}/{len(outer_combos)}: "
            f"wavelet={wv}, level={lv}, lags={lg}")
        res = _run_one_combination(
            y_values, y_original, time_labels,
            wv, lv, test_size, exog_full,
            params, device, log, lg,
        )
        if res is not None:
            all_results.append(res)
            log(f"    Test RMSE={res['rmse_test']:.4f} "
                f"MAE={res['mae_test']:.4f} "
                f"MAPE={res['mape_test']:.2f}%")

    if not all_results:
        raise RuntimeError("All wavelet-Transformer combinations failed.")

    # ── Select best ───────────────────────────────────────────────────────
    best             = min(all_results, key=lambda r: r["rmse_test"])
    wavelet          = best["wavelet"]
    level            = best["level"]
    lags             = best["lags"]
    transformer_cfgs = best["transformer_cfgs"]

    log(f"\nBest: wavelet={wavelet.upper()}  level={level}  lags={lags}")
    log(f"Train RMSE={best['rmse_train']:.4f} | Test RMSE={best['rmse_test']:.4f}")

    # ── Re-fit on full training set with best config ───────────────────────
    os.makedirs("static", exist_ok=True)
    details_b, smooth_b = modwt(y_values, wavelet, level)
    all_coefs_b         = details_b + [smooth_b]
    train_coefs_b       = [c[:-test_size] for c in all_coefs_b]
    coef_names          = [f"W{j+1}" for j in range(level)] + [f"V{level}"]
    exog_train_b        = (exog_full[:-test_size] if exog_full is not None else None)
    n_feat              = 1 + len(exog_cols)

    arch_list     = []
    scaler_list   = []
    last_wins     = []
    fitted_list_b = []
    bundle_data   = {}

    for name, train in zip(coef_names, train_coefs_b):
        best_p = transformer_cfgs[name]
        out    = _fit_transformer_on_coefficient(
            train, exog_train_b, params, device,
            lags, best_p, name, log,
        )
        fitted_full, x_sc, y_sc, model, last_win = out
        fitted_list_b.append(fitted_full)

        arch_list.append({
            "n_features"     : n_feat,
            "d_model"        : best_p["d_model"],
            "nhead"          : best_p["nhead"],
            "num_layers"     : best_p["num_layers"],
            "dim_feedforward": best_p.get("dim_feedforward",
                                           max(best_p["d_model"] * 2, 16)),
            "dropout"        : best_p["dropout"],
        })
        scaler_list.append((x_sc, y_sc))
        last_wins.append(last_win)
        bundle_data[f"weights_{name}"] = (
            {k: v.cpu() for k, v in model.state_dict().items()}
            if model is not None else {}
        )

    # ── Future time labels ─────────────────────────────────────────────────
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

    # ── Save bundle ────────────────────────────────────────────────────────
    bundle_data.update({
        "wavelet"           : wavelet,
        "level"             : level,
        "lags"              : lags,
        "coef_names"        : coef_names,
        "arch_list"         : arch_list,
        "scaler_list"       : scaler_list,
        "last_wins"         : last_wins,
        "target_col"        : target_col,
        "time_col"          : time_col,
        "exog_cols"         : exog_cols,
        "has_exogenous"     : has_exogenous,
        "future_time_labels": future_time_labels,
        "params"            : params,
    })
    torch.save(bundle_data, "static/wavelet_transformer_bundle.pt")

    # ── Plot ───────────────────────────────────────────────────────────────
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
    ax.set_title(f"Wavelet-Transformer ({wavelet.upper()}, L={level}) — "
                 f"{'Multivariate' if has_exogenous else 'Univariate'}")
    ax.set_xlabel(time_col); ax.set_ylabel(target_col)
    ax.legend(); plt.tight_layout()
    plt.savefig("static/wavelet_transformer_train.png", dpi=120)
    plt.savefig("static/wavelet_transformer_test.png",  dpi=120)
    plt.close()

    # ── Comparison table ───────────────────────────────────────────────────
    comparison_rows = [
        {
            "Wavelet"   : r["wavelet"].upper(),
            "Level"     : r["level"],
            "Lags"      : r["lags"],
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

    model_config = {
        "wavelet"         : wavelet.upper(),
        "level"           : level,
        "lags"            : lags,
        "data_type"       : "Multivariate" if has_exogenous else "Univariate",
        "exog_cols"       : exog_cols,
        "coefficients"    : coef_names,
        "transformer_cfgs": {
            name: {k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in cfg.items()}
            for name, cfg in transformer_cfgs.items()
        },
        "all_combinations": comparison_df.to_dict(orient="records"),
    }

    return {
        "model_key"       : "wavelet_transformer",
        "model_name"      : f"Wavelet-Transformer ({wavelet.upper()}, L={level})",
        "frequency"       : frequency,
        "horizon"         : horizon,
        "data_type"       : "Multivariate" if has_exogenous else "Univariate",
        "model_config"    : model_config,
        "auto_tuned"      : auto_tune,

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

        "forecast_table"  : None,
        "comparison_table": comparison_df,
    }