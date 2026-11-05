"""
Fast wavelet hybrid engine for DATAI
====================================
Shared by:
    models/wavelet_ann_model.py          run_wavelet_ann
    models/wavelet_lstm_model.py         run_wavelet_lstm
    models/wavelet_transformer_model.py  run_wavelet_transformer
    models/wavelet_arima_model.py        run_wavelet_arima

How it works
------------
1. Causal additive wavelet decomposition (redundant "a trous" transform —
   shift-invariant like MODWT, works with any PyWavelets filter):
        y(t) = D1(t) + ... + DJ(t) + AJ(t)          (exact)
   Each component at time t uses only y(t), y(t-1), ...  so the train/test
   evaluation never sees future values (circular MODWT on the full series does).

2. Models
   * Wavelet-ARIMA : ARMA(X) on every detail Dj, ARIMA(X) on the trend AJ;
                     each exogenous variable is decomposed the same way and component k
                     of y uses component k of the exogenous variables. Forecasts summed.
   * Wavelet-ANN / LSTM / Transformer : ONE network reads recent values of all
     components (+ exogenous variables) and predicts the change y(t) - y(t-1),
     so forecasts can follow rising / falling trends.

3. Speed
   * Wavelet filter x level x lags are ranked with a closed-form ridge model
     (milliseconds each) instead of training networks for every combination.
   * Only the best 2 set-ups are trained as networks, with at most 4 architectures
     from the ranges chosen in the form, using full-batch training + early stopping.
   * The final model is a small seed ensemble.
"""

import os
import sys
import time
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Windows DLL fix for PyTorch (same as the other DATAI models) ──
try:
    _dll = os.path.join(sys.exec_prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(_dll):
        os.add_dll_directory(_dll)
except Exception:
    pass

STATIC_DIR = "static"

# Fallback filters if PyWavelets is missing (low-pass, normalised to sum 1)
_BUILTIN_FILTERS = {
    "haar": [0.5, 0.5],
    "db2":  [0.341506350946, 0.591506350946, 0.158493649054, -0.091506350946],
    "db4":  [0.162901714026, 0.505472857546, 0.446100069123, -0.019787513118,
             -0.132253583685, 0.021808150237, 0.023251800535, -0.007493494665],
    "sym4": [0.022785172948, -0.008912350721, -0.070158812089, 0.210617267102,
             0.568329121704, 0.351869534328, -0.020955482563, -0.053574450709],
}
ALIASES = {"la8": "sym8", "la16": "sym16", "la20": "sym20", "d4": "db2", "d8": "db4"}


# ============================================================
# 1. DECOMPOSITION
# ============================================================
def resolve_filters(names):
    """Map UI names to valid filter names; drop unknown ones."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out = []
    try:
        import pywt
        valid = set(pywt.wavelist(kind="discrete"))
    except Exception:
        valid = set(_BUILTIN_FILTERS)
    for n in names or []:
        n = ALIASES.get(str(n).strip().lower(), str(n).strip().lower())
        if n in valid and n not in out:
            out.append(n)
    return out or ["haar", "db2", "db4"]


_filter_cache = {}


def lowpass(name):
    if name not in _filter_cache:
        try:
            import pywt
            g = np.asarray(pywt.Wavelet(name).rec_lo, dtype=float)
        except Exception:
            g = np.asarray(_BUILTIN_FILTERS.get(name, _BUILTIN_FILTERS["haar"]), dtype=float)
        s = g.sum()
        _filter_cache[name] = g / (s if abs(s) > 1e-12 else 1.0)
    return _filter_cache[name]


def wavelet_decompose(x, filt, level):
    """Rows D1..DJ, AJ (shape (J+1, n)); rows sum exactly to x; causal."""
    g = lowpass(filt)
    c = np.asarray(x, dtype=float).copy()
    n = len(c)
    comps = []
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        pad = step * (len(g) - 1)
        cp = np.concatenate([np.full(pad, c[0]), c])
        smooth = np.zeros(n)
        for l, gl in enumerate(g):
            s = pad - l * step
            smooth += gl * cp[s:s + n]
        comps.append(c - smooth)
        c = smooth
    comps.append(c)
    return np.vstack(comps)


def max_level_for(filt, n_train):
    """Deepest level whose filter warm-up stays within ~35% of the training data."""
    L = len(lowpass(filt))
    budget = max(1.0, 0.35 * n_train)
    lvl = 1
    while (L - 1) * (2 ** (lvl + 1) - 1) <= budget and lvl < 8:
        lvl += 1
    return lvl


# ============================================================
# 2. FEATURES
# ============================================================
def build_features(x, comps, lags, t_idx, exog=None):
    """
    For each target time t (t >= lags):
      channels 0..J : component values at t-lags .. t-1   (trend made relative to y(t-1))
      exog channels : exogenous values at t-lags+1 .. t   (last step = current period)
      target        : y(t) - y(t-1)
    Returns seq (rows, lags, channels) and base y(t-1).
    """
    J1 = comps.shape[0]
    n_ex = 0 if exog is None else exog.shape[1]
    t_idx = np.asarray(t_idx, dtype=int)
    seq = np.empty((len(t_idx), lags, J1 + n_ex))
    base = x[t_idx - 1]
    for r, t in enumerate(t_idx):
        block = comps[:, t - lags:t].T.copy()
        block[:, J1 - 1] -= x[t - 1]
        if n_ex:
            block = np.hstack([block, exog[t - lags + 1:t + 1]])
        seq[r] = block
    return seq, base


class Scaler:
    def fit(self, a):
        flat = a.reshape(-1, a.shape[-1])
        self.mu = flat.mean(axis=0)
        self.sd = flat.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def tf(self, a):
        return (a - self.mu) / self.sd


# ============================================================
# 3. FAST SCREENING (ridge proxy)
# ============================================================
def _ridge(Xtr, ytr, Xte, alpha):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + reg, A.T @ (ytr - ym))
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    return B @ w + ym


def _proxy_score(x, exog, n_fit, n_end, filt, level, lags):
    comps = wavelet_decompose(x[:n_end], filt, level)
    t_tr, t_va = np.arange(lags, n_fit), np.arange(n_fit, n_end)
    if len(t_tr) < 6 or len(t_va) < 2:
        return np.inf
    Str, btr = build_features(x, comps, lags, t_tr, exog)
    Sva, bva = build_features(x, comps, lags, t_va, exog)
    Xtr, Xva = Str.reshape(len(Str), -1), Sva.reshape(len(Sva), -1)
    ytr = x[t_tr] - btr
    best = np.inf
    for alpha in (0.3, 3.0, 30.0):
        pred = _ridge(Xtr, ytr, Xva, alpha) + bva
        best = min(best, float(np.sqrt(np.mean((x[t_va] - pred) ** 2))))
    return best


def screen_setups(x, exog, n_train, settings, log, top_k=2, use_lags=True):
    n_val = max(3, int(round(0.15 * n_train)))
    n_fit = n_train - n_val
    cands = []
    for f in settings["filters"]:
        lmax = max_level_for(f, n_fit)
        for lv in settings["levels"]:
            lv = min(lv, lmax)
            for lg in (settings["lags"] if use_lags else [settings["lags"][0]]):
                lg = max(1, min(lg, n_fit // 3))
                if (f, lv, lg) not in cands:
                    cands.append((f, lv, lg))
    log(f"   ⚡ Fast screening of {len(cands)} filter × level{' × lags' if use_lags else ''} "
        f"set-ups (validation = last {n_val} training points)")
    t0 = time.time()
    scored = sorted((_proxy_score(x, exog, n_fit, n_train, f, lv, lg), f, lv, lg)
                    for f, lv, lg in cands)
    for s, f, lv, lg in scored[:5]:
        log(f"      {f.upper():<8} L={lv}" + (f"  lags={lg:<2}" if use_lags else "")
            + f" → val RMSE={s:.4f}")
    log(f"   ✅ Screening finished in {time.time() - t0:.2f}s")
    return [(f, lv, lg) for _, f, lv, lg in scored[:top_k]], n_fit


# ============================================================
# 4. NEURAL NETWORKS
# ============================================================
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_net(kind, lags, n_ch, arch, dropout):
    torch, nn = _torch()

    class ANN(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = arch
            self.f = nn.Sequential(nn.Flatten(),
                                   nn.Linear(lags * n_ch, h1), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(h1, h2), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(h2, 1))

        def forward(self, s):
            return self.f(s).squeeze(-1)

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, layers = arch
            self.rnn = nn.LSTM(n_ch, hidden, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.skip = nn.Linear(lags * n_ch, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, s):
            o, _ = self.rnn(s)
            return (self.head(o[:, -1]) + self.skip(s.flatten(1))).squeeze(-1)

    class GRU(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, layers = arch
            self.rnn = nn.GRU(n_ch, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.skip = nn.Linear(lags * n_ch, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, s):
            o, _ = self.rnn(s)
            return (self.head(o[:, -1]) + self.skip(s.flatten(1))).squeeze(-1)

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, layers = arch
            self.rnn = nn.RNN(n_ch, hidden, num_layers=layers, batch_first=True, nonlinearity="tanh",
                              dropout=dropout if layers > 1 else 0.0)
            self.skip = nn.Linear(lags * n_ch, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, s):
            o, _ = self.rnn(s)
            return (self.head(o[:, -1]) + self.skip(s.flatten(1))).squeeze(-1)

    class Trans(nn.Module):
        def __init__(self):
            super().__init__()
            d_model, nhead, layers = arch
            self.inp = nn.Linear(n_ch, d_model)
            self.pos = nn.Parameter(torch.zeros(1, lags, d_model))
            enc = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=2 * d_model,
                                             dropout=dropout, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc, num_layers=layers)
            self.skip = nn.Linear(lags * n_ch, 1)
            self.head = nn.Linear(d_model, 1)

        def forward(self, s):
            h = self.enc(self.inp(s) + self.pos)
            return (self.head(h[:, -1]) + self.skip(s.flatten(1))).squeeze(-1)

    return {"ann": ANN, "lstm": LSTM, "gru": GRU, "rnn": RNN, "transformer": Trans}[kind]()


def train_net(kind, arch, Xtr, ytr, Xva, yva, settings, seed, fixed_epochs=None):
    """Full-batch AdamW + early stopping on validation. Returns (model, best_epoch, val_rmse)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    model = make_net(kind, Xtr.shape[1], Xtr.shape[2], arch, settings["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["wd"])
    loss_fn = nn.HuberLoss(delta=settings["huber"])
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    use_val = fixed_epochs is None and Xva is not None and len(Xva) > 0
    if use_val:
        Xv, yv = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    max_ep = fixed_epochs or settings["epochs"]
    best, best_ep, best_state, wait = np.inf, max_ep, None, 0
    for ep in range(1, max_ep + 1):
        model.train()
        opt.zero_grad()
        loss_fn(model(Xt), yt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if use_val and (ep % 5 == 0 or ep == max_ep):
            model.eval()
            with torch.no_grad():
                v = float(torch.sqrt(torch.mean((model(Xv) - yv) ** 2)))
            if v < best - 1e-6:
                best, best_ep, wait = v, ep, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                wait += 5
                if wait >= settings["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_ep, best


class Ensemble:
    def __init__(self, kind, arch, nets, sx, y_sd):
        self.kind, self.arch, self.nets, self.sx, self.y_sd = kind, arch, nets, sx, y_sd

    def predict_change(self, seq):
        torch, _ = _torch()
        X = torch.tensor(self.sx.tf(seq), dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(X).numpy() for n in self.nets], axis=0) * self.y_sd

    def state(self):
        return {"kind": self.kind, "arch": self.arch, "y_sd": self.y_sd,
                "sx_mu": self.sx.mu, "sx_sd": self.sx.sd,
                "nets": [{k: v.cpu() for k, v in n.state_dict().items()} for n in self.nets]}

    @classmethod
    def from_state(cls, st, lags, n_ch, dropout):
        nets = []
        for sd in st["nets"]:
            m = make_net(st["kind"], lags, n_ch, tuple(st["arch"]), dropout)
            m.load_state_dict(sd)
            m.eval()
            nets.append(m)
        sx = Scaler()
        sx.mu, sx.sd = st["sx_mu"], st["sx_sd"]
        return cls(st["kind"], tuple(st["arch"]), nets, sx, st["y_sd"])


def fit_ensemble(kind, arch, x, exog, comps, lags, t_idx, settings, epochs):
    S, b = build_features(x, comps, lags, t_idx, exog)
    y = x[t_idx] - b
    sx = Scaler().fit(S)
    y_sd = float(np.std(y)) or 1.0
    nets = [train_net(kind, arch, sx.tf(S), y / y_sd, None, None, settings, seed=s,
                      fixed_epochs=epochs)[0] for s in range(settings["ensemble"])]
    return Ensemble(kind, arch, nets, sx, y_sd)


def neural_search(kind, x, exog, n_train, settings, log):
    torch, _ = _torch()
    prev = torch.get_num_threads()
    torch.set_num_threads(1)              # tiny networks: one thread is fastest
    try:
        if settings["auto"]:
            setups, n_fit = screen_setups(x, exog, n_train, settings, log, top_k=2)
        else:
            f = settings["filters"][0]
            lv = min(settings["levels"][0], max_level_for(f, n_train))
            lg = max(1, min(settings["lags"][0], n_train // 4))
            setups = [(f, lv, lg)]
            n_fit = n_train - max(3, int(round(0.15 * n_train)))
            log(f"   🔧 Manual set-up: {f.upper()}, level {lv}, lags {lg}")

        archs = settings["archs"]
        log(f"   🧠 Training {len(setups)} set-up(s) × {len(archs)} architecture(s)")
        best, t0 = None, time.time()
        for (f, lv, lg) in setups:
            comps = wavelet_decompose(x[:n_train], f, lv)
            t_fit, t_val = np.arange(lg, n_fit), np.arange(n_fit, n_train)
            Sf, bf = build_features(x, comps, lg, t_fit, exog)
            Sv, bv = build_features(x, comps, lg, t_val, exog)
            yf, yv = x[t_fit] - bf, x[t_val] - bv
            sx = Scaler().fit(Sf)
            y_sd = float(np.std(yf)) or 1.0
            for arch in archs:
                _, ep, v = train_net(kind, arch, sx.tf(Sf), yf / y_sd, sx.tf(Sv), yv / y_sd,
                                     settings, seed=0)
                rmse = v * y_sd
                log(f"      {f.upper():<8} L={lv} lags={lg:<2} arch={arch} → val RMSE={rmse:.4f} (epoch {ep})")
                if best is None or rmse < best["rmse"]:
                    best = {"rmse": rmse, "filter": f, "level": lv, "lags": lg,
                            "arch": arch, "epochs": max(30, ep)}
        best["final_epochs"] = int(best["epochs"] * n_train / max(1, n_fit))
        log(f"   ✅ Best: {best['filter'].upper()}, level {best['level']}, lags {best['lags']}, "
            f"arch {best['arch']}  ({time.time() - t0:.1f}s)")
        log(f"   🔁 Training final {settings['ensemble']}-seed ensemble...")
        comps = wavelet_decompose(x[:n_train], best["filter"], best["level"])
        ens = fit_ensemble(kind, best["arch"], x, exog, comps, best["lags"],
                           np.arange(best["lags"], n_train), settings, best["final_epochs"])
        return best, ens
    finally:
        torch.set_num_threads(prev)


def neural_one_step(ens, x, exog, filt, level, lags, t_idx):
    comps = wavelet_decompose(x, filt, level)
    S, b = build_features(x, comps, lags, t_idx, exog)
    return b + ens.predict_change(S)


def neural_recursive(ens, x_hist, exog_hist, exog_future, filt, level, lags, horizon):
    xs = list(np.asarray(x_hist, float))
    ex = None if exog_hist is None else np.asarray(exog_hist, float)
    out = []
    for h in range(horizon):
        arr = np.asarray(xs + [xs[-1]])           # placeholder for the step being predicted
        t = len(arr) - 1
        ex_t = None if ex is None else np.vstack([ex, exog_future[:h + 1]])
        comps = wavelet_decompose(arr, filt, level)
        S, b = build_features(arr, comps, lags, [t], ex_t)
        nxt = float(b[0] + ens.predict_change(S)[0])
        out.append(nxt)
        xs.append(nxt)
    return np.array(out)


# ============================================================
# 4b. MACHINE-LEARNING LEARNERS (Random Forest, SVR, XGBoost)
#     Same wavelet features (flattened); target = change y(t) - y(t-1)
# ============================================================
ML_KINDS = ("rf", "svr", "xgb")


def make_ml(kind, cfg, seed=0):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=int(cfg["n_estimators"]),
                                     max_depth=(int(cfg["max_depth"]) or None),
                                     min_samples_leaf=int(cfg["min_samples_leaf"]),
                                     max_features=cfg.get("max_features", 1.0),
                                     random_state=seed, n_jobs=-1)
    if kind == "svr":
        from sklearn.svm import SVR
        return SVR(kernel=cfg.get("kernel", "rbf"), C=float(cfg["C"]),
                   epsilon=float(cfg["epsilon"]), gamma=cfg.get("gamma", "scale"))
    if kind == "xgb":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=int(cfg["n_estimators"]),
                                learning_rate=float(cfg["learning_rate"]),
                                max_depth=int(cfg["max_depth"]), subsample=0.9,
                                colsample_bytree=0.9, reg_lambda=1.0,
                                random_state=seed, n_jobs=1, verbosity=0)
        except ImportError:   # fall back to scikit-learn boosting if xgboost is missing
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=int(cfg["n_estimators"]),
                                             learning_rate=float(cfg["learning_rate"]),
                                             max_depth=int(cfg["max_depth"]), subsample=0.9,
                                             random_state=seed)
    raise ValueError(f"Unknown ML kind: {kind}")


def ml_config_text(kind, cfg):
    if kind == "rf":
        return (f"trees={cfg['n_estimators']}, max_depth={cfg['max_depth'] or 'None'}, "
                f"min_samples_leaf={cfg['min_samples_leaf']}")
    if kind == "svr":
        return f"kernel={cfg.get('kernel', 'rbf')}, C={cfg['C']}, epsilon={cfg['epsilon']}"
    return (f"trees={cfg['n_estimators']}, learning_rate={cfg['learning_rate']}, "
            f"max_depth={cfg['max_depth']}")


class MLModel:
    """Same interface as Ensemble: predict_change(seq) -> change in y."""
    def __init__(self, kind, cfg, est, sx, y_sd):
        self.kind, self.cfg, self.est, self.sx, self.y_sd = kind, cfg, est, sx, y_sd

    def predict_change(self, seq):
        X = self.sx.tf(seq).reshape(len(seq), -1)
        return np.asarray(self.est.predict(X), float) * self.y_sd

    def state(self):
        return {"kind": self.kind, "cfg": self.cfg, "est": self.est,
                "sx_mu": self.sx.mu, "sx_sd": self.sx.sd, "y_sd": self.y_sd}

    @classmethod
    def from_state(cls, st):
        sx = Scaler()
        sx.mu, sx.sd = st["sx_mu"], st["sx_sd"]
        return cls(st["kind"], st["cfg"], st["est"], sx, st["y_sd"])


def fit_ml(kind, cfg, x, exog, comps, lags, t_idx):
    S, b = build_features(x, comps, lags, t_idx, exog)
    y = x[t_idx] - b
    sx = Scaler().fit(S)
    y_sd = float(np.std(y)) or 1.0
    est = make_ml(kind, cfg)
    est.fit(sx.tf(S).reshape(len(S), -1), y / y_sd)
    return MLModel(kind, cfg, est, sx, y_sd)


def ml_search(kind, x, exog, n_train, settings, log):
    if settings["auto"]:
        setups, n_fit = screen_setups(x, exog, n_train, settings, log, top_k=2)
    else:
        f = settings["filters"][0]
        lv = min(settings["levels"][0], max_level_for(f, n_train))
        lg = max(1, min(settings["lags"][0], n_train // 4))
        setups = [(f, lv, lg)]
        n_fit = n_train - max(3, int(round(0.15 * n_train)))
        log(f"   🔧 Manual set-up: {f.upper()}, level {lv}, lags {lg}")

    cfgs = settings["ml_grid"]
    log(f"   🤖 Testing {len(setups)} set-up(s) × {len(cfgs)} configuration(s)")
    best, t0 = None, time.time()
    for (f, lv, lg) in setups:
        comps = wavelet_decompose(x[:n_train], f, lv)
        t_fit, t_val = np.arange(lg, n_fit), np.arange(n_fit, n_train)
        for cfg in cfgs:
            m = fit_ml(kind, cfg, x, exog, comps, lg, t_fit)
            Sv, bv = build_features(x, comps, lg, t_val, exog)
            pred = bv + m.predict_change(Sv)
            rmse = float(np.sqrt(np.mean((x[t_val] - pred) ** 2)))
            log(f"      {f.upper():<8} L={lv} lags={lg:<2} {ml_config_text(kind, cfg)} → val RMSE={rmse:.4f}")
            if best is None or rmse < best["rmse"]:
                best = {"rmse": rmse, "filter": f, "level": lv, "lags": lg, "cfg": cfg}
    log(f"   ✅ Best: {best['filter'].upper()}, level {best['level']}, lags {best['lags']}, "
        f"{ml_config_text(kind, best['cfg'])}  ({time.time() - t0:.1f}s)")
    comps = wavelet_decompose(x[:n_train], best["filter"], best["level"])
    model = fit_ml(kind, best["cfg"], x, exog, comps, best["lags"], np.arange(best["lags"], n_train))
    return best, model


# ============================================================
# 5. WAVELET-ARIMA
# ============================================================
def _fit_component(y, ex, is_trend, max_p, max_q):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    d, trend = 0, "n"
    if is_trend:
        from statsmodels.tsa.stattools import adfuller
        d = 1
        try:
            if adfuller(np.diff(y), autolag="AIC")[1] > 0.05:
                d = 2
        except Exception:
            pass
        trend = "t" if d == 1 else "n"
    use_ex = ex
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0 and not is_trend:
                continue
            try:
                r = SARIMAX(y, exog=use_ex, order=(p, d, q), trend=trend,
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False, maxiter=200)
                aic = float(r.aic)
                if not np.isfinite(aic) or np.max(np.abs(r.params)) > 1e6:
                    continue
                if best is None or aic < best[0]:
                    best = (aic, (p, d, q), trend, r)
            except Exception:
                continue
    if best is None:
        r = SARIMAX(y, exog=use_ex, order=(1, d, 0), trend=trend).fit(disp=False)
        best = (float(r.aic), (1, d, 0), trend, r)
    return best


def decompose_exog(exog, filt, level):
    """Decompose every exogenous column with the same filter -> (J+1, n, n_exog) or None.
    Component k of y is then explained by component k of each exogenous variable."""
    if exog is None:
        return None
    return np.stack([wavelet_decompose(exog[:, j], filt, level) for j in range(exog.shape[1])], axis=2)


def wavelet_arima_fit(x, exog, n_train, filt, level, settings):
    comps = wavelet_decompose(x, filt, level)
    ex_c = decompose_exog(exog, filt, level)
    J1 = comps.shape[0]
    fits, train_fit, test_pred = [], np.zeros(n_train), np.zeros(len(x) - n_train)
    for k in range(J1):
        is_trend = k == J1 - 1
        ex_k = None if ex_c is None else ex_c[k]
        aic, order, trend, res = _fit_component(comps[k, :n_train],
                                                None if ex_k is None else ex_k[:n_train],
                                                is_trend, settings["max_p"], settings["max_q"])
        full = res.apply(comps[k], exog=ex_k)
        pred = np.asarray(full.predict(start=0, end=len(x) - 1))
        train_fit += pred[:n_train]
        test_pred += pred[n_train:]
        fits.append({"component": f"A{level}" if is_trend else f"D{k + 1}",
                     "order": order, "trend": trend, "aic": aic})
    return comps, fits, train_fit, test_pred


def arima_search(x, exog, n_train, settings, log):
    if settings["auto"]:
        setups, n_fit = screen_setups(x, exog, n_train, settings, log, top_k=2, use_lags=False)
        setups = list(dict.fromkeys((f, lv) for f, lv, _ in setups))
    else:
        f = settings["filters"][0]
        setups = [(f, min(settings["levels"][0], max_level_for(f, n_train)))]
        n_fit = n_train
        log(f"   🔧 Manual set-up: {f.upper()}, level {setups[0][1]}")
    if len(setups) == 1:
        return setups[0]
    best = None
    for f, lv in setups:
        t0 = time.time()
        ex = None if exog is None else exog[:n_train]
        _, _, _, vp = wavelet_arima_fit(x[:n_train], ex, n_fit, f, lv, settings)
        rmse = float(np.sqrt(np.mean((x[n_fit:n_train] - vp) ** 2)))
        log(f"      {f.upper():<8} L={lv} → validation RMSE={rmse:.4f} ({time.time() - t0:.1f}s)")
        if best is None or rmse < best[0]:
            best = (rmse, f, lv)
    return best[1], best[2]


# ============================================================
# 6. HELPERS
# ============================================================
def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return (float(np.sqrt(np.mean((a - p) ** 2))), float(np.mean(np.abs(a - p))),
            float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100))


_DATE_FORMATS = ["%b-%y", "%b-%Y", "%B-%y", "%B-%Y", "%b %y", "%b %Y", "%B %Y",
                 "%Y-%m", "%Y/%m", "%m-%Y", "%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
                 "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]


def future_time_labels(time_values, horizon):
    """Continue the time column in its own format: 2023→2024, 'Jan-12'→'Feb-12', '2012-01'→'2012-02'."""
    vals = [v for v in list(time_values) if v is not None]
    if not vals:
        return list(range(1, horizon + 1))
    try:
        nums = [float(v) for v in vals[-3:]]
        if nums[-1] > 1e11:
            raise ValueError
        step = (nums[-1] - nums[-2]) if len(nums) > 1 else 1.0
        step = step or 1.0
        out = [nums[-1] + step * i for i in range(1, horizon + 1)]
        if all(float(n).is_integer() for n in nums) and float(step).is_integer():
            out = [int(o) for o in out]
        return out
    except (TypeError, ValueError):
        pass
    strs = [str(v).strip() for v in vals]
    dts, fmt = None, None
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals[-2:]):
        try:
            dts, fmt = pd.to_datetime(pd.Series(vals[-2:]), unit="ms"), "%Y-%m-%d"
        except Exception:
            dts = None
    if dts is None:
        for f in _DATE_FORMATS:
            try:
                dts, fmt = pd.to_datetime(pd.Series(strs[-2:]), format=f), f
                break
            except Exception:
                continue
    if dts is not None and len(dts) == 2:
        a, b = dts.iloc[0], dts.iloc[1]
        months = (b.year - a.year) * 12 + (b.month - a.month)
        if months > 0 and a.day == b.day:
            nxt = [b + pd.DateOffset(months=months * i) for i in range(1, horizon + 1)]
        else:
            delta = (b - a) if b > a else pd.Timedelta(days=1)
            nxt = [b + delta * i for i in range(1, horizon + 1)]
        return [d.strftime(fmt) for d in nxt]
    return [f"{strs[-1]} +{i}" for i in range(1, horizon + 1)]


def interval_table(labels, mean, sigma1):
    sd = sigma1 * np.sqrt(np.arange(1, len(mean) + 1))
    lo95, hi95 = mean - 1.96 * sd, mean + 1.96 * sd
    lo80, hi80 = mean - 1.2816 * sd, mean + 1.2816 * sd
    return pd.DataFrame({
        "Period": labels,
        "Forecast": np.round(mean, 3),
        "Lower 80%": np.round(lo80, 3), "Upper 80%": np.round(hi80, 3),
        "Lower 95%": np.round(lo95, 3), "Upper 95%": np.round(hi95, 3),
        "Interval (95%)": [f"[{l:.2f}, {u:.2f}]" for l, u in zip(lo95, hi95)],
    })


def save_plot(model_key, title, train_t, tr_act, tr_pred, test_t, te_act, te_pred,
              time_col, target_col):
    os.makedirs(STATIC_DIR, exist_ok=True)
    tr_x = list(range(len(tr_act)))
    te_x = list(range(len(tr_act), len(tr_act) + len(te_act)))
    all_x, all_lbl = tr_x + te_x, [str(t) for t in list(train_t) + list(test_t)]
    step = max(1, len(all_x) // 12)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(tr_x, tr_act, color="steelblue", label="Actual (Train)")
    ax.plot(tr_x, tr_pred, color="steelblue", linestyle="--", marker="o", ms=3, alpha=0.8,
            label="Predicted (Train)")
    ax.plot(te_x, te_act, color="darkorange", label="Actual (Test)")
    ax.plot(te_x, te_pred, color="darkorange", linestyle="--", marker="o", ms=4, alpha=0.9,
            label="Predicted (Test)")
    ax.axvline(x=len(tr_act) - 1, color="black", linestyle=":", linewidth=2, label="Train/Test Split")
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_lbl[::step], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f"{model_key}_train.png"), dpi=120)
    plt.savefig(os.path.join(STATIC_DIR, f"{model_key}_test.png"), dpi=120)
    plt.close(fig)


def energy_share(comps, level):
    v = np.array([np.var(c) for c in comps])
    tot = v.sum() or 1.0
    names = [f"D{k + 1}" for k in range(level)] + [f"A{level}"]
    return ", ".join(f"{n}: {100 * e / tot:.0f}%" for n, e in zip(names, v))


def prepare(data, params):
    time_col = params.get("time_col", "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols = [c for c in (params.get("exog_cols") or []) if c in data.columns]
    df = data[[time_col, target_col] + exog_cols].copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col)
    df = df.reset_index(drop=True)
    for c in [target_col] + exog_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").interpolate(method="linear").ffill().bfill()
    x = df[target_col].to_numpy(float)
    if len(x) < 20:
        raise ValueError(f"Wavelet models need at least 20 observations (got {len(x)}).")
    exog = df[exog_cols].to_numpy(float) if exog_cols else None
    split = float(params.get("split", 0.85) or 0.85)
    n_train = min(max(int(split * len(x)), 15), len(x) - 2)
    return df, x, exog, exog_cols, n_train, time_col, target_col


def _future_exog_matrix(params, exog_cols, horizon):
    if not exog_cols:
        return None
    fe = params.get("future_exog") or {}
    if isinstance(fe, pd.DataFrame):
        fe = {c: fe[c].tolist() for c in fe.columns}
    missing = [c for c in exog_cols if c not in fe]
    if missing:
        raise ValueError(f"This model was trained with exogenous variables. "
                         f"Please enter future values for: {missing}")
    mat = np.column_stack([np.asarray(fe[c], float)[:horizon] for c in exog_cols])
    if len(mat) < horizon:
        raise ValueError(f"Please enter {horizon} future values for each exogenous variable.")
    return mat


# ============================================================
# 7. FORECAST MODE
# ============================================================
def forecast(model_key, params, horizon):
    path = os.path.join(STATIC_DIR, f"{model_key}_bundle.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please train the model first.")
    with open(path, "rb") as f:
        b = pickle.load(f)
    x = np.asarray(b["series"], float)
    ex_hist = None if b["exog"] is None else np.asarray(b["exog"], float)
    ex_fut = _future_exog_matrix(params, b["exog_cols"], horizon)
    labels = future_time_labels(b["time_values"], horizon)

    if b["kind"] == "arima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        comps = wavelet_decompose(x, b["filter"], b["level"])
        n = len(x)
        ex_all = None if ex_hist is None else decompose_exog(np.vstack([ex_hist, ex_fut]),
                                                             b["filter"], b["level"])
        mean = np.zeros(horizon)
        for k, c in enumerate(b["components"]):
            ex_h = None if ex_all is None else ex_all[k][:n]
            ex_f = None if ex_all is None else ex_all[k][n:n + horizon]
            res = SARIMAX(comps[k], exog=ex_h, order=tuple(c["order"]), trend=c["trend"],
                          enforce_stationarity=False, enforce_invertibility=False
                          ).filter(np.asarray(c["params"]))
            mean += np.asarray(res.forecast(horizon, exog=ex_f))
    else:
        if b["kind"] in ML_KINDS:
            ens = MLModel.from_state(b["ml_model"])
        else:
            n_ch = b["level"] + 1 + (0 if ex_hist is None else ex_hist.shape[1])
            ens = Ensemble.from_state(b["ensemble"], b["lags"], n_ch, b["dropout"])
        mean = neural_recursive(ens, x, ex_hist, ex_fut, b["filter"], b["level"], b["lags"], horizon)

    return {"forecast_table": interval_table(labels, mean, b["sigma1"]),
            "model_name": b.get("model_name", model_key)}


# ============================================================
# 8. TRAIN
# ============================================================
def train(model_key, kind, label, data, params, settings, horizon, frequency, log):
    t_start = time.time()
    df, x, exog, exog_cols, n_train, time_col, target_col = prepare(data, params)
    times = df[time_col].tolist()
    is_multi = exog is not None
    log(f"🌊 {label} | {'Multivariate' if is_multi else 'Univariate'}")
    log(f"   Target: {target_col}  |  Exog: {exog_cols if exog_cols else 'None'}")
    log(f"   Dataset size: {len(x)}  |  Train: {n_train}  Test: {len(x) - n_train}")
    log(f"   Mode: {'Auto-tune' if settings['auto'] else 'Manual'}  |  Filters: "
        f"{', '.join(f.upper() for f in settings['filters'])}")
    log("=" * 60)

    if kind == "arima":
        filt, level = arima_search(x, exog, n_train, settings, log)
        log(f"   📐 Fitting component models ({filt.upper()}, level {level})...")
        comps, fits, train_fit, test_pred = wavelet_arima_fit(x, exog, n_train, filt, level, settings)
        for c in fits:
            log(f"      {c['component']:<4} → ARIMA{c['order']}  trend={c['trend']}  AIC={c['aic']:.2f}")
        start = min(n_train - 1, max(2 * level, 2))
        tr_idx = np.arange(start, n_train)
        train_pred, lags, arch_txt = train_fit[start:], None, \
            " + ".join(f"{c['component']}:ARIMA{c['order']}" for c in fits)
    else:
        if kind in ML_KINDS:
            best, ens = ml_search(kind, x, exog, n_train, settings, log)
        else:
            best, ens = neural_search(kind, x, exog, n_train, settings, log)
        filt, level, lags = best["filter"], best["level"], best["lags"]
        tr_idx = np.arange(lags, n_train)
        train_pred = neural_one_step(ens, x[:n_train], None if exog is None else exog[:n_train],
                                     filt, level, lags, tr_idx)
        test_pred = neural_one_step(ens, x, exog, filt, level, lags, np.arange(n_train, len(x)))
        arch_txt = ml_config_text(kind, best["cfg"]) if kind in ML_KINDS else str(best["arch"])

    tr_act, te_act = x[tr_idx], x[n_train:]
    test_pred = np.asarray(test_pred, float)
    rmse_tr, mae_tr, mape_tr = metrics(tr_act, train_pred)
    rmse_te, mae_te, mape_te = metrics(te_act, test_pred)
    log(f"\n📊 Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"📊 Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    te_res, tr_res = te_act - test_pred, tr_act - train_pred
    sigma1 = float(np.std(te_res, ddof=1)) if len(te_res) >= 5 else float(np.std(tr_res, ddof=1))
    if not np.isfinite(sigma1) or sigma1 <= 0:
        sigma1 = float(np.std(np.diff(x))) or 1.0

    model_name = f"{label} ({filt.upper()}, L={level})"

    # ── Refit on the full series and save the bundle ──────────────
    log("🔧 Refitting on the full dataset for future forecasting...")
    bundle = {"model_key": model_key, "kind": kind, "model_name": model_name,
              "filter": filt, "level": level, "lags": lags,
              "series": x.tolist(), "exog": None if exog is None else exog.tolist(),
              "exog_cols": exog_cols, "time_values": times, "time_col": time_col,
              "target_col": target_col, "sigma1": sigma1, "dropout": settings.get("dropout", 0.1),
              "trained_at": datetime.utcnow().isoformat()}
    if kind == "arima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        full = wavelet_decompose(x, filt, level)
        ex_c = decompose_exog(exog, filt, level)
        comp_b = []
        for k, c in enumerate(fits):
            r = SARIMAX(full[k], exog=None if ex_c is None else ex_c[k],
                        order=c["order"], trend=c["trend"], enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
            comp_b.append({"order": list(c["order"]), "trend": c["trend"],
                           "params": np.asarray(r.params).tolist()})
        bundle["components"] = comp_b
    elif kind in ML_KINDS:
        full_model = fit_ml(kind, best["cfg"], x, exog, wavelet_decompose(x, filt, level),
                            lags, np.arange(lags, len(x)))
        bundle["ml_model"] = full_model.state()
    else:
        torch, _ = _torch()
        prev = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            ens_full = fit_ensemble(kind, best["arch"], x, exog, wavelet_decompose(x, filt, level),
                                    lags, np.arange(lags, len(x)), settings,
                                    int(best["final_epochs"] * len(x) / n_train))
        finally:
            torch.set_num_threads(prev)
        bundle["ensemble"] = ens_full.state()
    os.makedirs(STATIC_DIR, exist_ok=True)
    with open(os.path.join(STATIC_DIR, f"{model_key}_bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    train_t, test_t = [times[i] for i in tr_idx], times[n_train:]
    save_plot(model_key, f"{model_name} — {'Multivariate' if is_multi else 'Univariate'} — Train & Test",
              train_t, tr_act, train_pred, test_t, te_act, test_pred, time_col, target_col)

    config = {
        "wavelet": filt.upper(),
        "level": level,
        "components": f"D1–D{level} + A{level}",
        "energy_share": energy_share(wavelet_decompose(x[:n_train], filt, level), level),
        "data_type": "Multivariate" if is_multi else "Univariate",
        "exog_cols": ", ".join(exog_cols) if exog_cols else "None",
        "decomposition": "Causal additive (MODWT-type), no look-ahead",
    }
    if kind == "arima":
        config["component_models"] = arch_txt
        config["max_p_q"] = f"({settings['max_p']}, {settings['max_q']})"
    elif kind in ML_KINDS:
        config["lags"] = lags
        config["model_settings"] = arch_txt
        config["target"] = "Change y(t) − y(t−1)"
    else:
        config["lags"] = lags
        config["architecture"] = arch_txt
        config["target"] = "Change y(t) − y(t−1)"
        config["epochs"] = best["final_epochs"]
        config["ensemble_seeds"] = settings["ensemble"]
    config["training_time"] = f"{time.time() - t_start:.1f} s"
    log(f"✅ {label} complete in {time.time() - t_start:.1f}s")

    return {
        "model_key": model_key, "model_name": model_name,
        "frequency": frequency, "horizon": horizon,
        "data_type": "Multivariate" if is_multi else "Univariate",
        "model_config": config,
        "auto_tuned": bool(settings["auto"]),
        "rmse_train": round(rmse_tr, 4), "mae_train": round(mae_tr, 4), "mape_train": round(mape_tr, 2),
        "rmse_test": round(rmse_te, 4), "mae_test": round(mae_te, 4), "mape_test": round(mape_te, 2),
        "residuals": tr_res.tolist(),
        "train_table": pd.DataFrame({time_col: train_t, "Actual": np.round(tr_act, 4),
                                     "Predicted": np.round(train_pred, 4)}),
        "test_table": pd.DataFrame({time_col: test_t, "Actual": np.round(te_act, 4),
                                    "Predicted": np.round(test_pred, 4)}),
        "forecast_table": None,
    }


# ============================================================
# 9. ENTRY USED BY THE MODEL FILES
# ============================================================
def run(model_key, kind, label, data, params, settings, horizon, frequency,
        mode="train", log_callback=None):
    def log(msg):
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        print(msg)

    try:  # statsmodels adds an "always" warning filter on first import -> import first
        import statsmodels.tsa.statespace.sarimax  # noqa: F401
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "forecast":
            return forecast(model_key, params or {}, horizon)
        if data is None:
            raise ValueError("data must be provided in train mode.")
        return train(model_key, kind, label, data, dict(params or {}), settings,
                     horizon, frequency, log)


# ============================================================
# 10. SETTINGS HELPERS (used by the model files)
# ============================================================
def num(params, key, default, cast=float):
    try:
        v = params.get(key)
        return cast(float(v)) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def int_range(params, prefix_min, prefix_max, prefix_step, d_min, d_max, d_step, cap=8):
    lo = num(params, prefix_min, d_min, int)
    hi = max(lo, num(params, prefix_max, d_max, int))
    st = max(1, num(params, prefix_step, d_step, int))
    vals = list(range(lo, hi + 1, st)) or [lo]
    return vals[:cap]


def spread(vals, k=2):
    """Pick up to k values spread over a list (min, max, ...)."""
    vals = sorted(set(vals))
    if len(vals) <= k:
        return vals
    idx = np.linspace(0, len(vals) - 1, k).round().astype(int)
    return [vals[i] for i in sorted(set(idx))]


def base_settings(params, auto, levels, lags):
    """Common settings; in auto mode training uses fast, robust defaults."""
    filters = resolve_filters(params.get("wann_wavelets", ["haar", "db2", "db4", "sym8"]))
    s = {
        "auto": bool(auto),
        "filters": filters if auto else filters[:1],
        "levels": levels,
        "lags": lags,
        "ensemble": 3 if auto else 2,
        "dropout": min(0.3, max(0.0, num(params, "ann_dropout_1_min", 0.1))),
        "wd": max(1e-6, num(params, "ann_weight_decay_min", 1e-4)),
        "huber": max(0.1, num(params, "ann_huber_delta_min", 1.0)),
    }
    if auto:
        s.update(lr=0.005, epochs=max(400, num(params, "ann_epochs_max", 200, int)),
                 patience=max(40, num(params, "ann_patience_max", 30, int)))
    else:
        s.update(lr=max(1e-4, num(params, "ann_learning_rate_min", 0.003)),
                 epochs=max(100, num(params, "ann_epochs_min", 200, int)),
                 patience=max(20, num(params, "ann_patience_min", 30, int)))
    return s


def wavelet_levels_lags(params):
    lv_min = max(1, num(params, "wann_min_level", 1, int))
    lv_max = max(lv_min, min(8, num(params, "wann_max_level", 3, int)))
    lag_min = max(1, num(params, "wann_lags_min", 2, int))
    lag_max = max(lag_min, min(24, num(params, "wann_lags_max", 4, int)))
    lag_step = max(1, num(params, "wann_lags_step", 2, int))
    return list(range(lv_min, lv_max + 1)), (list(range(lag_min, lag_max + 1, lag_step)) or [lag_min])


# ============================================================
# 11. SETTINGS FOR WAVELET-RF / SVR / XGBOOST / GRU / RNN
#     Form fields use a short prefix: wrf_, wsvr_, wxgb_, wgru_, wrnn_
# ============================================================
def extra_settings(params, p, kind):
    """Build engine settings from the form fields of one of the newer wavelet models."""
    auto = bool(params.get(f"auto_tune_{p}", True))
    if auto:
        filters = resolve_filters(params.get(f"{p}_wavelets") or ["haar", "db2", "db4", "sym8"])
        lv_min = max(1, num(params, f"{p}_min_level", 1, int))
        lv_max = max(lv_min, min(8, num(params, f"{p}_max_level", 3, int)))
        lag_min = max(1, num(params, f"{p}_lags_min", 2, int))
        lag_max = max(lag_min, min(24, num(params, f"{p}_lags_max", 4, int)))
        levels = list(range(lv_min, lv_max + 1))
        lags = spread(list(range(lag_min, lag_max + 1)), 3)
    else:
        filters = resolve_filters([params.get(f"{p}_manual_wavelet") or "sym8"])[:1]
        levels = [max(1, min(8, num(params, f"{p}_manual_level", 2, int)))]
        lags = [max(1, min(24, num(params, f"{p}_manual_lags", 3, int)))]

    s = {"auto": auto, "filters": filters, "levels": levels, "lags": lags,
         "ensemble": 3 if auto else 2, "dropout": 0.1, "wd": 1e-4, "huber": 1.0,
         "lr": 0.005, "epochs": 400, "patience": 40}

    if kind in ("gru", "rnn"):
        if auto:
            s["archs"] = [(8, 1), (16, 1), (16, 2)]
        else:
            hidden = max(2, min(256, num(params, f"{p}_hidden", 16, int)))
            layers = max(1, min(3, num(params, f"{p}_layers", 1, int)))
            s["archs"] = [(hidden, layers)]
            s["lr"] = max(1e-4, min(0.1, num(params, f"{p}_lr", 0.005)))
            s["epochs"] = max(50, min(3000, num(params, f"{p}_epochs", 400, int)))
            s["patience"] = max(20, s["epochs"] // 10)
    elif kind == "rf":
        s["ml_grid"] = ([{"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 1},
                         {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 2},
                         {"n_estimators": 200, "max_depth": 0, "min_samples_leaf": 4}] if auto else
                        [{"n_estimators": max(10, min(2000, num(params, f"{p}_n_estimators", 200, int))),
                          "max_depth": max(0, min(50, num(params, f"{p}_max_depth", 0, int))),
                          "min_samples_leaf": max(1, min(50, num(params, f"{p}_min_samples_leaf", 1, int)))}])
    elif kind == "svr":
        kernel = params.get(f"{p}_kernel") or "rbf"
        kernel = kernel if kernel in ("rbf", "linear", "poly", "sigmoid") else "rbf"
        s["ml_grid"] = ([{"kernel": "rbf", "C": C, "epsilon": e}
                         for C in (1.0, 10.0, 100.0) for e in (0.05, 0.2)] if auto else
                        [{"kernel": kernel,
                          "C": max(1e-3, num(params, f"{p}_C", 10.0)),
                          "epsilon": max(0.0, num(params, f"{p}_epsilon", 0.1))}])
    elif kind == "xgb":
        s["ml_grid"] = ([{"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
                         {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
                         {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 3}] if auto else
                        [{"n_estimators": max(10, min(3000, num(params, f"{p}_n_estimators", 300, int))),
                          "learning_rate": max(1e-3, min(1.0, num(params, f"{p}_learning_rate", 0.05))),
                          "max_depth": max(1, min(15, num(params, f"{p}_max_depth", 3, int)))}])
    return s
