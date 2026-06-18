import numpy as np
import pandas as pd
import os
import sys
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    torch_dll_path = os.path.join(
        sys.exec_prefix, "Lib", "site-packages", "torch", "lib"
    )
    if os.path.exists(torch_dll_path):
        os.add_dll_directory(torch_dll_path)
except Exception:
    pass

import torch
import torch.nn as nn

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error,mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

def clean_dataframe(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
    df[obj_cols] = df[obj_cols].ffill().bfill()
    return df.dropna()
# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class ANNModel(nn.Module):
    """
    ANN for time-series regression.
    Input shape : (batch, n_features)  — flattened lags
    Output shape: (batch,)  — single-step prediction (scaled)
    """
    def __init__(self, input_dim, hidden_layer_1, hidden_layer_2,
                 dropout_1, dropout_2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),          # dropout for MC sampling

            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),          # dropout for MC sampling

            nn.Linear(hidden_layer_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# AUTO-TUNE  (grid search over user-defined ranges)
# ============================================================
def auto_tune_ann(
    X_train, y_train, X_val, y_val,
    device, input_dim, params,
    log_callback=None
):
    import itertools

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting ANN Grid Search Auto-Tuning")

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, decimals=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, decimals))
            v += step
        return vals

    grid = {
        
        "hidden_layer_1": irange(params["ann_hidden_layer_1_min"],
                                 params["ann_hidden_layer_1_max"],
                                 params["ann_hidden_layer_1_step"]),
        "hidden_layer_2": irange(params["ann_hidden_layer_2_min"],
                                 params["ann_hidden_layer_2_max"],
                                 params["ann_hidden_layer_2_step"]),
        "dropout_1":      frange(params["ann_dropout_1_min"],
                                 params["ann_dropout_1_max"],
                                 params["ann_dropout_1_step"], 3),
        "dropout_2":      frange(params["ann_dropout_2_min"],
                                 params["ann_dropout_2_max"],
                                 params["ann_dropout_2_step"], 3),
        "learning_rate":  frange(params["ann_learning_rate_min"],
                                 params["ann_learning_rate_max"],
                                 params["ann_learning_rate_step"], 6),
        "weight_decay":   frange(params["ann_weight_decay_min"],
                                 params["ann_weight_decay_max"],
                                 params["ann_weight_decay_step"], 7),
        "huber_delta":    frange(params["ann_huber_delta_min"],
                                 params["ann_huber_delta_max"],
                                 params["ann_huber_delta_step"], 2),
        "patience":       irange(params["ann_patience_min"],
                                 params["ann_patience_max"],
                                 params["ann_patience_step"]),
         "epochs":         irange(params["ann_epochs_min"],
                                 params["ann_epochs_max"],
                                 params["ann_epochs_step"]),
    }
    
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    log(f"🔢 Total combinations: {len(combos)}")

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val,   dtype=torch.float32, device=device)
    yva = torch.tensor(y_val,   dtype=torch.float32, device=device)

    best_val, best_p = float("inf"), None

    for idx, p in enumerate(combos, 1):
        if idx % 50 == 0:
            log(f"   Progress: {idx}/{len(combos)} ({idx/len(combos)*100:.1f}%)")
        try:
            model = ANNModel(
                input_dim=input_dim,
                hidden_layer_1=p["hidden_layer_1"],
                hidden_layer_2=p["hidden_layer_2"],
                dropout_1=p["dropout_1"],
                dropout_2=p["dropout_2"],
            ).to(device)

            crit = nn.HuberLoss(delta=p["huber_delta"])
            opt  = torch.optim.Adam(model.parameters(),
                                    lr=p["learning_rate"],
                                    weight_decay=p["weight_decay"])
            best_ep, pat = float("inf"), 0

            for _ in range(p["epochs"]):
                model.train(); opt.zero_grad()
                crit(model(Xtr), ytr).backward(); opt.step()

                model.eval()
                with torch.no_grad():
                    vl = crit(model(Xva), yva).item()
                if vl < best_ep:
                    best_ep, pat = vl, 0
                else:
                    pat += 1
                    if pat >= p["patience"]:
                        break

            if best_ep < best_val:
                best_val, best_p = best_ep, p.copy()
                log(f"   ✨ New best Val Loss: {best_val:.6f} at #{idx}")

        except Exception as e:
            log(f"   ⚠️ Skipped #{idx}: {e}")
            continue

    log(f"✅ Best params: {best_p}  |  Val Loss: {best_val:.6f}")
    return best_p


# ============================================================
# MC DROPOUT AUTOREGRESSIVE FORECAST
# ============================================================
def _forecast_autoregressive_mc(model, last_window_flat, horizon,
                                  n_vars, target_idx,
                                  future_exog_scaled,
                                  x_scaler_mean, x_scaler_scale,
                                  y_scaler_mean, y_scaler_scale,
                                  lags, device, n_samples=200):
    """
    Monte Carlo Dropout forecast for ANN.
    last_window_flat : (LAGS * n_vars,)  — X-scaled, flattened
    Returns: mean_pred, lower_95, upper_95, lower_80, upper_80
    """
    all_runs = []
    model.train()   # activate dropout for MC sampling

    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            # Reshape to (LAGS, n_vars) for easier rolling
            win = last_window_flat.clone().reshape(lags, n_vars)

            for step in range(horizon):
                x_flat = win.reshape(1, -1)
                pred_scaled = model(x_flat).item()
                pred_actual = pred_scaled * y_scaler_scale + y_scaler_mean
                preds.append(pred_actual)

                # Roll window: drop oldest lag, append new row
                new_row = win[-1].clone()
                new_row[target_idx] = (
                    (pred_actual - x_scaler_mean[target_idx])
                    / x_scaler_scale[target_idx]
                )
                if future_exog_scaled is not None:
                    exog_ptr = 0
                    for i in range(n_vars):
                        if i != target_idx:
                            new_row[i] = future_exog_scaled[step, exog_ptr]
                            exog_ptr += 1

                win = torch.cat([win[1:], new_row.unsqueeze(0)], dim=0)

            all_runs.append(preds)

    model.eval()

    all_runs  = np.array(all_runs)          # (n_samples, horizon)
    mean_pred = all_runs.mean(axis=0)
    std_pred  = all_runs.std(axis=0)

    lower_95 = mean_pred - 1.96 * std_pred
    upper_95 = mean_pred + 1.96 * std_pred
    lower_80 = mean_pred - 1.28 * std_pred
    upper_80 = mean_pred + 1.28 * std_pred

    return mean_pred, lower_95, upper_95, lower_80, upper_80


# ============================================================
# FORECAST-ONLY  (loads saved bundle, no retraining)
# ============================================================
def _forecast_only_ann(params, horizon, future_rain, future_mean_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in ("static/ann_bundle.pt", "static/ann_weights.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train the model first.")

    bundle = torch.load("static/ann_bundle.pt", weights_only=False)

    model = ANNModel(
        input_dim     = bundle["input_dim"],
        hidden_layer_1= bundle["hidden_layer_1"],
        hidden_layer_2= bundle["hidden_layer_2"],
        dropout_1     = bundle["dropout_1"],
        dropout_2     = bundle["dropout_2"],
    ).to(device)
    model.load_state_dict(
        torch.load("static/ann_weights.pt", map_location=device, weights_only=False)
    )

    saved_exog_cols = bundle["exog_cols"]
    all_cols        = bundle["all_cols"]
    target_idx      = bundle["target_idx"]
    x_scaler_mean   = bundle["x_scaler_mean"]
    x_scaler_scale  = bundle["x_scaler_scale"]
    y_scaler_mean   = bundle["y_scaler_mean"]
    y_scaler_scale  = bundle["y_scaler_scale"]
    last_window     = bundle["last_window"].to(device)   # (LAGS*n_vars,) flat
    time_labels     = bundle["time_labels"]
    n_vars          = len(all_cols)
    lags            = bundle["lags"]

    # ── Scale future exog if needed ──────────────────────────────────
    future_exog_scaled = None
    if saved_exog_cols:
        future_exog_dict = params.get("future_exog")
        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))

        if future_exog_dict:
            # Dynamic mode — any column names supported
            for j, col in enumerate(saved_exog_cols):
                if col not in future_exog_dict:
                    raise ValueError(f"Missing future values for column '{col}'.")
                col_idx = all_cols.index(col)
                vals = np.array(future_exog_dict[col][:horizon], dtype=float)
                future_exog_scaled[:, j] = (
                    (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
                )

        elif future_rain is not None and future_mean_t is not None:
            # Legacy mode — name-based mapping
            for j, col in enumerate(saved_exog_cols):
                col_lower = col.lower()
                col_idx   = all_cols.index(col)
                if "rain" in col_lower:
                    vals = np.array(future_rain[:horizon], dtype=float)
                elif "temp" in col_lower or "mean_t" in col_lower:
                    vals = np.array(future_mean_t[:horizon], dtype=float)
                else:
                    raise ValueError(
                        f"Cannot auto-map column '{col}'. "
                        f"Pass values via future_exog dict instead."
                    )
                future_exog_scaled[:, j] = (
                    (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
                )
        else:
            raise ValueError(
                "Model was trained with exogenous variables. "
                "Provide future_exog dict or future_rain/future_mean_t."
            )

    # ── Reset seed for reproducible MC Dropout ───────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── MC Dropout forecast ──────────────────────────────────────────
    mean_pred, lower_95, upper_95, lower_80, upper_80 = _forecast_autoregressive_mc(
        model, last_window, horizon,
        n_vars, target_idx,
        future_exog_scaled,
        x_scaler_mean, x_scaler_scale,
        y_scaler_mean, y_scaler_scale,
        lags, device, n_samples=200,
    )

    future_labels = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    df = pd.DataFrame({
        "Period"        : future_labels,
        "Forecast"      : np.round(mean_pred, 3),
        "Lower 80%"     : np.round(lower_80,  3),
        "Upper 80%"     : np.round(upper_80,  3),
        "Lower 95%"     : np.round(lower_95,  3),
        "Upper 95%"     : np.round(upper_95,  3),
        "Interval (95%)": [f"[{l:.1f}, {u:.1f}]"
                           for l, u in zip(lower_95, upper_95)],
    })

    return {"forecast_table": df}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_ann(
    data,
    params,
    horizon,
    frequency,
    future_rain   = None,
    future_mean_t = None,
    mode          : str = "train",
    log_callback  = None,
):
    """
    ANN model — lag-based, raw values, no log differencing.
    Mirrors the LSTM/GRU/RNN implementation exactly.

    Parameters
    ----------
    data         : Raw DataFrame
    target_col   : read from params["target_col"]
    time_col     : read from params["time_col"]
    exog_cols    : read from params["exog_cols"]
    mode         : "train" | "forecast"
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ------------------------------------------------------------------
    # FORECAST-ONLY MODE
    # ------------------------------------------------------------------
    if mode == "forecast":
        if future_rain is None:
            future_rain = params.get("future_rain")
        if future_mean_t is None:
            future_mean_t = params.get("future_mean_t")
        return _forecast_only_ann(params, horizon, future_rain, future_mean_t)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if data is None:
        raise ValueError("data must be provided in train mode.")

    target_col = params.get("target_col", "Yield")
    time_col   = params.get("time_col",   "Year")
    exog_cols  = params.get("exog_cols",  [])
    if exog_cols is None:
        exog_cols = []

    has_exogenous = len(exog_cols) > 0

    log(f"📊 Mode: {'Multivariate' if has_exogenous else 'Univariate'}")
    log(f"   Study variable : {target_col}")
    log(f"   Time column    : {time_col}")
    log(f"   Exog variables : {exog_cols if exog_cols else 'None'}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    LAGS         = params.get("ann_lags") or params.get("lags", 3)
    split        = params.get("split", 0.85)
    use_tune     = params.get("auto_tune_ann", True)

    hidden_layer_1 = params.get("hidden_layer_1", 16)
    hidden_layer_2 = params.get("hidden_layer_2", 8)
    dropout_1      = params.get("dropout_1", 0.2)
    dropout_2      = params.get("dropout_2", 0.3)
    lr             = params.get("learning_rate", 0.002)
    weight_decay   = params.get("weight_decay",  1e-4)
    epochs         = params.get("epochs",        200)
    patience       = params.get("patience",      30)
    huber_delta    = params.get("huber_delta",   1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    cols_needed = [time_col, target_col] + exog_cols
    df = data[cols_needed].copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df = clean_dataframe(df)

    # Variable order: target first, then exog (mirrors LSTM/GRU/RNN)
    all_var_cols = [target_col] + exog_cols
    n_vars       = len(all_var_cols)
    target_idx   = 0

    # ------------------------------------------------------------------
    # Build lag features on raw values  (NO log differencing)
    # ------------------------------------------------------------------
    for lag in range(1, LAGS + 1):
        df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
        for ec in exog_cols:
            df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)

    df = df.dropna().reset_index(drop=True)

    log(f"   Dataset size   : {len(df)} samples")

    # Build X: (N, LAGS, n_vars) then flatten to (N, LAGS*n_vars)
    def build_X(df_):
        samples = []
        for i in range(len(df_)):
            seq = []
            for lag in range(1, LAGS + 1):
                row = [df_[f"__lag_{lag}_{target_col}"].iloc[i]]
                for ec in exog_cols:
                    row.append(df_[f"__lag_{lag}_{ec}"].iloc[i])
                seq.append(row)
            samples.append(seq)
        arr = np.array(samples, dtype=np.float32)   # (N, LAGS, n_vars)
        return arr.reshape(arr.shape[0], -1)         # (N, LAGS*n_vars)

    X = build_X(df)
    y = df[target_col].values.astype(np.float32)
    input_dim = X.shape[1]

    log(f"   Input shape    : {X.shape}  (samples, lags×variables)")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    split_idx       = int(split * len(X))
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]

    # ------------------------------------------------------------------
    # Scale X  (fit on train only)
    # ------------------------------------------------------------------
    x_scaler  = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)
    X_all_s   = x_scaler.transform(X)

    # ------------------------------------------------------------------
    # Scale y  (fit on train only)
    # ------------------------------------------------------------------
    y_scaler  = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # AUTO-TUNE
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # AUTO-TUNE
    # ------------------------------------------------------------------
    if use_tune:
        log("\n🔧 Auto-tuning ANN hyperparameters...")

        lags_values = list(range(
            params.get("ann_lags_min", 2),
            params.get("ann_lags_max", 5) + 1,
            params.get("ann_lags_step", 1),
        ))

        global_best_val  = float("inf")
        global_best_p    = None
        global_best_lags = LAGS

        for lag_val in lags_values:
            log(f"\n🔁 Trying lags={lag_val}...")

            # ── Rebuild X for this lag ──────────────────────────────
            df_lag = data[cols_needed].copy()
            if pd.api.types.is_numeric_dtype(df_lag[time_col]):
                df_lag = df_lag.sort_values(time_col).reset_index(drop=True)
            else:
                df_lag = df_lag.reset_index(drop=True)
            df_lag = clean_dataframe(df_lag)

            for l in range(1, lag_val + 1):
                df_lag[f"__lag_{l}_{target_col}"] = df_lag[target_col].shift(l)
                for ec in exog_cols:
                    df_lag[f"__lag_{l}_{ec}"] = df_lag[ec].shift(l)
            df_lag = df_lag.dropna().reset_index(drop=True)

            X_lag = []
            for i in range(len(df_lag)):
                seq = []
                for l in range(1, lag_val + 1):
                    row = [df_lag[f"__lag_{l}_{target_col}"].iloc[i]]
                    for ec in exog_cols:
                        row.append(df_lag[f"__lag_{l}_{ec}"].iloc[i])
                    seq.append(row)
                X_lag.append(seq)
            X_lag = np.array(X_lag, dtype=np.float32)
            X_lag = X_lag.reshape(X_lag.shape[0], -1)   # ✅ Flatten: (N, lag*n_vars)
            y_lag = df_lag[target_col].values.astype(np.float32)
            lag_input_dim = X_lag.shape[1]               # ✅ input_dim for this lag

            # ── Split & scale ───────────────────────────────────────
            sp = int(split * len(X_lag))
            X_tr_lag, X_va_lag = X_lag[:sp], X_lag[sp:]
            y_tr_lag, y_va_lag = y_lag[:sp], y_lag[sp:]

            sc_x = StandardScaler()
            X_tr_s = sc_x.fit_transform(X_tr_lag)
            X_va_s = sc_x.transform(X_va_lag)

            sc_y = StandardScaler()
            y_tr_s = sc_y.fit_transform(y_tr_lag.reshape(-1, 1)).flatten()
            y_va_s = sc_y.transform(y_va_lag.reshape(-1, 1)).flatten()

            # ── Inner grid search (no lags in grid) ─────────────────
            val_sp = int(0.8 * len(X_tr_s))
            best_p = auto_tune_ann(
                X_tr_s[:val_sp], y_tr_s[:val_sp],
                X_tr_s[val_sp:], y_tr_s[val_sp:],
                device, lag_input_dim, params,   # ✅ Pass correct input_dim
                log_callback=log_callback,
            )

            if best_p is None:
                continue

            # ── Re-evaluate on full val set ──────────────────────────
            model_tmp = ANNModel(
                input_dim=lag_input_dim,
                hidden_layer_1=best_p["hidden_layer_1"],
                hidden_layer_2=best_p["hidden_layer_2"],
                dropout_1=best_p["dropout_1"],
                dropout_2=best_p["dropout_2"],
            ).to(device)

            crit_tmp = nn.HuberLoss(delta=best_p["huber_delta"])
            opt_tmp  = torch.optim.Adam(model_tmp.parameters(),
                                        lr=best_p["learning_rate"],
                                        weight_decay=best_p["weight_decay"])

            Xtr_t = torch.tensor(X_tr_s, dtype=torch.float32, device=device)
            ytr_t = torch.tensor(y_tr_s, dtype=torch.float32, device=device)
            Xva_t = torch.tensor(X_va_s, dtype=torch.float32, device=device)
            yva_t = torch.tensor(y_va_s, dtype=torch.float32, device=device)

            pat_tmp, best_ep_tmp = 0, float("inf")
            for _ in range(best_p["epochs"]):
                model_tmp.train(); opt_tmp.zero_grad()
                crit_tmp(model_tmp(Xtr_t), ytr_t).backward(); opt_tmp.step()
                model_tmp.eval()
                with torch.no_grad():
                    vl = crit_tmp(model_tmp(Xva_t), yva_t).item()
                if vl < best_ep_tmp:
                    best_ep_tmp, pat_tmp = vl, 0
                else:
                    pat_tmp += 1
                    if pat_tmp >= best_p["patience"]:
                        break

            if best_ep_tmp < global_best_val:
                global_best_val  = best_ep_tmp
                global_best_p    = best_p.copy()
                global_best_lags = lag_val
                log(f"   🏆 New global best: lags={lag_val}, val={global_best_val:.6f}")

        # ── Apply best found ─────────────────────────────────────────
        LAGS           = global_best_lags
        best_p         = global_best_p
        hidden_layer_1 = best_p["hidden_layer_1"]
        hidden_layer_2 = best_p["hidden_layer_2"]
        dropout_1      = best_p["dropout_1"]
        dropout_2      = best_p["dropout_2"]
        lr             = best_p["learning_rate"]
        weight_decay   = best_p["weight_decay"]
        huber_delta    = best_p["huber_delta"]
        patience       = best_p["patience"]
        epochs         = best_p["epochs"]

        log(f"\n✅ Best overall: lags={LAGS}, params={best_p}")

        # ── Rebuild final X/y/scalers with best LAGS ─────────────────
        for lag in range(1, LAGS + 1):
            df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
            for ec in exog_cols:
                df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)
        df = df.dropna().reset_index(drop=True)

        X = build_X(df)          # already flattened by build_X
        y = df[target_col].values.astype(np.float32)
        input_dim = X.shape[1]   # ✅ Recalculate with best LAGS

        split_idx       = int(split * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_s = x_scaler.fit_transform(X_train)
        X_test_s  = x_scaler.transform(X_test)
        X_all_s   = x_scaler.transform(X)
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        # Rebuild tensors
        X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
        y_te_t = torch.tensor(y_test_s,  dtype=torch.float32, device=device)

    else:
        LAGS = params.get("ann_lags") or params.get("lags", 3)
        X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
        y_te_t = torch.tensor(y_test_s,  dtype=torch.float32, device=device)
    # Reset seed before model creation for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ------------------------------------------------------------------
    # Build tensors & model
    # ------------------------------------------------------------------
    

    model = ANNModel(
        input_dim=input_dim,
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
    ).to(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn    = nn.HuberLoss(delta=huber_delta)
    best_loss  = float("inf")
    best_state = None
    counter    = 0

    log(f"\n🏋️ Training for up to {epochs} epochs (patience={patience})...")
    for ep in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward(); optimizer.step()

        if (ep + 1) % 50 == 0 or ep == 0:
            log(f"   Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                log(f"   Early stopping at epoch {ep+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ------------------------------------------------------------------
    # Evaluation — inverse-scale to actual values
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        tr_preds_s = model(X_tr_t).cpu().numpy()
        te_preds_s = model(X_te_t).cpu().numpy()

    pred_tr   = y_scaler.inverse_transform(tr_preds_s.reshape(-1, 1)).flatten()
    pred_te   = y_scaler.inverse_transform(te_preds_s.reshape(-1, 1)).flatten()
    actual_tr = y_train.copy()
    actual_te = y_test.copy()

    min_tr = min(len(actual_tr), len(pred_tr))
    min_te = min(len(actual_te), len(pred_te))
    actual_tr, pred_tr = actual_tr[:min_tr], pred_tr[:min_tr]
    actual_te, pred_te = actual_te[:min_te], pred_te[:min_te]

    time_labels_seq = df[time_col].tolist()
    train_times = time_labels_seq[:split_idx][:min_tr]
    test_times  = time_labels_seq[split_idx:][:min_te]

    def mape(a, p):
        return float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    os.makedirs("static", exist_ok=True)

    # Save last window as flat (LAGS*n_vars,) — already flat from X_all_s
    last_window_np = X_all_s[-1]                    # (LAGS*n_vars,)
    last_window_t  = torch.tensor(last_window_np, dtype=torch.float32)

    # Auto future time labels
    last_t = df[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig_sequence = data[time_col].tolist()
        first_val = orig_sequence[0]
        try:
            cycle_len = orig_sequence[1:].index(first_val) + 1
        except ValueError:
            cycle_len = len(orig_sequence)
        cycle = orig_sequence[:cycle_len]
        last_cycle_pos = cycle.index(last_t)
        future_time_labels = [
            cycle[(last_cycle_pos + i) % cycle_len]
            for i in range(1, horizon + 1)
        ]

    torch.save(model.state_dict(), "static/ann_weights.pt")
    torch.save({
        # Architecture
        "input_dim"     : input_dim,
        "hidden_layer_1": hidden_layer_1,
        "hidden_layer_2": hidden_layer_2,
        "dropout_1"     : dropout_1,
        "dropout_2"     : dropout_2,
        # X scaler
        "x_scaler_mean" : x_scaler.mean_,
        "x_scaler_scale": x_scaler.scale_,
        # y scaler
        "y_scaler_mean" : float(y_scaler.mean_[0]),
        "y_scaler_scale": float(y_scaler.scale_[0]),
        # Column metadata
        "target_col"    : target_col,
        "time_col"      : time_col,
        "exog_cols"     : exog_cols,
        "all_cols"      : all_var_cols,
        "target_idx"    : target_idx,
        # Forecast state
        "last_window"   : last_window_t,
        "last_time"     : df[time_col].iloc[-1],
        "time_labels"   : future_time_labels,
        # Metadata
        "params"        : params,
        "has_exogenous" : has_exogenous,
        "lags"          : LAGS,
        "n_vars"        : n_vars,
    }, "static/ann_bundle.pt")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    train_x = list(range(len(actual_tr)))
    test_x  = list(range(len(actual_tr), len(actual_tr) + len(actual_te)))

    ax.plot(train_x, actual_tr, color="steelblue",  label="Actual (Train)")
    ax.plot(train_x, pred_tr,   color="steelblue",  label="Predicted (Train)",
            linestyle="--", marker="o", ms=4, alpha=0.8)
    ax.plot(test_x,  actual_te, color="darkorange", label="Actual (Test)")
    ax.plot(test_x,  pred_te,   color="darkorange", label="Predicted (Test)",
            linestyle="--", marker="o", ms=4, alpha=0.8)

    ax.axvline(x=len(actual_tr) - 1, color="black", linestyle=":", linewidth=2,
                label="Train/Test Split")

    all_times = train_times + test_times
    all_x     = train_x + test_x
    step = max(1, len(all_x) // 12) if len(all_x) > 24 else 1
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_times[::step], rotation=45, ha="right")

    ax.set_title("ANN — Training & Testing Set")
    ax.set_xlabel(time_col); ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/ann_train.png", dpi=120)
    plt.savefig("static/ann_test.png",  dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------
    model_config = {
        "hidden_layer_1": hidden_layer_1,
        "hidden_layer_2": hidden_layer_2,
        "dropout_1"     : round(dropout_1, 3),
        "dropout_2"     : round(dropout_2, 3),
        "learning_rate" : lr,
        "weight_decay"  : weight_decay,
        "huber_delta"   : round(huber_delta, 2),
        "patience"      : patience,
        "lags"          : LAGS,
    }

    results = {
        "model_key"   : "ann",
        "model_name"  : "ANN",
        "frequency"   : frequency,
        "horizon"     : horizon,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",
        "model_config": model_config,
        "auto_tuned"  : use_tune,

        "rmse_train"  : round(float(root_mean_squared_error(actual_tr, pred_tr)), 3),
        "mae_train"    : round(float(mean_absolute_error(actual_tr, pred_tr)), 3),
        "mape_train"  : round(mape(actual_tr, pred_tr), 2),

        "rmse_test"   : round(float(root_mean_squared_error(actual_te, pred_te)), 3),
        "mae_test"     : round(float(mean_absolute_error(actual_te, pred_te)), 3),
        "mape_test"   : round(mape(actual_te, pred_te), 2),
        "residuals"   : (actual_tr - pred_tr).tolist(),   # ← ADD THIS

        "train_table" : pd.DataFrame({
            time_col    : train_times,
            "Actual"    : actual_tr,
            "Predicted" : pred_tr,
        }),

        "test_table"  : pd.DataFrame({
            time_col    : test_times,
            "Actual"    : actual_te,
            "Predicted" : pred_te,
        }),

        "forecast_table": None,
    }

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {results['rmse_train']}, MAE: {results['mae_train']}, MAPE: {results['mape_train']}%")
    log(f"   Test  — RMSE: {results['rmse_test']},  MAE: {results['mae_test']},  MAPE: {results['mape_test']}%")

    return results