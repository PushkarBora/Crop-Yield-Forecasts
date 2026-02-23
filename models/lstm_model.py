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
    """
    Safely handles missing values for any column type.
    - Numeric columns: linear interpolation → forward fill → backward fill
    - String/object columns: forward fill → backward fill
    Works regardless of whether samay is numeric (2001, 2002) or string (January, February).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
    df[obj_cols] = df[obj_cols].ffill().bfill()

    return df.dropna()
# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class LSTMModel(nn.Module):
    """
    LSTM for time-series regression.
    Input shape : (batch, seq_len, n_features)   — seq_len = LAGS
    Output shape: (batch,)  — single-step prediction (scaled)
    """
    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),          # ← extra dropout for MC sampling
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]              # take last time-step
        return self.fc(out).squeeze(-1)


# ============================================================
# AUTO-TUNE  (grid search over user-defined ranges)
# ============================================================
def auto_tune_lstm(
    X_train, y_train, X_val, y_val,
    device, n_features, params,
    log_callback=None
):
    import itertools

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting LSTM Grid Search Auto-Tuning")

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, decimals=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, decimals))
            v += step
        return vals

    # ── Read lags range from params ──────────────────────────────────
    lags_values = irange(
        params.get("lstm_lags_min",  2),
        params.get("lstm_lags_max",  5),
        params.get("lstm_lags_step", 1),
    )

    grid = {
        "hidden_size":   irange(params["lstm_hidden_size_min"],
                                params["lstm_hidden_size_max"],
                                params["lstm_hidden_size_step"]),
        "num_layers":    irange(params["lstm_num_layers_min"],
                                params["lstm_num_layers_max"],
                                params["lstm_num_layers_step"]),
        "dropout":       frange(params["lstm_dropout_min"],
                                params["lstm_dropout_max"],
                                params["lstm_dropout_step"], 3),
        "learning_rate": frange(params["lstm_learning_rate_min"],
                                params["lstm_learning_rate_max"],
                                params["lstm_learning_rate_step"], 6),
        "weight_decay":  frange(params["lstm_weight_decay_min"],
                                params["lstm_weight_decay_max"],
                                params["lstm_weight_decay_step"], 7),
        "huber_delta":   frange(params["lstm_huber_delta_min"],
                                params["lstm_huber_delta_max"],
                                params["lstm_huber_delta_step"], 2),
        "patience":      irange(params["lstm_patience_min"],
                                params["lstm_patience_max"],
                                params["lstm_patience_step"]),
        "epochs":        irange(params["lstm_epochs_min"],
                                params["lstm_epochs_max"],
                                params["lstm_epochs_step"]),
    }

    # NOTE: lags grid search is handled at the outer level in run_lstm
    # because changing lags requires rebuilding X entirely.
    # Here we just tune model/training hyperparameters.

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
            model = LSTMModel(
                n_features=n_features,
                hidden_size=p["hidden_size"],
                num_layers=p["num_layers"],
                dropout=p["dropout"],
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
def _forecast_autoregressive_mc(model, last_window, horizon,
                                  n_vars, target_idx,
                                  future_exog_scaled,
                                  x_scaler_mean, x_scaler_scale,
                                  y_scaler_mean, y_scaler_scale,
                                  device, n_samples=200):
    """
    Monte Carlo Dropout forecast.
    last_window shape : (LAGS, n_vars)  — X-scaled
    Returns: mean_pred, lower_95, upper_95, lower_80, upper_80
    """
    all_runs = []
    model.train()   # activate dropout for MC sampling

    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            win = last_window.clone()   # (LAGS, n_vars)

            for step in range(horizon):
                pred_scaled = model(win.unsqueeze(0)).item()
                pred_actual = pred_scaled * y_scaler_scale + y_scaler_mean
                preds.append(pred_actual)

                # Roll window
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
def _forecast_only_lstm(params, horizon, future_rain, future_mean_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in ("static/lstm_bundle.pt", "static/lstm_weights.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train the model first.")

    bundle = torch.load("static/lstm_bundle.pt", weights_only=False)

    model = LSTMModel(
        n_features  = bundle["n_features"],
        hidden_size = bundle["hidden_size"],
        num_layers  = bundle["num_layers"],
        dropout     = bundle["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load("static/lstm_weights.pt", map_location=device, weights_only=False)
    )

    saved_exog_cols = bundle["exog_cols"]       # [] if univariate
    all_cols        = bundle["all_cols"]         # [target, exog1, ...]
    target_idx      = bundle["target_idx"]
    x_scaler_mean   = bundle["x_scaler_mean"]
    x_scaler_scale  = bundle["x_scaler_scale"]
    y_scaler_mean   = bundle["y_scaler_mean"]
    y_scaler_scale  = bundle["y_scaler_scale"]
    last_window     = bundle["last_window"].to(device)
    time_labels     = bundle["time_labels"]
    n_vars          = len(all_cols)

    # ── Scale future exog if needed ──────────────────────────────────
    # ── Scale future exog if needed ──────────────────────────────────
    future_exog_scaled = None
    if saved_exog_cols:
        future_exog_dict = params.get("future_exog")  # dict: {col: [vals]}

        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))

        if future_exog_dict:
            # ✅ NEW: Dynamic mode — any column names supported
            for j, col in enumerate(saved_exog_cols):
                if col not in future_exog_dict:
                    raise ValueError(f"Missing future values for column '{col}'.")
                col_idx = all_cols.index(col)
                vals = np.array(future_exog_dict[col][:horizon], dtype=float)
                future_exog_scaled[:, j] = (
                    (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
                )

        elif future_rain is not None and future_mean_t is not None:
            # Legacy mode — name-based mapping (rain/temp only)
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
        device, n_samples=200,
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
def run_lstm(
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
    LSTM model — lag-based, raw values, no log differencing.
    Mirrors the transformer implementation exactly.

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
        return _forecast_only_lstm(params, horizon, future_rain, future_mean_t)

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
    LAGS         = params.get("lstm_lags") or params.get("lags", 3)
    split        = params.get("split", 0.85)
    use_tune     = params.get("auto_tune_lstm", True)

    hidden_size  = params.get("hidden_size",   8)
    num_layers   = params.get("num_layers",    2)
    dropout      = params.get("dropout",       0.3)
    lr           = params.get("learning_rate", 0.003)
    weight_decay = params.get("weight_decay",  1e-4)
    epochs       = params.get("epochs",        200)
    patience     = params.get("patience",      30)
    huber_delta  = params.get("huber_delta",   1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    cols_needed = [time_col, target_col] + exog_cols
    df = data[cols_needed].copy()
    # Safe universal approach
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)  # preserve user's original order
    df = clean_dataframe(df)

    # Variable order: target first, then exog (mirrors transformer)
    all_var_cols = [target_col] + exog_cols
    n_vars       = len(all_var_cols)
    target_idx   = 0

    # ------------------------------------------------------------------
    # Build lag features on raw values
    # ------------------------------------------------------------------
    for lag in range(1, LAGS + 1):
        df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
        for ec in exog_cols:
            df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)

    df = df.dropna().reset_index(drop=True)

    log(f"   Dataset size   : {len(df)} samples")

    # Build X: (N, LAGS, n_vars)  — seq[0]=lag1, seq[-1]=most_recent_lag
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
        return np.array(samples, dtype=np.float32)   # (N, LAGS, n_vars)

    X = build_X(df)
    y = df[target_col].values.astype(np.float32)

    log(f"   Input shape    : {X.shape}  (samples, lags, variables)")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    split_idx       = int(split * len(X))
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]

    # ------------------------------------------------------------------
    # Scale X  (fit on train only)
    # ------------------------------------------------------------------
    x_scaler = StandardScaler()
    n_s, n_l, n_v = X_train.shape
    X_train_s = x_scaler.fit_transform(X_train.reshape(-1, n_v)).reshape(X_train.shape)
    X_test_s  = x_scaler.transform(X_test.reshape(-1, n_v)).reshape(X_test.shape)
    X_all_s   = x_scaler.transform(X.reshape(-1, n_v)).reshape(X.shape)

    # ------------------------------------------------------------------
    # Scale y  (fit on train only)
    # ------------------------------------------------------------------
    y_scaler  = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # AUTO-TUNE
    # ------------------------------------------------------------------
    if use_tune:
        log("\n🔧 Auto-tuning LSTM hyperparameters...")
        val_sp = int(0.8 * len(X_train_s))
        best_p = auto_tune_lstm(
            X_train_s[:val_sp], y_train_s[:val_sp],
            X_train_s[val_sp:], y_train_s[val_sp:],
            device, n_v, params,
            log_callback=log_callback,
        )
        hidden_size  = best_p["hidden_size"]
        num_layers   = best_p["num_layers"]
        dropout      = best_p["dropout"]
        lr           = best_p["learning_rate"]
        weight_decay = best_p["weight_decay"]
        huber_delta  = best_p["huber_delta"]
        patience     = best_p["patience"]
        epochs       = best_p["epochs"]
        log("✅ Using auto-tuned parameters")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # ------------------------------------------------------------------
    # Build tensors & model
    # ------------------------------------------------------------------
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test_s,  dtype=torch.float32, device=device)

    model = LSTMModel(
        n_features=n_v,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
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

    last_window_np = X_all_s[-1]   # (LAGS, n_vars) — X-scaled
    last_window_t  = torch.tensor(last_window_np, dtype=torch.float32)

    # Auto future time labels
    last_t = df[time_col].iloc[-1]
    try:
    # Numeric: just increment
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
    # String: detect cycle length then continue from last position
        orig_sequence = data[time_col].tolist()
    
    # Detect cycle length (e.g. 12 for months, 7 for weeks, 4 for quarters)
        first_val = orig_sequence[0]
        try:
            cycle_len = orig_sequence[1:].index(first_val) + 1
        except ValueError:
            cycle_len = len(orig_sequence)  # no repeat found, use full length
    
    # Find position of last_t within one cycle
        cycle = orig_sequence[:cycle_len]
        last_cycle_pos = cycle.index(last_t)
    
    # Continue from next position in cycle
        future_time_labels = [
            cycle[(last_cycle_pos + i) % cycle_len]
            for i in range(1, horizon + 1)
        ]

    torch.save(model.state_dict(), "static/lstm_weights.pt")
    torch.save({
        # Architecture
        "n_features"    : n_v,
        "hidden_size"   : hidden_size,
        "num_layers"    : num_layers,
        "dropout"       : dropout,
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
    }, "static/lstm_bundle.pt")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

# ✅ Use numeric index for x-axis to avoid repeating label stacking
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

    # ✅ Show actual month labels as x-ticks (evenly spaced, no overlap)
    all_times = train_times + test_times
    all_x     = train_x + test_x
    step = max(1, len(all_x) // 12)  # show ~12 labels max
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_times[::step], rotation=45, ha="right")

    ax.set_title("LSTM — Training & Testing Set")
    ax.set_xlabel(time_col); ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/lstm_train.png", dpi=120)
    plt.savefig("static/lstm_test.png",  dpi=120)
    plt.close()
    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------
    model_config = {
        "hidden_size"  : hidden_size,
        "num_layers"   : num_layers,
        "dropout"      : dropout,
        "learning_rate": lr,
        "weight_decay" : weight_decay,
        "huber_delta"  : round(huber_delta, 2),
        "patience"     : patience,
        "lags"         : LAGS,
    }

    results = {
        "model_key"   : "lstm",
        "model_name"  : "LSTM",
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