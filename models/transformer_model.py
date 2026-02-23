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

# ------------------------------------------------------------
# Torch DLL fix (Windows)
# ------------------------------------------------------------
try:
    torch_dll_path = os.path.join(
        sys.exec_prefix, "Lib", "site-packages", "torch", "lib"
    )
    if os.path.exists(torch_dll_path):
        os.add_dll_directory(torch_dll_path)
except Exception:
    pass  # Non-Windows systems

import torch
import torch.nn as nn

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

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
class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only Transformer for time-series regression.
    Input shape : (batch, seq_len, n_features)
    Output shape: (batch,)  — single-step prediction (scaled)
    """

    def __init__(self, n_features, d_model, nhead, num_layers,
                 dim_feedforward, dropout, activation):
        super().__init__()

        self.embedding  = nn.Linear(n_features, d_model)
        self.embed_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.embed_drop(self.embedding(x))
        x = self.transformer(x)
        x = self.norm(x[:, -1, :])      # Take last position
        return self.fc(x).squeeze(-1)


# ============================================================
# AUTO-TUNE  (grid search over user-defined ranges)
# ============================================================
def auto_tune_transformer(
    X_train, y_train, X_val, y_val,
    device, n_features, params,
    max_epochs=100, log_callback=None
):
    import itertools

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting Transformer Grid Search Auto-Tuning")

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, decimals=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, decimals))
            v += step
        return vals

    grid = {
        "d_model":         irange(params["transformer_d_model_min"],
                                  params["transformer_d_model_max"],
                                  params["transformer_d_model_step"]),
        "nhead":           irange(params["transformer_nhead_min"],
                                  params["transformer_nhead_max"],
                                  params["transformer_nhead_step"]),
        "num_layers":      irange(params["transformer_num_layers_min"],
                                  params["transformer_num_layers_max"],
                                  params["transformer_num_layers_step"]),
        "dim_feedforward": irange(params["transformer_dim_feedforward_min"],
                                  params["transformer_dim_feedforward_max"],
                                  params["transformer_dim_feedforward_step"]),
        "dropout":         frange(params["transformer_dropout_min"],
                                  params["transformer_dropout_max"],
                                  params["transformer_dropout_step"], 3),
        "activation":      ["relu", "gelu"],
        "learning_rate":   frange(params["transformer_learning_rate_min"],
                                  params["transformer_learning_rate_max"],
                                  params["transformer_learning_rate_step"], 6),
        "weight_decay":    frange(params["transformer_weight_decay_min"],
                                  params["transformer_weight_decay_max"],
                                  params["transformer_weight_decay_step"], 7),
        "huber_delta":     frange(params["transformer_huber_delta_min"],
                                  params["transformer_huber_delta_max"],
                                  params["transformer_huber_delta_step"], 2),
        "patience":        irange(params["transformer_patience_min"],
                                  params["transformer_patience_max"],
                                  params["transformer_patience_step"]),
    }

    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    log(f"🔢 Total combinations: {len(combos)}")

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val,   dtype=torch.float32, device=device)
    yva = torch.tensor(y_val,   dtype=torch.float32, device=device)

    best_val, best_p = float("inf"), None

    for idx, p in enumerate(combos, 1):
        if idx % 100 == 0:
            log(f"   Progress: {idx}/{len(combos)} ({idx/len(combos)*100:.1f}%)")
        if p["d_model"] % p["nhead"] != 0:
            continue
        try:
            model = TimeSeriesTransformer(
                n_features=n_features,
                **{k: p[k] for k in
                   ["d_model","nhead","num_layers","dim_feedforward","dropout","activation"]}
            ).to(device)

            crit = nn.HuberLoss(delta=p["huber_delta"])
            opt  = torch.optim.Adam(model.parameters(),
                                    lr=p["learning_rate"],
                                    weight_decay=p["weight_decay"])
            best_ep, pat = float("inf"), 0

            for _ in range(max_epochs):
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
        except Exception:
            continue

    log(f"✅ Best params: {best_p}  |  Val Loss: {best_val:.6f}")
    return best_p


# ============================================================
# AUTOREGRESSIVE FORECAST HELPER
# ============================================================
def _forecast_autoregressive_mc(model, last_window, horizon,
                                  n_vars, target_idx,
                                  future_exog_scaled,
                                  x_scaler_mean, x_scaler_scale,
                                  y_scaler_mean, y_scaler_scale,
                                  device, n_samples=100):
    """
    Monte Carlo Dropout forecast — returns mean + 95% interval.
    Runs n_samples stochastic forward passes with dropout active.
    """
    all_runs = []

    # Enable dropout at inference by setting model to TRAIN mode
    model.train()

    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            win = last_window.clone()

            for step in range(horizon):
                pred_scaled = model(win.unsqueeze(0)).item()
                pred_actual = pred_scaled * y_scaler_scale + y_scaler_mean
                preds.append(pred_actual)

                new_row = win[-1].clone()
                new_row[target_idx] = (pred_actual - x_scaler_mean[target_idx]) / x_scaler_scale[target_idx]

                if future_exog_scaled is not None:
                    exog_ptr = 0
                    for i in range(n_vars):
                        if i != target_idx:
                            new_row[i] = future_exog_scaled[step, exog_ptr]
                            exog_ptr += 1

                win = torch.cat([win[1:], new_row.unsqueeze(0)], dim=0)

            all_runs.append(preds)

    model.eval()

    all_runs = np.array(all_runs)  # (n_samples, horizon)
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
def _forecast_only(params, horizon, future_exog_df, target_col, exog_cols):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in ("static/transformer_bundle.pt", "static/transformer_weights.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train the model first.")

    bundle = torch.load("static/transformer_bundle.pt", weights_only=False)

    model = TimeSeriesTransformer(
        n_features      = bundle["n_features"],
        d_model         = bundle["d_model"],
        nhead           = bundle["nhead"],
        num_layers      = bundle["num_layers"],
        dim_feedforward = bundle["dim_feedforward"],
        dropout         = bundle["dropout"],
        activation      = bundle["activation"],
    ).to(device)
    model.load_state_dict(
        torch.load("static/transformer_weights.pt", map_location=device, weights_only=False)
    )

    saved_exog_cols = bundle["exog_cols"]
    all_cols        = bundle["all_cols"]
    target_idx      = bundle["target_idx"]
    x_scaler_mean   = bundle["x_scaler_mean"]
    x_scaler_scale  = bundle["x_scaler_scale"]
    y_scaler_mean   = bundle["y_scaler_mean"]
    y_scaler_scale  = bundle["y_scaler_scale"]
    last_window     = bundle["last_window"].to(device)
    time_labels     = bundle["time_labels"]
    n_vars          = len(all_cols)

    future_exog_scaled = None
    if saved_exog_cols:
        if future_exog_df is None or future_exog_df.empty:
            raise ValueError(f"Model needs exogenous variables: {saved_exog_cols}")
        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))
        for j, col in enumerate(saved_exog_cols):
            if col not in future_exog_df.columns:
                raise ValueError(f"Missing column '{col}' in future exog data.")
            col_idx = all_cols.index(col)
            vals = future_exog_df[col].values[:horizon].astype(float)
            future_exog_scaled[:, j] = (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
    
    # ✅ Use MC Dropout forecast
    mean_pred, lower_95, upper_95, lower_80, upper_80 = _forecast_autoregressive_mc(
        model, last_window, horizon,
        n_vars, target_idx,
        future_exog_scaled,
        x_scaler_mean, x_scaler_scale,
        y_scaler_mean, y_scaler_scale,
        device,
        n_samples=200  # More samples = smoother intervals
    )

    future_labels = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    df = pd.DataFrame({
        "Period"        : future_labels,
        "Forecast"      : np.round(mean_pred, 3),
        "Lower 80%"     : np.round(lower_80, 3),
        "Upper 80%"     : np.round(upper_80, 3),
        "Lower 95%"     : np.round(lower_95, 3),
        "Upper 95%"     : np.round(upper_95, 3),
        "Interval (95%)": [f"[{l:.1f}, {u:.1f}]"
                           for l, u in zip(lower_95, upper_95)],
    })

    return {"forecast_table": df}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_transformer(
    data               : pd.DataFrame,
    params             : dict,
    horizon            : int,
    frequency          : str,
    target_col         : str   = None,
    time_col           : str   = None,
    exog_cols          : list  = None,
    future_exog        : pd.DataFrame = None,
    future_time_labels : list  = None,
    mode               : str   = "train",
    log_callback       = None,
):
    """
    Generalized Transformer model — lag-based, no log differencing.

    Parameters
    ----------
    data         : Raw DataFrame uploaded by user
    target_col   : Column name of the study variable  (e.g. "Yield")
    time_col     : Column name of time index           (e.g. "Year")
    exog_cols    : Exogenous variable column names. [] = univariate.
    future_exog  : DataFrame with exog values for forecast mode.
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
        return _forecast_only(
            params=params,
            horizon=horizon,
            future_exog_df=future_exog,
            target_col=target_col,
            exog_cols=exog_cols or [],
        )

    # ------------------------------------------------------------------
    # Validate required inputs
    # ------------------------------------------------------------------
    if data is None:
        raise ValueError("data must be provided in train mode.")
    if target_col is None or time_col is None:
        raise ValueError("target_col and time_col must be specified.")
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
    #✅ Fix:
    LAGS = params.get("transformer_lags") or params.get("lags", 3)
    split        = params.get("split", 0.85)
    use_tune     = params.get("auto_tune_transformer", True)

    d_model      = params.get("d_model", 8)
    nhead        = params.get("nhead", 2)
    num_layers   = params.get("num_layers", 2)
    dim_ff       = params.get("dim_feedforward", 16)
    dropout      = params.get("dropout", 0.2)
    activation   = params.get("activation", "gelu")
    lr           = params.get("learning_rate", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)
    epochs       = params.get("epochs", 200)
    patience     = params.get("patience", 30)
    huber_delta  = params.get("huber_delta", 1.0)

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

    # Variable order: target first, then exog
    all_var_cols = [target_col] + exog_cols
    n_vars       = len(all_var_cols)
    target_idx   = 0   # target is always first

    # ------------------------------------------------------------------
    # Build lag features directly on raw values
    # ------------------------------------------------------------------
    for lag in range(1, LAGS + 1):
        df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
        for ec in exog_cols:
            df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)

    df = df.dropna().reset_index(drop=True)

    log(f"   Dataset size   : {len(df)} samples")

    # Build X: (N, LAGS, n_vars)  — seq[0]=lag1, seq[1]=lag2, ...
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
    y = df[target_col].values.astype(np.float32)     # raw target values

    log(f"   Input shape    : {X.shape}  (samples, lags, variables)")

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    split_idx        = int(split * len(X))
    X_train, X_test  = X[:split_idx],  X[split_idx:]
    y_train, y_test  = y[:split_idx],  y[split_idx:]

    # ------------------------------------------------------------------
    # Scale X features  (fit on train, apply to test)
    # ------------------------------------------------------------------
    x_scaler  = StandardScaler()
    n_s, n_l, n_v = X_train.shape
    X_train_s = x_scaler.fit_transform(X_train.reshape(-1, n_v)).reshape(X_train.shape)
    X_test_s  = x_scaler.transform(X_test.reshape(-1, n_v)).reshape(X_test.shape)
    X_all_s   = x_scaler.transform(X.reshape(-1, n_v)).reshape(X.shape)

    # ------------------------------------------------------------------
    # Scale y  (fit on train only)
    # ------------------------------------------------------------------
    y_scaler      = StandardScaler()
    y_train_s     = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s      = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # AUTO-TUNE (optional)
    # ------------------------------------------------------------------
    if use_tune:
        log("\n🔧 Auto-tuning Transformer hyperparameters...")
        val_sp = int(0.8 * len(X_train_s))
        best_p = auto_tune_transformer(
            X_train_s[:val_sp], y_train_s[:val_sp],
            X_train_s[val_sp:], y_train_s[val_sp:],
            device, n_v, params,
            max_epochs=100, log_callback=log_callback,
        )
        d_model, nhead, num_layers   = best_p["d_model"], best_p["nhead"], best_p["num_layers"]
        dim_ff, dropout, activation  = best_p["dim_feedforward"], best_p["dropout"], best_p["activation"]
        lr, weight_decay             = best_p["learning_rate"], best_p["weight_decay"]
        huber_delta                  = best_p["huber_delta"]
        patience                     = best_p["patience"]
        log("✅ Using auto-tuned parameters")

    # ------------------------------------------------------------------
    # Build tensors & model
    # ------------------------------------------------------------------
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test_s,  dtype=torch.float32, device=device)

    model = TimeSeriesTransformer(
        n_features=n_v, d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_ff,
        dropout=dropout, activation=activation,
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
    # Evaluation — inverse-scale predictions to actual values
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        tr_preds_s = model(X_tr_t).cpu().numpy()
        te_preds_s = model(X_te_t).cpu().numpy()

    # Inverse-scale: scaled → actual
    pred_tr = y_scaler.inverse_transform(tr_preds_s.reshape(-1, 1)).flatten()
    pred_te = y_scaler.inverse_transform(te_preds_s.reshape(-1, 1)).flatten()
    actual_tr = y_train.copy()
    actual_te = y_test.copy()

    # Safety trim (lengths should already match, but just in case)
    min_tr = min(len(actual_tr), len(pred_tr))
    min_te = min(len(actual_te), len(pred_te))
    actual_tr, pred_tr = actual_tr[:min_tr], pred_tr[:min_tr]
    actual_te, pred_te = actual_te[:min_te], pred_te[:min_te]

    # Time labels (aligned with y, which starts at row 0 of df after lag-dropna)
    time_labels_seq = df[time_col].tolist()
    train_times = time_labels_seq[:split_idx][:min_tr]
    test_times  = time_labels_seq[split_idx:][:min_te]

    def mape(a, p):
        return float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    os.makedirs("static", exist_ok=True)

    # Last window = last sample's input window in X-scaled space
    last_window_np = X_all_s[-1]   # (LAGS, n_vars)
    last_window_t  = torch.tensor(last_window_np, dtype=torch.float32)

    # Auto-generate future time labels if not provided
    if future_time_labels is None:
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

    torch.save(model.state_dict(), "static/transformer_weights.pt")
    torch.save({
        # Architecture
        "n_features"     : n_v,
        "d_model"        : d_model,
        "nhead"          : nhead,
        "num_layers"     : num_layers,
        "dim_feedforward": dim_ff,
        "dropout"        : dropout,
        "activation"     : activation,
        # X scaler (per-variable, shape = n_vars)
        "x_scaler_mean"  : x_scaler.mean_,
        "x_scaler_scale" : x_scaler.scale_,
        # y scaler (scalar)
        "y_scaler_mean"  : float(y_scaler.mean_[0]),
        "y_scaler_scale" : float(y_scaler.scale_[0]),
        # Column metadata
        "target_col"     : target_col,
        "time_col"       : time_col,
        "exog_cols"      : exog_cols,
        "all_cols"       : all_var_cols,
        "target_idx"     : target_idx,
        # Forecasting state
        "last_window"    : last_window_t,
        "last_time"      : df[time_col].iloc[-1],
        "time_labels"    : future_time_labels,
        # Training info
        "params"         : params,
        "has_exogenous"  : has_exogenous,
        "lags"           : LAGS,
    }, "static/transformer_bundle.pt")

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

    ax.set_title("Transformer — Training & Testing Set")
    ax.set_xlabel(time_col); ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/transformer_train.png", dpi=120)
    plt.savefig("static/transformer_test.png",  dpi=120)
    plt.close()
    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------
    model_config = {
        "d_model"        : d_model,
        "nhead"          : nhead,
        "num_layers"     : num_layers,
        "dim_feedforward": dim_ff,
        "dropout"        : dropout,
        "activation"     : activation,
        "learning_rate"  : lr,
        "weight_decay"   : weight_decay,
        "huber_delta"    : round(huber_delta, 2),
        "patience"       : patience,
        "lags"           : LAGS,
    }

    results = {
        "model_key"   : "transformer",
        "model_name"  : "Transformer (Encoder)",
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