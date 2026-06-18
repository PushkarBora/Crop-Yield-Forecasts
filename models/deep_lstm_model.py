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
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# HELPER
# ============================================================
def clean_dataframe(df):
    """
    Safely handles missing values for any column type.
    - Numeric columns: linear interpolation → forward fill → backward fill
    - String/object columns: forward fill → backward fill
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
    df[obj_cols] = df[obj_cols].ffill().bfill()
    return df.dropna()


# ============================================================
# RESIDUAL LSTM LAYER
# ============================================================
class ResidualLSTMLayer(nn.Module):
    """
    A single LSTM layer wrapped with a residual (skip) connection.

    Forward pass
    ────────────
    1.  Run input sequence through a single-layer nn.LSTM
    2.  Apply LayerNorm to the LSTM output sequence
    3.  Add the residual:
        •  If input_size == hidden_size  → direct skip  (out + x)
        •  If input_size != hidden_size  → project x first via a
           learned Linear(input_size → hidden_size) then add
    4.  Apply Dropout (MC-Dropout compatible)

    Why residuals in LSTMs?
    ───────────────────────
    In very deep stacked LSTMs (4+ layers), gradients can vanish during
    back-propagation through many recurrent steps AND many layers.
    Residual connections provide a direct gradient highway from the
    loss back to each layer — the same idea that made ResNet work for
    deep convolutional networks.  Even with only 2–3 layers the skip
    connection helps the network learn identity mappings easily, leading
    to faster convergence and more stable training.
    """

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.lstm       = nn.LSTM(input_size, hidden_size,
                                  num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)

        # Projection for the skip connection when dims differ
        self.projection = (
            nn.Linear(input_size, hidden_size, bias=False)
            if input_size != hidden_size else None
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)                    # (batch, seq_len, hidden_size)
        out    = self.layer_norm(out)

        # Residual
        skip = self.projection(x) if self.projection is not None else x
        out  = out + skip                        # (batch, seq_len, hidden_size)

        return self.dropout(out)                 # MC-Dropout compatible


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class DeepLSTMModel(nn.Module):
    """
    Deep LSTM with Residual Connections for time-series regression.

    Architecture
    ────────────
    • num_layers stacked ResidualLSTMLayer blocks.
      Each block: LSTM → LayerNorm → Residual Skip → Dropout
    • All hidden layers share the same hidden_size.
    • The projection inside ResidualLSTMLayer handles the
      input→hidden dimension mismatch only at the first layer.
    • After the last residual layer the final time-step is taken.
    • Deep FC head:
        Linear(hidden_size → hidden_size)  → ReLU → Dropout
        Linear(hidden_size → hidden_size//2) → ReLU → Dropout
        Linear(hidden_size//2 → 1)

    Input  : (batch, seq_len, n_features)   — seq_len = LAGS
    Output : (batch,)                        — single-step prediction (scaled)

    Key distinction from Stacked LSTM
    ──────────────────────────────────
    Stacked LSTM (ModuleList)  : sequential layers, no skip connections,
                                  each layer's output is the next layer's input.
    Deep LSTM (this model)     : each layer adds its output to its input
                                  via a residual connection — identity mappings
                                  can be learned trivially, enabling genuinely
                                  deep networks (4–8 layers) without gradient
                                  vanishing.  The deeper FC head also provides
                                  richer non-linear feature transformation at
                                  the output stage.
    """

    def __init__(self, n_features, hidden_size, num_layers, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Residual LSTM stack ──────────────────────────────────────
        layers = []
        for i in range(num_layers):
            in_size = n_features if i == 0 else hidden_size
            layers.append(ResidualLSTMLayer(in_size, hidden_size, dropout))
        self.residual_layers = nn.ModuleList(layers)

        # ── Deep FC head ─────────────────────────────────────────────
        half = max(hidden_size // 2, 1)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),                  # MC-Dropout
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Dropout(dropout),                  # MC-Dropout
            nn.Linear(half, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        for layer in self.residual_layers:
            x = layer(x)               # (batch, seq_len, hidden_size)

        last = x[:, -1, :]             # (batch, hidden_size)
        return self.fc_head(last).squeeze(-1)   # (batch,)


# ============================================================
# AUTO-TUNE  (grid search over user-defined ranges)
# ============================================================
def auto_tune_deep_lstm(
    X_train, y_train, X_val, y_val,
    device, n_features, params,
    log_callback=None
):
    import itertools

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting Deep LSTM Grid Search Auto-Tuning")

    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    def frange(lo, hi, step, decimals=6):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, decimals))
            v += step
        return vals

    grid = {
        "hidden_size"   : irange(params["dlstm_hidden_size_min"],
                                  params["dlstm_hidden_size_max"],
                                  params["dlstm_hidden_size_step"]),
        "num_layers"    : irange(params["dlstm_num_layers_min"],
                                  params["dlstm_num_layers_max"],
                                  params["dlstm_num_layers_step"]),
        "dropout"       : frange(params["dlstm_dropout_min"],
                                  params["dlstm_dropout_max"],
                                  params["dlstm_dropout_step"], 3),
        "learning_rate" : frange(params["dlstm_learning_rate_min"],
                                  params["dlstm_learning_rate_max"],
                                  params["dlstm_learning_rate_step"], 6),
        "weight_decay"  : frange(params["dlstm_weight_decay_min"],
                                  params["dlstm_weight_decay_max"],
                                  params["dlstm_weight_decay_step"], 7),
        "huber_delta"   : frange(params["dlstm_huber_delta_min"],
                                  params["dlstm_huber_delta_max"],
                                  params["dlstm_huber_delta_step"], 2),
        "patience"      : irange(params["dlstm_patience_min"],
                                  params["dlstm_patience_max"],
                                  params["dlstm_patience_step"]),
        "epochs"        : irange(params["dlstm_epochs_min"],
                                  params["dlstm_epochs_max"],
                                  params["dlstm_epochs_step"]),
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
            model = DeepLSTMModel(
                n_features  = n_features,
                hidden_size = p["hidden_size"],
                num_layers  = p["num_layers"],
                dropout     = p["dropout"],
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
def _forecast_autoregressive_mc(
    model, last_window, horizon,
    n_vars, target_idx,
    future_exog_scaled,
    x_scaler_mean, x_scaler_scale,
    y_scaler_mean, y_scaler_scale,
    device, n_samples=200
):
    """
    Monte Carlo Dropout autoregressive forecast.
    last_window shape : (LAGS, n_vars)  — X-scaled
    Returns: mean_pred, lower_95, upper_95, lower_80, upper_80
    """
    all_runs = []
    model.train()   # keep Dropout active for MC sampling

    with torch.no_grad():
        for _ in range(n_samples):
            preds = []
            win   = last_window.clone()   # (LAGS, n_vars)

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
def _forecast_only_deep_lstm(params, horizon, future_rain, future_mean_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in ("static/deep_lstm_bundle.pt", "static/deep_lstm_weights.pt"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Please train the model first."
            )

    bundle = torch.load("static/deep_lstm_bundle.pt", weights_only=False)

    model = DeepLSTMModel(
        n_features  = bundle["n_features"],
        hidden_size = bundle["hidden_size"],
        num_layers  = bundle["num_layers"],
        dropout     = bundle["dropout"],
    ).to(device)
    model.load_state_dict(
        torch.load("static/deep_lstm_weights.pt",
                   map_location=device, weights_only=False)
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

    # ── Scale future exog if needed ──────────────────────────────────
    future_exog_scaled = None
    if saved_exog_cols:
        future_exog_dict = params.get("future_exog")
        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))

        if future_exog_dict:
            for j, col in enumerate(saved_exog_cols):
                if col not in future_exog_dict:
                    raise ValueError(
                        f"Missing future values for column '{col}'."
                    )
                col_idx = all_cols.index(col)
                vals = np.array(future_exog_dict[col][:horizon], dtype=float)
                future_exog_scaled[:, j] = (
                    (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
                )

        elif future_rain is not None and future_mean_t is not None:
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

    mean_pred, lower_95, upper_95, lower_80, upper_80 = \
        _forecast_autoregressive_mc(
            model, last_window, horizon,
            n_vars, target_idx,
            future_exog_scaled,
            x_scaler_mean, x_scaler_scale,
            y_scaler_mean, y_scaler_scale,
            device, n_samples=200,
        )

    future_labels = (
        time_labels[:horizon] if time_labels else list(range(1, horizon + 1))
    )

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
def run_deep_lstm(
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
    Deep LSTM with Residual Connections — each LSTM layer adds its
    output back to its input via a skip connection (like ResNet),
    enabling genuinely deep stacks (4–8 layers) without gradient
    vanishing.  A deeper FC head (hidden → hidden//2 → 1) provides
    richer non-linear feature transformation at the output stage.
    MC-Dropout produces 80% and 95% uncertainty intervals.

    Mirrors the existing LSTM / Stacked LSTM / BD LSTM / Conv-LSTM
    implementations exactly in API and pipeline.

    Parameters
    ----------
    data          : Raw DataFrame
    target_col    : read from params["target_col"]
    time_col      : read from params["time_col"]
    exog_cols     : read from params["exog_cols"]
    mode          : "train" | "forecast"

    Deep LSTM-specific params keys
    ──────────────────────────────
    dlstm_lags                  : sequence length
    dlstm_lags_min/max/step     : lags grid-search range
    dlstm_hidden_size           : hidden units per residual layer
    dlstm_num_layers            : number of residual LSTM layers
    dlstm_dropout               : dropout (inside layers + FC head)
    dlstm_learning_rate         : Adam LR
    dlstm_weight_decay          : Adam weight decay
    dlstm_huber_delta           : Huber loss delta
    dlstm_patience              : early-stopping patience
    dlstm_epochs                : max training epochs
    auto_tune_deep_lstm         : bool — run grid search if True
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
        return _forecast_only_deep_lstm(params, horizon, future_rain, future_mean_t)

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
    LAGS        = params.get("dlstm_lags") or params.get("lags", 3)
    split       = params.get("split", 0.85)
    use_tune    = params.get("auto_tune_deep_lstm", True)

    hidden_size = params.get("dlstm_hidden_size",    8)
    num_layers  = params.get("dlstm_num_layers",     4)
    dropout     = params.get("dlstm_dropout",        0.3)
    lr          = params.get("dlstm_learning_rate",  0.003)
    weight_decay= params.get("dlstm_weight_decay",   1e-4)
    epochs      = params.get("dlstm_epochs",         200)
    patience    = params.get("dlstm_patience",       30)
    huber_delta = params.get("dlstm_huber_delta",    1.0)

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

    all_var_cols = [target_col] + exog_cols
    n_vars       = len(all_var_cols)
    target_idx   = 0

    # ------------------------------------------------------------------
    # Build lag features
    # ------------------------------------------------------------------
    for lag in range(1, LAGS + 1):
        df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
        for ec in exog_cols:
            df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)

    df = df.dropna().reset_index(drop=True)
    log(f"   Dataset size   : {len(df)} samples")

    # Build X: (N, LAGS, n_vars)
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
        return np.array(samples, dtype=np.float32)

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
    _, _, n_v = X_train.shape
    X_train_s = x_scaler.fit_transform(
        X_train.reshape(-1, n_v)).reshape(X_train.shape)
    X_test_s  = x_scaler.transform(
        X_test.reshape(-1, n_v)).reshape(X_test.shape)
    X_all_s   = x_scaler.transform(
        X.reshape(-1, n_v)).reshape(X.shape)

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
        log("\n🔧 Auto-tuning Deep LSTM hyperparameters...")

        lags_values = list(range(
            params.get("dlstm_lags_min",  2),
            params.get("dlstm_lags_max",  5) + 1,
            params.get("dlstm_lags_step", 1),
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
            y_lag = df_lag[target_col].values.astype(np.float32)

            # ── Split & scale ───────────────────────────────────────
            sp = int(split * len(X_lag))
            X_tr_lag, X_va_lag = X_lag[:sp], X_lag[sp:]
            y_tr_lag, y_va_lag = y_lag[:sp], y_lag[sp:]

            sc_x = StandardScaler()
            _, nl, nv = X_tr_lag.shape
            X_tr_s = sc_x.fit_transform(
                X_tr_lag.reshape(-1, nv)).reshape(X_tr_lag.shape)
            X_va_s = sc_x.transform(
                X_va_lag.reshape(-1, nv)).reshape(X_va_lag.shape)

            sc_y = StandardScaler()
            y_tr_s = sc_y.fit_transform(y_tr_lag.reshape(-1, 1)).flatten()
            y_va_s = sc_y.transform(y_va_lag.reshape(-1, 1)).flatten()

            # ── Inner grid search ────────────────────────────────────
            val_sp = int(0.8 * len(X_tr_s))
            best_p = auto_tune_deep_lstm(
                X_tr_s[:val_sp], y_tr_s[:val_sp],
                X_tr_s[val_sp:], y_tr_s[val_sp:],
                device, nv, params,
                log_callback=log_callback,
            )

            if best_p is None:
                continue

            # ── Re-evaluate on full val set ──────────────────────────
            model_tmp = DeepLSTMModel(
                n_features  = nv,
                hidden_size = best_p["hidden_size"],
                num_layers  = best_p["num_layers"],
                dropout     = best_p["dropout"],
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
                log(f"   🏆 New global best: lags={lag_val}, "
                    f"val={global_best_val:.6f}")

        # ── Apply best found ─────────────────────────────────────────
        LAGS        = global_best_lags
        best_p      = global_best_p
        hidden_size = best_p["hidden_size"]
        num_layers  = best_p["num_layers"]
        dropout     = best_p["dropout"]
        lr          = best_p["learning_rate"]
        weight_decay= best_p["weight_decay"]
        huber_delta = best_p["huber_delta"]
        patience    = best_p["patience"]
        epochs      = best_p["epochs"]

        log(f"\n✅ Best overall: lags={LAGS}, params={best_p}")

        # ── Rebuild final X/y/scalers with best LAGS ─────────────────
        df_final = data[cols_needed].copy()
        if pd.api.types.is_numeric_dtype(df_final[time_col]):
            df_final = df_final.sort_values(time_col).reset_index(drop=True)
        else:
            df_final = df_final.reset_index(drop=True)
        df_final = clean_dataframe(df_final)

        for lag in range(1, LAGS + 1):
            df_final[f"__lag_{lag}_{target_col}"] = df_final[target_col].shift(lag)
            for ec in exog_cols:
                df_final[f"__lag_{lag}_{ec}"] = df_final[ec].shift(lag)
        df_final = df_final.dropna().reset_index(drop=True)
        df = df_final

        X = build_X(df)
        y = df[target_col].values.astype(np.float32)

        split_idx       = int(split * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        _, _, n_v = X_train.shape
        X_train_s = x_scaler.fit_transform(
            X_train.reshape(-1, n_v)).reshape(X_train.shape)
        X_test_s  = x_scaler.transform(
            X_test.reshape(-1, n_v)).reshape(X_test.shape)
        X_all_s   = x_scaler.transform(
            X.reshape(-1, n_v)).reshape(X.shape)
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    else:
        LAGS = params.get("dlstm_lags") or params.get("lags", 3)
        _, _, n_v = X_train_s.shape

    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test_s,  dtype=torch.float32, device=device)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    _, _, n_v = X_train_s.shape

    model = DeepLSTMModel(
        n_features  = n_v,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
    ).to(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
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
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                log(f"   Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(
        {k: v.to(device) for k, v in best_state.items()}
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        tr_preds_s = model(X_tr_t).cpu().numpy()
        te_preds_s = model(X_te_t).cpu().numpy()

    pred_tr   = y_scaler.inverse_transform(
        tr_preds_s.reshape(-1, 1)).flatten()
    pred_te   = y_scaler.inverse_transform(
        te_preds_s.reshape(-1, 1)).flatten()
    actual_tr = y_train.copy()
    actual_te = y_test.copy()

    min_tr = min(len(actual_tr), len(pred_tr))
    min_te = min(len(actual_te), len(pred_te))
    actual_tr, pred_tr = actual_tr[:min_tr], pred_tr[:min_tr]
    actual_te, pred_te = actual_te[:min_te], pred_te[:min_te]

    time_labels_seq = df[time_col].tolist()
    train_times     = time_labels_seq[:split_idx][:min_tr]
    test_times      = time_labels_seq[split_idx:][:min_te]

    def mape(a, p):
        return float(
            np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100
        )

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    os.makedirs("static", exist_ok=True)

    last_window_np = X_all_s[-1]   # (LAGS, n_vars) — X-scaled
    last_window_t  = torch.tensor(last_window_np, dtype=torch.float32)

    # Auto future time labels
    last_t = df[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig_sequence  = data[time_col].tolist()
        first_val      = orig_sequence[0]
        try:
            cycle_len = orig_sequence[1:].index(first_val) + 1
        except ValueError:
            cycle_len = len(orig_sequence)
        cycle          = orig_sequence[:cycle_len]
        last_cycle_pos = cycle.index(last_t)
        future_time_labels = [
            cycle[(last_cycle_pos + i) % cycle_len]
            for i in range(1, horizon + 1)
        ]

    torch.save(model.state_dict(), "static/deep_lstm_weights.pt")
    torch.save({
        # Architecture
        "n_features"  : n_v,
        "hidden_size" : hidden_size,
        "num_layers"  : num_layers,
        "dropout"     : dropout,
        # X scaler
        "x_scaler_mean"  : x_scaler.mean_,
        "x_scaler_scale" : x_scaler.scale_,
        # y scaler
        "y_scaler_mean"  : float(y_scaler.mean_[0]),
        "y_scaler_scale" : float(y_scaler.scale_[0]),
        # Column metadata
        "target_col"  : target_col,
        "time_col"    : time_col,
        "exog_cols"   : exog_cols,
        "all_cols"    : all_var_cols,
        "target_idx"  : target_idx,
        # Forecast state
        "last_window" : last_window_t,
        "last_time"   : df[time_col].iloc[-1],
        "time_labels" : future_time_labels,
        # Metadata
        "params"        : params,
        "has_exogenous" : has_exogenous,
        "lags"          : LAGS,
    }, "static/deep_lstm_bundle.pt")

    # ------------------------------------------------------------------
    # Plot
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

    ax.axvline(x=len(actual_tr) - 1, color="black",
               linestyle=":", linewidth=2, label="Train/Test Split")

    all_times = train_times + test_times
    all_x     = train_x + test_x
    step      = max(1, len(all_x) // 12)
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_times[::step], rotation=45, ha="right")

    ax.set_title("Deep LSTM (Residual) — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/deep_lstm_train.png", dpi=120)
    plt.savefig("static/deep_lstm_test.png",  dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------
    model_config = {
        "hidden_size"      : hidden_size,
        "num_layers"       : num_layers,
        "dropout"          : dropout,
        "learning_rate"    : lr,
        "weight_decay"     : weight_decay,
        "huber_delta"      : round(huber_delta, 2),
        "patience"         : patience,
        "lags"             : LAGS,
        "residual_skip"    : True,
        "fc_head_depth"    : 3,           # hidden → hidden//2 → 1
    }

    results = {
        "model_key"   : "deep_lstm",
        "model_name"  : "Deep LSTM",
        "frequency"   : frequency,
        "horizon"     : horizon,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",
        "model_config": model_config,
        "auto_tuned"  : use_tune,

        "rmse_train"  : round(float(
            root_mean_squared_error(actual_tr, pred_tr)), 3),
        "mae_train"   : round(float(
            mean_absolute_error(actual_tr, pred_tr)), 3),
        "mape_train"  : round(mape(actual_tr, pred_tr), 2),

        "rmse_test"   : round(float(
            root_mean_squared_error(actual_te, pred_te)), 3),
        "mae_test"    : round(float(
            mean_absolute_error(actual_te, pred_te)), 3),
        "mape_test"   : round(mape(actual_te, pred_te), 2),

        "residuals"   : (actual_tr - pred_tr).tolist(),

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
    log(f"   Train — RMSE: {results['rmse_train']}, "
        f"MAE: {results['mae_train']}, "
        f"MAPE: {results['mape_train']}%")
    log(f"   Test  — RMSE: {results['rmse_test']},  "
        f"MAE: {results['mae_test']},  "
        f"MAPE: {results['mape_test']}%")

    return results