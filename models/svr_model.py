import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clean_dataframe(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
    df[obj_cols] = df[obj_cols].ffill().bfill()
    return df.dropna()
# ============================================================
# AUTO-TUNE  (grid search over user-defined ranges)
# ============================================================
def auto_tune_svr(X_train, y_train, X_val, y_val, params, log_callback=None):
    """
    Auto-tune SVR using USER-DEFINED grid search ranges from params.
    """
    from sklearn.model_selection import GridSearchCV

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting SVR Grid Search Auto-Tuning")

    # ── Build value ranges ─────────────────────────────────────────────
    def frange(lo, hi, step):
        vals, v = [], lo
        while v <= hi + 1e-9:
            vals.append(round(v, 4))
            v += step
        return vals

    C_values       = frange(params["svr_C_min"],       params["svr_C_max"],       params["svr_C_step"])
    epsilon_values = frange(params["svr_epsilon_min"], params["svr_epsilon_max"], params["svr_epsilon_step"])
    gamma_numeric  = frange(params["svr_gamma_min"],   params["svr_gamma_max"],   params["svr_gamma_step"])
    gamma_values   = ['scale', 'auto'] + gamma_numeric

    kernel_options = []
    if params.get('svr_kernel_rbf',     False): kernel_options.append('rbf')
    if params.get('svr_kernel_poly',    False): kernel_options.append('poly')
    if params.get('svr_kernel_sigmoid', False): kernel_options.append('sigmoid')
    if not kernel_options:
        log("⚠️  No kernels selected — defaulting to RBF")
        kernel_options = ['rbf']

    param_grid = {
        'kernel' : kernel_options,
        'C'      : C_values,
        'epsilon': epsilon_values,
        'gamma'  : gamma_values,
    }

    total = len(kernel_options) * len(C_values) * len(epsilon_values) * len(gamma_values)
    log(f"🔢 Total combinations: {total}")
    log(f"   kernel : {kernel_options}")
    log(f"   C      : {C_values}")
    log(f"   epsilon: {epsilon_values}")
    log(f"   gamma  : {gamma_values}")
    log("⏳ Running 3-fold cross-validation...")

    gs = GridSearchCV(
        SVR(), param_grid,
        cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)

    best = gs.best_params_
    log(f"✅ Best params: {best}  |  CV MSE: {-gs.best_score_:.4f}")
    val_rmse = root_mean_squared_error(y_val, gs.best_estimator_.predict(X_val))
    log(f"   Validation RMSE: {val_rmse:.4f}")
    return best


# ============================================================
# BOOTSTRAP PREDICTION INTERVALS
# ============================================================
def _bootstrap_forecast(model, X_last, horizon, n_vars, target_idx,
                         future_exog_scaled,
                         x_scaler_mean, x_scaler_scale,
                         y_scaler_mean, y_scaler_scale,
                         lags, residuals, n_boot=200):
    """
    Bootstrap forecast for SVR prediction intervals.
    Adds bootstrapped residual noise to produce interval estimates.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    all_runs = []
    for _ in range(n_boot):
        preds = []
        win = X_last.copy()   # (LAGS * n_vars,) flat, X-scaled

        for step in range(horizon):
            x_in = win.reshape(1, -1)
            pred_scaled = model.predict(x_in)[0]
            # Add bootstrapped residual noise
            noise = np.random.choice(residuals)
            pred_actual = (pred_scaled + noise) * y_scaler_scale + y_scaler_mean
            preds.append(pred_actual)

            # Roll window: reshape → update target → flatten back
            win_2d = win.reshape(lags, n_vars)
            new_row = win_2d[-1].copy()
            new_row[target_idx] = (
                (pred_actual - x_scaler_mean[target_idx]) / x_scaler_scale[target_idx]
            )
            if future_exog_scaled is not None:
                exog_ptr = 0
                for i in range(n_vars):
                    if i != target_idx:
                        new_row[i] = future_exog_scaled[step, exog_ptr]
                        exog_ptr += 1
            win_2d = np.vstack([win_2d[1:], new_row])
            win = win_2d.flatten()

        all_runs.append(preds)

    all_runs  = np.array(all_runs)
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
def _forecast_only_svr(params, horizon, future_rain, future_mean_t):
    for path in ("static/svr_bundle.pkl", "static/svr_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train SVR first.")

    with open("static/svr_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("static/svr_model.pkl", "rb") as f:
        model = pickle.load(f)

    saved_exog_cols  = bundle["exog_cols"]
    all_cols         = bundle["all_cols"]
    target_idx       = bundle["target_idx"]
    x_scaler_mean    = bundle["x_scaler_mean"]
    x_scaler_scale   = bundle["x_scaler_scale"]
    y_scaler_mean    = bundle["y_scaler_mean"]
    y_scaler_scale   = bundle["y_scaler_scale"]
    last_window      = bundle["last_window"]   # (LAGS*n_vars,) flat, X-scaled
    time_labels      = bundle["time_labels"]
    lags             = bundle["lags"]
    n_vars           = len(all_cols)
    residuals        = bundle["residuals"]     # scaled residuals for bootstrap

    # ── Scale future exog if needed ────────────────────────────────────
    future_exog_scaled = None
    if saved_exog_cols:
        future_exog_dict = params.get("future_exog")
        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))

        if future_exog_dict:
            for j, col in enumerate(saved_exog_cols):
                if col not in future_exog_dict:
                    raise ValueError(f"Missing future values for column '{col}'.")
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
                        "Pass values via future_exog dict instead."
                    )
                future_exog_scaled[:, j] = (
                    (vals - x_scaler_mean[col_idx]) / x_scaler_scale[col_idx]
                )
        else:
            raise ValueError(
                "Model was trained with exogenous variables. "
                "Provide future_exog dict or future_rain/future_mean_t."
            )

    # ── Bootstrap forecast ──────────────────────────────────────────────
    mean_pred, lower_95, upper_95, lower_80, upper_80 = _bootstrap_forecast(
        model, last_window, horizon, n_vars, target_idx,
        future_exog_scaled,
        x_scaler_mean, x_scaler_scale,
        y_scaler_mean, y_scaler_scale,
        lags, residuals, n_boot=200,
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
def run_svr(
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
    SVR model — raw lag features, no log differencing, no sliding window.
    Mirrors the LSTM/GRU/RNN/ANN implementation exactly.

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

    # ── FORECAST MODE ──────────────────────────────────────────────────
    if mode == "forecast":
        if future_rain is None:
            future_rain = params.get("future_rain")
        if future_mean_t is None:
            future_mean_t = params.get("future_mean_t")
        return _forecast_only_svr(params, horizon, future_rain, future_mean_t)

    # ── Validate ────────────────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ─────────────────────────────────────────────────
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

    # ── Hyperparameters ─────────────────────────────────────────────────
    LAGS      = params.get("svr_lags") or params.get("lags", 3)
    split     = params.get("split", 0.85)
    use_tune  = params.get("auto_tune_svr", True)

    kernel  = params.get("kernel",  "rbf")
    C       = params.get("C",       1.0)
    epsilon = params.get("epsilon", 0.1)
    gamma   = params.get("gamma",   "scale")

    log(f"   Lags           : {LAGS}")

    # ── Data preparation ────────────────────────────────────────────────
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
    target_idx   = 0

    # ── Build raw lag features  (NO log differencing) ───────────────────
    for lag in range(1, LAGS + 1):
        df[f"__lag_{lag}_{target_col}"] = df[target_col].shift(lag)
        for ec in exog_cols:
            df[f"__lag_{lag}_{ec}"] = df[ec].shift(lag)

    df = df.dropna().reset_index(drop=True)
    log(f"   Dataset size   : {len(df)} samples after lagging")

    # ── Build X flat: (N, LAGS*n_vars) ──────────────────────────────────
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

    log(f"   Input shape    : {X.shape}  (samples, lags×variables)")

    # ── Train / test split ───────────────────────────────────────────────
    split_idx       = int(split * len(X))
    X_train, X_test = X[:split_idx],  X[split_idx:]
    y_train, y_test = y[:split_idx],  y[split_idx:]

    # ── Scale X ──────────────────────────────────────────────────────────
    x_scaler  = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)
    X_all_s   = x_scaler.transform(X)

    # ── Scale y ──────────────────────────────────────────────────────────
    y_scaler  = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # ── AUTO-TUNE ────────────────────────────────────────────────────────
    if use_tune:
        log("\n🔧 Auto-tuning SVR hyperparameters...")
        val_sp = int(0.8 * len(X_train_s))
        best_p = auto_tune_svr(
            X_train_s[:val_sp], y_train_s[:val_sp],
            X_train_s[val_sp:], y_train_s[val_sp:],
            params, log_callback=log_callback,
        )
        kernel  = best_p['kernel']
        C       = best_p['C']
        epsilon = best_p['epsilon']
        gamma   = best_p['gamma']
        log(f"✅ Using auto-tuned SVR parameters")

    # ── Fit SVR ──────────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)

    svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    svr.fit(X_train_s, y_train_s)

    # ── Predict & inverse-scale ──────────────────────────────────────────
    tr_preds_s = svr.predict(X_train_s)
    te_preds_s = svr.predict(X_test_s)

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

    # ── Scaled residuals for bootstrap intervals ─────────────────────────
    residuals_scaled = y_train_s[:min_tr] - tr_preds_s[:min_tr]

    # ── Metrics ──────────────────────────────────────────────────────────
    def mape(a, p):
        return float(np.mean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(actual_tr, pred_tr))
    mae_tr   = float(mean_absolute_error(actual_tr, pred_tr))
    mape_tr = mape(actual_tr, pred_tr)

    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te   = float(mean_absolute_error(actual_te, pred_te))
    mape_te = mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Save bundle ───────────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)

    last_window_np = X_all_s[-1]   # (LAGS*n_vars,) flat, X-scaled

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

    with open("static/svr_model.pkl", "wb") as f:
        pickle.dump(svr, f)

    with open("static/svr_bundle.pkl", "wb") as f:
        pickle.dump({
            # Scaler info
            "x_scaler_mean"  : x_scaler.mean_,
            "x_scaler_scale" : x_scaler.scale_,
            "y_scaler_mean"  : float(y_scaler.mean_[0]),
            "y_scaler_scale" : float(y_scaler.scale_[0]),
            # Column metadata
            "target_col"     : target_col,
            "time_col"       : time_col,
            "exog_cols"      : exog_cols,
            "all_cols"       : all_var_cols,
            "target_idx"     : target_idx,
            # Forecast state
            "last_window"    : last_window_np,
            "last_time"      : df[time_col].iloc[-1],
            "time_labels"    : future_time_labels,
            # For bootstrap intervals
            "residuals"      : residuals_scaled,
            # Metadata
            "params"         : params,
            "has_exogenous"  : has_exogenous,
            "lags"           : LAGS,
            "n_vars"         : n_vars,
        }, f)

    # ── Plots ─────────────────────────────────────────────────────────────
    # ── Plots ─────────────────────────────────────────────────────────────
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

    ax.set_title("SVR — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()

    # Save as both so compare.html works for both sections
    plt.savefig("static/svr_train.png", dpi=120)
    plt.savefig("static/svr_test.png",  dpi=120)
    plt.close()
    # ── Results dict ──────────────────────────────────────────────────────
    return {
        "model_key"   : "svr",
        "model_name"  : "SVR",
        "frequency"   : frequency,
        "horizon"     : horizon,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",

        "model_config": {
            "kernel" : kernel,
            "C"      : C,
            "epsilon": epsilon,
            "gamma"  : gamma if isinstance(gamma, str) else round(float(gamma), 3),
            "lags"   : LAGS,
        },
        "auto_tuned": use_tune,

        "rmse_train" : round(rmse_tr, 3),
        "mae_train"   : round(mae_tr,   3),
        "mape_train" : round(mape_tr,  2),

        "rmse_test"  : round(rmse_te, 3),
        "mae_test"    : round(mae_te,   3),
        "mape_test"  : round(mape_te,  2),

        "train_table": pd.DataFrame({
            time_col   : train_times,
            "Actual"   : actual_tr,
            "Predicted": pred_tr,
        }),

        "test_table": pd.DataFrame({
            time_col   : test_times,
            "Actual"   : actual_te,
            "Predicted": pred_te,
        }),

        "forecast_table": None,
    }