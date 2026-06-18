import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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
# HELPER: build lag features for a given lag value
# ============================================================
def _build_lag_features(df_clean, lag, target_col, exog_cols):
    """
    Add lag columns to df_clean for the given lag value.
    Returns (df_lagged, X, y).
    """
    df_lag = df_clean.copy()
    for l in range(1, lag + 1):
        df_lag[f"__lag_{l}_{target_col}"] = df_lag[target_col].shift(l)
        for ec in exog_cols:
            df_lag[f"__lag_{l}_{ec}"] = df_lag[ec].shift(l)
    df_lag = df_lag.dropna().reset_index(drop=True)

    samples = []
    for i in range(len(df_lag)):
        seq = []
        for l in range(1, lag + 1):
            row = [df_lag[f"__lag_{l}_{target_col}"].iloc[i]]
            for ec in exog_cols:
                row.append(df_lag[f"__lag_{l}_{ec}"].iloc[i])
            seq.append(row)
        samples.append(seq)

    arr = np.array(samples, dtype=np.float32)       # (N, lag, n_vars)
    X   = arr.reshape(arr.shape[0], -1)              # (N, lag*n_vars)
    y   = df_lag[target_col].values.astype(np.float32)
    return df_lag, X, y


# ============================================================
# AUTO-TUNE (outer loop over lags, inner grid over KNN params)
# ============================================================
def auto_tune_knn(df_clean, target_col, exog_cols, split, params, log_callback=None):
    """
    Auto-tune KNN using USER-DEFINED grid search ranges from params.

    lags is searched in an OUTER loop because it changes the shape of X.
    KNN hyperparameters (n_neighbors, weights, metric, p) are searched
    in the inner loop using TimeSeriesSplit cross-validation.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from itertools import product as iproduct

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Starting KNN Grid Search Auto-Tuning")

    # ── Build value ranges ─────────────────────────────────────────────
    def irange(lo, hi, step):
        return list(range(lo, hi + 1, step))

    lags_values       = irange(params["knn_lags_min"],        params["knn_lags_max"],        params["knn_lags_step"])
    k_values          = irange(params["knn_k_min"],           params["knn_k_max"],           params["knn_k_step"])

    # Weights options from checkboxes
    weights_values = []
    if params.get("knn_weights_uniform", True):   weights_values.append("uniform")
    if params.get("knn_weights_distance", True):  weights_values.append("distance")
    if not weights_values:
        weights_values = ["uniform"]

    # Metric options from checkboxes
    metric_values = []
    if params.get("knn_metric_euclidean", True):  metric_values.append("euclidean")
    if params.get("knn_metric_manhattan", True):  metric_values.append("manhattan")
    if params.get("knn_metric_minkowski", False): metric_values.append("minkowski")
    if not metric_values:
        metric_values = ["euclidean"]

    # p (Minkowski power) — only relevant when metric="minkowski"
    p_values = irange(params.get("knn_p_min", 1), params.get("knn_p_max", 3), params.get("knn_p_step", 1))

    total = len(lags_values) * len(k_values) * len(weights_values) * len(metric_values)
    log(f"🔢 Total combinations: {total}")
    log(f"   lags    : {lags_values}")
    log(f"   k       : {k_values}")
    log(f"   weights : {weights_values}")
    log(f"   metric  : {metric_values}")

    best_overall = None
    best_rmse    = float('inf')

    # ── Outer loop over lags ────────────────────────────────────────────
    for lag_idx, lag in enumerate(lags_values, 1):
        log(f"  🔄 [{lag_idx}/{len(lags_values)}] Testing lags={lag}...")

        df_lag, X, y = _build_lag_features(df_clean, lag, target_col, exog_cols)

        split_idx  = int(split * len(X))
        X_tr, y_tr = X[:split_idx], y[:split_idx]

        x_sc   = StandardScaler()
        X_tr_s = x_sc.fit_transform(X_tr)
        y_sc   = StandardScaler()
        y_tr_s = y_sc.fit_transform(y_tr.reshape(-1, 1)).flatten()

        # Cap k to valid range (can't have more neighbours than samples)
        max_valid_k = max(1, len(X_tr_s) - 1)

        # ── TimeSeriesSplit ─────────────────────────────────────────────
        n_splits = min(5, max(2, len(X_tr_s) // 10))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        val_start = int(0.8 * len(X_tr_s))
        X_val_gs  = X_tr_s[val_start:]
        y_val_gs  = y_tr_s[val_start:]

        combos = list(iproduct(k_values, weights_values, metric_values))
        n_combos = len(combos)
        log(f"     📐 Training samples: {len(X_tr_s)} | "
            f"Val samples: {len(X_val_gs)} | "
            f"Combinations: {n_combos} | "
            f"TimeSeriesSplit(n_splits={n_splits})")

        best_lag_rmse   = float('inf')
        best_lag_params = None

        for combo_idx, (k, weights, metric) in enumerate(combos, 1):
            # Skip if k exceeds available training samples
            effective_k = min(k, max_valid_k)
            if effective_k < 1:
                continue

            # Use p=2 (Euclidean) for non-Minkowski, p varies for Minkowski
            p_list = p_values if metric == "minkowski" else [2]

            for p in p_list:
                try:
                    knn_cv = KNeighborsRegressor(
                        n_neighbors = effective_k,
                        weights     = weights,
                        metric      = metric,
                        p           = p,
                        n_jobs      = -1,
                    )

                    # ── TimeSeriesSplit cross-validation ──────────────
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_tr_s):
                        X_cv_tr, X_cv_val = X_tr_s[train_idx], X_tr_s[val_idx]
                        y_cv_tr, y_cv_val = y_tr_s[train_idx], y_tr_s[val_idx]
                        # Further cap k for smaller CV folds
                        fold_k = min(effective_k, len(X_cv_tr) - 1)
                        if fold_k < 1:
                            continue
                        knn_fold = KNeighborsRegressor(
                            n_neighbors=fold_k, weights=weights,
                            metric=metric, p=p, n_jobs=-1
                        )
                        knn_fold.fit(X_cv_tr, y_cv_tr)
                        fold_rmse = root_mean_squared_error(
                            y_cv_val, knn_fold.predict(X_cv_val)
                        )
                        cv_scores.append(fold_rmse)

                    if not cv_scores:
                        continue
                    mean_cv_rmse = float(np.mean(cv_scores))

                    # ── Final val score on temporal holdout ───────────
                    knn_cv.fit(X_tr_s[:val_start], y_tr_s[:val_start])
                    if len(X_val_gs) > 0:
                        val_rmse_combo = root_mean_squared_error(
                            y_val_gs, knn_cv.predict(X_val_gs)
                        )
                    else:
                        val_rmse_combo = mean_cv_rmse

                    p_str = f" p={p}" if metric == "minkowski" else ""
                    is_best = val_rmse_combo < best_lag_rmse
                    log(f"     [{combo_idx:>4}/{n_combos}] "
                        f"k={effective_k:<3} w={weights:<8} m={metric:<10}{p_str} "
                        f"| CV={mean_cv_rmse:.4f} | Val={val_rmse_combo:.4f}"
                        + (" ✅ NEW BEST" if is_best else ""))

                    if is_best:
                        best_lag_rmse   = val_rmse_combo
                        best_lag_params = {
                            'n_neighbors': effective_k,
                            'weights'    : weights,
                            'metric'     : metric,
                            'p'          : p,
                        }

                except Exception as e:
                    log(f"     [{combo_idx:>4}/{n_combos}] "
                        f"k={effective_k} w={weights} m={metric} "
                        f"| ❌ FAILED: {str(e)}")
                    continue

        if best_lag_params is None:
            log(f"     ⚠️  All combinations failed for lags={lag}, skipping.")
            continue

        log(f"     🏁 lags={lag} done → best: {best_lag_params} | "
            f"val RMSE: {best_lag_rmse:.4f}")

        if best_lag_rmse < best_rmse:
            best_rmse    = best_lag_rmse
            best_overall = {**best_lag_params, 'lags': lag}
            log(f"     🏆 Global best updated! lags={lag}, val RMSE={best_rmse:.4f}")

    log(f"✅ Best overall: {best_overall} | val RMSE: {best_rmse:.4f}")
    return best_overall


# ============================================================
# BOOTSTRAP PREDICTION INTERVALS
# ============================================================
def _bootstrap_forecast(model, X_last, horizon, n_vars, target_idx,
                         future_exog_scaled,
                         x_scaler_mean, x_scaler_scale,
                         y_scaler_mean, y_scaler_scale,
                         lags, residuals, n_boot=200):
    random.seed(SEED)
    np.random.seed(SEED)

    all_runs = []
    for _ in range(n_boot):
        preds = []
        win   = X_last.copy()

        for step in range(horizon):
            x_in        = win.reshape(1, -1)
            pred_scaled = model.predict(x_in)[0]
            noise       = np.random.choice(residuals)
            pred_actual = (pred_scaled + noise) * y_scaler_scale + y_scaler_mean
            preds.append(pred_actual)

            win_2d  = win.reshape(lags, n_vars)
            new_row = win_2d[-1].copy()
            new_row[target_idx] = (
                (pred_actual - x_scaler_mean[target_idx]) / x_scaler_scale[target_idx]
            )
            if future_exog_scaled is not None:
                exog_ptr = 0
                for i in range(n_vars):
                    if i != target_idx:
                        new_row[i] = future_exog_scaled[step, exog_ptr]
                        exog_ptr  += 1
            win_2d = np.vstack([win_2d[1:], new_row])
            win    = win_2d.flatten()

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
# FORECAST-ONLY
# ============================================================
def _forecast_only_knn(params, horizon, future_rain, future_mean_t):
    for path in ("static/knn_bundle.pkl", "static/knn_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Please train KNN first.")

    with open("static/knn_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("static/knn_model.pkl", "rb") as f:
        model  = pickle.load(f)

    saved_exog_cols = bundle["exog_cols"]
    all_cols        = bundle["all_cols"]
    target_idx      = bundle["target_idx"]
    x_scaler_mean   = bundle["x_scaler_mean"]
    x_scaler_scale  = bundle["x_scaler_scale"]
    y_scaler_mean   = bundle["y_scaler_mean"]
    y_scaler_scale  = bundle["y_scaler_scale"]
    last_window     = bundle["last_window"]
    time_labels     = bundle["time_labels"]
    lags            = bundle["lags"]
    n_vars          = len(all_cols)
    residuals       = bundle["residuals"]

    future_exog_scaled = None
    if saved_exog_cols:
        future_exog_dict   = params.get("future_exog")
        future_exog_scaled = np.zeros((horizon, len(saved_exog_cols)))

        if future_exog_dict:
            for j, col in enumerate(saved_exog_cols):
                if col not in future_exog_dict:
                    raise ValueError(f"Missing future values for column '{col}'.")
                col_idx = all_cols.index(col)
                vals    = np.array(future_exog_dict[col][:horizon], dtype=float)
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
def run_knn(
    data,
    params,
    horizon,
    frequency,
    future_rain   = None,
    future_mean_t = None,
    mode          : str = "train",
    log_callback  = None,
):
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    if mode == "forecast":
        if future_rain is None:
            future_rain = params.get("future_rain")
        if future_mean_t is None:
            future_mean_t = params.get("future_mean_t")
        return _forecast_only_knn(params, horizon, future_rain, future_mean_t)

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

    LAGS     = params.get("knn_lags") or params.get("lags", 3)
    split    = params.get("split", 0.85)
    use_tune = params.get("auto_tune_knn", True)

    # Default KNN hyperparameters (used when auto_tune_knn=False)
    n_neighbors = params.get("knn_n_neighbors",  5)
    weights     = params.get("knn_weights",  "uniform")
    metric      = params.get("knn_metric",   "euclidean")
    p           = params.get("knn_p",        2)

    log(f"   Lags           : {LAGS}")

    # ── Prepare clean dataframe ─────────────────────────────────────────
    cols_needed = [time_col, target_col] + exog_cols
    df_clean    = data[cols_needed].copy()
    if pd.api.types.is_numeric_dtype(df_clean[time_col]):
        df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
    else:
        df_clean = df_clean.reset_index(drop=True)
    df_clean = clean_dataframe(df_clean)

    all_var_cols = [target_col] + exog_cols
    n_vars       = len(all_var_cols)
    target_idx   = 0

    # ── AUTO-TUNE ────────────────────────────────────────────────────────
    if use_tune:
        log("\n🔧 Auto-tuning KNN hyperparameters...")
        best_p = auto_tune_knn(
            df_clean, target_col, exog_cols, split,
            params, log_callback=log_callback,
        )
        LAGS        = best_p['lags']
        n_neighbors = best_p['n_neighbors']
        weights     = best_p['weights']
        metric      = best_p['metric']
        p           = best_p['p']
        log(f"✅ Using auto-tuned: lags={LAGS}, k={n_neighbors}, "
            f"weights={weights}, metric={metric}, p={p}")

    # ── Build final lag features ─────────────────────────────────────────
    df, X, y = _build_lag_features(df_clean, LAGS, target_col, exog_cols)
    log(f"   Dataset size   : {len(df)} samples after lagging")
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

    # ── Cap k to valid range ─────────────────────────────────────────────
    n_neighbors = min(n_neighbors, len(X_train_s) - 1)
    if n_neighbors < 1:
        n_neighbors = 1

    # ── Fit final KNN ────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)

    knn = KNeighborsRegressor(
        n_neighbors = n_neighbors,
        weights     = weights,
        metric      = metric,
        p           = p,
        n_jobs      = -1,
    )
    knn.fit(X_train_s, y_train_s)

    # ── Predict & inverse-scale ──────────────────────────────────────────
    # ── Test predictions (standard) ─────────────────────────────────────────
    te_preds_s = knn.predict(X_test_s)
    pred_te    = y_scaler.inverse_transform(te_preds_s.reshape(-1, 1)).flatten()

    # ── Train predictions via Leave-One-Out to avoid distance=0 self-prediction ──
    # When weights='distance', KNN finds itself (d=0) → perfect fit → RMSE=0.
    # LOO gives a realistic train error instead.
    from sklearn.model_selection import LeaveOneOut
    loo      = LeaveOneOut()
    loo_preds = np.zeros(len(X_train_s))

    for train_idx, val_idx in loo.split(X_train_s):
        eff_k = min(n_neighbors, len(train_idx) - 1)
        if eff_k < 1:
            eff_k = 1
        knn_loo = KNeighborsRegressor(
            n_neighbors = eff_k,
            weights     = weights,
            metric      = metric,
            p           = p,
            n_jobs      = -1,
        )
        knn_loo.fit(X_train_s[train_idx], y_train_s[train_idx])
        loo_preds[val_idx] = knn_loo.predict(X_train_s[val_idx])

    pred_tr = y_scaler.inverse_transform(loo_preds.reshape(-1, 1)).flatten()
    actual_tr = y_train.copy()
    actual_te = y_test.copy()

    min_tr = min(len(actual_tr), len(pred_tr))
    min_te = min(len(actual_te), len(pred_te))
    actual_tr, pred_tr = actual_tr[:min_tr], pred_tr[:min_tr]
    actual_te, pred_te = actual_te[:min_te], pred_te[:min_te]

    time_labels_seq = df[time_col].tolist()
    train_times     = time_labels_seq[:split_idx][:min_tr]
    test_times      = time_labels_seq[split_idx:][:min_te]

    residuals_scaled = y_train_s[:min_tr] - loo_preds[:min_tr]

    # ── Metrics ──────────────────────────────────────────────────────────
    def mape(a, p_):
        return float(np.mean(np.abs((a - p_) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(actual_tr, pred_tr))
    mae_tr  = float(mean_absolute_error(actual_tr,  pred_tr))
    mape_tr = mape(actual_tr, pred_tr)

    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te  = float(mean_absolute_error(actual_te,  pred_te))
    mape_te = mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Train / Test fit plot ─────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)

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
    step_x    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1
    ax.set_xticks(all_x[::step_x])
    ax.set_xticklabels(all_times[::step_x], rotation=45, ha="right")

    ax.set_title(f"KNN (k={n_neighbors}, {weights}, {metric}) — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()

    plt.savefig("static/knn_train.png", dpi=120)
    plt.savefig("static/knn_test.png",  dpi=120)
    plt.close()

    # ── k vs RMSE diagnostic plot ─────────────────────────────────────────
    # Shows how RMSE changes with k on training data — helps interpret the chosen k
    k_range    = list(range(1, min(20, len(X_train_s))))
    k_rmses    = []
    for k_val in k_range:
        knn_tmp = KNeighborsRegressor(
            n_neighbors=k_val, weights=weights,
            metric=metric, p=p, n_jobs=-1
        )
        knn_tmp.fit(X_train_s, y_train_s)
        k_rmses.append(root_mean_squared_error(
            y_train_s, knn_tmp.predict(X_train_s)
        ))

    fig_k, ax_k = plt.subplots(figsize=(8, 4))
    ax_k.plot(k_range, k_rmses, color="steelblue", marker="o", ms=5)
    ax_k.axvline(x=n_neighbors, color="red", linestyle="--",
                 label=f"Chosen k={n_neighbors}")
    ax_k.set_xlabel("k (Number of Neighbours)")
    ax_k.set_ylabel("Train RMSE (scaled)")
    ax_k.set_title("KNN — k vs Train RMSE")
    ax_k.legend()
    plt.tight_layout()
    plt.savefig("static/knn_k_rmse.png", dpi=120)
    plt.close()

    # ── Save bundle ───────────────────────────────────────────────────────
    last_window_np = X_all_s[-1]

    last_t = df[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig_sequence  = data[time_col].tolist()
        first_val      = orig_sequence[0]
        try:
            cycle_len  = orig_sequence[1:].index(first_val) + 1
        except ValueError:
            cycle_len  = len(orig_sequence)
        cycle          = orig_sequence[:cycle_len]
        last_cycle_pos = cycle.index(last_t)
        future_time_labels = [
            cycle[(last_cycle_pos + i) % cycle_len]
            for i in range(1, horizon + 1)
        ]

    with open("static/knn_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    with open("static/knn_bundle.pkl", "wb") as f:
        pickle.dump({
            "x_scaler_mean"  : x_scaler.mean_,
            "x_scaler_scale" : x_scaler.scale_,
            "y_scaler_mean"  : float(y_scaler.mean_[0]),
            "y_scaler_scale" : float(y_scaler.scale_[0]),
            "target_col"     : target_col,
            "time_col"       : time_col,
            "exog_cols"      : exog_cols,
            "all_cols"       : all_var_cols,
            "target_idx"     : target_idx,
            "last_window"    : last_window_np,
            "last_time"      : df[time_col].iloc[-1],
            "time_labels"    : future_time_labels,
            "residuals"      : residuals_scaled,
            "params"         : params,
            "has_exogenous"  : has_exogenous,
            "lags"           : LAGS,
            "n_vars"         : n_vars,
        }, f)

    return {
        "model_key"   : "knn",
        "model_name"  : "KNN",
        "frequency"   : frequency,
        "horizon"     : horizon,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",

        "model_config": {
            "n_neighbors": n_neighbors,
            "weights"    : weights,
            "metric"     : metric,
            "p"          : p,
            "lags"       : LAGS,
        },
        "auto_tuned": use_tune,

        "rmse_train" : round(rmse_tr, 3),
        "mae_train"  : round(mae_tr,  3),
        "mape_train" : round(mape_tr, 2),

        "rmse_test"  : round(rmse_te, 3),
        "mae_test"   : round(mae_te,  3),
        "mape_test"  : round(mape_te, 2),
        "residuals"  : (actual_tr - pred_tr).tolist(),

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