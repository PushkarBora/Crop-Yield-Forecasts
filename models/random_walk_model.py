import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FORECAST-ONLY  (loads saved bundle, no retraining)
# ============================================================
def _forecast_only_rw(params, horizon):
    if not os.path.exists("static/rw_bundle.pkl"):
        raise FileNotFoundError(
            "Random Walk model not trained yet. Please train the model first before forecasting."
        )

    with open("static/rw_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    last_value   = bundle["last_value"]
    drift        = bundle["drift"]            # float (0.0 if no drift)
    resid_std    = bundle["resid_std"]        # std of one-step residuals
    time_labels  = bundle["time_labels"]
    seasonal     = bundle["seasonal"]
    season_s     = bundle["season_s"]
    tail_values  = bundle["tail_values"]      # last `season_s` values for seasonal RW

    mean_vals  = []
    lower_95   = []
    upper_95   = []
    lower_80   = []
    upper_80   = []

    for h in range(1, horizon + 1):
        if seasonal and season_s > 0:
            # Seasonal naïve: ŷ_{T+h} = y_{T+h-s}
            idx = (h - 1) % season_s
            fc  = float(tail_values[-(season_s - idx)])
        else:
            # RW (with or without drift): ŷ_{T+h} = y_T + h * drift
            fc = last_value + h * drift

        # h-step prediction intervals: σ * sqrt(h)
        se = resid_std * np.sqrt(h)
        mean_vals.append(fc)
        lower_95.append(fc - 1.96 * se)
        upper_95.append(fc + 1.96 * se)
        lower_80.append(fc - 1.28 * se)
        upper_80.append(fc + 1.28 * se)

    future_labels = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    df = pd.DataFrame({
        "Period"        : future_labels,
        "Forecast"      : np.round(mean_vals, 3),
        "Lower 80%"     : np.round(lower_80,  3),
        "Upper 80%"     : np.round(upper_80,  3),
        "Lower 95%"     : np.round(lower_95,  3),
        "Upper 95%"     : np.round(upper_95,  3),
        "Interval (95%)": [f"[{l:.1f}, {u:.1f}]"
                           for l, u in zip(lower_95, upper_95)],
    })

    return {"forecast_table": df}


# ============================================================
# HELPER — parameter estimates table
# ============================================================
def _param_estimates_df(drift, resid_std, n, use_drift, seasonal, season_s):
    """
    Returns a 2-column DataFrame: Parameter | Value
    (matches the TBATS-style param table — no std err / z / p columns
     because Random Walk has no estimated coefficients in the MLE sense)
    """
    rows = []

    if seasonal:
        rows.append(("Model Type",   f"Seasonal Naïve (s={season_s})"))
        rows.append(("Seasonal Period (s)", season_s))
    elif use_drift:
        rows.append(("Model Type",   "Random Walk with Drift"))
        rows.append(("Drift (μ)",    round(float(drift), 6)))
    else:
        rows.append(("Model Type",   "Pure Random Walk (no drift)"))
        rows.append(("Drift (μ)",    0.0))

    rows.append(("Residual Std Dev (σ)", round(float(resid_std), 4)))
    rows.append(("Number of Observations", int(n)))

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ============================================================
# HELPER — information criteria
# (RW has no likelihood-based IC; we report MAE/RMSE on residuals only)
# ============================================================
def _info_criteria(resid_std, n):
    """
    Random Walk has no AIC/BIC from MLE.
    We return the only meaningful summary statistics available.
    """
    criteria = {}
    try:
        criteria["Residual Std Dev"] = round(float(resid_std), 4)
    except Exception:
        pass
    try:
        criteria["N (training obs)"] = int(n)
    except Exception:
        pass
    return criteria


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_random_walk(
    data        : pd.DataFrame,
    params      : dict,
    horizon     : int,
    frequency   : str,
    mode        : str = "train",
    log_callback      = None,
):
    """
    Random Walk / Naïve Benchmark model.

    Variants (controlled by params):
      - Pure Random Walk   : ŷ_{t+1} = y_t
      - RW with Drift      : ŷ_{t+1} = y_t + μ  (μ = mean of first differences)
      - Seasonal Naïve     : ŷ_{t+h} = y_{t+h-s} (last observed same season)

    Always univariate — no exogenous variable support.
    Rolling one-step-ahead forecast on the test set.
    Prediction intervals scale as σ√h.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ───────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_rw(params=params, horizon=horizon)

    # ── Validate ────────────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ──────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")

    # ── Hyperparameters ──────────────────────────────────────────────
    split      = params.get("split", 0.85)
    use_drift  = params.get("rw_drift",   False)   # include drift term
    seasonal   = params.get("rw_seasonal", False)   # seasonal naïve
    season_s   = int(params.get("rw_seasonal_period",
                                params.get("rw_s", 12)))  # seasonal period

    # Seasonal naïve overrides drift
    if seasonal:
        use_drift = False

    log(f"📊 Random Walk — Univariate")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")
    log("=" * 60)
    if seasonal:
        log(f"   Variant        : Seasonal Naïve (s={season_s})")
    elif use_drift:
        log(f"   Variant        : Random Walk with Drift")
    else:
        log(f"   Variant        : Pure Random Walk (no drift)")
    log("=" * 60)

    # ── Data preparation ─────────────────────────────────────────────
    df = data[[time_col, target_col]].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    series = df[target_col].copy().reset_index(drop=True)

    # ── Train / test split ───────────────────────────────────────────
    split_idx    = int(split * len(df))
    train_series = series.iloc[:split_idx]
    test_series  = series.iloc[split_idx:]
    train_times  = df[time_col].iloc[:split_idx]
    test_times   = df[time_col].iloc[split_idx:]

    # ── Compute drift from training data ─────────────────────────────
    first_diffs = train_series.diff().dropna()
    drift       = float(first_diffs.mean()) if use_drift else 0.0
    resid_std   = float(first_diffs.std())   # 1-step residual std

    if seasonal and len(train_series) < season_s:
        log(f"   ⚠️  Training series too short for seasonal period {season_s}. "
            f"Falling back to pure Random Walk.")
        seasonal = False

    log(f"\n   Estimated drift (μ) : {drift:.4f}")
    log(f"   Residual std (σ)    : {resid_std:.4f}")

    # ── In-sample (train) predictions ────────────────────────────────
    # One-step-ahead in-sample fit
    train_pred = []
    for i in range(1, len(train_series)):
        if seasonal and i >= season_s:
            pred = float(train_series.iloc[i - season_s])
        else:
            pred = float(train_series.iloc[i - 1]) + drift
        train_pred.append(pred)

    # Drop the first observation (no prior value available)
    train_act  = train_series.iloc[1:].values
    train_pred = np.array(train_pred)
    train_t    = train_times.iloc[1:].values

    # ── Rolling one-step-ahead forecast on test ───────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")

    history      = train_series.copy().reset_index(drop=True)
    test_preds   = []
    test_lower95 = []
    test_upper95 = []
    test_lower80 = []
    test_upper80 = []

    for i in range(len(test_series)):
        h = 1   # always one-step-ahead in rolling forecast

        if seasonal and len(history) >= season_s:
            fc = float(history.iloc[-season_s])
        else:
            fc = float(history.iloc[-1]) + drift

        se = resid_std * np.sqrt(h)
        test_preds.append(fc)
        test_lower95.append(fc - 1.96 * se)
        test_upper95.append(fc + 1.96 * se)
        test_lower80.append(fc - 1.28 * se)
        test_upper80.append(fc + 1.28 * se)

        # Append actual observation to history
        new_obs = pd.Series([test_series.iloc[i]])
        history = pd.concat([history, new_obs], ignore_index=True)

    pred_te   = np.array(test_preds)
    actual_te = test_series.values
    l95_te    = np.array(test_lower95)
    u95_te    = np.array(test_upper95)
    l80_te    = np.array(test_lower80)
    u80_te    = np.array(test_upper80)

    # ── Metrics ──────────────────────────────────────────────────────
    def mape(a, p_):
        return float(np.mean(np.abs((a - p_) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(train_act, train_pred))
    mae_tr  = float(mean_absolute_error(train_act, train_pred))
    mape_tr = mape(train_act, train_pred)

    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te  = float(mean_absolute_error(actual_te, pred_te))
    mape_te = mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Model label ───────────────────────────────────────────────────
    if seasonal:
        model_label = f"Seasonal Naïve (s={season_s})"
    elif use_drift:
        model_label = f"Random Walk with Drift (μ={drift:.4f})"
    else:
        model_label = "Random Walk"

    # ── Parameter estimates & information criteria ───────────────────
    param_estimates_table = _param_estimates_df(
        drift, resid_std, len(train_series), use_drift, seasonal, season_s
    )
    info_criteria = _info_criteria(resid_std, len(train_series))

    # ── Future time labels ────────────────────────────────────────────
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

    # ── Save bundle for forecast mode ────────────────────────────────
    log("\n🔧 Saving model bundle for future forecasting...")
    os.makedirs("static", exist_ok=True)

    # Keep last `season_s` values for seasonal naïve forecasting
    tail_values = series.values.tolist()

    with open("static/rw_bundle.pkl", "wb") as f:
        pickle.dump({
            "last_value"  : float(series.iloc[-1]),
            "drift"       : drift,
            "resid_std"   : resid_std,
            "seasonal"    : seasonal,
            "season_s"    : season_s,
            "tail_values" : tail_values,
            "time_labels" : future_time_labels,
            "time_col"    : time_col,
            "target_col"  : target_col,
            "params"      : params,
        }, f)

    # ── Combined train + test plot ────────────────────────────────────
    train_x = list(range(len(train_act)))
    test_x  = list(range(len(train_act), len(train_act) + len(actual_te)))
    all_x   = train_x + test_x
    all_t   = list(train_t) + list(test_times.values)
    step    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train_x, train_act,  color="steelblue", label="Actual (Train)")
    ax.plot(train_x, train_pred, color="steelblue", label="Predicted (Train)",
            linestyle="--", marker="o", ms=4, alpha=0.8)

    ax.plot(test_x, actual_te, color="darkorange", label="Actual (Test)")
    ax.plot(test_x, pred_te,   color="darkorange", label="Predicted (Test)",
            linestyle="--", marker="o", ms=4, alpha=0.8)

    

    ax.axvline(x=len(train_act) - 1, color="black", linestyle=":",
               linewidth=2, label="Train/Test Split")

    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_t[::step], rotation=45, ha="right")
    ax.set_title(f"{model_label} — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/rw_train.png", dpi=120)
    plt.savefig("static/rw_test.png",  dpi=120)
    plt.close()

    # ── Output tables ────────────────────────────────────────────────
    train_table = pd.DataFrame({
        time_col   : train_t,
        "Actual"   : np.round(train_act,  3),
        "Predicted": np.round(train_pred, 3),
    })

    test_table = pd.DataFrame({
        time_col   : test_times.values,
        "Actual"   : np.round(actual_te, 3),
        "Predicted": np.round(pred_te,   3),
    })

    return {
        "model_key"   : "rw",
        "model_name"  : model_label,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Univariate (Random Walk)",

        "model_config": {
            "Variant"           : ("Seasonal Naïve" if seasonal
                                   else "RW with Drift" if use_drift
                                   else "Pure Random Walk"),
            "Drift (μ)"         : round(drift, 6) if use_drift else "N/A",
            "Seasonal"          : seasonal,
            "Seasonal Period (s)": season_s if seasonal else "N/A",
            "Residual Std (σ)"  : round(resid_std, 4),
        },

        "auto_tuned"  : False,   # no hyperparameter search

        # ── Parameter estimates & information criteria ───────────────
        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,

        "rmse_train"  : round(rmse_tr, 3),
        "mae_train"   : round(mae_tr,  3),
        "mape_train"  : round(mape_tr, 2),

        "rmse_test"   : round(rmse_te, 3),
        "mae_test"    : round(mae_te,  3),
        "mape_test"   : round(mape_te, 2),
        "residuals"   : (train_act - train_pred).tolist(),

        "train_table"   : train_table,
        "test_table"    : test_table,
        "forecast_table": None,
    }