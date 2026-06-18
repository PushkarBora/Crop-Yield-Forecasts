import numpy as np
import pandas as pd
import os
import pickle
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error,mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FORECAST-ONLY  (loads saved bundle, no retraining)
# ============================================================
def _forecast_only_arima(params, horizon):
    if not os.path.exists("static/arima_bundle.pkl"):
        raise FileNotFoundError(
            "ARIMA model not trained yet. Please train the model first before forecasting."
        )

    with open("static/arima_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    with open("static/arima_model.pkl", "rb") as f:
        arima_fit = pickle.load(f)

    time_labels = bundle["time_labels"]

    # Forecast with confidence intervals (raw values, no log)
    forecast_result = arima_fit.get_forecast(steps=horizon)
    forecast_mean   = forecast_result.predicted_mean
    conf_int_95     = forecast_result.conf_int(alpha=0.05)
    conf_int_80     = forecast_result.conf_int(alpha=0.20)

    mean_vals = forecast_mean.values
    lower_95  = conf_int_95.iloc[:, 0].values
    upper_95  = conf_int_95.iloc[:, 1].values
    lower_80  = conf_int_80.iloc[:, 0].values
    upper_80  = conf_int_80.iloc[:, 1].values

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
# HELPER — build parameter-estimates DataFrame from fitted model
# ============================================================
def _param_estimates_df(fit):
    """
    Returns a tidy DataFrame matching the statsmodels summary block:
      Parameter | Coef | Std Err | z | P>|z| | [0.025 | 0.975]
    """
    ci = fit.conf_int()
    df = pd.DataFrame({
        "Parameter": fit.param_names,
        "Coef"     : np.round(fit.params,  4),
        "Std Err"  : np.round(fit.bse,     4),
        "z"        : np.round(fit.tvalues, 4),
        "P>|z|"    : np.round(fit.pvalues, 4),
        "[0.025"   : np.round(ci.iloc[:, 0], 4),
        "0.975]"   : np.round(ci.iloc[:, 1], 4),
    })
    return df


# ============================================================
# HELPER — collect information criteria from fitted model
# ============================================================
def _info_criteria(fit):
    """
    Returns a dict of available information criteria.
    AICC is computed manually if not directly exposed.
    """
    criteria = {}

    # AIC
    try:
        criteria["AIC"] = round(float(fit.aic), 4)
    except Exception:
        pass

    # AICC  (statsmodels exposes .aicc on some result classes)
    try:
        criteria["AICc"] = round(float(fit.aicc), 4)
    except Exception:
        # Manual AICC = AIC + 2k(k+1)/(n-k-1)
        try:
            k = len(fit.params)
            n = int(fit.nobs)
            aic = float(fit.aic)
            if n - k - 1 > 0:
                criteria["AICc"] = round(aic + 2 * k * (k + 1) / (n - k - 1), 4)
        except Exception:
            pass

    # BIC
    try:
        criteria["BIC"] = round(float(fit.bic), 4)
    except Exception:
        pass

    # HQIC
    try:
        criteria["HQIC"] = round(float(fit.hqic), 4)
    except Exception:
        pass

    # Log-likelihood
    try:
        criteria["Log-Likelihood"] = round(float(fit.llf), 4)
    except Exception:
        pass

    return criteria


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_arima(
    data        : pd.DataFrame,
    params      : dict,
    horizon     : int,
    frequency   : str,
    mode        : str = "train",
    log_callback      = None,
):
    """
    ARIMA model — univariate, dynamic column names, NO log transform.
    Combined train+test plot with vertical split line (same style as LSTM).
    Forecast includes 80% and 95% prediction intervals.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ───────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_arima(params=params, horizon=horizon)

    # ── Validate ────────────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings (dynamic, like LSTM) ─────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")

    log(f"📊 ARIMA — Univariate")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")

    # ── Hyperparameters ─────────────────────────────────────────────
    split          = params.get("split", 0.85)
    use_auto_arima = params.get("auto_arima", True)

    p = params.get("arima_p", 1)
    d = params.get("arima_d", 1)
    q = params.get("arima_q", 0)

    log("=" * 60)
    log(f"   auto_arima : {use_auto_arima}")
    log(f"   Manual p={p}, d={d}, q={q}")
    log("=" * 60)

    # ── Data preparation — raw values, NO log transform ──────────────
    df = data[[time_col, target_col]].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    series = df[target_col].copy()   # raw values

    # ── Train / test split ───────────────────────────────────────────
    split_idx    = int(split * len(df))
    train_series = series.iloc[:split_idx]
    test_series  = series.iloc[split_idx:]
    train_times  = df[time_col].iloc[:split_idx]
    test_times   = df[time_col].iloc[split_idx:]

    # ── AUTO-ARIMA grid search ───────────────────────────────────────
    if use_auto_arima:
        log("\n🔍 Running Auto-ARIMA grid search...")
        best_aic   = np.inf
        best_order = None

        for p_try in range(0, 4):
            for d_try in range(0, 3):
                for q_try in range(0, 4):
                    try:
                        m = ARIMA(train_series, order=(p_try, d_try, q_try), trend="n")
                        r = m.fit()
                        log(f"   Testing ARIMA({p_try},{d_try},{q_try}) → AIC={r.aic:.2f}")
                        if r.aic < best_aic:
                            best_aic   = r.aic
                            best_order = (p_try, d_try, q_try)
                            log(f"   ✨ New best: ARIMA{best_order}  AIC={best_aic:.2f}")
                    except Exception:
                        continue

        if best_order is not None:
            p, d, q = best_order
            log(f"\n✅ Best order: ARIMA({p},{d},{q})")
            params["arima_p"] = p
            params["arima_d"] = d
            params["arima_q"] = q

    # ── Fit on train ─────────────────────────────────────────────────
    model_train = ARIMA(train_series, order=(p, d, q), trend="n")
    arima_fit   = model_train.fit()

    # ── In-sample (train) predictions ────────────────────────────────
    fitted_vals = arima_fit.fittedvalues
    valid_start = d if d > 0 else 0      # rows lost to differencing
    train_pred  = fitted_vals.iloc[valid_start:].values
    train_act   = train_series.iloc[valid_start:].values
    train_t     = train_times.iloc[valid_start:].values

    # ── Rolling one-step-ahead forecast on test ───────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")
    test_preds   = []
    test_lower95 = []
    test_upper95 = []
    test_lower80 = []
    test_upper80 = []
    history      = train_series.copy()

    for i in range(len(test_series)):
        m_tmp = ARIMA(history, order=(p, d, q), trend="n")
        f_tmp = m_tmp.fit()

        fc   = f_tmp.get_forecast(steps=1)
        ci95 = fc.conf_int(alpha=0.05)
        ci80 = fc.conf_int(alpha=0.20)

        test_preds.append(fc.predicted_mean.iloc[0])
        test_lower95.append(ci95.iloc[0, 0])
        test_upper95.append(ci95.iloc[0, 1])
        test_lower80.append(ci80.iloc[0, 0])
        test_upper80.append(ci80.iloc[0, 1])

        history = pd.concat([history, test_series.iloc[i:i+1]])

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
    mae_tr   = float(mean_absolute_error(train_act, train_pred))
    mape_tr = mape(train_act, train_pred)

    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te   = float(mean_absolute_error(actual_te, pred_te))
    mape_te = mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Parameter estimates & information criteria ───────────────────
    param_estimates_table = _param_estimates_df(arima_fit)
    info_criteria         = _info_criteria(arima_fit)

    # ── Future time labels (same cycle logic as LSTM) ─────────────────
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

    # ── Refit on full data for future forecasting ────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    model_full     = ARIMA(series, order=(p, d, q), trend="n")
    arima_fit_full = model_full.fit()

    # ── Save bundle ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)

    with open("static/arima_model.pkl", "wb") as f:
        pickle.dump(arima_fit_full, f)

    with open("static/arima_bundle.pkl", "wb") as f:
        pickle.dump({
            "order"      : (p, d, q),
            "last_time"  : last_t,
            "time_labels": future_time_labels,
            "time_col"   : time_col,
            "target_col" : target_col,
            "params"     : params,
        }, f)

    # ── Combined train + test plot — same style as LSTM ───────────────
    train_x = list(range(len(train_act)))
    test_x  = list(range(len(train_act), len(train_act) + len(actual_te)))
    all_x   = train_x + test_x
    all_t   = list(train_t) + list(test_times.values)
    step    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1

    fig, ax = plt.subplots(figsize=(14, 5))

    # Train lines
    ax.plot(train_x, train_act,  color="steelblue", label="Actual (Train)")
    ax.plot(train_x, train_pred, color="steelblue", label="Predicted (Train)",
            linestyle="--", marker="o", ms=4, alpha=0.8)

    # Test lines
    ax.plot(test_x, actual_te, color="darkorange", label="Actual (Test)")
    ax.plot(test_x, pred_te,   color="darkorange", label="Predicted (Test)",
            linestyle="--", marker="o", ms=4, alpha=0.8)

    # Vertical train/test split line — same as LSTM
    ax.axvline(x=len(train_act) - 1, color="black", linestyle=":",
               linewidth=2, label="Train/Test Split")

    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_t[::step], rotation=45, ha="right")
    ax.set_title(f"ARIMA({p},{d},{q}) — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/arima_train.png", dpi=120)
    plt.savefig("static/arima_test.png",  dpi=120)
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
        "model_key"   : "arima",
        "model_name"  : f"ARIMA({p},{d},{q})",
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Univariate",
        "model_config": {
            "p (AR Order)"    : p,
            "d (Differencing)": d,
            "q (MA Order)"    : q,
        },
        "auto_tuned"    : use_auto_arima,
        "order"         : f"ARIMA({p},{d},{q})",
        "auto_arima"    : use_auto_arima,

        # ── NEW: parameter estimates & information criteria ──────────
        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,
        # ─────────────────────────────────────────────────────────────

        "rmse_train"    : round(rmse_tr, 3),
        "mae_train"      : round(mae_tr,   3),
        "mape_train"    : round(mape_tr,  2),

        "rmse_test"     : round(rmse_te, 3),
        "mae_test"       : round(mae_te,   3),
        "mape_test"     : round(mape_te,  2),
        "residuals"   : (train_act - train_pred).tolist(),   # ← ADD THIS

        "train_table"   : train_table,
        "test_table"    : test_table,
        "forecast_table": None,
    }