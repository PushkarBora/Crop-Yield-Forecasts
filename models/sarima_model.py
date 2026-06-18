import numpy as np
import pandas as pd
import os
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FORECAST-ONLY  (loads saved bundle, no retraining)
# ============================================================
def _forecast_only_sarima(params, horizon):
    if not os.path.exists("static/sarima_bundle.pkl"):
        raise FileNotFoundError(
            "SARIMA model not trained yet. Please train the model first before forecasting."
        )

    with open("static/sarima_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    with open("static/sarima_model.pkl", "rb") as f:
        sarima_fit = pickle.load(f)

    time_labels  = bundle["time_labels"]
    exog_cols    = bundle.get("exog_cols", [])
    future_exog  = None

    # ── Multivariate: need future exogenous values ───────────────────
    if exog_cols:
        future_exog_data = params.get("future_exog", None)
        if future_exog_data is None:
            raise ValueError(
                f"This SARIMA model was trained with exogenous variables: {exog_cols}. "
                f"Please provide 'future_exog' in params as a dict of lists with {horizon} values each.\n"
                f"Example: params['future_exog'] = {{{', '.join([repr(c)+': [...]' for c in exog_cols])}}}"
            )
        future_exog = pd.DataFrame(future_exog_data)[exog_cols].values[:horizon]
        if len(future_exog) < horizon:
            raise ValueError(
                f"'future_exog' must have at least {horizon} rows, got {len(future_exog)}."
            )

    # ── Forecast with confidence intervals ──────────────────────────
    forecast_result = sarima_fit.get_forecast(steps=horizon, exog=future_exog)
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
    criteria = {}

    try:
        criteria["AIC"] = round(float(fit.aic), 4)
    except Exception:
        pass

    try:
        criteria["AICc"] = round(float(fit.aicc), 4)
    except Exception:
        try:
            k = len(fit.params)
            n = int(fit.nobs)
            aic = float(fit.aic)
            if n - k - 1 > 0:
                criteria["AICc"] = round(aic + 2 * k * (k + 1) / (n - k - 1), 4)
        except Exception:
            pass

    try:
        criteria["BIC"] = round(float(fit.bic), 4)
    except Exception:
        pass

    try:
        criteria["HQIC"] = round(float(fit.hqic), 4)
    except Exception:
        pass

    try:
        criteria["Log-Likelihood"] = round(float(fit.llf), 4)
    except Exception:
        pass

    return criteria


# ============================================================
# HELPER — safe SARIMAX fit (retries without seasonal if needed)
# ============================================================
def _safe_fit(endog, order, seasonal_order, exog=None, trend="n"):
    """
    Tries to fit SARIMAX. If it fails with the given seasonal_order,
    falls back to seasonal_order=(0,0,0,0) i.e. plain ARIMAX.
    Returns (fit_result, seasonal_order_used).
    """
    try:
        m = SARIMAX(
            endog,
            exog           = exog,
            order          = order,
            seasonal_order = seasonal_order,
            trend          = trend,
            enforce_stationarity  = False,
            enforce_invertibility = False,
        )
        return m.fit(disp=False), seasonal_order
    except Exception:
        # Fallback — drop seasonal component
        fallback = (0, 0, 0, 0)
        m = SARIMAX(
            endog,
            exog           = exog,
            order          = order,
            seasonal_order = fallback,
            trend          = trend,
            enforce_stationarity  = False,
            enforce_invertibility = False,
        )
        return m.fit(disp=False), fallback


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_sarima(
    data        : pd.DataFrame,
    params      : dict,
    horizon     : int,
    frequency   : str,
    mode        : str = "train",
    log_callback      = None,
):
    """
    SARIMA / SARIMAX model.
    - Univariate  : no exog_cols → pure SARIMA
    - Multivariate: exog_cols specified → SARIMAX with exogenous regressors
    Combined train+test plot with vertical split line (same style as LSTM/ARIMA).
    Forecast includes 80% and 95% prediction intervals.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ───────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_sarima(params=params, horizon=horizon)

    # ── Validate ────────────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ──────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols  = params.get("exog_cols",  [])          # [] → univariate

    is_multivariate = len(exog_cols) > 0
    data_type_label = "Multivariate (SARIMAX)" if is_multivariate else "Univariate (SARIMA)"

    log(f"📊 SARIMA — {data_type_label}")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")
    if is_multivariate:
        log(f"   Exog variables : {exog_cols}")

    # ── Hyperparameters ──────────────────────────────────────────────
    split          = params.get("split", 0.85)
    use_auto_arima = params.get("auto_sarima", params.get("auto_arima", True))

    # ── Safe param getter: 0 is a valid value, must NOT be treated as falsy ──
    def _sp(d, key, default):
        v = d.get(key)
        return v if v is not None else default

    # Non-seasonal orders — read sarima_* first, fall back to arima_* then hard default
    p = _sp(params, "sarima_p", _sp(params, "arima_p", 1))
    d = _sp(params, "sarima_d", _sp(params, "arima_d", 1))
    q = _sp(params, "sarima_q", _sp(params, "arima_q", 0))   # 0 is valid — never use `or`!

    # Seasonal orders
    P = _sp(params, "sarima_P", 1)
    D = _sp(params, "sarima_D", 1)
    Q = _sp(params, "sarima_Q", 0)   # 0 is valid — never use `or`!
    s = _sp(params, "sarima_s", 12)

    log("=" * 60)
    log(f"   auto_arima    : {use_auto_arima}")
    log(f"   Manual order  : SARIMA({p},{d},{q})({P},{D},{Q})[{s}]")
    if is_multivariate:
        log(f"   Exogenous cols: {exog_cols}")
    log("=" * 60)

    # ── Data preparation ─────────────────────────────────────────────
    required_cols = [time_col, target_col] + exog_cols
    df = data[required_cols].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Clean target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    # Clean exogenous
    for col in exog_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="linear").ffill().bfill()

    series = df[target_col].copy()

    # ── Train / test split ───────────────────────────────────────────
    split_idx    = int(split * len(df))
    train_series = series.iloc[:split_idx]
    test_series  = series.iloc[split_idx:]
    train_times  = df[time_col].iloc[:split_idx]
    test_times   = df[time_col].iloc[split_idx:]

    # Exogenous arrays
    train_exog = df[exog_cols].iloc[:split_idx].values if is_multivariate else None
    test_exog  = df[exog_cols].iloc[split_idx:].values if is_multivariate else None

    # ── AUTO-ARIMA / AUTO-SARIMA grid search ─────────────────────────
    if use_auto_arima:
        log("\n🔍 Running Auto-SARIMA grid search (this may take a while)...")
        best_aic   = np.inf
        best_order = (p, d, q)
        best_seas  = (P, D, Q, s)

        # Keep grid small but meaningful
        p_range = range(0, 3)
        d_range = range(0, 2)
        q_range = range(0, 3)
        P_range = range(0, 2)
        D_range = range(0, 2)
        Q_range = range(0, 2)

        for p_t in p_range:
            for d_t in d_range:
                for q_t in q_range:
                    for P_t in P_range:
                        for D_t in D_range:
                            for Q_t in Q_range:
                                try:
                                    fit_tmp, _ = _safe_fit(
                                        endog          = train_series,
                                        order          = (p_t, d_t, q_t),
                                        seasonal_order = (P_t, D_t, Q_t, s),
                                        exog           = train_exog,
                                    )
                                    aic_val = fit_tmp.aic
                                    log(
                                        f"   Testing SARIMA({p_t},{d_t},{q_t})"
                                        f"({P_t},{D_t},{Q_t})[{s}] → AIC={aic_val:.2f}"
                                    )

                                    # ✅ Reject degenerate fits
                                    if np.isnan(aic_val) or np.isinf(aic_val):
                                        continue
                                    if aic_val < 50:  # Suspiciously low AIC
                                        continue
                                    non_sigma_params = [
                                        v for name, v in zip(fit_tmp.param_names, fit_tmp.params)
                                        if "sigma" not in name.lower()
                                    ]
                                    if non_sigma_params:
                                        max_coef = np.max(np.abs(non_sigma_params))
                                        if max_coef > 1e6:
                                            continue

                                    if aic_val < best_aic:
                                        best_aic   = aic_val
                                        best_order = (p_t, d_t, q_t)
                                        best_seas  = (P_t, D_t, Q_t, s)
                                        log(
                                            f"   ✨ New best: SARIMA{best_order}"
                                            f"{best_seas}  AIC={best_aic:.2f}"
                                        )
                                except Exception:
                                    continue

        p, d, q       = best_order
        P, D, Q, s    = best_seas
        params["arima_p"] = p
        params["arima_d"] = d
        params["arima_q"] = q
        params["sarima_P"] = P
        params["sarima_D"] = D
        params["sarima_Q"] = Q
        params["sarima_s"] = s
        log(f"\n✅ Best order: SARIMA({p},{d},{q})({P},{D},{Q})[{s}]")

    # ── Fit on train ─────────────────────────────────────────────────
    arima_fit, (P, D, Q, s) = _safe_fit(
        endog          = train_series,
        order          = (p, d, q),
        seasonal_order = (P, D, Q, s),
        exog           = train_exog,
    )

    model_label = f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"
    if is_multivariate:
        model_label = f"SARIMAX({p},{d},{q})({P},{D},{Q})[{s}]"

    # ── In-sample (train) predictions ────────────────────────────────
    fitted_vals = arima_fit.fittedvalues

    # Rows lost to differencing (non-seasonal d + seasonal D*s)
    valid_start = d + D * s if (d + D * s) > 0 else 0
    valid_start = min(valid_start, len(train_series) - 1)

    train_pred = fitted_vals.iloc[valid_start:].values
    train_act  = train_series.iloc[valid_start:].values
    train_t    = train_times.iloc[valid_start:].values

    # ── Rolling one-step-ahead forecast on test ───────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")
    test_preds   = []
    test_lower95 = []
    test_upper95 = []
    test_lower80 = []
    test_upper80 = []
    history      = train_series.copy()
    history_exog = train_exog.tolist() if is_multivariate else None

    for i in range(len(test_series)):
        exog_fit   = np.array(history_exog) if is_multivariate else None
        exog_fc    = test_exog[i:i+1]       if is_multivariate else None

        try:
            f_tmp, _ = _safe_fit(
                endog          = history,
                order          = (p, d, q),
                seasonal_order = (P, D, Q, s),
                exog           = exog_fit,
            )
            fc   = f_tmp.get_forecast(steps=1, exog=exog_fc)
            ci95 = fc.conf_int(alpha=0.05)
            ci80 = fc.conf_int(alpha=0.20)

            test_preds.append(fc.predicted_mean.iloc[0])
            test_lower95.append(ci95.iloc[0, 0])
            test_upper95.append(ci95.iloc[0, 1])
            test_lower80.append(ci80.iloc[0, 0])
            test_upper80.append(ci80.iloc[0, 1])
        except Exception as e:
            log(f"   ⚠️  Step {i+1} forecast failed ({e}), using last prediction.")
            last_val = test_preds[-1] if test_preds else float(history.iloc[-1])
            test_preds.append(last_val)
            test_lower95.append(last_val)
            test_upper95.append(last_val)
            test_lower80.append(last_val)
            test_upper80.append(last_val)

        history = pd.concat([history, test_series.iloc[i:i+1]])
        if is_multivariate:
            history_exog.append(test_exog[i].tolist())

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

    # ✅ Sanity check — if metrics are astronomically large, 
    # fall back to manual order instead of crashing
    if rmse_te > 1e10 or np.isnan(rmse_te) or np.isinf(rmse_te):
        log("\n⚠️  Auto-SARIMA selected a degenerate model. Falling back to manual order...")
        p, d, q = (_sp(params, "sarima_p", 1), _sp(params, "sarima_d", 1), _sp(params, "sarima_q", 0))
        P, D, Q = (_sp(params, "sarima_P", 1), _sp(params, "sarima_D", 1), _sp(params, "sarima_Q", 0))

        arima_fit, (P, D, Q, s) = _safe_fit(
            endog=train_series, order=(p, d, q),
            seasonal_order=(P, D, Q, s), exog=train_exog,
        )
        fitted_vals  = arima_fit.fittedvalues
        valid_start  = min(d + D * s, len(train_series) - 1)
        train_pred   = fitted_vals.iloc[valid_start:].values
        train_act    = train_series.iloc[valid_start:].values
        train_t      = train_times.iloc[valid_start:].values
        model_label  = f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"

        # Recompute test predictions with fallback order
        test_preds, test_lower95, test_upper95, test_lower80, test_upper80 = [], [], [], [], []
        history = train_series.copy()
        history_exog = train_exog.tolist() if is_multivariate else None
        for i in range(len(test_series)):
            exog_fit = np.array(history_exog) if is_multivariate else None
            exog_fc  = test_exog[i:i+1] if is_multivariate else None
            try:
                f_tmp, _ = _safe_fit(history, (p,d,q), (P,D,Q,s), exog_fit)
                fc   = f_tmp.get_forecast(steps=1, exog=exog_fc)
                ci95 = fc.conf_int(alpha=0.05)
                ci80 = fc.conf_int(alpha=0.20)
                test_preds.append(fc.predicted_mean.iloc[0])
                test_lower95.append(ci95.iloc[0, 0])
                test_upper95.append(ci95.iloc[0, 1])
                test_lower80.append(ci80.iloc[0, 0])
                test_upper80.append(ci80.iloc[0, 1])
            except Exception:
                last_val = test_preds[-1] if test_preds else float(history.iloc[-1])
                test_preds.extend([last_val]*1)
                test_lower95.append(last_val); test_upper95.append(last_val)
                test_lower80.append(last_val); test_upper80.append(last_val)
            history = pd.concat([history, test_series.iloc[i:i+1]])
            if is_multivariate:
                history_exog.append(test_exog[i].tolist())

        pred_te  = np.array(test_preds)
        actual_te = test_series.values
        rmse_tr  = float(root_mean_squared_error(train_act, train_pred))
        mae_tr   = float(mean_absolute_error(train_act, train_pred))
        mape_tr  = mape(train_act, train_pred)
        rmse_te  = float(root_mean_squared_error(actual_te, pred_te))
        mae_te   = float(mean_absolute_error(actual_te, pred_te))
        mape_te  = mape(actual_te, pred_te)
        param_estimates_table = _param_estimates_df(arima_fit)
        info_criteria         = _info_criteria(arima_fit)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ✅ Only compute for normal path — fallback already sets these
    if 'param_estimates_table' not in dir() or param_estimates_table is None:
        param_estimates_table = _param_estimates_df(arima_fit)
        info_criteria         = _info_criteria(arima_fit)

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

    # ── Refit on full data for future forecasting ────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    full_exog = df[exog_cols].values if is_multivariate else None
    arima_fit_full, _ = _safe_fit(
        endog          = series,
        order          = (p, d, q),
        seasonal_order = (P, D, Q, s),
        exog           = full_exog,
    )

    # ── Save bundle ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)

    with open("static/sarima_model.pkl", "wb") as f:
        pickle.dump(arima_fit_full, f)

    with open("static/sarima_bundle.pkl", "wb") as f:
        pickle.dump({
            "order"         : (p, d, q),
            "seasonal_order": (P, D, Q, s),
            "last_time"     : last_t,
            "time_labels"   : future_time_labels,
            "time_col"      : time_col,
            "target_col"    : target_col,
            "exog_cols"     : exog_cols,
            "params"        : params,
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
    plt.savefig("static/sarima_train.png", dpi=120)
    plt.savefig("static/sarima_test.png",  dpi=120)
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
        "model_key"   : "sarima",
        "model_name"  : model_label,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : data_type_label,
        "model_config": {
            "p (AR Order)"         : p,
            "d (Differencing)"     : d,
            "q (MA Order)"         : q,
            "P (Seasonal AR)"      : P,
            "D (Seasonal Diff)"    : D,
            "Q (Seasonal MA)"      : Q,
            "s (Seasonal Period)"  : s,
            "Exogenous Variables"  : ", ".join(exog_cols) if exog_cols else "None",
        },
        "auto_tuned"  : use_auto_arima,
        "order"       : model_label,
        "auto_arima"  : use_auto_arima,

        # ── Parameter estimates & information criteria ───────────────
        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,
        # ─────────────────────────────────────────────────────────────

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