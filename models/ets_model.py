import numpy as np
import pandas as pd
import os
import pickle
import warnings

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ETS CONFIGURATIONS FOR AUTO MODE
# ─────────────────────────────────────────────────────────────────────────────
_AUTO_CONFIGS = [
    # (error, trend,  damped, seasonal,  label)
    ("add",  None,   False,  None,      "ETS(A,N,N) — Simple Exponential Smoothing"),
    ("add",  "add",  False,  None,      "ETS(A,A,N) — Holt's Linear"),
    ("add",  "add",  True,   None,      "ETS(A,Ad,N) — Damped Holt"),
    ("add",  "add",  False,  "add",     "ETS(A,A,A) — Holt-Winters Additive"),
    ("add",  "add",  True,   "add",     "ETS(A,Ad,A) — Damped HW Additive"),
    ("mul",  "add",  False,  "mul",     "ETS(M,A,M) — Holt-Winters Multiplicative"),
    ("mul",  "add",  True,   "mul",     "ETS(M,Ad,M) — Damped HW Multiplicative"),
    ("add",  "add",  False,  "mul",     "ETS(A,A,M) — HW with Multiplicative Seasonal"),
    ("add",  None,   False,  "add",     "ETS(A,N,A) — Additive Seasonal, No Trend"),
    ("mul",  None,   False,  "mul",     "ETS(M,N,M) — Multiplicative Seasonal, No Trend"),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — try fitting one ETS configuration
# ─────────────────────────────────────────────────────────────────────────────
def _try_ets(y, error, trend, damped, seasonal, seasonal_periods, log):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    try:
        sp = seasonal_periods if seasonal is not None else None
        model = ETSModel(
            y,
            error=error,
            trend=trend,
            damped_trend=damped,
            seasonal=seasonal,
            seasonal_periods=sp,
        )
        fit = model.fit(disp=False, maxiter=500)
        return fit
    except Exception as e:
        log(f"      ↳ skipped: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — OLS regression
# ─────────────────────────────────────────────────────────────────────────────
def _fit_ols(X, y):
    """Fit OLS: y = X @ beta. Returns (beta, fitted_values)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    return beta, fitted


def _make_X(df_subset, exog_cols):
    """Build design matrix with intercept."""
    X = df_subset[exog_cols].values.astype(float)
    return np.column_stack([np.ones(len(X)), X])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — parameter estimates table (2-column like TBATS/RW)
# ─────────────────────────────────────────────────────────────────────────────
def _param_table(fit, model_label, seasonal_periods, ols_beta=None, exog_cols=None):
    rows = [("Model Type", model_label)]

    if ols_beta is not None and exog_cols is not None:
        rows.append(("── OLS Regression Coefficients ──", ""))
        rows.append(("Intercept (β₀)", round(float(ols_beta[0]), 6)))
        for i, col in enumerate(exog_cols):
            rows.append((f"β({col})", round(float(ols_beta[i + 1]), 6)))
        rows.append(("── ETS Residual Model ──", ""))

    try:
        # Build name→value dict safely (handles numpy array or pandas Series)
        raw = fit.params
        if hasattr(raw, "index"):
            p = dict(zip(raw.index, raw.values))
        elif hasattr(fit, "param_names"):
            p = dict(zip(fit.param_names, raw))
        else:
            p = {}

        display = {
            "smoothing_level":    "α (level smoothing)",
            "smoothing_trend":    "β (trend smoothing)",
            "smoothing_seasonal": "γ (seasonal smoothing)",
            "damping_trend":      "φ (damping)",
        }
        aliases = {
            "smoothing_level":    ["smoothing_level",    "alpha"],
            "smoothing_trend":    ["smoothing_trend",     "beta"],
            "smoothing_seasonal": ["smoothing_seasonal",  "gamma"],
            "damping_trend":      ["damping_trend",       "phi"],
        }
        for canonical, label in display.items():
            for key in aliases[canonical]:
                if key in p:
                    try:
                        val = float(p[key])
                        if not np.isnan(val):
                            rows.append((label, round(val, 6)))
                            break
                    except Exception:
                        pass
        if seasonal_periods:
            rows.append(("Seasonal Period (s)", int(seasonal_periods)))
        rows.append(("AIC",  round(float(fit.aic),  4)))
        rows.append(("BIC",  round(float(fit.bic),  4)))
        rows.append(("AICc", round(float(fit.aicc), 4)))
    except Exception:
        pass

    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — information criteria dict
# ─────────────────────────────────────────────────────────────────────────────
def _info_criteria(fit):
    crit = {}
    try: crit["AIC"]            = round(float(fit.aic),  4)
    except Exception: pass
    try: crit["AICc"]           = round(float(fit.aicc), 4)
    except Exception: pass
    try: crit["BIC"]            = round(float(fit.bic),  4)
    except Exception: pass
    try: crit["Log-Likelihood"] = round(float(fit.llf),  4)
    except Exception: pass
    return crit


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST-ONLY (loads saved bundle, no retraining)
# ─────────────────────────────────────────────────────────────────────────────
def _forecast_only_ets(params, horizon):
    bundle_path = "static/ets_bundle.pkl"
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            "ETS model not trained yet. Please train the model first before forecasting."
        )

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    fit             = bundle["fit"]
    time_labels     = bundle["time_labels"]
    is_multivariate = bundle.get("is_multivariate", False)
    ols_beta        = bundle.get("ols_beta")
    exog_cols       = bundle.get("exog_cols", [])

    fc_ets    = fit.forecast(horizon)
    resid_std = float(np.std(fit.resid, ddof=1)) if hasattr(fit, "resid") else 0.0

    # OLS component for multivariate
    if is_multivariate and ols_beta is not None and exog_cols:
        future_exog = params.get("future_exog", {})
        if not future_exog:
            raise ValueError(
                f"This ETS model was trained with exogenous variables: {exog_cols}. "
                f"Please provide 'future_exog' in params as a dict of lists with "
                f"{horizon} values each."
            )
        X_future = np.column_stack([
            np.ones(horizon),
            *[np.array(future_exog[col], dtype=float) for col in exog_cols]
        ])
        ols_fc = X_future @ ols_beta
    else:
        ols_fc = np.zeros(horizon)

    rows = []
    for h in range(1, horizon + 1):
        ets_val = float(fc_ets.iloc[h - 1]) if hasattr(fc_ets, "iloc") else float(fc_ets[h - 1])
        f_val   = ets_val + float(ols_fc[h - 1])
        se      = resid_std * np.sqrt(h)
        rows.append({
            "Period":         time_labels[h - 1] if h - 1 < len(time_labels) else h,
            "Forecast":       round(f_val, 3),
            "Lower 80%":      round(f_val - 1.28 * se, 3),
            "Upper 80%":      round(f_val + 1.28 * se, 3),
            "Lower 95%":      round(f_val - 1.96 * se, 3),
            "Upper 95%":      round(f_val + 1.96 * se, 3),
            "Interval (95%)": f"[{f_val - 1.96*se:.1f}, {f_val + 1.96*se:.1f}]",
        })

    return {"forecast_table": pd.DataFrame(rows)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_ets(
    data,
    params,
    horizon,
    frequency,
    mode         = "train",
    log_callback = None,
):
    """
    ETS (Error-Trend-Seasonality) model — Holt-Winters family.

    Univariate mode:
        Standard ETS fitted directly on the target series.

    Multivariate mode (Regression with ETS Errors):
        Step 1 — OLS regression: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
        Step 2 — ETS fitted on regression residuals ε
        Forecast = OLS forecast (using future exog) + ETS residual forecast

        This is the standard "Regression with ETS Errors" approach,
        analogous to how SARIMAX extends SARIMA.

    Auto mode tries 10 common ETS configurations and selects best by AIC.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ──────────────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_ets(params=params, horizon=horizon)

    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ────────────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols  = params.get("exog_cols",  [])
    split      = params.get("split", 0.85)

    is_multivariate = len(exog_cols) > 0

    # ── ETS hyperparameters ────────────────────────────────────────────────
    auto_ets             = params.get("auto_ets", True)
    ets_error            = params.get("ets_error",    "add")
    ets_trend_raw        = params.get("ets_trend",    "add")
    ets_seasonal_raw     = params.get("ets_seasonal", "add")
    ets_seasonal_periods = int(params.get("ets_seasonal_periods", 12))

    if ets_trend_raw in (None, "none", ""):
        ets_trend, ets_damped = None, False
    elif "damped" in str(ets_trend_raw):
        ets_trend  = ets_trend_raw.replace("_damped", "")
        ets_damped = True
    else:
        ets_trend, ets_damped = ets_trend_raw, False

    ets_seasonal = None if ets_seasonal_raw in (None, "none", "") else ets_seasonal_raw

    # ── Data preparation ───────────────────────────────────────────────────
    if is_multivariate:
        log(f"📊 ETS — Multivariate (Regression + ETS Errors)")
        log(f"   Time column    : {time_col}")
        log(f"   Study variable : {target_col}")
        log(f"   Exog variables : {exog_cols}")
        needed_cols = [time_col, target_col] + exog_cols
    else:
        log(f"📊 ETS — Univariate")
        log(f"   Time column    : {time_col}")
        log(f"   Study variable : {target_col}")
        needed_cols = [time_col, target_col]

    log("=" * 60)

    df = data[needed_cols].copy()
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df[target_col] = (pd.to_numeric(df[target_col], errors="coerce")
                      .interpolate(method="linear").ffill().bfill())
    for col in exog_cols:
        df[col] = (pd.to_numeric(df[col], errors="coerce")
                   .interpolate(method="linear").ffill().bfill())

    series = df[target_col].reset_index(drop=True)

    # ── Train / test split ─────────────────────────────────────────────────
    split_idx    = int(split * len(df))
    train_df     = df.iloc[:split_idx].copy()
    test_df      = df.iloc[split_idx:].copy()
    train_series = series.iloc[:split_idx]
    test_series  = series.iloc[split_idx:]
    train_times  = df[time_col].iloc[:split_idx]
    test_times   = df[time_col].iloc[split_idx:]
    n_train      = len(train_series)

    # ── Step 1 (Multivariate): OLS on exogenous variables ────────────────
    ols_beta         = None
    train_ols_fitted = np.zeros(n_train)
    test_ols_pred    = np.zeros(len(test_series))

    if is_multivariate:
        log(f"\n   Step 1: OLS regression on exogenous variables...")
        X_train = _make_X(train_df, exog_cols)
        y_train_arr = train_series.values.astype(float)
        ols_beta, train_ols_fitted = _fit_ols(X_train, y_train_arr)
        X_test = _make_X(test_df, exog_cols)
        test_ols_pred = X_test @ ols_beta
        log(f"   Intercept: {ols_beta[0]:.4f}")
        for i, col in enumerate(exog_cols):
            log(f"   β({col}): {ols_beta[i+1]:.4f}")
        train_for_ets = train_series.values.astype(float) - train_ols_fitted
        log(f"   Step 2: Fitting ETS on OLS residuals...")
    else:
        train_for_ets = train_series.values.astype(float)

    # ── Step 2: Fit ETS ────────────────────────────────────────────────────
    if ets_seasonal and n_train < 2 * ets_seasonal_periods:
        log(f"   ⚠️  Not enough training data for seasonal period {ets_seasonal_periods}. Forcing seasonal=None.")
        ets_seasonal = None

    best_fit = None
    best_aic = np.inf
    best_label = ""
    best_sp = None

    if auto_ets:
        series_name = "residuals" if is_multivariate else "series"
        log(f"   auto_ets: True | Seasonal period: {ets_seasonal_periods}")
        log(f"   Trying {len(_AUTO_CONFIGS)} ETS configs on {series_name}...")
        log("=" * 60)
        for error, trend, damped, seasonal, label in _AUTO_CONFIGS:
            if seasonal and n_train < 2 * ets_seasonal_periods:
                continue
            log(f"   Testing: {label}")
            sp  = ets_seasonal_periods if seasonal else None
            fit = _try_ets(train_for_ets, error, trend, damped, seasonal, sp, log)
            if fit is not None:
                try:
                    aic = float(fit.aic)
                    log(f"      AIC = {aic:.4f}")
                    if aic < best_aic:
                        best_aic, best_fit, best_label, best_sp = aic, fit, label, sp
                except Exception:
                    pass
        if best_fit is None:
            raise RuntimeError(
                "All ETS configurations failed. Try manual mode or check your data."
            )
        log(f"\n✅ Best ETS: {best_label}  (AIC={best_aic:.4f})")
    else:
        log(f"   auto_ets: False | Error:{ets_error} Trend:{ets_trend} "
            f"Damped:{ets_damped} Seasonal:{ets_seasonal}")
        log("=" * 60)
        sp       = ets_seasonal_periods if ets_seasonal else None
        best_sp  = sp
        best_fit = _try_ets(train_for_ets, ets_error, ets_trend, ets_damped, ets_seasonal, sp, log)
        if best_fit is None:
            raise RuntimeError(
                f"ETS({ets_error},{ets_trend},{ets_seasonal}) failed. "
                "Try a different configuration or use Auto mode."
            )
        e_s = ets_error[0].upper() if ets_error else "A"
        t_s = (ets_trend[0].upper() + ("d" if ets_damped else "")) if ets_trend else "N"
        s_s = ets_seasonal[0].upper() if ets_seasonal else "N"
        best_label = f"ETS({e_s},{t_s},{s_s})" + (f"[{sp}]" if sp else "")
        log(f"\n✅ Fitted: {best_label}")

    # ── Extract ETS spec for rolling forecast ──────────────────────────────
    try:
        bf        = best_fit.model
        _error    = bf.error          # already "add" or "mul"
        _trend    = bf.trend          # already "add", "mul", or None
        _damped   = bf.damped_trend   # bool
        _seasonal = bf.seasonal       # already "add", "mul", or None
    except Exception:
        _error, _trend, _damped, _seasonal = ets_error, ets_trend, ets_damped, ets_seasonal

    # ── In-sample train fitted values ──────────────────────────────────────
    try:
        ets_fv_train = np.array(best_fit.fittedvalues, dtype=float)
        # Replace any NaN (init period) with forward fill
        nan_mask = np.isnan(ets_fv_train)
        if nan_mask.any() and not nan_mask.all():
            ets_fv_train[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                np.where(~nan_mask)[0],
                ets_fv_train[~nan_mask]
            )
        elif nan_mask.all():
            ets_fv_train = np.full(n_train, float(np.mean(train_for_ets)))
    except Exception:
        ets_fv_train = np.full(n_train, float(np.mean(train_for_ets)))

    min_len      = min(n_train, len(ets_fv_train))
    train_act    = train_series.iloc[-min_len:].values
    ets_fv       = ets_fv_train[-min_len:]
    ols_fv_trim  = train_ols_fitted[-min_len:]
    train_pred_v = ols_fv_trim + ets_fv    # OLS component + ETS component
    train_t      = train_times.iloc[-min_len:].values

    # ── Rolling one-step-ahead test forecast ──────────────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    history_resid = list(train_for_ets)   # residuals if multivariate, full series if univariate
    test_preds    = []

    for i in range(len(test_series)):
        y_hist = np.array(history_resid, dtype=float)
        try:
            _sp = best_sp if _seasonal else None
            if _seasonal and len(y_hist) < 2 * (_sp or 2):
                _seas_use, _sp_use = None, None
            else:
                _seas_use, _sp_use = _seasonal, _sp
            m  = ETSModel(y_hist, error=_error, trend=_trend,
                          damped_trend=_damped, seasonal=_seas_use,
                          seasonal_periods=_sp_use)
            f  = m.fit(disp=False, maxiter=300)
            fc = float(f.forecast(1).iloc[0])
        except Exception:
            fc = float(history_resid[-1]) if history_resid else 0.0

        # Add OLS component
        total_pred = fc + float(test_ols_pred[i])
        test_preds.append(total_pred)

        # Update residual history with actual residual
        actual_resid = float(test_series.iloc[i]) - float(test_ols_pred[i])
        history_resid.append(actual_resid)

    pred_te   = np.array(test_preds)
    actual_te = test_series.values

    # ── Metrics ──────────────────────────────────────────────────────────
    def _mape(a, p):
        return float(np.nanmean(np.abs((a - p) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(train_act, train_pred_v))
    mae_tr  = float(mean_absolute_error(train_act, train_pred_v))
    mape_tr = _mape(train_act, train_pred_v)
    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te  = float(mean_absolute_error(actual_te, pred_te))
    mape_te = _mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Full-data refit for forecasting ────────────────────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    if is_multivariate:
        X_full = _make_X(df, exog_cols)
        y_full = series.values.astype(float)
        ols_beta_full, ols_full_fitted = _fit_ols(X_full, y_full)
        resid_full = y_full - ols_full_fitted
    else:
        ols_beta_full = None
        resid_full    = series.values.astype(float)

    try:
        full_sp  = best_sp if _seasonal else None
        full_fit = ETSModel(
            resid_full, error=_error, trend=_trend,
            damped_trend=_damped, seasonal=_seasonal,
            seasonal_periods=full_sp,
        ).fit(disp=False, maxiter=500)
    except Exception:
        full_fit = best_fit

    # ── Future time labels ────────────────────────────────────────────────
    last_t = df[time_col].iloc[-1]
    try:
        future_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig      = data[time_col].tolist()
        first_val = orig[0]
        try:    cycle_len = orig[1:].index(first_val) + 1
        except: cycle_len = len(orig)
        cycle    = orig[:cycle_len]
        last_pos = cycle.index(last_t)
        future_labels = [cycle[(last_pos + i) % cycle_len] for i in range(1, horizon + 1)]

    # ── Save bundle ───────────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)
    with open("static/ets_bundle.pkl", "wb") as f:
        pickle.dump({
            "fit":            full_fit,
            "time_labels":    future_labels,
            "model_label":    best_label,
            "is_multivariate": is_multivariate,
            "ols_beta":       ols_beta_full if is_multivariate else None,
            "exog_cols":      exog_cols,
            "params":         params,
        }, f)

    # ── Plot ──────────────────────────────────────────────────────────────
    train_x = list(range(len(train_act)))
    test_x  = list(range(len(train_act), len(train_act) + len(actual_te)))
    all_x   = train_x + test_x
    all_t   = list(train_t) + list(test_times.values)
    step    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1

    resid_std = float(np.std(best_fit.resid, ddof=1)) if hasattr(best_fit, "resid") else 0.0
    l95_te = pred_te - 1.96 * resid_std * np.sqrt(np.arange(1, len(pred_te) + 1))
    u95_te = pred_te + 1.96 * resid_std * np.sqrt(np.arange(1, len(pred_te) + 1))

    mode_str   = "Multivariate — Regression + ETS Errors" if is_multivariate else "Univariate"
    plot_title = f"{best_label} [{mode_str}] — Training & Testing Set"

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_x, train_act,    color="steelblue",  label="Actual (Train)")
    ax.plot(train_x, train_pred_v, color="steelblue",  label="Fitted (Train)",
            linestyle="--", marker="o", ms=4, alpha=0.8)
    ax.plot(test_x, actual_te, color="darkorange", label="Actual (Test)")
    ax.plot(test_x, pred_te,   color="darkorange", label="Predicted (Test)",
            linestyle="--", marker="o", ms=4, alpha=0.8)
    ax.axvline(x=len(train_act) - 1, color="black", linestyle=":", linewidth=2,
               label="Train/Test Split")
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_t[::step], rotation=45, ha="right")
    ax.set_title(plot_title)
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/ets_train.png", dpi=120)
    plt.savefig("static/ets_test.png",  dpi=120)
    plt.close()

    # ── Parameter table & IC ──────────────────────────────────────────────
    param_table   = _param_table(
        best_fit, best_label, best_sp,
        ols_beta=ols_beta if is_multivariate else None,
        exog_cols=exog_cols if is_multivariate else None,
    )
    info_criteria = _info_criteria(best_fit)

    # ── Model config dict ──────────────────────────────────────────────────
    # ── Model config dict ──────────────────────────────────────────────────
    # Build a proper name→value dict regardless of whether .params is a
    # numpy array (older statsmodels) or a pandas Series (newer statsmodels)
    def _get_params_dict(fit):
        """Return {param_name: value} regardless of statsmodels version."""
        try:
            p = fit.params
            if hasattr(p, "index"):          # pandas Series — new statsmodels
                return dict(zip(p.index, p.values))
            elif hasattr(fit, "param_names"): # numpy array — old statsmodels
                return dict(zip(fit.param_names, p))
            else:
                return {}
        except Exception:
            return {}

    _ALIASES = {
        "smoothing_level":    ["smoothing_level",    "alpha"],
        "smoothing_trend":    ["smoothing_trend",     "beta"],
        "smoothing_seasonal": ["smoothing_seasonal",  "gamma"],
        "damping_trend":      ["damping_trend",       "phi"],
    }

    def _safe(name):
        p = _get_params_dict(best_fit)
        # Pass 1: exact key match
        for key in _ALIASES.get(name, [name]):
            if key in p:
                try:
                    val = float(p[key])
                    if not np.isnan(val):
                        return round(val, 4)
                except Exception:
                    pass
        # Pass 2: substring match (handles unusual key names)
        for idx_key, val in p.items():
            for candidate in _ALIASES.get(name, [name]):
                if candidate in str(idx_key).lower():
                    try:
                        fval = float(val)
                        if not np.isnan(fval):
                            return round(fval, 4)
                    except Exception:
                        pass
        return "N/A"

    
    model_config = {
        "Model":              best_label,
        "Mode":               "Regression + ETS Errors" if is_multivariate else "Univariate ETS",
        "Error Component":    _error.capitalize() if _error else "Additive",
        "Trend Component":    ((_trend.capitalize() if _trend else "None") +
                               (" (Damped)" if _damped else "")),
        "Seasonal Component": _seasonal.capitalize() if _seasonal else "None",
        "Seasonal Period":    best_sp if best_sp else "N/A",
        "α (level)":          _safe("smoothing_level"),
        "β (trend)":          _safe("smoothing_trend"),
        "γ (seasonal)":       _safe("smoothing_seasonal"),
        "φ (damping)":        _safe("damping_trend"),
        "AIC":                round(float(best_fit.aic), 4),
        "BIC":                round(float(best_fit.bic), 4),
    }
    if is_multivariate:
        model_config["Exog Variables"] = ", ".join(exog_cols)

    data_type = "Multivariate (ETS)" if is_multivariate else "Univariate (ETS)"

    # ── Output tables ──────────────────────────────────────────────────────
    train_table = pd.DataFrame({
        time_col:    train_t,
        "Actual":    np.round(train_act,    3),
        "Predicted": np.round(train_pred_v, 3),
    })
    test_table = pd.DataFrame({
        time_col:    test_times.values,
        "Actual":    np.round(actual_te, 3),
        "Predicted": np.round(pred_te,   3),
    })

    return {
        "model_key":   "ets",
        "model_name":  best_label,
        "horizon":     horizon,
        "frequency":   frequency,
        "data_type":   data_type,

        "model_config": model_config,
        "auto_tuned":   auto_ets,

        "param_estimates_table": param_table,
        "info_criteria":         info_criteria,

        "rmse_train": round(rmse_tr, 3),
        "mae_train":  round(mae_tr,  3),
        "mape_train": round(mape_tr, 2),
        "rmse_test":  round(rmse_te, 3),
        "mae_test":   round(mae_te,  3),
        "mape_test":  round(mape_te, 2),

        "residuals":      (train_act - train_pred_v).tolist(),
        "train_table":    train_table,
        "test_table":     test_table,
        "forecast_table": None,
    }