import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA as smARIMA
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPER: clean dataframe
# ============================================================
def clean_dataframe(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(exclude=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    df[obj_cols] = df[obj_cols].ffill().bfill()
    return df.dropna()


# ============================================================
# HELPER: build lagged exogenous matrix
# ============================================================
def _build_exog_matrix(df, exog_cols, exog_lags):
    frames = {}
    for col in exog_cols:
        for lag in range(1, exog_lags + 1):
            frames[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(frames, index=df.index)


# ============================================================
# HELPER: build + scale exog array
# ============================================================
def _make_exog_array(raw_df, exog_cols, exog_lags):
    if raw_df is None or exog_lags == 0 or not exog_cols:
        return None, None, None
    lagged = _build_exog_matrix(raw_df.reset_index(drop=True), exog_cols, exog_lags)
    valid  = lagged.notna().all(axis=1).values
    scaler = StandardScaler()
    arr    = scaler.fit_transform(lagged[valid].values.astype(float))
    return valid, arr, scaler


# ============================================================
# HELPER: fit ARMA mean model (statsmodels)
# ============================================================
def _fit_arma(series, ar_lags, ma_lags, exog=None):
    """
    Fits ARMA(ar_lags, ma_lags) via statsmodels ARIMA(p, 0, q).
    Returns the fitted result object.
    """
    y = pd.Series(series.values.astype(float)).reset_index(drop=True)
    model = smARIMA(
        y,
        order=(ar_lags, 0, ma_lags),
        exog=exog,
        trend="c",
    )
    return model.fit()


# ============================================================
# HELPER: parameter estimates DataFrame
# ============================================================
def _param_estimates_df(arma_fit, garch_fit):
    """
    Combines ARMA + GARCH parameter estimates into one tidy DataFrame.
    """
    rows = []

    # ARMA parameters
    try:
        for name, coef, se, pval in zip(
            arma_fit.param_names,
            arma_fit.params,
            arma_fit.bse,
            arma_fit.pvalues,
        ):
            rows.append({
                "Parameter": f"[Mean] {name}",
                "Coef"     : round(float(coef),  4),
                "Std Err"  : round(float(se),    4),
                "z"        : round(float(coef / se) if se != 0 else np.nan, 4),
                "P>|z|"    : round(float(pval),  4),
                "[0.025"   : round(float(coef - 1.96 * se), 4),
                "0.975]"   : round(float(coef + 1.96 * se), 4),
            })
    except Exception:
        pass

    # GARCH parameters
    try:
        ci = garch_fit.conf_int()
        lower_col = ci.columns[0]
        upper_col = ci.columns[1]
        for name, coef, se, pval, lo, hi in zip(
            garch_fit.params.index,
            garch_fit.params.values,
            garch_fit.std_err.values,
            garch_fit.pvalues.values,
            ci[lower_col].values,
            ci[upper_col].values,
        ):
            rows.append({
                "Parameter": f"[Var] {name}",
                "Coef"     : round(float(coef), 4),
                "Std Err"  : round(float(se),   4),
                "z"        : round(float(coef / se) if se != 0 else np.nan, 4),
                "P>|z|"    : round(float(pval), 4),
                "[0.025"   : round(float(lo),   4),
                "0.975]"   : round(float(hi),   4),
            })
    except Exception:
        pass

    if not rows:
        return None
    return pd.DataFrame(rows)


# ============================================================
# HELPER: information criteria
# ============================================================
def _info_criteria(arma_fit, garch_fit):
    criteria = {}
    try:
        criteria["Log-Likelihood (ARMA)"] = round(float(arma_fit.llf),  4)
        criteria["AIC (ARMA)"]            = round(float(arma_fit.aic),  4)
        criteria["BIC (ARMA)"]            = round(float(arma_fit.bic),  4)
    except Exception:
        pass
    try:
        criteria["Log-Likelihood (GARCH)"] = round(float(garch_fit.loglikelihood), 4)
        criteria["AIC (GARCH)"]            = round(float(garch_fit.aic),  4)
        criteria["BIC (GARCH)"]            = round(float(garch_fit.bic),  4)
    except Exception:
        pass
    return criteria


# ============================================================
# AUTO-TUNE
# ============================================================
def auto_tune_armagarch(train_series, exog_train, has_exogenous,
                        params, log_callback=None):
    """
    Grid search over ar_lags in [1,3], ma_lags in [0,2],
    garch_p in [1,2], garch_q in [1,2].
    Two-step: statsmodels ARMA mean → arch GARCH on residuals.
    Best model selected by combined AIC.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Running Auto ARMA-GARCH grid search (ar_lags, ma_lags, p, q)...")

    best_aic   = np.inf
    best_combo = None

    for ar_try in range(1, 4):
        for ma_try in range(0, 3):
            for p_try in range(1, 3):
                for q_try in range(1, 3):
                    try:
                        arma_fit = _fit_arma(
                            train_series, ar_try, ma_try, exog=exog_train
                        )
                        arma_resid = pd.Series(
                            arma_fit.resid.values.astype(float)
                        ).dropna().reset_index(drop=True)

                        gm = arch_model(
                            arma_resid,
                            mean="Zero",
                            vol="GARCH",
                            p=p_try,
                            q=q_try,
                            dist="normal",
                        )
                        gr = gm.fit(disp="off", show_warning=False)

                        combined_aic = float(arma_fit.aic) + float(gr.aic)

                        prefix = "ARMAX" if has_exogenous else "ARMA"
                        tag    = f"{prefix}({ar_try},{ma_try})-GARCH({p_try},{q_try})"
                        log(f"   Testing {tag} → Combined AIC={combined_aic:.2f}")

                        if combined_aic < best_aic:
                            best_aic   = combined_aic
                            best_combo = {
                                "ar_lags": ar_try,
                                "ma_lags": ma_try,
                                "garch_p": p_try,
                                "garch_q": q_try,
                            }
                            log(f"   ✨ New best: {tag}  Combined AIC={best_aic:.2f}")
                    except Exception:
                        continue

    return best_combo


# ============================================================
# ARCH LM TEST
# ============================================================
def _arch_lm_test(series, lags=5, log=None):
    import pmdarima as pm

    y = pd.Series(series.values.astype(float)).reset_index(drop=True)

    try:
        arima_fit = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            information_criterion='aicc',
            test='kpss',
            max_p=5, max_q=5,
            max_d=2,
            start_p=2, start_q=2,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order   = arima_fit.order
        p, d, q = order
        trend   = 'c' if d == 0 else ('t' if d == 1 else 'n')
        sm_fit      = smARIMA(y, order=order, trend=trend).fit()
        residuals   = sm_fit.resid
        arima_order = order
    except Exception as e:
        if log:
            log(f"\n⚠️  Auto-ARIMA failed: {str(e)}")
            log("   Treating as NOT significant — ARMA-GARCH will not be fitted.")
        return 0.0, 1.0, False

    lm_stat, p_value, f_stat, f_p_value = het_arch(residuals, nlags=lags)

    if log:
        log("\n" + "=" * 60)
        log("📋 ARCH LM Test (Engle's Test for ARCH Effects)")
        drift_str = "with drift" if d == 1 else ("with mean" if d == 0 else "no constant")
        log(f"   Mean model   : Auto-ARIMA{arima_order} {drift_str} residuals")
        log(f"   ARCH lags    : {lags}")
        log(f"   LM Statistic : {lm_stat:.4f}")
        log(f"   p-value      : {p_value:.4f}")
        if p_value < 0.05:
            log(f"   ✅ ARCH effects detected (p={p_value:.4f} < 0.05) → Proceed with ARMA-GARCH")
        else:
            log(f"   ❌ No ARCH effects (p={p_value:.4f} ≥ 0.05) → ARMA-GARCH NOT appropriate")
        log("=" * 60)

    return lm_stat, p_value, p_value < 0.05


# ============================================================
# FORECAST-ONLY
# ============================================================
def _forecast_only_armagarch(params, horizon):
    for path in ("static/armagarch_bundle.pkl",
                 "static/armagarch_arma.pkl",
                 "static/armagarch_garch.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Please train the ARMA-GARCH model first."
            )

    with open("static/armagarch_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("static/armagarch_arma.pkl",  "rb") as f:
        arma_fit_full = pickle.load(f)
    with open("static/armagarch_garch.pkl", "rb") as f:
        garch_fit_full = pickle.load(f)

    time_labels    = bundle["time_labels"]
    diff_order     = bundle.get("diff_order",     0)
    last_val       = bundle.get("last_val",       None)
    has_exogenous  = bundle.get("has_exogenous",  False)
    exog_cols      = bundle.get("exog_cols",      [])
    exog_lags      = bundle.get("exog_lags",      0)
    n_exog_feats   = bundle.get("n_exog_feats",   0)
    x_scaler_mean  = bundle.get("x_scaler_mean",  None)
    x_scaler_scale = bundle.get("x_scaler_scale", None)

    future_labels = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    # ── ARMA mean forecast ────────────────────────────────────────────
    if has_exogenous:
        future_exog_dict = params.get("future_exog")
        if not future_exog_dict:
            raise ValueError("Provide future exog values via params['future_exog'].")
        missing = [c for c in exog_cols if c not in future_exog_dict]
        if missing:
            raise ValueError(f"Missing future exog values for: {missing}")
        future_exog_raw = np.column_stack([
            np.array(future_exog_dict[c][:horizon], dtype=float)
            for c in exog_cols
        ])
        future_exog_tiled  = np.hstack([future_exog_raw] * exog_lags)[:, :n_exog_feats]
        future_exog_scaled = (future_exog_tiled - x_scaler_mean) / x_scaler_scale
        arma_fc = arma_fit_full.forecast(steps=horizon, exog=future_exog_scaled)
    else:
        arma_fc = arma_fit_full.forecast(steps=horizon)

    mean_fc = np.array(arma_fc, dtype=float)
    if diff_order == 1 and last_val is not None:
        mean_fc = last_val + np.cumsum(mean_fc)

    # ── GARCH variance forecast ───────────────────────────────────────
    method = "analytic" if horizon == 1 else "simulation"
    try:
        fc_var = garch_fit_full.forecast(
            horizon=horizon, method=method, reindex=False
        ).variance.values.flatten()[:horizon]
    except Exception:
        fc_var = np.full(
            horizon,
            float(garch_fit_full.conditional_volatility.values[-1] ** 2)
        )

    std_v    = np.sqrt(np.maximum(fc_var, 0.0))
    lower_95 = mean_fc - 1.960 * std_v
    upper_95 = mean_fc + 1.960 * std_v
    lower_80 = mean_fc - 1.282 * std_v
    upper_80 = mean_fc + 1.282 * std_v

    df_out = pd.DataFrame({
        "Period"        : future_labels,
        "Forecast"      : np.round(mean_fc,  3),
        "Lower 80%"     : np.round(lower_80, 3),
        "Upper 80%"     : np.round(upper_80, 3),
        "Lower 95%"     : np.round(lower_95, 3),
        "Upper 95%"     : np.round(upper_95, 3),
        "Interval (95%)": [f"[{l:.1f}, {u:.1f}]"
                           for l, u in zip(lower_95, upper_95)],
    })
    return {"forecast_table": df_out}


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_armagarch(
    data         : pd.DataFrame,
    params       : dict,
    horizon      : int,
    frequency    : str,
    mode         : str = "train",
    log_callback       = None,
):
    """
    ARMA-GARCH / ARMAX-GARCH  —  two-step implementation.

    WHY TWO-STEP:
      The arch library does NOT support mean='ARMA' directly.
      Valid mean models in arch are: Constant, Zero, AR, ARX, LS, HAR, HARX.
      Solution: fit ARMA mean via statsmodels, then fit GARCH on residuals via arch.

    Step 1 — Mean  : ARMA(ar_lags, ma_lags)  [statsmodels]
    Step 2 — Var   : GARCH(p, q) on residuals [arch, mean='Zero']
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    if mode == "forecast":
        return _forecast_only_armagarch(params=params, horizon=horizon)

    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ───────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols  = params.get("exog_cols",  []) or []

    has_exogenous = len(exog_cols) > 0
    model_label   = "ARMAX-GARCH" if has_exogenous else "ARMA-GARCH"

    log(f"📊 {model_label} — {'Multivariate' if has_exogenous else 'Univariate'}")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")
    log(f"   Exog variables : {exog_cols if exog_cols else 'None'}")
    log(f"   Approach       : statsmodels ARMA mean → arch GARCH on residuals")

    # ── Hyperparameters ───────────────────────────────────────────────
    split      = params.get("split",          0.85)
    use_auto   = params.get("auto_armagarch", True)
    diff_order = params.get("diff_order",      0)

    ar_lags   = params.get("ar_lags",   1)
    ma_lags   = params.get("ma_lags",   1)
    garch_p   = params.get("garch_p",   1)
    garch_q   = params.get("garch_q",   1)
    exog_lags = params.get("exog_lags", 1)
    arch_lags = params.get("arch_lags", 5)

    log("=" * 60)
    log(f"   auto_armagarch : {use_auto}")
    log(f"   diff_order     : {diff_order}")
    log(f"   Manual: ar_lags={ar_lags}, ma_lags={ma_lags}, p={garch_p}, q={garch_q}")
    log(f"   arch_lags      : {arch_lags}")
    log("=" * 60)

    # ── Data preparation ──────────────────────────────────────────────
    cols_needed = [time_col, target_col] + exog_cols
    df_full = data[cols_needed].copy()

    if pd.api.types.is_numeric_dtype(df_full[time_col]):
        df_full = df_full.sort_values(time_col).reset_index(drop=True)
    else:
        df_full = df_full.reset_index(drop=True)

    df_full = clean_dataframe(df_full)
    series  = df_full[target_col].copy().reset_index(drop=True)
    times   = df_full[time_col].copy().reset_index(drop=True)

    last_val_before_diff = float(series.iloc[-1])

    if diff_order == 1:
        series_model = series.diff().dropna().reset_index(drop=True)
        times_model  = times.iloc[1:].reset_index(drop=True)
        exog_raw_df  = df_full[exog_cols].iloc[1:].reset_index(drop=True) if has_exogenous else None
        log("   ℹ️  Applied 1st-order differencing.")
    else:
        series_model = series.reset_index(drop=True)
        times_model  = times.reset_index(drop=True)
        exog_raw_df  = df_full[exog_cols].reset_index(drop=True) if has_exogenous else None

    exog_valid_mask, exog_arr_all, exog_scaler = _make_exog_array(
        exog_raw_df, exog_cols, exog_lags if has_exogenous else 0
    )

    if has_exogenous:
        series_aligned = series_model[exog_valid_mask].reset_index(drop=True)
        times_aligned  = times_model[exog_valid_mask].reset_index(drop=True)
    else:
        series_aligned = series_model
        times_aligned  = times_model
        exog_arr_all   = None

    # ── Train / test split ────────────────────────────────────────────
    n         = len(series_aligned)
    split_idx = int(split * n)

    train_series = series_aligned.iloc[:split_idx].reset_index(drop=True)
    test_series  = series_aligned.iloc[split_idx:].reset_index(drop=True)
    train_times  = times_aligned.iloc[:split_idx].reset_index(drop=True)
    test_times   = times_aligned.iloc[split_idx:].reset_index(drop=True)

    exog_train = exog_arr_all[:split_idx] if exog_arr_all is not None else None
    exog_test  = exog_arr_all[split_idx:] if exog_arr_all is not None else None

    # ── ARCH LM Test ──────────────────────────────────────────────────
    lm_stat, lm_pvalue, arch_significant = _arch_lm_test(
        series_aligned, lags=arch_lags, log=log
    )
    if not arch_significant:
        log("\n⚠️  ARCH LM test not significant — ARMA-GARCH is not appropriate.")
        return {
            "model_key"        : "armagarch",
            "model_name"       : model_label,
            "data_type"        : "Univariate" if not has_exogenous else "Multivariate",
            "arch_lm_stat"     : round(lm_stat,   4),
            "arch_lm_pvalue"   : round(lm_pvalue, 4),
            "arch_significant" : False,
            "arch_not_significant_message": (
                f"The ARCH LM test is NOT significant (LM Stat = {lm_stat:.4f}, "
                f"p-value = {lm_pvalue:.4f} ≥ 0.05). "
                f"No ARCH effects detected — {model_label} is not appropriate. "
                f"Consider using ARIMA instead."
            ),
            "train_table"   : None, "test_table"    : None, "forecast_table": None,
            "rmse_train"    : None, "mae_train"     : None, "mape_train"    : None,
            "rmse_test"     : None, "mae_test"      : None, "mape_test"     : None,
        }

    # ── AUTO-TUNE ─────────────────────────────────────────────────────
    if use_auto:
        best = auto_tune_armagarch(
            train_series, exog_train, has_exogenous, params, log_callback=log
        )
        if best:
            ar_lags = best["ar_lags"]
            ma_lags = best["ma_lags"]
            garch_p = best["garch_p"]
            garch_q = best["garch_q"]
            prefix  = "ARMAX" if has_exogenous else "ARMA"
            log(f"\n✅ Best: {prefix}({ar_lags},{ma_lags})-GARCH({garch_p},{garch_q})")
            params.update(best)

    # ── Step 1: Fit ARMA on TRAIN ────────────────────────────────────
    log(f"\n🔧 Fitting ARMA({ar_lags},{ma_lags}) mean on training set...")
    arma_fit_train  = _fit_arma(train_series, ar_lags, ma_lags, exog=exog_train)
    arma_resid_train = pd.Series(
        arma_fit_train.resid.values.astype(float)
    ).dropna().reset_index(drop=True)

    # ── Step 2: Fit GARCH on residuals ───────────────────────────────
    log(f"🔧 Fitting GARCH({garch_p},{garch_q}) on ARMA residuals...")
    garch_model_train = arch_model(
        arma_resid_train,
        mean="Zero",
        vol="GARCH",
        p=garch_p,
        q=garch_q,
        dist="normal",
    )
    garch_fit_train = garch_model_train.fit(disp="off", show_warning=False)

    # ── In-sample train predictions ───────────────────────────────────
    try:
        burn = int(arma_fit_train.loglikelihood_burn)
    except Exception:
        burn = 0
    # Conservative burn-in: 2*(p+q) + 10, minimum 15
    # Ensures Kalman filter has fully converged for ARMAX models
    valid_start = max(2 * (ar_lags + ma_lags) + burn + 5, 15)
    # Safety: ensure at least 10 observations remain for metrics
    if valid_start >= len(train_series) - 10:
        valid_start = max(ar_lags + ma_lags + burn, ar_lags + ma_lags)

    # Use fittedvalues directly (= y - resid, no Kalman call needed)
    train_fitted = train_series.values - arma_fit_train.resid.values
    train_pred   = train_fitted[valid_start:]
    train_act   = train_series.values[valid_start:]
    train_t     = train_times.values[valid_start:]

    cond_vol      = garch_fit_train.conditional_volatility.values
    residuals_std = (
        arma_resid_train.values[:len(cond_vol)] / (cond_vol + 1e-8)
    )

    param_estimates_table = _param_estimates_df(arma_fit_train, garch_fit_train)
    info_criteria         = _info_criteria(arma_fit_train, garch_fit_train)

    # ── Rolling one-step-ahead test forecast ─────────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")
    test_preds   = []
    test_lower95 = []
    test_upper95 = []
    z95 = 1.960

    history_s = train_series.copy().tolist()
    history_x = list(exog_train) if exog_train is not None else None

    # WITH (adds jitter retry before fallback):
    last_good_fc  = float(pd.Series(history_s).iloc[-1])
    last_good_std = float(pd.Series(history_s).std())

    for i in range(len(test_series)):
        arma_fc = last_good_fc
        std_v   = last_good_std
        try:
            hs = pd.Series(history_s)
            hx = np.array(history_x) if history_x is not None else None

            # Try fitting; if LU decomposition fails, add small jitter and retry once
            try:
                arma_tmp = _fit_arma(hs, ar_lags, ma_lags, exog=hx)
            except Exception:
                # Jitter retry: add tiny noise to break numerical singularity
                hs_jitter = hs + np.random.normal(0, 1e-6, len(hs))
                arma_tmp  = _fit_arma(hs_jitter, ar_lags, ma_lags, exog=hx)

            next_x  = exog_test[i:i+1] if exog_test is not None else None
            arma_fc = float(arma_tmp.forecast(steps=1, exog=next_x).iloc[0])

            resid_tmp = pd.Series(arma_tmp.resid.values.astype(float)).dropna()
            gm_tmp    = arch_model(
                resid_tmp, mean="Zero", vol="GARCH",
                p=garch_p, q=garch_q, dist="normal",
            )
            gf_tmp = gm_tmp.fit(disp="off", show_warning=False)
            fc_var = float(
                gf_tmp.forecast(horizon=1, reindex=False).variance.values.flatten()[0]
            )
            std_v = np.sqrt(max(fc_var, 0.0))
            last_good_fc  = arma_fc
            last_good_std = std_v

        except Exception as e:
            log(f"⚠️ Rolling step {i} failed: {e} — using last valid prediction")
            # arma_fc and std_v already set to last_good values above

        test_preds.append(arma_fc)
        test_lower95.append(arma_fc - z95 * std_v)
        test_upper95.append(arma_fc + z95 * std_v)

        history_s.append(float(test_series.iloc[i]))
        if history_x is not None:
            history_x.append(exog_test[i])

    pred_te   = np.array(test_preds)
    actual_te = test_series.values

    # ── Undo differencing ─────────────────────────────────────────────
    if diff_order == 1:
        offset     = exog_lags if has_exogenous else 0
        base_idx   = split_idx + offset
        base_level = series.iloc[base_idx] if base_idx < len(series) else series.iloc[-1]
        actual_te_ = series.values[base_idx + 1: base_idx + 1 + len(pred_te)]
        pred_te_   = base_level + np.cumsum(pred_te)
        train_act_ = series.values[(1 + offset): split_idx + 1 + offset][valid_start:]
        train_pred_= train_act_ - arma_fit_train.resid.values[valid_start:]
    else:
        actual_te_ = actual_te
        pred_te_   = pred_te
        train_act_ = train_act
        train_pred_= train_pred

    min_tr = min(len(train_act_), len(train_pred_))
    min_te = min(len(actual_te_), len(pred_te_))
    train_act_  = train_act_[:min_tr]
    train_pred_ = train_pred_[:min_tr]
    actual_te_  = actual_te_[:min_te]
    pred_te_    = pred_te_[:min_te]
    train_t     = train_t[:min_tr]

    # ── Metrics ───────────────────────────────────────────────────────
    def mape(a, p_):
        return float(np.mean(np.abs((a - p_) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(train_act_, train_pred_))
    mae_tr  = float(mean_absolute_error(train_act_,     train_pred_))
    mape_tr = mape(train_act_, train_pred_)
    rmse_te = float(root_mean_squared_error(actual_te_, pred_te_))
    mae_te  = float(mean_absolute_error(actual_te_,     pred_te_))
    mape_te = mape(actual_te_, pred_te_)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Future time labels ────────────────────────────────────────────
    last_t = df_full[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig_sequence = data[time_col].tolist()
        first_val     = orig_sequence[0]
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

    # ── Refit on full data ────────────────────────────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    arma_fit_full  = _fit_arma(series_aligned, ar_lags, ma_lags, exog=exog_arr_all)
    resid_full     = pd.Series(
        arma_fit_full.resid.values.astype(float)
    ).dropna().reset_index(drop=True)
    garch_fit_full = arch_model(
        resid_full, mean="Zero", vol="GARCH", p=garch_p, q=garch_q, dist="normal"
    ).fit(disp="off", show_warning=False)

    # ── Save pickles ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)
    n_exog_feats = int(exog_arr_all.shape[1]) if exog_arr_all is not None else 0

    prefix    = "ARMAX" if has_exogenous else "ARMA"
    title_str = (
        f"{prefix}({ar_lags},{ma_lags})-GARCH({garch_p},{garch_q})"
        + (" [Multivariate]" if has_exogenous else " [Univariate]")
    )

    with open("static/armagarch_arma.pkl",  "wb") as f:
        pickle.dump(arma_fit_full,  f)
    with open("static/armagarch_garch.pkl", "wb") as f:
        pickle.dump(garch_fit_full, f)
    with open("static/armagarch_model.pkl", "wb") as f:  # legacy key
        pickle.dump(garch_fit_full, f)
    with open("static/armagarch_bundle.pkl", "wb") as f:
        pickle.dump({
            "ar_lags"       : ar_lags,
            "ma_lags"       : ma_lags,
            "garch_p"       : garch_p,
            "garch_q"       : garch_q,
            "diff_order"    : diff_order,
            "last_val"      : last_val_before_diff,
            "last_time"     : last_t,
            "time_labels"   : future_time_labels,
            "time_col"      : time_col,
            "target_col"    : target_col,
            "exog_cols"     : exog_cols,
            "exog_lags"     : exog_lags,
            "has_exogenous" : has_exogenous,
            "n_exog_feats"  : n_exog_feats,
            "x_scaler_mean" : exog_scaler.mean_  if exog_scaler else None,
            "x_scaler_scale": exog_scaler.scale_ if exog_scaler else None,
            "residuals_std" : residuals_std.tolist(),
            "params"        : params,
        }, f)

    # ── Plot ──────────────────────────────────────────────────────────
    train_x = list(range(len(train_act_)))
    test_x  = list(range(len(train_act_), len(train_act_) + len(actual_te_)))
    all_x   = train_x + test_x
    all_t   = list(train_t) + list(test_times.values[:min_te])
    step    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_x, train_act_,  color="steelblue",  label="Actual (Train)")
    ax.plot(train_x, train_pred_, color="steelblue",
            linestyle="--", marker="o", ms=4, alpha=0.8, label="Predicted (Train)")
    ax.plot(test_x,  actual_te_,  color="darkorange", label="Actual (Test)")
    ax.plot(test_x,  pred_te_,    color="darkorange",
            linestyle="--", marker="o", ms=4, alpha=0.8, label="Predicted (Test)")
    ax.axvline(x=len(train_act_) - 1, color="black", linestyle=":", linewidth=2,
               label="Train/Test Split")
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_t[::step], rotation=45, ha="right")
    ax.set_title(f"{title_str} — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/armagarch_train.png", dpi=120)
    plt.savefig("static/armagarch_test.png",  dpi=120)
    plt.close()

    # ── Output tables ─────────────────────────────────────────────────
    train_table = pd.DataFrame({
        time_col   : train_t,
        "Actual"   : np.round(train_act_,  3),
        "Predicted": np.round(train_pred_, 3),
    })
    test_table = pd.DataFrame({
        time_col   : test_times.values[:min_te],
        "Actual"   : np.round(actual_te_, 3),
        "Predicted": np.round(pred_te_,   3),
    })

    return {
        "model_key"   : "armagarch",
        "model_name"  : title_str,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",
        "model_config": {
            "Mean Model"     : f"{prefix}({ar_lags},{ma_lags})",
            "AR Lags"        : ar_lags,
            "MA Lags"        : ma_lags,
            "p (ARCH Order)" : garch_p,
            "q (GARCH Order)": garch_q,
            "Differencing"   : diff_order,
            "Exog Cols"      : exog_cols if exog_cols else "None",
            "Exog Lags"      : exog_lags if has_exogenous else "N/A",
        },
        "auto_tuned"     : use_auto,
        "order"          : title_str,
        "auto_armagarch" : use_auto,

        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,

        "rmse_train" : round(rmse_tr, 3),
        "mae_train"  : round(mae_tr,  3),
        "mape_train" : round(mape_tr, 2),
        "rmse_test"  : round(rmse_te, 3),
        "mae_test"   : round(mae_te,  3),
        "mape_test"  : round(mape_te, 2),
        "residuals"  : (train_act - train_pred).tolist(),

        "train_table"   : train_table,
        "test_table"    : test_table,
        "forecast_table": None,

        "arch_lm_stat"     : round(lm_stat,   4),
        "arch_lm_pvalue"   : round(lm_pvalue, 4),
        "arch_significant" : arch_significant,
    }