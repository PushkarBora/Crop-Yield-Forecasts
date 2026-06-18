import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch
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
    """
    Lags each exog col from 1..exog_lags.
    Returns DataFrame; NaN rows NOT yet dropped — caller handles alignment.
    """
    frames = {}
    for col in exog_cols:
        for lag in range(1, exog_lags + 1):
            frames[f"{col}_lag{lag}"] = df[col].shift(lag)
    return pd.DataFrame(frames, index=df.index)


# ============================================================
# HELPER: build + scale exog array
# ============================================================
def _make_exog_array(raw_df, exog_cols, exog_lags):
    """
    Returns (valid_mask, exog_arr_scaled, fitted_scaler).
    valid_mask       : boolean array marking non-NaN rows after lagging.
    exog_arr_scaled  : 2-D float array shape (sum(valid_mask), n_features).
    """
    if raw_df is None or exog_lags == 0 or not exog_cols:
        return None, None, None
    lagged = _build_exog_matrix(raw_df.reset_index(drop=True), exog_cols, exog_lags)
    valid  = lagged.notna().all(axis=1).values
    scaler = StandardScaler()
    arr    = scaler.fit_transform(lagged[valid].values.astype(float))
    return valid, arr, scaler


# ============================================================
# HELPER: parameter estimates DataFrame
# ============================================================
def _param_estimates_df(fit):
    try:
        ci = fit.conf_int()
        lower_col = ci.columns[0]
        upper_col = ci.columns[1]
        return pd.DataFrame({
            "Parameter": fit.params.index.tolist(),
            "Coef"     : np.round(fit.params.values,    4),
            "Std Err"  : np.round(fit.std_err.values,   4),
            "z"        : np.round(fit.tvalues.values,   4),
            "P>|z|"    : np.round(fit.pvalues.values,   4),
            "[0.025"   : np.round(ci[lower_col].values, 4),
            "0.975]"   : np.round(ci[upper_col].values, 4),
        })
    except Exception:
        return None


# ============================================================
# HELPER: information criteria
# ============================================================
def _info_criteria(fit):
    criteria = {}
    try:
        criteria["Log-Likelihood"] = round(float(fit.loglikelihood), 4)
    except Exception:
        pass
    try:
        criteria["AIC"] = round(float(fit.aic), 4)
    except Exception:
        pass
    try:
        k = int(len(fit.params))
        n = int(fit.nobs)
        if n - k - 1 > 0:
            criteria["AICc"] = round(float(fit.aic) + 2 * k * (k + 1) / (n - k - 1), 4)
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
    return criteria


# ============================================================
# AUTO-TUNE — grid search over (ar_lags, p, o, q)
# ============================================================
def auto_tune_argjrgarch(train_series, exog_train, has_exogenous,
                       params, log_callback=None):
    """
    Grid search over:
      ar_lags ∈ [1, 3]  — AR order in mean equation
      p       ∈ [1, 2]  — ARCH order
      o       ∈ [1, 2]  — Threshold / asymmetry order (always ≥ 1)
      q       ∈ [1, 2]  — GARCH order

    Mean  : AR (univariate) / ARX (multivariate)
    Var   : GJR-GARCH(p, o, q)  via arch: vol='GARCH', o=o_order
    Best model selected by AIC.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    log("\n🔍 Running Auto AR-GJR-GARCH grid search (ar_lags, p, o, q)...")

    mean_model = "ARX" if has_exogenous else "AR"
    best_aic   = np.inf
    best_combo = None

    for ar_try in range(1, 4):
        for p_try in range(1, 3):
            for o_try in range(1, 3):          # o ≥ 1 — threshold term always present
                for q_try in range(1, 3):
                    try:
                        m = arch_model(
                            train_series,
                            x=exog_train,
                            mean=mean_model,
                            lags=ar_try,
                            vol="GARCH",
                            p=p_try,
                            o=o_try,
                            q=q_try,
                            dist="normal",
                        )
                        r = m.fit(disp="off", show_warning=False)

                        prefix = "ARX" if has_exogenous else "AR"
                        tag    = f"{prefix}({ar_try})-GJR-GARCH({p_try},{o_try},{q_try})"
                        log(f"   Testing {tag} → AIC={r.aic:.2f}")

                        if r.aic < best_aic:
                            best_aic   = r.aic
                            best_combo = {
                                "ar_lags" : ar_try,
                                "gjrgarch_p": p_try,
                                "gjrgarch_o": o_try,
                                "gjrgarch_q": q_try,
                            }
                            log(f"   ✨ New best: {tag}  AIC={best_aic:.2f}")
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
            information_criterion='aic',
            test='kpss',
            max_p=5, max_q=5,
            max_d=2,
            start_p=0, start_q=0,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order       = arima_fit.order
        p, d, q     = order
        # Drop first residual (Kalman init artifact) — matches ar_egarch_model.py
        residuals   = pd.Series(arima_fit.resid()).iloc[1:].reset_index(drop=True)
        arima_order = order

    except Exception as e:
        if log:
            log(f"\n⚠️  Auto-ARIMA failed: {str(e)}")
            log("   Cannot perform ARCH LM test — AR-GJR-GARCH will not be fitted.")
        return 0.0, 1.0, False

    lm_stat, p_value, f_stat, f_p_value = het_arch(residuals, nlags=lags)

    if log:
        log("\n" + "=" * 60)
        log("📋 ARCH LM Test (Engle's Test for ARCH Effects)")
        drift_str = "with drift" if d == 1 else ("with mean" if d == 0 else "no constant")
        log(f"   Mean model   : Auto-ARIMA{arima_order} {drift_str} residuals")
        log(f"   ARCH lags    : {lags}")
        log(f"   H0 : No ARCH effects in residuals")
        log(f"   H1 : ARCH effects present → AR-GJR-GARCH is appropriate")
        log(f"   LM Statistic : {lm_stat:.4f}")
        log(f"   p-value      : {p_value:.4f}")
        log(f"   F-Statistic  : {f_stat:.4f}")
        log(f"   F p-value    : {f_p_value:.4f}")
        if p_value < 0.05:
            log(f"   ✅ ARCH effects detected (p={p_value:.4f} < 0.05) → Proceed with AR-GJR-GARCH")
        else:
            log(f"   ❌ No ARCH effects (p={p_value:.4f} ≥ 0.05) → AR-GJR-GARCH NOT appropriate")
        log("=" * 60)

    return lm_stat, p_value, p_value < 0.05


# ============================================================
# FORECAST-ONLY
# ============================================================
def _forecast_only_argjrgarch(params, horizon):
    for path in ("static/argjrgarch_bundle.pkl", "static/argjrgarch_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Please train the AR-GJR-GARCH model first."
            )

    with open("static/argjrgarch_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    with open("static/argjrgarch_model.pkl", "rb") as f:
        fit_full = pickle.load(f)

    time_labels    = bundle["time_labels"]
    diff_order     = bundle.get("diff_order",     0)
    last_val       = bundle.get("last_val",       None)
    has_exogenous  = bundle.get("has_exogenous",  False)
    exog_cols      = bundle.get("exog_cols",      [])
    exog_lags      = bundle.get("exog_lags",      0)
    n_exog_feats   = bundle.get("n_exog_feats",   0)
    x_scaler_mean  = bundle.get("x_scaler_mean",  None)
    x_scaler_scale = bundle.get("x_scaler_scale", None)
    ar_lags        = bundle.get("ar_lags",        1)
    gjrgarch_p       = bundle.get("gjrgarch_p",       1)
    gjrgarch_o       = bundle.get("gjrgarch_o",       1)
    gjrgarch_q       = bundle.get("gjrgarch_q",       1)

    future_labels = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    # ── Build future exog if multivariate ────────────────────────────────
    if has_exogenous:
        future_exog_dict = params.get("future_exog")
        if not future_exog_dict:
            raise ValueError(
                "Model was trained with exogenous variables. "
                "Provide future values via params['future_exog']."
            )
        missing = [c for c in exog_cols if c not in future_exog_dict]
        if missing:
            raise ValueError(f"Missing future exog values for: {missing}")

        future_exog_raw = np.column_stack([
            np.array(future_exog_dict[c][:horizon], dtype=float)
            for c in exog_cols
        ])
        future_exog_tiled  = np.hstack([future_exog_raw] * exog_lags)[:, :n_exog_feats]
        future_exog_scaled = (future_exog_tiled - x_scaler_mean) / x_scaler_scale

        last_fitted_mean = bundle.get("last_fitted_mean", float(last_val) if last_val else 0.0)
        mean_fc = np.full(horizon, last_fitted_mean)

        if diff_order == 1 and last_val is not None:
            mean_fc = last_val + np.cumsum(mean_fc)

        try:
            n_exog = future_exog_scaled.shape[1]
            x_fc   = (
                future_exog_scaled.reshape(n_exog, horizon, 1)
                if n_exog > 1
                else future_exog_scaled.reshape(1, horizon, 1)
            )
            fc_result = fit_full.forecast(horizon=horizon, x=x_fc, reindex=False)
            var_fc    = fc_result.variance.values.flatten()[:horizon]
        except Exception:
            var_fc = np.full(
                horizon,
                float(fit_full.conditional_volatility.values[-1] ** 2)
            )
    else:
        method = "analytic" if horizon == 1 else "simulation"
        fc     = fit_full.forecast(horizon=horizon, method=method, reindex=False)
        mean_fc = fc.mean.values.flatten()[:horizon]
        var_fc  = fc.variance.values.flatten()[:horizon]

        if diff_order == 1 and last_val is not None:
            mean_fc = last_val + np.cumsum(mean_fc)

    std_v    = np.sqrt(np.maximum(var_fc, 0.0))
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
def run_argjrgarch(
    data         : pd.DataFrame,
    params       : dict,
    horizon      : int,
    frequency    : str,
    mode         : str = "train",
    log_callback       = None,
):
    """
    AR-GJR-GARCH / ARX-GJR-GARCH model.

    Combines:
      Mean equation : AR(ar_lags)  — autoregressive terms
                      ARX(ar_lags) — AR + exogenous regressors
      Variance eq.  : GJR-GARCH(p, o, q) — T-GARCH with leverage effect
                      σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·σ²_{t-1}
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ─────────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_argjrgarch(params=params, horizon=horizon)

    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ───────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols  = params.get("exog_cols",  []) or []

    has_exogenous = len(exog_cols) > 0
    model_label   = "ARX-GJR-GARCH" if has_exogenous else "AR-GJR-GARCH"

    log(f"📊 {model_label} — {'Multivariate' if has_exogenous else 'Univariate'}")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")
    log(f"   Exog variables : {exog_cols if exog_cols else 'None'}")
    log(f"   Mean model     : {'ARX(ar_lags) — AR + Exog regressors' if has_exogenous else 'AR(ar_lags) — AR terms only'}")
    log(f"   Variance model : T-GARCH / GJR-GARCH (p, o, q)")
    log(f"   Asymmetry      : γ·I(ε<0) threshold term captures leverage effect")

    # ── Hyperparameters ───────────────────────────────────────────────
    split      = params.get("split",        0.85)
    use_auto   = params.get("auto_argjrgarch", True)
    diff_order = params.get("diff_order",    0)

    ar_lags   = params.get("ar_lags",   1)
    gjrgarch_p  = params.get("gjrgarch_p",  1)
    gjrgarch_o  = params.get("gjrgarch_o",  1)
    gjrgarch_q  = params.get("gjrgarch_q",  1)
    exog_lags = params.get("exog_lags", 1)
    arch_lags = params.get("arch_lags", 5)

    log("=" * 60)
    log(f"   auto_argjrgarch : {use_auto}")
    log(f"   diff_order    : {diff_order}")
    log(f"   Manual: ar_lags={ar_lags}, p={gjrgarch_p}, o={gjrgarch_o}, q={gjrgarch_q}"
        + (f", exog_lags={exog_lags}" if has_exogenous else ""))
    log(f"   arch_lags     : {arch_lags}")
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

    # ── ARCH LM Test — use raw target series, same as ar_egarch_model.py ──
    raw_target = df_full[target_col].copy().reset_index(drop=True)
    lm_stat, lm_pvalue, arch_significant = _arch_lm_test(
        raw_target, lags=arch_lags, log=log
    )
    if not arch_significant:
        log("\n⚠️  ARCH LM test not significant — AR-GJR-GARCH is not appropriate.")
        return {
            "model_key"        : "argjrgarch",
            "model_name"       : model_label,
            "data_type"        : "Univariate" if not has_exogenous else "Multivariate",
            "arch_lm_stat"     : round(lm_stat,   4),
            "arch_lm_pvalue"   : round(lm_pvalue, 4),
            "arch_significant" : False,
            "arch_not_significant_message": (
                "Auto-ARIMA could not be fitted to compute ARCH LM test residuals. "
                "AR-GJR-GARCH cannot be applied without a valid pre-test."
            ) if lm_stat == 0.0 and lm_pvalue == 1.0 else (
                f"The ARCH LM test is NOT significant (LM Stat = {lm_stat:.4f}, "
                f"p-value = {lm_pvalue:.4f} ≥ 0.05). "
                f"No ARCH effects detected — {model_label} is not appropriate. "
                f"Consider using ARIMA instead."
            ),
            "train_table"   : None,
            "test_table"    : None,
            "forecast_table": None,
            "rmse_train"    : None,
            "mae_train"     : None,
            "mape_train"    : None,
            "rmse_test"     : None,
            "mae_test"      : None,
            "mape_test"     : None,
        }

    # ── AUTO-TUNE ─────────────────────────────────────────────────────
    if use_auto:
        best = auto_tune_argjrgarch(
            train_series, exog_train, has_exogenous,
            params, log_callback=log,
        )
        if best:
            ar_lags  = best["ar_lags"]
            gjrgarch_p = best["gjrgarch_p"]
            gjrgarch_o = best["gjrgarch_o"]
            gjrgarch_q = best["gjrgarch_q"]
            prefix   = "ARX" if has_exogenous else "AR"
            log(f"\n✅ Best: {prefix}({ar_lags})-GJRGARCH({gjrgarch_p},{gjrgarch_o},{gjrgarch_q})")
            params.update(best)

    # ── Fit on TRAIN ──────────────────────────────────────────────────
    mean_model = "ARX" if has_exogenous else "AR"
    model_train = arch_model(
        train_series,
        x=exog_train,
        mean=mean_model,
        lags=ar_lags,
        vol="GARCH",
        p=gjrgarch_p,
        o=gjrgarch_o,
        q=gjrgarch_q,
        dist="normal",
    )
    fit_train = model_train.fit(disp="off", show_warning=False)

    # ── In-sample (train) fitted values ───────────────────────────────
    # Use valid_start = ar_lags to skip NaN-producing initial AR rows
    # This matches ar_egarch_model.py exactly and prevents NaN in metrics
    valid_start = ar_lags
    fitted_mean = train_series.values - fit_train.resid.values
    train_pred  = fitted_mean[valid_start:]
    train_act   = train_series.values[valid_start:]
    train_t     = train_times.values[valid_start:]

    cond_vol      = fit_train.conditional_volatility.values[valid_start:]
    residuals_std = fit_train.resid.values[valid_start:] / (cond_vol + 1e-8)

    param_estimates_table = _param_estimates_df(fit_train)
    info_criteria         = _info_criteria(fit_train)

    # ── Rolling one-step-ahead test forecast ─────────────────────────
    log("\n🔮 Rolling one-step-ahead forecast on test set...")
    test_preds   = []
    test_lower95 = []
    test_upper95 = []
    z95 = 1.960

    history_s = train_series.copy().tolist()
    history_x = list(exog_train) if exog_train is not None else None

    for i in range(len(test_series)):
        try:
            hs = pd.Series(history_s)
            hx = np.array(history_x) if history_x is not None else None

            # Jitter retry for numerical stability
            try:
                m_tmp = arch_model(
                    hs, x=hx, mean=mean_model, lags=ar_lags,
                    vol="GARCH", p=gjrgarch_p, o=gjrgarch_o, q=gjrgarch_q,
                    dist="normal",
                )
                f_tmp = m_tmp.fit(disp="off", show_warning=False)
            except Exception:
                hs_j  = hs + np.random.normal(0, 1e-6, len(hs))
                m_tmp = arch_model(
                    hs_j, x=hx, mean=mean_model, lags=ar_lags,
                    vol="GARCH", p=gjrgarch_p, o=gjrgarch_o, q=gjrgarch_q,
                    dist="normal",
                )
                f_tmp = m_tmp.fit(disp="off", show_warning=False)

            # Shape next_x correctly for N exog variables
            if exog_test is not None:
                row    = exog_test[i]
                n_exog = row.shape[0]
                next_x = row.reshape(1, 1) if n_exog == 1 else row.reshape(n_exog, 1, 1)
            else:
                next_x = None

            fc    = f_tmp.forecast(horizon=1, x=next_x, reindex=False)
            mu    = float(fc.mean.values.flatten()[0])
            var   = float(fc.variance.values.flatten()[0])
            std_v = np.sqrt(max(var, 0.0))

        except Exception as e:
            log(f"⚠️ Rolling step {i} failed: {e} — using last valid prediction")
            mu    = float(pd.Series(history_s).iloc[-1])
            std_v = float(pd.Series(history_s).std())

        test_preds.append(mu)
        test_lower95.append(mu - z95 * std_v)
        test_upper95.append(mu + z95 * std_v)

        history_s.append(float(test_series.iloc[i]))
        if history_x is not None:
            history_x.append(exog_test[i])

    pred_te   = np.array(test_preds)
    actual_te = test_series.values

    # ── Undo differencing — modelled exactly after ar_egarch_model.py ─
    if diff_order == 1:
        offset     = exog_lags if has_exogenous else 0
        base_idx   = split_idx + offset
        base_level = series.iloc[base_idx] if base_idx < len(series) else series.iloc[-1]

        # Test: cumsum differenced predictions back to original scale
        actual_te_ = series.values[base_idx + 1: base_idx + 1 + len(pred_te)]
        pred_te_   = base_level + np.cumsum(pred_te)

        # Train: slice original-scale actuals with valid_start offset applied
        # This matches ar_egarch_model.py:
        #   train_act_ = series.values[(1 + offset): split_idx + 1 + offset][valid_start:]
        #   train_pred_= train_act_ - fit_train.resid.values[valid_start:]
        train_act_ = series.values[(1 + offset): split_idx + 1 + offset][valid_start:]
        train_pred_= train_act_ - fit_train.resid.values[valid_start:]
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

    # ── NaN guard before metrics ──────────────────────────────────────
    def _clean_pair(a, p):
        """Remove NaN/Inf pairs so sklearn metrics don't crash."""
        a = np.array(a, dtype=float)
        p = np.array(p, dtype=float)
        mask = np.isfinite(a) & np.isfinite(p)
        if mask.sum() == 0:
            raise ValueError(
                "All predictions are NaN/Inf — "
                "model failed to converge on this dataset."
            )
        return a[mask], p[mask]

    train_act_c,  train_pred_c = _clean_pair(train_act_,  train_pred_)
    actual_te_c,  pred_te_c    = _clean_pair(actual_te_,  pred_te_)

    # ── Metrics ───────────────────────────────────────────────────────
    def mape(a, p_):
        return float(np.mean(np.abs((a - p_) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(train_act_c, train_pred_c))
    mae_tr  = float(mean_absolute_error(train_act_c,     train_pred_c))
    mape_tr = mape(train_act_c, train_pred_c)
    rmse_te = float(root_mean_squared_error(actual_te_c, pred_te_c))
    mae_te  = float(mean_absolute_error(actual_te_c,     pred_te_c))
    mape_te = mape(actual_te_c, pred_te_c)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Future time labels ────────────────────────────────────────────
    last_t = df_full[time_col].iloc[-1]
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

    # ── Refit on FULL data ────────────────────────────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    model_full = arch_model(
        series_aligned,
        x=exog_arr_all,
        mean=mean_model,
        lags=ar_lags,
        vol="GARCH",
        p=gjrgarch_p,
        o=gjrgarch_o,
        q=gjrgarch_q,
        dist="normal",
    )
    fit_full = model_full.fit(disp="off", show_warning=False)

    last_fitted_mean = float(
        series_aligned.iloc[-1] - fit_full.resid.values[-1]
    )

    # ── Save pickles ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)
    n_exog_feats = int(exog_arr_all.shape[1]) if exog_arr_all is not None else 0

    prefix    = "ARX-GJR-GARCH" if has_exogenous else "AR-GJR-GARCH"
    title_str = (
        f"{prefix}({ar_lags},{gjrgarch_p},{gjrgarch_o},{gjrgarch_q})"
        + (" [Multivariate]" if has_exogenous else " [Univariate]")
    )

    with open("static/argjrgarch_model.pkl", "wb") as f:
        pickle.dump(fit_full, f)

    with open("static/argjrgarch_bundle.pkl", "wb") as f:
        pickle.dump({
            "ar_lags"         : ar_lags,
            "gjrgarch_p"        : gjrgarch_p,
            "gjrgarch_o"        : gjrgarch_o,
            "gjrgarch_q"        : gjrgarch_q,
            "diff_order"      : diff_order,
            "last_val"        : last_val_before_diff,
            "last_time"       : last_t,
            "time_labels"     : future_time_labels,
            "time_col"        : time_col,
            "target_col"      : target_col,
            "exog_cols"       : exog_cols,
            "exog_lags"       : exog_lags,
            "has_exogenous"   : has_exogenous,
            "n_exog_feats"    : n_exog_feats,
            "x_scaler_mean"   : exog_scaler.mean_  if exog_scaler else None,
            "x_scaler_scale"  : exog_scaler.scale_ if exog_scaler else None,
            "last_fitted_mean": last_fitted_mean,
            "params"          : params,
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
    ax.axvline(x=len(train_act_) - 1, color="black", linestyle=":",
               linewidth=2, label="Train/Test Split")
    ax.set_xticks(all_x[::step])
    ax.set_xticklabels(all_t[::step], rotation=45, ha="right")
    ax.set_title(f"{title_str} — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/argjrgarch_train.png", dpi=120)
    plt.savefig("static/argjrgarch_test.png",  dpi=120)
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
        "model_key"   : "argjrgarch",
        "model_name"  : title_str,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",
        "model_config": {
            "Mean Model"          : f"{'ARX' if has_exogenous else 'AR'}({ar_lags})",
            "AR Lags"             : ar_lags,
            "Variance Model"      : f"T-GARCH / GJR-GARCH({gjrgarch_p},{gjrgarch_o},{gjrgarch_q})",
            "p (ARCH Order)"      : gjrgarch_p,
            "o (Threshold Order)" : gjrgarch_o,
            "q (GARCH Order)"     : gjrgarch_q,
            "Differencing"        : diff_order,
            "Exog Cols"           : exog_cols if exog_cols else "None",
            "Exog Lags"           : exog_lags if has_exogenous else "N/A",
        },
        "auto_tuned"   : use_auto,
        "order"        : title_str,
        "auto_argjrgarch": use_auto,

        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,

        "rmse_train" : round(rmse_tr, 3),
        "mae_train"  : round(mae_tr,  3),
        "mape_train" : round(mape_tr, 2),
        "rmse_test"  : round(rmse_te, 3),
        "mae_test"   : round(mae_te,  3),
        "mape_test"  : round(mape_te, 2),

        # Use cleaned train residuals (no NaN)
        "residuals"  : (train_act_c - train_pred_c).tolist(),

        "train_table"   : train_table,
        "test_table"    : test_table,
        "forecast_table": None,

        "arch_lm_stat"     : round(lm_stat,   4),
        "arch_lm_pvalue"   : round(lm_pvalue, 4),
        "arch_significant" : arch_significant,
    }