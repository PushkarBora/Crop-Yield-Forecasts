"""
AR-TGARCH / ARX-TGARCH — True Zakoian (1994) Threshold GARCH
=============================================================
Key distinction from AR-GJR-GARCH:
  GJR-GARCH  : models conditional VARIANCE  →  σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε<0) + β·σ²_{t-1}
  T-GARCH    : models conditional STD DEV   →  σ_t  = ω + α⁺·max(ε_{t-1},0) + α⁻·max(-ε_{t-1},0) + β·σ_{t-1}

Because no Python library (arch, statsmodels) implements the true Zakoian T-GARCH,
this module provides a custom implementation via:
  • Two-step estimation: OLS for AR mean, scipy MLE for T-GARCH variance
  • Rolling 1-step-ahead forecast for test evaluation
  • Multi-step σ convergence formula for future intervals

Reference: Zakoian, J.-M. (1994). Threshold heteroskedastic models.
           Journal of Economic Dynamics and Control, 18(5), 931–955.
"""

import numpy as np
import pandas as pd
import os
import pickle
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from statsmodels.stats.diagnostic import het_arch
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
# STEP 1 — FIT AR MEAN EQUATION VIA OLS
# ============================================================
def _fit_ar_ols(series_values, ar_lags, exog=None):
    """
    Fit AR(ar_lags) or ARX(ar_lags) mean equation via ordinary least squares.

    Design matrix (n_valid × k):
      [1, y_{t-1}, y_{t-2}, ..., y_{t-ar_lags}, exog_{t}, ...]

    Returns dict with:
      params     : OLS coefficients  [const, φ₁,...,φₖ, exog params...]
      fitted     : in-sample fitted values (length n - ar_lags)
      residuals  : OLS residuals      (length n - ar_lags)
      y_valid    : target y values    (length n - ar_lags)
      X          : design matrix
      se         : OLS standard errors
      t_stats    : t-statistics
      p_values   : p-values
    """
    y_full = np.array(series_values, dtype=float)
    n_full = len(y_full)

    # Valid targets: skip first ar_lags obs (consumed as lags)
    y = y_full[ar_lags:]
    n_valid = len(y)

    # Build lag columns
    lag_cols = [y_full[ar_lags - i - 1: n_full - i - 1] for i in range(ar_lags)]
    X = np.column_stack(lag_cols) if lag_cols else np.empty((n_valid, 0))

    # Append exogenous (already aligned: same length as series_values)
    if exog is not None:
        exog_arr = np.array(exog, dtype=float)
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        exog_slice = exog_arr[ar_lags:]    # align: drop first ar_lags rows
        X = np.column_stack([X, exog_slice]) if X.shape[1] > 0 else exog_slice

    # Prepend constant column
    ones = np.ones((n_valid, 1))
    X = np.column_stack([ones, X]) if X.shape[1] > 0 else ones

    # OLS solution
    try:
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        params = np.zeros(X.shape[1])

    fitted    = X @ params
    residuals = y - fitted

    # OLS standard errors (for inference)
    k       = X.shape[1]
    df_res  = max(n_valid - k, 1)
    rss     = float(np.sum(residuals ** 2))
    s2      = rss / df_res
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se      = np.sqrt(s2 * np.maximum(np.diag(XtX_inv), 0.0))
    except Exception:
        se = np.full(k, np.nan)

    t_stats = params / np.where(se > 0, se, np.nan)
    p_vals  = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=df_res))

    return {
        "params"   : params,
        "fitted"   : fitted,
        "residuals": residuals,
        "y_valid"  : y,
        "X"        : X,
        "n_valid"  : n_valid,
        "se"       : se,
        "t_stats"  : t_stats,
        "p_values" : p_vals,
        "df_res"   : df_res,
    }


# ============================================================
# STEP 2 — ZAKOIAN T-GARCH VARIANCE RECURSION
# ============================================================
def _tgarch_sigma_recursion(var_params, residuals, p, q):
    """
    Compute the T-GARCH conditional std-dev sequence σ_t.

    Zakoian (1994) recursion (TGARCH(p,q)):
      σ_t = ω  +  Σᵢ₌₁ᵖ [ α⁺ᵢ · max(ε_{t-i}, 0) + α⁻ᵢ · max(-ε_{t-i}, 0) ]
               +  Σⱼ₌₁q   βⱼ · σ_{t-j}

    Parameter vector layout:
      [ω, α⁺₁,...,α⁺ₚ, α⁻₁,...,α⁻ₚ, β₁,...,βq]

    Returns σ array (same length as residuals).
    """
    n        = len(residuals)
    max_lag  = max(p, q)

    omega     = var_params[0]
    alpha_pos = var_params[1       : 1 + p]
    alpha_neg = var_params[1 + p   : 1 + 2 * p]
    beta      = var_params[1 + 2*p : 1 + 2*p + q]

    sigma = np.zeros(n)
    # Back-cast initialisation: unconditional std dev
    sigma[:max_lag] = max(np.std(residuals), 1e-8)

    for t in range(max_lag, n):
        s = omega
        for i in range(p):
            eps_i = residuals[t - 1 - i]
            s += alpha_pos[i] * max(eps_i,  0.0)
            s += alpha_neg[i] * max(-eps_i, 0.0)
        for j in range(q):
            s += beta[j] * sigma[t - 1 - j]
        sigma[t] = max(s, 1e-8)          # ensure positivity

    return sigma


def _tgarch_one_step_ahead(var_params, p, q, resid_hist, sigma_hist):
    """
    Compute the one-step-ahead σ forecast given histories.
    Used in both rolling test evaluation and multi-step forecasting.
    """
    omega     = var_params[0]
    alpha_pos = var_params[1       : 1 + p]
    alpha_neg = var_params[1 + p   : 1 + 2 * p]
    beta      = var_params[1 + 2*p : 1 + 2*p + q]

    s = omega
    for i in range(p):
        if len(resid_hist) > i:
            eps_i = float(resid_hist[-1 - i])
            s += alpha_pos[i] * max(eps_i,  0.0)
            s += alpha_neg[i] * max(-eps_i, 0.0)
    for j in range(q):
        if len(sigma_hist) > j:
            s += beta[j] * float(sigma_hist[-1 - j])
    return max(s, 1e-8)


# ============================================================
# STEP 2b — NEGATIVE LOG-LIKELIHOOD FOR MLE
# ============================================================
def _tgarch_nll(var_params, residuals, p, q):
    """
    Negative Gaussian log-likelihood for T-GARCH variance equation.

    ε_t | F_{t-1} ~ N(0, σ²_t),  σ_t from T-GARCH recursion.
    LL = -Σ [ log(2π)/2 + log(σ_t) + ε²_t / (2σ²_t) ]
    """
    # Hard parameter constraints
    if var_params[0] <= 0 or np.any(var_params < 0):
        return 1e10

    alpha_pos = var_params[1       : 1 + p]
    alpha_neg = var_params[1 + p   : 1 + 2 * p]
    beta      = var_params[1 + 2*p : 1 + 2*p + q]

    # Stationarity: E[(α⁺+α⁻)/2 per lag] + sum(β) < 1
    stationarity = np.sum((alpha_pos + alpha_neg) / 2.0) + np.sum(beta)
    if stationarity >= 0.9999:
        return 1e10

    sigma   = _tgarch_sigma_recursion(var_params, residuals, p, q)
    max_lag = max(p, q)
    s = sigma[max_lag:]
    e = residuals[max_lag:]

    if np.any(s <= 0) or not np.all(np.isfinite(s)):
        return 1e10

    ll = -0.5 * np.sum(np.log(2 * np.pi) + 2 * np.log(s) + (e / s) ** 2)
    return -ll if np.isfinite(ll) else 1e10


# ============================================================
# STEP 2c — FIT T-GARCH VARIANCE BY MLE
# ============================================================
def _fit_tgarch_mle(residuals, p, q, n_restarts=5):
    """
    Estimate T-GARCH(p,q) parameters via maximum likelihood.

    Parameter vector: [ω, α⁺₁,...,α⁺ₚ, α⁻₁,...,α⁻ₚ, β₁,...,βq]
    Bounds: ω > 0,  all α ≥ 0,  all β ∈ [0, 0.99)
    Constraint: stationarity  Σ(α⁺+α⁻)/2 + Σβ < 1

    Returns: (var_params, sigma, loglik, aic, bic, hqic, converged)
    """
    n_vp       = 1 + 2 * p + q
    uncond_std = max(float(np.std(residuals)), 1e-8)
    n_obs      = len(residuals) - max(p, q)

    bounds = (
        [(1e-8, uncond_std * 2.0)] +   # ω
        [(0.0, 0.95)] * (2 * p)    +   # α⁺ᵢ, α⁻ᵢ
        [(0.0, 0.95)] * q              # βⱼ
    )

    best_result = None
    best_nll    = np.inf

    for trial in range(n_restarts):
        rng = np.random.default_rng(SEED + trial * 31)

        omega_0    = uncond_std * rng.uniform(0.02, 0.15)
        alpha_p_0  = rng.uniform(0.04, 0.18, size=p)
        alpha_n_0  = rng.uniform(0.07, 0.22, size=p)   # leverage: neg > pos
        beta_0     = rng.uniform(0.35, 0.65, size=q)

        x0 = np.concatenate([[omega_0], alpha_p_0, alpha_n_0, beta_0])

        try:
            res = minimize(
                _tgarch_nll,
                x0,
                args=(residuals, p, q),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-9},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll    = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or not np.isfinite(best_nll):
        # Fallback: simple plausible starting values
        var_params = np.zeros(n_vp)
        var_params[0]            = uncond_std * 0.05    # ω
        var_params[1 : 1+p]      = 0.08                 # α⁺
        var_params[1+p : 1+2*p]  = 0.14                 # α⁻ (leverage)
        var_params[1+2*p :]      = 0.50                 # β
        loglik    = np.nan
        converged = False
    else:
        var_params = best_result.x
        loglik     = -best_nll
        converged  = best_result.success

    sigma = _tgarch_sigma_recursion(var_params, residuals, p, q)

    # ── Numerical Hessian → standard errors ───────────────────────
    var_se = np.full(n_vp, np.nan)
    if best_result is not None and np.isfinite(best_nll):
        try:
            from scipy.optimize import approx_fprime
            eps   = 1e-5 * np.abs(var_params) + 1e-8
            hess  = np.zeros((n_vp, n_vp))
            f0    = _tgarch_nll(var_params, residuals, p, q)
            for i in range(n_vp):
                for j in range(i, n_vp):
                    ei = np.zeros(n_vp); ei[i] = eps[i]
                    ej = np.zeros(n_vp); ej[j] = eps[j]
                    fij  = _tgarch_nll(var_params + ei + ej, residuals, p, q)
                    fi   = _tgarch_nll(var_params + ei,      residuals, p, q)
                    fj   = _tgarch_nll(var_params + ej,      residuals, p, q)
                    hess[i, j] = (fij - fi - fj + f0) / (eps[i] * eps[j])
                    hess[j, i] = hess[i, j]
            cov = np.linalg.inv(hess)
            diag = np.diag(cov)
            var_se = np.where(diag > 0, np.sqrt(diag), np.nan)
        except Exception:
            var_se = np.full(n_vp, np.nan)

    # Information criteria (variance model only; AR params accounted for separately)
    aic  = (2 * n_vp - 2 * loglik)                if np.isfinite(loglik) else np.inf
    bic  = (np.log(n_obs) * n_vp - 2 * loglik)    if np.isfinite(loglik) else np.inf
    hqic = (2 * np.log(np.log(n_obs)) * n_vp
             - 2 * loglik)                          if (np.isfinite(loglik) and n_obs > 1) else np.inf

    return var_params, sigma, loglik, aic, bic, hqic, converged, var_se


# ============================================================
# COMBINED PARAMETER ESTIMATES TABLE
# ============================================================
def _build_param_table(ar_fit, var_params, p, q, ar_lags,
                       has_exogenous, exog_cols, exog_lags, var_se=None):
    """
    Build a unified parameter estimates DataFrame for display.
    Mean eq: OLS estimates with standard errors and p-values.
    Var  eq: MLE point estimates (std errors via numerical Hessian if feasible).
    """
    rows = []

    # ── Mean equation ──────────────────────────────────────────────
    mean_labels = ["Const (μ)"] + [f"AR({i+1}) (φ{i+1})" for i in range(ar_lags)]
    if has_exogenous:
        for col in exog_cols:
            for lag in range(1, exog_lags + 1):
                mean_labels.append(f"{col}_lag{lag}")

    params = ar_fit["params"]
    se     = ar_fit["se"]
    t_stat = ar_fit["t_stats"]
    pv     = ar_fit["p_values"]

    for i, lbl in enumerate(mean_labels):
        if i < len(params):
            coef_i = float(params[i])
            se_i   = float(se[i])
            t_i    = float(t_stat[i])
            pv_i   = float(pv[i])
            ci_lo  = coef_i - 1.96 * se_i if np.isfinite(se_i) else np.nan
            ci_hi  = coef_i + 1.96 * se_i if np.isfinite(se_i) else np.nan
            rows.append({
                "Parameter" : lbl,
                "Coef"      : round(coef_i, 4),
                "Std Err"   : round(se_i,   4) if np.isfinite(se_i) else "—",
                "z"         : round(t_i,    4) if np.isfinite(t_i)  else "—",
                "P>|z|"     : round(pv_i,   4) if np.isfinite(pv_i) else "—",
                "[0.025"    : round(ci_lo,  4) if np.isfinite(ci_lo) else "—",
                "0.975]"    : round(ci_hi,  4) if np.isfinite(ci_hi) else "—",
            })

    # ── Variance equation ──────────────────────────────────────────
    var_labels = ["ω (const)"]
    for i in range(p):
        var_labels.append(f"α⁺({i+1}) (pos shock)")
    for i in range(p):
        var_labels.append(f"α⁻({i+1}) (neg shock, leverage)")
    for j in range(q):
        var_labels.append(f"β({j+1}) (GARCH)")

    for i, lbl in enumerate(var_labels):
        if i < len(var_params):
            coef = float(var_params[i])
            se_i = float(var_se[i]) if (var_se is not None and i < len(var_se) and np.isfinite(var_se[i])) else np.nan
            z_i  = coef / se_i if (np.isfinite(se_i) and se_i > 0) else np.nan
            pv_i = float(2 * (1 - t_dist.cdf(abs(z_i), df=max(ar_fit["n_valid"] - len(var_params), 1)))) if np.isfinite(z_i) else np.nan
            ci_lo = coef - 1.96 * se_i if np.isfinite(se_i) else np.nan
            ci_hi = coef + 1.96 * se_i if np.isfinite(se_i) else np.nan

            rows.append({
                "Parameter" : lbl,
                "Coef"      : round(coef, 4),
                "Std Err"   : round(se_i,  4) if np.isfinite(se_i)  else "—",
                "z"         : round(z_i,   4) if np.isfinite(z_i)   else "—",
                "P>|z|"     : round(pv_i,  4) if np.isfinite(pv_i)  else "—",
                "[0.025"    : round(ci_lo, 4) if np.isfinite(ci_lo) else "—",
                "0.975]"    : round(ci_hi, 4) if np.isfinite(ci_hi) else "—",
            })

    return pd.DataFrame(rows)


# ============================================================
# INFORMATION CRITERIA (COMBINED)
# ============================================================
def _combined_info_criteria(ar_fit, loglik_var, p, q):
    """
    Compute combined AIC/BIC/HQIC for the full AR + T-GARCH model.
    Total log-likelihood = OLS Gaussian LL (mean) + TGARCH MLE LL (variance).
    """
    n       = ar_fit["n_valid"]
    rss     = float(np.sum(ar_fit["residuals"] ** 2))
    s2_ols  = rss / n
    # Gaussian LL for AR mean equation
    ll_mean = -n / 2 * (np.log(2 * np.pi) + np.log(max(s2_ols, 1e-15)) + 1)

    if np.isfinite(loglik_var):
        ll_total = ll_mean + loglik_var
    else:
        return {}

    k_ar    = len(ar_fit["params"])
    k_var   = 1 + 2 * p + q
    k_total = k_ar + k_var

    criteria = {"Log-Likelihood": round(float(ll_total), 4)}
    criteria["AIC"]  = round(2 * k_total - 2 * ll_total, 4)
    if n > k_total + 1:
        criteria["AICc"] = round(
            criteria["AIC"] + 2 * k_total * (k_total + 1) / (n - k_total - 1), 4
        )
    criteria["BIC"]  = round(np.log(n) * k_total - 2 * ll_total, 4)
    if n > 1:
        criteria["HQIC"] = round(2 * np.log(np.log(n)) * k_total - 2 * ll_total, 4)
    return criteria


# ============================================================
# AUTO-TUNE — grid search over (ar_lags, p, q)
# ============================================================
def auto_tune_artgarch(train_series, exog_train, has_exogenous,
                       params, log_callback=None):
    """
    Grid search:
      ar_lags ∈ [1, 3]   — AR mean order
      p       ∈ [1, 2]   — T-GARCH ARCH order
      q       ∈ [1, 2]   — T-GARCH GARCH order

    Selects best (ar_lags, p, q) by combined AIC.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    prefix = "ARX" if has_exogenous else "AR"
    log(f"\n🔍 Auto T-GARCH grid search — {prefix}(ar) × TGARCH(p,q) ...")

    best_aic   = np.inf
    best_combo = None

    for ar_try in range(1, 4):
        ar_fit = _fit_ar_ols(train_series.values, ar_try, exog_train)
        resid  = ar_fit["residuals"]

        for p_try in range(1, 3):
            for q_try in range(1, 3):
                try:
                    _, _, loglik_v, _, _, _, ok, _ = _fit_tgarch_mle(
                        resid, p_try, q_try, n_restarts=2
                    )
                    crit = _combined_info_criteria(ar_fit, loglik_v, p_try, q_try)
                    total_aic = crit.get("AIC", np.inf)

                    tag = f"{prefix}({ar_try})-TGARCH({p_try},{q_try})"
                    log(f"   {tag}  AIC={total_aic:.2f}"
                        + ("" if ok else " [⚠ not converged]"))

                    if total_aic < best_aic:
                        best_aic   = total_aic
                        best_combo = {
                            "ar_lags" : ar_try,
                            "tgarch_p": p_try,
                            "tgarch_o": p_try,   # T-GARCH: o always = p
                            "tgarch_q": q_try,
                        }
                        log(f"   ✨ New best: {tag}  AIC={best_aic:.2f}")

                except Exception:
                    continue

    return best_combo


# ============================================================
# ARCH LM PRE-TEST (identical to other volatility models)
# ============================================================
def _arch_lm_test(series, lags=5, log=None):
    import pmdarima as pm

    y = pd.Series(series.values.astype(float)).reset_index(drop=True)

    try:
        arima_fit = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            information_criterion="aic",
            test="kpss",
            max_p=5, max_q=5, max_d=2,
            start_p=0, start_q=0,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order    = arima_fit.order
        p, d, q  = order
        # Drop first residual (Kalman init artifact) — same as other models
        residuals = pd.Series(arima_fit.resid()).iloc[1:].reset_index(drop=True)
    except Exception as e:
        if log:
            log(f"\n⚠️  Auto-ARIMA failed: {e}")
            log("   Cannot perform ARCH LM test — AR-TGARCH will not be fitted.")
        return 0.0, 1.0, False

    lm_stat, p_value, f_stat, f_pval = het_arch(residuals, nlags=lags)

    if log:
        log("\n" + "=" * 60)
        log("📋 ARCH LM Test (Engle's Test for ARCH Effects)")
        drift_str = "with drift" if d == 1 else ("with mean" if d == 0 else "no constant")
        log(f"   Mean model   : Auto-ARIMA{order} {drift_str} residuals")
        log(f"   ARCH lags    : {lags}")
        log(f"   H0 : No ARCH effects  |  H1 : ARCH effects present")
        log(f"   LM Statistic : {lm_stat:.4f}")
        log(f"   p-value      : {p_value:.4f}")
        log(f"   F-Statistic  : {f_stat:.4f}")
        log(f"   F p-value    : {f_pval:.4f}")
        if p_value < 0.05:
            log(f"   ✅ ARCH effects detected (p={p_value:.4f} < 0.05) → Proceed with AR-TGARCH")
        else:
            log(f"   ❌ No ARCH effects (p={p_value:.4f} ≥ 0.05) → AR-TGARCH NOT appropriate")
        log("=" * 60)

    return lm_stat, p_value, p_value < 0.05


# ============================================================
# FORECAST-ONLY MODE
# ============================================================
def _forecast_only_artgarch(params, horizon):
    """
    Load saved bundle and produce h-step-ahead forecasts.

    Mean forecast : AR recursion using last observed y values.
    σ forecast    : T-GARCH 1-step, then long-run convergence for h > 1.
    Intervals     : ŷ ± z·σ  (95% and 80%).
    """
    for path in ("static/artgarch_bundle.pkl", "static/artgarch_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Please train the AR-TGARCH model first."
            )

    with open("static/artgarch_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    ar_params      = bundle["ar_params"]       # [const, φ₁,...,φₖ, exog_p...]
    var_params     = bundle["var_params"]       # [ω, α⁺..., α⁻..., β...]
    ar_lags        = bundle["ar_lags"]
    tgarch_p       = bundle["tgarch_p"]
    tgarch_q       = bundle["tgarch_q"]
    diff_order     = bundle.get("diff_order", 0)
    last_val       = bundle.get("last_val", None)
    has_exogenous  = bundle.get("has_exogenous", False)
    exog_cols      = bundle.get("exog_cols", [])
    exog_lags      = bundle.get("exog_lags", 0)
    n_exog_feats   = bundle.get("n_exog_feats", 0)
    x_scaler_mean  = bundle.get("x_scaler_mean", None)
    x_scaler_scale = bundle.get("x_scaler_scale", None)
    last_y         = bundle.get("last_y", [])        # last ar_lags observed y values
    last_resid     = bundle.get("last_resid", [])    # last max(p,q) residuals
    last_sigma     = bundle.get("last_sigma", [])    # last max(p,q) σ values
    time_labels    = bundle.get("time_labels", [])

    future_labels  = time_labels[:horizon] if time_labels else list(range(1, horizon + 1))

    # ── Build future exog if multivariate ────────────────────────────
    future_exog_scaled = None
    if has_exogenous:
        future_exog_dict = params.get("future_exog")
        if not future_exog_dict:
            raise ValueError("Provide future exog values via params['future_exog'].")
        missing = [c for c in exog_cols if c not in future_exog_dict]
        if missing:
            raise ValueError(f"Missing future exog for: {missing}")
        raw = np.column_stack([
            np.array(future_exog_dict[c][:horizon], dtype=float)
            for c in exog_cols
        ])
        tiled = np.hstack([raw] * exog_lags)[:, :n_exog_feats]
        future_exog_scaled = (tiled - x_scaler_mean) / x_scaler_scale

    # ── AR mean forecasts ─────────────────────────────────────────────
    const       = ar_params[0]
    ar_phi      = ar_params[1 : 1 + ar_lags]
    exog_coefs  = ar_params[1 + ar_lags:] if has_exogenous else np.array([])

    y_hist = list(last_y)     # circular buffer of recent y values
    mean_fc = []

    for h in range(horizon):
        mu = const
        for j in range(ar_lags):
            mu += ar_phi[j] * (y_hist[-1 - j] if len(y_hist) > j else 0.0)
        if has_exogenous and future_exog_scaled is not None and len(exog_coefs) > 0:
            exog_row = future_exog_scaled[h] if h < len(future_exog_scaled) else future_exog_scaled[-1]
            mu += float(np.dot(exog_coefs, exog_row[:len(exog_coefs)]))
        mean_fc.append(mu)
        y_hist.append(mu)     # use forecast as future lagged value

    mean_fc = np.array(mean_fc)

    if diff_order == 1 and last_val is not None:
        mean_fc = last_val + np.cumsum(mean_fc)

    # ── T-GARCH σ forecasts ───────────────────────────────────────────
    resid_hist = list(last_resid)
    sigma_hist = list(last_sigma)
    alpha_pos  = var_params[1          : 1 + tgarch_p]
    alpha_neg  = var_params[1+tgarch_p : 1 + 2*tgarch_p]
    beta       = var_params[1+2*tgarch_p : 1 + 2*tgarch_p + tgarch_q]
    persistence = np.sum((alpha_pos + alpha_neg) / 2.0) + np.sum(beta)

    sigma_fc = []
    for h in range(horizon):
        if h == 0:
            s = _tgarch_one_step_ahead(var_params, tgarch_p, tgarch_q,
                                       resid_hist, sigma_hist)
        else:
            # Long-run convergence formula for h > 1
            # E[σ_{t+h}] = σ_∞ + persistence^h * (σ_{t+1} - σ_∞)
            omega    = var_params[0]
            s_inf    = omega / max(1.0 - persistence, 1e-8)
            s        = s_inf + persistence ** h * (sigma_fc[0] - s_inf)
            s        = max(s, 1e-8)
        sigma_fc.append(s)
        # Propagate: expected residual = 0, so E[max(ε,0)] = σ * sqrt(1/(2π))
        resid_hist.append(0.0)
        sigma_hist.append(s)

    sigma_fc = np.array(sigma_fc)

    lower_95 = mean_fc - 1.960 * sigma_fc
    upper_95 = mean_fc + 1.960 * sigma_fc
    lower_80 = mean_fc - 1.282 * sigma_fc
    upper_80 = mean_fc + 1.282 * sigma_fc

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
def run_artgarch(
    data        : pd.DataFrame,
    params      : dict,
    horizon     : int,
    frequency   : str,
    mode        : str = "train",
    log_callback      = None,
):
    """
    AR-TGARCH / ARX-TGARCH — True Zakoian (1994) model.

    Variance equation models conditional STD DEV (not variance):
      σ_t = ω + α⁺·max(ε_{t-1},0) + α⁻·max(-ε_{t-1},0) + β·σ_{t-1}

    This is fundamentally different from GJR-GARCH which models σ²_t.
    Implemented from scratch via OLS (mean) + scipy MLE (variance).

    Same API as all other DATAI volatility models.
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # ── FORECAST MODE ─────────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_artgarch(params=params, horizon=horizon)

    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Column mappings ───────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")
    exog_cols  = params.get("exog_cols",  []) or []

    has_exogenous = len(exog_cols) > 0
    model_label   = "ARX-TGARCH" if has_exogenous else "AR-TGARCH"

    log(f"📊 {model_label} — True Zakoian (1994) Threshold GARCH")
    log(f"   Variance: σ_t = ω + α⁺·ε⁺_{{t-1}} + α⁻·ε⁻_{{t-1}} + β·σ_{{t-1}}  (std dev, NOT variance)")
    log(f"   Note: different from GJR-GARCH which models σ²_t")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")
    log(f"   Exog variables : {exog_cols if exog_cols else 'None'}")

    # ── Hyperparameters ───────────────────────────────────────────────
    split      = params.get("split",        0.85)
    use_auto   = params.get("auto_artgarch", True)
    diff_order = params.get("diff_order",    0)

    ar_lags   = params.get("ar_lags",   1)
    tgarch_p  = params.get("tgarch_p",  1)
    tgarch_o  = params.get("tgarch_o",  1)    # display only; T-GARCH has o=p inherently
    tgarch_q  = params.get("tgarch_q",  1)
    exog_lags = params.get("exog_lags", 1)
    arch_lags = params.get("arch_lags", 5)

    log("=" * 60)
    log(f"   auto_artgarch : {use_auto}")
    log(f"   diff_order    : {diff_order}")
    log(f"   Manual: ar_lags={ar_lags}, p={tgarch_p}, q={tgarch_q}"
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

    # ── ARCH LM Pre-Test (on raw target series) ───────────────────────
    raw_target = df_full[target_col].copy().reset_index(drop=True)
    lm_stat, lm_pvalue, arch_significant = _arch_lm_test(
        raw_target, lags=arch_lags, log=log
    )

    if not arch_significant:
        log("\n⚠️  ARCH LM test not significant — AR-TGARCH is not appropriate.")
        return {
            "model_key"        : "artgarch",
            "model_name"       : model_label,
            "data_type"        : "Univariate" if not has_exogenous else "Multivariate",
            "arch_lm_stat"     : round(lm_stat,   4),
            "arch_lm_pvalue"   : round(lm_pvalue, 4),
            "arch_significant" : False,
            "arch_not_significant_message": (
                "Auto-ARIMA could not be fitted to compute ARCH LM test residuals. "
                "AR-TGARCH cannot be applied without a valid pre-test."
            ) if lm_stat == 0.0 and lm_pvalue == 1.0 else (
                f"The ARCH LM test is NOT significant "
                f"(LM Stat = {lm_stat:.4f}, p-value = {lm_pvalue:.4f} ≥ 0.05). "
                f"No ARCH effects detected — {model_label} is not appropriate. "
                f"Consider using ARIMA instead."
            ),
            "train_table"   : None, "test_table"    : None,
            "forecast_table": None,
            "rmse_train": None, "mae_train": None, "mape_train": None,
            "rmse_test" : None, "mae_test" : None, "mape_test" : None,
        }

    # ── AUTO-TUNE ─────────────────────────────────────────────────────
    if use_auto:
        best = auto_tune_artgarch(
            train_series, exog_train, has_exogenous,
            params, log_callback=log,
        )
        if best:
            ar_lags  = best["ar_lags"]
            tgarch_p = best["tgarch_p"]
            tgarch_o = best.get("tgarch_o", tgarch_p)
            tgarch_q = best["tgarch_q"]
            px       = "ARX" if has_exogenous else "AR"
            log(f"\n✅ Best: {px}({ar_lags})-TGARCH({tgarch_p},{tgarch_q})")
            params.update(best)

    # ── Step 1: Fit AR mean on TRAIN ──────────────────────────────────
    log(f"\n🔧 Fitting AR({ar_lags}) mean equation (OLS)...")
    ar_fit_train = _fit_ar_ols(train_series.values, ar_lags, exog_train)

    train_residuals = ar_fit_train["residuals"]
    train_fitted    = ar_fit_train["fitted"]
    valid_start     = ar_lags        # first ar_lags rows consumed as lags

    # ── Step 2: Fit T-GARCH variance on train residuals ───────────────
    log(f"🔧 Fitting T-GARCH({tgarch_p},{tgarch_q}) variance (MLE)...")
    var_params, sigma_train, loglik_var, aic_v, bic_v, hqic_v, converged, var_se = (
        _fit_tgarch_mle(train_residuals, tgarch_p, tgarch_q, n_restarts=5)
    )
    log(f"   Converged: {converged}  |  Var AIC: {aic_v:.2f}")

    # Extract AR params for rolling forecast
    ar_params_arr  = ar_fit_train["params"]
    ar_const       = ar_params_arr[0]
    ar_phi         = ar_params_arr[1 : 1 + ar_lags]
    exog_coefs_tr  = ar_params_arr[1 + ar_lags:] if has_exogenous else np.array([])

    # ── Rolling 1-step-ahead forecast on TEST ─────────────────────────
    log("\n🔮 Rolling 1-step-ahead forecast on test set...")
    max_vlag = max(tgarch_p, tgarch_q)

    resid_hist = list(train_residuals)       # grows as actual values arrive
    sigma_hist = list(sigma_train)

    test_preds  = []
    test_sigmas = []

    for i in range(len(test_series)):
        # AR mean forecast using fixed training AR params
        y_hist_arr = np.concatenate([
            train_series.values,
            test_series.values[:i]
        ])
        mu = ar_const
        for j in range(ar_lags):
            if len(y_hist_arr) > j:
                mu += ar_phi[j] * float(y_hist_arr[-1 - j])
        if has_exogenous and exog_test is not None and len(exog_coefs_tr) > 0:
            exog_row = exog_test[i]
            k_exog   = min(len(exog_coefs_tr), len(exog_row))
            mu += float(np.dot(exog_coefs_tr[:k_exog], exog_row[:k_exog]))

        # T-GARCH σ forecast
        sigma_next = _tgarch_one_step_ahead(
            var_params, tgarch_p, tgarch_q, resid_hist, sigma_hist
        )

        test_preds.append(mu)
        test_sigmas.append(sigma_next)

        # Update history with actual values
        actual_resid = float(test_series.iloc[i]) - mu
        resid_hist.append(actual_resid)
        sigma_hist.append(sigma_next)

    pred_te   = np.array(test_preds)
    actual_te = test_series.values

    # ── Undo differencing ─────────────────────────────────────────────
    if diff_order == 1:
        offset     = exog_lags if has_exogenous else 0
        base_idx   = split_idx + offset
        base_level = series.iloc[base_idx] if base_idx < len(series) else series.iloc[-1]

        actual_te_ = series.values[base_idx + 1 : base_idx + 1 + len(pred_te)]
        pred_te_   = base_level + np.cumsum(pred_te)

        train_act_  = series.values[(1 + offset) : split_idx + 1 + offset][valid_start:]
        train_pred_ = train_act_ - ar_fit_train["residuals"][valid_start:]
    else:
        actual_te_  = actual_te
        pred_te_    = pred_te
        train_act_  = train_series.values[valid_start:]
        train_pred_ = train_fitted[valid_start:]

    min_tr = min(len(train_act_), len(train_pred_))
    min_te = min(len(actual_te_), len(pred_te_))
    train_act_  = train_act_[:min_tr]
    train_pred_ = train_pred_[:min_tr]
    actual_te_  = actual_te_[:min_te]
    pred_te_    = pred_te_[:min_te]
    train_t     = train_times.values[valid_start : valid_start + min_tr]

    # ── NaN guard ─────────────────────────────────────────────────────
    def _clean_pair(a, p_):
        a  = np.array(a, dtype=float)
        p_ = np.array(p_, dtype=float)
        m  = np.isfinite(a) & np.isfinite(p_)
        if m.sum() == 0:
            raise ValueError("All predictions are NaN/Inf — model failed to converge.")
        return a[m], p_[m]

    train_act_c, train_pred_c = _clean_pair(train_act_,  train_pred_)
    actual_te_c, pred_te_c   = _clean_pair(actual_te_,   pred_te_)

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

    # ── Parameter estimates & info criteria ───────────────────────────
    param_estimates_table = _build_param_table(
        ar_fit_train, var_params,
        tgarch_p, tgarch_q,
        ar_lags, has_exogenous, exog_cols, exog_lags,
        var_se=var_se
    )
    info_criteria = _combined_info_criteria(ar_fit_train, loglik_var, tgarch_p, tgarch_q)

    # ── Future time labels ────────────────────────────────────────────
    last_t = df_full[time_col].iloc[-1]
    try:
        future_time_labels = [int(last_t) + i for i in range(1, horizon + 1)]
    except (TypeError, ValueError):
        orig_seq = data[time_col].tolist()
        first_v  = orig_seq[0]
        try:
            cyc_len = orig_seq[1:].index(first_v) + 1
        except ValueError:
            cyc_len = len(orig_seq)
        cyc      = orig_seq[:cyc_len]
        last_pos = cyc.index(last_t)
        future_time_labels = [
            cyc[(last_pos + i) % cyc_len]
            for i in range(1, horizon + 1)
        ]

    # ── Refit on FULL data ────────────────────────────────────────────
    log("\n🔧 Refitting on full dataset for future forecasting...")
    ar_fit_full = _fit_ar_ols(series_aligned.values, ar_lags, exog_arr_all)
    var_params_full, sigma_full, loglik_full, *_ = _fit_tgarch_mle(
        ar_fit_full["residuals"], tgarch_p, tgarch_q, n_restarts=5
    )

    # Save last window for forecast initialisation
    last_ar_lags = ar_lags
    save_window  = max(last_ar_lags, max(tgarch_p, tgarch_q))
    last_y_vals  = list(series_aligned.values[-save_window:])
    last_resid   = list(ar_fit_full["residuals"][-save_window:])
    last_sigma   = list(sigma_full[-save_window:])

    # ── Save pickles ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)
    n_exog_feats = int(exog_arr_all.shape[1]) if exog_arr_all is not None else 0

    prefix    = "ARX-TGARCH" if has_exogenous else "AR-TGARCH"
    title_str = (
        f"{prefix}({ar_lags},{tgarch_p},{tgarch_q})"
        + (" [Multivariate]" if has_exogenous else " [Univariate]")
    )

    with open("static/artgarch_model.pkl", "wb") as f:
        pickle.dump({
            "ar_fit"    : ar_fit_full,
            "var_params": var_params_full,
            "sigma"     : sigma_full,
        }, f)

    with open("static/artgarch_bundle.pkl", "wb") as f:
        pickle.dump({
            "ar_params"      : ar_fit_full["params"],
            "var_params"     : var_params_full,
            "ar_lags"        : ar_lags,
            "tgarch_p"       : tgarch_p,
            "tgarch_o"       : tgarch_o,
            "tgarch_q"       : tgarch_q,
            "diff_order"     : diff_order,
            "last_val"       : last_val_before_diff,
            "last_time"      : last_t,
            "time_labels"    : future_time_labels,
            "time_col"       : time_col,
            "target_col"     : target_col,
            "exog_cols"      : exog_cols,
            "exog_lags"      : exog_lags,
            "has_exogenous"  : has_exogenous,
            "n_exog_feats"   : n_exog_feats,
            "x_scaler_mean"  : exog_scaler.mean_  if exog_scaler else None,
            "x_scaler_scale" : exog_scaler.scale_ if exog_scaler else None,
            "last_y"         : last_y_vals,
            "last_resid"     : last_resid,
            "last_sigma"     : last_sigma,
            "params"         : params,
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
    ax.set_title(f"{title_str} — Zakoian (1994) T-GARCH — Training & Testing Set")
    ax.set_xlabel(time_col)
    ax.set_ylabel(target_col)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/artgarch_train.png", dpi=120)
    plt.savefig("static/artgarch_test.png",  dpi=120)
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
        "model_key"   : "artgarch",
        "model_name"  : title_str,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Multivariate" if has_exogenous else "Univariate",
        "model_config": {
            "Mean Model"         : f"{'ARX' if has_exogenous else 'AR'}({ar_lags})",
            "AR Lags"            : ar_lags,
            "Variance Model"     : f"T-GARCH({tgarch_p},{tgarch_q}) — Zakoian (1994)",
            "Variance Recursion" : "σ_t (std dev) — NOT σ²_t like GJR-GARCH",
            "p (ARCH Order)"     : tgarch_p,
            "q (GARCH Order)"    : tgarch_q,
            "Differencing"       : diff_order,
            "Exog Cols"          : exog_cols if exog_cols else "None",
            "Exog Lags"          : exog_lags if has_exogenous else "N/A",
            "Estimation"         : "OLS (mean) + MLE (variance, custom scipy)",
            "Reference"          : "Zakoian (1994), J. Econ. Dyn. Control",
        },
        "auto_tuned"   : use_auto,
        "order"        : title_str,
        "auto_artgarch": use_auto,

        "param_estimates_table": param_estimates_table,
        "info_criteria"        : info_criteria,

        "rmse_train": round(rmse_tr, 3),
        "mae_train" : round(mae_tr,  3),
        "mape_train": round(mape_tr, 2),
        "rmse_test" : round(rmse_te, 3),
        "mae_test"  : round(mae_te,  3),
        "mape_test" : round(mape_te, 2),

        "residuals": (train_act_c - train_pred_c).tolist(),

        "train_table"   : train_table,
        "test_table"    : test_table,
        "forecast_table": None,

        "arch_lm_stat"     : round(lm_stat,   4),
        "arch_lm_pvalue"   : round(lm_pvalue, 4),
        "arch_significant" : arch_significant,
    }