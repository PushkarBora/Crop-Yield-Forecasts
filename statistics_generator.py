"""
Statistics Generator for Time Series Data
Fully generic — works with any column names passed from session.

Fixes applied:
1. perform_stationarity_tests returns {actual_col_name: results} — no more hardcoded "Yield"
2. generate_plots line plot uses numeric index for x-axis — no more compressed string-date display
3. _resolve_target_col picks second column as fallback — prevents last-column wrong detection
4. valid_exog always excludes time_col and target_col — prevents correlogram/scatter confusion
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================
# HELPERS — resolve column names robustly
# ============================================================

def _resolve_target_col(data, target_col):
    """
    Priority: explicit target_col → col with 'yield' in name
              → second column (after time) → first numeric col.
    Never picks the time column or last column blindly.
    """
    if target_col and target_col in data.columns:
        return target_col
    for col in data.columns:
        if "yield" in col.lower():
            return col
    # Second column is almost always the study variable
    if len(data.columns) >= 2:
        return data.columns[1]
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    raise ValueError("Could not find a target column in data.")


def _resolve_time_col(data, time_col):
    if time_col and time_col in data.columns:
        return time_col
    return data.columns[0]


def get_yield_series(data, target_col=None):
    col = _resolve_target_col(data, target_col)
    return pd.to_numeric(data[col], errors="coerce").dropna()


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def calculate_summary_statistics(data, selected_stats,
                                   target_col=None, exog_cols=None):
    from scipy import stats as _stats

    exog_cols       = exog_cols or []
    resolved_target = _resolve_target_col(data, target_col)

    col_pairs = [(resolved_target, resolved_target)]
    for ec in exog_cols:
        if ec in data.columns and ec != resolved_target:
            col_pairs.append((ec, ec))

    results = {}
    for col, label in col_pairs:
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        vstats = {}
        try:
            if 'mean'     in selected_stats: vstats['mean']     = round(float(series.mean()), 4)
            if 'median'   in selected_stats: vstats['median']   = round(float(series.median()), 4)
            if 'mode'     in selected_stats:
                m = _stats.mode(series, keepdims=True)
                vstats['mode'] = round(float(m.mode[0]), 4)
            if 'range'    in selected_stats: vstats['range']    = round(float(series.max() - series.min()), 4)
            if 'variance' in selected_stats: vstats['variance'] = round(float(series.var()), 4)
            if 'std'      in selected_stats: vstats['std']      = round(float(series.std()), 4)
            if 'cv'       in selected_stats: vstats['cv']       = round(float((series.std() / series.mean()) * 100), 4)
            if 'se'       in selected_stats: vstats['se']       = round(float(series.std() / np.sqrt(len(series))), 6)
            if 'skewness' in selected_stats: vstats['skewness'] = round(float(series.skew()), 4)
            if 'kurtosis' in selected_stats: vstats['kurtosis'] = round(float(series.kurtosis()), 4)
        except Exception as e:
            print(f"Error in summary stats for {label}: {e}")
        results[label] = vstats

    return results


# ============================================================
# NORMALITY TESTS
# ============================================================

def perform_normality_tests(data, selected_tests,
                             target_col=None, exog_cols=None):
    from scipy import stats as _stats

    exog_cols       = exog_cols or []
    resolved_target = _resolve_target_col(data, target_col)

    col_pairs = [(resolved_target, resolved_target)]
    for ec in exog_cols:
        if ec in data.columns and ec != resolved_target:
            col_pairs.append((ec, ec))

    results = {}
    for col, label in col_pairs:
        series = pd.to_numeric(data[col], errors="coerce").dropna()
        vtests = {}
        try:
            if 'shapiro' in selected_tests:
                stat, p = _stats.shapiro(series)
                vtests['shapiro'] = {'statistic': round(float(stat), 4), 'p_value': round(float(p), 4),
                                     'interpretation': 'Normal' if p > 0.05 else 'Not Normal'}
            if 'ks' in selected_tests:
                stat, p = _stats.kstest(series, 'norm')
                vtests['ks'] = {'statistic': round(float(stat), 4), 'p_value': round(float(p), 4),
                                'interpretation': 'Normal' if p > 0.05 else 'Not Normal'}
            if 'anderson' in selected_tests:
                from statsmodels.stats.diagnostic import normal_ad
                stat, p = normal_ad(series.values)
                vtests['anderson'] = {
                    'statistic':      round(float(stat), 4),
                    'p_value':        round(float(p), 4),
                    'interpretation': 'Normal' if p > 0.05 else 'Not Normal'
                }
            if 'jarque_bera' in selected_tests:
                stat, p = _stats.jarque_bera(series)
                vtests['jarque_bera'] = {'statistic': round(float(stat), 4), 'p_value': round(float(p), 4),
                                         'interpretation': 'Normal' if p > 0.05 else 'Not Normal'}
        except Exception as e:
            print(f"Normality test error for {label}: {e}")
        results[label] = vtests

    return results


# ============================================================
# STATIONARITY TESTS
# ✅ FIX: returns {actual_col_name: {test_results}}
#         so template shows the real variable name, not hardcoded "Yield"
# ============================================================

def perform_stationarity_tests(data, selected_tests, target_col=None):
    """
    Returns: { actual_col_name: { 'adf': {...}, 'pp': {...}, 'kpss': {...} } }
    """
    col_name    = _resolve_target_col(data, target_col)
    col_results = {}

    try:
        series = pd.to_numeric(data[col_name], errors="coerce").dropna()

        if 'adf' in selected_tests:
            result = adfuller(series, autolag='AIC')
            col_results['adf'] = {
                'statistic':       round(result[0], 4),
                'p_value':         round(result[1], 4),
                
                'interpretation':  'Stationary' if result[1] < 0.05 else 'Non-Stationary'
            }

        if 'pp' in selected_tests:
            try:
                from arch.unitroot import PhillipsPerron
                pp = PhillipsPerron(series, trend='c', test_type='tau')
                col_results['pp'] = {
                    'statistic':       round(float(pp.stat), 4),
                    'p_value':         round(float(pp.pvalue), 4),
                'lags':            int(pp.lags),
                'interpretation': 'Stationary' if pp.pvalue < 0.05 else 'Non-Stationary',
                
            }
            except ImportError:
                # Fallback if arch not installed — warn clearly
                col_results['pp'] = {
                    'statistic':      None,
                    'p_value':        None,
                    'interpretation': 'Error',
                    'note':           'arch library not installed. Run: pip install arch'
                }
            except Exception as pp_err:
                col_results['pp'] = {
                    'statistic':      None,
                    'p_value':        None,
                    'interpretation': 'Error',
                    'note':           str(pp_err)
                }

        if 'kpss' in selected_tests:
            result = kpss(series, regression='c', nlags='auto')
            col_results['kpss'] = {
                'statistic':       round(result[0], 4),
                'p_value':         round(result[1], 4),
                'interpretation':  'Non-Stationary' if result[1] < 0.05 else 'Stationary'
            }

    except Exception as e:
        print(f"Stationarity test error on '{col_name}': {e}")

    # ✅ Keyed by actual column name so template can display it
    return {col_name: col_results}


# ============================================================
# TREND TESTS
# ============================================================

def run_trend_tests(series: pd.Series, test_names: list) -> dict:
    try:
        import pymannkendall as mk
    except ImportError:
        raise ImportError("Install pymannkendall:  pip install pymannkendall")

    results = {}
    arr = series.dropna().values

    if "mann_kendall" in test_names:
        res       = mk.original_test(arr)
        direction = res.trend
        interp    = (
            f"A statistically significant {direction} trend was detected "
            f"(p = {res.p:.4f}, τ = {res.Tau:.3f}). Null hypothesis rejected at 5%."
            if res.h else
            f"No statistically significant trend detected (p = {res.p:.4f})."
        )
        results["mann_kendall"] = {
            "trend": direction, "h": bool(res.h),
            "p_value": round(float(res.p), 6), "z": round(float(res.z), 4),
            "tau": round(float(res.Tau), 4), "s": int(res.s),
            "var_s": round(float(res.var_s), 4), "interpretation": interp,
        }

    if "sens_slope" in test_names:
        res_ss    = mk.original_test(arr)
        slope     = round(float(res_ss.slope), 6)
        intercept = round(float(res_ss.intercept), 6)
        sign_str  = "positive (upward)" if slope > 0 else ("negative (downward)" if slope < 0 else "zero (no)")
        results["sens_slope"] = {
            "slope": slope, "intercept": intercept,
            "interpretation": (f"Sen's Slope = {slope:.4f} units/period, {sign_str} trend. "
                               f"Intercept = {intercept:.4f}."),
        }

    if "linear_regression" in test_names:
        results["linear_regression"] = _linear_regression_trend(arr)

    return results


def _linear_regression_trend(arr: np.ndarray) -> dict:
    """
    Ordinary Least Squares trend line with full significance metrics.

    Returns: slope, intercept, R², Adj-R², SE(slope), SE(intercept),
             t-statistic, p-value(slope), F-statistic, p-value(F),
             95% CI for slope and intercept, interpretation.
    """
    from scipy import stats as sc

    n   = len(arr)
    x   = np.arange(n, dtype=float)           # time index 0,1,2,...
    y   = arr.astype(float)

    # OLS via scipy linregress (exact, no matrix inversion needed)
    slope, intercept, r_value, p_value, se_slope = sc.linregress(x, y)

    r2     = r_value ** 2
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)    # n-2 because 1 predictor

    # Residuals & MSE
    y_hat  = slope * x + intercept
    resid  = y - y_hat
    mse    = np.sum(resid ** 2) / (n - 2)          # unbiased

    # SE of intercept
    se_int = np.sqrt(mse * (1/n + x.mean()**2 / np.sum((x - x.mean())**2)))

    # t-statistics
    t_slope = slope / se_slope
    t_int   = intercept / se_int

    # p-values (two-tailed)
    p_slope = float(2 * sc.t.sf(abs(t_slope), df=n-2))
    p_int   = float(2 * sc.t.sf(abs(t_int),   df=n-2))

    # F-statistic  (df1=1, df2=n-2)
    ss_reg = np.sum((y_hat - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    f_stat = (ss_reg / 1) / (ss_res / (n - 2))
    p_f    = float(sc.f.sf(f_stat, 1, n - 2))

    # 95% Confidence Intervals  (t_{0.025, n-2})
    t_crit       = sc.t.ppf(0.975, df=n-2)
    ci_slope     = (round(slope - t_crit * se_slope, 6),
                    round(slope + t_crit * se_slope, 6))
    ci_intercept = (round(intercept - t_crit * se_int, 6),
                    round(intercept + t_crit * se_int, 6))

    # Interpretation
    sig      = p_slope < 0.05
    dir_str  = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "no")
    sig_str  = "statistically significant" if sig else "not statistically significant"
    interp   = (
        f"Linear trend is {sig_str} (p = {p_slope:.4f}). "
        f"Slope = {slope:.4f} units/period — {dir_str} trend. "
        f"The model explains {r2*100:.1f}% of variance (R² = {r2:.4f})."
    )

    return {
        # Core equation
        "slope"         : round(float(slope),     6),
        "intercept"     : round(float(intercept), 6),
        # Goodness of fit
        "r2"            : round(float(r2),     4),
        "adj_r2"        : round(float(adj_r2), 4),
        # Slope significance
        "se_slope"      : round(float(se_slope), 6),
        "t_statistic"   : round(float(t_slope),  4),
        "p_value_slope" : round(p_slope, 6),
        # Intercept significance
        "se_intercept"  : round(float(se_int), 6),
        "t_intercept"   : round(float(t_int),  4),
        "p_value_intercept": round(p_int, 6),
        # Overall model
        "f_statistic"   : round(float(f_stat), 4),
        "p_value_f"     : round(float(p_f),    6),
        "n"             : int(n),
        # Confidence intervals
        "ci_slope_lower"    : ci_slope[0],
        "ci_slope_upper"    : ci_slope[1],
        "ci_intercept_lower": ci_intercept[0],
        "ci_intercept_upper": ci_intercept[1],
        # Summary
        "significant"   : sig,
        "direction"     : dir_str,
        "interpretation": interp,
    }


# ============================================================
# NON-LINEARITY TESTS
# ============================================================

def perform_nonlinearity_tests(data, selected_tests, target_col=None):
    results = {}
    try:
        series = get_yield_series(data, target_col)
        if 'bds' in selected_tests:  results['bds'] = bds_test(series)
        if 'tnn' in selected_tests:  results['tnn'] = terasvirta_test(series)
        if 'nn'  in selected_tests or 'wnn' in selected_tests:
            results['nn'] = white_test(series)
    except Exception as e:
        print(f"Nonlinearity test error: {e}")
    return results


def bds_test(series, m=2, eps_multiplier=0.7):
    try:
        from scipy import stats as sc
        series     = np.array(series)
        n          = len(series)
        ss         = (series - series.mean()) / series.std()
        eps        = eps_multiplier * ss.std()

        def ci(d, m_d, e):
            pts = len(d) - m_d + 1
            vecs = np.array([d[i:i+m_d] for i in range(pts)])
            cnt  = sum(1 for i in range(pts) for j in range(i+1, pts)
                       if np.max(np.abs(vecs[i]-vecs[j])) < e)
            return 2*cnt/(pts*(pts-1))

        cm, c1 = ci(ss, m, eps), ci(ss, 1, eps)
        if cm <= 0 or c1 <= 0:
            return {'test_name':'BDS','statistic':np.nan,'p_value':np.nan,'interpretation':'Unable to compute'}
        z = np.sqrt(n)*(cm - c1**m) / max(np.sqrt(m*c1**(2*m-2)), 1e-10)
        p = 2*(1 - sc.norm.cdf(abs(z)))
        return {'test_name':'BDS (Brock-Dechert-Scheinkman)',
                'statistic': round(float(z),4), 'p_value': round(float(p),4),
                'embedding_dim': m, 'epsilon': round(eps,4),
                'interpretation': 'IID' if p > 0.05 else 'Non-linear dependence detected'}
    except Exception as e:
        return {'test_name':'BDS','statistic':np.nan,'p_value':np.nan,'interpretation':f'Error:{e}'}


def terasvirta_test(series, lag=1):
    try:
        from scipy import stats as sc
        from numpy.linalg import lstsq
        s = np.array(series); n = len(s)
        if n < lag+10:
            return {'test_name':'TNN','statistic':np.nan,'p_value':np.nan,'interpretation':'Insufficient data'}
        y = s[lag:]; X = s[:-lag].reshape(-1,1)
        Xl = np.column_stack([np.ones(len(X)), X])
        Xu = np.column_stack([np.ones(len(X)), X, X**2, X**3])
        bl = lstsq(Xl, y, rcond=None)[0]; bu = lstsq(Xu, y, rcond=None)[0]
        rr = np.sum((y-Xl@bl)**2); ru = np.sum((y-Xu@bu)**2)
        kr,ku,n2 = Xl.shape[1], Xu.shape[1], len(y)
        F = ((rr-ru)/(ku-kr))/(ru/(n2-ku))
        p = 1-sc.f.cdf(F, ku-kr, n2-ku)
        return {'test_name':'TNN (Teräsvirta Neural Network)',
                'statistic':round(float(F),4),'p_value':round(float(p),4),'lag':lag,
                'interpretation':'Linear' if p>0.05 else 'Nonlinearity detected'}
    except Exception as e:
        return {'test_name':'TNN','statistic':np.nan,'p_value':np.nan,'interpretation':f'Error:{e}'}


def white_test(series, lag=1):
    try:
        from scipy import stats as sc
        from numpy.linalg import lstsq
        s = np.array(series); n = len(s)
        if n < lag+10:
            return {'test_name':'WNN','statistic':np.nan,'p_value':np.nan,'interpretation':'Insufficient data'}
        y = s[lag:]; X = s[:-lag].reshape(-1,1)
        Xl = np.column_stack([np.ones(len(X)), X])
        Xu = np.column_stack([np.ones(len(X)), X, X**2])
        bu = lstsq(Xu, y, rcond=None)[0]
        sst = np.sum((y-y.mean())**2)
        ssr = np.sum((y-Xu@bu)**2)
        lm  = len(y)*(1-ssr/sst)
        df  = Xu.shape[1]-Xl.shape[1]
        p   = 1-sc.chi2.cdf(lm, df)
        return {'test_name':'WNN (White Neural Network)',
                'statistic':round(float(lm),4),'p_value':round(float(p),4),'lag':lag,
                'interpretation':'Linear' if p>0.05 else 'Nonlinearity detected'}
    except Exception as e:
        return {'test_name':'WNN','statistic':np.nan,'p_value':np.nan,'interpretation':f'Error:{e}'}


# ============================================================
# PLOTS
# ============================================================

def generate_plots(data: pd.DataFrame, plot_types: list,
                   time_col=None, target_col=None, exog_cols=None) -> dict:

    os.makedirs("static/stats_plots", exist_ok=True)
    plot_files = {}
    exog_cols  = exog_cols or []

    # ── Resolve columns ───────────────────────────────────────────────────────
    time_col   = _resolve_time_col(data, time_col)
    target_col = _resolve_target_col(data, target_col)

    print(f"📊 generate_plots: time='{time_col}', target='{target_col}', exog={exog_cols}")

    # Exog: exclude time and target to avoid confusion
    valid_exog     = [c for c in exog_cols
                      if c in data.columns and c != time_col and c != target_col]
    is_multivariate = len(valid_exog) > 0

    PURPLE = "#667eea"; GREEN = "#28a745"; ORANGE = "#fd7f28"
    COLORS = [PURPLE, GREEN, ORANGE, "#e74c3c", "#f39c12", "#8e44ad"]

    def save(fig, name):
        path = f"static/stats_plots/{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return f"stats_plots/{name}.png"

    # ── Boxplot ───────────────────────────────────────────────────────────────
    if "boxplot" in plot_types:
        arr = pd.to_numeric(data[target_col], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot(arr, patch_artist=True,
                   boxprops=dict(facecolor="#dde4ff", color=PURPLE),
                   medianprops=dict(color=PURPLE, linewidth=2.5),
                   whiskerprops=dict(color=PURPLE, linewidth=1.5),
                   capprops=dict(color=PURPLE, linewidth=1.5),
                   flierprops=dict(marker="o", color=PURPLE, alpha=0.5, markersize=6))
        ax.set_title(f"Boxplot — {target_col}", fontsize=14, fontweight="bold")
        ax.set_ylabel(target_col); ax.set_xticks([1]); ax.set_xticklabels([target_col])
        ax.grid(axis="y", alpha=0.3)
        plot_files["boxplot"] = save(fig, "boxplot")

        if valid_exog:
            n_exog = len(valid_exog)
            fig, axes = plt.subplots(1, n_exog, figsize=(6*n_exog, 5))
            if n_exog == 1: axes = [axes]
            fig.suptitle("Boxplots — Exogenous Variables", fontsize=14, fontweight="bold", y=1.02)
            for ax, ec, color in zip(axes, valid_exog, COLORS[1:]):
                ec_arr = pd.to_numeric(data[ec], errors="coerce").dropna()
                ax.boxplot(ec_arr, patch_artist=True,
                           boxprops=dict(facecolor=color+"33", color=color),
                           medianprops=dict(color=color, linewidth=2.5),
                           whiskerprops=dict(color=color, linewidth=1.5),
                           capprops=dict(color=color, linewidth=1.5),
                           flierprops=dict(marker="o", color=color, alpha=0.5, markersize=6))
                ax.set_title(f"Boxplot — {ec}", fontsize=12, fontweight="bold")
                ax.set_ylabel(ec); ax.set_xticks([1]); ax.set_xticklabels([ec])
                ax.grid(axis="y", alpha=0.3)
                if len(ec_arr) > 0:
                    q1, med, q3 = ec_arr.quantile([0.25,0.5,0.75])
                    ax.text(1.35, med, f"Median={med:.2f}\nQ1={q1:.2f}\nQ3={q3:.2f}\nIQR={q3-q1:.2f}",
                            va="center", ha="left", fontsize=8.5, transform=ax.get_yaxis_transform())
            fig.tight_layout()
            plot_files["boxplot_exog"] = save(fig, "boxplot_exog")

    # ── Line plot ─────────────────────────────────────────────────────────────
    # ✅ FIX: use numeric index for x-axis — avoids compressed string-date display
    if "lineplot" in plot_types:
        y        = pd.to_numeric(data[target_col], errors="coerce").values
        x_idx    = list(range(len(y)))
        x_labels = [str(v) for v in data[time_col]]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_idx, y, color=PURPLE, linewidth=2, marker="o", markersize=4)
        ax.set_title(f"Line Plot — {target_col} over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel(time_col); ax.set_ylabel(target_col)

        step = max(1, len(x_idx) // 12)
        ax.set_xticks(x_idx[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45, ha="right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_files["lineplot"] = save(fig, "lineplot")

    # ── Histogram ─────────────────────────────────────────────────────────────
    if "histogram" in plot_types:
        from scipy.stats import gaussian_kde

        # ── Target variable histogram ──────────────────────────────────────
        arr = pd.to_numeric(data[target_col], errors="coerce").dropna().values
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(arr, bins=15, color=PURPLE, alpha=0.5, density=True, edgecolor="white")
        xs = np.linspace(arr.min(), arr.max(), 200)
        ax.plot(xs, gaussian_kde(arr)(xs), color="#764ba2", linewidth=2)
        ax.set_title(f"Histogram & Density — {target_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(target_col); ax.set_ylabel("Density"); ax.grid(alpha=0.3)
        plot_files["histogram"] = save(fig, "histogram")

        # ── Exog variables histogram (multivariate only) ───────────────────
        if valid_exog:
            n_exog = len(valid_exog)
            fig, axes = plt.subplots(1, n_exog, figsize=(7 * n_exog, 5))
            if n_exog == 1: axes = [axes]
            fig.suptitle("Histogram & Density — Exogenous Variables",
                         fontsize=14, fontweight="bold", y=1.02)
            for ax, ec, color in zip(axes, valid_exog, COLORS[1:]):
                ec_arr = pd.to_numeric(data[ec], errors="coerce").dropna().values
                if len(ec_arr) < 2:
                    ax.set_title(f"{ec} — insufficient data"); continue
                ax.hist(ec_arr, bins=15, color=color, alpha=0.5,
                        density=True, edgecolor="white")
                ec_xs = np.linspace(ec_arr.min(), ec_arr.max(), 200)
                kde_color = "#" + "".join(
                    f"{max(0, int(c, 16) - 30):02x}"
                    for c in [color[1:3], color[3:5], color[5:7]]
                ) if color.startswith("#") else color
                ax.plot(ec_xs, gaussian_kde(ec_arr)(ec_xs),
                        color=kde_color, linewidth=2)
                ax.set_title(f"Histogram & Density — {ec}",
                             fontsize=11, fontweight="bold")
                ax.set_xlabel(ec); ax.set_ylabel("Density"); ax.grid(alpha=0.3)
            fig.tight_layout()
            plot_files["histogram_exog"] = save(fig, "histogram_exog")

    # ── Scatter (each exog vs target) ────────────────────────────────────────
    if "scatter" in plot_types:
        yv   = pd.to_numeric(data[target_col], errors="coerce")
        x_idx = np.arange(len(yv))                 # numeric positions for regression
        mask_t = yv.notna()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(x_idx[mask_t], yv[mask_t], color=PURPLE, alpha=0.75,
                   edgecolors="white", linewidth=0.5, zorder=3)

        if mask_t.sum() >= 2:
            xm = x_idx[mask_t]; ym = yv[mask_t].values
            b_coef, a_coef = np.polyfit(xm, ym, 1)          # b=slope, a=intercept
            xs = np.linspace(xm.min(), xm.max(), 200)
            ax.plot(xs, a_coef + b_coef * xs, color="#764ba2",
                    linewidth=2, linestyle="--", zorder=4,
                    label=f"$y = {a_coef:.3f} + {b_coef:.3f}x$")
            ax.legend(fontsize=10, loc="best")

        # X-axis: use actual time labels at readable intervals
        x_labels = [str(v) for v in data[time_col]]
        step = max(1, len(x_idx) // 12)
        ax.set_xticks(x_idx[::step])
        ax.set_xticklabels(x_labels[::step], rotation=45, ha="right")

        ax.set_xlabel(time_col); ax.set_ylabel(target_col)
        ax.set_title(f"Scatter Plot — {time_col} vs {target_col}",
                     fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_files["scatter_time"] = save(fig, "scatter_time")

    # ── Scatter: each exog vs target (multivariate only) ─────────────────────
    if "scatter" in plot_types and valid_exog:
        n_exog = len(valid_exog)
        fig, axes = plt.subplots(1, n_exog, figsize=(6 * n_exog, 5))
        if n_exog == 1: axes = [axes]

        for ax, ec, color in zip(axes, valid_exog, COLORS[1:]):
            xv = pd.to_numeric(data[ec],         errors="coerce")
            yv = pd.to_numeric(data[target_col], errors="coerce")
            mask = xv.notna() & yv.notna()
            ax.scatter(xv[mask], yv[mask], color=color, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)

            if mask.sum() >= 2:
                xm = xv[mask].values; ym = yv[mask].values
                b_coef, a_coef = np.polyfit(xm, ym, 1)      # b=slope, a=intercept
                xs = np.linspace(xm.min(), xm.max(), 200)
                eq = f"$y = {a_coef:.3f} + {b_coef:.3f}x$"
                ax.plot(xs, a_coef + b_coef * xs, "k--",
                        linewidth=1.8, zorder=4, label=eq)
                ax.legend(fontsize=9, loc="best")

            ax.set_xlabel(ec); ax.set_ylabel(target_col)
            ax.set_title(f"{ec} vs {target_col}", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Scatter Plots — Exogenous Variables vs {target_col}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot_files["scatter"] = save(fig, "scatter")

    # ── Correlogram (target + exog, no time column) ───────────────────────────
    if "correlogram" in plot_types and valid_exog:
        from scipy.stats import pearsonr
        import matplotlib.transforms as mtransforms

        corr_cols = [target_col] + valid_exog
        corr_df   = data[corr_cols].apply(pd.to_numeric, errors="coerce").dropna()
        n_obs = len(corr_df); n = len(corr_cols)

        r_mat = np.ones((n, n)); p_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r, p = pearsonr(corr_df.iloc[:, i], corr_df.iloc[:, j])
                    r_mat[i, j] = r; p_mat[i, j] = p

        def sig_sup(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            else:
                return "ns"
        sz  = max(6, n * 1.8)
        fig, ax = plt.subplots(figsize=(sz + 2, sz + 2))

        # ✅ pcolormesh: zero inter-cell gaps — no white stripes ever
        # pcolormesh origin is bottom-left so flip the matrix rows
        r_plot = r_mat[::-1]
        mesh   = ax.pcolormesh(r_plot, cmap="coolwarm", vmin=-1, vmax=1,
                               linewidth=0, edgecolors="none")
        fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02).set_label("Pearson r", fontsize=10)

        # Cell centres for pcolormesh are at 0.5, 1.5, ...
        centres = np.arange(n) + 0.5

        # X ticks (top) — columns
        ax.set_xlim(0, n); ax.set_ylim(0, n)
        ax.set_xticks(centres)
        ax.set_xticklabels(corr_cols, fontsize=10, fontweight="bold")
        ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")

        # Y ticks — rows are flipped so reverse label order
        ax.set_yticks(centres)
        ax.set_yticklabels(corr_cols[::-1], fontsize=10, fontweight="bold")

        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Annotate each cell ────────────────────────────────────────────────
        for i in range(n):                       # data row (top = 0)
            py = n - i - 0.5                     # pcolormesh y (flipped)
            for j in range(n):
                px  = j + 0.5                    # pcolormesh x
                rv  = r_mat[i, j]
                if i == j:
                    sup = ""
                else:
                    sup = sig_sup(p_mat[i, j])
                tc  = "white" if abs(rv) > 0.55 else "#1a202c"
                val = f"{rv:.3f}"

                if not sup:
                    # No stars — just centred value
                    ax.text(px, py, val,
                            ha="center", va="center",
                            fontsize=14, fontweight="bold", color=tc)
                else:
                    # ✅ Value slightly left of centre; stars as tight superscript
                    # Shift the number left by ~0.13 so stars fit within cell
                    star_color = "#FFD700" if abs(rv) > 0.55 else "#c0392b"
                    ax.text(px - 0.13, py, val,
                            ha="center", va="center",
                            fontsize=14, fontweight="bold", color=tc)
                    # Stars: same horizontal end as number, raised by 0.18 units
                    ax.text(px + 0.22, py + 0.18, sup,
                            ha="left", va="center",
                            fontsize=14, fontweight="bold", color=star_color)

        ax.set_title("Correlation Matrix (Pearson r)",
                     fontsize=13, fontweight="bold", pad=18)
        fig.text(0.5, -0.02,
                 "*** p<0.001   ** p<0.01   * p<0.05   ns = not significant   "
                 f"n = {n_obs}",
                 ha="center", fontsize=11, style="italic",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="#f7fafc", edgecolor="#cbd5e0"))
        fig.tight_layout()
        plot_files["correlogram"] = save(fig, "CORRELOGRAM")
    # ── ACF ───────────────────────────────────────────────────────────────────
    if "acf" in plot_types:
        from statsmodels.tsa.stattools import acf as sm_acf
        arr      = pd.to_numeric(data[target_col], errors="coerce").dropna()
        max_lags = min(20, len(arr)//2 - 1)

        # Compute ACF values + confidence intervals, skip lag 0
        acf_vals, confint = sm_acf(arr, nlags=max_lags, alpha=0.05)
        lags      = np.arange(1, max_lags + 1)        # start from 1
        acf_vals  = acf_vals[1:]                       # drop lag 0
        ci_lower  = confint[1:, 0] - acf_vals          # relative lower CI
        ci_upper  = confint[1:, 1] - acf_vals          # relative upper CI
        conf_band = 1.96 / np.sqrt(len(arr))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(lags, acf_vals, color=PURPLE, alpha=0.7, width=0.4)
        ax.axhline( conf_band, color="red", linestyle="--", linewidth=1, label="95% CI")
        ax.axhline(-conf_band, color="red", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(lags, -conf_band, conf_band, alpha=0.1, color="blue")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.set_title(f"ACF — {target_col}", fontsize=13, fontweight="bold")
        ax.set_xticks(lags); ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plot_files["acf"] = save(fig, "acf")

    # ── PACF ──────────────────────────────────────────────────────────────────
    if "pacf" in plot_types:
        from statsmodels.tsa.stattools import pacf as sm_pacf
        arr      = pd.to_numeric(data[target_col], errors="coerce").dropna()
        max_lags = min(20, len(arr)//2 - 1)

        # Compute PACF values + confidence intervals, skip lag 0
        pacf_vals, confint = sm_pacf(arr, nlags=max_lags, alpha=0.05, method="ywm")
        lags       = np.arange(1, max_lags + 1)       # start from 1
        pacf_vals  = pacf_vals[1:]                     # drop lag 0
        conf_band  = 1.96 / np.sqrt(len(arr))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(lags, pacf_vals, color="#764ba2", alpha=0.7, width=0.4)
        ax.axhline( conf_band, color="red", linestyle="--", linewidth=1, label="95% CI")
        ax.axhline(-conf_band, color="red", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.fill_between(lags, -conf_band, conf_band, alpha=0.1, color="blue")
        ax.set_xlabel("Lag"); ax.set_ylabel("PACF")
        ax.set_title(f"PACF — {target_col}", fontsize=13, fontweight="bold")
        ax.set_xticks(lags); ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plot_files["pacf"] = save(fig, "PACF")

    # ── CCF ───────────────────────────────────────────────────────────────────
    if "ccf" in plot_types and valid_exog:
        from statsmodels.tsa.stattools import ccf as sm_ccf
        n_exog = len(valid_exog)
        fig, axes = plt.subplots(1, n_exog, figsize=(7*n_exog, 5), squeeze=False)
        max_lags   = min(15, len(data)//3)
        conf_level = 1.96/np.sqrt(len(data.dropna()))
        for ax, ec, color in zip(axes[0], valid_exog, COLORS[1:]):
            x = pd.to_numeric(data[ec],         errors="coerce").dropna()
            y = pd.to_numeric(data[target_col], errors="coerce").reindex(x.index).dropna()
            x = x.reindex(y.index)
            ccf_vals = sm_ccf(x.values, y.values, nlags=max_lags, adjusted=True)
            lags     = np.arange(len(ccf_vals))
            ax.bar(lags, ccf_vals, color=color, alpha=0.7, width=0.4)
            ax.axhline( conf_level, color="red", linestyle="--", linewidth=1, label="95% CI")
            ax.axhline(-conf_level, color="red", linestyle="--", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"CCF: {ec} → {target_col}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Lag (periods)"); ax.set_ylabel("Cross-Correlation")
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.suptitle(f"Cross-Correlation Function (CCF)\nExogenous Variables vs {target_col}",
                     fontsize=14, fontweight="bold", y=1.03)
        plot_files["ccf"] = save(fig, "CCF")

    return plot_files


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def generate_all_statistics(
    data,
    summary_stats=None, normality_tests=None, stationarity_tests=None,
    nonlinearity_tests=None, trend_tests=None, plots=None,
    time_col=None, target_col=None, exog_cols=None,
):
    summary_stats      = summary_stats      or []
    normality_tests    = normality_tests    or []
    stationarity_tests = stationarity_tests or []
    nonlinearity_tests = nonlinearity_tests or []
    trend_tests        = trend_tests        or []
    plots              = plots              or []
    exog_cols          = exog_cols          or []

    # Resolve once
    resolved_target = _resolve_target_col(data, target_col)
    resolved_time   = _resolve_time_col(data, time_col)

    print(f"📊 generate_all_statistics: time='{resolved_time}', "
          f"target='{resolved_target}', exog={exog_cols}")

    results = {}

    if summary_stats:
        results["summary_stats"] = calculate_summary_statistics(
            data, summary_stats, target_col=resolved_target, exog_cols=exog_cols)

    if normality_tests:
        results["normality_tests"] = perform_normality_tests(
            data, normality_tests, target_col=resolved_target, exog_cols=exog_cols)

    if stationarity_tests:
        results["stationarity_tests"] = perform_stationarity_tests(
            data, stationarity_tests, target_col=resolved_target)

    if nonlinearity_tests:
        results["nonlinearity_tests"] = perform_nonlinearity_tests(
            data, nonlinearity_tests, target_col=resolved_target)

    if trend_tests:
        series = get_yield_series(data, resolved_target)
        results["trend_tests"] = run_trend_tests(series, trend_tests)

    if plots:
        results["plots"] = generate_plots(
            data, plots,
            time_col=resolved_time,
            target_col=resolved_target,
            exog_cols=exog_cols,
        )

    return results


# ============================================================
# TESTING (if run directly)
# ============================================================

if __name__ == "__main__":
    # Example usage
    print("Statistics Generator Module - Test Mode")
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Year': range(2000, 2024),
        'Yield': np.random.normal(100, 15, 24) + np.arange(24) * 0.5
    })
    
    # Test all functions
    results = generate_all_statistics(
        data=sample_data,
        summary_stats=['mean', 'median', 'std', 'variance'],
        normality_tests=['shapiro', 'jarque_bera'],
        stationarity_tests=['adf', 'kpss'],
        plots=['lineplot', 'histogram', 'acf']
    )
    
    print("\n📊 Sample Results:")
    print(f"Summary Stats: {results['summary_stats']}")
    print(f"Normality Tests: {results['normality_tests']}")
    print(f"Stationarity Tests: {results['stationarity_tests']}")
    print(f"Plots Generated: {list(results['plots'].keys())}")