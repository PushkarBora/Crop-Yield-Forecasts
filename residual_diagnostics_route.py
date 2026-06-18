"""
residual_diagnostics_route.py
─────────────────────────────
Paste this into your main Flask app file (app.py / routes.py).

Assumes:
  • results_store  – your existing global dict  {model_key: results_dict}
  • results_dict contains a key  "residuals"  – a 1-D numpy array of in-sample residuals
    (training residuals are preferred; add   results["residuals"] = residuals
     wherever you already compute  y_train - y_pred_train).
  • STATIC_DIR  –  app.static_folder  (already available in Flask)
"""

import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.tsa.stattools import adfuller, kpss, bds as bds_test
from scipy import stats

# ─── helper ──────────────────────────────────────────────────────────────────

def _fmt(v, dec=4):
    """Format a float safely."""
    try:
        return f"{float(v):.{dec}f}"
    except Exception:
        return "N/A"


def run_residual_diagnostics(residuals: np.ndarray) -> dict:
    """
    Run a full suite of residual diagnostic tests.
    Returns a nested dict ready for the Jinja2 template.
    """
    res = {}
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]          # drop NaN / Inf

    n = len(r)

    # ── 1. NORMALITY ──────────────────────────────────────────────────────────

    # Shapiro-Wilk  (reliable up to n~5000)
    sw_stat, sw_p = stats.shapiro(r[:5000] if n > 5000 else r)
    res["shapiro"] = {
        "stat": _fmt(sw_stat), "p": _fmt(sw_p),
        "reject": sw_p < 0.05,
        "interpretation": (
            "Residuals are NOT normally distributed (p < 0.05)."
            if sw_p < 0.05 else
            "No evidence against normality (p ≥ 0.05)."
        ),
    }

    # Kolmogorov-Smirnov
    ks_stat, ks_p = stats.kstest(r, "norm", args=(r.mean(), r.std(ddof=1)))
    res["ks"] = {
        "stat": _fmt(ks_stat), "p": _fmt(ks_p),
        "reject": ks_p < 0.05,
        "interpretation": (
            "Residuals deviate significantly from normality (p < 0.05)."
            if ks_p < 0.05 else
            "No significant deviation from normality (p ≥ 0.05)."
        ),
    }

    # Anderson-Darling
    ad = stats.anderson(r, dist="norm")
    ad_reject_5 = ad.statistic > ad.critical_values[2]   # index 2 → 5 % level
    res["anderson"] = {
        "stat": _fmt(ad.statistic),
        "critical_5pct": _fmt(ad.critical_values[2]),
        "reject": ad_reject_5,
        "interpretation": (
            f"Statistic ({_fmt(ad.statistic)}) > critical value at 5 % ({_fmt(ad.critical_values[2])}): "
            "NOT normally distributed."
            if ad_reject_5 else
            f"Statistic ({_fmt(ad.statistic)}) ≤ critical value at 5 % ({_fmt(ad.critical_values[2])}): "
            "No evidence against normality."
        ),
    }

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(r)
    res["jarque_bera"] = {
        "stat": _fmt(jb_stat), "p": _fmt(jb_p),
        "reject": jb_p < 0.05,
        "skewness": _fmt(stats.skew(r)),
        "kurtosis": _fmt(stats.kurtosis(r)),
        "interpretation": (
            "Residuals are NOT normally distributed — skewness/kurtosis (p < 0.05)."
            if jb_p < 0.05 else
            "No evidence of non-normality via skewness/kurtosis (p ≥ 0.05)."
        ),
    }

    # ── 2. AUTOCORRELATION — Ljung-Box ────────────────────────────────────────

    lags_to_test = [5, 10, 15, 20]
    lags_to_test = [l for l in lags_to_test if l < n]
    lb_df = acorr_ljungbox(r, lags=lags_to_test, return_df=True)
    lb_rows = []
    for lag, row in lb_df.iterrows():
        lb_rows.append({
            "lag":  int(lag),
            "stat": _fmt(row["lb_stat"]),
            "p":    _fmt(row["lb_pvalue"]),
            "reject": row["lb_pvalue"] < 0.05,
        })
    # primary verdict at lag-10 (or last available)
    primary_p = lb_df["lb_pvalue"].iloc[-1]
    res["ljung_box"] = {
        "rows": lb_rows,
        "reject": primary_p < 0.05,
        "interpretation": (
            "Significant autocorrelation detected (p < 0.05) — residuals are not white noise."
            if primary_p < 0.05 else
            "No significant autocorrelation — residuals resemble white noise (p ≥ 0.05)."
        ),
    }

    # ── 3. HETEROSKEDASTICITY — White's Test ──────────────────────────────────

    try:
        X = sm.add_constant(np.arange(n, dtype=float).reshape(-1, 1))
        lm_stat, lm_p, f_stat, f_p = het_white(r, X)
        res["white"] = {
            "stat": _fmt(lm_stat), "p": _fmt(lm_p),
            "f_stat": _fmt(f_stat), "f_p": _fmt(f_p),
            "reject": lm_p < 0.05,
            "interpretation": (
                "Heteroskedasticity detected — variance is NOT constant (p < 0.05)."
                if lm_p < 0.05 else
                "No evidence of heteroskedasticity — residual variance is constant (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["white"] = {"error": str(exc)}

    # ── 4. STATIONARITY ───────────────────────────────────────────────────────

    # ADF
    adf_out = adfuller(r, autolag="AIC")
    adf_cv  = adf_out[4]
    res["adf"] = {
        "stat": _fmt(adf_out[0]), "p": _fmt(adf_out[1]),
        "cv_1pct":  _fmt(adf_cv["1%"]),
        "cv_5pct":  _fmt(adf_cv["5%"]),
        "cv_10pct": _fmt(adf_cv["10%"]),
        "reject": adf_out[1] < 0.05,          # reject unit-root → stationary
        "interpretation": (
            "Residuals are STATIONARY — unit root rejected (p < 0.05)."
            if adf_out[1] < 0.05 else
            "Cannot reject unit root — residuals may be NON-STATIONARY (p ≥ 0.05)."
        ),
    }

    # Phillips-Perron  (via arch)
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(r)
        res["pp"] = {
            "stat": _fmt(pp.stat), "p": _fmt(pp.pvalue),
            "reject": pp.pvalue < 0.05,
            "interpretation": (
                "Residuals are STATIONARY — unit root rejected (p < 0.05)."
                if pp.pvalue < 0.05 else
                "Cannot reject unit root — residuals may be NON-STATIONARY (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["pp"] = {"error": str(exc)}

    # KPSS  (H0: stationary → reject = non-stationary)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, _, kpss_cv = kpss(r, regression="c", nlags="auto")
        res["kpss"] = {
            "stat": _fmt(kpss_stat), "p": _fmt(kpss_p),
            "cv_5pct": _fmt(kpss_cv["5%"]),
            "reject": kpss_p < 0.05,          # reject stationarity
            "interpretation": (
                "Residuals are NON-STATIONARY — stationarity rejected (p < 0.05)."
                if kpss_p < 0.05 else
                "No evidence against stationarity (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["kpss"] = {"error": str(exc)}

    # ── 5. NON-LINEARITY ──────────────────────────────────────────────────────

    # BDS
    try:
        max_dim = min(6, n // 10)
        bds_stats, bds_ps = bds_test(r, max_dim=max(2, max_dim))
        res["bds"] = {
            "stat": _fmt(bds_stats[0]), "p": _fmt(bds_ps[0]),
            "reject": float(bds_ps[0]) < 0.05,
            "interpretation": (
                "Non-random structure / non-linearity detected (p < 0.05)."
                if float(bds_ps[0]) < 0.05 else
                "No significant non-random structure detected (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["bds"] = {"error": str(exc)}

    # Terasvirta Neural Network Test (TNN)
    try:
        from statsmodels.stats.diagnostic import linear_reset
        # Build an AR(1) regression for the NL tests
        y_tnn  = r[1:]
        X_tnn  = sm.add_constant(r[:-1])
        ols    = sm.OLS(y_tnn, X_tnn).fit()
        tnn    = linear_reset(ols, power=3, use_f=False)
        res["tnn"] = {
            "stat": _fmt(tnn.statistic), "p": _fmt(tnn.pvalue),
            "reject": tnn.pvalue < 0.05,
            "interpretation": (
                "Non-linearity detected via smooth transition (p < 0.05)."
                if tnn.pvalue < 0.05 else
                "No smooth-transition non-linearity detected (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["tnn"] = {"error": str(exc)}

    # White Neural Network Test (WNN)
    try:
        from statsmodels.stats.diagnostic import linear_reset
        y_wnn  = r[1:]
        X_wnn  = sm.add_constant(r[:-1])
        ols_w  = sm.OLS(y_wnn, X_wnn).fit()
        wnn    = linear_reset(ols_w, power=2, use_f=True)
        res["wnn"] = {
            "stat": _fmt(wnn.statistic), "p": _fmt(wnn.pvalue),
            "reject": wnn.pvalue < 0.05,
            "interpretation": (
                "Neglected non-linearity detected (p < 0.05)."
                if wnn.pvalue < 0.05 else
                "No neglected non-linearity detected (p ≥ 0.05)."
            ),
        }
    except Exception as exc:
        res["wnn"] = {"error": str(exc)}

    return res


# ─── plot generator ──────────────────────────────────────────────────────────

def generate_residual_plot(residuals: np.ndarray, model_key: str,
                           static_folder: str) -> str:
    """
    Generates the 3-panel diagnostic plot (residuals / ACF / histogram).
    Saves as  static/<model_key>_residuals.png
    Returns the filename  (model_key + '_residuals.png')
    """
    r   = np.asarray(residuals, dtype=float)
    r   = r[np.isfinite(r)]
    n   = len(r)

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1, 1.2],
                   hspace=0.45, wspace=0.35)

    # ── Panel 1: residuals time series (top, full width) ─────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(np.arange(n), r, color="#2d3748", linewidth=0.9, alpha=0.85)
    ax1.axhline(0, color="#e53e3e", linewidth=1.2, linestyle="--", alpha=0.7)
    ax1.set_title("Residuals over Time", fontsize=13, fontweight="bold",
                  color="#2d3748", pad=8)
    ax1.set_xlabel("Observation", fontsize=10)
    ax1.set_ylabel("Residual", fontsize=10)
    ax1.tick_params(labelsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_facecolor("#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: ACF (bottom-left) ────────────────────────────────────────────
    ax2  = fig.add_subplot(gs[1, 0])
    nlags = min(20, n // 2 - 1)
    sm.graphics.tsa.plot_acf(r, lags=nlags, ax=ax2, color="#667eea",
                              vlines_kwargs={"colors": "#2d3748", "linewidth": 1.4},
                              alpha=0.05, zero=False)
    ax2.set_title("ACF of Residuals", fontsize=12, fontweight="bold",
                  color="#2d3748", pad=6)
    ax2.set_xlabel("Lag", fontsize=10)
    ax2.set_ylabel("ACF", fontsize=10)
    ax2.tick_params(labelsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_facecolor("#f8fafc")
    for line in ax2.get_lines():
        if line.get_linestyle() == "--":
            line.set_color("#e53e3e")
            line.set_alpha(0.7)

    # ── Panel 3: Histogram + normal curve (bottom-right) ─────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(r, bins=min(30, int(np.sqrt(n) * 1.5)),
             color="#667eea", edgecolor="white",
             linewidth=0.4, alpha=0.85, density=True)

    # overlay normal PDF
    xmin, xmax = ax3.get_xlim()
    xs  = np.linspace(xmin, xmax, 300)
    pdf = stats.norm.pdf(xs, r.mean(), r.std(ddof=1))
    ax3.plot(xs, pdf, color="#e53e3e", linewidth=2.2, alpha=0.9, label="Normal")

    ax3.set_title("Histogram of Residuals", fontsize=12, fontweight="bold",
                  color="#2d3748", pad=6)
    ax3.set_xlabel("Residuals", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.tick_params(labelsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_facecolor("#f8fafc")
    ax3.legend(fontsize=9)

    fig.suptitle("Residual Diagnostic Plots", fontsize=15,
                 fontweight="bold", color="#2d3748", y=1.01)

    fname = f"{model_key}_residuals.png"
    fpath = os.path.join(static_folder, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return fname


# ─── Flask route ─────────────────────────────────────────────────────────────
# Paste this block into your app.py / routes.py alongside your other routes.

