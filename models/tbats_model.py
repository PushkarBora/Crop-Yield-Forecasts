import numpy as np
import pandas as pd
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# FORECAST-ONLY  (loads saved bundle, no retraining)
# ============================================================
def _forecast_only_tbats(params, horizon):
    if not os.path.exists("static/tbats_bundle.pkl"):
        raise FileNotFoundError(
            "TBATS model not trained yet. Please train the model first before forecasting."
        )

    with open("static/tbats_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    with open("static/tbats_model.pkl", "rb") as f:
        tbats_fit = pickle.load(f)

    time_labels  = bundle["time_labels"]
    resid_std    = bundle.get("resid_std", 1.0)

    # ── Forecast ────────────────────────────────────────────────────
    mean_vals = tbats_fit.forecast(steps=horizon)

    # ── Prediction intervals using residual std ──────────────────────
    # h-step ahead variance grows as std * sqrt(h)
    h_arr     = np.arange(1, horizon + 1)
    std_arr   = resid_std * np.sqrt(h_arr)

    lower_95  = mean_vals - 1.960 * std_arr
    upper_95  = mean_vals + 1.960 * std_arr
    lower_80  = mean_vals - 1.282 * std_arr
    upper_80  = mean_vals + 1.282 * std_arr

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
# HELPER — build parameter-estimates DataFrame from TBATS model
# ============================================================
def _param_estimates_df(model):
    """
    Extracts TBATS model parameters into a tidy DataFrame.
    TBATS params: alpha, beta, phi, Box-Cox lambda, ARMA p/q coefficients,
    seasonal harmonics (k per period).
    """
    try:
        p = model.params
        rows = []

        # Core smoothing params
        if p.alpha is not None:
            rows.append(("alpha (Level)",       round(float(p.alpha),  6)))
        if p.beta is not None:
            rows.append(("beta (Trend)",         round(float(p.beta),   6)))
        if p.phi is not None:
            rows.append(("phi (Damping)",        round(float(p.phi),    6)))
        if p.box_cox_lambda is not None:
            rows.append(("Box-Cox λ",            round(float(p.box_cox_lambda), 6)))

        # Seasonal harmonics
        if p.components.seasonal_harmonics is not None:
            for i, k in enumerate(p.components.seasonal_harmonics):
                rows.append((f"Harmonics (Season {i+1})", int(k)))

        # ARMA coefficients
        ar_coeffs = getattr(p, 'ar_coeffs', None) or []
        ma_coeffs = getattr(p, 'ma_coeffs', None) or []
        for i, v in enumerate(ar_coeffs, 1):
            rows.append((f"AR({i})", round(float(v), 6)))
        for i, v in enumerate(ma_coeffs, 1):
            rows.append((f"MA({i})", round(float(v), 6)))

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["Parameter", "Coef"])
        # TBATS has no std errors or p-values — fill with "—"
        df["Std Err"] = "—"
        df["z"]       = "—"
        df["P>|z|"]   = "—"
        return df

    except Exception:
        return None


# ============================================================
# HELPER — collect information criteria from TBATS model
# ============================================================
def _info_criteria(model):
    criteria = {}
    try:
        criteria["AIC"] = round(float(model.aic), 4)
    except Exception:
        pass
    return criteria


# ============================================================
# HELPER — describe fitted TBATS components
# ============================================================
def _model_description(model):
    """Returns a human-readable string describing the fitted TBATS."""
    try:
        c = model.params.components
        parts = []
        if c.use_box_cox:
            parts.append(f"BoxCox(λ={round(float(model.params.box_cox_lambda), 3)})")
        if c.use_trend:
            parts.append("Trend")
            if c.use_damped_trend:
                parts.append("Damped")
        sp_list = c.seasonal_periods if c.seasonal_periods is not None and len(c.seasonal_periods) > 0 else []
        for i, sp in enumerate(sp_list):
            k = c.seasonal_harmonics[i] if c.seasonal_harmonics is not None else "?"
            parts.append(f"S{int(sp)}(k={k})")
        if c.use_arma_errors:
            p_ar = len(getattr(model.params, 'ar_coeffs', None) or [])
            q_ma = len(getattr(model.params, 'ma_coeffs', None) or [])
            parts.append(f"ARMA({p_ar},{q_ma})")
        return "TBATS(" + ", ".join(parts) + ")" if parts else "TBATS"
    except Exception:
        return "TBATS"


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_tbats(
    data        : pd.DataFrame,
    params      : dict,
    horizon     : int,
    frequency   : str,
    mode        : str = "train",
    log_callback      = None,
):
    """
    TBATS model — Trigonometric seasonality, Box-Cox transformation,
    ARMA errors, Trend and Seasonal components.

    Always univariate — no exogenous variable support.
    Supports multiple seasonal periods (e.g. daily + weekly + annual).

    Forecast includes 80% and 95% prediction intervals derived from
    residual standard deviation (h-step scaling: std × √h).
    """

    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    import threading
    # ── FORECAST MODE ───────────────────────────────────────────────
    if mode == "forecast":
        return _forecast_only_tbats(params=params, horizon=horizon)

    # ── Validate ────────────────────────────────────────────────────
    if data is None:
        raise ValueError("data must be provided in train mode.")

    # ── Lazy import (so missing package gives clear error) ───────────
    try:
        from tbats import TBATS as TBATSEstimator
    except ImportError:
        raise ImportError(
            "The 'tbats' package is required. Install it with: pip install tbats"
        )

    # ── Column mappings ──────────────────────────────────────────────
    time_col   = params.get("time_col",   "Year")
    target_col = params.get("target_col", "Yield")

    log(f"📊 TBATS — Univariate")
    log(f"   Time column    : {time_col}")
    log(f"   Study variable : {target_col}")

    # ── Hyperparameters ──────────────────────────────────────────────
    split              = params.get("split",           0.85)
    use_auto           = params.get("auto_tbats",      True)

    # Seasonal periods — can be a list for multiple seasonalities
    # e.g. [12] for monthly, [7, 365] for daily data, [4] for quarterly
    seasonal_periods_raw = params.get("tbats_seasonal_periods", [12])
    if isinstance(seasonal_periods_raw, (int, float)):
        seasonal_periods_raw = [int(seasonal_periods_raw)]
    elif isinstance(seasonal_periods_raw, str):
        try:
            seasonal_periods_raw = [int(x.strip()) for x in seasonal_periods_raw.split(",")]
        except Exception:
            seasonal_periods_raw = [12]
    seasonal_periods = [int(x) for x in seasonal_periods_raw if int(x) > 1]
    if not seasonal_periods:
        seasonal_periods = None   # None = TBATS auto-detects or no seasonality

    # Manual flags (None = auto-select)
    use_box_cox      = params.get("tbats_use_box_cox",      None)
    use_trend        = params.get("tbats_use_trend",         None)
    use_damped_trend = params.get("tbats_use_damped_trend",  None)
    use_arma_errors  = params.get("tbats_use_arma_errors",   True)

    log("=" * 60)
    log(f"   auto_tbats         : {use_auto}")
    log(f"   seasonal_periods   : {seasonal_periods}")
    log(f"   use_box_cox        : {use_box_cox}")
    log(f"   use_trend          : {use_trend}")
    log(f"   use_damped_trend   : {use_damped_trend}")
    log(f"   use_arma_errors    : {use_arma_errors}")
    log("=" * 60)

    # ── Data preparation ─────────────────────────────────────────────
    df = data[[time_col, target_col]].copy()

    if pd.api.types.is_numeric_dtype(df[time_col]):
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].interpolate(method="linear").ffill().bfill()

    series = df[target_col].values.astype(float)

    # ── Train / test split ───────────────────────────────────────────
    split_idx    = int(split * len(df))
    train_series = series[:split_idx]
    test_series  = series[split_idx:]
    train_times  = df[time_col].iloc[:split_idx].values
    test_times   = df[time_col].iloc[split_idx:].values

    # ── AUTO-TBATS: grid over seasonal period candidates ─────────────
    if use_auto:
        

        log("\n🔍 Running Auto-TBATS (searching seasonal period candidates)...")

        if seasonal_periods is not None:
            base_candidates = [seasonal_periods]
            for sp in seasonal_periods:
                if [sp] not in base_candidates:
                    base_candidates.append([sp])
            base_candidates.append(None)
        else:
            base_candidates = [None, [12], [4], [7]]

        total_candidates = len(base_candidates)
        log(f"   📋 Will test {total_candidates} candidate(s): {base_candidates}")
        log(f"   ⏳ Each fit may take 30–90s. Live heartbeat shown every 5s.")

        best_aic   = np.inf
        best_sp    = seasonal_periods
        best_model = None

        for i, sp_candidate in enumerate(base_candidates, 1):
            log(f"\n   [{i}/{total_candidates}] 🔄 Fitting seasonal_periods={sp_candidate} ...")
            log(f"   ⚙️  TBATS internally searches: Box-Cox(on/off) × Trend(on/off) × "
                f"Damped(on/off) × ARMA(p,q) — all auto-selected by AIC")
            log(f"   ⏳ Running fit... (heartbeat every 5s)")

            # ── Run fit in background thread, emit heartbeats ─────────
            result_holder = [None]
            error_holder  = [None]
            done_flag     = threading.Event()

            def _fit_thread():
                try:
                    est = TBATSEstimator(
                        seasonal_periods  = sp_candidate,
                        use_box_cox       = use_box_cox,
                        use_trend         = use_trend,
                        use_damped_trend  = use_damped_trend,
                        use_arma_errors   = use_arma_errors,
                        n_jobs            = 1,
                    )
                    result_holder[0] = est.fit(train_series)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    done_flag.set()

            t = threading.Thread(target=_fit_thread, daemon=True)
            t.start()

            # ── Heartbeat loop ─────────────────────────────────────────
            elapsed = 0
            interval = 5   # seconds between heartbeat messages
            while not done_flag.wait(timeout=interval):
                elapsed += interval
                log(f"   ⏱  Still fitting seasonal_periods={sp_candidate} "
                    f"... {elapsed}s elapsed")

            t.join()

            if error_holder[0] is not None:
                log(f"   ⚠️  [{i}/{total_candidates}] FAILED: {error_holder[0]}")
                continue

            fit_tmp = result_holder[0]
            aic_val = float(fit_tmp.aic)

            # ── Log what TBATS actually chose ──────────────────────────
            c = fit_tmp.params.components
            p_ar = len(getattr(fit_tmp.params, 'ar_coeffs', None) or [])
            q_ma = len(getattr(fit_tmp.params, 'ma_coeffs', None) or [])
            log(f"   [{i}/{total_candidates}] ✅ Fit complete!")
            log(f"   📊 AIC = {aic_val:.4f}")
            log(f"   🔍 Chosen components:")
            log(f"      Box-Cox transform : {c.use_box_cox}")
            if c.use_box_cox and fit_tmp.params.box_cox_lambda is not None:
                log(f"      Box-Cox λ         : {round(float(fit_tmp.params.box_cox_lambda), 4)}")
            log(f"      Trend             : {c.use_trend}")
            log(f"      Damped trend      : {c.use_damped_trend}")
            log(f"      ARMA errors       : {c.use_arma_errors} → ARMA({p_ar},{q_ma})")
            sp_list = c.seasonal_periods if c.seasonal_periods is not None and len(c.seasonal_periods) > 0 else []
            if sp_list:
                for j, sp in enumerate(sp_list):
                    k = c.seasonal_harmonics[j] if c.seasonal_harmonics is not None else "?"
                    log(f"      Seasonal period {j+1} : {int(sp)} (harmonics k={k})")
            else:
                log(f"      Seasonal period   : None")

            if aic_val < best_aic:
                best_aic   = aic_val
                best_sp    = sp_candidate
                best_model = fit_tmp
                log(f"   🏆 New best! seasonal_periods={sp_candidate}  AIC={best_aic:.4f}")
            else:
                log(f"   ↩️  Not better than current best AIC={best_aic:.4f}, skipping.")

        if best_model is None:
            log("   ⚠️  All candidates failed — falling back to no seasonality.")
            estimator  = TBATSEstimator(seasonal_periods=None, n_jobs=1)
            best_model = estimator.fit(train_series)
            best_sp    = None

        tbats_fit_train  = best_model
        seasonal_periods = best_sp
        log(f"\n✅ Auto-TBATS complete! Best: seasonal_periods={seasonal_periods}, AIC={best_aic:.4f}")
    else:
        # Manual fit
        log("\n🔧 Fitting TBATS with manual configuration...")
        estimator = TBATSEstimator(
            seasonal_periods   = seasonal_periods,
            use_box_cox        = use_box_cox,
            use_trend          = use_trend,
            use_damped_trend   = use_damped_trend,
            use_arma_errors    = use_arma_errors,
            n_jobs             = 1,
        )
        tbats_fit_train = estimator.fit(train_series)

    # ── Model description string ──────────────────────────────────────
    model_label = _model_description(tbats_fit_train)
    log(f"   Model: {model_label}")

    # ── In-sample (train) fitted values ──────────────────────────────
    fitted_vals = tbats_fit_train.y_hat  # in-sample predictions
    train_pred  = np.array(fitted_vals)
    train_act   = train_series.copy()
    train_t     = train_times.copy()

    # ── Residual std for prediction intervals ─────────────────────────
    resid       = train_act - train_pred
    resid_std   = float(np.std(resid))

    # ── Multi-step forecast on test set (fast) ────────────────────────
    log("\n🔮 Generating multi-step forecast on test set...")
    n_test    = len(test_series)
    h_arr     = np.arange(1, n_test + 1)
    std_arr   = resid_std * np.sqrt(h_arr)

    try:
        pred_te = tbats_fit_train.forecast(steps=n_test)
    except Exception as e:
        log(f"   ⚠️  Multi-step forecast failed ({e}), using last train value.")
        pred_te = np.full(n_test, float(train_series[-1]))

    actual_te = test_series.copy()

    # ── Metrics ──────────────────────────────────────────────────────
    def mape(a, p_):
        return float(np.mean(np.abs((a - p_) / np.where(a == 0, 1e-8, a))) * 100)

    rmse_tr = float(root_mean_squared_error(train_act, train_pred))
    mae_tr  = float(mean_absolute_error(train_act,     train_pred))
    mape_tr = mape(train_act, train_pred)

    rmse_te = float(root_mean_squared_error(actual_te, pred_te))
    mae_te  = float(mean_absolute_error(actual_te,     pred_te))
    mape_te = mape(actual_te, pred_te)

    log(f"\n📊 Results:")
    log(f"   Train — RMSE: {rmse_tr:.3f}, MAE: {mae_tr:.3f}, MAPE: {mape_tr:.2f}%")
    log(f"   Test  — RMSE: {rmse_te:.3f}, MAE: {mae_te:.3f}, MAPE: {mape_te:.2f}%")

    # ── Parameter estimates & information criteria ───────────────────
    param_estimates_table = _param_estimates_df(tbats_fit_train)
    info_criteria         = _info_criteria(tbats_fit_train)

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
    log(f"   📊 Using best config: seasonal_periods={seasonal_periods}")
    log(f"   📏 Training on {len(series)} observations (full dataset)...")
    log(f"   ⏳ This is the final fit — please wait ~30–60s...")
    estimator_full = TBATSEstimator(
        seasonal_periods   = seasonal_periods,
        use_box_cox        = use_box_cox,
        use_trend          = use_trend,
        use_damped_trend   = use_damped_trend,
        use_arma_errors    = use_arma_errors,
        n_jobs             = 1,
    )
    result_holder2 = [None]
    done_flag2     = threading.Event()

    def _refit_thread():
        result_holder2[0] = estimator_full.fit(series)
        done_flag2.set()

    t2 = threading.Thread(target=_refit_thread, daemon=True)
    t2.start()
    elapsed2 = 0
    while not done_flag2.wait(timeout=5):
        elapsed2 += 5
        log(f"   ⏱  Refitting on full data... {elapsed2}s elapsed")
    t2.join()
    tbats_fit_full = result_holder2[0]
    resid_full_std = float(np.std(series - tbats_fit_full.y_hat))
    log(f"   ✅ Full-data fit complete!")
    log(f"   📉 Residual std (full): {resid_full_std:.4f}")

    # ── Save bundle ──────────────────────────────────────────────────
    os.makedirs("static", exist_ok=True)

    with open("static/tbats_model.pkl", "wb") as f:
        pickle.dump(tbats_fit_full, f)

    with open("static/tbats_bundle.pkl", "wb") as f:
        pickle.dump({
            "seasonal_periods"  : seasonal_periods,
            "use_box_cox"       : use_box_cox,
            "use_trend"         : use_trend,
            "use_damped_trend"  : use_damped_trend,
            "use_arma_errors"   : use_arma_errors,
            "last_time"         : last_t,
            "time_labels"       : future_time_labels,
            "time_col"          : time_col,
            "target_col"        : target_col,
            "resid_std"         : resid_full_std,
            "params"            : params,
        }, f)

    # ── Combined train + test plot ────────────────────────────────────
    train_x = list(range(len(train_act)))
    test_x  = list(range(len(train_act), len(train_act) + len(actual_te)))
    all_x   = train_x + test_x
    all_t   = list(train_t) + list(test_times)
    step    = max(1, len(all_x) // 12) if len(all_x) > 24 else 1

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train_x, train_act,  color="steelblue",  label="Actual (Train)")
    ax.plot(train_x, train_pred, color="steelblue",  label="Predicted (Train)",
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
    plt.savefig("static/tbats_train.png", dpi=120)
    plt.savefig("static/tbats_test.png",  dpi=120)
    plt.close()

    # ── Output tables ────────────────────────────────────────────────
    train_table = pd.DataFrame({
        time_col   : train_t,
        "Actual"   : np.round(train_act,  3),
        "Predicted": np.round(train_pred, 3),
    })

    test_table = pd.DataFrame({
        time_col   : test_times,
        "Actual"   : np.round(actual_te, 3),
        "Predicted": np.round(pred_te,   3),
    })
    log("\n✅ TBATS training complete!")
    log(f"   Model saved → static/tbats_model.pkl")
    log(f"   Bundle saved → static/tbats_bundle.pkl")
    log(f"   Train RMSE: {rmse_tr:.3f} | Test RMSE: {rmse_te:.3f}")
    log(f"   🎯 Ready for forecasting.")
    return {
        "model_key"   : "tbats",
        "model_name"  : model_label,
        "horizon"     : horizon,
        "frequency"   : frequency,
        "data_type"   : "Univariate (TBATS)",
        "model_config": {
            "Seasonal Periods"  : str(seasonal_periods) if seasonal_periods else "None",
            "Box-Cox Transform" : str(tbats_fit_train.params.components.use_box_cox),
            "Trend"             : str(tbats_fit_train.params.components.use_trend),
            "Damped Trend"      : str(tbats_fit_train.params.components.use_damped_trend),
            "ARMA Errors"       : str(tbats_fit_train.params.components.use_arma_errors),
        },
        "auto_tuned"  : use_auto,
        "order"       : model_label,
        "auto_tbats"  : use_auto,

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