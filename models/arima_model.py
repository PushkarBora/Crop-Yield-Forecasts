import numpy as np
import pandas as pd
import os
import pickle
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error, r2_score

# ============================================================
# FORECAST-ONLY (NO TRAINING, NO PLOTS)
# ============================================================
def _forecast_only_arima(
    params,
    horizon,
):
    # Load saved bundle
    if not os.path.exists("static/arima_bundle.pkl"):
        raise FileNotFoundError(
            "ARIMA model not trained yet. Please train the model first before forecasting."
        )
    
    with open("static/arima_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    # Load fitted model
    with open("static/arima_model.pkl", "rb") as f:
        arima_fit = pickle.load(f)
    
    # Forecast future values
    forecast_log = arima_fit.forecast(steps=horizon)
    forecast_yield = np.exp(forecast_log.values)
    
    future_years = np.arange(
        bundle["last_year"] + 1,
        bundle["last_year"] + 1 + horizon
    )
    
    return {
        "forecast_table": pd.DataFrame({
            "Year": future_years,
            "Forecast": forecast_yield
        })
    }


# Main Entry Function
def run_arima(
    data: pd.DataFrame,
    params: dict,
    horizon: int,
    frequency: str,
    mode: str = "train",
    log_callback=None
):
    """
    Website-ready ARIMA model (UNIVARIATE)
    Input  : DataFrame with columns [year, yield]
    Output : dict (metrics + train/test tables)
    """
    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # Forecast mode
    if mode == "forecast":
        return _forecast_only_arima(
            params=params,
            horizon=horizon,
        )
    
    # Validate training mode
    if mode == "train" and data is None:
        raise ValueError(
            "Training ARIMA requires historical data, but data=None was received."
        )
    
    # HYPERPARAMETERS
    split = params.get("split", 0.85)
    
    # ARIMA order (manual override allowed)
    use_auto_arima = params.get("auto_arima", False)
    # Add this right after line 74 in arima_model.py
    log("=" * 80)
    log("🔍 ARIMA PARAMETER DEBUG")
    log("=" * 80)
    log(f"use_auto_arima: {use_auto_arima}")
    log(f"Manual p: {params.get('arima_p')}")
    log(f"Manual d: {params.get('arima_d')}")
    log(f"Manual q: {params.get('arima_q')}")
    log("=" * 80)
    
    p = params.get("arima_p", 1)
    d = params.get("arima_d", 1)
    q = params.get("arima_q", 0)
    
    # Data Preparation
    data = data.rename(columns={
        "Year": "year",
        "Yield": "yield"
    })
    
    data = data[["year", "yield"]]
    data = data.sort_values("year").reset_index(drop=True)
    data["yield"] = data["yield"].interpolate(method="linear")
    
    # Log Transform
    data["log_yield"] = np.log(data["yield"])
    
    # Train–Test Split
    split_idx = int(split * len(data))
    
    train_log = data["log_yield"].iloc[:split_idx]
    test_log  = data["log_yield"].iloc[split_idx:]
    
    years_train = data["year"].iloc[:split_idx]
    years_test  = data["year"].iloc[split_idx:]
    
    # AUTO-ARIMA (DEFAULT) OR MANUAL ARIMA
    if use_auto_arima:
        log("🔍 Running Auto-ARIMA grid search...")  # ✅ ADD THIS
        # Conservative auto-selection
        best_aic = np.inf
        best_order = None
        
        for p_try in range(0, 3):
            for q_try in range(0, 3):
                try:
                    log(f"  → Testing ARIMA({p_try},{d},{q_try})...")  # ✅ ADD THIS
                    model = ARIMA(train_log, order=(p_try, d, q_try))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p_try, d, q_try)
                        log(f"  ✨ New best: AIC={best_aic:.2f}")  # ✅ ADD THIS
                except:
                    continue
        
        if best_order is not None:
            p, d, q = best_order
            log(f"✅ Best order found: ({p},{d},{q})")  # ✅ ADD THIS
    # Final ARIMA fit on TRAINING data
    model = ARIMA(
        train_log,
        order=(p, d, q),
        trend="n"
    )
    
    arima_fit = model.fit()
    
    # In-Sample (Training) Prediction
    fitted_log = arima_fit.fittedvalues
    
    # Align due to differencing
    train_pred = np.exp(fitted_log.iloc[1:].values)
    train_actual = np.exp(train_log.iloc[1:].values)
    train_years = years_train.iloc[1:].values
    
    # ✅ FIX: Rolling forecast on Test Period (one-step-ahead)
    test_predictions = []
    history = train_log.copy()
    
    for i in range(len(test_log)):
        # Fit model on history
        model_temp = ARIMA(history, order=(p, d, q), trend="n")
        fit_temp = model_temp.fit()
        
        # One-step forecast
        forecast_val = fit_temp.forecast(steps=1).iloc[0]
        test_predictions.append(forecast_val)
        
        # Add actual observation to history
        history = pd.concat([history, test_log.iloc[i:i+1]])
    
    pred_test = np.exp(test_predictions)
    actual_test = np.exp(test_log.values)
    
    # Accuracy Metrics
    rmse_train = root_mean_squared_error(train_actual, train_pred)
    r2_train = r2_score(train_actual, train_pred)
    mape_train = np.mean(
        np.abs((train_actual - train_pred) / train_actual)
    ) * 100
    
    rmse_test = root_mean_squared_error(actual_test, pred_test)
    r2_test = r2_score(actual_test, pred_test)
    mape_test = np.mean(
        np.abs((actual_test - pred_test) / actual_test)
    ) * 100
    
    # Save model and bundle (ONLY IN TRAIN MODE)
    if mode == "train":
        import matplotlib.pyplot as plt
        
        os.makedirs("static", exist_ok=True)
        
        # ✅ FIX: Refit on FULL dataset for future forecasting
        full_log = data["log_yield"]
        model_full = ARIMA(full_log, order=(p, d, q), trend="n")
        arima_fit_full = model_full.fit()
        
        # Save the FULL model
        with open("static/arima_model.pkl", "wb") as f:
            pickle.dump(arima_fit_full, f)
        
        # Save bundle
        with open("static/arima_bundle.pkl", "wb") as f:
            pickle.dump({
                "last_year": int(data["year"].iloc[-1]),
                "order": (p, d, q),
                "params": params,
            }, f)
        
        # Training plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_years, train_actual, label="Actual (Train)")
        ax.plot(
            train_years,
            train_pred,
            label="Predicted (Train)",
            linestyle="--",
            marker="o",
            markersize=4
        )
        ax.set_title(f"ARIMA({p},{d},{q}) Model - Training Set")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield")
        ax.legend()
        fig.tight_layout()
        fig.savefig("static/arima_train.png", dpi=150)
        plt.close(fig)
        
        # Testing plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years_test, actual_test, label="Actual (Test)")
        ax.plot(
            years_test,
            pred_test,
            label="Predicted (Test)",
            linestyle="--",
            marker="o",
            markersize=4
        )
        ax.set_title(f"ARIMA({p},{d},{q}) Model - Testing Set")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield")
        ax.legend()
        fig.tight_layout()
        fig.savefig("static/arima_test.png", dpi=150)
        plt.close(fig)
    
    # OUTPUT TABLES
    train_table = pd.DataFrame({
        "Year": train_years,
        "Actual": train_actual,
        "Predicted": train_pred
    })
    
    test_table = pd.DataFrame({
        "Year": years_test.values,
        "Actual": actual_test,
        "Predicted": pred_test
    })
    
    # RETURN
    return {
        "model_key": "arima",
        "model_name": f"ARIMA({p},{d},{q})",
        "horizon": horizon,
        "frequency": frequency,
        
        # ✅ ADD MODEL CONFIGURATION
        "model_config": {
            "p (AR Order)": p,
            "d (Differencing)": d,
            "q (MA Order)": q,
    },
    "auto_tuned": use_auto_arima,
    
        "order": f"ARIMA({p},{d},{q})",
        "auto_arima": use_auto_arima,
        
        "rmse_train": round(rmse_train, 3),
        "r2_train": round(r2_train, 3),
        "mape_train": round(mape_train, 2),
        
        "rmse_test": round(rmse_test, 3),
        "r2_test": round(r2_test, 3),
        "mape_test": round(mape_test, 2),
        
        "train_table": train_table,
        "test_table": test_table,
        
        "forecast_table": None
    }