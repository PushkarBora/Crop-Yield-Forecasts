import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def generate_param_range(min_val, max_val, step):
    """
    Generate a list of values from min to max with given step.
    Works for both integers and floats.
    """
    import numpy as np
    if isinstance(step, int) and isinstance(min_val, int):
        return list(range(min_val, max_val + 1, step))
    else:
        return list(np.arange(min_val, max_val + step, step))

# ============================================================
# AUTO-TUNING FUNCTION WITH USER-DEFINED GRID SEARCH
# ============================================================
def auto_tune_svr(X_train, y_train, X_val, y_val, params,log_callback=None):
    """
    Auto-tune SVR using USER-DEFINED grid search ranges from params.
    """
    from sklearn.model_selection import GridSearchCV
    import numpy as np
    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    log("\n🔍 Starting SVR Grid Search Auto-Tuning (USER RANGES)")
    
    # ============================================================
    # READ USER-DEFINED RANGES - USE WHILE LOOP TO FIX FLOATING POINT
    # ============================================================
    C_values = []
    val = params["svr_C_min"]
    while val <= params["svr_C_max"] + 1e-9:
        C_values.append(round(val, 3))
        val += params["svr_C_step"]
    
    epsilon_values = []
    val = params["svr_epsilon_min"]
    while val <= params["svr_epsilon_max"] + 1e-9:
        epsilon_values.append(round(val, 3))
        val += params["svr_epsilon_step"]
    
    # Gamma: numeric values + 'scale' and 'auto'
    gamma_numeric_values = []
    val = params["svr_gamma_min"]
    while val <= params["svr_gamma_max"] + 1e-9:
        gamma_numeric_values.append(round(val, 3))
        val += params["svr_gamma_step"]
    
    gamma_values = ['scale', 'auto'] + gamma_numeric_values
    
    # ✅ READ KERNEL SELECTIONS FROM USER CHECKBOXES
    kernel_options = []
    if params.get('svr_kernel_rbf', False):
        kernel_options.append('rbf')
    if params.get('svr_kernel_poly', False):
        kernel_options.append('poly')
    if params.get('svr_kernel_sigmoid', False):
        kernel_options.append('sigmoid')
    
    # ✅ VALIDATE: At least one kernel must be selected
    if not kernel_options:
        log("⚠️  WARNING: No kernels selected, defaulting to RBF")
        kernel_options = ['rbf']
    # ============================================================
    # PARAMETER GRID
    # ============================================================
    param_grid = {
        'kernel': kernel_options,
        'C': C_values,
        'epsilon': epsilon_values,
        'gamma': gamma_values
    }
    
    # Calculate total combinations
    total_combinations = (
        len(kernel_options) * 
        len(C_values) * 
        len(epsilon_values) * 
        len(gamma_values)
    )
    
    # ✅ PRINT ALL PARAMETERS
    log("📦 Grid summary:")
    log(f"   kernel: {kernel_options}")
    log(f"   C: {C_values}")
    log(f"   epsilon: {epsilon_values}")
    log(f"   gamma: {gamma_values}")
    log(f"🔢 Total combinations: {total_combinations}")
    log("⏳ This may take a while...\n")
    
    # Create base model
    base_model = SVR()
    
    # GridSearchCV - tests ALL combinations
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    # Fit on training data
    log("   Running grid search with 3-fold cross-validation...")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    
    log(f"\n✅ Grid Search Complete!")
    log(f"   Best Parameters: {best_params}")
    log(f"   Best CV MSE: {best_score:.4f}")
    
    # Validate on validation set
    best_model = grid_search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, val_pred)
    
    log(f"   Validation RMSE: {val_rmse:.4f}\n")
    
    return best_params


# ============================================================
# FORECAST-ONLY (NO TRAINING, NO PLOTS)
# ============================================================
def _forecast_only_svr(
    params,
    horizon,
    future_rain,
    future_mean_t,
):
    # Load saved bundle
    if not os.path.exists("static/svr_bundle.pkl"):
        raise FileNotFoundError(
            "SVR model not trained yet. Please train the model first before forecasting."
        )
    
    with open("static/svr_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    
    # Load model
    with open("static/svr_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    FEATURES = bundle["features"]
    feat_idx = {f: i for i, f in enumerate(FEATURES)}
    WINDOW = bundle["window"]
    
    # Load last window and scaler
    last_window = bundle["last_window"]
    scaler_mean = bundle["scaler_mean"]
    scaler_scale = bundle["scaler_scale"]
    
    # Scale future inputs
    future_rain_scaled = (
        np.array(future_rain) - scaler_mean[feat_idx["rain_lag_1"]]
    ) / scaler_scale[feat_idx["rain_lag_1"]]
    
    future_mean_t_scaled = (
        np.array(future_mean_t) - scaler_mean[feat_idx["mean_t_lag_1"]]
    ) / scaler_scale[feat_idx["mean_t_lag_1"]]
    
    # Forecast loop
    future_log_diffs = []
    
    for h in range(horizon):
        # Flatten window for SVR
        x_flat = last_window.reshape(1, -1)
        
        # Predict
        pred = model.predict(x_flat)[0]
        future_log_diffs.append(pred)
        
        # Shift window
        last_window = np.roll(last_window, -1, axis=0)
        
        # Update with new values
        last_window[-1, feat_idx["log_diff_lag_1"]] = pred
        last_window[-1, feat_idx["rain_lag_1"]] = future_rain_scaled[h]
        last_window[-1, feat_idx["mean_t_lag_1"]] = future_mean_t_scaled[h]
        
        if "time_lag_1" in feat_idx:
            last_window[-1, feat_idx["time_lag_1"]] += 1
    
    # Reconstruct yield
    last_log = bundle["last_log"]
    future_log = np.cumsum(future_log_diffs) + last_log
    future_yield = np.exp(future_log)
    
    future_years = np.arange(
        bundle["last_year"] + 1,
        bundle["last_year"] + 1 + horizon
    )
    
    return {
        "forecast_table": pd.DataFrame({
            "Year": future_years,
            "Forecast": future_yield
        })
    }


# ============================================================
# MAIN ENTRY FUNCTION
# ============================================================
def run_svr(
    data: pd.DataFrame,
    params: dict,
    horizon: int,
    frequency: str,
    future_rain=None,
    future_mean_t=None,
    mode: str = "train",
    log_callback=None,
):
    """
    Website-ready SVR model (flattened sliding window)
    Input  : DataFrame with columns [year, mean_t, rain, yield]
    Output : dict (metrics + train/test tables)
    """
    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    # Forecast mode
    if mode == "forecast":
        return _forecast_only_svr(
            params=params,
            horizon=horizon,
            future_rain=future_rain,
            future_mean_t=future_mean_t,
        )
    
    # Validate training mode
    if mode == "train" and data is None:
        raise ValueError(
            "Training SVR requires historical data, but data=None was received."
        )
    
    # ===============================
    # HYPERPARAMETERS
    # ===============================
    WINDOW = params.get("window", 4)
    LAGS = params.get("lags", 3)
    split = params.get("split", 0.85)
    
    # Check if auto-tuning is enabled
    use_auto_tune = params.get("auto_tune_svr", True)
    
    # Default SVR parameters (used if auto-tune disabled)
    kernel = params.get("kernel", "rbf")
    C = params.get("C", 1.0)
    epsilon = params.get("epsilon", 0.1)
    gamma = params.get("gamma", "scale")
    
    # ===============================
    # DATA CLEANING
    # ===============================
    data = data.rename(columns={
        "Year": "year",
        "Mean_T": "mean_t",
        "Mean_Rain": "rain",
        "Yield": "yield"
    })
    
    data = data[["year", "mean_t", "rain", "yield"]]
    data = data.sort_values("year").reset_index(drop=True)
    data = data.interpolate(method="linear").dropna()
    
    # Log-Difference Target
    data["log_yield"] = np.log(data["yield"])
    data["log_diff"] = data["log_yield"].diff()
    data["time_idx"] = np.arange(len(data))
    data = data.dropna().reset_index(drop=True)
    
    # Explicit Lags
    for lag in range(1, LAGS + 1):
        data[f"log_diff_lag_{lag}"] = data["log_diff"].shift(lag)
        data[f"rain_lag_{lag}"] = data["rain"].shift(lag)
        data[f"mean_t_lag_{lag}"] = data["mean_t"].shift(lag)
        data[f"time_lag_{lag}"] = data["time_idx"].shift(lag)
    
    data = data.dropna().reset_index(drop=True)
    
    # Feature Set
    FEATURES = [c for c in data.columns if "lag_" in c]
    
    # ===============================
    # SLIDING WINDOW
    # ===============================
    def make_sequences(df):
        X, y = [], []
        for i in range(len(df) - WINDOW):
            X.append(df[FEATURES].iloc[i:i+WINDOW].values)
            y.append(df["log_diff"].iloc[i+WINDOW])
        return np.array(X), np.array(y)
    
    X, y = make_sequences(data)
    
    # Train–Test Split
    split_idx = int(split * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # ✅ AUTO-TUNE (if enabled)
    if use_auto_tune:
        log("\n🔧 Auto-tuning enabled - searching for optimal hyperparameters...")
        
        # Create validation set from training data (80-20 split)
        val_split = int(0.8 * len(X_train))
        X_train_tune = X_train[:val_split]
        y_train_tune = y_train[:val_split]
        X_val_tune = X_train[val_split:]
        y_val_tune = y_train[val_split:]
        
        # Scale for tuning
        scaler_tune = StandardScaler()
        X_train_tune_scaled = scaler_tune.fit_transform(
            X_train_tune.reshape(-1, X_train_tune.shape[-1])
        ).reshape(X_train_tune.shape)
        X_val_tune_scaled = scaler_tune.transform(
            X_val_tune.reshape(-1, X_val_tune.shape[-1])
        ).reshape(X_val_tune.shape)
        
        # Flatten for SVR
        X_train_tune_flat = X_train_tune_scaled.reshape(X_train_tune_scaled.shape[0], -1)
        X_val_tune_flat = X_val_tune_scaled.reshape(X_val_tune_scaled.shape[0], -1)
        
        best_params_svr = auto_tune_svr(
            X_train_tune_flat, y_train_tune,
            X_val_tune_flat, y_val_tune,
            params , # ✅ Pass params dict
            log_callback=log_callback
        )
        
        # Override with best parameters
        kernel = best_params_svr['kernel']
        C = best_params_svr['C']
        epsilon = best_params_svr['epsilon']
        gamma = best_params_svr['gamma']
        
        log(f"✅ Using optimized SVR parameters")
    
    # ===============================
    # SCALING (FULL TRAINING SET)
    # ===============================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)
    
    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)
    
    # Flatten for SVR
    X_train_svr = X_train.reshape(X_train.shape[0], -1)
    X_test_svr  = X_test.reshape(X_test.shape[0], -1)
    
    # ===============================
    # FIT SVR
    # ===============================
    svr = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma
    )
    
    svr.fit(X_train_svr, y_train)
    
    # ===============================
    # PREDICTION & RECONSTRUCTION
    # ===============================
    train_preds = svr.predict(X_train_svr)
    test_preds  = svr.predict(X_test_svr)
    
    # Training Reconstruction
    last_log_train = data["log_yield"].iloc[WINDOW - 1]
    pred_log_train = np.cumsum(train_preds) + last_log_train
    pred_train = np.exp(pred_log_train)
    
    actual_train = data["yield"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values
    
    years_train = data["year"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values
    
    # Test Reconstruction
    start_idx = split_idx + WINDOW
    last_log_test = data["log_yield"].iloc[start_idx - 1]
    
    pred_log_test = np.cumsum(test_preds) + last_log_test
    pred_test = np.exp(pred_log_test)
    
    actual_test = np.exp(
        np.cumsum(y_test[:len(test_preds)]) + last_log_test
    )
    
    years_test = data["year"].iloc[
        start_idx:start_idx + len(actual_test)
    ].values
    
    # ===============================
    # METRICS
    # ===============================
    rmse_train = root_mean_squared_error(actual_train, pred_train)
    r2_train = r2_score(actual_train, pred_train)
    mape_train = np.mean(
        np.abs((actual_train - pred_train) / actual_train)
    ) * 100
    
    rmse_test = root_mean_squared_error(actual_test, pred_test)
    r2_test = r2_score(actual_test, pred_test)
    mape_test = np.mean(
        np.abs((actual_test - pred_test) / actual_test)
    ) * 100
    
    # ===============================
    # SAVE MODEL AND BUNDLE (ONLY IN TRAIN MODE)
    # ===============================
    if mode == "train":
        import matplotlib.pyplot as plt
        
        os.makedirs("static", exist_ok=True)
        
        # Save model
        with open("static/svr_model.pkl", "wb") as f:
            pickle.dump(svr, f)
        
        # Save bundle
        with open("static/svr_bundle.pkl", "wb") as f:
            pickle.dump({
                "last_window": X_test[-1],
                "features": FEATURES,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "last_log": np.log(pred_test[-1]),
                "last_year": int(data["year"].iloc[-1]),
                "window": WINDOW,
                "params": params,
            }, f)
        
        # Training plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years_train, actual_train, label="Actual (Train)")
        ax.plot(
            years_train,
            pred_train,
            label="Predicted (Train)",
            linestyle="--",
            marker="o",
            markersize=4
        )
        ax.set_title("SVR Model - Training Set")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield")
        ax.legend()
        fig.tight_layout()
        fig.savefig("static/svr_train.png", dpi=150)
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
        ax.set_title("SVR Model - Testing Set")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield")
        ax.legend()
        fig.tight_layout()
        fig.savefig("static/svr_test.png", dpi=150)
        plt.close(fig)
    
    # ===============================
    # RETURN
    # ===============================
    return {
        "model_key": "svr",
        "model_name": "SVR",
        "horizon": horizon,
        "frequency": frequency,
        
        # ✅ MODEL CONFIGURATION
        "model_config": {
            "kernel": kernel,
            "C": C,
            "epsilon": epsilon,
            "gamma": gamma if isinstance(gamma, str) else round(gamma, 3),
        },
        "auto_tuned": use_auto_tune,
        
        "rmse_train": round(rmse_train, 3),
        "r2_train": round(r2_train, 3),
        "mape_train": round(mape_train, 2),
        
        "rmse_test": round(rmse_test, 3),
        "r2_test": round(r2_test, 3),
        "mape_test": round(mape_test, 2),
        
        "train_table": pd.DataFrame({
            "Year": years_train,
            "Actual": actual_train,
            "Predicted": pred_train
        }),
        
        "test_table": pd.DataFrame({
            "Year": years_test,
            "Actual": actual_test,
            "Predicted": pred_test
        }),
        
        "forecast_table": None
    }
