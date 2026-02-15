import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# AUTO-TUNING FUNCTION WITH GRID SEARCH
# ============================================================
def auto_tune_ann(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    input_dim,
    params,  # ✅ ADD THIS
    max_epochs=100,
    log_callback=None
):
    """
    Auto-tune ANN using USER-DEFINED grid search ranges from params.
    """
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn

    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
    
    log("\n🔍 Starting ANN Grid Search Auto-Tuning (USER RANGES)")
    
    # ============================================================
    # READ USER-DEFINED RANGES - USE WHILE LOOP TO FIX FLOATING POINT
    # ============================================================
    hidden_layer_1_values = list(range(
        params["ann_hidden_layer_1_min"],
        params["ann_hidden_layer_1_max"] + 1,
        params["ann_hidden_layer_1_step"]
    ))
    
    hidden_layer_2_values = list(range(
        params["ann_hidden_layer_2_min"],
        params["ann_hidden_layer_2_max"] + 1,
        params["ann_hidden_layer_2_step"]
    ))
    
    # ✅ FIX: Use while loop to avoid floating point errors
    dropout_1_values = []
    val = params["ann_dropout_1_min"]
    while val <= params["ann_dropout_1_max"] + 1e-9:
        dropout_1_values.append(round(val, 3))
        val += params["ann_dropout_1_step"]
    
    dropout_2_values = []
    val = params["ann_dropout_2_min"]
    while val <= params["ann_dropout_2_max"] + 1e-9:
        dropout_2_values.append(round(val, 3))
        val += params["ann_dropout_2_step"]
    
    learning_rate_values = []
    val = params["ann_learning_rate_min"]
    while val <= params["ann_learning_rate_max"] + 1e-9:
        learning_rate_values.append(round(val, 6))
        val += params["ann_learning_rate_step"]
    
    weight_decay_values = []
    val = params["ann_weight_decay_min"]
    while val <= params["ann_weight_decay_max"] + 1e-9:
        weight_decay_values.append(round(val, 7))
        val += params["ann_weight_decay_step"]
    
    huber_delta_values = []
    val = params["ann_huber_delta_min"]
    while val <= params["ann_huber_delta_max"] + 1e-9:
        huber_delta_values.append(round(val, 2))
        val += params["ann_huber_delta_step"]
    
    patience_values = list(range(
        params["ann_patience_min"],
        params["ann_patience_max"] + 1,
        params["ann_patience_step"]
    ))
    
    # ============================================================
    # PARAMETER GRID
    # ============================================================
    param_grid = {
        'hidden_layer_1': hidden_layer_1_values,
        'hidden_layer_2': hidden_layer_2_values,
        'dropout_1': dropout_1_values,
        'dropout_2': dropout_2_values,
        'learning_rate': learning_rate_values,
        'weight_decay': weight_decay_values,
        'huber_delta': huber_delta_values,
        'patience': patience_values
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]
    
    total_combinations = len(combinations)
    
    # ✅ PRINT ALL PARAMETERS
    log("📦 Grid summary:")
    log(f"   hidden_layer_1: {hidden_layer_1_values}")
    log(f"   hidden_layer_2: {hidden_layer_2_values}")
    log(f"   dropout_1: {dropout_1_values}")
    log(f"   dropout_2: {dropout_2_values}")
    log(f"   learning_rate: {learning_rate_values}")
    log(f"   weight_decay: {weight_decay_values}")
    log(f"   huber_delta: {huber_delta_values}")
    log(f"   patience: {patience_values}")
    log(f"🔢 Total combinations: {total_combinations}")
    log("⏳ This may take a while...\n")
    
    best_val_loss = float('inf')
    best_params = None
    
    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    # Test each combination
    for idx, p in enumerate(combinations, 1):
        if idx % 100 == 0:
            log(f"   Progress: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%)")
        
        try:
            # Create model
            model = ANNModel(
                input_dim=input_dim,
                hidden_layer_1=p['hidden_layer_1'],
                hidden_layer_2=p['hidden_layer_2'],
                dropout_1=p['dropout_1'],
                dropout_2=p['dropout_2']
            ).to(device)
            
            # Setup training
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=p['learning_rate'],
                weight_decay=p['weight_decay']
            )
            
            loss_fn = nn.HuberLoss(delta=p['huber_delta'])
            
            # Training loop with early stopping
            best_epoch_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(max_epochs):
                # Train
                model.train()
                optimizer.zero_grad()
                preds = model(X_train_t)
                loss = loss_fn(preds, y_train_t)
                loss.backward()
                optimizer.step()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val_t)
                    val_loss = loss_fn(val_preds, y_val_t).item()
                
                # Early stopping
                if val_loss < best_epoch_loss:
                    best_epoch_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= p['patience']:
                        break
            
            # Update best if better
            if best_epoch_loss < best_val_loss:
                best_val_loss = best_epoch_loss
                best_params = p.copy()
                log(f"   ✨ New best! Val Loss: {best_val_loss:.6f} at combination {idx}/{total_combinations}")
        
        except Exception as e:
            continue
    
    log(f"✅ Grid Search Complete!")
    log(f"   Best Parameters: {best_params}")
    log(f"   Best Validation Loss: {best_val_loss:.6f}")
    
    return best_params


#ANN Architecture
class ANNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_1,
        hidden_layer_2,
        dropout_1,
        dropout_2
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_1),
            nn.ReLU(),                      # activation = ReLU
            nn.Dropout(dropout_1),

            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),

            nn.Linear(hidden_layer_2, 1)    # output_dim = 1
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def _forecast_only_ann(
    params,
    horizon,
    future_rain,
    future_mean_t,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torch.load("static/ann_bundle.pt", weights_only=False)
    FEATURES = bundle["features"]
    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    model = ANNModel(
    input_dim=bundle["input_dim"],
    hidden_layer_1=bundle["hidden_layer_1"],  # ✅ CORRECT
    hidden_layer_2=bundle["hidden_layer_2"],  # ✅ CORRECT
    dropout_1=bundle["dropout_1"],            # ✅ CORRECT
    dropout_2=bundle["dropout_2"],            # ✅ CORRECT
).to(device)
    

    model.load_state_dict(
        torch.load("static/ann_weights.pt", map_location=device, weights_only=False)
    )
    model.eval()

    last_window = bundle["last_window"]
    scaler_mean = bundle["scaler_mean"]
    scaler_scale = bundle["scaler_scale"]

    future_rain_scaled = (
        np.array(future_rain) - scaler_mean[feat_idx["rain_lag_1"]]
    ) / scaler_scale[feat_idx["rain_lag_1"]]

    future_mean_t_scaled = (
        np.array(future_mean_t) - scaler_mean[feat_idx["mean_t_lag_1"]]
    ) / scaler_scale[feat_idx["mean_t_lag_1"]]

    future_log_diffs = []

    with torch.no_grad():
        for h in range(horizon):

            x_flat = torch.tensor(
                last_window.reshape(1, -1),
                dtype=torch.float32,
                device=device
            )

            pred = model(x_flat).item()
            future_log_diffs.append(pred)

            last_window = np.roll(last_window, -1, axis=0)
            last_window[-1, feat_idx["log_diff_lag_1"]] = pred
            last_window[-1, feat_idx["rain_lag_1"]] = future_rain_scaled[h]
            last_window[-1, feat_idx["mean_t_lag_1"]] = future_mean_t_scaled[h]

            if "time_lag_1" in feat_idx:
                last_window[-1, feat_idx["time_lag_1"]] += 1

    future_log = np.cumsum(future_log_diffs) + bundle["last_log"]
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

#MAIN FUNCTION
def run_ann(
    data: pd.DataFrame,
    params: dict,
    horizon: int,
    frequency: str,
    log_callback=None,
):

    """
    Website-ready ANN model (flattened sliding window)
    Input  : DataFrame with columns [year, mean_t, rain, tobaco]
    Output : dict (metrics + train/test tables)
    """
    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    if data is None:
        return _forecast_only_ann(
            params=params,
            horizon=horizon,
            future_rain=params["future_rain"],
            future_mean_t=params["future_mean_t"],
    )

#MANUAL HYPERPARAMETER BLOCK
    # ===============================
    # ANN (MLP) HYPERPARAMETERS
    # ===============================

    # Time-series / data
    WINDOW = params.get("window", 4)
    LAGS = params.get("lags", 3)
    forecast_horizon = horizon
    split = params.get("split", 0.85)


    # Check if auto-tuning is enabled
    use_auto_tune = params.get("auto_tune_ann", True)
    # Model architecture
    hidden_layer_1 = params.get("hidden_layer_1", 16)
    hidden_layer_2 = params.get("hidden_layer_2", 8)
    dropout_1 = params.get("dropout_1", 0.2)
    dropout_2 = params.get("dropout_2", 0.3)

    # Training
    lr = params.get("learning_rate", 0.002)
    weight_decay = params.get("weight_decay", 1e-4)
    num_epochs = params.get("epochs", 150)
    patience = params.get("patience", 30)

    # Optimization
    huber_delta = params.get("huber_delta", 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
#Data Cleaning
    data = data.rename(columns={
        "Year": "year",
        "Mean_T": "mean_t",
        "Mean_Rain": "rain",
        "Yield": "yield"
    })

    data = data[["year", "mean_t", "rain", "yield"]]
    data = data.sort_values("year").reset_index(drop=True)
    data = data.interpolate(method="linear").dropna()

# Log-Difference Target (MISSING FIX)
    data["log_yield"] = np.log(data["yield"])
    data["log_diff"] = data["log_yield"].diff()
    data["time_idx"] = np.arange(len(data))
    data = data.dropna().reset_index(drop=True)

#Explicit Lags
    
    for lag in range(1, LAGS + 1):
        data[f"log_diff_lag_{lag}"] = data["log_diff"].shift(lag)
        data[f"rain_lag_{lag}"] = data["rain"].shift(lag)
        data[f"mean_t_lag_{lag}"] = data["mean_t"].shift(lag)
        data[f"time_lag_{lag}"] = data["time_idx"].shift(lag)

    data = data.dropna().reset_index(drop=True)


    # Define features programmatically
    FEATURES = [c for c in data.columns if "lag_" in c]

#Sliding Window
    

    def make_sequences(df):
        X, y = [], []
        for i in range(len(df) - WINDOW):
            X.append(df[FEATURES].iloc[i:i+WINDOW].values)
            y.append(df["log_diff"].iloc[i+WINDOW])
        return np.array(X), np.array(y)

    X, y = make_sequences(data)
    input_dim = X.shape[1] * X.shape[2]

#Train–Test Split
    split_idx = int(split * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ✅ AUTO-TUNE (if enabled)
    if use_auto_tune:
        log("\n🔧 Auto-tuning enabled - searching for optimal hyperparameters...")
        
        # Create validation set from training data (80-20 split of training)
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
    
        # Flatten
        X_train_tune_flat = X_train_tune_scaled.reshape(X_train_tune_scaled.shape[0], -1)
        X_val_tune_flat = X_val_tune_scaled.reshape(X_val_tune_scaled.shape[0], -1)
    
        best_params_ann = auto_tune_ann(
            X_train_tune_flat, y_train_tune,
            X_val_tune_flat, y_val_tune,
            device, X_train_tune_flat.shape[1],
            params, #✅ ADD THIS
            max_epochs=100,
            log_callback=log_callback
        )
    
        # Override with best parameters
        hidden_layer_1 = best_params_ann['hidden_layer_1']
        hidden_layer_2 = best_params_ann['hidden_layer_2']
        dropout_1 = best_params_ann['dropout_1']
        dropout_2 = best_params_ann['dropout_2']
        lr = best_params_ann['learning_rate']  # ✅ CORRECT KEY
        weight_decay = best_params_ann['weight_decay']
        huber_delta = best_params_ann['huber_delta']
        patience = best_params_ann['patience']
        
        log(f"✅ Using optimized ANN parameters")

#Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

#Flatten for ANN
    X_train_ann = X_train.reshape(X_train.shape[0], -1)
    X_test_ann  = X_test.reshape(X_test.shape[0], -1)

#Torch Setup
    

    X_train_ann = torch.tensor(X_train_ann, dtype=torch.float32).to(device)
    X_test_ann  = torch.tensor(X_test_ann,  dtype=torch.float32).to(device)
    y_train     = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test      = torch.tensor(y_test,  dtype=torch.float32).to(device)

#Training
    model = ANNModel(
        input_dim=input_dim,
        hidden_layer_1=hidden_layer_1,
        hidden_layer_2=hidden_layer_2,
        dropout_1=dropout_1,
        dropout_2=dropout_2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    loss_fn = nn.HuberLoss(delta=huber_delta)

    best_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_ann)
        loss = loss_fn(preds, y_train)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)

#Prediction & Reconstruction
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_ann).cpu().numpy()
        test_preds = model(X_test_ann).cpu().numpy()

    # Training reconstruction
    last_log_train = data["log_yield"].iloc[WINDOW - 1]
    pred_log_train = np.cumsum(train_preds) + last_log_train
    pred_train = np.exp(pred_log_train)

    actual_train = data["yield"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values

    years_train = data["year"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values

    # Testing reconstruction
    start_idx = split_idx + WINDOW
    last_log_test = data["log_yield"].iloc[start_idx - 1]

    
    pred_log_test = np.cumsum(test_preds) + last_log_test
    pred_test = np.exp(pred_log_test)

    actual_test = np.exp(
        np.cumsum(y_test.cpu().numpy()) + last_log_test
    )

    years_test = data["year"].iloc[
        start_idx:start_idx + len(actual_test)
    ].values

#Metrics
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

    import os
    os.makedirs("static", exist_ok=True)

    torch.save(model.state_dict(), "static/ann_weights.pt")

    torch.save({
    "last_window": X_test[-1],  # ✅ Already numpy, save directly
    "features": FEATURES,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "last_log": np.log(pred_test[-1]),
    "last_year": int(data["year"].iloc[-1]),
    
    # ✅ SAVE MODEL ARCHITECTURE (critical for loading)
    "input_dim": input_dim,
    "hidden_layer_1": hidden_layer_1,
    "hidden_layer_2": hidden_layer_2,
    "dropout_1": dropout_1,
    "dropout_2": dropout_2,
    
    "params": params,
}, "static/ann_bundle.pt")

    import matplotlib.pyplot as plt

# TRAIN
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_train, actual_train, label="Actual (Train)")
    ax.plot(years_train, pred_train, "--o", label="Predicted (Train)", markersize=4)
    ax.set_title("ANN Model - Training Set")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig("static/ann_train.png", dpi=150)
    plt.close(fig)

# TEST
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_test, actual_test, label="Actual (Test)")
    ax.plot(years_test, pred_test, "--o", label="Predicted (Test)", markersize=4)
    ax.set_title("ANN Model - Testing Set")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield")
    ax.legend()
    fig.tight_layout()
    fig.savefig("static/ann_test.png", dpi=150)
    plt.close(fig)

#RETURN
    return {
        "model_name": "ANN",
        "model_key": "ann",
        "horizon": horizon,
        
        "frequency": frequency,
        # ✅ ADD MODEL CONFIGURATION
        "model_config": {
            "hidden_layer_1": hidden_layer_1,
            "hidden_layer_2": hidden_layer_2,
            "dropout_1": round(dropout_1, 3),
            "dropout_2": round(dropout_2, 3),
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "huber_delta": round(huber_delta,2),
            "epochs_used": num_epochs,
            "patience": patience,
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
        "forecast_table": None #Always None during training - only populated in forecast mode
    }