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
def auto_tune_gru(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    n_features,
    params,  # ✅ ADD THIS
    max_epochs=100,
    log_callback=None
):
    """
    Auto-tune GRU using USER-DEFINED grid search ranges from params.
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

    log("\n🔍 Starting GRU Grid Search Auto-Tuning (USER RANGES)")
    
    # ============================================================
    # READ USER-DEFINED RANGES - USE WHILE LOOP TO FIX FLOATING POINT
    # ============================================================
    hidden_size_values = list(range(
        params["gru_hidden_size_min"],
        params["gru_hidden_size_max"] + 1,
        params["gru_hidden_size_step"]
    ))
    
    num_layers_values = list(range(
        params["gru_num_layers_min"],
        params["gru_num_layers_max"] + 1,
        params["gru_num_layers_step"]
    ))
    
    # ✅ FIX: Use while loop to avoid floating point errors
    gru_dropout_values = []
    val = params["gru_gru_dropout_min"]
    while val <= params["gru_gru_dropout_max"] + 1e-9:
        gru_dropout_values.append(round(val, 3))
        val += params["gru_gru_dropout_step"]
    
    fc_dropout_values = []
    val = params["gru_fc_dropout_min"]
    while val <= params["gru_fc_dropout_max"] + 1e-9:
        fc_dropout_values.append(round(val, 3))
        val += params["gru_fc_dropout_step"]
    
    learning_rate_values = []
    val = params["gru_learning_rate_min"]
    while val <= params["gru_learning_rate_max"] + 1e-9:
        learning_rate_values.append(round(val, 6))
        val += params["gru_learning_rate_step"]
    
    weight_decay_values = []
    val = params["gru_weight_decay_min"]
    while val <= params["gru_weight_decay_max"] + 1e-9:
        weight_decay_values.append(round(val, 7))
        val += params["gru_weight_decay_step"]
    
    huber_delta_values = []
    val = params["gru_huber_delta_min"]
    while val <= params["gru_huber_delta_max"] + 1e-9:
        huber_delta_values.append(round(val, 2))
        val += params["gru_huber_delta_step"]
    
    patience_values = list(range(
        params["gru_patience_min"],
        params["gru_patience_max"] + 1,
        params["gru_patience_step"]
    ))
    
    # ============================================================
    # PARAMETER GRID
    # ============================================================
    param_grid = {
        'hidden_size': hidden_size_values,
        'num_layers': num_layers_values,
        'gru_dropout': gru_dropout_values,
        'fc_dropout': fc_dropout_values,
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
    log(f"   hidden_size: {hidden_size_values}")
    log(f"   num_layers: {num_layers_values}")
    log(f"   gru_dropout: {gru_dropout_values}")
    log(f"   fc_dropout: {fc_dropout_values}")
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
            model = GRUModel(
                n_features=n_features,
                hidden_size=p['hidden_size'],
                num_layers=p['num_layers'],
                gru_dropout=p['gru_dropout'],
                fc_dropout=p['fc_dropout']
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


#GRU Architecture
class GRUModel(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size,
        num_layers,
        gru_dropout,
        fc_dropout
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(fc_dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),                 # activation_fc = ReLU
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]           # last time step
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)

def _forecast_only_gru(
    params,
    horizon,
    future_rain,
    future_mean_t,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torch.load("static/gru_bundle.pt", weights_only=False)
    FEATURES = bundle["features"]
    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    model = GRUModel(
        n_features=len(FEATURES),
        hidden_size=bundle["hidden_size"],  # ✅ CORRECT - from trained model
        num_layers=bundle["num_layers"],  # ✅ CORRECT
        gru_dropout=bundle["gru_dropout"],  # ✅ CORRECT
        fc_dropout=bundle["fc_dropout"],  # ✅ CORRECT
    ).to(device)

    model.load_state_dict(
        torch.load("static/gru_weights.pt", map_location=device, weights_only=False)
    )
    model.eval()

    last_window = bundle["last_window"].to(device).unsqueeze(0)

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
            pred = model(last_window).item()
            future_log_diffs.append(pred)

            last_window = torch.roll(last_window, shifts=-1, dims=1)

            last_window[0, -1, feat_idx["log_diff_lag_1"]] = pred
            last_window[0, -1, feat_idx["rain_lag_1"]] = future_rain_scaled[h]
            last_window[0, -1, feat_idx["mean_t_lag_1"]] = future_mean_t_scaled[h]

            if "time_lag_1" in feat_idx:
                last_window[0, -1, feat_idx["time_lag_1"]] += 1

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
def run_gru(
    data: pd.DataFrame,
    params: dict,
    horizon: int,
    frequency: str,
    log_callback=None
):

    """
    Website-ready GRU model
    Input  : DataFrame with columns [year, mean_t, rain, yield]
    Output : dict (metrics + train/test tables)
    """
    # ✅ ADD LOG FUNCTION
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)
#MANUAL HYPERPARAMETER BLOCK
    # ===============================
    # GRU HYPERPARAMETERS
    # ===============================
    if data is None:
        return _forecast_only_gru(
        params=params,
        horizon=horizon,
        future_rain=params["future_rain"],
        future_mean_t=params["future_mean_t"],
    )

    # Time-series
    WINDOW = params.get("window", 4)
    LAGS = params.get("lags", 3)
    split = params.get("split", 0.85)
    
    #Check if auto-tuning is enabled
    use_auto_tune = params.get("auto_tune_gru", True)
    # Model architecture
    hidden_size = params.get("hidden_size", 8)
    num_layers = params.get("num_layers", 2)
    gru_dropout = params.get("gru_dropout", 0.0)
    fc_dropout = params.get("fc_dropout", 0.3)

    # Training
    lr = params.get("learning_rate", 0.002)
    weight_decay = params.get("weight_decay", 1e-4)
    num_epochs = params.get("epochs", 150)
    patience = params.get("patience", 30)

    # Optimization
    huber_delta = params.get("huber_delta", 1.0)

    #    ✅ DEFINE DEVICE EARLY
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

#Log-Difference Target
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

#Feature Set
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

#Train–Test Split
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
    
        best_params_gru = auto_tune_gru(
            X_train_tune_scaled, y_train_tune,
            X_val_tune_scaled, y_val_tune,
            device, X_train.shape[2],
            params,  # ✅ ADD THIS
            max_epochs=100,
            log_callback=log_callback
        )
    
        # Override with best parameters
        hidden_size = best_params_gru['hidden_size']
        num_layers = best_params_gru['num_layers']
        gru_dropout = best_params_gru['gru_dropout']
        fc_dropout = best_params_gru['fc_dropout']
        lr = best_params_gru['learning_rate']  # ✅ CORRECT KEY
        weight_decay = best_params_gru['weight_decay']
        huber_delta = best_params_gru['huber_delta']
        patience = best_params_gru['patience']
    
        log(f"✅ Using optimized GRU parameters")

#Scaling (TRAIN ONLY)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test = scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

#Torch Setup
    

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).to(device)

#Training
    model = GRUModel(
        n_features=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        gru_dropout=gru_dropout,
        fc_dropout=fc_dropout
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

        preds = model(X_train)
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
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    # Training
    last_log_train = data["log_yield"].iloc[WINDOW - 1]
    pred_log_train = np.cumsum(train_preds) + last_log_train
    pred_train = np.exp(pred_log_train)

    actual_train = data["yield"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values

    years_train = data["year"].iloc[
        WINDOW:WINDOW + len(pred_train)
    ].values

    # Testing (forecast horizon aware)
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

    import matplotlib.pyplot as plt
    import os

    os.makedirs("static", exist_ok=True)
    torch.save(model.state_dict(), "static/gru_weights.pt")

    torch.save({
    "last_window": X_test[-1].detach().cpu(),
    "features": FEATURES,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "last_log": np.log(pred_test[-1]),
    "last_year": int(data["year"].iloc[-1]),
    
    # ✅ SAVE MODEL ARCHITECTURE (critical for loading)
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "gru_dropout": gru_dropout,
    "fc_dropout": fc_dropout,
    
    "params": params,
}, "static/gru_bundle.pt")


# -------------------------
# TRAINING PLOT
# -------------------------
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

    ax.set_title("GRU Model - Training Set")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield")
    ax.legend()

    fig.tight_layout()
    fig.savefig("static/gru_train.png", dpi=150)
    plt.close(fig)

# -------------------------
# TESTING PLOT
# -------------------------
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

    ax.set_title("GRU Model - Testing Set")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield")
    ax.legend()

    fig.tight_layout()
    fig.savefig("static/gru_test.png", dpi=150)
    plt.close(fig)


#RETURN
    return {
        "model_key": "gru",
        "model_name": "GRU",
        "horizon": horizon,

        "frequency": frequency,

        # ✅ ADD MODEL CONFIGURATION
        "model_config": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "gru_dropout": gru_dropout,
            "fc_dropout": fc_dropout,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "huber_delta": round(huber_delta, 2),
            "patience": patience,
            "epochs_used": num_epochs,
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