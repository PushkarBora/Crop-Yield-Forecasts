import numpy as np
import pandas as pd
import os
import sys
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Torch DLL fix (Windows)
# ------------------------------------------------------------
torch_dll_path = os.path.join(
    sys.exec_prefix, "Lib", "site-packages", "torch", "lib"
)
os.add_dll_directory(torch_dll_path)

import torch
import torch.nn as nn

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

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
# AUTO-TUNING FUNCTION
# ============================================================
def auto_tune_transformer(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    n_features,
    params,
    max_epochs=100,
    log_callback=None
):
    
    """
    Auto-tune Transformer using USER-DEFINED grid search ranges from params.
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

    log("\n🔍 Starting Transformer Grid Search Auto-Tuning (USER RANGES)")

    # ============================================================
    # READ USER-DEFINED RANGES - USE PROPER ROUNDING
    # ============================================================
    d_model_values = list(range(
        params["transformer_d_model_min"],
        params["transformer_d_model_max"] + 1,
        params["transformer_d_model_step"]
    ))

    nhead_values = list(range(
        params["transformer_nhead_min"],
        params["transformer_nhead_max"] + 1,
        params["transformer_nhead_step"]
    ))

    num_layers_values = list(range(
        params["transformer_num_layers_min"],
        params["transformer_num_layers_max"] + 1,
        params["transformer_num_layers_step"]
    ))

    dim_feedforward_values = list(range(
        params["transformer_dim_feedforward_min"],
        params["transformer_dim_feedforward_max"] + 1,
        params["transformer_dim_feedforward_step"]
    ))

    # ✅ FIX: Use proper rounding to avoid floating point errors
    dropout_values = []
    val = params["transformer_dropout_min"]
    while val <= params["transformer_dropout_max"] + 1e-9:
        dropout_values.append(round(val, 3))
        val += params["transformer_dropout_step"]

    learning_rate_values = []
    val = params["transformer_learning_rate_min"]
    while val <= params["transformer_learning_rate_max"] + 1e-9:
        learning_rate_values.append(round(val, 6))
        val += params["transformer_learning_rate_step"]

    weight_decay_values = []
    val = params["transformer_weight_decay_min"]
    while val <= params["transformer_weight_decay_max"] + 1e-9:
        weight_decay_values.append(round(val, 7))
        val += params["transformer_weight_decay_step"]

    huber_delta_values = []
    val = params["transformer_huber_delta_min"]
    while val <= params["transformer_huber_delta_max"] + 1e-9:
        huber_delta_values.append(round(val, 2))
        val += params["transformer_huber_delta_step"]

    patience_values = list(range(
        params["transformer_patience_min"],
        params["transformer_patience_max"] + 1,
        params["transformer_patience_step"]
    ))

    # Activation functions
    activation_values = ["relu", "gelu"]

    # ============================================================
    # PARAMETER GRID
    # ============================================================
    param_grid = {
        "d_model": d_model_values,
        "nhead": nhead_values,
        "num_layers": num_layers_values,
        "dim_feedforward": dim_feedforward_values,
        "dropout": dropout_values,
        "activation": activation_values,
        "learning_rate": learning_rate_values,
        "weight_decay": weight_decay_values,
        "huber_delta": huber_delta_values,
        "patience": patience_values,
    }

    keys = param_grid.keys()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

    total_combinations = len(combinations)

    # ✅ PRINT ALL PARAMETERS
    log("📦 Grid summary:")
    log(f"   d_model: {d_model_values}")
    log(f"   nhead: {nhead_values}")
    log(f"   num_layers: {num_layers_values}")
    log(f"   dim_feedforward: {dim_feedforward_values}")
    log(f"   dropout: {dropout_values}")
    log(f"   activation: {activation_values}")
    log(f"   learning_rate: {learning_rate_values}")
    log(f"   weight_decay: {weight_decay_values}")
    log(f"   huber_delta: {huber_delta_values}")
    log(f"   patience: {patience_values}")
    log(f"🔢 Total combinations: {total_combinations}")
    log("⏳ This may take a while...\n")
    
    best_val_loss = float('inf')
    best_params = None
    
    # ✅ Convert to tensors ONCE
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    # Test each combination
    for idx, p in enumerate(combinations, 1):
        if idx % 100 == 0:
            log(f"   Progress: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%)")
        
        try:
            # Validate nhead divides d_model
            if p['d_model'] % p['nhead'] != 0:
                continue
            
            # Create model
            model = TimeSeriesTransformer(
                n_features=n_features,  # ✅ Use the parameter!
                d_model=p['d_model'],
                nhead=p['nhead'],
                num_layers=p['num_layers'],
                dim_feedforward=p['dim_feedforward'],
                dropout=p['dropout'],
                activation=p['activation']
            ).to(device)
            
            # Setup training
            criterion = torch.nn.HuberLoss(delta=p['huber_delta'])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=p['learning_rate'],
                weight_decay=p['weight_decay']
            )
            
            # Training loop with early stopping
            best_epoch_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(max_epochs):
                # Train
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()
                
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


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        activation,
    ):
        super().__init__()

        self.embedding = nn.Linear(n_features, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)

        x = self.transformer(x)
        x = x[:, -1, :]
        x = self.norm(x)

        return self.fc(x).squeeze(-1)

def transformer_exogenous_forecast(
    model,
    last_window,
    future_exog,
    feature_names,
    device,
):
    preds_scaled = []
    current_window = last_window.clone()

    model.eval()
    with torch.no_grad():
        for step in range(len(future_exog)):

            pred_scaled = model(current_window.unsqueeze(0))
            preds_scaled.append(pred_scaled.item())

            new_row = []
            for feat in feature_names:

                if feat.startswith("log_diff_lag_"):
                    lag = int(feat.split("_")[-1])
                    if lag == 1:
                        new_row.append(pred_scaled.item())  # already scaled
                    else:
                        new_row.append(
                            current_window[-1, feature_names.index(f"log_diff_lag_{lag-1}")].item()
                        )

                elif feat.startswith("rain_lag_"):
                    lag = int(feat.split("_")[-1])
                    if lag == 1:
                        new_row.append(future_exog.iloc[step]["rain_scaled"])
                    else:
                        new_row.append(
                            current_window[-1, feature_names.index(f"rain_lag_{lag-1}")].item()
                        )

                elif feat.startswith("mean_t_lag_"):
                    lag = int(feat.split("_")[-1])
                    if lag == 1:
                        new_row.append(future_exog.iloc[step]["mean_t_scaled"])
                    else:
                        new_row.append(
                            current_window[-1, feature_names.index(f"mean_t_lag_{lag-1}")].item()
                        )

                elif feat.startswith("time_lag_"):
                    lag = int(feat.split("_")[-1])
                    if lag == 1:
                        new_row.append(
                            current_window[-1, feature_names.index("time_lag_1")].item() + 1
                        )
                    else:
                        new_row.append(
                            current_window[-1, feature_names.index(f"time_lag_{lag-1}")].item()
                        )

            new_row = torch.tensor(new_row, dtype=torch.float32, device=device)
            current_window = torch.cat([current_window[1:], new_row.unsqueeze(0)], dim=0)

    return np.array(preds_scaled)


# ============================================================
# FORECAST-ONLY (NO TRAINING, NO PLOTS)
# ============================================================
def _forecast_only(
    params,
    horizon,
    future_exog,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ✅ ADD THESE SAFETY CHECKS
    if not os.path.exists("static/transformer_bundle.pt"):
        raise FileNotFoundError(
            "Transformer model not trained yet. Please train the model first before forecasting."
        )
    
    if not os.path.exists("static/transformer_weights.pt"):
        raise FileNotFoundError(
            "Transformer weights not found. Please train the model first before forecasting."
        )
    bundle = torch.load("static/transformer_bundle.pt",weights_only=False)
    FEATURES = bundle["features"]

    # ✅ LOAD ARCHITECTURE FROM BUNDLE
    model = TimeSeriesTransformer(
        n_features=len(FEATURES),
        d_model=bundle["d_model"],                  # ✅ CORRECT
        nhead=bundle["nhead"],                      # ✅ CORRECT
        num_layers=bundle["num_layers"],            # ✅ CORRECT
        dim_feedforward=bundle["dim_feedforward"],  # ✅ CORRECT
        dropout=bundle["dropout"],                  # ✅ CORRECT
        activation=bundle["activation"],            # ✅ CORRECT
    ).to(device)

    model.load_state_dict(
        torch.load("static/transformer_weights.pt", map_location=device,weights_only=False)
    )
    model.eval()

    last_window = bundle["last_window"].to(device)

    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    future_exog["rain_scaled"] = (
        future_exog["rain"] - bundle["scaler_mean"][feat_idx["rain_lag_1"]]
    ) / bundle["scaler_scale"][feat_idx["rain_lag_1"]]

    future_exog["mean_t_scaled"] = (
        future_exog["mean_t"] - bundle["scaler_mean"][feat_idx["mean_t_lag_1"]]
    ) / bundle["scaler_scale"][feat_idx["mean_t_lag_1"]]

    future_log_diffs = transformer_exogenous_forecast(
        model,
        last_window,
        future_exog.iloc[:horizon],
        FEATURES,
        device,
    )

    forecast_log = np.cumsum(future_log_diffs) + bundle["last_log"]
    forecast_yield = np.exp(forecast_log)

    years = np.arange(
        bundle["last_year"] + 1,
        bundle["last_year"] + 1 + horizon,
    )

    return {
        "forecast_table": pd.DataFrame({
            "Year": years,
            "Forecast": forecast_yield,
        })
    }




# ============================================================
# MAIN ENTRY (TRAIN ONCE, METRICS STABLE)
# ============================================================
def run_transformer(
    data: pd.DataFrame,
    params: dict,
    horizon: int,
    frequency: str,
    future_exog: pd.DataFrame = None,
    mode: str = "train",
    log_callback=None,
):
    # ✅ ADD LOG FUNCTION RIGHT HERE
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(msg)

    if mode == "forecast":
        return _forecast_only(
            
            params=params,
            horizon=horizon,
            
            future_exog=future_exog
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------
    LAGS = params.get("lags", 3)
    WINDOW = params.get("window", 4)
    split = params.get("split", 0.85)

    # ✅ Check if auto-tuning is enabled
    use_auto_tune = params.get("auto_tune_transformer", True)

    # Default architecture
    d_model = params.get("d_model", 8)
    nhead = params.get("nhead", 2)
    num_layers = params.get("num_layers", 2)
    dim_feedforward = params.get("dim_feedforward", 16)
    dropout = params.get("dropout", 0.2)
    activation = params.get("activation", "gelu")

    # Training
    lr = params.get("learning_rate", 1e-3)
    weight_decay = params.get("weight_decay", 1e-4)
    epochs = params.get("epochs", 200)
    patience = params.get("patience", 30)
    huber_delta = params.get("huber_delta", 1.0)

    # --------------------------------------------------------
    # Data preparation
    # --------------------------------------------------------
    data = data.rename(
        columns={
            "Year": "year",
            "Mean_T": "mean_t",
            "Mean_Rain": "rain",
            "Yield": "yield",
        }
    )

    data = data[["year", "mean_t", "rain", "yield"]]
    data = data.sort_values("year").reset_index(drop=True)
    data = data.interpolate().dropna()

    data["log_yield"] = np.log(data["yield"])
    data["log_diff"] = data["log_yield"].diff()
    data["time_idx"] = np.arange(len(data))
    data = data.dropna().reset_index(drop=True)

    for lag in range(1, LAGS + 1):
        data[f"log_diff_lag_{lag}"] = data["log_diff"].shift(lag)
        data[f"rain_lag_{lag}"] = data["rain"].shift(lag)
        data[f"mean_t_lag_{lag}"] = data["mean_t"].shift(lag)
        data[f"time_lag_{lag}"] = data["time_idx"].shift(lag)

    data = data.dropna().reset_index(drop=True)
    FEATURES = [c for c in data.columns if "lag_" in c]

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------
    X, y = [], []
    for i in range(len(data) - WINDOW):
        X.append(data[FEATURES].iloc[i:i + WINDOW].values)
        y.append(data["log_diff"].iloc[i + WINDOW])

    X, y = np.array(X), np.array(y)

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
        
        # ✅ CORRECT FUNCTION CALL
        best_params_transformer = auto_tune_transformer(
            X_train_tune_scaled, y_train_tune,
            X_val_tune_scaled, y_val_tune,
            device,                # ✅ Pass device
            X_train.shape[2],      # ✅ n_features
            params,                # ✅ Pass params dict
            max_epochs=100,
            log_callback = log_callback
        )
        
        # ✅ CORRECT KEY NAMES (matching what auto_tune_transformer returns)
        d_model = best_params_transformer['d_model']
        nhead = best_params_transformer['nhead']
        num_layers = best_params_transformer['num_layers']
        dim_feedforward = best_params_transformer['dim_feedforward']
        dropout = best_params_transformer['dropout']
        activation = best_params_transformer['activation']
        lr = best_params_transformer['learning_rate']  # ✅ CORRECT KEY
        weight_decay = best_params_transformer['weight_decay']
        huber_delta = best_params_transformer['huber_delta']
        patience = best_params_transformer['patience']
        
        log(f"✅ Using optimized Transformer parameters")

    # ✅ NOW SCALE FULL TRAINING SET
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[2])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X.shape[2])).reshape(X_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    

    # --------------------------------------------------------
    # Model training (ONCE)
    # --------------------------------------------------------

    model = TimeSeriesTransformer(
        n_features=X.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    ).to(device)
    
    if mode == "train":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr,
            weight_decay=weight_decay
        )
        loss_fn = nn.HuberLoss(delta=huber_delta)

        best_loss = float("inf")
        best_state = None
        counter = 0

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()

            loss = loss_fn(model(X_train),y_train)
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

    # --------------------------------------------------------
    # Evaluation (FIXED)
    # --------------------------------------------------------

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    last_log_train = data["log_yield"].iloc[WINDOW - 1]
    pred_train = np.exp(np.cumsum(train_preds) + last_log_train)
    actual_train = data["yield"].iloc[WINDOW:WINDOW + len(pred_train)].values

    start_idx = split_idx + WINDOW
    last_log_test = data["log_yield"].iloc[start_idx - 1]
    pred_test = np.exp(np.cumsum(test_preds) + last_log_test)
    actual_test = np.exp(np.cumsum(y_test.cpu().numpy()) + last_log_test)
    mape_train = np.mean(
        np.abs((actual_train - pred_train) / actual_train)
    ) * 100
    mape_test = np.mean(
        np.abs((actual_test - pred_test) / actual_test)
    ) * 100

    # --------------------------------------------------------
# Year indices for plotting & tables
# --------------------------------------------------------
    years_train = data["year"].iloc[
    WINDOW : WINDOW + len(actual_train)
].values

    years_test = data["year"].iloc[
    start_idx : start_idx + len(actual_test)
].values

    

    if mode == "train":
        os.makedirs("static", exist_ok=True)
        torch.save(model.state_dict(), "static/transformer_weights.pt")

        # ✅ SAVE ARCHITECTURE IN BUNDLE
        torch.save({
            "last_window": X_test[-1].detach().cpu(),
            "features": FEATURES,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "last_log": np.log(pred_test[-1]),
            "last_year": int(data["year"].iloc[-1]),
            
            # ✅ SAVE MODEL ARCHITECTURE (critical for loading)
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            
            "params": params,
        }, "static/transformer_bundle.pt")


        

    # TRAINING PLOT
        plt.figure(figsize=(10, 5))
        plt.plot(years_train, actual_train, label="Actual (Train)")
        plt.plot(
        years_train,
        pred_train,
        label="Predicted (Train)",
        linestyle="--",
        marker="o",
        markersize=4,
        )
        plt.legend()
        plt.title("Transformer Model - Training Set")
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.tight_layout()
        plt.savefig("static/transformer_train.png")
        plt.close()

    # TESTING PLOT
        plt.figure(figsize=(10, 5))
        plt.plot(years_test, actual_test, label="Actual (Test)")
        plt.plot(
        years_test,
        pred_test,
        label="Predicted (Test)",
        linestyle="--",
        marker="o",
        markersize=4,
        )
        plt.legend()
        plt.title("Transformer Model - Testing Set")    
        plt.xlabel("Year")
        plt.ylabel("Yield")
        plt.tight_layout()
        plt.savefig("static/transformer_test.png")
        plt.close()


    # --------------------------------------------------------
    # Results (METRICS FROZEN)
    # --------------------------------------------------------
    results = {
        "model_key": "transformer",    # 🔑 for logic
        "model_name": "Transformer (Encoder)",
        "frequency": frequency,
        "horizon": horizon,

        # ✅ ADD MODEL CONFIGURATION
        "model_config": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "huber_delta": round(huber_delta, 2),
            "patience": patience,
            "epochs_used": epochs,
        },
        "auto_tuned": use_auto_tune,

        "rmse_train": round(root_mean_squared_error(actual_train, pred_train), 3),
        "r2_train": round(r2_score(actual_train, pred_train), 3),
        "mape_train": round(mape_train, 2),


        "rmse_test": round(root_mean_squared_error(actual_test, pred_test), 3),
        "r2_test": round(r2_score(actual_test, pred_test), 3),
        "mape_test": round(mape_test, 2),

        "train_table": pd.DataFrame({
            "Year": data["year"].iloc[WINDOW:WINDOW + len(pred_train)],
            "Actual": actual_train,
            "Predicted": pred_train,
        }),

        "test_table": pd.DataFrame({
            "Year": data["year"].iloc[start_idx:start_idx + len(actual_test)],
            "Actual": actual_test,
            "Predicted": pred_test,
        }),

        "forecast_table": None,
    }

    return results