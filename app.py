from flask import Flask, render_template, request, session, flash, redirect, url_for
from models.wavelet_ann_model import run_wavelet_ann
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest
from io import BytesIO
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from models.ar_garch_model import run_argarch
from models.ar_egarch_model import run_aregarch
from models.arma_garch_model import run_armagarch
from models.ar_tgarch_model import run_artgarch
from models.ar_gjrgarch_model import run_argjrgarch
import time
import os
import pycountry
import requests
os.makedirs('./flask_session', exist_ok=True)
import re  # ← ADD THIS
from dotenv import load_dotenv  # ← Add this import
from faqs import FAQ_DATABASE, get_all_faqs, get_faq_answer
# Load environment variables from .env file
load_dotenv()


from flask_session import Session
import pandas as pd
from io import StringIO

from models.transformer_model import run_transformer
from models.lstm_model import run_lstm
from models.rnn_model import run_rnn
from models.gru_model import run_gru
from models.cnn_model import run_cnn
from models.stacked_lstm_model import run_stacked_lstm
from models.bd_lstm_model import run_bd_lstm
from models.conv_lstm_model import run_conv_lstm
from models.deep_lstm_model import run_deep_lstm
from models.ann_model import run_ann
from models.svr_model import run_svr
from models.rf_model import run_rf
from models.arima_model import run_arima
from models.sarima_model import run_sarima
from models.tbats_model import run_tbats
from models.random_walk_model import run_random_walk
from models.ets_model import run_ets
from models.xgboost_model import run_xgb
from models.gbm_model import run_gbm
from models.knn_model import run_knn
from models.wavelet_lstm_model import run_wavelet_lstm
from models.wavelet_transformer_model import run_wavelet_transformer
from flask import Response, stream_with_context
import google.generativeai as genai
import queue
import threading
import time
import json
import requests
  # Add this at the top with other imports
# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)
STATIC_WRITEABLE = os.environ.get("STATIC_DIR", os.path.join(app.root_path, "static"))
os.makedirs(STATIC_WRITEABLE, exist_ok=True)
app.config["STATIC_WRITEABLE"] = STATIC_WRITEABLE
app.secret_key = "crop-yield-secret"   # REQUIRED for session storage

 #ADD SERVER-SIDE SESSION (REQUIRED!)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.environ.get("SESSION_FILE_DIR", "/tmp/flask_sessions")
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
app.config['SESSION_PERMANENT'] = False
Session(app)

#=====================================================
# INITIALIZE GOOGLE GEMINI CLIENT
# =====================================================
#=====================================================
# INITIALIZE GOOGLE GEMINI CLIENT
# =====================================================
# After load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    print("⚠️ WARNING: GOOGLE_API_KEY not found!")
else:
    print("✅ Google Gemini API Key loaded successfully")

# Global progress tracking
# ✅ REPLACE WITH — file-based, works across all workers
import json, threading
progress_lock = threading.Lock()
PROGRESS_DIR  = os.environ.get("PROGRESS_DIR", "/tmp/training_progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

def _progress_path(session_id):
    return os.path.join(PROGRESS_DIR, f"{session_id}.json")

def update_progress(session_id, data):
    with progress_lock:
        path = _progress_path(session_id)
        try:
            with open(path, "r") as f:
                current = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current = {
                'current_model': '', 'current_step': '',
                'models_completed': 0, 'total_models': 0,
                'progress_percent': 0, 'logs': [],
                'status': 'running', 'error': None
            }
        current.update(data)
        current['logs'] = current.get('logs', [])[-20:]
        with open(path, "w") as f:
            json.dump(current, f)

def get_progress(session_id):
    try:
        with open(_progress_path(session_id), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def clear_progress(session_id):
    try:
        os.remove(_progress_path(session_id))
    except FileNotFoundError:
        pass

# ======================================================
# CHAT ROUTES (CLEAN & MINIMAL)
# =====================================================

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat messages - use FAQ database for predefined questions, Gemini for others
    """
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        faq_key = data.get("faq_key")

        if not user_message and not faq_key:
            return {"error": "No message provided"}, 400

        if "chat_history" not in session:
            session["chat_history"] = []

        chat_history = session["chat_history"]

        # ✅ Handle FAQ clicks
        if faq_key:
            faq_item = get_faq_answer(faq_key)
            if faq_item:
                user_message = faq_item["question"]
                assistant_message = faq_item["answer"]
                
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": assistant_message})
                session["chat_history"] = chat_history[-20:]
                session.modified = True
                
                return {"response": assistant_message, "status": "success"}

        # ✅ For non-FAQ messages, use Gemini
        context = """You are an AI assistant for a Crop Yield Prediction platform.
Help users with ML models, hyperparameters, and metrics. Be concise and professional."""

        for entry in chat_history[-5:]:
            role = "User" if entry["role"] == "user" else "Assistant"
            context += f"\n{role}: {entry['content']}"

        context += f"\nUser: {user_message}\nAssistant:"

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context)
        assistant_message = response.text

        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": assistant_message})
        session["chat_history"] = chat_history[-20:]
        session.modified = True

        return {"response": assistant_message, "status": "success"}

    except Exception as e:
        print("Chat error:", str(e))
        return {
            "response": "I'm here to help! Try clicking one of the suggested questions above.",
            "status": "error"
        }, 500


@app.route("/get_faqs", methods=["GET"])
def get_faqs():
    """Return list of available FAQs"""
    return {"faqs": get_all_faqs()}


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    """Clear chat history"""
    session["chat_history"] = []
    session.modified = True
    return {"status": "success"}


# =====================================================
# CENTRALIZED HYPERPARAMETER COLLECTOR
# =====================================================
def collect_hyperparameters(form):
    """
    Collect hyperparameters from form, including grid search ranges for all models.
    Supports: LSTM, Transformer, RNN, GRU, ANN, SVR, Random Forest, ARIMA
    """
    params = {}
    
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    def safe_int(val, default):
        """Safely convert to integer with fallback"""
        try:
            return int(val) if val else default
        except (ValueError, TypeError):
            return default
    
    def safe_float(val, default):
        """Safely convert to float with fallback"""
        try:
            return float(val) if val else default
        except (ValueError, TypeError):
            return default
    
    def safe_bool(val):
        """Convert checkbox value to boolean"""
        return val == "true"
    
    # ============================================================
    # SHARED TIME-SERIES PARAMETERS
    # ============================================================
    params["split"] = safe_float(form.get("split"), 0.85)
    
    # ============================================================
    # LSTM PARAMETERS
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_lstm"] = safe_bool(form.get("auto_tune_lstm"))
    
    # Grid Search Ranges
    #LSTM lags
    params["lstm_lags_min"]  = safe_int(form.get("lstm_lags_min"),  2)
    params["lstm_lags_max"]  = safe_int(form.get("lstm_lags_max"),  5)
    params["lstm_lags_step"] = safe_int(form.get("lstm_lags_step"), 1)

    params["lstm_hidden_size_min"] = safe_int(form.get("lstm_hidden_size_min"), 4)
    params["lstm_hidden_size_max"] = safe_int(form.get("lstm_hidden_size_max"), 16)
    params["lstm_hidden_size_step"] = safe_int(form.get("lstm_hidden_size_step"), 4)
    
    params["lstm_num_layers_min"] = safe_int(form.get("lstm_num_layers_min"), 1)
    params["lstm_num_layers_max"] = safe_int(form.get("lstm_num_layers_max"), 3)
    params["lstm_num_layers_step"] = safe_int(form.get("lstm_num_layers_step"), 1)
    
    params["lstm_dropout_min"] = safe_float(form.get("lstm_dropout_min"), 0.0)
    params["lstm_dropout_max"] = safe_float(form.get("lstm_dropout_max"), 0.3)
    params["lstm_dropout_step"] = safe_float(form.get("lstm_dropout_step"), 0.1)
    
    params["lstm_learning_rate_min"] = safe_float(form.get("lstm_learning_rate_min"), 0.0005)
    params["lstm_learning_rate_max"] = safe_float(form.get("lstm_learning_rate_max"), 0.005)
    params["lstm_learning_rate_step"] = safe_float(form.get("lstm_learning_rate_step"), 0.001)
    
    params["lstm_weight_decay_min"] = safe_float(form.get("lstm_weight_decay_min"), 1e-5)
    params["lstm_weight_decay_max"] = safe_float(form.get("lstm_weight_decay_max"), 5e-4)
    params["lstm_weight_decay_step"] = safe_float(form.get("lstm_weight_decay_step"), 1e-4)
    
    params["lstm_huber_delta_min"] = safe_float(form.get("lstm_huber_delta_min"), 0.7)
    params["lstm_huber_delta_max"] = safe_float(form.get("lstm_huber_delta_max"), 1.5)
    params["lstm_huber_delta_step"] = safe_float(form.get("lstm_huber_delta_step"), 0.4)
    
    params["lstm_patience_min"] = safe_int(form.get("lstm_patience_min"), 20)
    params["lstm_patience_max"] = safe_int(form.get("lstm_patience_max"), 50)
    params["lstm_patience_step"] = safe_int(form.get("lstm_patience_step"), 10)
    
    #✅ ADD EPOCHS RANGE
    params["lstm_epochs_min"] = safe_int(form.get("lstm_epochs_min"), 100)
    params["lstm_epochs_max"] = safe_int(form.get("lstm_epochs_max"), 300)
    params["lstm_epochs_step"] = safe_int(form.get("lstm_epochs_step"), 100)
    
    # Manual parameters (used if auto-tune disabled)
    #✅ ADD THIS:
    params["lstm_lags"] = safe_int(
    form.get("lstm_lags") or form.get("lags"), 3
    )
    params["hidden_size"] = safe_int(form.get("hidden_size"), 8)
    params["num_layers"] = safe_int(form.get("num_layers"), 2)
    params["dropout"] = safe_float(form.get("dropout"), 0.3)
    params["learning_rate"] = safe_float(form.get("learning_rate"), 0.003)
    params["weight_decay"] = safe_float(form.get("weight_decay"), 1e-4)
    params["huber_delta"] = safe_float(form.get("huber_delta"), 1.0)
    params["patience"] = safe_int(form.get("patience"), 20)
    params["epochs"] = safe_int(form.get("epochs"), 200)
    
    # ============================================================
    # TRANSFORMER PARAMETERS
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_transformer"] = safe_bool(form.get("auto_tune_transformer"))
    
    # Grid Search Ranges
    #Transformer lags
    params["transformer_lags_min"]  = safe_int(form.get("transformer_lags_min"),  2)
    params["transformer_lags_max"]  = safe_int(form.get("transformer_lags_max"),  5)
    params["transformer_lags_step"] = safe_int(form.get("transformer_lags_step"), 1)
    
    params["transformer_d_model_min"] = safe_int(form.get("transformer_d_model_min"), 4)
    params["transformer_d_model_max"] = safe_int(form.get("transformer_d_model_max"), 16)
    params["transformer_d_model_step"] = safe_int(form.get("transformer_d_model_step"), 4)
    
    params["transformer_nhead_min"] = safe_int(form.get("transformer_nhead_min"), 1)
    params["transformer_nhead_max"] = safe_int(form.get("transformer_nhead_max"), 4)
    params["transformer_nhead_step"] = safe_int(form.get("transformer_nhead_step"), 1)
    
    params["transformer_num_layers_min"] = safe_int(form.get("transformer_num_layers_min"), 1)
    params["transformer_num_layers_max"] = safe_int(form.get("transformer_num_layers_max"), 3)
    params["transformer_num_layers_step"] = safe_int(form.get("transformer_num_layers_step"), 1)
    
    params["transformer_dim_feedforward_min"] = safe_int(form.get("transformer_dim_feedforward_min"), 8)
    params["transformer_dim_feedforward_max"] = safe_int(form.get("transformer_dim_feedforward_max"), 32)
    params["transformer_dim_feedforward_step"] = safe_int(form.get("transformer_dim_feedforward_step"), 8)
    
    params["transformer_dropout_min"] = safe_float(form.get("transformer_dropout_min"), 0.1)
    params["transformer_dropout_max"] = safe_float(form.get("transformer_dropout_max"), 0.3)
    params["transformer_dropout_step"] = safe_float(form.get("transformer_dropout_step"), 0.1)
    
    params["transformer_learning_rate_min"] = safe_float(form.get("transformer_learning_rate_min"), 0.0005)
    params["transformer_learning_rate_max"] = safe_float(form.get("transformer_learning_rate_max"), 0.005)
    params["transformer_learning_rate_step"] = safe_float(form.get("transformer_learning_rate_step"), 0.001)
    
    params["transformer_weight_decay_min"] = safe_float(form.get("transformer_weight_decay_min"), 1e-5)
    params["transformer_weight_decay_max"] = safe_float(form.get("transformer_weight_decay_max"), 5e-4)
    params["transformer_weight_decay_step"] = safe_float(form.get("transformer_weight_decay_step"), 1e-4)
    
    params["transformer_huber_delta_min"] = safe_float(form.get("transformer_huber_delta_min"), 0.7)
    params["transformer_huber_delta_max"] = safe_float(form.get("transformer_huber_delta_max"), 1.5)
    params["transformer_huber_delta_step"] = safe_float(form.get("transformer_huber_delta_step"), 0.4)
    
    params["transformer_patience_min"] = safe_int(form.get("transformer_patience_min"), 20)
    params["transformer_patience_max"] = safe_int(form.get("transformer_patience_max"), 50)
    params["transformer_patience_step"] = safe_int(form.get("transformer_patience_step"), 10)
    
    params["transformer_epochs_min"]  = safe_int(form.get("transformer_epochs_min"),  100)
    params["transformer_epochs_max"]  = safe_int(form.get("transformer_epochs_max"),  300)
    params["transformer_epochs_step"] = safe_int(form.get("transformer_epochs_step"), 100)
    # Manual parameters (used if auto-tune disabled)
    # After transformer manual params section:
    params["transformer_lags"] = safe_int(
    form.get("transformer_lags") or form.get("lags"), 3
    )
    params["d_model"] = safe_int(form.get("d_model"), 8)
    params["nhead"] = safe_int(form.get("nhead"), 2)
    params["dim_feedforward"] = safe_int(form.get("dim_feedforward"), 16)
    params["activation"] = form.get("activation", "gelu")
    params["transformer_num_layers"]   = safe_int(form.get("transformer_num_layers"), 2)
    params["transformer_dropout"]      = safe_float(form.get("transformer_dropout"), 0.2)
    params["transformer_learning_rate"]= safe_float(form.get("transformer_learning_rate"), 0.001)
    params["transformer_weight_decay"] = safe_float(form.get("transformer_weight_decay"), 1e-4)
    params["transformer_epochs"]       = safe_int(form.get("transformer_epochs"), 200)
    params["transformer_patience"]     = safe_int(form.get("transformer_patience"), 30)
    params["transformer_huber_delta"]  = safe_float(form.get("transformer_huber_delta"), 1.0)
    # ============================================================
    # RNN PARAMETERS
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_rnn"] = safe_bool(form.get("auto_tune_rnn"))
    
    # Grid Search Ranges
    #RNN lags
    params["rnn_lags_min"]  = safe_int(form.get("rnn_lags_min"),  2)
    params["rnn_lags_max"]  = safe_int(form.get("rnn_lags_max"),  5)
    params["rnn_lags_step"] = safe_int(form.get("rnn_lags_step"), 1)

    params["rnn_hidden_size_min"] = safe_int(form.get("rnn_hidden_size_min"), 4)
    params["rnn_hidden_size_max"] = safe_int(form.get("rnn_hidden_size_max"), 16)
    params["rnn_hidden_size_step"] = safe_int(form.get("rnn_hidden_size_step"), 4)
    
    params["rnn_num_layers_min"] = safe_int(form.get("rnn_num_layers_min"), 1)
    params["rnn_num_layers_max"] = safe_int(form.get("rnn_num_layers_max"), 3)
    params["rnn_num_layers_step"] = safe_int(form.get("rnn_num_layers_step"), 1)
    
    params["rnn_dropout_rnn_min"] = safe_float(form.get("rnn_dropout_rnn_min"), 0.0)
    params["rnn_dropout_rnn_max"] = safe_float(form.get("rnn_dropout_rnn_max"), 0.2)
    params["rnn_dropout_rnn_step"] = safe_float(form.get("rnn_dropout_rnn_step"), 0.1)
    
    params["rnn_dropout_fc_min"] = safe_float(form.get("rnn_dropout_fc_min"), 0.2)
    params["rnn_dropout_fc_max"] = safe_float(form.get("rnn_dropout_fc_max"), 0.4)
    params["rnn_dropout_fc_step"] = safe_float(form.get("rnn_dropout_fc_step"), 0.1)
    
    params["rnn_learning_rate_min"] = safe_float(form.get("rnn_learning_rate_min"), 0.0005)
    params["rnn_learning_rate_max"] = safe_float(form.get("rnn_learning_rate_max"), 0.005)
    params["rnn_learning_rate_step"] = safe_float(form.get("rnn_learning_rate_step"), 0.001)
    
    params["rnn_weight_decay_min"] = safe_float(form.get("rnn_weight_decay_min"), 1e-5)
    params["rnn_weight_decay_max"] = safe_float(form.get("rnn_weight_decay_max"), 5e-4)
    params["rnn_weight_decay_step"] = safe_float(form.get("rnn_weight_decay_step"), 1e-4)
    
    params["rnn_huber_delta_min"] = safe_float(form.get("rnn_huber_delta_min"), 0.7)
    params["rnn_huber_delta_max"] = safe_float(form.get("rnn_huber_delta_max"), 1.5)
    params["rnn_huber_delta_step"] = safe_float(form.get("rnn_huber_delta_step"), 0.4)
    
    params["rnn_patience_min"] = safe_int(form.get("rnn_patience_min"), 20)
    params["rnn_patience_max"] = safe_int(form.get("rnn_patience_max"), 50)
    params["rnn_patience_step"] = safe_int(form.get("rnn_patience_step"), 10)
    
    params["rnn_epochs_min"]  = safe_int(form.get("rnn_epochs_min"),  100)
    params["rnn_epochs_max"]  = safe_int(form.get("rnn_epochs_max"),  300)
    params["rnn_epochs_step"] = safe_int(form.get("rnn_epochs_step"), 100)
    # Manual parameters (used if auto-tune disabled)
    params["rnn_lags"] = safe_int(form.get("rnn_lags") or form.get("lags"), 3)
    params["lags"] = safe_int(form.get("lags"), 3)
    params["dropout_rnn"] = safe_float(form.get("dropout_rnn"), 0.2)
    params["dropout_fc"] = safe_float(form.get("dropout_fc"), 0.3)
    params["nonlinearity"] = form.get("nonlinearity", "tanh")
    
    # ============================================================
    # GRU PARAMETERS
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_gru"] = safe_bool(form.get("auto_tune_gru"))
    
    # Grid Search Ranges
    #GRU lags
    params["gru_lags_min"]  = safe_int(form.get("gru_lags_min"),  2)
    params["gru_lags_max"]  = safe_int(form.get("gru_lags_max"),  5)
    params["gru_lags_step"] = safe_int(form.get("gru_lags_step"), 1)

    params["gru_hidden_size_min"] = safe_int(form.get("gru_hidden_size_min"), 4)
    params["gru_hidden_size_max"] = safe_int(form.get("gru_hidden_size_max"), 16)
    params["gru_hidden_size_step"] = safe_int(form.get("gru_hidden_size_step"), 4)
    
    params["gru_num_layers_min"] = safe_int(form.get("gru_num_layers_min"), 1)
    params["gru_num_layers_max"] = safe_int(form.get("gru_num_layers_max"), 3)
    params["gru_num_layers_step"] = safe_int(form.get("gru_num_layers_step"), 1)
    
    params["gru_gru_dropout_min"] = safe_float(form.get("gru_gru_dropout_min"), 0.0)
    params["gru_gru_dropout_max"] = safe_float(form.get("gru_gru_dropout_max"), 0.1)
    params["gru_gru_dropout_step"] = safe_float(form.get("gru_gru_dropout_step"), 0.05)
    
    params["gru_fc_dropout_min"] = safe_float(form.get("gru_fc_dropout_min"), 0.2)
    params["gru_fc_dropout_max"] = safe_float(form.get("gru_fc_dropout_max"), 0.4)
    params["gru_fc_dropout_step"] = safe_float(form.get("gru_fc_dropout_step"), 0.1)
    
    params["gru_learning_rate_min"] = safe_float(form.get("gru_learning_rate_min"), 0.0005)
    params["gru_learning_rate_max"] = safe_float(form.get("gru_learning_rate_max"), 0.005)
    params["gru_learning_rate_step"] = safe_float(form.get("gru_learning_rate_step"), 0.001)
    
    params["gru_weight_decay_min"] = safe_float(form.get("gru_weight_decay_min"), 1e-5)
    params["gru_weight_decay_max"] = safe_float(form.get("gru_weight_decay_max"), 5e-4)
    params["gru_weight_decay_step"] = safe_float(form.get("gru_weight_decay_step"), 1e-4)
    
    params["gru_huber_delta_min"] = safe_float(form.get("gru_huber_delta_min"), 0.7)
    params["gru_huber_delta_max"] = safe_float(form.get("gru_huber_delta_max"), 1.5)
    params["gru_huber_delta_step"] = safe_float(form.get("gru_huber_delta_step"), 0.4)
    
    params["gru_patience_min"] = safe_int(form.get("gru_patience_min"), 20)
    params["gru_patience_max"] = safe_int(form.get("gru_patience_max"), 50)
    params["gru_patience_step"] = safe_int(form.get("gru_patience_step"), 10)
    
    params["gru_epochs_min"]  = safe_int(form.get("gru_epochs_min"),  100)
    params["gru_epochs_max"]  = safe_int(form.get("gru_epochs_max"),  300)
    params["gru_epochs_step"] = safe_int(form.get("gru_epochs_step"), 100)
    # Manual parameters (used if auto-tune disabled)
    params["gru_lags"] = safe_int(form.get("gru_lags") or form.get("lags"), 3)
    params["lags"] = safe_int(form.get("lags"), 3)
    params["gru_dropout"] = safe_float(form.get("gru_dropout"), 0.0)
    params["fc_dropout"] = safe_float(form.get("fc_dropout"), 0.3)
    
    # ============================================================
    # ANN PARAMETERS
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_ann"] = safe_bool(form.get("auto_tune_ann"))
    
    # Grid Search Ranges
    #ANN lags
    params["ann_lags_min"]  = safe_int(form.get("ann_lags_min"),  2)
    params["ann_lags_max"]  = safe_int(form.get("ann_lags_max"),  5)
    params["ann_lags_step"] = safe_int(form.get("ann_lags_step"), 1)
    
    params["ann_hidden_layer_1_min"] = safe_int(form.get("ann_hidden_layer_1_min"), 8)
    params["ann_hidden_layer_1_max"] = safe_int(form.get("ann_hidden_layer_1_max"), 32)
    params["ann_hidden_layer_1_step"] = safe_int(form.get("ann_hidden_layer_1_step"), 8)
    
    params["ann_hidden_layer_2_min"] = safe_int(form.get("ann_hidden_layer_2_min"), 4)
    params["ann_hidden_layer_2_max"] = safe_int(form.get("ann_hidden_layer_2_max"), 16)
    params["ann_hidden_layer_2_step"] = safe_int(form.get("ann_hidden_layer_2_step"), 4)
    
    params["ann_dropout_1_min"] = safe_float(form.get("ann_dropout_1_min"), 0.1)
    params["ann_dropout_1_max"] = safe_float(form.get("ann_dropout_1_max"), 0.3)
    params["ann_dropout_1_step"] = safe_float(form.get("ann_dropout_1_step"), 0.1)
    
    params["ann_dropout_2_min"] = safe_float(form.get("ann_dropout_2_min"), 0.1)
    params["ann_dropout_2_max"] = safe_float(form.get("ann_dropout_2_max"), 0.4)
    params["ann_dropout_2_step"] = safe_float(form.get("ann_dropout_2_step"), 0.1)
    
    params["ann_learning_rate_min"] = safe_float(form.get("ann_learning_rate_min"), 0.0005)
    params["ann_learning_rate_max"] = safe_float(form.get("ann_learning_rate_max"), 0.005)
    params["ann_learning_rate_step"] = safe_float(form.get("ann_learning_rate_step"), 0.001)
    
    params["ann_weight_decay_min"] = safe_float(form.get("ann_weight_decay_min"), 1e-5)
    params["ann_weight_decay_max"] = safe_float(form.get("ann_weight_decay_max"), 5e-4)
    params["ann_weight_decay_step"] = safe_float(form.get("ann_weight_decay_step"), 1e-4)
    
    params["ann_huber_delta_min"] = safe_float(form.get("ann_huber_delta_min"), 0.5)
    params["ann_huber_delta_max"] = safe_float(form.get("ann_huber_delta_max"), 1.5)
    params["ann_huber_delta_step"] = safe_float(form.get("ann_huber_delta_step"), 0.5)
    
    params["ann_epochs_min"]  = safe_int(form.get("ann_epochs_min"),  100)
    params["ann_epochs_max"]  = safe_int(form.get("ann_epochs_max"),  300)
    params["ann_epochs_step"] = safe_int(form.get("ann_epochs_step"), 100)
    
    params["ann_patience_min"] = safe_int(form.get("ann_patience_min"), 20)
    params["ann_patience_max"] = safe_int(form.get("ann_patience_max"), 50)
    params["ann_patience_step"] = safe_int(form.get("ann_patience_step"), 10)
    
    # Manual parameters (used if auto-tune disabled)
    params["ann_lags"] = safe_int(form.get("ann_lags") or form.get("lags"), 3)
    params["lags"] = safe_int(form.get("lags"), 3)
    params["hidden_layer_1"] = safe_int(form.get("hidden_layer_1"), 16)
    params["hidden_layer_2"] = safe_int(form.get("hidden_layer_2"), 8)
    params["dropout_1"] = safe_float(form.get("dropout_1"), 0.2)
    params["dropout_2"] = safe_float(form.get("dropout_2"), 0.3)
    
    # ============================================================
    # WAVELET-ANN PARAMETERS
    # ============================================================
    # ── WAVELET-ANN PARAMETERS ──────────────────────────────────────
    # ── WAVELET-ANN PARAMETERS ──────────────────────────────────────────────────
    WAVELET_ALIAS_MAP = {
        "la8":  "sym8",   # legacy alias → valid pywt name
        "haar": "haar",
        "db2":  "db2",
        "db4":  "db4",
    }

    auto_tune_wann = form.get("auto_tune_wann") == "true"
    params["auto_tune_wann"] = auto_tune_wann

    if auto_tune_wann:
        # ── GRID SEARCH MODE ────────────────────────────────────────────────────
        raw_wavelets = form.get("wann_wavelets", "").strip()
        if raw_wavelets:
            params["wann_wavelets"] = [
                WAVELET_ALIAS_MAP.get(w.strip(), w.strip())
                for w in raw_wavelets.split(",") if w.strip()
            ]
        else:
            # Fallback — JS should always populate this, but just in case
            params["wann_wavelets"] = ["haar", "db2", "db4", "sym8"]

        params["wann_min_level"]           = safe_int(form.get("wann_min_level"), 1)
        params["wann_max_level"]           = safe_int(form.get("wann_max_level"), 3)
        params["wann_lags_min"]            = safe_int(form.get("wann_lags_min"),  2)
        params["wann_lags_max"]            = safe_int(form.get("wann_lags_max"),  4)
        params["wann_lags_step"]           = safe_int(form.get("wann_lags_step"), 2)

        params["ann_hidden_layer_1_min"]   = safe_int(form.get("ann_hidden_layer_1_min"),  8)
        params["ann_hidden_layer_1_max"]   = safe_int(form.get("ann_hidden_layer_1_max"), 16)
        params["ann_hidden_layer_1_step"]  = safe_int(form.get("ann_hidden_layer_1_step"), 8)

        params["ann_hidden_layer_2_min"]   = safe_int(form.get("ann_hidden_layer_2_min"),  4)
        params["ann_hidden_layer_2_max"]   = safe_int(form.get("ann_hidden_layer_2_max"),  8)
        params["ann_hidden_layer_2_step"]  = safe_int(form.get("ann_hidden_layer_2_step"), 4)

        params["ann_dropout_1_min"]        = safe_float(form.get("ann_dropout_1_min"),  0.2)
        params["ann_dropout_1_max"]        = safe_float(form.get("ann_dropout_1_max"),  0.2)
        params["ann_dropout_1_step"]       = safe_float(form.get("ann_dropout_1_step"), 0.1)

        params["ann_dropout_2_min"]        = safe_float(form.get("ann_dropout_2_min"),  0.3)
        params["ann_dropout_2_max"]        = safe_float(form.get("ann_dropout_2_max"),  0.3)
        params["ann_dropout_2_step"]       = safe_float(form.get("ann_dropout_2_step"), 0.1)

        params["ann_learning_rate_min"]    = safe_float(form.get("ann_learning_rate_min"),  0.001)
        params["ann_learning_rate_max"]    = safe_float(form.get("ann_learning_rate_max"),  0.003)
        params["ann_learning_rate_step"]   = safe_float(form.get("ann_learning_rate_step"), 0.002)

        params["ann_weight_decay_min"]     = safe_float(form.get("ann_weight_decay_min"),  0.0001)
        params["ann_weight_decay_max"]     = safe_float(form.get("ann_weight_decay_max"),  0.0001)
        params["ann_weight_decay_step"]    = safe_float(form.get("ann_weight_decay_step"), 0.0001)

        params["ann_huber_delta_min"]      = safe_float(form.get("ann_huber_delta_min"),  1.0)
        params["ann_huber_delta_max"]      = safe_float(form.get("ann_huber_delta_max"),  1.0)
        params["ann_huber_delta_step"]     = safe_float(form.get("ann_huber_delta_step"), 0.5)

        params["ann_epochs_min"]           = safe_int(form.get("ann_epochs_min"),  200)
        params["ann_epochs_max"]           = safe_int(form.get("ann_epochs_max"),  200)
        params["ann_epochs_step"]          = safe_int(form.get("ann_epochs_step"), 100)

        params["ann_patience_min"]         = safe_int(form.get("ann_patience_min"),  30)
        params["ann_patience_max"]         = safe_int(form.get("ann_patience_max"),  30)
        params["ann_patience_step"]        = safe_int(form.get("ann_patience_step"), 10)

    else:
        # ── MANUAL MODE ─────────────────────────────────────────────────────────
        manual_wavelet = form.get("wann_manual_wavelet", "sym8").strip()
        params["wann_wavelets"] = [WAVELET_ALIAS_MAP.get(manual_wavelet, manual_wavelet)]

        # Fixed level — min == max so the model only tests one level
        manual_level = safe_int(form.get("wann_manual_level"), 3)
        params["wann_min_level"] = manual_level
        params["wann_max_level"] = manual_level

        # All ANN params fixed at the single submitted value (min == max, step=1)
        lags = safe_int(form.get("wann_lags_min"), 3)
        params["wann_lags_min"]  = lags
        params["wann_lags_max"]  = lags
        params["wann_lags_step"] = 1

        hl1 = safe_int(form.get("ann_hidden_layer_1_min"), 16)
        params["ann_hidden_layer_1_min"]  = hl1
        params["ann_hidden_layer_1_max"]  = hl1
        params["ann_hidden_layer_1_step"] = 1

        hl2 = safe_int(form.get("ann_hidden_layer_2_min"), 8)
        params["ann_hidden_layer_2_min"]  = hl2
        params["ann_hidden_layer_2_max"]  = hl2
        params["ann_hidden_layer_2_step"] = 1

        d1 = safe_float(form.get("ann_dropout_1_min"), 0.2)
        params["ann_dropout_1_min"]  = d1
        params["ann_dropout_1_max"]  = d1
        params["ann_dropout_1_step"] = 0.1

        d2 = safe_float(form.get("ann_dropout_2_min"), 0.3)
        params["ann_dropout_2_min"]  = d2
        params["ann_dropout_2_max"]  = d2
        params["ann_dropout_2_step"] = 0.1

        lr = safe_float(form.get("ann_learning_rate_min"), 0.001)
        params["ann_learning_rate_min"]  = lr
        params["ann_learning_rate_max"]  = lr
        params["ann_learning_rate_step"] = 0.001

        wd = safe_float(form.get("ann_weight_decay_min"), 0.0001)
        params["ann_weight_decay_min"]  = wd
        params["ann_weight_decay_max"]  = wd
        params["ann_weight_decay_step"] = 0.0001

        hd = safe_float(form.get("ann_huber_delta_min"), 1.0)
        params["ann_huber_delta_min"]  = hd
        params["ann_huber_delta_max"]  = hd
        params["ann_huber_delta_step"] = 0.5

        ep = safe_int(form.get("ann_epochs_min"), 200)
        params["ann_epochs_min"]  = ep
        params["ann_epochs_max"]  = ep
        params["ann_epochs_step"] = 100

        pa = safe_int(form.get("ann_patience_min"), 30)
        params["ann_patience_min"]  = pa
        params["ann_patience_max"]  = pa
        params["ann_patience_step"] = 10

        # Remove old boolean flags — checkboxes no longer exist in the form
        # (kept here as None so downstream code doesn't KeyError if it reads them)
    params["wann_haar"] = None
    params["wann_db2"]  = None
    params["wann_db4"]  = None
    params["wann_la8"]  = None

    # ANN architecture grid (same keys as standalone ANN — reused automatically)
    # These are already collected in the ANN section above, so no duplication needed.
    # Just make sure ann_hidden_layer_1_min/max/step etc. are present — they are.

    # ============================================================
    # WAVELET-LSTM PARAMETERS
    # ============================================================
    # ============================================================
    # WAVELET-LSTM PARAMETERS
    # ============================================================
    # ============================================================
    # WAVELET-LSTM PARAMETERS
    # ============================================================
    # ============================================================
    # WAVELET-LSTM PARAMETERS
    # ============================================================
    auto_tune_wlstm = form.get("auto_tune_wlstm") == "true"
    params["auto_tune_wlstm"] = auto_tune_wlstm

    if auto_tune_wlstm:
        raw_wlstm = form.get("wlstm_wavelets", "").strip()
        params["wlstm_wavelets_resolved"] = [w.strip() for w in raw_wlstm.split(",") if w.strip()] \
                                         or ["haar", "db2", "db4", "sym8"]
        params["wlstm_min_level_resolved"]  = safe_int(form.get("wlstm_min_level"),  1)
        params["wlstm_max_level_resolved"]  = safe_int(form.get("wlstm_max_level"),  3)
        params["wlstm_lags_min_resolved"]   = safe_int(form.get("wlstm_lags_min"),   2)
        params["wlstm_lags_max_resolved"]   = safe_int(form.get("wlstm_lags_max"),   4)
        params["wlstm_lags_step_resolved"]  = safe_int(form.get("wlstm_lags_step"),  2)
        params["wlstm_hs_min_resolved"]     = safe_int(form.get("wlstm_hs_min"),     8)
        params["wlstm_hs_max_resolved"]     = safe_int(form.get("wlstm_hs_max"),     16)
        params["wlstm_hs_step_resolved"]    = safe_int(form.get("wlstm_hs_step"),    8)
        params["wlstm_num_layers_min_resolved"]  = safe_int(form.get("wlstm_num_layers_min"),  1)
        params["wlstm_num_layers_max_resolved"]  = safe_int(form.get("wlstm_num_layers_max"),  2)
        params["wlstm_num_layers_step_resolved"] = safe_int(form.get("wlstm_num_layers_step"), 1)
        params["wlstm_dr_min_resolved"]  = safe_float(form.get("wlstm_dr_min"),  0.2)
        params["wlstm_dr_max_resolved"]  = safe_float(form.get("wlstm_dr_max"),  0.2)
        params["wlstm_dr_step_resolved"] = safe_float(form.get("wlstm_dr_step"), 0.1)
        params["wlstm_lr_min_resolved"]  = safe_float(form.get("wlstm_lr_min"),  0.001)
        params["wlstm_lr_max_resolved"]  = safe_float(form.get("wlstm_lr_max"),  0.003)
        params["wlstm_lr_step_resolved"] = safe_float(form.get("wlstm_lr_step"), 0.002)
        params["wlstm_wd_min_resolved"]  = safe_float(form.get("wlstm_wd_min"),  1e-4)
        params["wlstm_wd_max_resolved"]  = safe_float(form.get("wlstm_wd_max"),  1e-4)
        params["wlstm_wd_step_resolved"] = safe_float(form.get("wlstm_wd_step"), 1e-4)
        params["wlstm_hd_min_resolved"]  = safe_float(form.get("wlstm_hd_min"),  1.0)
        params["wlstm_hd_max_resolved"]  = safe_float(form.get("wlstm_hd_max"),  1.0)
        params["wlstm_hd_step_resolved"] = safe_float(form.get("wlstm_hd_step"), 0.5)
        params["wlstm_ep_min_resolved"]  = safe_int(form.get("wlstm_ep_min"),  200)
        params["wlstm_ep_max_resolved"]  = safe_int(form.get("wlstm_ep_max"),  200)
        params["wlstm_ep_step_resolved"] = safe_int(form.get("wlstm_ep_step"), 100)
        params["wlstm_pa_min_resolved"]  = safe_int(form.get("wlstm_pa_min"),  30)
        params["wlstm_pa_max_resolved"]  = safe_int(form.get("wlstm_pa_max"),  30)
        params["wlstm_pa_step_resolved"] = safe_int(form.get("wlstm_pa_step"), 10)

    else:
        params["wlstm_wavelets_resolved"] = [form.get("wlstm_manual_wavelet", "sym8").strip()]
        lv = safe_int(form.get("wlstm_manual_level"), 3)
        params["wlstm_min_level_resolved"] = lv
        params["wlstm_max_level_resolved"] = lv
        lags = safe_int(form.get("wlstm_lags_min"), 3)
        params["wlstm_lags_min_resolved"]  = lags
        params["wlstm_lags_max_resolved"]  = lags
        params["wlstm_lags_step_resolved"] = 1
        hs = safe_int(form.get("wlstm_hs_min"), 8)
        params["wlstm_hs_min_resolved"]  = hs
        params["wlstm_hs_max_resolved"]  = hs
        params["wlstm_hs_step_resolved"] = 1
        nl = safe_int(form.get("wlstm_num_layers_min"), 1)
        params["wlstm_num_layers_min_resolved"]  = nl
        params["wlstm_num_layers_max_resolved"]  = nl
        params["wlstm_num_layers_step_resolved"] = 1
        dr = safe_float(form.get("wlstm_dr_min"), 0.2)
        params["wlstm_dr_min_resolved"]  = dr
        params["wlstm_dr_max_resolved"]  = dr
        params["wlstm_dr_step_resolved"] = 0.1
        lr = safe_float(form.get("wlstm_lr_min"), 0.001)
        params["wlstm_lr_min_resolved"]  = lr
        params["wlstm_lr_max_resolved"]  = lr
        params["wlstm_lr_step_resolved"] = 0.001
        wd = safe_float(form.get("wlstm_wd_min"), 1e-4)
        params["wlstm_wd_min_resolved"]  = wd
        params["wlstm_wd_max_resolved"]  = wd
        params["wlstm_wd_step_resolved"] = 1e-4
        hd = safe_float(form.get("wlstm_hd_min"), 1.0)
        params["wlstm_hd_min_resolved"]  = hd
        params["wlstm_hd_max_resolved"]  = hd
        params["wlstm_hd_step_resolved"] = 0.5
        ep = safe_int(form.get("wlstm_ep_min"), 200)
        params["wlstm_ep_min_resolved"]  = ep
        params["wlstm_ep_max_resolved"]  = ep
        params["wlstm_ep_step_resolved"] = 100
        pa = safe_int(form.get("wlstm_pa_min"), 30)
        params["wlstm_pa_min_resolved"]  = pa
        params["wlstm_pa_max_resolved"]  = pa
        params["wlstm_pa_step_resolved"] = 10

    # ============================================================
    # WAVELET-TRANSFORMER PARAMETERS
    # ============================================================
    # ============================================================
    # WAVELET-TRANSFORMER PARAMETERS
    # ============================================================
    auto_tune_wtransformer = form.get("auto_tune_wtransformer") == "true"
    params["auto_tune_wtransformer"] = auto_tune_wtransformer

    if auto_tune_wtransformer:
        raw_wt = form.get("wtransformer_wavelets", "").strip()
        params["wtransformer_wavelets_resolved"] = [w.strip() for w in raw_wt.split(",") if w.strip()] \
                                                or ["haar", "db2", "db4", "sym8"]
        params["wtransformer_min_level_resolved"]  = safe_int(form.get("wtransformer_min_level"),  1)
        params["wtransformer_max_level_resolved"]  = safe_int(form.get("wtransformer_max_level"),  3)
        params["wtransformer_lags_min_resolved"]   = safe_int(form.get("wtransformer_lags_min"),   2)
        params["wtransformer_lags_max_resolved"]   = safe_int(form.get("wtransformer_lags_max"),   4)
        params["wtransformer_lags_step_resolved"]  = safe_int(form.get("wtransformer_lags_step"),  2)
        params["wtransformer_dm_min_resolved"]     = safe_int(form.get("wtransformer_dm_min"),     8)
        params["wtransformer_dm_max_resolved"]     = safe_int(form.get("wtransformer_dm_max"),     16)
        params["wtransformer_dm_step_resolved"]    = safe_int(form.get("wtransformer_dm_step"),    8)
        params["wtransformer_nhead_min_resolved"]       = safe_int(form.get("wtransformer_nhead_min"),  1)
        params["wtransformer_nhead_max_resolved"]       = safe_int(form.get("wtransformer_nhead_max"),  2)
        params["wtransformer_nhead_step_resolved"]      = safe_int(form.get("wtransformer_nhead_step"), 1)
        params["wtransformer_num_layers_min_resolved"]  = safe_int(form.get("wtransformer_num_layers_min"),  1)
        params["wtransformer_num_layers_max_resolved"]  = safe_int(form.get("wtransformer_num_layers_max"),  2)
        params["wtransformer_num_layers_step_resolved"] = safe_int(form.get("wtransformer_num_layers_step"), 1)
        params["wtransformer_dr_min_resolved"]  = safe_float(form.get("wtransformer_dr_min"),  0.1)
        params["wtransformer_dr_max_resolved"]  = safe_float(form.get("wtransformer_dr_max"),  0.2)
        params["wtransformer_dr_step_resolved"] = safe_float(form.get("wtransformer_dr_step"), 0.1)
        params["wtransformer_lr_min_resolved"]  = safe_float(form.get("wtransformer_lr_min"),  0.001)
        params["wtransformer_lr_max_resolved"]  = safe_float(form.get("wtransformer_lr_max"),  0.003)
        params["wtransformer_lr_step_resolved"] = safe_float(form.get("wtransformer_lr_step"), 0.002)
        params["wtransformer_wd_min_resolved"]  = safe_float(form.get("wtransformer_wd_min"),  1e-4)
        params["wtransformer_wd_max_resolved"]  = safe_float(form.get("wtransformer_wd_max"),  1e-4)
        params["wtransformer_wd_step_resolved"] = safe_float(form.get("wtransformer_wd_step"), 1e-4)
        params["wtransformer_hd_min_resolved"]  = safe_float(form.get("wtransformer_hd_min"),  1.0)
        params["wtransformer_hd_max_resolved"]  = safe_float(form.get("wtransformer_hd_max"),  1.0)
        params["wtransformer_hd_step_resolved"] = safe_float(form.get("wtransformer_hd_step"), 0.5)
        params["wtransformer_ep_min_resolved"]  = safe_int(form.get("wtransformer_ep_min"),  200)
        params["wtransformer_ep_max_resolved"]  = safe_int(form.get("wtransformer_ep_max"),  200)
        params["wtransformer_ep_step_resolved"] = safe_int(form.get("wtransformer_ep_step"), 100)
        params["wtransformer_pa_min_resolved"]  = safe_int(form.get("wtransformer_pa_min"),  30)
        params["wtransformer_pa_max_resolved"]  = safe_int(form.get("wtransformer_pa_max"),  30)
        params["wtransformer_pa_step_resolved"] = safe_int(form.get("wtransformer_pa_step"), 10)

    else:
        params["wtransformer_wavelets_resolved"] = [form.get("wtransformer_manual_wavelet", "sym8").strip()]
        lv = safe_int(form.get("wtransformer_manual_level"), 3)
        params["wtransformer_min_level_resolved"] = lv
        params["wtransformer_max_level_resolved"] = lv
        lags = safe_int(form.get("wtransformer_lags_min"), 3)
        params["wtransformer_lags_min_resolved"]  = lags
        params["wtransformer_lags_max_resolved"]  = lags
        params["wtransformer_lags_step_resolved"] = 1
        dmodel = safe_int(form.get("wtransformer_dm_min"), 8)
        params["wtransformer_dm_min_resolved"]  = dmodel
        params["wtransformer_dm_max_resolved"]  = dmodel
        params["wtransformer_dm_step_resolved"] = 1
        nhead = safe_int(form.get("wtransformer_nhead_min"), 2)
        params["wtransformer_nhead_min_resolved"]  = nhead
        params["wtransformer_nhead_max_resolved"]  = nhead
        params["wtransformer_nhead_step_resolved"] = 1
        nl = safe_int(form.get("wtransformer_num_layers_min"), 1)
        params["wtransformer_num_layers_min_resolved"]  = nl
        params["wtransformer_num_layers_max_resolved"]  = nl
        params["wtransformer_num_layers_step_resolved"] = 1
        dr = safe_float(form.get("wtransformer_dr_min"), 0.1)
        params["wtransformer_dr_min_resolved"]  = dr
        params["wtransformer_dr_max_resolved"]  = dr
        params["wtransformer_dr_step_resolved"] = 0.1
        lr = safe_float(form.get("wtransformer_lr_min"), 0.001)
        params["wtransformer_lr_min_resolved"]  = lr
        params["wtransformer_lr_max_resolved"]  = lr
        params["wtransformer_lr_step_resolved"] = 0.001
        wd = safe_float(form.get("wtransformer_wd_min"), 1e-4)
        params["wtransformer_wd_min_resolved"]  = wd
        params["wtransformer_wd_max_resolved"]  = wd
        params["wtransformer_wd_step_resolved"] = 1e-4
        hd = safe_float(form.get("wtransformer_hd_min"), 1.0)
        params["wtransformer_hd_min_resolved"]  = hd
        params["wtransformer_hd_max_resolved"]  = hd
        params["wtransformer_hd_step_resolved"] = 0.5
        ep = safe_int(form.get("wtransformer_ep_min"), 200)
        params["wtransformer_ep_min_resolved"]  = ep
        params["wtransformer_ep_max_resolved"]  = ep
        params["wtransformer_ep_step_resolved"] = 100
        pa = safe_int(form.get("wtransformer_pa_min"), 30)
        params["wtransformer_pa_min_resolved"]  = pa
        params["wtransformer_pa_max_resolved"]  = pa
        params["wtransformer_pa_step_resolved"] = 10
    # ============================================================
    # SVR PARAMETERS (GRID SEARCH + MANUAL)
    # ============================================================
    
    # Auto-tune checkbox
    params['auto_tune_svr'] = safe_bool(form.get('auto_tune_svr'))  # ✅ FIXED

    if params['auto_tune_svr']:
        # Grid search ranges
        #SVR lags
        params["svr_lags_min"]  = safe_int(form.get("svr_lags_min"),  2)
        params["svr_lags_max"]  = safe_int(form.get("svr_lags_max"),  5)
        params["svr_lags_step"] = safe_int(form.get("svr_lags_step"), 1)
        
        params['svr_C_min'] = safe_float(form.get('svr_C_min'), 0.1)  # ✅ FIXED
        params['svr_C_max'] = safe_float(form.get('svr_C_max'), 3.0)  # ✅ FIXED
        params['svr_C_step'] = safe_float(form.get('svr_C_step'), 0.5)  # ✅ FIXED
    
        params['svr_epsilon_min'] = safe_float(form.get('svr_epsilon_min'), 0.01)  # ✅ FIXED
        params['svr_epsilon_max'] = safe_float(form.get('svr_epsilon_max'), 0.3)  # ✅ FIXED
        params['svr_epsilon_step'] = safe_float(form.get('svr_epsilon_step'), 0.05)  # ✅ FIXED
    
        params['svr_gamma_min'] = safe_float(form.get('svr_gamma_min'), 0.1)  # ✅ FIXED
        params['svr_gamma_max'] = safe_float(form.get('svr_gamma_max'), 1.0)  # ✅ FIXED
        params['svr_gamma_step'] = safe_float(form.get('svr_gamma_step'), 0.2)  # ✅ FIXED
    
        # Kernel checkboxes
        params['svr_kernel_rbf'] = bool(form.get('svr_kernel_rbf'))  # ✅ FIXED
        params['svr_kernel_poly'] = bool(form.get('svr_kernel_poly'))  # ✅ FIXED
        params['svr_kernel_sigmoid'] = bool(form.get('svr_kernel_sigmoid'))  # ✅ FIXED

        # Manual parameters (fallback when auto-tune disabled)
    params["svr_lags"] = safe_int(form.get("svr_lags") or form.get("lags"), 3)  # ✅ FIXED
    params["lags"] = safe_int(form.get("lags"), 3)
    params['kernel'] = form.get('kernel', 'rbf')  # ✅ FIXED
    params['C'] = safe_float(form.get('C'), 1.0)  # ✅ FIXED
    params['epsilon'] = safe_float(form.get('epsilon'), 0.1)  # ✅ FIXED
    gamma_value = form.get('gamma', 'scale')  # ✅ FIXED
    try:
        params['gamma'] = float(gamma_value)
    except ValueError:
        params['gamma'] = gamma_value  # Keep as string ('scale', 'auto')
    
    # ============================================================
    # RANDOM FOREST PARAMETERS (GRID SEARCH + MANUAL)
    # ============================================================
    
    # Auto-tune checkbox
    params["auto_tune_rf"] = safe_bool(form.get("auto_tune_rf"))
    
    # Grid search ranges
    #RF lags
    params["rf_lags_min"]  = safe_int(form.get("rf_lags_min"),  2)
    params["rf_lags_max"]  = safe_int(form.get("rf_lags_max"),  5)
    params["rf_lags_step"] = safe_int(form.get("rf_lags_step"), 1)

    params["rf_n_estimators_min"] = safe_int(form.get("rf_n_estimators_min"), 100)
    params["rf_n_estimators_max"] = safe_int(form.get("rf_n_estimators_max"), 300)
    params["rf_n_estimators_step"] = safe_int(form.get("rf_n_estimators_step"), 100)
    
    params["rf_max_depth_min"] = safe_int(form.get("rf_max_depth_min"), 2)
    params["rf_max_depth_max"] = safe_int(form.get("rf_max_depth_max"), 6)
    params["rf_max_depth_step"] = safe_int(form.get("rf_max_depth_step"), 2)
    params["rf_max_depth_include_none"] = safe_bool(form.get("rf_max_depth_include_none"))
    
    params["rf_min_samples_split_min"] = safe_int(form.get("rf_min_samples_split_min"), 2)
    params["rf_min_samples_split_max"] = safe_int(form.get("rf_min_samples_split_max"), 6)
    params["rf_min_samples_split_step"] = safe_int(form.get("rf_min_samples_split_step"), 2)
    
    params["rf_min_samples_leaf_min"] = safe_int(form.get("rf_min_samples_leaf_min"), 1)
    params["rf_min_samples_leaf_max"] = safe_int(form.get("rf_min_samples_leaf_max"), 2)
    params["rf_min_samples_leaf_step"] = safe_int(form.get("rf_min_samples_leaf_step"), 1)
    
    # Max features checkboxes
    params["rf_max_features_sqrt"] = safe_bool(form.get("rf_max_features_sqrt"))
    params["rf_max_features_log2"] = safe_bool(form.get("rf_max_features_log2"))
    params["rf_max_features_none"] = safe_bool(form.get("rf_max_features_none"))
    
    # Bootstrap checkboxes
    params["rf_bootstrap_true"] = safe_bool(form.get("rf_bootstrap_true"))
    params["rf_bootstrap_false"] = safe_bool(form.get("rf_bootstrap_false"))
    
    # Manual parameters (fallback when auto-tune disabled)
    params["rf_lags"] = safe_int(form.get("rf_lags") or form.get("lags"), 3)  # ✅ FIXED
    params["lags"] = safe_int(form.get("lags"), 3)
    params["n_estimators"] = safe_int(form.get("n_estimators"), 350)
    params["max_depth"] = safe_int(form.get("max_depth"), 3)
    params["min_samples_split"] = safe_int(form.get("min_samples_split"), 6)
    params["min_samples_leaf"] = safe_int(form.get("min_samples_leaf"), 1)
    mf = form.get("max_features", "sqrt")
    try:
        params["max_features"] = int(mf)
    except (ValueError, TypeError):
        params["max_features"] = mf  # keeps "sqrt", "log2", None as-is
    params["bootstrap"] = safe_bool(form.get("bootstrap"))
    params["random_state"] = safe_int(form.get("random_state"), 42)
    
    # ============================================================
    # ARIMA PARAMETERS (if applicable)
    # ============================================================
    params["auto_arima"] = safe_bool(form.get("auto_arima"))  # ✅ ADD THIS LINE
    params["arima_p"] = safe_int(form.get("arima_p"), 1)
    params["arima_d"] = safe_int(form.get("arima_d"), 1)
    params["arima_q"] = safe_int(form.get("arima_q"), 0)

    # ============================================================
    # SARIMA PARAMETERS
    # ============================================================
    auto_sarima = safe_bool(form.get("auto_sarima"))
    params["auto_sarima"] = auto_sarima

    # Both grid section and manual section use same field names.
    # Grid inputs appear FIRST in DOM → getlist()[0]
    # Manual inputs appear LAST  in DOM → getlist()[-1]
    # Pick the correct one based on which mode is active.
    def _sarima_int(field, default):
        vals = form.getlist(field)
        if not vals:
            return default
        raw = vals[0] if auto_sarima else vals[-1]
        return safe_int(raw, default)

    params["sarima_p"] = _sarima_int("sarima_p", 1)
    params["sarima_d"] = _sarima_int("sarima_d", 1)
    params["sarima_q"] = _sarima_int("sarima_q", 0)
    params["sarima_P"] = _sarima_int("sarima_P", 1)
    params["sarima_D"] = _sarima_int("sarima_D", 1)
    params["sarima_Q"] = _sarima_int("sarima_Q", 0)
    params["sarima_s"] = _sarima_int("sarima_s", 12)

    # Non-seasonal aliases for sarima_model.py
    def _coalesce(a, b):
        return a if a is not None else b
    params["arima_p"] = _coalesce(params.get("sarima_p"), params.get("arima_p", 1))
    params["arima_d"] = _coalesce(params.get("sarima_d"), params.get("arima_d", 1))
    params["arima_q"] = _coalesce(params.get("sarima_q"), params.get("arima_q", 0))
    # ============================================================
    # TBATS PARAMETERS
    # ============================================================
    params["auto_tbats"]               = safe_bool(form.get("auto_tbats"))
    params["tbats_seasonal_periods"]   = form.get("tbats_seasonal_periods", "12")
    params["tbats_use_box_cox"]        = None   # always auto
    params["tbats_use_trend"]          = None   # always auto
    params["tbats_use_damped_trend"]   = None   # always auto
    params["tbats_use_arma_errors"]    = True

    params["rw_drift"]           = safe_bool(form.get("rw_drift"))
    params["rw_seasonal"]        = safe_bool(form.get("rw_seasonal"))
    params["rw_seasonal_period"] = safe_int(form.get("rw_seasonal_period"), 12)

    params["auto_ets"]               = safe_bool(form.get("auto_ets"))
    params["ets_error"]              = form.get("ets_error", "add")
    params["ets_trend"]              = form.get("ets_trend", "add")
    params["ets_seasonal"]           = form.get("ets_seasonal", "add")
    params["ets_seasonal_periods"]   = safe_int(form.get("ets_seasonal_periods"), 12)
    # ============================================================
# EGARCH PARAMETERS
# ============================================================
    params["auto_egarch"] = safe_bool(form.get("auto_egarch"))
    params["egarch_p_min"]  = safe_int(form.get("egarch_p_min"),  1)
    params["egarch_p_max"]  = safe_int(form.get("egarch_p_max"),  2)
    params["egarch_p_step"] = safe_int(form.get("egarch_p_step"), 1)
    params["egarch_o_min"]  = safe_int(form.get("egarch_o_min"),  0)
    params["egarch_o_max"]  = safe_int(form.get("egarch_o_max"),  1)
    params["egarch_o_step"] = safe_int(form.get("egarch_o_step"), 1)
    params["egarch_q_min"]  = safe_int(form.get("egarch_q_min"),  1)
    params["egarch_q_max"]  = safe_int(form.get("egarch_q_max"),  2)
    params["egarch_q_step"] = safe_int(form.get("egarch_q_step"), 1)
    params["egarch_p"] = safe_int(form.get("egarch_p"), 1)
    params["egarch_o"] = safe_int(form.get("egarch_o"), 1)
    params["egarch_q"] = safe_int(form.get("egarch_q"), 1)
    params["diff_order"]  = safe_int(form.get("diff_order"),  0)
    params["exog_lags"]   = safe_int(form.get("exog_lags"),   1)
    params["arch_lags"]  = safe_int(form.get("arch_lags"),  5)
# ============================================================
# AR-EGARCH PARAMETERS
# ============================================================
    params["auto_aregarch"] = safe_bool(form.get("auto_aregarch"))
    params["ar_lags_min"]  = safe_int(form.get("ar_lags_min"),  1)
    params["ar_lags_max"]  = safe_int(form.get("ar_lags_max"),  3)
    params["ar_lags_step"] = safe_int(form.get("ar_lags_step"), 1)
    params["ar_lags"] = safe_int(form.get("ar_lags"), 1)

    # ============================================================
# AR-GARCH PARAMETERS
# ============================================================
    params["auto_argarch"] = safe_bool(form.get("auto_argarch"))
# ar_lags, garch_p, garch_q, diff_order, exog_lags, arch_lags already collected — shared

# ============================================================
    # ARMA-GARCH PARAMETERS
    # ============================================================
    params["auto_armagarch"] = safe_bool(form.get("auto_armagarch"))
    params["ma_lags"] = safe_int(form.get("ma_lags"), 1)
    # ar_lags, garch_p, garch_q, garch_p_min/max/step, garch_q_min/max/step,
    # ar_lags_min/max/step, diff_order, exog_lags, arch_lags already collected — shared

    # ============================================================
# GARCH PARAMETERS
# ============================================================
    params["auto_garch"] = safe_bool(form.get("auto_garch"))
    params["garch_p"]    = safe_int(form.get("garch_p"), 1)
    params["garch_q"]    = safe_int(form.get("garch_q"), 1)
# diff_order, exog_lags, arch_lags already collected in EGARCH section — shared

# ============================================================
    # T-GARCH PARAMETERS
    # ============================================================
    params["auto_tgarch"] = safe_bool(form.get("auto_tgarch"))
    params["tgarch_p"]    = safe_int(form.get("tgarch_p"), 1)
    params["tgarch_o"]    = safe_int(form.get("tgarch_o"), 1)
    params["tgarch_q"]    = safe_int(form.get("tgarch_q"), 1)
    # diff_order, exog_lags, arch_lags already collected — shared
    
    # ============================================================
    # AR-TGARCH PARAMETERS
    # ============================================================
    params["auto_argjrgarch"] = safe_bool(form.get("auto_argjrgarch"))
    # ar_lags, tgarch_p, tgarch_o, tgarch_q already collected — shared
    # diff_order, exog_lags, arch_lags already collected — shared
    # ============================================================
    # XGBOOST PARAMETERS (GRID SEARCH + MANUAL)
    # ============================================================
    params["auto_tune_xgb"] = safe_bool(form.get("auto_tune_xgb"))

    params["xgb_lags_min"]  = safe_int(form.get("xgb_lags_min"),  2)
    params["xgb_lags_max"]  = safe_int(form.get("xgb_lags_max"),  5)
    params["xgb_lags_step"] = safe_int(form.get("xgb_lags_step"), 1)

    params["xgb_n_estimators_min"]  = safe_int(form.get("xgb_n_estimators_min"),  50)
    params["xgb_n_estimators_max"]  = safe_int(form.get("xgb_n_estimators_max"),  300)
    params["xgb_n_estimators_step"] = safe_int(form.get("xgb_n_estimators_step"), 50)

    params["xgb_max_depth_min"]  = safe_int(form.get("xgb_max_depth_min"),  2)
    params["xgb_max_depth_max"]  = safe_int(form.get("xgb_max_depth_max"),  6)
    params["xgb_max_depth_step"] = safe_int(form.get("xgb_max_depth_step"), 1)

    params["xgb_lr_min"]  = safe_float(form.get("xgb_lr_min"),  0.01)
    params["xgb_lr_max"]  = safe_float(form.get("xgb_lr_max"),  0.2)
    params["xgb_lr_step"] = safe_float(form.get("xgb_lr_step"), 0.05)

    params["xgb_subsample_min"]  = safe_float(form.get("xgb_subsample_min"),  0.6)
    params["xgb_subsample_max"]  = safe_float(form.get("xgb_subsample_max"),  1.0)
    params["xgb_subsample_step"] = safe_float(form.get("xgb_subsample_step"), 0.2)

    params["xgb_colsample_min"]  = safe_float(form.get("xgb_colsample_min"),  0.6)
    params["xgb_colsample_max"]  = safe_float(form.get("xgb_colsample_max"),  1.0)
    params["xgb_colsample_step"] = safe_float(form.get("xgb_colsample_step"), 0.2)

    # Manual fallback params
    params["xgb_lags"]        = safe_int(form.get("xgb_lags") or form.get("lags"), 3)
    params["n_estimators"]    = safe_int(form.get("n_estimators"),    100)
    params["max_depth"]       = safe_int(form.get("max_depth"),         4)
    params["learning_rate"]   = safe_float(form.get("learning_rate"),  0.1)
    params["subsample"]       = safe_float(form.get("subsample"),       0.8)
    params["colsample_bytree"]= safe_float(form.get("colsample_bytree"),0.8)

    # ============================================================
    # GBM PARAMETERS (GRID SEARCH + MANUAL)
    # ============================================================
    params["auto_tune_gbm"] = safe_bool(form.get("auto_tune_gbm"))

    params["gbm_lags_min"]  = safe_int(form.get("gbm_lags_min"),  2)
    params["gbm_lags_max"]  = safe_int(form.get("gbm_lags_max"),  5)
    params["gbm_lags_step"] = safe_int(form.get("gbm_lags_step"), 1)

    params["gbm_n_estimators_min"]  = safe_int(form.get("gbm_n_estimators_min"),  50)
    params["gbm_n_estimators_max"]  = safe_int(form.get("gbm_n_estimators_max"),  300)
    params["gbm_n_estimators_step"] = safe_int(form.get("gbm_n_estimators_step"), 50)

    params["gbm_max_depth_min"]  = safe_int(form.get("gbm_max_depth_min"),  2)
    params["gbm_max_depth_max"]  = safe_int(form.get("gbm_max_depth_max"),  6)
    params["gbm_max_depth_step"] = safe_int(form.get("gbm_max_depth_step"), 1)

    params["gbm_lr_min"]  = safe_float(form.get("gbm_lr_min"),  0.01)
    params["gbm_lr_max"]  = safe_float(form.get("gbm_lr_max"),  0.2)
    params["gbm_lr_step"] = safe_float(form.get("gbm_lr_step"), 0.05)

    params["gbm_subsample_min"]  = safe_float(form.get("gbm_subsample_min"),  0.6)
    params["gbm_subsample_max"]  = safe_float(form.get("gbm_subsample_max"),  1.0)
    params["gbm_subsample_step"] = safe_float(form.get("gbm_subsample_step"), 0.2)

    # max_features checkboxes
    params["gbm_max_features_sqrt"] = safe_bool(form.get("gbm_max_features_sqrt"))
    params["gbm_max_features_log2"] = safe_bool(form.get("gbm_max_features_log2"))
    params["gbm_max_features_none"] = safe_bool(form.get("gbm_max_features_none"))

    # Manual fallback params
    params["gbm_lags"]          = safe_int(form.get("gbm_lags") or form.get("lags"), 3)
    params["gbm_n_estimators"]  = safe_int(form.get("gbm_n_estimators"),   100)
    params["gbm_max_depth"]     = safe_int(form.get("gbm_max_depth"),         3)
    params["gbm_learning_rate"] = safe_float(form.get("gbm_learning_rate"),  0.1)
    params["gbm_subsample"]     = safe_float(form.get("gbm_subsample"),       0.8)
    params["gbm_max_features"]  = form.get("gbm_max_features", "sqrt")

    # ============================================================
    # KNN PARAMETERS (GRID SEARCH + MANUAL)
    # ============================================================
    params["auto_tune_knn"] = safe_bool(form.get("auto_tune_knn"))

    params["knn_lags_min"]  = safe_int(form.get("knn_lags_min"),  2)
    params["knn_lags_max"]  = safe_int(form.get("knn_lags_max"),  5)
    params["knn_lags_step"] = safe_int(form.get("knn_lags_step"), 1)

    params["knn_k_min"]  = safe_int(form.get("knn_k_min"),  1)
    params["knn_k_max"]  = safe_int(form.get("knn_k_max"),  10)
    params["knn_k_step"] = safe_int(form.get("knn_k_step"), 1)

    params["knn_weights_uniform"]  = safe_bool(form.get("knn_weights_uniform"))
    params["knn_weights_distance"] = safe_bool(form.get("knn_weights_distance"))

    params["knn_metric_euclidean"] = safe_bool(form.get("knn_metric_euclidean"))
    params["knn_metric_manhattan"] = safe_bool(form.get("knn_metric_manhattan"))
    params["knn_metric_minkowski"] = safe_bool(form.get("knn_metric_minkowski"))

    params["knn_p_min"]  = safe_int(form.get("knn_p_min"),  1)
    params["knn_p_max"]  = safe_int(form.get("knn_p_max"),  3)
    params["knn_p_step"] = safe_int(form.get("knn_p_step"), 1)

    # Manual fallback params
    params["knn_lags"]        = safe_int(form.get("knn_lags") or form.get("lags"), 3)
    params["knn_n_neighbors"] = safe_int(form.get("knn_n_neighbors"), 5)
    params["knn_weights"]     = form.get("knn_weights",  "uniform")
    params["knn_metric"]      = form.get("knn_metric",   "euclidean")
    params["knn_p"]           = safe_int(form.get("knn_p"), 2)

    # ============================================================
    # CNN PARAMETERS
    # ============================================================
    params["auto_tune_cnn"] = safe_bool(form.get("auto_tune_cnn"))

    params["cnn_lags_min"]  = safe_int(form.get("cnn_lags_min"),  2)
    params["cnn_lags_max"]  = safe_int(form.get("cnn_lags_max"),  5)
    params["cnn_lags_step"] = safe_int(form.get("cnn_lags_step"), 1)

    params["cnn_num_filters_min"]  = safe_int(form.get("cnn_num_filters_min"),  8)
    params["cnn_num_filters_max"]  = safe_int(form.get("cnn_num_filters_max"),  32)
    params["cnn_num_filters_step"] = safe_int(form.get("cnn_num_filters_step"), 8)

    params["cnn_kernel_size_min"]  = safe_int(form.get("cnn_kernel_size_min"),  3)
    params["cnn_kernel_size_max"]  = safe_int(form.get("cnn_kernel_size_max"),  7)
    params["cnn_kernel_size_step"] = safe_int(form.get("cnn_kernel_size_step"), 2)

    params["cnn_num_layers_min"]  = safe_int(form.get("cnn_num_layers_min"),  1)
    params["cnn_num_layers_max"]  = safe_int(form.get("cnn_num_layers_max"),  3)
    params["cnn_num_layers_step"] = safe_int(form.get("cnn_num_layers_step"), 1)

    params["cnn_dropout_min"]  = safe_float(form.get("cnn_dropout_min"),  0.0)
    params["cnn_dropout_max"]  = safe_float(form.get("cnn_dropout_max"),  0.3)
    params["cnn_dropout_step"] = safe_float(form.get("cnn_dropout_step"), 0.1)

    params["cnn_learning_rate_min"]  = safe_float(form.get("cnn_learning_rate_min"),  0.0005)
    params["cnn_learning_rate_max"]  = safe_float(form.get("cnn_learning_rate_max"),  0.005)
    params["cnn_learning_rate_step"] = safe_float(form.get("cnn_learning_rate_step"), 0.001)

    params["cnn_weight_decay_min"]  = safe_float(form.get("cnn_weight_decay_min"),  1e-5)
    params["cnn_weight_decay_max"]  = safe_float(form.get("cnn_weight_decay_max"),  5e-4)
    params["cnn_weight_decay_step"] = safe_float(form.get("cnn_weight_decay_step"), 1e-4)

    params["cnn_huber_delta_min"]  = safe_float(form.get("cnn_huber_delta_min"),  0.7)
    params["cnn_huber_delta_max"]  = safe_float(form.get("cnn_huber_delta_max"),  1.5)
    params["cnn_huber_delta_step"] = safe_float(form.get("cnn_huber_delta_step"), 0.4)

    params["cnn_patience_min"]  = safe_int(form.get("cnn_patience_min"),  20)
    params["cnn_patience_max"]  = safe_int(form.get("cnn_patience_max"),  50)
    params["cnn_patience_step"] = safe_int(form.get("cnn_patience_step"), 10)

    params["cnn_epochs_min"]  = safe_int(form.get("cnn_epochs_min"),  100)
    params["cnn_epochs_max"]  = safe_int(form.get("cnn_epochs_max"),  300)
    params["cnn_epochs_step"] = safe_int(form.get("cnn_epochs_step"), 100)

    # Manual fallback
    params["cnn_lags"]      = safe_int(form.get("cnn_lags") or form.get("lags"), 3)
    params["num_filters"]   = safe_int(form.get("num_filters"),   16)
    params["kernel_size"]   = safe_int(form.get("kernel_size"),    3)
    params["cnn_dropout"]   = safe_float(form.get("cnn_dropout"), 0.3)
    
    # ============================================================
    # STACKED LSTM PARAMETERS
    # ============================================================
    params["auto_tune_stacked_lstm"] = safe_bool(form.get("auto_tune_stacked_lstm"))

    params["slstm_lags_min"]  = safe_int(form.get("slstm_lags_min"),  2)
    params["slstm_lags_max"]  = safe_int(form.get("slstm_lags_max"),  5)
    params["slstm_lags_step"] = safe_int(form.get("slstm_lags_step"), 1)

    params["slstm_hidden_size_min"]  = safe_int(form.get("slstm_hidden_size_min"),  4)
    params["slstm_hidden_size_max"]  = safe_int(form.get("slstm_hidden_size_max"),  16)
    params["slstm_hidden_size_step"] = safe_int(form.get("slstm_hidden_size_step"), 4)

    params["slstm_num_stacked_layers_min"]  = safe_int(form.get("slstm_num_stacked_layers_min"),  2)
    params["slstm_num_stacked_layers_max"]  = safe_int(form.get("slstm_num_stacked_layers_max"),  5)
    params["slstm_num_stacked_layers_step"] = safe_int(form.get("slstm_num_stacked_layers_step"), 1)

    params["slstm_dropout_min"]  = safe_float(form.get("slstm_dropout_min"),  0.0)
    params["slstm_dropout_max"]  = safe_float(form.get("slstm_dropout_max"),  0.3)
    params["slstm_dropout_step"] = safe_float(form.get("slstm_dropout_step"), 0.1)

    params["slstm_learning_rate_min"]  = safe_float(form.get("slstm_learning_rate_min"),  0.0005)
    params["slstm_learning_rate_max"]  = safe_float(form.get("slstm_learning_rate_max"),  0.005)
    params["slstm_learning_rate_step"] = safe_float(form.get("slstm_learning_rate_step"), 0.001)

    params["slstm_weight_decay_min"]  = safe_float(form.get("slstm_weight_decay_min"),  1e-5)
    params["slstm_weight_decay_max"]  = safe_float(form.get("slstm_weight_decay_max"),  5e-4)
    params["slstm_weight_decay_step"] = safe_float(form.get("slstm_weight_decay_step"), 1e-4)

    params["slstm_huber_delta_min"]  = safe_float(form.get("slstm_huber_delta_min"),  0.7)
    params["slstm_huber_delta_max"]  = safe_float(form.get("slstm_huber_delta_max"),  1.5)
    params["slstm_huber_delta_step"] = safe_float(form.get("slstm_huber_delta_step"), 0.4)

    params["slstm_patience_min"]  = safe_int(form.get("slstm_patience_min"),  20)
    params["slstm_patience_max"]  = safe_int(form.get("slstm_patience_max"),  50)
    params["slstm_patience_step"] = safe_int(form.get("slstm_patience_step"), 10)

    params["slstm_epochs_min"]  = safe_int(form.get("slstm_epochs_min"),  100)
    params["slstm_epochs_max"]  = safe_int(form.get("slstm_epochs_max"),  300)
    params["slstm_epochs_step"] = safe_int(form.get("slstm_epochs_step"), 100)

    # Manual fallback
    params["slstm_lags"]               = safe_int(form.get("slstm_lags") or form.get("lags"), 3)
    params["slstm_hidden_size"]        = safe_int(form.get("slstm_hidden_size"),        8)
    params["slstm_num_stacked_layers"] = safe_int(form.get("slstm_num_stacked_layers"), 3)
    params["slstm_dropout"]            = safe_float(form.get("slstm_dropout"),          0.3)
    params["slstm_learning_rate"]      = safe_float(form.get("slstm_learning_rate"),    0.003)
    params["slstm_weight_decay"]       = safe_float(form.get("slstm_weight_decay"),     1e-4)
    params["slstm_huber_delta"]        = safe_float(form.get("slstm_huber_delta"),      1.0)
    params["slstm_patience"]           = safe_int(form.get("slstm_patience"),           30)
    params["slstm_epochs"]             = safe_int(form.get("slstm_epochs"),             200)

    # ============================================================
    # BIDIRECTIONAL LSTM PARAMETERS
    # ============================================================
    params["auto_tune_bd_lstm"] = safe_bool(form.get("auto_tune_bd_lstm"))

    params["bdlstm_lags_min"]  = safe_int(form.get("bdlstm_lags_min"),  2)
    params["bdlstm_lags_max"]  = safe_int(form.get("bdlstm_lags_max"),  5)
    params["bdlstm_lags_step"] = safe_int(form.get("bdlstm_lags_step"), 1)

    params["bdlstm_hidden_size_min"]  = safe_int(form.get("bdlstm_hidden_size_min"),  4)
    params["bdlstm_hidden_size_max"]  = safe_int(form.get("bdlstm_hidden_size_max"),  16)
    params["bdlstm_hidden_size_step"] = safe_int(form.get("bdlstm_hidden_size_step"), 4)

    params["bdlstm_num_layers_min"]  = safe_int(form.get("bdlstm_num_layers_min"),  1)
    params["bdlstm_num_layers_max"]  = safe_int(form.get("bdlstm_num_layers_max"),  3)
    params["bdlstm_num_layers_step"] = safe_int(form.get("bdlstm_num_layers_step"), 1)

    params["bdlstm_dropout_min"]  = safe_float(form.get("bdlstm_dropout_min"),  0.0)
    params["bdlstm_dropout_max"]  = safe_float(form.get("bdlstm_dropout_max"),  0.3)
    params["bdlstm_dropout_step"] = safe_float(form.get("bdlstm_dropout_step"), 0.1)

    params["bdlstm_learning_rate_min"]  = safe_float(form.get("bdlstm_learning_rate_min"),  0.0005)
    params["bdlstm_learning_rate_max"]  = safe_float(form.get("bdlstm_learning_rate_max"),  0.005)
    params["bdlstm_learning_rate_step"] = safe_float(form.get("bdlstm_learning_rate_step"), 0.001)

    params["bdlstm_weight_decay_min"]  = safe_float(form.get("bdlstm_weight_decay_min"),  1e-5)
    params["bdlstm_weight_decay_max"]  = safe_float(form.get("bdlstm_weight_decay_max"),  5e-4)
    params["bdlstm_weight_decay_step"] = safe_float(form.get("bdlstm_weight_decay_step"), 1e-4)

    params["bdlstm_huber_delta_min"]  = safe_float(form.get("bdlstm_huber_delta_min"),  0.7)
    params["bdlstm_huber_delta_max"]  = safe_float(form.get("bdlstm_huber_delta_max"),  1.5)
    params["bdlstm_huber_delta_step"] = safe_float(form.get("bdlstm_huber_delta_step"), 0.4)

    params["bdlstm_patience_min"]  = safe_int(form.get("bdlstm_patience_min"),  20)
    params["bdlstm_patience_max"]  = safe_int(form.get("bdlstm_patience_max"),  50)
    params["bdlstm_patience_step"] = safe_int(form.get("bdlstm_patience_step"), 10)

    params["bdlstm_epochs_min"]  = safe_int(form.get("bdlstm_epochs_min"),  100)
    params["bdlstm_epochs_max"]  = safe_int(form.get("bdlstm_epochs_max"),  300)
    params["bdlstm_epochs_step"] = safe_int(form.get("bdlstm_epochs_step"), 100)

    # Manual fallback
    params["bdlstm_lags"]         = safe_int(form.get("bdlstm_lags") or form.get("lags"), 3)
    params["bdlstm_hidden_size"]  = safe_int(form.get("bdlstm_hidden_size"),   8)
    params["bdlstm_num_layers"]   = safe_int(form.get("bdlstm_num_layers"),    2)
    params["bdlstm_dropout"]      = safe_float(form.get("bdlstm_dropout"),     0.3)
    params["bdlstm_learning_rate"]= safe_float(form.get("bdlstm_learning_rate"), 0.003)
    params["bdlstm_weight_decay"] = safe_float(form.get("bdlstm_weight_decay"),  1e-4)
    params["bdlstm_huber_delta"]  = safe_float(form.get("bdlstm_huber_delta"),   1.0)
    params["bdlstm_patience"]     = safe_int(form.get("bdlstm_patience"),       30)
    params["bdlstm_epochs"]       = safe_int(form.get("bdlstm_epochs"),         200)

    # ============================================================
    # CONV-LSTM PARAMETERS
    # ============================================================
    params["auto_tune_conv_lstm"] = safe_bool(form.get("auto_tune_conv_lstm"))

    params["clstm_lags_min"]  = safe_int(form.get("clstm_lags_min"),  2)
    params["clstm_lags_max"]  = safe_int(form.get("clstm_lags_max"),  5)
    params["clstm_lags_step"] = safe_int(form.get("clstm_lags_step"), 1)

    params["clstm_num_filters_min"]  = safe_int(form.get("clstm_num_filters_min"),  8)
    params["clstm_num_filters_max"]  = safe_int(form.get("clstm_num_filters_max"),  32)
    params["clstm_num_filters_step"] = safe_int(form.get("clstm_num_filters_step"), 8)

    params["clstm_kernel_size_min"]  = safe_int(form.get("clstm_kernel_size_min"),  3)
    params["clstm_kernel_size_max"]  = safe_int(form.get("clstm_kernel_size_max"),  5)
    params["clstm_kernel_size_step"] = safe_int(form.get("clstm_kernel_size_step"), 2)

    params["clstm_num_conv_layers_min"]  = safe_int(form.get("clstm_num_conv_layers_min"),  1)
    params["clstm_num_conv_layers_max"]  = safe_int(form.get("clstm_num_conv_layers_max"),  3)
    params["clstm_num_conv_layers_step"] = safe_int(form.get("clstm_num_conv_layers_step"), 1)

    params["clstm_lstm_hidden_size_min"]  = safe_int(form.get("clstm_lstm_hidden_size_min"),  4)
    params["clstm_lstm_hidden_size_max"]  = safe_int(form.get("clstm_lstm_hidden_size_max"),  16)
    params["clstm_lstm_hidden_size_step"] = safe_int(form.get("clstm_lstm_hidden_size_step"), 4)

    params["clstm_num_lstm_layers_min"]  = safe_int(form.get("clstm_num_lstm_layers_min"),  1)
    params["clstm_num_lstm_layers_max"]  = safe_int(form.get("clstm_num_lstm_layers_max"),  2)
    params["clstm_num_lstm_layers_step"] = safe_int(form.get("clstm_num_lstm_layers_step"), 1)

    params["clstm_dropout_min"]  = safe_float(form.get("clstm_dropout_min"),  0.0)
    params["clstm_dropout_max"]  = safe_float(form.get("clstm_dropout_max"),  0.3)
    params["clstm_dropout_step"] = safe_float(form.get("clstm_dropout_step"), 0.1)

    params["clstm_learning_rate_min"]  = safe_float(form.get("clstm_learning_rate_min"),  0.0005)
    params["clstm_learning_rate_max"]  = safe_float(form.get("clstm_learning_rate_max"),  0.005)
    params["clstm_learning_rate_step"] = safe_float(form.get("clstm_learning_rate_step"), 0.001)

    params["clstm_weight_decay_min"]  = safe_float(form.get("clstm_weight_decay_min"),  1e-5)
    params["clstm_weight_decay_max"]  = safe_float(form.get("clstm_weight_decay_max"),  5e-4)
    params["clstm_weight_decay_step"] = safe_float(form.get("clstm_weight_decay_step"), 1e-4)

    params["clstm_huber_delta_min"]  = safe_float(form.get("clstm_huber_delta_min"),  0.7)
    params["clstm_huber_delta_max"]  = safe_float(form.get("clstm_huber_delta_max"),  1.5)
    params["clstm_huber_delta_step"] = safe_float(form.get("clstm_huber_delta_step"), 0.4)

    params["clstm_patience_min"]  = safe_int(form.get("clstm_patience_min"),  20)
    params["clstm_patience_max"]  = safe_int(form.get("clstm_patience_max"),  50)
    params["clstm_patience_step"] = safe_int(form.get("clstm_patience_step"), 10)

    params["clstm_epochs_min"]  = safe_int(form.get("clstm_epochs_min"),  100)
    params["clstm_epochs_max"]  = safe_int(form.get("clstm_epochs_max"),  300)
    params["clstm_epochs_step"] = safe_int(form.get("clstm_epochs_step"), 100)

    # Manual fallback
    params["clstm_lags"]             = safe_int(form.get("clstm_lags") or form.get("lags"), 3)
    params["clstm_num_filters"]      = safe_int(form.get("clstm_num_filters"),       16)
    params["clstm_kernel_size"]      = safe_int(form.get("clstm_kernel_size"),         3)
    params["clstm_num_conv_layers"]  = safe_int(form.get("clstm_num_conv_layers"),     2)
    params["clstm_lstm_hidden_size"] = safe_int(form.get("clstm_lstm_hidden_size"),    8)
    params["clstm_num_lstm_layers"]  = safe_int(form.get("clstm_num_lstm_layers"),     1)
    params["clstm_dropout"]          = safe_float(form.get("clstm_dropout"),          0.3)
    params["clstm_learning_rate"]    = safe_float(form.get("clstm_learning_rate"),    0.003)
    params["clstm_weight_decay"]     = safe_float(form.get("clstm_weight_decay"),     1e-4)
    params["clstm_huber_delta"]      = safe_float(form.get("clstm_huber_delta"),      1.0)
    params["clstm_patience"]         = safe_int(form.get("clstm_patience"),           30)
    params["clstm_epochs"]           = safe_int(form.get("clstm_epochs"),             200)  

    # ============================================================
    # DEEP LSTM PARAMETERS
    # ============================================================
    params["auto_tune_deep_lstm"] = safe_bool(form.get("auto_tune_deep_lstm"))

    params["dlstm_lags_min"]  = safe_int(form.get("dlstm_lags_min"),  2)
    params["dlstm_lags_max"]  = safe_int(form.get("dlstm_lags_max"),  5)
    params["dlstm_lags_step"] = safe_int(form.get("dlstm_lags_step"), 1)

    params["dlstm_hidden_size_min"]  = safe_int(form.get("dlstm_hidden_size_min"),  4)
    params["dlstm_hidden_size_max"]  = safe_int(form.get("dlstm_hidden_size_max"),  16)
    params["dlstm_hidden_size_step"] = safe_int(form.get("dlstm_hidden_size_step"), 4)

    params["dlstm_num_layers_min"]  = safe_int(form.get("dlstm_num_layers_min"),  2)
    params["dlstm_num_layers_max"]  = safe_int(form.get("dlstm_num_layers_max"),  6)
    params["dlstm_num_layers_step"] = safe_int(form.get("dlstm_num_layers_step"), 2)

    params["dlstm_dropout_min"]  = safe_float(form.get("dlstm_dropout_min"),  0.0)
    params["dlstm_dropout_max"]  = safe_float(form.get("dlstm_dropout_max"),  0.3)
    params["dlstm_dropout_step"] = safe_float(form.get("dlstm_dropout_step"), 0.1)

    params["dlstm_learning_rate_min"]  = safe_float(form.get("dlstm_learning_rate_min"),  0.0005)
    params["dlstm_learning_rate_max"]  = safe_float(form.get("dlstm_learning_rate_max"),  0.005)
    params["dlstm_learning_rate_step"] = safe_float(form.get("dlstm_learning_rate_step"), 0.001)

    params["dlstm_weight_decay_min"]  = safe_float(form.get("dlstm_weight_decay_min"),  1e-5)
    params["dlstm_weight_decay_max"]  = safe_float(form.get("dlstm_weight_decay_max"),  5e-4)
    params["dlstm_weight_decay_step"] = safe_float(form.get("dlstm_weight_decay_step"), 1e-4)

    params["dlstm_huber_delta_min"]  = safe_float(form.get("dlstm_huber_delta_min"),  0.7)
    params["dlstm_huber_delta_max"]  = safe_float(form.get("dlstm_huber_delta_max"),  1.5)
    params["dlstm_huber_delta_step"] = safe_float(form.get("dlstm_huber_delta_step"), 0.4)

    params["dlstm_patience_min"]  = safe_int(form.get("dlstm_patience_min"),  20)
    params["dlstm_patience_max"]  = safe_int(form.get("dlstm_patience_max"),  50)
    params["dlstm_patience_step"] = safe_int(form.get("dlstm_patience_step"), 10)

    params["dlstm_epochs_min"]  = safe_int(form.get("dlstm_epochs_min"),  100)
    params["dlstm_epochs_max"]  = safe_int(form.get("dlstm_epochs_max"),  300)
    params["dlstm_epochs_step"] = safe_int(form.get("dlstm_epochs_step"), 100)

    # Manual fallback
    params["dlstm_lags"]         = safe_int(form.get("dlstm_lags") or form.get("lags"), 3)
    params["dlstm_hidden_size"]  = safe_int(form.get("dlstm_hidden_size"),   8)
    params["dlstm_num_layers"]   = safe_int(form.get("dlstm_num_layers"),    4)
    params["dlstm_dropout"]      = safe_float(form.get("dlstm_dropout"),     0.3)
    params["dlstm_learning_rate"]= safe_float(form.get("dlstm_learning_rate"), 0.003)
    params["dlstm_weight_decay"] = safe_float(form.get("dlstm_weight_decay"),  1e-4)
    params["dlstm_huber_delta"]  = safe_float(form.get("dlstm_huber_delta"),   1.0)
    params["dlstm_patience"]     = safe_int(form.get("dlstm_patience"),       30)
    params["dlstm_epochs"]       = safe_int(form.get("dlstm_epochs"),         200) 

    
    # ============================================================
    # RETURN COMPLETE PARAMS DICTIONARY
    # ============================================================
    return params

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/models")
def models():
    return render_template("models.html")

#✅ ADD THIS NEW ROUTE
@app.route("/team")
def team():
    """Team page - display team members"""
    return render_template("team.html")

@app.route("/privacy-policy")
def privacy_policy():
    """Privacy Policy page"""
    return render_template("privacy_policy.html")

# =====================================================
# ADD THIS ROUTE TO YOUR app.py FILE
# =====================================================

@app.route("/traffic")
def user_traffic():
    """
    Display global user traffic visualization
    """
    return render_template("user_traffic_globe.html")

def get_country_coords(country):
    try:
        url = f"https://restcountries.com/v3.1/name/{country}"
        r = requests.get(url, timeout=5).json()
        latlng = r[0]["latlng"]
        return latlng[0], latlng[1]
    except:
        return None


@app.route("/api/visitor-countries")
def visitor_countries():
    """
    Return visitor counts by country from Google Analytics
    """
    try:
        PROPERTY_ID = "527652952"

        client = BetaAnalyticsDataClient()

        request = RunReportRequest(
            property=f"properties/{PROPERTY_ID}",
            dimensions=[{"name": "country"}],
            metrics=[{"name": "activeUsers"}],
            date_ranges=[{"start_date": "1daysAgo", "end_date": "today"}],
        )

        response = client.run_report(request)

        data = []

        for row in response.rows:
            country = row.dimension_values[0].value
            users = int(row.metric_values[0].value)

            coords = get_country_coords(country)

            if coords:
                lat, lng = coords

                data.append({
                    "name": country,
                    "count": users,
                    "lat": lat,
                    "lng": lng
                })

        return {"data": data}

    except Exception as e:
        return {"data": [], "error": str(e)}

# =====================================================
# STEP 3 — CONFIGURE HYPERPARAMETERS
# =====================================================
@app.route("/configure", methods=["GET", "POST"])
def configure():
    if request.method == "POST":
        # Get selected models
        selected_models = request.form.getlist("models")
        
        if not selected_models:
            flash("Please select at least one model!", "error")
            return redirect(url_for("models"))
        
        # Store dataset
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("No file uploaded!", "error")
            return redirect(url_for("models"))
        
        # Read and store data in session
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file)
            else:
                data = pd.read_excel(file)
            
            # Store data as JSON
        
            session["data"] = data.to_json(orient="split")
        except Exception as e:
            flash(f"Error reading file: {str(e)}", "error")
            return redirect(url_for("models"))
        
        # Store configuration in session
        session["models"] = selected_models
        session["frequency"] = request.form.get("frequency", "annual")
        session["horizon"] = int(request.form.get("horizon", 1))
        session["window"] = int(request.form.get("window", 4))
        session["lags"] = int(request.form.get("lags", 3))
        session["split"] = float(request.form.get("split", 0.85))
        session.modified = True
        
        # If single model, use existing hyperparams workflow
        if len(selected_models) == 1:
            session["model"] = selected_models[0]
            return redirect(url_for("hyperparams"))
        else:
            # Multiple models - go to comparison hyperparams
            return redirect(url_for("configure_comparison"))
    
    return render_template("models.html")


@app.route("/hyperparams")
def hyperparams():
    """
    Hyperparameter configuration page for single model
    """
    model_name = session.get("model")
    if not model_name:
        flash("No model selected!", "error")
        return redirect(url_for("models"))
    
    return render_template(
        "hyperparams.html",
        model_name=model_name,
        frequency=session.get("frequency", "annual"),
        horizon=session.get("horizon", 1)
    )

#=====================================================
# ADD THESE ROUTES TO YOUR app.py FILE
# =====================================================

# =====================================================
# REPLACE your existing /upload_data route with this
# =====================================================

@app.route("/upload_data", methods=["GET","POST"])
def upload_data():
    
    if request.method == "GET":
        return render_template("models.html")
    # ✅ CLEAR STALE SESSION DATA FROM PREVIOUS UPLOADS
    for key in ["data", "exog_cols", "time_col", "target_col", 
                "comparison_model_keys", "comparison_results",
                "comparison_forecasts", "comparison_id","statistics_results","selected_stats",]:
        session.pop(key, None)
    session.modified = True
    # ── 1. Read yield file ───────────────────────────────────────────────────
    yield_file = request.files.get("dataset_yield")
    if not yield_file or yield_file.filename == "":
        flash("No file uploaded!", "error")
        return redirect(url_for("models"))

    try:
        if yield_file.filename.lower().endswith(".csv"):
            df_yield = pd.read_csv(yield_file)
        else:
            df_yield = pd.read_excel(yield_file)
        df_yield.columns = [c.strip() for c in df_yield.columns]
        # ✅ ADD THESE DEBUG LINES
        print(f"✅ Uploaded file: {yield_file.filename}")
        print(f"✅ Data shape: {df_yield.shape}")
        print(f"✅ First 3 rows:\n{df_yield.head(3)}")
    except Exception as e:
        flash(f"Error reading file: {str(e)}", "error")
        return redirect(url_for("models"))

    # ── 2. Read optional exog file ───────────────────────────────────────────
    exog_file = request.files.get("dataset_exog")
    df_exog = None
    if exog_file and exog_file.filename != "":
        try:
            if exog_file.filename.lower().endswith(".csv"):
                df_exog = pd.read_csv(exog_file)
            else:
                df_exog = pd.read_excel(exog_file)
            df_exog.columns = [c.strip() for c in df_exog.columns]
        except Exception as e:
            flash(f"Error reading exogenous file: {str(e)}", "error")
            return redirect(url_for("models"))

    # ── 3. Get user column mappings ──────────────────────────────────────────
    time_col   = request.form.get("time_col", "").strip()
    target_col = request.form.get("target_col", "").strip()
    exog_cols  = request.form.getlist("exog_cols")  # list of checked exog columns

    # Fallback auto-detection if not provided (e.g. Excel upload)
    if not time_col:
        for col in df_yield.columns:
            if re.search(r'year|month|date|time|period', col, re.I):
                time_col = col
                break
        if not time_col:
            time_col = df_yield.columns[0]  # first column as fallback

    if not target_col:
        for col in df_yield.columns:
            if re.search(r'yield|production|output|sales|revenue', col, re.I):
                target_col = col
                break
        if not target_col:
            target_col = df_yield.columns[-1]  # last column as fallback

    # ── 4. Validate ──────────────────────────────────────────────────────────
    if time_col not in df_yield.columns:
        flash(f"Time column '{time_col}' not found in file. Columns: {list(df_yield.columns)}", "error")
        return redirect(url_for("models"))

    if target_col not in df_yield.columns:
        flash(f"Target column '{target_col}' not found in file. Columns: {list(df_yield.columns)}", "error")
        return redirect(url_for("models"))

    # ── 5. Merge if exog file provided ───────────────────────────────────────
    try:
        if df_exog is not None:
            if time_col not in df_exog.columns:
                # Try to find time col in exog by same pattern
                for col in df_exog.columns:
                    if re.search(r'year|month|date|time|period', col, re.I):
                        df_exog = df_exog.rename(columns={col: time_col})
                        break

            # Auto-detect exog_cols from exog file if not specified
            if not exog_cols:
                exog_cols = [c for c in df_exog.columns if c != time_col]

            # Keep only needed columns
            keep_exog = [time_col] + [c for c in exog_cols if c in df_exog.columns]
            df_exog_slim = df_exog[keep_exog]

            keep_yield = [time_col, target_col]
            df_yield_slim = df_yield[keep_yield]

            # ✅ Use positional alignment instead of name-based merge
            # (month names repeat, so merge would create cartesian explosion)
            min_rows = min(len(df_yield_slim), len(df_exog_slim))
            df_yield_slim = df_yield_slim.reset_index(drop=True).iloc[:min_rows]
            df_exog_slim  = df_exog_slim.reset_index(drop=True).iloc[:min_rows]

            data = pd.concat(
                [df_yield_slim, df_exog_slim.drop(columns=[time_col])],
                axis=1
            )

            if data.empty:
                flash("No common rows found after merging. Check that both files share the same time values.", "error")
                return redirect(url_for("models"))

            flash(f"✅ Merged successfully — {len(data)} rows, {len(exog_cols)} exogenous variable(s).", "success")

        else:
            # Single file — validate it has exactly 2 columns (Time + Target)
            non_time_cols = [c for c in df_yield.columns if c != time_col]

            if len(non_time_cols) > 1:
                flash(
                    f"⚠️ Your Study Variable file has {len(df_yield.columns)} columns "
                    f"({', '.join(df_yield.columns.tolist())}). "
                    f"It should contain only 2 columns: Time + Study Variable. "
                    f"For multivariate analysis, upload exogenous variables in a separate Exog file.",
                    "error"
                )
                return redirect(url_for("models"))

            # Safe: only 2 columns — second one is the target
            if not target_col and non_time_cols:
                target_col = non_time_cols[0]

            data     = df_yield[[time_col, target_col]]
            exog_cols = []  # no exog from single file
            flash(f"✅ Data uploaded — {len(data)} rows, Univariate mode.", "success")

    except Exception as e:
        flash(f"Error processing files: {str(e)}", "error")
        import traceback; traceback.print_exc()
        return redirect(url_for("models"))

    # ── 6. Store in session ──────────────────────────────────────────────────
    data[time_col] = data[time_col].astype(str)  # Ensure time column is string for JSON serialization
    session["data"]       = data.to_json(orient="split")
    session["time_col"]   = time_col
    session["target_col"] = target_col
    session["exog_cols"]  = exog_cols
    session["frequency"]  = request.form.get("frequency", "annual")
    session["horizon"]    = int(request.form.get("horizon", 1))
    session["split"]      = float(request.form.get("split", 0.85))
    session.modified = True

    next_step = request.form.get("next_step", "upload")
    if next_step == "models":
        return redirect(url_for("select_models"))
    elif next_step == "statistics":
        return redirect(url_for("summary_statistics"))
    else:
        flash("✅ Data uploaded successfully! Use the navbar to proceed.", "success")
        return redirect(url_for("upload_data"))
        
    


# =====================================================
# REPLACE your existing /summary_statistics route
# with this one in app.py
# =====================================================
# =====================================================================
# REPLACE ALL your statistics-related routes in app.py with this block
# =====================================================================

# ── helper reused by all three generate routes ────────────────────────
def _get_data_and_multivariate():
    """Load DataFrame from session. Returns (data, is_multivariate) or (None, False)."""
    data_json = session.get("data")
    if not data_json:
        return None, False
    data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    is_multivariate = len(session.get("exog_cols", [])) > 0
    return data, is_multivariate


# ─────────────────────────────────────────────────────────────────────
# SELECTION PAGES  (GET)
# ─────────────────────────────────────────────────────────────────────

@app.route("/summary_statistics")
def summary_statistics():
    """Selection page: only descriptive statistics checkboxes."""
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("models"))
    _, is_multivariate = _get_data_and_multivariate()
    return render_template(
        "summary_statistics.html",
        is_multivariate=is_multivariate,
    )


@app.route("/statistical_tests_select")
def statistical_tests_select():
    """Selection page: normality / stationarity / trend / non-linearity tests."""
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("models"))
    _, is_multivariate = _get_data_and_multivariate()
    return render_template(
        "statistical_tests_select.html",
        is_multivariate=is_multivariate,
    )


@app.route("/data_visualization_select")
def data_visualization_select():
    """Selection page: plot checkboxes."""
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("models"))
    _, is_multivariate = _get_data_and_multivariate()
    return render_template(
        "data_visualization_select.html",
        is_multivariate=is_multivariate,
    )


# ─────────────────────────────────────────────────────────────────────
# GENERATE ROUTES  (POST)
# ─────────────────────────────────────────────────────────────────────

@app.route("/generate_descriptive", methods=["POST"])
def generate_descriptive():
    """Generate ONLY summary / descriptive statistics."""
    data, _ = _get_data_and_multivariate()
    if data is None:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))

    summary_stats = request.form.getlist("summary_stats")
    if not summary_stats:
        flash("Please select at least one statistic.", "warning")
        return redirect(url_for("summary_statistics"))

    try:
        from statistics_generator import generate_all_statistics
        results = generate_all_statistics(
            data=data,
            summary_stats=summary_stats,
            normality_tests=[],
            stationarity_tests=[],
            nonlinearity_tests=[],
            trend_tests=[],
            plots=[],
            time_col=session.get("time_col"),
            target_col=session.get("target_col"),
            exog_cols=session.get("exog_cols", []),
        )

        # Merge into existing session results so each section stays independent
        existing = session.get("statistics_results", {})
        existing.update(results)
        session["statistics_results"] = existing

        existing_sel = session.get("selected_stats", {})
        existing_sel["summary_stats"] = summary_stats
        session["selected_stats"] = existing_sel
        session.modified = True

        flash("Summary statistics generated successfully!", "success")
        return redirect(url_for("descriptive_stats"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error generating statistics: {str(e)}", "error")
        return redirect(url_for("summary_statistics"))


@app.route("/generate_tests", methods=["POST"])
def generate_tests():
    """Generate ONLY statistical tests (normality / stationarity / trend / non-linearity)."""
    data, _ = _get_data_and_multivariate()
    if data is None:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))

    normality_tests    = request.form.getlist("normality_tests")
    stationarity_tests = request.form.getlist("stationarity_tests")
    trend_tests        = request.form.getlist("trend_tests")
    nonlinearity_tests = request.form.getlist("nonlinearity_tests")

    if not any([normality_tests, stationarity_tests, trend_tests, nonlinearity_tests]):
        flash("Please select at least one test.", "warning")
        return redirect(url_for("statistical_tests_select"))

    print("\n" + "=" * 80)
    print("🔍 FLASK ROUTE DEBUG — generate_tests")
    print("=" * 80)
    print(f"normality_tests    : {normality_tests}")
    print(f"stationarity_tests : {stationarity_tests}")
    print(f"trend_tests        : {trend_tests}")
    print(f"nonlinearity_tests : {nonlinearity_tests}")
    print("=" * 80 + "\n")

    try:
        from statistics_generator import generate_all_statistics
        results = generate_all_statistics(
            data=data,
            summary_stats=[],
            normality_tests=normality_tests,
            stationarity_tests=stationarity_tests,
            nonlinearity_tests=nonlinearity_tests,
            trend_tests=trend_tests,
            plots=[],
            time_col=session.get("time_col"),
            target_col=session.get("target_col"),
            exog_cols=session.get("exog_cols", []),
        )

        existing = session.get("statistics_results", {})
        existing.update(results)
        session["statistics_results"] = existing

        existing_sel = session.get("selected_stats", {})
        existing_sel.update({
            "normality_tests":    normality_tests,
            "stationarity_tests": stationarity_tests,
            "trend_tests":        trend_tests,
            "nonlinearity_tests": nonlinearity_tests,
        })
        session["selected_stats"] = existing_sel
        session.modified = True

        flash("Statistical tests completed successfully!", "success")
        return redirect(url_for("statistical_tests"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error running tests: {str(e)}", "error")
        return redirect(url_for("statistical_tests_select"))


@app.route("/generate_plots_only", methods=["POST"])
def generate_plots_only():
    """Generate ONLY the selected plots."""
    data, _ = _get_data_and_multivariate()
    if data is None:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))

    plots = request.form.getlist("plots")
    if not plots:
        flash("Please select at least one plot.", "warning")
        return redirect(url_for("data_visualization_select"))

    try:
        from statistics_generator import generate_all_statistics
        results = generate_all_statistics(
            data=data,
            summary_stats=[],
            normality_tests=[],
            stationarity_tests=[],
            nonlinearity_tests=[],
            trend_tests=[],
            plots=plots,
            time_col=session.get("time_col"),
            target_col=session.get("target_col"),
            exog_cols=session.get("exog_cols", []),
        )

        existing = session.get("statistics_results", {})
        existing.update(results)
        session["statistics_results"] = existing

        existing_sel = session.get("selected_stats", {})
        existing_sel["plots"] = plots
        session["selected_stats"] = existing_sel
        session.modified = True

        flash("Plots generated successfully!", "success")
        return redirect(url_for("data_visualization"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error generating plots: {str(e)}", "error")
        return redirect(url_for("data_visualization_select"))


# ─────────────────────────────────────────────────────────────────────
# RESULTS PAGES  (GET)  — keep your exact existing logic, unchanged
# ─────────────────────────────────────────────────────────────────────

@app.route("/statistics_results")
def show_statistics_results():
    """Overview page — shows everything generated so far."""
    results       = session.get("statistics_results", {})
    selected_stats = session.get("selected_stats", {})
    if not selected_stats:
        flash("No statistics selected.", "error")
        return redirect(url_for("summary_statistics"))
    if not results:
        flash("No statistics generated yet. Please select and generate statistics first.", "info")
        return redirect(url_for("summary_statistics"))
    return render_template(
        "statistics_results.html",
        results=results,
        selected_stats=selected_stats,
        exog_cols=session.get("exog_cols", []),
        is_multivariate=len(session.get("exog_cols", [])) > 0,
        show_section=None,
    )


@app.route("/descriptive_stats")
def descriptive_stats():
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("upload_data"))
    results = session.get("statistics_results", {})
    if not results:
        flash("Please generate statistics first.", "info")
        return redirect(url_for("summary_statistics"))
    if not results.get("summary_stats"):
        flash(
            "No descriptive statistics were generated. "
            "Please go back and tick at least one summary statistic (Mean, Std, etc.).",
            "warning"
        )
    return render_template(
        "statistics_results.html",
        results=results,
        selected_stats=session.get("selected_stats", {}),
        exog_cols=session.get("exog_cols", []),
        is_multivariate=len(session.get("exog_cols", [])) > 0,
        show_section="descriptive",
    )


@app.route("/statistical_tests")
def statistical_tests():
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("upload_data"))
    results = session.get("statistics_results", {})
    if not results:
        flash("Please generate statistics first.", "info")
        return redirect(url_for("statistical_tests_select"))
    has_tests = any(results.get(k) for k in
                    ["normality_tests", "stationarity_tests",
                     "trend_tests", "nonlinearity_tests"])
    if not has_tests:
        flash(
            "No statistical tests were generated. "
            "Please go back and tick at least one test "
            "(Normality, Stationarity, Trend, or Non-Linearity).",
            "warning"
        )
    return render_template(
        "statistics_results.html",
        results=results,
        selected_stats=session.get("selected_stats", {}),
        exog_cols=session.get("exog_cols", []),
        is_multivariate=len(session.get("exog_cols", [])) > 0,
        show_section="tests",
    )


@app.route("/data_visualization")
def data_visualization():
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("upload_data"))
    results = session.get("statistics_results", {})
    if not results:
        flash("Please generate statistics first.", "info")
        return redirect(url_for("data_visualization_select"))
    if not results.get("plots"):
        flash(
            "No plots were generated. "
            "Please go back and tick at least one plot type "
            "(Line Plot, Histogram, Boxplot, etc.).",
            "warning"
        )
    return render_template(
        "statistics_results.html",
        results=results,
        selected_stats=session.get("selected_stats", {}),
        exog_cols=session.get("exog_cols", []),
        is_multivariate=len(session.get("exog_cols", [])) > 0,
        show_section="plots",
    )


# ─────────────────────────────────────────────────────────────────────
# LEGACY ROUTE — kept for backward compatibility only
# (the new pages no longer post here, but old bookmarks still work)
# ─────────────────────────────────────────────────────────────────────

@app.route("/generate_statistics", methods=["POST"])
def generate_statistics():
    """Legacy single-form route — redirects to the correct section."""
    data, _ = _get_data_and_multivariate()
    if data is None:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))

    summary_stats      = request.form.getlist("summary_stats")
    normality_tests    = request.form.getlist("normality_tests")
    stationarity_tests = request.form.getlist("stationarity_tests")
    nonlinearity_tests = request.form.getlist("nonlinearity_tests")
    trend_tests        = request.form.getlist("trend_tests")
    plots              = request.form.getlist("plots")

    print("\n" + "=" * 80)
    print("🔍 FLASK ROUTE DEBUG — generate_statistics (legacy)")
    print("=" * 80)
    print(f"nonlinearity_tests : {nonlinearity_tests}")
    print(f"trend_tests        : {trend_tests}")
    print("=" * 80 + "\n")

    try:
        from statistics_generator import generate_all_statistics
        results = generate_all_statistics(
            data=data,
            summary_stats=summary_stats,
            normality_tests=normality_tests,
            stationarity_tests=stationarity_tests,
            nonlinearity_tests=nonlinearity_tests,
            trend_tests=trend_tests,
            plots=plots,
            time_col=session.get("time_col"),
            target_col=session.get("target_col"),
            exog_cols=session.get("exog_cols", []),
        )

        session["statistics_results"] = results
        session["selected_stats"] = {
            "summary_stats":      summary_stats,
            "normality_tests":    normality_tests,
            "stationarity_tests": stationarity_tests,
            "nonlinearity_tests": nonlinearity_tests,
            "trend_tests":        trend_tests,
            "plots":              plots,
        }
        session.modified = True

        flash("Statistics generated successfully!", "success")
        if summary_stats:
            return redirect(url_for("descriptive_stats"))
        elif any([normality_tests, stationarity_tests, trend_tests, nonlinearity_tests]):
            return redirect(url_for("statistical_tests"))
        elif plots:
            return redirect(url_for("data_visualization"))
        else:
            return redirect(url_for("show_statistics_results"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error generating statistics: {str(e)}", "error")
        return redirect(url_for("summary_statistics"))

# NEW ROUTE: Model selection page (replaces old /configure)
@app.route("/select_models")
def select_models():
    """
    Display model selection page after statistics
    """
    # Check if data exists in session
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("models"))
    
    return render_template("model_selection.html" , target_col=session.get("target_col", ""))


# NEW ROUTE: Handle model selection and go to hyperparameters
@app.route("/configure_models", methods=["POST"])
def configure_models():
    """
    Handle model selection and redirect to hyperparameter configuration
    """
    # Get selected models
    selected_models = request.form.getlist("models")
    
    if not selected_models:
        flash("Please select at least one model!", "error")
        return redirect(url_for("select_models"))
    
    # Store selected models in session
    session["models"] = selected_models
    session.modified = True
    
    # If single model, use existing hyperparams workflow
    if len(selected_models) == 1:
        session["model"] = selected_models[0]
        return redirect(url_for("hyperparams"))
    else:
        # Multiple models - go to comparison hyperparams
        return redirect(url_for("configure_comparison"))
# =====================================================
# Configure comparison route for multiple models#
## =====================================================
@app.route("/configure_comparison")
def configure_comparison():
    """
    Hyperparameter configuration for multiple models
    """
    selected_models = session.get("models", [])
    
    if not selected_models:
        flash("No models selected!", "error")
        return redirect(url_for("index"))
    
    return render_template(
        "hyperparams_comparison.html",
        models=selected_models,
        frequency=session.get("frequency", "annual"),
        horizon=session.get("horizon", 1)
    )
@app.route("/train_single", methods=["POST"])
def train_single():
    """
    Train a single model with progress tracking
    """
    # ✅ CLEAR OLD COMPARISON DATA
    if "comparison_model_keys" in session:
        del session["comparison_model_keys"]
    if "comparison_results" in session:
        del session["comparison_results"]
    session.modified = True

    model_name = session.get("model")
    
    if not model_name:
        flash("No model selected!", "error")
        return redirect(url_for("models"))
    
    # Get data from session
    data_json = session.get("data")
    if not data_json:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))
    
    # Collect hyperparameters
    params = collect_hyperparameters(request.form)
    
    # Override with session values
    params["split"] = session.get("split", 0.85)
    
    #✅ Add column mappings from session
    params["time_col"]   = session.get("time_col", "Year")
    params["target_col"] = session.get("target_col", "Yield")
    params["exog_cols"]  = session.get("exog_cols", [])
    
    horizon = session.get("horizon", 5)
    frequency = session.get("frequency", "annual")
    
    # Generate unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    session['single_training_id'] = session_id
    session.modified = True
    
    # Initialize progress
    update_progress(session_id, {
        'current_model': model_name.upper(),
        'current_step': 'Initializing...',
        'models_completed': 0,
        'total_models': 1,
        'progress_percent': 0,
        'logs': [f'🚀 Starting {model_name.upper()} training...'],
        'status': 'running',
        'data_json': data_json,          # ← ADD THIS LINE
        'exog_cols': session.get("exog_cols", []),   # ← ADD THIS LINE
        'horizon': session.get("horizon", 1),         # ← ADD THIS LINE
    })
    
    # Start training in background thread
    def train_in_background():
        try:
            data = pd.read_json(StringIO(data_json), orient="split",convert_dates=False)
            
            # Update progress
            update_progress(session_id, {
                'current_step': 'Loading model architecture...',
                'logs': get_progress(session_id).get('logs', []) + [f'▶️ Training {model_name.upper()}...']
            })
            
            # Train model with progress tracking
            result = train_model_with_progress(
                model_name, 
                data, 
                params, 
                horizon, 
                frequency, 
                session_id
            )
            
            if result:
                
                # Store full results in training_progress for redirect
                # ✅ In train_in_background(), replace the full_results storage:
                with progress_lock:
                    path = _progress_path(session_id)
                    with open(path, "r") as f:
                        current = json.load(f)
                    current['full_results'] = result   # result is already JSON-safe (DataFrames serialized)
                    with open(path, "w") as f:
                        json.dump(current, f)
                
                # Mark as complete
                update_progress(session_id, {
                    'status': 'complete',
                    'current_step': 'Training complete!',
                    'progress_percent': 100,
                    'models_completed': 1,
                    'logs': get_progress(session_id).get('logs' , []) + [f'✅ {model_name.upper()} training complete!', '📊 Preparing results...']
                })
            else:
                raise Exception("Training failed")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            update_progress(session_id, {
                'status': 'error',
                'error': str(e),
                'logs': get_progress(session_id).get('logs' , [] ) + [f'❌ ERROR: {str(e)}', error_details]
            })
    
    # Start background thread
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
    
    # Redirect to progress page
    return redirect(url_for('show_single_training_progress', session_id=session_id))

@app.route("/single_training_progress/<session_id>")
def show_single_training_progress(session_id):
    """Show single model training progress page"""
    model_name = session.get("model", "Model")
    return render_template('training_progress.html', 
                          session_id=session_id,
                          is_single_model=True,
                          model_name=model_name)

@app.route("/show_single_results")
def show_single_results():
    """Show results for single model training"""
    session_id = request.args.get('session_id')
    model_name = session.get("model")
    
    print(f"🔍 DEBUG: show_single_results called with session_id={session_id}")  # ✅ ADD DEBUG

    progress_data = get_progress(session_id)
    if not session_id or not progress_data:
        flash("Training session not found.", "error")
        return redirect(url_for("models"))
    full_results = progress_data.get('full_results')
    
    if not full_results:
        flash("Results not found.", "error")
        return redirect(url_for("models"))
    # ── If ARCH test failed, render directly ──────────────────
    if full_results.get("arch_significant") is False:
        return render_template(
            "results.html",
            results={
                "model_name": full_results.get("model_name", "AR-EGARCH"),
                "model_key":  model_name,
                "arch_lm_stat":    full_results.get("arch_lm_stat"),
                "arch_lm_pvalue":  full_results.get("arch_lm_pvalue"),
                "arch_significant": False,
                "arch_not_significant_message": full_results.get("arch_not_significant_message"),
            },
            show_future_inputs=False,
            exog_cols=session.get("exog_cols", []),
            ts=int(time.time()),
        )
    
    
    # ✅ Convert JSON strings back to DataFrames for template
    results_for_template = {
        "model_key": model_name,
        "model_name": full_results["model_name"],
        "frequency": session.get("frequency", "annual"),
        "horizon": session.get("horizon", 5),
        "rmse_train": full_results["rmse_train"],
        "mae_train": full_results["mae_train"],
        "mape_train": full_results["mape_train"],
        "rmse_test": full_results["rmse_test"],
        "mae_test": full_results["mae_test"],
        "mape_test": full_results["mape_test"],
        "model_config": full_results["model_config"],
        "auto_tuned": full_results["auto_tuned"],
        "train_table": pd.read_json(StringIO(full_results["train_table"]), orient="split"),  # ✅ Convert JSON to DataFrame
        "test_table": pd.read_json(StringIO(full_results["test_table"]), orient="split"),    # ✅ Convert JSON to DataFrame
        "arch_lm_stat"    : full_results.get("arch_lm_stat"),
        "arch_lm_pvalue"  : full_results.get("arch_lm_pvalue"),
        "arch_significant": full_results.get("arch_significant"),
        "arch_not_significant_message": full_results.get("arch_not_significant_message"),
        "param_estimates_table": pd.read_json(StringIO(full_results["param_estimates_table"]), orient="split")
            if full_results.get("param_estimates_table") else None,
        "info_criteria": full_results.get("info_criteria"),   
    }
    # ✅ NOW save to session (we're in request context here)
    session[f"{model_name}_results"] = {
        "meta": {
            "model_key": model_name,
            "model_name": full_results["model_name"],
            "frequency": session.get("frequency", "annual"),
            "horizon": session.get("horizon", 5),
            "rmse_train": full_results["rmse_train"],
            "mae_train": full_results["mae_train"],
            "mape_train": full_results["mape_train"],
            "rmse_test": full_results["rmse_test"],
            "mae_test": full_results["mae_test"],
            "mape_test": full_results["mape_test"],
            "model_config": full_results["model_config"],
            "auto_tuned": full_results["auto_tuned"],
            "param_estimates_table": full_results.get("param_estimates_table"),
            "info_criteria": full_results.get("info_criteria"),
        },
        "train_table": full_results["train_table"],
        "test_table": full_results["test_table"],

        # ✅ ADD THIS LINE FOR ARIMA
        "trained_params": full_results.get("trained_params") if model_name == "arima" else None,

    }
    session.modified = True

    # ✅ Check if model is univariate (no exogenous variables needed)
    is_univariate = (
        model_name == "arima" or 
        "Univariate" in str(full_results.get("data_type", ""))
    )
    if full_results.get("param_estimates_table"):
        import json
        pt = pd.read_json(StringIO(full_results["param_estimates_table"]), orient="split")
        print(f"🔍 param_estimates_table columns: {list(pt.columns)}")
        print(pt.head())
    return render_template(
        "results.html",
        results=results_for_template,
        show_future_inputs=True,
        is_arima=(model_name == "arima"),
        is_univariate=is_univariate , # ✅ NEW FLAG
        is_wavelet_ann=(model_name in ("wavelet_ann", "wavelet_lstm", "wavelet_transformer")),
        exog_cols=session.get("exog_cols", []),
        ts=int(time.time()),
    )
##########################################################
#Training progress stream for SSE
#############################################################
@app.route("/training_progress_stream")
def training_progress_stream():
    """Server-Sent Events endpoint for training progress"""
    session_id = request.args.get('session_id', 'default')
    
    def generate():
        last_data = None
        last_ping_time = time.time()   # ✅ Time-based, not counter-based

        while True:
            current_data = get_progress(session_id)

            if current_data != last_data:
                send_data = {
                    'current_model':    current_data.get('current_model', ''),
                    'current_step':     current_data.get('current_step', ''),
                    'models_completed': current_data.get('models_completed', 0),
                    'total_models':     current_data.get('total_models', 0),
                    'progress_percent': current_data.get('progress_percent', 0),
                    'status':           current_data.get('status', 'running'),
                    'error':            current_data.get('error'),
                    'logs':             current_data.get('logs', [])[-10:],
                }
                yield f"data: {json.dumps(send_data)}\n\n"
                last_data = current_data.copy()
                last_ping_time = time.time()   # ✅ Reset ping timer on real data

                if current_data.get('status') in ['complete', 'error']:
                    break
            else:
                # ✅ Send keepalive every 15s of silence (not counter-based)
                if time.time() - last_ping_time > 15:
                    yield f"data: {json.dumps({'type': 'keepalive', 'status': 'running'})}\n\n"
                    last_ping_time = time.time()

            time.sleep(0.5)   # ✅ Poll every 0.5s for faster response
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection'        : 'keep-alive',
            'Keep-Alive'        : 'timeout=3600',
            'Transfer-Encoding'     : 'chunked',        # ✅ Add this
            'Content-Type'          : 'text/event-stream; charset=utf-8',
        }
    )
# =====================================================
## Train multiple models for comparison
## =====================================================
@app.route("/train_comparison", methods=["POST"])
def train_comparison():
    """Train multiple models in parallel with progress tracking"""
    selected_models = session.get("models",[])

    #ADD THIS DEBUG LINE
    print(f"🔍 DEBUG: Selected models from session: {selected_models}")
    print(f"🔍 DEBUG: Number of models: {len(selected_models)}")
    
    if not selected_models:
        flash("Please select at least one model.", "error")
        return redirect(url_for("models"))
    
    # ✅ COPY ALL SESSION DATA BEFORE STARTING THREAD
    data_json = session.get("data")
    if not data_json:
        flash("No data found. Please upload data first.", "error")
        return redirect(url_for("models"))
    
    # Get other session data
    comparison_horizon = session.get("horizon", 5)
    comparison_frequency = session.get("frequency", "annual")
    
    # Collect hyperparameters from form
    params = collect_hyperparameters(request.form)
    
    #✅ Add column mappings from session
    params["time_col"]   = session.get("time_col", "Year")
    params["target_col"] = session.get("target_col", "Yield")
    params["exog_cols"]  = session.get("exog_cols", [])
    
    # Generate unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    session['comparison_id'] = session_id

    # Store horizon and frequency in session for later use
    session["comparison_horizon"] = comparison_horizon
    session["comparison_frequency"] = comparison_frequency
    session.modified = True
    
    # Initialize progress
    update_progress(session_id, {
        'current_model': 'Initializing...',
        'current_step': 'Loading data...',
        'models_completed': 0,
        'total_models': len(selected_models),
        'progress_percent': 0,
        'logs': ['🚀 Starting training process...'],
        'status': 'running'
    })
    
    # Start training in background thread
    def train_in_background():
        try:
            
            data = pd.read_json(StringIO(data_json), orient="split",convert_dates=False)
            
            results = {}
            total = len(selected_models)

            # ✅ ADD THIS DEBUG
            print(f"🔍 DEBUG: About to train {total} models: {selected_models}")
            
            for idx, model_name in enumerate(selected_models, 1):
                # Update progress
                update_progress(session_id, {
                    'current_model': f'{model_name.upper()}',
                    'current_step': f'Training model {idx} of {total}...',
                    'models_completed': idx - 1,
                    'progress_percent': ((idx - 1) / total) * 100,
                    'logs': get_progress(session_id).get('logs' , [] ) + [f'\n▶️ Training {model_name.upper()} (Model {idx}/{total})']
                })
                
                # Train model (existing code with progress callbacks)
                result = train_model_with_progress(
                model_name, 
                data.copy(),
                params, 
                comparison_horizon,  # ✅ From outer scope
                comparison_frequency,  # ✅ From outer scope
                session_id
            )
                # ✅ NEW: Only add successful results
                if result is not None:
                    results[model_name] = result
                    # Update completion
                    update_progress(session_id, {
                        'models_completed': idx,
                        'progress_percent': (idx / total) * 100,
                        'logs': get_progress(session_id).get('logs',[]) + [f'✅ {model_name.upper()} training complete!']
                    })
                else:
                    # Model failed - log it but continue
                    update_progress(session_id, {
                        'logs': get_progress(session_id).get('logs',[]) + [f'⚠️ {model_name.upper()} training failed - skipped']
                })
                
            #✅✅✅ ADD THIS DEBUG BLOCK RIGHT HERE (AFTER THE FOR LOOP) ✅✅✅
            print("=" * 80)
            print("🔍 TRAINING LOOP COMPLETE")
            print("=" * 80)
            print(f"Results dict keys: {list(results.keys())}")
            for key in results.keys():
                print(f"  {key}: {list(results[key].keys())}")
            print("=" * 80)
            # ✅✅✅ END OF DEBUG BLOCK ✅✅✅

            #✅✅✅ NEW: SINGLE, SAFE STORAGE WITH DEBUG ✅✅✅
            print("=" * 80)
            print("🔍 ABOUT TO STORE RESULTS IN training_progress")
            print("=" * 80)
            print(f"Results to store: {list(results.keys())}")
            print("=" * 80)

            # Store results using thread lock for safety
            with progress_lock:
                path = _progress_path(session_id)
                with open(path, "r") as f:
                    current = json.load(f)
                current['results'] = results.copy()
                with open(path, "w") as f:
                    json.dump(current, f)
            # ✅✅✅ END OF NEW STORAGE BLOCK ✅✅✅

            
            # Mark as complete
            update_progress(session_id, {
                'status': 'complete',
                'current_model': 'All models trained!',
                'current_step': 'Redirecting to results...',
                'progress_percent': 100,
                'logs': get_progress(session_id).get('logs',[]) + ['\n🎉 All models trained successfully!', '📊 Preparing results...']
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            update_progress(session_id, {
                'status': 'error',
                'error': str(e),
                'logs': get_progress(session_id).get('logs', []) + [f'\n❌ ERROR: {str(e)}', error_details]
            })
    
    # Start background thread
    thread = threading.Thread(target=train_in_background)
    thread.daemon = True
    thread.start()
    
    # Redirect to progress page
    return redirect(url_for('show_training_progress', session_id=session_id))

@app.route("/training_progress/<session_id>")
def show_training_progress(session_id):
    """Show training progress page"""
    return render_template('training_progress.html', session_id=session_id)

def train_model_with_progress(model_name, data, params, horizon, frequency, session_id):
    """Train a model with progress updates"""
    
    print(f"🔍 train_model_with_progress called with model_name: '{model_name}'")

    def log(message):
        """Add log message"""
        current = get_progress(session_id)
        update_progress(session_id, {
            'logs': current.get('logs', []) + [message]
        })
    
    try:
        log(f"  → Initializing {model_name.upper()}...")
        
        if model_name == "lstm":
            log(f"  → Loading LSTM architecture...")
            # ✅ PASS log_callback HERE
            result = run_lstm(data, params, horizon, frequency, mode="train", log_callback=log)
            
        elif model_name == "gru":
            log(f"  → Loading GRU architecture...")
            print(f"🔍 About to call run_gru with data shape: {data.shape}")
            # ✅ PASS log_callback HERE
            result = run_gru(data, params, horizon, frequency, log_callback=log)
            print(f"🔍 run_gru returned successfully")
        
        elif model_name == "rnn":
            log(f"  → Loading RNN architecture...")
            # ✅ PASS log_callback HERE
            result = run_rnn(data, params, horizon, frequency, log_callback=log)
            
        elif model_name == "transformer":
            log(f"  → Loading Transformer architecture...")

            # ✅ Use column mappings passed via params (stored from session)
            target_col = params.get("target_col", "Yield")
            time_col   = params.get("time_col", "Year")
            exog_cols  = params.get("exog_cols", [])

            result = run_transformer(
                data, params, horizon, frequency,
                target_col=target_col,
                time_col=time_col,
                exog_cols=exog_cols,
                mode="train",
                log_callback=log
            )
            
        elif model_name == "ann":
            log(f"  → Loading ANN architecture...")
            # ✅ PASS log_callback HERE
            result = run_ann(data, params, horizon, frequency, log_callback=log)

        elif model_name == "wavelet_ann":
            log(f"  → Loading Wavelet-ANN model...")
            result = run_wavelet_ann(
                data, params, horizon, frequency,
                mode="train", log_callback=log
            )
        
        elif model_name == "wavelet_lstm":
            log(f"  → Loading Wavelet-LSTM model...")
            # Build isolated params for this model — override shared wann_* keys
            wlstm_params = dict(params)
            wlstm_params["wann_wavelets"]             = params.get("wlstm_wavelets_resolved", params.get("wann_wavelets", ["sym8"]))
            wlstm_params["wann_min_level"]            = params.get("wlstm_min_level_resolved", params.get("wann_min_level", 1))
            wlstm_params["wann_max_level"]            = params.get("wlstm_max_level_resolved", params.get("wann_max_level", 3))
            wlstm_params["wann_lags_min"]             = params.get("wlstm_lags_min_resolved",  params.get("wann_lags_min",  2))
            wlstm_params["wann_lags_max"]             = params.get("wlstm_lags_max_resolved",  params.get("wann_lags_max",  4))
            wlstm_params["wann_lags_step"]            = params.get("wlstm_lags_step_resolved", params.get("wann_lags_step", 2))
            wlstm_params["ann_hidden_layer_1_min"]    = params.get("wlstm_hs_min_resolved",    8)
            wlstm_params["ann_hidden_layer_1_max"]    = params.get("wlstm_hs_max_resolved",    16)
            wlstm_params["ann_hidden_layer_1_step"]   = params.get("wlstm_hs_step_resolved",   8)
            wlstm_params["wlstm_num_layers_min"]      = params.get("wlstm_num_layers_min_resolved", 1)
            wlstm_params["wlstm_num_layers_max"]      = params.get("wlstm_num_layers_max_resolved", 2)
            wlstm_params["wlstm_num_layers_step"]     = params.get("wlstm_num_layers_step_resolved", 1)
            wlstm_params["ann_dropout_1_min"]         = params.get("wlstm_dr_min_resolved",  0.2)
            wlstm_params["ann_dropout_1_max"]         = params.get("wlstm_dr_max_resolved",  0.2)
            wlstm_params["ann_dropout_1_step"]        = params.get("wlstm_dr_step_resolved", 0.1)
            wlstm_params["ann_learning_rate_min"]     = params.get("wlstm_lr_min_resolved",  0.001)
            wlstm_params["ann_learning_rate_max"]     = params.get("wlstm_lr_max_resolved",  0.003)
            wlstm_params["ann_learning_rate_step"]    = params.get("wlstm_lr_step_resolved", 0.002)
            wlstm_params["ann_weight_decay_min"]      = params.get("wlstm_wd_min_resolved",  1e-4)
            wlstm_params["ann_weight_decay_max"]      = params.get("wlstm_wd_max_resolved",  1e-4)
            wlstm_params["ann_weight_decay_step"]     = params.get("wlstm_wd_step_resolved", 1e-4)
            wlstm_params["ann_huber_delta_min"]       = params.get("wlstm_hd_min_resolved",  1.0)
            wlstm_params["ann_huber_delta_max"]       = params.get("wlstm_hd_max_resolved",  1.0)
            wlstm_params["ann_huber_delta_step"]      = params.get("wlstm_hd_step_resolved", 0.5)
            wlstm_params["ann_epochs_min"]            = params.get("wlstm_ep_min_resolved",  200)
            wlstm_params["ann_epochs_max"]            = params.get("wlstm_ep_max_resolved",  200)
            wlstm_params["ann_epochs_step"]           = params.get("wlstm_ep_step_resolved", 100)
            wlstm_params["ann_patience_min"]          = params.get("wlstm_pa_min_resolved",  30)
            wlstm_params["ann_patience_max"]          = params.get("wlstm_pa_max_resolved",  30)
            wlstm_params["ann_patience_step"]         = params.get("wlstm_pa_step_resolved", 10)
            result = run_wavelet_lstm(
            data, wlstm_params, horizon, frequency,
            mode="train", log_callback=log
        )

        elif model_name == "wavelet_transformer":
            log(f"  → Loading Wavelet-Transformer model...")
            # Build isolated params for this model
            wt_params = dict(params)
            wt_params["wann_wavelets"]             = params.get("wtransformer_wavelets_resolved", params.get("wann_wavelets", ["sym8"]))
            wt_params["wann_min_level"]            = params.get("wtransformer_min_level_resolved", 1)
            wt_params["wann_max_level"]            = params.get("wtransformer_max_level_resolved", 3)
            wt_params["wann_lags_min"]             = params.get("wtransformer_lags_min_resolved",  2)
            wt_params["wann_lags_max"]             = params.get("wtransformer_lags_max_resolved",  4)
            wt_params["wann_lags_step"]            = params.get("wtransformer_lags_step_resolved", 2)
            wt_params["ann_hidden_layer_1_min"]    = params.get("wtransformer_dm_min_resolved",    8)
            wt_params["ann_hidden_layer_1_max"]    = params.get("wtransformer_dm_max_resolved",    16)
            wt_params["ann_hidden_layer_1_step"]   = params.get("wtransformer_dm_step_resolved",   8)
            wt_params["wtransformer_nhead_min"]    = params.get("wtransformer_nhead_min_resolved",  1)
            wt_params["wtransformer_nhead_max"]    = params.get("wtransformer_nhead_max_resolved",  2)
            wt_params["wtransformer_nhead_step"]   = params.get("wtransformer_nhead_step_resolved", 1)
            wt_params["wtransformer_num_layers_min"]  = params.get("wtransformer_num_layers_min_resolved",  1)
            wt_params["wtransformer_num_layers_max"]  = params.get("wtransformer_num_layers_max_resolved",  2)
            wt_params["wtransformer_num_layers_step"] = params.get("wtransformer_num_layers_step_resolved", 1)
            wt_params["ann_dropout_1_min"]         = params.get("wtransformer_dr_min_resolved",  0.1)
            wt_params["ann_dropout_1_max"]         = params.get("wtransformer_dr_max_resolved",  0.2)
            wt_params["ann_dropout_1_step"]        = params.get("wtransformer_dr_step_resolved", 0.1)
            wt_params["ann_learning_rate_min"]     = params.get("wtransformer_lr_min_resolved",  0.001)
            wt_params["ann_learning_rate_max"]     = params.get("wtransformer_lr_max_resolved",  0.003)
            wt_params["ann_learning_rate_step"]    = params.get("wtransformer_lr_step_resolved", 0.002)
            wt_params["ann_weight_decay_min"]      = params.get("wtransformer_wd_min_resolved",  1e-4)
            wt_params["ann_weight_decay_max"]      = params.get("wtransformer_wd_max_resolved",  1e-4)
            wt_params["ann_weight_decay_step"]     = params.get("wtransformer_wd_step_resolved", 1e-4)
            wt_params["ann_huber_delta_min"]       = params.get("wtransformer_hd_min_resolved",  1.0)
            wt_params["ann_huber_delta_max"]       = params.get("wtransformer_hd_max_resolved",  1.0)
            wt_params["ann_huber_delta_step"]      = params.get("wtransformer_hd_step_resolved", 0.5)
            wt_params["ann_epochs_min"]            = params.get("wtransformer_ep_min_resolved",  200)
            wt_params["ann_epochs_max"]            = params.get("wtransformer_ep_max_resolved",  200)
            wt_params["ann_epochs_step"]           = params.get("wtransformer_ep_step_resolved", 100)
            wt_params["ann_patience_min"]          = params.get("wtransformer_pa_min_resolved",  30)
            wt_params["ann_patience_max"]          = params.get("wtransformer_pa_max_resolved",  30)
            wt_params["ann_patience_step"]         = params.get("wtransformer_pa_step_resolved", 10)
    
            result = run_wavelet_transformer(
                data, wt_params, horizon, frequency,
                mode="train", log_callback=log
            )

        elif model_name == "svr":
            log(f"  → Loading SVR model...")
            # ✅ PASS log_callback HERE
            result = run_svr(data, params, horizon, frequency, mode="train", log_callback=log)
            
        elif model_name == "rf":
            log(f"  → Loading Random Forest model...")
            # ✅ PASS log_callback HERE
            result = run_rf(data, params, horizon, frequency, mode="train", log_callback=log)

        elif model_name == "xgb":
            log(f"  → Loading XGBoost model...")
            result = run_xgb(data, params, horizon, frequency, mode="train", log_callback=log)

        elif model_name == "gbm":
            log(f"  → Loading GBM model...")
            result = run_gbm(data, params, horizon, frequency, mode="train", log_callback=log)
        
        elif model_name == "knn":
            log(f"  → Loading KNN model...")
            result = run_knn(
                data         = data,
                params       = params,
                horizon      = horizon,
                frequency    = frequency,
                mode         = "train",
                log_callback = log,
            )

        elif model_name == "cnn":
            log(f"  → Loading Deep CNN architecture...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_cnn(data, params, horizon, frequency,
                         mode="train", log_callback=log)
        
        elif model_name == "stacked_lstm":
            log(f"  → Loading Stacked LSTM architecture...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_stacked_lstm(data, params, horizon, frequency,
                                  mode="train", log_callback=log)
        
        elif model_name == "bd_lstm":
            log(f"  → Loading Bidirectional LSTM architecture...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_bd_lstm(data, params, horizon, frequency,
                             mode="train", log_callback=log)
        
        elif model_name == "conv_lstm":
            log(f"  → Loading Conv-LSTM architecture...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_conv_lstm(data, params, horizon, frequency,
                               mode="train", log_callback=log)
            
        elif model_name == "deep_lstm":
            log(f"  → Loading Deep LSTM (Residual) architecture...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_deep_lstm(data, params, horizon, frequency,
                               mode="train", log_callback=log)

        elif model_name == "aregarch":
            log(f"  → Loading AR-EGARCH model...")
            params["time_col"]   = params.get("time_col", "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols", [])
            result = run_aregarch(data, params, horizon, frequency,
                                  mode="train", log_callback=log)
        
        elif model_name == "argarch":
            log(f"  → Loading AR-GARCH model...")
            params["time_col"]   = params.get("time_col", "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols", [])
            result = run_argarch(data, params, horizon, frequency,
                                 mode="train", log_callback=log)
        
        elif model_name == "armagarch":
            log(f"  → Loading ARMA-GARCH model...")
            params["time_col"]   = params.get("time_col", "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols", [])
            result = run_armagarch(data, params, horizon, frequency,
                                   mode="train", log_callback=log)
        
        elif model_name == "artgarch":
            log(f"  → Loading AR-TGARCH model...")
            params["time_col"]   = params.get("time_col", "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols", [])
            result = run_artgarch(data, params, horizon, frequency,
                                  mode="train", log_callback=log)
            
        elif model_name == "argjrgarch":
            log(f"  → Loading AR-TGARCH model...")
            params["time_col"]   = params.get("time_col", "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols", [])
            result = run_argjrgarch(data, params, horizon, frequency,
                                  mode="train", log_callback=log)
        
        elif model_name == "arima":
            log(f"  → Loading ARIMA model...")
            # ✅ PASS log_callback HERE
            result = run_arima(data, params, horizon, frequency, mode="train", log_callback=log)

        elif model_name == "sarima":
            log(f"  → Loading SARIMA/SARIMAX model...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            params["exog_cols"]  = params.get("exog_cols",  [])
            result = run_sarima(data, params, horizon, frequency,
                                mode="train", log_callback=log)
        
        elif model_name == "tbats":
            log(f"  → Loading TBATS model...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            result = run_tbats(data, params, horizon, frequency,
                               mode="train", log_callback=log)
        elif model_name == "rw":
            log(f"  → Loading Random Walk model...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            result = run_random_walk(data, params, horizon, frequency,
                                 mode="train", log_callback=log)
        
        elif model_name == "ets":
            log(f"  → Loading ETS model...")
            params["time_col"]   = params.get("time_col",   "Year")
            params["target_col"] = params.get("target_col", "Yield")
            result = run_ets(data, params, horizon, frequency,
                             mode="train", log_callback=log)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        log(f"  ✅ Training complete!")
        log(f"  📊 Test RMSE: {result['rmse_test']}, MAE: {result['mae_test']}")
        
        print(f"🔍 About to return result for {model_name}")  # ✅ ADD THIS
        
         #ADD THIS DEBUG BLOCK
        print(f"🔍 Result keys: {list(result.keys())}")
        print(f"🔍 Has train_table: {'train_table' in result}")
        print(f"🔍 Has test_table: {'test_table' in result}")
        if 'train_table' in result:
            print(f"🔍 train_table type: {type(result['train_table'])}")
        if 'test_table' in result:
            print(f"🔍 test_table type: {type(result['test_table'])}")

        #Return in comparison format
        formatted_result = {
            "model_name": result.get("model_name", model_name.upper()),
            "rmse_train": result["rmse_train"],
            "mae_train": result["mae_train"],
            "mape_train": result["mape_train"],
            "rmse_test": result["rmse_test"],
            "mae_test": result["mae_test"],
            "mape_test": result["mape_test"],
            # AFTER (safe guard)
            "train_table": result["train_table"].to_json(orient="split") if result.get("train_table") is not None else None,
            "test_table":  result["test_table"].to_json(orient="split")  if result.get("test_table")  is not None else None,
            "auto_tuned": result.get("auto_tuned") or params.get("auto_egarch", False) or params.get("auto_aregarch", False) or params.get("auto_argarch", False) or params.get("auto_armagarch", False) or params.get("auto_artgarch", False) or params.get("auto_argjrgarch", False),
            "model_config": result.get("model_config", {}),
            "data_type": result.get("data_type", "Multivariate"),  # ✅ ADD THIS LINE
             # ✅ ADD THIS LINE FOR ARIMA ONLY
            
            "trained_params": params if model_name == "arima" else None,
            "arch_lm_stat"    : float(result.get("arch_lm_stat"))   if result.get("arch_lm_stat")   is not None else None,
            "arch_lm_pvalue"  : float(result.get("arch_lm_pvalue")) if result.get("arch_lm_pvalue") is not None else None,
            "arch_significant": bool(result.get("arch_significant")) if result.get("arch_significant") is not None else None,
            "arch_not_significant_message": result.get("arch_not_significant_message"),
            "param_estimates_table": result.get("param_estimates_table").to_json(orient="split")
                if result.get("param_estimates_table") is not None else None,
            "info_criteria": result.get("info_criteria"),
            "residuals": result.get("residuals"),
            
}       

        # ✅✅✅ ADD THIS DEBUG BLOCK RIGHT HERE ✅✅✅
        print(f"🔍 Formatted result for {model_name}:")
        print(f"   Keys: {list(formatted_result.keys())}")
        print(f"   model_name: {formatted_result['model_name']}")
        print(f"   rmse_test: {formatted_result['rmse_test']}")
        # ✅✅✅ END OF DEBUG BLOCK ✅✅✅

        return formatted_result
        
    except Exception as e:
         # ✅ Make error logging even more aggressive
        import traceback
        import sys
        
        error_msg = f"  ❌ Error training {model_name}: {str(e)}"
        traceback_str = traceback.format_exc()
        
        # Log to progress
        log(error_msg)
        log(f"  📋 Traceback:\n{traceback_str}")
        
        # Print to console
        print("=" * 80)  # ✅ Make it VERY visible
        print(f"❌❌❌ EXCEPTION IN {model_name.upper()} ❌❌❌")
        print("=" * 80)
        print(error_msg)
        print(traceback_str)
        print("=" * 80)
        
        # Also print to stderr
        print(error_msg, file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        
        # Return None to continue with other models
        return None


#################################################################
#ARIMA forecast route (no exogenous inputs needed)
################################################################
@app.route("/generate_arima_forecast", methods=["POST"])
def generate_arima_forecast():
    """
    Generate ARIMA forecast (no exogenous inputs needed)
    """
    # ✅ REBUILD comparison_results from individual session storage
    comparison_results = {}
    model_keys = session.get("comparison_model_keys", [])
    
    for key in model_keys:
        model_data = session.get(f"comparison_{key}")
        if model_data:
            comparison_results[key] = model_data
    
    if not comparison_results or "arima" not in comparison_results:
        flash("ARIMA model not found in comparison results.", "error")
        return redirect(url_for("show_comparison"))
    
    # ✅ CLEAR PREVIOUS FORECASTS - ONLY SHOW ARIMA
    session["comparison_forecasts"] = {}
    session.modified = True

    horizon = session.get("comparison_horizon", 5)
    frequency = session.get("comparison_frequency", "annual")
    
    # Get the original data
    data_json = session.get("data")
    if not data_json:
        flash("Original data not found. Please retrain models.", "error")
        return redirect(url_for("models"))
    
    data = pd.read_json(StringIO(data_json), orient="split",convert_dates=False)
    
    try:
        # Get params from form (or use stored ones)
        params = collect_hyperparameters(request.form)
        
        # Generate ARIMA forecast
        forecast_result = run_arima(
            data=data,  # No data needed for forecast mode
            params=params,
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )
        
        # Store the forecast
        if "comparison_forecasts" not in session:
            session["comparison_forecasts"] = {}
        
        session["comparison_forecasts"]["arima"] = forecast_result["forecast_table"].to_json(orient="split")
        session.modified = True
        
        flash("ARIMA forecast generated successfully!", "success")
        return redirect(url_for("show_forecasts"))
        
    except Exception as e:
        print(f"Error generating ARIMA forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"Error generating ARIMA forecast: {str(e)}", "error")
        return redirect(url_for("show_comparison"))
# =====================================================
## Show comparison Route
## =====================================================
@app.route("/show_comparison")
def show_comparison():
    """
    Display comparison dashboard with forecast inputs
    """
    # ✅ ALWAYS check training_progress FIRST (fresh data), then fall back to session
    comparison_results = None
    session_id = session.get('comparison_id')
    
    
    # Try to get fresh results from training_progress
    progress_data = get_progress(session_id)
    if progress_data:
        fresh_results = progress_data.get('results')
        if fresh_results:
            print("=" * 80)
            print("🔍 FOUND FRESH RESULTS IN training_progress")
            print(f"Fresh results keys: {list(fresh_results.keys())}")
            print("=" * 80)

                # ✅ FIX: Store each model individually to avoid bulk serialization issues
                #Store each model individually with error handling
            # Store in session for persistence
            successfully_stored = []
            for model_key, model_data in fresh_results.items():
                try:
                    print(f"🔍 Storing {model_key} to session...")
                    session[f"comparison_{model_key}"] = model_data
                    successfully_stored.append(model_key)
                    print(f"✅ Stored {model_key}")
                except Exception as e:
                    print(f"❌ Failed to store {model_key}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                # Store in session for future use
                # ✅ FIX: Use deep copy to avoid serialization issues
            import copy
            session["comparison_model_keys"] = successfully_stored
            session.modified = True

            print(f"🔍 Successfully stored models: {successfully_stored}")
            print("=" * 80)
                
    
    #If not in training_progress, try to rebuild from session
    if not comparison_results:
        print("🔍 No fresh results, rebuilding from session...")
        model_keys = session.get("comparison_model_keys", [])
        if model_keys:
            comparison_results = {}
            for key in model_keys:
                model_data = session.get(f"comparison_{key}")
                if model_data:
                    comparison_results[key] = model_data
    
    print("=" * 80)
    print("🔍 FINAL comparison_results")
    print(f"Keys: {list(comparison_results.keys()) if comparison_results else 'None'}")
    print("=" * 80)

    if not comparison_results:
        flash("No comparison data available. Please train models first.", "error")
        return redirect(url_for("models"))
    
    # Get the original data for calculating historical averages
    data_json = session.get("data")
    historical_avg_rain = 50.0  # Default
    historical_avg_temp = 25.0  # Default
    last_rain = 45.0  # Default
    last_temp = 28.0  # Default
    
    if data_json:
        try:
            data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
            
            # Calculate historical averages
            #✅ NEW
            exog_cols = session.get("exog_cols", [])
            if exog_cols and len(exog_cols) >= 2:
                historical_avg_rain = float(data[exog_cols[1]].mean())  # 2nd exog col
                historical_avg_temp = float(data[exog_cols[0]].mean())  # 1st exog col
        except Exception as e:
            print(f"Warning: Could not calculate historical data: {str(e)}")
    
    # Prepare data for template
    models_data = []
    for model_key, data in comparison_results.items():
        # Deserialize param_estimates_table from JSON string back to DataFrame
        pet_json = data.get("param_estimates_table")
        pet_df = pd.read_json(StringIO(pet_json), orient="split") if pet_json else None

        models_data.append({
            "key": model_key,
            "name": data["model_name"],
            "rmse_train": data["rmse_train"],
            "mae_train": data["mae_train"],
            "mape_train": data["mape_train"],
            "rmse_test": data["rmse_test"],
            "mae_test": data["mae_test"],
            "mape_test": data["mape_test"],
            "auto_tuned": data["auto_tuned"],
            "model_config": data.get("model_config", {}),
            "arch_significant": data.get("arch_significant"),
            "arch_lm_stat":    data.get("arch_lm_stat"),
            "arch_lm_pvalue":  data.get("arch_lm_pvalue"),
            "arch_not_significant_message": data.get("arch_not_significant_message"),
            "param_estimates_table": pet_df,
        })
    
    # Sort by test RMSE (best first)
    # Then filter best_model to only fitted models:
    fitted_models = [m for m in models_data if m["rmse_test"] is not None]
    best_model = fitted_models[0] if fitted_models else None
    
    
    
    #Detect univariate models
    # Detect univariate vs multivariate models
    univariate_keys   = []
    multivariate_keys = []

    for model_key, model_data in comparison_results.items():
        if model_data.get("arch_significant") is False:
            continue
        data_type = model_data.get("data_type", "Multivariate")
        if "Univariate" in str(data_type):
            univariate_keys.append(model_key)
        else:
            multivariate_keys.append(model_key)

    has_multivariate = bool(multivariate_keys)
    has_univariate   = bool(univariate_keys)
    
    return render_template(
        "compare.html",
        models=models_data,
        best_model=best_model,
        horizon=session.get("comparison_horizon", 5),
        frequency=session.get("comparison_frequency", "yearly"),
        # ✅ ADD THESE NEW PARAMETERS
        historical_avg_rain=historical_avg_rain,
        historical_avg_temp=historical_avg_temp,
        last_rain=last_rain,
        last_temp=last_temp,
        has_multivariate=has_multivariate,
        has_univariate=has_univariate,
        all_univariate=(not has_multivariate),
        univariate_model_keys=univariate_keys,
        multivariate_model_keys=multivariate_keys, #ADD THIS FLAG TO INDICATE IF ALL MODELS ARE UNIVARIATE
        exog_cols=session.get("exog_cols", []),
        ts=int(time.time()),
    )

# =====================================================
## Detailed comparison route for tables and forecasts
## =====================================================

@app.route("/detailed_comparison")
def detailed_comparison():
    """
    Show detailed prediction tables for all models
    """
    # ✅ REBUILD comparison_results from individual session storage
    comparison_results = {}
    model_keys = session.get("comparison_model_keys", [])
    
    for key in model_keys:
        model_data = session.get(f"comparison_{key}")
        if model_data:
            comparison_results[key] = model_data
    
    if not comparison_results:
        flash("No comparison data available.", "error")
        return redirect(url_for("models"))
    
    # Prepare detailed data
    models_data = []
    for model_key, data in comparison_results.items():
        if data.get("train_table") is None:   # skip ARCH-failed
            continue
        models_data.append({
            "key": model_key,
            "name": data["model_name"],
            "train_table": pd.read_json(StringIO(data["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(data["test_table"]), orient="split"),
            "rmse_test": data["rmse_test"],
        })
    
    # Sort by RMSE
    # Then filter best_model to only fitted models:
    fitted_models = [m for m in models_data if m["rmse_test"] is not None]
    best_model = fitted_models[0] if fitted_models else None
    
    return render_template("detailed_comparison.html", models=models_data)

# =====================================================
## Forecasting route for compared models
# =====================================================
@app.route("/forecast_comparison")
def forecast_comparison():
    """
    Forecasting page for compared models
    """
    comparison_results = session.get("comparison_results")
    
    if not comparison_results:
        flash("No comparison data available.", "error")
        return redirect(url_for("models"))
    
    horizon = session.get("comparison_horizon", 5)
    frequency = session.get("comparison_frequency", "annual")
    
    # Get list of models
    models_list = list(comparison_results.keys())
    
    return render_template(
        "forecast_comparison.html",
        models=models_list,
        horizon=horizon,
        frequency=frequency
    )


@app.route("/generate_forecasts", methods=["POST"])
def generate_forecasts():
    """
    Generate forecasts for all compared models using new input format
    Supports both multivariate (with climate inputs) and univariate (no inputs) modes
    """
    # ✅ REBUILD comparison_results from individual session storage
    comparison_results = {}
    model_keys = session.get("comparison_model_keys", [])
    
    for key in model_keys:
        model_data = session.get(f"comparison_{key}")
        if model_data:
            comparison_results[key] = model_data
    
    if not comparison_results:
        flash("No comparison data available.", "error")
        return redirect(url_for("models"))
    
    horizon = session.get("comparison_horizon", 5)
    frequency = session.get("comparison_frequency", "annual")
    
    # ✅ CLEAR PREVIOUS FORECASTS
    session["comparison_forecasts"] = {}
    session.modified = True

    # ✅ NEW: Check if this is univariate mode
    univariate_mode = request.form.get("univariate_mode", "false") == "true"
    
    # Get list of models from form
    models_str = request.form.get("models", "")
    if not models_str:
        flash("No models specified.", "error")
        return redirect(url_for("show_comparison"))
    
    model_names = [m.strip() for m in models_str.split(',') if m.strip()]
    # Filter out ARCH-failed models
    valid_model_names = []
    for m in model_names:
        model_data = session.get(f"comparison_{m}", {})
        if model_data.get("arch_significant") is not False:
            valid_model_names.append(m)
        else:
            print(f"⏭️ Skipping {m} — ARCH test not significant")
    model_names = valid_model_names
    
    # ✅ NEW: Handle univariate mode
    if univariate_mode:
        print(f"\n{'='*60}")
        print(f"🔍 UNIVARIATE MODE: Generating forecasts for {len(model_names)} models")
        print(f"Models: {model_names}")
        print(f"{'='*60}\n")
        
        # Get original data for forecasting
        data_json = session.get("data")
        if not data_json:
            flash("Original data not found. Please retrain models.", "error")
            return redirect(url_for("models"))
        
        data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
        
        # Collect hyperparameters
        params = collect_hyperparameters(request.form)
        
        # Generate forecasts for each univariate model
        forecasts = {}
        
        for model_name in model_names:
            print(f"\n{'─'*60}")
            print(f"Forecasting {model_name.upper()} (univariate)...")
            print(f"{'─'*60}")
            
            try:
                # Run the appropriate model in forecast mode (NO climate inputs)
                if model_name == "lstm":
                    forecast_result = run_lstm(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "gru":
                    forecast_result = run_gru(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "rnn":
                    forecast_result = run_rnn(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast",
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "transformer":
                    forecast_result = run_transformer(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        future_exog=None,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "ann":
                    forecast_result = run_ann(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast",
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "wavelet_ann":
                    forecast_result = run_wavelet_ann(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "wavelet_lstm":
                    forecast_result = run_wavelet_lstm(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "wavelet_transformer":
                    forecast_result = run_wavelet_transformer(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "svr":
                    forecast_result = run_svr(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "rf":
                    forecast_result = run_rf(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "xgb":
                    forecast_result = run_xgb(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]

                elif model_name == "gbm":
                    forecast_result = run_gbm(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "knn":
                    forecast_result = run_knn(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]

                elif model_name == "cnn":
                    forecast_result = run_cnn(
                        data=None,
                        params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "stacked_lstm":
                    forecast_result = run_stacked_lstm(
                        data=None,
                        params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]

                elif model_name == "bd_lstm":
                    forecast_result = run_bd_lstm(
                        data=None,
                        params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]

                elif model_name == "conv_lstm":
                    forecast_result = run_conv_lstm(
                        data=None,
                        params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "deep_lstm":
                    forecast_result = run_deep_lstm(
                        data=None,
                        params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                    
                elif model_name == "arima":
                    forecast_result = run_arima(
                        data=None,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "sarima":
                    forecast_result = run_sarima(
                        data=data,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "tbats":
                    forecast_result = run_tbats(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "rw":
                    forecast_result = run_random_walk(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "ets":
                    forecast_result = run_ets(
                        data=None, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                

                elif model_name == "aregarch":
                    forecast_result = run_aregarch(
                    data=data,
                    params=params,
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                    forecasts[model_name] = forecast_result["forecast_table"]
                


                elif model_name == "argarch":
                    forecast_result = run_argarch(
                    data=data,
                    params=params,
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                    forecasts[model_name] = forecast_result["forecast_table"]

                elif model_name == "armagarch":
                    forecast_result = run_armagarch(
                        data=data,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "artgarch":
                    forecast_result = run_artgarch(
                        data=data, params=params,
                        horizon=horizon, frequency=frequency, mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]
                
                elif model_name == "argjrgarch":
                    forecast_result = run_argjrgarch(
                        data=data,
                        params=params,
                        horizon=horizon,
                        frequency=frequency,
                        mode="forecast"
                    )
                    forecasts[model_name] = forecast_result["forecast_table"]

                print(f"✅ {model_name.upper()} forecast generated successfully!")
                
            except Exception as e:
                print(f"❌ Error forecasting {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                flash(f"Error generating forecast for {model_name}: {str(e)}", "error")
                continue
        
        if not forecasts:
            flash("No forecasts were successfully generated.", "error")
            return redirect(url_for("show_comparison"))
        
        # Store forecasts in session
        session["comparison_forecasts"] = {
            k: v.to_json(orient="split") for k, v in forecasts.items()
        }
        session.modified = True
        
        print(f"\n{'='*60}")
        print(f"UNIVARIATE FORECAST GENERATION COMPLETE - {len(forecasts)} models")
        print(f"{'='*60}\n")
        
        return redirect(url_for("show_forecasts"))
    
    # ✅ EXISTING: Multivariate mode (with climate inputs)
    # ... rest of your existing multivariate code below ...
    
    # ✅ NEW: Collect rain and temperature from individual inputs
    # ✅ Collect exog values dynamically from form (per-column, per-period)
    exog_cols = session.get("exog_cols", [])
    future_exog_dict = {}

    for col in exog_cols:
        vals = []
        for i in range(1, horizon + 1):
            v = request.form.get(f"future_exog_{col}_{i}", "").strip()
            if not v:
                flash(f"Missing input for '{col}' period {i}. Please fill all fields.", "error")
                return redirect(url_for("show_comparison"))
            try:
                vals.append(float(v))
            except ValueError:
                flash(f"Invalid number for '{col}' period {i}.", "error")
                return redirect(url_for("show_comparison"))
        future_exog_dict[col] = vals
    
    # Filter out ARIMA (it has its own route)
    multivariate_models = [m for m in model_names if m != 'arima']
    
    if not multivariate_models:
        flash("No multivariate models to forecast. Use the ARIMA button for univariate forecasting.", "info")
        return redirect(url_for("show_comparison"))
    
    # Get original data for forecasting
    data_json = session.get("data")
    if not data_json:
        flash("Original data not found. Please retrain models.", "error")
        return redirect(url_for("models"))
    
    data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    
    # Collect hyperparameters
    params = collect_hyperparameters(request.form)
    
    # Generate forecasts for each multivariate model
    forecasts = {}
    
    print(f"\n{'='*60}")
    print(f"GENERATING MULTIVARIATE FORECASTS FOR {len(multivariate_models)} MODELS")
    print(f"{'='*60}\n")
    
    for model_name in multivariate_models:
        print(f"\n{'─'*60}")
        print(f"Forecasting {model_name.upper()}...")
        print(f"{'─'*60}")
        
        try:
            # Create future exog DataFrame
            #Fix — use session exog_cols:
            # future_exog_dict already built above (per-column, per-period)
            future_exog = pd.DataFrame(future_exog_dict) if future_exog_dict else None

            
            # Run the appropriate model in forecast mode
            if model_name == "transformer":
                forecast_result = run_transformer(
                    data=None,
                    params=params,
                    horizon=horizon,
                    frequency=frequency,
                    future_exog=future_exog,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "lstm":
                forecast_result = run_lstm(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "gru":
                forecast_result = run_gru(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "rnn":
                forecast_result = run_rnn(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "ann":
                forecast_result = run_ann(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]

            elif model_name == "wavelet_ann":
                forecast_result = run_wavelet_ann(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "wavelet_lstm":
                forecast_result = run_wavelet_lstm(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "wavelet_transformer":
                forecast_result = run_wavelet_transformer(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "svr":
                forecast_result = run_svr(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
                
            elif model_name == "rf":
                forecast_result = run_rf(
                    data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "xgb":
                forecast_result = run_xgb(
                data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "gbm":
                forecast_result = run_gbm(
                data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]

            elif model_name == "knn":
                forecast_result = run_knn(
                data=None,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]

            elif model_name == "cnn":
                forecast_result = run_cnn(
                data=None,
                    params={**params, "future_exog": future_exog_dict} if future_exog_dict else params,
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "stacked_lstm":
                forecast_result = run_stacked_lstm(
                data=None,
                    params={**params, "future_exog": future_exog_dict} if future_exog_dict else params,
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]

            elif model_name == "bd_lstm":
                forecast_result = run_bd_lstm(
                data=None,
                    params={**params, "future_exog": future_exog_dict} if future_exog_dict else params,
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "conv_lstm":
                forecast_result = run_conv_lstm(
                data=None,
                    params={**params, "future_exog": future_exog_dict} if future_exog_dict else params,
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]

            elif model_name == "deep_lstm":
                forecast_result = run_deep_lstm(
                data=None,
                    params={**params, "future_exog": future_exog_dict} if future_exog_dict else params,
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            

            elif model_name == "aregarch":
                forecast_result = run_aregarch(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            

            
            elif model_name == "argarch":
                forecast_result = run_argarch(
                data=data,
                params={**params, "future_exog": future_exog_dict},
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "sarima":
                forecast_result = run_sarima(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "ets":
                forecast_result = run_ets(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "armagarch":
                forecast_result = run_armagarch(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            elif model_name == "artgarch":
                forecast_result = run_artgarch(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon, frequency=frequency, mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            

            
            elif model_name == "argjrgarch":
                forecast_result = run_argjrgarch(
                    data=data,
                    params={**params, "future_exog": future_exog_dict},
                    horizon=horizon,
                    frequency=frequency,
                    mode="forecast"
                )
                forecasts[model_name] = forecast_result["forecast_table"]
            
            print(f"✅ {model_name.upper()} forecast generated successfully!")
            
        except Exception as e:
            print(f"❌ Error forecasting {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f"Error generating forecast for {model_name}: {str(e)}", "error")
            continue
    
    if not forecasts:
        flash("No forecasts were successfully generated.", "error")
        return redirect(url_for("show_comparison"))
    
    # Store forecasts in session
    session["comparison_forecasts"] = {
        k: v.to_json(orient="split") for k, v in forecasts.items()
    }
    session["forecast_exog_inputs"] = future_exog_dict
    session.modified = True
    
    print(f"\n{'='*60}")
    print(f"MULTIVARIATE FORECAST GENERATION COMPLETE - {len(forecasts)} models")
    print(f"{'='*60}\n")
    
    return redirect(url_for("show_forecasts"))


@app.route("/show_forecasts")
def show_forecasts():
    """
    Display forecasts for all models
    """
    forecasts_json = session.get("comparison_forecasts")
    
    if not forecasts_json:
        flash("No forecasts available.", "error")
        return redirect(url_for("forecast_comparison"))
    
    forecasts = {
        k: pd.read_json(StringIO(v), orient="split")
        for k, v in forecasts_json.items()
    }
    
    return render_template(
        "show_forecasts.html",
        forecasts=forecasts,
        time_col=session.get("time_col", "Year")
    )
# =====================================================
# STEP 4 — FINAL PREDICTION (FIXED, STABLE)
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():

    # --------------------
    # RESTORE DATASET
    # --------------------
    data_json = session.get("data")
    if data_json is None:
        return "Session expired. Please upload the dataset again."

    data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)

    model_name = request.form["model"]
    #NEW - fallback to session values
    horizon = session.get("horizon") or int(request.form.get("horizon", 5))
    frequency = request.form.get("frequency") or session.get("frequency", "annual")

    params = collect_hyperparameters(request.form)

    # ==================================================
    # TRANSFORMER (NO METRIC DRIFT)
    # ==================================================
    if model_name == "transformer":
    
    # ✅ CHECK FOR UNIVARIATE FORECAST FIRST
        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
    
        # ✅ Collect future exog values dynamically
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        if exog_cols:
            for col in exog_cols:
                raw = request.form.get(f"future_exog_{col}", "").strip()
                if not raw:
                    return f"Missing future values for '{col}'.", 400
                try:
                    vals = list(map(float, raw.split()))
                except ValueError:
                    return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                if len(vals) != horizon:
                    return f"Enter exactly {horizon} values for '{col}'.", 400
                future_exog_dict[col] = vals

            future_exog = pd.DataFrame(future_exog_dict)
        else:
            future_exog = None

    # ==================================================
    # STEP 1 — TRAIN + EVALUATE (ONCE)
    # ==================================================
         #Fix:
        if not is_univariate_forecast and not exog_cols:
            results = run_transformer(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                target_col=session.get("target_col", "Yield"),  # whatever user named it
                time_col=session.get("time_col", "Year"),        # whatever user named it
                exog_cols=session.get("exog_cols", []),
                future_exog=None,
                mode="train"
            )
            # 🔐 STORE ONLY JSON-SAFE OBJECTS
            session["transformer_results"] = {
                "meta": {
                "model_key": "transformer",
                "model_name": "Transformer (Encoder)",
                "frequency": results["frequency"],
                "horizon": results["horizon"],

                "rmse_train": results["rmse_train"],
                "mae_train": results["mae_train"],
                "mape_train": results["mape_train"],

                "rmse_test": results["rmse_test"],
                "mae_test": results["mae_test"],
                "mape_test": results["mape_test"],
        
                # ✅ ADD THESE TWO LINES
                "model_config": results["model_config"],
                "auto_tuned": results["auto_tuned"],
            },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
            }
            session.modified = True  # ✅ Force session save
            


            # ✅ Check if this is univariate data
            is_univariate = (results.get("data_type") == "Univariate")
            
            exog_cols = session.get("exog_cols", [])
            
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True, # ✅ Hide if univariate!
                is_univariate=is_univariate,  # ✅ NEW FLAG
                exog_cols=exog_cols,
            )
        
        # ==================================================
        # STEP 2 — FORECAST (METRICS FROZEN)
        # ==================================================
                  
        # ==================================================
        # STEP 2 — FORECAST (METRICS FROZEN)
        # ==================================================
        
        # ✅ Check if this is univariate forecasting
        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()

        if is_univariate_forecast:
    # Get session data
            stored = session.get("transformer_results")
            if stored is None:
                return "Session expired. Please run the model again."
    
    # Rebuild results structure
            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }
    
        # Generate univariate forecast
            forecast_only = run_transformer(
                data=None,
                params=params,
                horizon=horizon,
                frequency=None,
                future_exog=None,
                mode="forecast"
            )
    
            forecast_df = forecast_only["forecast_table"].copy()
            results["forecast_table"] = forecast_df
            session["transformer_forecast"] = forecast_df.to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='transformer'))
        
        # Multivariate forecast - process climate inputs
        # ✅ REPLACE WITH (uses future_exog already built at top of transformer block):
        if future_exog is None:
            return "No exogenous inputs provided.", 400

        exog_signature = future_exog.to_json()
        
        stored = session.get("transformer_results")
        if stored is None:
            return "Session expired. Please run the model again."

        # 🔁 REBUILD RESULTS STRUCTURE
        

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
}
        # ==================================================
        # 🔑 RECOMPUTE ONLY IF FUTURE INPUTS CHANGED
        # ==================================================
        if (
            session.get("last_future_exog") != exog_signature
            or session.get("transformer_forecast") is None
        ):

        # ⚠️ Run transformer ONLY to get forecast_table
            forecast_only = run_transformer(
                data=None,  # No data needed
                params=params,
                horizon=horizon,
                frequency=None,
                future_exog=future_exog,
                mode="forecast" # Special mode to skip retraining
            )

            forecast_df = forecast_only["forecast_table"].copy()

            #REPLACE WITH:
            for i, col in enumerate(exog_cols):
                forecast_df.insert(i + 1, col, future_exog[col].values)

            results["forecast_table"] = forecast_df
         # 🔐 Store forecast permanently for this session
            session["transformer_forecast"] = (
                results["forecast_table"].to_json(orient="split")
        )
            session["last_future_exog"] = exog_signature
        else:
            # 🔁 Load from session cache
            forecast_json = session.get("transformer_forecast")
            results["forecast_table"] = pd.read_json(
                StringIO(forecast_json), orient="split"
        )
        return redirect(url_for('show_forecast_result', model_key='transformer'))

    # ==================================================
    # OTHER MODELS (UNCHANGED)
    # ==================================================
    elif model_name == "lstm":
    
    # ✅ CHECK FOR UNIVARIATE FORECAST FIRST
        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
    
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

        # ✅ Also check for dynamic exog fields
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )
    # ==============================
    # STEP 1 — TRAIN + EVALUATE
    # ==============================
        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:
            
            # ✅ ADD THESE 3 LINES HERE
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])
            
            results = run_lstm(
            data=data,
            params=params,
            horizon=horizon,
            frequency=frequency,
            future_rain=None,
            future_mean_t=None,
            mode="train" # Normal training mode
        )

        # 🔐 STORE BASE RESULTS
            session["lstm_results"] = {
            "meta": {
                "model_key": "lstm",     # 🔑 REQUIRED
                "model_name": "LSTM",
                "frequency": results["frequency"],
                "horizon": horizon,

                "rmse_train": results["rmse_train"],
                "mae_train": results["mae_train"],
                "mape_train": results["mape_train"],

                "rmse_test": results["rmse_test"],
                "mae_test": results["mae_test"],
                "mape_test": results["mape_test"],
                    #✅ ADD THESE TWO LINES
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
            },
            "train_table": results["train_table"].to_json(orient="split"),
            "test_table": results["test_table"].to_json(orient="split"),
        }
         #ADD THIS DEBUG
            print("DEBUG: Stored in session:", session.get("lstm_results") is not None)
            print("DEBUG: Session keys:", list(session.keys()))
            session.modified = True  # ✅ Force session save
        
        #✅ Define is_univariate first
            is_univariate = (results.get("data_type") == "Univariate")
            
            return render_template(
            "results.html",
            results=results,
            show_future_inputs=True,
            is_univariate=is_univariate  # ✅ Add this line
        )

     # ==============================
    # STEP 2 — FORECAST ONLY
    # ==============================
    
    # ✅ Handle univariate forecast FIRST
        if is_univariate_forecast:
            stored = session.get("lstm_results")
            if stored is None:
                return "Session expired. Please run the model again."
    
            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }
    
        # Generate univariate forecast (no climate inputs needed)
            forecast_only = run_lstm(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )
    
            results["forecast_table"] = forecast_only["forecast_table"]
            session["lstm_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='lstm'))
        
        # STEP 2 — MULTIVARIATE FORECAST (dynamic exog columns)
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("lstm_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='lstm'))
    

    elif model_name == "gru":
    
        # ✅ CHECK FOR UNIVARIATE FORECAST FIRST
        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
    
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

    # ==============================
    # STEP 1 — TRAIN + EVALUATE
    # ==============================
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:
            
            # ✅ ADD COLUMN MAPPINGS
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])

            results = run_gru(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency
        )

        # 🔐 STORE BASE RESULTS
            session["gru_results"] = {
                "meta": {
                    "model_key": "gru",
                    "model_name": "GRU",
                    "frequency": frequency,
                    "horizon": horizon,

                    "rmse_train": results["rmse_train"],
                    "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"],

                    "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],
                    "mape_test": results["mape_test"],

                     #✅ ADD THESE TWO LINES
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
            },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
        }
            session.modified = True  # ✅ Force session save

            #✅ Define is_univariate first
            is_univariate = (results.get("data_type") == "Univariate")

            return render_template(
                "results.html",
                results=results,
               show_future_inputs=True,
               is_univariate=is_univariate  # ✅ Add this line
        )

    # ==============================
    # STEP 2 — FORECAST ONLY
    # ==============================
    
    # ✅ Handle univariate forecast FIRST
        if is_univariate_forecast:
            stored = session.get("gru_results")
            if stored is None:
                return "Session expired. Please run the model again."
    
            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }
    
        # Generate univariate forecast (no climate inputs needed)
            forecast_only = run_gru(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast",
                
            )
    
            results["forecast_table"] = forecast_only["forecast_table"]
            session["gru_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True

            return redirect(url_for('show_forecast_result', model_key='gru'))
        # STEP 2 — MULTIVARIATE FORECAST (dynamic exog columns)
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("gru_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_gru(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["gru_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='gru'))
    
    elif model_name == "cnn":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip() for col in exog_cols
        )

        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_cnn(data, params, horizon, frequency, mode="train")

            session["cnn_results"] = {
                "meta": {
                    "model_key": "cnn", "model_name": "Deep CNN",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            is_univariate = (results.get("data_type") == "Univariate")
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=is_univariate,
                                   exog_cols=session.get("exog_cols", []))

        if is_univariate_forecast:
            stored = session.get("cnn_results")
            if stored is None:
                return "Session expired. Please run the model again."
            forecast_only = run_cnn(data=None, params=params,
                                    horizon=horizon, frequency=frequency, mode="forecast")
            session["cnn_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='cnn'))

        # Multivariate forecast
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw: return f"Missing future values for '{col}'.", 400
            try: vals = list(map(float, raw.split()))
            except ValueError: return f"Invalid values for '{col}'.", 400
            if len(vals) != horizon: return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("cnn_results")
        if stored is None: return "Session expired. Please run the model again."

        forecast_only = run_cnn(data=None,
                                params={**params, "future_exog": future_exog_dict},
                                horizon=horizon, frequency=frequency, mode="forecast")
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["cnn_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='cnn'))


    elif model_name == "rnn":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

        # ✅ Check for dynamic exog fields
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # ==============================
        # STEP 1 — TRAIN + EVALUATE
        # ==============================
        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:

            # ✅ Column mappings from session
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])

            results = run_rnn(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="train"
            )

            session["rnn_results"] = {
                "meta": {
                    "model_key": "rnn",
                    "model_name": "RNN",
                    "frequency": frequency,
                    "horizon": horizon,
                    "rmse_train": results["rmse_train"],
                    "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"],
                    "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],
                    "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # ==============================
        # STEP 2 — FORECAST ONLY
        # ==============================

        # ✅ Univariate forecast
        if is_univariate_forecast:
            stored = session.get("rnn_results")
            if stored is None:
                return "Session expired. Please run the model again."

            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }

            forecast_only = run_rnn(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )

            results["forecast_table"] = forecast_only["forecast_table"]
            session["rnn_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='rnn'))

        # ✅ Multivariate forecast — dynamic exog columns
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("rnn_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_rnn(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["rnn_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='rnn'))


    elif model_name == "ann":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

        # ✅ Check for dynamic exog fields
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # ==============================
        # STEP 1 — TRAIN + EVALUATE
        # ==============================
        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:

            # ✅ Column mappings from session
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])

            results = run_ann(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="train"
            )

            session["ann_results"] = {
                "meta": {
                    "model_key": "ann",
                    "model_name": "ANN",
                    "frequency": frequency,
                    "horizon": horizon,
                    "rmse_train": results["rmse_train"],
                    "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"],
                    "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],
                    "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # ==============================
        # STEP 2 — FORECAST ONLY
        # ==============================

        # ✅ Univariate forecast
        if is_univariate_forecast:
            stored = session.get("ann_results")
            if stored is None:
                return "Session expired. Please run the model again."

            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }

            forecast_only = run_ann(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )

            results["forecast_table"] = forecast_only["forecast_table"]
            session["ann_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='ann'))

        # ✅ Multivariate forecast — dynamic exog columns
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("ann_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_ann(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["ann_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='ann'))

    
    elif model_name == "wavelet_ann":

        is_forecast = request.form.get("wavelet_ann_forecast_trigger", "").strip()
    
        # ✅ Always inject column mappings including exog
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])
        has_exog  = len(exog_cols) > 0

        # ── STEP 1: TRAIN ──────────────────────────────────────────
        if not is_forecast:
            results = run_wavelet_ann(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="train",
            )

            session["wavelet_ann_results"] = {
                "meta": {
                    "model_key"   : "wavelet_ann",
                    "model_name"  : results["model_name"],
                    "frequency"   : frequency,
                    "horizon"     : horizon,
                    "rmse_train"  : results["rmse_train"],
                    "mae_train"   : results["mae_train"],
                    "mape_train"  : results["mape_train"],
                    "rmse_test"   : results["rmse_test"],
                    "mae_test"    : results["mae_test"],
                    "mape_test"   : results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned"  : results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table" : results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            # ✅ Dynamic — not hardcoded
            is_univariate = not has_exog

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=is_univariate,
                is_wavelet_ann=True,
                exog_cols=exog_cols,
            )

        # ── STEP 2: FORECAST ───────────────────────────────────────
        stored = session.get("wavelet_ann_results")
        if stored is None:
            return "Session expired. Please run the model again."

        # ── Univariate forecast (no exog inputs needed) ─────────────
        if not has_exog:
            forecast_only = run_wavelet_ann(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast",
            )
            session["wavelet_ann_forecast"] = \
                forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='wavelet_ann'))

        # ── Multivariate forecast — collect future exog from form ───
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        forecast_only = run_wavelet_ann(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast",
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        session["wavelet_ann_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='wavelet_ann'))

    elif model_name == "wavelet_lstm":

        is_forecast = (request.form.get("wavelet_lstm_forecast_trigger", "").strip() or
                       request.form.get("wavelet_ann_forecast_trigger", "").strip() or
                       request.form.get("univariate_forecast_trigger", "").strip())

        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])
        has_exog  = len(exog_cols) > 0

        # ── STEP 1: TRAIN ──────────────────────────────────────
        if not is_forecast:
            results = run_wavelet_lstm(
                data=data, params=params,
                horizon=horizon, frequency=frequency, mode="train",
            )
            session["wavelet_lstm_results"] = {
                "meta": {
                    "model_key"   : "wavelet_lstm",
                    "model_name"  : results["model_name"],
                    "frequency"   : frequency,
                    "horizon"     : horizon,
                    "rmse_train"  : results["rmse_train"],
                    "mae_train"   : results["mae_train"],
                    "mape_train"  : results["mape_train"],
                    "rmse_test"   : results["rmse_test"],
                    "mae_test"    : results["mae_test"],
                    "mape_test"   : results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned"  : results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table" : results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=not has_exog,
                is_wavelet_ann=(model_name == "wavelet_lstm"),
                exog_cols=exog_cols,
            )

        # ── STEP 2: FORECAST ───────────────────────────────────
        stored = session.get("wavelet_lstm_results")
        if stored is None:
            return "Session expired. Please run the model again."

        if not has_exog:
            forecast_only = run_wavelet_lstm(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast",
            )
            session["wavelet_lstm_forecast"] = \
                forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='wavelet_lstm'))

        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw: return f"Missing future values for '{col}'.", 400
            try: vals = list(map(float, raw.split()))
            except ValueError: return f"Invalid values for '{col}'.", 400
            if len(vals) != horizon: return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        forecast_only = run_wavelet_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast",
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["wavelet_lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='wavelet_lstm'))
    
    elif model_name == "wavelet_transformer":

        is_forecast = (request.form.get("wavelet_transformer_forecast_trigger", "").strip() or
                       request.form.get("wavelet_ann_forecast_trigger", "").strip() or
                       request.form.get("univariate_forecast_trigger", "").strip())

        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])
        has_exog  = len(exog_cols) > 0

        # ── STEP 1: TRAIN ──────────────────────────────────────
        if not is_forecast:
            results = run_wavelet_transformer(
                data=data, params=params,
                horizon=horizon, frequency=frequency, mode="train",
            )
            session["wavelet_transformer_results"] = {
                "meta": {
                    "model_key"   : "wavelet_transformer",
                    "model_name"  : results["model_name"],
                    "frequency"   : frequency,
                    "horizon"     : horizon,
                    "rmse_train"  : results["rmse_train"],
                    "mae_train"   : results["mae_train"],
                    "mape_train"  : results["mape_train"],
                    "rmse_test"   : results["rmse_test"],
                    "mae_test"    : results["mae_test"],
                    "mape_test"   : results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned"  : results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table" : results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=not has_exog,
                is_wavelet_ann=True,   # reuse same forecast button style
                exog_cols=exog_cols,
            )

        # ── STEP 2: FORECAST ───────────────────────────────────
        stored = session.get("wavelet_transformer_results")
        if stored is None:
            return "Session expired. Please run the model again."

        if not has_exog:
            forecast_only = run_wavelet_transformer(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast",
            )
            session["wavelet_transformer_forecast"] = \
                forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='wavelet_transformer'))

        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw: return f"Missing future values for '{col}'.", 400
            try: vals = list(map(float, raw.split()))
            except ValueError: return f"Invalid values for '{col}'.", 400
            if len(vals) != horizon: return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        forecast_only = run_wavelet_transformer(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast",
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["wavelet_transformer_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='wavelet_transformer'))

    elif model_name == "svr":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

        # ✅ Check for dynamic exog fields
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # ==============================
        # STEP 1 — TRAIN + EVALUATE
        # ==============================
        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:

            # ✅ Column mappings from session
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])

            results = run_svr(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="train"
            )

            session["svr_results"] = {
                "meta": {
                    "model_key": "svr",
                    "model_name": "SVR",
                    "frequency": frequency,
                    "horizon": horizon,
                    "rmse_train": results["rmse_train"],
                    "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"],
                    "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],
                    "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

    # ==============================
    # STEP 2 — FORECAST ONLY
    # ==============================

    # ✅ Univariate forecast
        if is_univariate_forecast:
            stored = session.get("svr_results")
            if stored is None:
                return "Session expired. Please run the model again."

            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }

            forecast_only = run_svr(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )

            results["forecast_table"] = forecast_only["forecast_table"]
            session["svr_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='svr'))

        # ✅ Multivariate forecast — dynamic exog columns
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("svr_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_svr(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["svr_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='svr'))

    elif model_name == "rf":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        future_mean_t = request.form.get("future_mean_t", "").strip()
        future_rain = request.form.get("future_rain", "").strip()

        # ✅ Check for dynamic exog fields
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # ==============================
        # STEP 1 — TRAIN + EVALUATE
        # ==============================
        if not is_univariate_forecast and not future_mean_t and not future_rain and not has_future_exog:

            # ✅ Column mappings from session
            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols", [])

            results = run_rf(
                data=data,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="train"
            )

            session["rf_results"] = {
                "meta": {
                    "model_key": "rf",
                    "model_name": "Random Forest",
                    "frequency": frequency,
                    "horizon": horizon,
                    "rmse_train": results["rmse_train"],
                    "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"],
                    "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],
                    "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned": results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table": results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # ==============================
        #    STEP 2 — FORECAST ONLY
        # ==============================

        # ✅ Univariate forecast
        if is_univariate_forecast:
            stored = session.get("rf_results")
            if stored is None:
                return "Session expired. Please run the model again."

            results = {
                **stored["meta"],
                "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
                "forecast_table": None,
            }

            forecast_only = run_rf(
                data=None,
                params=params,
                horizon=horizon,
                frequency=frequency,
                mode="forecast"
            )

            results["forecast_table"] = forecast_only["forecast_table"]
            session["rf_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='rf'))

        # ✅ Multivariate forecast — dynamic exog columns
        exog_cols = session.get("exog_cols", [])
        future_exog_dict = {}

        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("rf_results")
        if stored is None:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
            "forecast_table": None,
        }

        forecast_only = run_rf(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon,
            frequency=frequency,
            mode="forecast"
        )

        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["rf_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='rf'))
    
    elif model_name == "xgb":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # STEP 1 — TRAIN
        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_xgb(data, params, horizon, frequency, mode="train")

            session["xgb_results"] = {
                "meta": {
                    "model_key": "xgb", "model_name": "XGBoost",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")
            return render_template(
                "results.html", results=results,
                show_future_inputs=True, is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # STEP 2a — UNIVARIATE FORECAST
        if is_univariate_forecast:
            stored = session.get("xgb_results")
            if stored is None:
                return "Session expired. Please run the model again."

            forecast_only = run_xgb(data=None, params=params, horizon=horizon,
                                    frequency=frequency, mode="forecast")
            session["xgb_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='xgb'))

        # STEP 2b — MULTIVARIATE FORECAST
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("xgb_results")
        if stored is None:
            return "Session expired. Please run the model again."

        forecast_only = run_xgb(
            data=None, params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        session["xgb_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='xgb'))
    
    elif model_name == "gbm":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # STEP 1 — TRAIN
        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_gbm(data, params, horizon, frequency, mode="train")

            session["gbm_results"] = {
                "meta": {
                    "model_key": "gbm", "model_name": "GBM",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")
            return render_template(
                "results.html", results=results,
                show_future_inputs=True, is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # STEP 2a — UNIVARIATE FORECAST
        if is_univariate_forecast:
            stored = session.get("gbm_results")
            if stored is None:
                return "Session expired. Please run the model again."

            forecast_only = run_gbm(data=None, params=params, horizon=horizon,
                                    frequency=frequency, mode="forecast")
            session["gbm_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='gbm'))

        # STEP 2b — MULTIVARIATE FORECAST
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("gbm_results")
        if stored is None:
            return "Session expired. Please run the model again."

        forecast_only = run_gbm(
            data=None, params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        session["gbm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='gbm'))
    
    elif model_name == "knn":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip()
            for col in exog_cols
        )

        # STEP 1 — TRAIN
        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_knn(data, params, horizon, frequency, mode="train")

            session["knn_results"] = {
                "meta": {
                    "model_key": "knn", "model_name": "KNN",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True

            is_univariate = (results.get("data_type") == "Univariate")
            return render_template(
                "results.html", results=results,
                show_future_inputs=True, is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # STEP 2a — UNIVARIATE FORECAST
        if is_univariate_forecast:
            stored = session.get("knn_results")
            if stored is None:
                return "Session expired. Please run the model again."

            forecast_only = run_knn(data=None, params=params, horizon=horizon,
                                    frequency=frequency, mode="forecast")
            session["knn_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='knn'))

        # STEP 2b — MULTIVARIATE FORECAST
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("knn_results")
        if stored is None:
            return "Session expired. Please run the model again."

        forecast_only = run_knn(
            data=None, params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        session["knn_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='knn'))
    
    elif model_name == "stacked_lstm":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip() for col in exog_cols
        )

        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_stacked_lstm(data, params, horizon, frequency, mode="train")

            session["stacked_lstm_results"] = {
                "meta": {
                    "model_key": "stacked_lstm", "model_name": "Stacked LSTM",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            is_univariate = (results.get("data_type") == "Univariate")
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=is_univariate,
                                   exog_cols=session.get("exog_cols", []))

        if is_univariate_forecast:
            stored = session.get("stacked_lstm_results")
            if stored is None:
                return "Session expired. Please run the model again."
            forecast_only = run_stacked_lstm(data=None, params=params,
                                             horizon=horizon, frequency=frequency,
                                             mode="forecast")
            session["stacked_lstm_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='stacked_lstm'))

        # Multivariate forecast
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw: return f"Missing future values for '{col}'.", 400
            try: vals = list(map(float, raw.split()))
            except ValueError: return f"Invalid values for '{col}'.", 400
            if len(vals) != horizon: return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("stacked_lstm_results")
        if stored is None: return "Session expired. Please run the model again."

        forecast_only = run_stacked_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["stacked_lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='stacked_lstm'))
    
    elif model_name == "bd_lstm":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip() for col in exog_cols
        )

        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_bd_lstm(data, params, horizon, frequency, mode="train")

            session["bd_lstm_results"] = {
                "meta": {
                    "model_key": "bd_lstm", "model_name": "Bidirectional LSTM",
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"],     "mape_test": results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned":   results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            is_univariate = (results.get("data_type") == "Univariate")
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=is_univariate,
                                   exog_cols=session.get("exog_cols", []))

        if is_univariate_forecast:
            stored = session.get("bd_lstm_results")
            if stored is None:
                return "Session expired. Please run the model again."
            forecast_only = run_bd_lstm(data=None, params=params,
                                        horizon=horizon, frequency=frequency,
                                        mode="forecast")
            session["bd_lstm_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='bd_lstm'))

        # Multivariate forecast
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw: return f"Missing future values for '{col}'.", 400
            try: vals = list(map(float, raw.split()))
            except ValueError: return f"Invalid values for '{col}'.", 400
            if len(vals) != horizon: return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("bd_lstm_results")
        if stored is None: return "Session expired. Please run the model again."

        forecast_only = run_bd_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["bd_lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='bd_lstm'))
    
    elif model_name == "conv_lstm":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip() for col in exog_cols
        )

        # ── STEP 1: TRAIN ──────────────────────────────────────────────
        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_conv_lstm(data, params, horizon, frequency, mode="train")

            session["conv_lstm_results"] = {
                "meta": {
                    "model_key"   : "conv_lstm",
                    "model_name"  : "Conv-LSTM",
                    "frequency"   : frequency,
                    "horizon"     : horizon,
                    "rmse_train"  : results["rmse_train"],
                    "mae_train"   : results["mae_train"],
                    "mape_train"  : results["mape_train"],
                    "rmse_test"   : results["rmse_test"],
                    "mae_test"    : results["mae_test"],
                    "mape_test"   : results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned"  : results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table" : results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            is_univariate = (results.get("data_type") == "Univariate")
            return render_template(
                "results.html", results=results,
                show_future_inputs=True, is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # ── STEP 2a: UNIVARIATE FORECAST ───────────────────────────────
        if is_univariate_forecast:
            stored = session.get("conv_lstm_results")
            if stored is None:
                return "Session expired. Please run the model again."
            forecast_only = run_conv_lstm(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast"
            )
            session["conv_lstm_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='conv_lstm'))

        # ── STEP 2b: MULTIVARIATE FORECAST ─────────────────────────────
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("conv_lstm_results")
        if stored is None:
            return "Session expired. Please run the model again."

        forecast_only = run_conv_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["conv_lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='conv_lstm'))
    
    elif model_name == "deep_lstm":

        is_univariate_forecast = request.form.get("univariate_forecast_trigger", "").strip()
        exog_cols = session.get("exog_cols", [])
        has_future_exog = any(
            request.form.get(f"future_exog_{col}", "").strip() for col in exog_cols
        )

        # ── STEP 1: TRAIN ──────────────────────────────────────────────
        if not is_univariate_forecast and not has_future_exog:
            params["time_col"]   = session.get("time_col",   "Year")
            params["target_col"] = session.get("target_col", "Yield")
            params["exog_cols"]  = session.get("exog_cols",  [])

            results = run_deep_lstm(data, params, horizon, frequency, mode="train")

            session["deep_lstm_results"] = {
                "meta": {
                    "model_key"   : "deep_lstm",
                    "model_name"  : "Deep LSTM",
                    "frequency"   : frequency,
                    "horizon"     : horizon,
                    "rmse_train"  : results["rmse_train"],
                    "mae_train"   : results["mae_train"],
                    "mape_train"  : results["mape_train"],
                    "rmse_test"   : results["rmse_test"],
                    "mae_test"    : results["mae_test"],
                    "mape_test"   : results["mape_test"],
                    "model_config": results["model_config"],
                    "auto_tuned"  : results["auto_tuned"],
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table" : results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            is_univariate = (results.get("data_type") == "Univariate")
            return render_template(
                "results.html", results=results,
                show_future_inputs=True, is_univariate=is_univariate,
                exog_cols=session.get("exog_cols", [])
            )

        # ── STEP 2a: UNIVARIATE FORECAST ───────────────────────────────
        if is_univariate_forecast:
            stored = session.get("deep_lstm_results")
            if stored is None:
                return "Session expired. Please run the model again."
            forecast_only = run_deep_lstm(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast"
            )
            session["deep_lstm_forecast"] = \
                forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='deep_lstm'))

        # ── STEP 2b: MULTIVARIATE FORECAST ─────────────────────────────
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        stored = session.get("deep_lstm_results")
        if stored is None:
            return "Session expired. Please run the model again."

        forecast_only = run_deep_lstm(
            data=None,
            params={**params, "future_exog": future_exog_dict},
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])
        session["deep_lstm_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='deep_lstm'))
    
    elif model_name == "sarima":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        is_univariate_forecast = request.form.get("sarima_forecast_trigger", "").strip()

        stored = session.get("sarima_results")

        # ── STEP 1: TRAIN ──────────────────────────────────────────────
        if not stored and not is_univariate_forecast:
            results = run_sarima(data, params, horizon, frequency, mode="train")
            session["sarima_results"] = {
                "meta": {
                    "model_key":    "sarima",
                    "model_name":   results["model_name"],
                    "frequency":    frequency,
                    "horizon":      horizon,
                    "rmse_train":   results["rmse_train"],
                    "mae_train":    results["mae_train"],
                    "mape_train":   results["mape_train"],
                    "rmse_test":    results["rmse_test"],
                    "mae_test":     results["mae_test"],
                    "mape_test":    results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=True,
                is_univariate=not exog_cols,
                exog_cols=exog_cols,
            )

        # ── STEP 2: FORECAST ───────────────────────────────────────────
        if not stored:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
            "forecast_table": None,
            "param_estimates_table": pd.read_json(
                StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                if stored["meta"].get("param_estimates_table") else None,
            "info_criteria": stored["meta"].get("info_criteria"),
        }

        # ── Univariate: no exog inputs needed ─────────────────────────
        if not exog_cols:
            forecast_only = run_sarima(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast"
            )
            results["forecast_table"] = forecast_only["forecast_table"]
            session["sarima_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='sarima'))

        # ── Multivariate: collect future exog values ───────────────────
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        params["future_exog"] = future_exog_dict
        forecast_only = run_sarima(
            data=None, params=params,
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["sarima_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='sarima'))
    
    elif model_name == "tbats":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        stored = session.get("tbats_results")
        is_forecast = request.form.get("tbats_forecast_trigger", "").strip()

        if not stored and not is_forecast:
            results = run_tbats(data, params, horizon, frequency, mode="train")
            session["tbats_results"] = {
                "meta": {
                    "model_key": "tbats", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"], "mape_test": results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned": results.get("auto_tuned", False),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=True, exog_cols=[])

        if not stored:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
            "forecast_table": None,
            "param_estimates_table": pd.read_json(
                StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                if stored["meta"].get("param_estimates_table") else None,
            "info_criteria": stored["meta"].get("info_criteria"),
        }
        forecast_only = run_tbats(data=None, params=params, horizon=horizon,
                                  frequency=frequency, mode="forecast")
        results["forecast_table"] = forecast_only["forecast_table"]
        session["tbats_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='tbats'))

    elif model_name == "arima":
        
    # Check if this is a forecast request (user clicked forecast button)
        is_forecast_request = request.form.get("arima_forecast_trigger", "").strip()

    # ==============================
    # STEP 1 — TRAIN + EVALUATE
    # ==============================
        if not is_forecast_request:

            params["time_col"]   = session.get("time_col", "Year")
            params["target_col"] = session.get("target_col", "Yield")
            
            results = run_arima(
            data=data,
            params=params,
            horizon=horizon,
            frequency=frequency,
            mode="train"  # Normal training mode
        )

        # 🔐 STORE BASE RESULTS
            session["arima_results"] = {
            "meta": {
                "model_key": "arima",
                "model_name": results["model_name"],  # Will be "ARIMA(p,d,q)"
                "frequency": frequency,
                "horizon": horizon,

                "rmse_train": results["rmse_train"],
                "mae_train": results["mae_train"],
                "mape_train": results["mape_train"],

                "rmse_test": results["rmse_test"],
                "mae_test": results["mae_test"],
                "mape_test": results["mape_test"],

                "order": results["order"],
                "auto_arima": results["auto_arima"],

                # ✅ ADD THESE TWO LINES
                "model_config": results.get("model_config", {}),
                "auto_tuned": results.get("auto_tuned", False),
            },
            "train_table": results["train_table"].to_json(orient="split"),
            "test_table": results["test_table"].to_json(orient="split"),

            # ✅ CRITICAL FIX: Store the trained ARIMA order for forecasting
              # e.g., "ARIMA(2,1,2)"
            "trained_params": params,  # Store original training params
        }
            session.modified = True  # ✅ Force session save

            return render_template(
            "results.html",
            results=results,
            show_future_inputs=True,
            is_arima=True , # ✅ Flag for template to hide input fields
            is_univariate=True  # ✅ ADD THIS LINE
        )

    # ==============================
    # STEP 2 — FORECAST ONLY
    # ==============================
        stored = session.get("arima_results")
        if stored is None:
            return "Session expired. Please run the model again."
        

        results = {
        **stored["meta"],
        "train_table": pd.read_json(StringIO(stored["train_table"]), orient="split"),
        "test_table": pd.read_json(StringIO(stored["test_table"]), orient="split"),
        "forecast_table": None,
    }
    # ✅ CRITICAL FIX: Use stored training params, not current form
        forecast_params = stored.get("trained_params")

        if not forecast_params:
            return "Training parameters not found. Please train the model again."
    # 🔮 Forecast using saved model (no future inputs needed)
        forecast_only = run_arima(
        data=None,  # No data needed
        params=forecast_params,
        horizon=horizon,
        frequency=frequency,
        mode="forecast"  # Special mode to skip retraining
    )

        results["forecast_table"] = forecast_only["forecast_table"]
        session["arima_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='arima'))
    
    elif model_name == "aregarch":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        stored = session.get("aregarch_results")

        if stored:
            # ── Collect future exog from form if multivariate ─────
            if exog_cols:
                future_exog_dict = {}
                for col in exog_cols:
                    raw = request.form.get(f"future_exog_{col}", "").strip()
                    if not raw:
                        return f"Missing future values for '{col}'.", 400
                    try:
                        vals = list(map(float, raw.split()))
                    except ValueError:
                        return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                    if len(vals) != horizon:
                        return f"Enter exactly {horizon} values for '{col}'.", 400
                    future_exog_dict[col] = vals
                params["future_exog"] = future_exog_dict

            results = {
                **stored["meta"],
                "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
                "forecast_table": None,
                "param_estimates_table": pd.read_json(
                    StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                    if stored["meta"].get("param_estimates_table") else None,
                "info_criteria": stored["meta"].get("info_criteria"),
            }
            forecast_only = run_aregarch(data=None, params=params, horizon=horizon,
                                     frequency=frequency, mode="forecast")
            results["forecast_table"] = forecast_only["forecast_table"]
            session["aregarch_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='aregarch'))
        else:
            results = run_aregarch(data, params, horizon, frequency, mode="train")
            session["aregarch_results"] = {
                "meta": {
                    "model_key": "aregarch", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"], "mape_train": results["mape_train"],
                    "rmse_test":  results["rmse_test"],  "mae_test":  results["mae_test"],  "mape_test":  results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                               show_future_inputs=True, is_univariate=not exog_cols,
                               exog_cols=exog_cols)

    elif model_name == "argarch":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        stored = session.get("argarch_results")

        if stored:
            if exog_cols:
                future_exog_dict = {}
                for col in exog_cols:
                    raw = request.form.get(f"future_exog_{col}", "").strip()
                    if not raw:
                        return f"Missing future values for '{col}'.", 400
                    try:
                        vals = list(map(float, raw.split()))
                    except ValueError:
                        return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                    if len(vals) != horizon:
                        return f"Enter exactly {horizon} values for '{col}'.", 400
                    future_exog_dict[col] = vals
                params["future_exog"] = future_exog_dict

            results = {
                **stored["meta"],
                "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
                "forecast_table": None,
                "param_estimates_table": pd.read_json(
                    StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                    if stored["meta"].get("param_estimates_table") else None,
                "info_criteria": stored["meta"].get("info_criteria"),
            }
            forecast_only = run_argarch(data=None, params=params, horizon=horizon,
                                        frequency=frequency, mode="forecast")
            results["forecast_table"] = forecast_only["forecast_table"]
            session["argarch_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='argarch'))

        else:
            results = run_argarch(data, params, horizon, frequency, mode="train")

            if results.get("arch_significant") is False:
                return render_template("results.html", results=results,
                                       show_future_inputs=False, exog_cols=exog_cols)

            session["argarch_results"] = {
                "meta": {
                    "model_key": "argarch", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"], "mape_train": results["mape_train"],
                    "rmse_test":  results["rmse_test"],  "mae_test":  results["mae_test"],  "mape_test":  results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "arch_lm_stat":     results.get("arch_lm_stat"),
                    "arch_lm_pvalue":   results.get("arch_lm_pvalue"),
                    "arch_significant": results.get("arch_significant"),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=not exog_cols,
                                   exog_cols=exog_cols)   
    
    elif model_name == "armagarch":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        stored = session.get("armagarch_results")

        if stored:
            if exog_cols:
                future_exog_dict = {}
                for col in exog_cols:
                    raw = request.form.get(f"future_exog_{col}", "").strip()
                    if not raw:
                        return f"Missing future values for '{col}'.", 400
                    try:
                        vals = list(map(float, raw.split()))
                    except ValueError:
                        return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                    if len(vals) != horizon:
                        return f"Enter exactly {horizon} values for '{col}'.", 400
                    future_exog_dict[col] = vals
                params["future_exog"] = future_exog_dict

            results = {
                **stored["meta"],
                "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
                "forecast_table": None,
                "param_estimates_table": pd.read_json(
                    StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                    if stored["meta"].get("param_estimates_table") else None,
                "info_criteria": stored["meta"].get("info_criteria"),
            }
            forecast_only = run_armagarch(data=None, params=params, horizon=horizon,
                                          frequency=frequency, mode="forecast")
            results["forecast_table"] = forecast_only["forecast_table"]
            session["armagarch_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='armagarch'))

        else:
            results = run_armagarch(data, params, horizon, frequency, mode="train")

            if results.get("arch_significant") is False:
                return render_template("results.html", results=results,
                                       show_future_inputs=False, exog_cols=exog_cols)

            session["armagarch_results"] = {
                "meta": {
                    "model_key": "armagarch", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"], "mape_train": results["mape_train"],
                    "rmse_test":  results["rmse_test"],  "mae_test":  results["mae_test"],  "mape_test":  results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "arch_lm_stat":     results.get("arch_lm_stat"),
                    "arch_lm_pvalue":   results.get("arch_lm_pvalue"),
                    "arch_significant": results.get("arch_significant"),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=not exog_cols,
                                   exog_cols=exog_cols)
    
    elif model_name == "artgarch":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        stored = session.get("artgarch_results")

        if stored:
            if exog_cols:
                future_exog_dict = {}
                for col in exog_cols:
                    raw = request.form.get(f"future_exog_{col}", "").strip()
                    if not raw:
                        return f"Missing future values for '{col}'.", 400
                    try:
                        vals = list(map(float, raw.split()))
                    except ValueError:
                        return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                    if len(vals) != horizon:
                        return f"Enter exactly {horizon} values for '{col}'.", 400
                    future_exog_dict[col] = vals
                params["future_exog"] = future_exog_dict

            results = {
                **stored["meta"],
                "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
                "forecast_table": None,
                "param_estimates_table": pd.read_json(
                    StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                    if stored["meta"].get("param_estimates_table") else None,
                "info_criteria": stored["meta"].get("info_criteria"),
            }
            forecast_only = run_artgarch(data=None, params=params, horizon=horizon,
                                       frequency=frequency, mode="forecast")
            results["forecast_table"] = forecast_only["forecast_table"]
            session["artgarch_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='artgarch'))

        else:
            results = run_artgarch(data, params, horizon, frequency, mode="train")

            if results.get("arch_significant") is False:
                return render_template("results.html", results=results,
                                       show_future_inputs=False, exog_cols=exog_cols)

            session["artgarch_results"] = {
                "meta": {
                    "model_key": "artgarch", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"], "mape_train": results["mape_train"],
                    "rmse_test":  results["rmse_test"],  "mae_test":  results["mae_test"],  "mape_test":  results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "arch_lm_stat":     results.get("arch_lm_stat"),
                    "arch_lm_pvalue":   results.get("arch_lm_pvalue"),
                    "arch_significant": results.get("arch_significant"),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=not exog_cols,
                                   exog_cols=exog_cols)
    
    elif model_name == "argjrgarch":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])
        exog_cols = session.get("exog_cols", [])

        stored = session.get("argjrgarch_results")

        if stored:
            if exog_cols:
                future_exog_dict = {}
                for col in exog_cols:
                    raw = request.form.get(f"future_exog_{col}", "").strip()
                    if not raw:
                        return f"Missing future values for '{col}'.", 400
                    try:
                        vals = list(map(float, raw.split()))
                    except ValueError:
                        return f"Invalid values for '{col}'. Use space-separated numbers.", 400
                    if len(vals) != horizon:
                        return f"Enter exactly {horizon} values for '{col}'.", 400
                    future_exog_dict[col] = vals
                params["future_exog"] = future_exog_dict

            results = {
                **stored["meta"],
                "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
                "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
                "forecast_table": None,
                "param_estimates_table": pd.read_json(
                    StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                    if stored["meta"].get("param_estimates_table") else None,
                "info_criteria": stored["meta"].get("info_criteria"),
            }
            forecast_only = run_argjrgarch(data=None, params=params, horizon=horizon,
                                         frequency=frequency, mode="forecast")
            results["forecast_table"] = forecast_only["forecast_table"]
            session["argjrgarch_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='argjrgarch'))

        else:
            results = run_argjrgarch(data, params, horizon, frequency, mode="train")

            if results.get("arch_significant") is False:
                return render_template("results.html", results=results,
                                       show_future_inputs=False, exog_cols=exog_cols)

            session["argjrgarch_results"] = {
                "meta": {
                    "model_key": "argjrgarch", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"], "mape_train": results["mape_train"],
                    "rmse_test":  results["rmse_test"],  "mae_test":  results["mae_test"],  "mape_test":  results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned":   results.get("auto_tuned", False),
                    "arch_lm_stat":     results.get("arch_lm_stat"),
                    "arch_lm_pvalue":   results.get("arch_lm_pvalue"),
                    "arch_significant": results.get("arch_significant"),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                                   show_future_inputs=True, is_univariate=not exog_cols,
                                   exog_cols=exog_cols)
    
    elif model_name == "rw":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        stored = session.get("rw_results")
        is_forecast = request.form.get("rw_forecast_trigger", "").strip()

        if not stored and not is_forecast:
            results = run_random_walk(data, params, horizon, frequency, mode="train")
            session["rw_results"] = {
                "meta": {
                    "model_key": "rw", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"], "mape_test": results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned": results.get("auto_tuned", False),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template("results.html", results=results,
                               show_future_inputs=True, is_univariate=True, exog_cols=[])

        if not stored:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
            "forecast_table": None,
            "param_estimates_table": pd.read_json(
                StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                if stored["meta"].get("param_estimates_table") else None,
            "info_criteria": stored["meta"].get("info_criteria"),
        }
        forecast_only = run_random_walk(data=None, params=params, horizon=horizon,
                                        frequency=frequency, mode="forecast")
        results["forecast_table"] = forecast_only["forecast_table"]
        session["rw_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='rw'))
    
    elif model_name == "ets":
        params["time_col"]   = session.get("time_col",   "Year")
        params["target_col"] = session.get("target_col", "Yield")
        params["exog_cols"]  = session.get("exog_cols",  [])   # ← was missing
        exog_cols = session.get("exog_cols", [])
        stored      = session.get("ets_results")
        is_forecast = request.form.get("ets_forecast_trigger", "").strip()

        if not stored and not is_forecast:
            results = run_ets(data, params, horizon, frequency, mode="train")
            session["ets_results"] = {
                "meta": {
                    "model_key": "ets", "model_name": results["model_name"],
                    "frequency": frequency, "horizon": horizon,
                    "rmse_train": results["rmse_train"], "mae_train": results["mae_train"],
                    "mape_train": results["mape_train"], "rmse_test": results["rmse_test"],
                    "mae_test": results["mae_test"], "mape_test": results["mape_test"],
                    "model_config": results.get("model_config", {}),
                    "auto_tuned": results.get("auto_tuned", False),
                    "param_estimates_table": results.get("param_estimates_table").to_json(orient="split")
                        if results.get("param_estimates_table") is not None else None,
                    "info_criteria": results.get("info_criteria"),
                },
                "train_table": results["train_table"].to_json(orient="split"),
                "test_table":  results["test_table"].to_json(orient="split"),
            }
            session.modified = True
            return render_template(
                "results.html", results=results,
                show_future_inputs=True,
                is_univariate=not exog_cols,   # ← was hardcoded True
                exog_cols=exog_cols,           # ← was hardcoded []
            )

        if not stored:
            return "Session expired. Please run the model again."

        results = {
            **stored["meta"],
            "train_table":    pd.read_json(StringIO(stored["train_table"]), orient="split"),
            "test_table":     pd.read_json(StringIO(stored["test_table"]),  orient="split"),
            "forecast_table": None,
            "param_estimates_table": pd.read_json(
                StringIO(stored["meta"]["param_estimates_table"]), orient="split")
                if stored["meta"].get("param_estimates_table") else None,
            "info_criteria": stored["meta"].get("info_criteria"),
        }

        # ── Univariate: no exog inputs needed ────────────────────────────
        if not exog_cols:
            forecast_only = run_ets(
                data=None, params=params,
                horizon=horizon, frequency=frequency, mode="forecast"
            )
            results["forecast_table"] = forecast_only["forecast_table"]
            session["ets_forecast"] = forecast_only["forecast_table"].to_json(orient="split")
            session.modified = True
            return redirect(url_for('show_forecast_result', model_key='ets'))

        # ── Multivariate: collect future exog values ──────────────────────
        future_exog_dict = {}
        for col in exog_cols:
            raw = request.form.get(f"future_exog_{col}", "").strip()
            if not raw:
                return f"Missing future values for '{col}'.", 400
            try:
                vals = list(map(float, raw.split()))
            except ValueError:
                return f"Invalid values for '{col}'. Use space-separated numbers.", 400
            if len(vals) != horizon:
                return f"Enter exactly {horizon} values for '{col}'.", 400
            future_exog_dict[col] = vals

        params["future_exog"] = future_exog_dict
        forecast_only = run_ets(
            data=None, params=params,
            horizon=horizon, frequency=frequency, mode="forecast"
        )
        forecast_df = forecast_only["forecast_table"].copy()
        for i, col in enumerate(exog_cols):
            forecast_df.insert(i + 1, col, future_exog_dict[col])

        results["forecast_table"] = forecast_df
        session["ets_forecast"] = forecast_df.to_json(orient="split")
        session.modified = True
        return redirect(url_for('show_forecast_result', model_key='ets'))
        
    else:
        return "Invalid model selected."

    # ✅ ADD THIS CHECK BEFORE RENDERING
    if results is None:
        return "Model execution failed. Please check your parameters and try again.", 500

    return render_template("results.html", results=results)

@app.route("/autofill_exog", methods=["POST"])
def autofill_exog():
    """
    Auto-fill future exogenous inputs using ARIMA forecast on historical data.
    """
    from statsmodels.tsa.arima.model import ARIMA as ARIMAModel

    data_json = session.get("data")
    if not data_json:
        # Try to recover from training_progress
        session_id = session.get("single_training_id") or session.get("comparison_id")
        progress_data = get_progress(session_id)
        if session_id and progress_data:
            data_json = progress_data.get("data_json")
    
    if not data_json:
        return {"error": "No data found in session. Please re-upload your dataset."}, 400

    data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    exog_cols = session.get("exog_cols", [])
    horizon   = session.get("horizon", 1)

    if not exog_cols or horizon == 1:
        session_id = session.get("single_training_id") or session.get("comparison_id")
        if session_id:
            progress_data = get_progress(session_id)
            if progress_data:
                if not exog_cols:
                    exog_cols = progress_data.get("exog_cols", [])
                if horizon == 1:
                    horizon = progress_data.get("horizon", 1)
    if not exog_cols:
        return {"error": "No exogenous columns found."}, 400

    forecasts = {}
    for col in exog_cols:
        try:
            series = pd.to_numeric(data[col], errors="coerce").dropna()
            model  = ARIMAModel(series, order=(1, 1, 0), trend="n").fit()
            vals   = model.forecast(steps=horizon).tolist()
            forecasts[col] = [round(v, 3) for v in vals]
        except Exception:
            # Fallback: repeat the column mean
            mean_val = float(pd.to_numeric(data[col], errors="coerce").mean())
            forecasts[col] = [round(mean_val, 3)] * horizon

    return {"status": "ok", "forecasts": forecasts, "horizon": horizon}

# =====================================================
# ARCH TEST API ROUTES (for model_selection.html panel)
# =====================================================

@app.route("/api/arch_acf", methods=["POST"])
def arch_acf():
    """
    Fits Auto-ARIMA on the target series alone (identical pipeline to
    _arch_lm_test in ar_garch_model.py), then computes ACF of residuals²
    up to max_lag. No time column or OLS step involved.
    """
    import pmdarima as pm
    from statsmodels.tsa.stattools import acf as sm_acf

    data_json = session.get("data")
    if not data_json:
        return {"error": "No data found. Please upload your dataset first."}, 400

    body       = request.get_json(force=True)
    target_col = body.get("target_col", "").strip()
    max_lag    = int(body.get("max_lag", 12))

    if not target_col:
        return {"error": "target_col is required."}, 400

    try:
        df = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}, 500

    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}, 400

    # ── Target series only — exactly as in _arch_lm_test ─────────────────
    y = pd.Series(df[target_col].values.astype(float)).reset_index(drop=True)
    n = len(y)

    if n < max_lag + 5:
        return {"error": f"Not enough data points ({n}) for max_lag={max_lag}."}, 400

    # ── Auto-ARIMA → residuals (same settings as ar_garch_model.py) ───────
    try:
        arima_fit = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            information_criterion="aic",
            test="kpss",
            max_p=5, max_q=5, max_d=2,
            start_p=0, start_q=0,
            max_order=5,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order   = arima_fit.order
        p, d, q = order

        # pmdarima.resid() = equivalent of R's residuals(auto.arima(y))
        # = Kalman filter innovation residuals — correct for ARCH test
        resid = pd.Series(arima_fit.resid()).iloc[1:]  # Drop first residual (Kalman init artifact)
        resid = resid.reset_index(drop=True)
    except Exception as e:
        return {"error": f"Auto-ARIMA failed: {str(e)}"}, 500
    

    # ── ACF of residuals² ─────────────────────────────────────────────────
    resid_sq = resid ** 2
    bound    = 1.96 / (n ** 0.5)
    acf_vals = sm_acf(resid_sq, nlags=max_lag, fft=True)

    rows = []
    for lag in range(1, max_lag + 1):
        v   = float(acf_vals[lag])
        sig = abs(v) > bound
        rows.append({
            "lag"           : lag,
            "acf_value"     : round(v, 4),
            "significant"   : sig,
            "interpretation": (
                f"Lag {lag}: |ACF|={abs(v):.3f} > {bound:.3f} → "
                "squared residuals autocorrelated → ARCH effect likely"
                if sig else
                f"Lag {lag}: |ACF|={abs(v):.3f} ≤ {bound:.3f} → not significant"
            ),
        })

    # Build residual table for display in frontend
    resid_list = [
        {
            "index"    : int(i),
            "residual" : round(float(resid.iloc[i]), 4),
            "resid_sq" : round(float(resid_sq.iloc[i]), 4),
        }
        for i in range(len(resid))
    ]

    # Extract MA(1) coefficient — must be done BEFORE the return dict
    # Extract MA(1) coefficient from statsmodels state-space fit
    ma1_val = None
    

    return {
        "acf_rows"   : rows,
        "bound"      : round(bound, 4),
        "arima_order": list(order),
        "n"          : n,
        "residuals"  : resid_list,
        "ma1_coef"   : ma1_val,
    }


@app.route("/api/arch_test", methods=["POST"])
def arch_test():
    """
    Fits Auto-ARIMA on target series alone, then runs Engle's ARCH LM test
    (het_arch) per lag — identical to _arch_lm_test in ar_garch_model.py.
    No time column or OLS step.
    """
    import pmdarima as pm
    from statsmodels.stats.diagnostic import het_arch

    data_json = session.get("data")
    if not data_json:
        return {"error": "No data found. Please upload your dataset first."}, 400

    body       = request.get_json(force=True)
    target_col = body.get("target_col", "").strip()
    lags       = body.get("lags", [1, 2, 3, 4, 5])

    if not target_col:
        return {"error": "target_col is required."}, 400
    if not lags:
        return {"error": "At least one lag must be specified."}, 400

    try:
        df = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    except Exception as e:
        return {"error": f"Failed to load data: {str(e)}"}, 500

    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found. Available: {list(df.columns)}"}, 400

    # ── Target series only ────────────────────────────────────────────────
    y = pd.Series(df[target_col].values.astype(float)).reset_index(drop=True)

    # ── Auto-ARIMA → residuals ────────────────────────────────────────────
    try:
        arima_fit = pm.auto_arima(
            y,
            seasonal=False,
            stepwise=True,
            information_criterion="aic",
            test="kpss",
            max_p=5, max_q=5, max_d=2,
            start_p=0, start_q=0,
            max_order=5,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order   = arima_fit.order
        p, d, q = order

        # pmdarima.resid() = equivalent of R's residuals(auto.arima(y))
        # = Kalman filter innovation residuals — correct for ARCH test
        resid = pd.Series(arima_fit.resid()).iloc[1:]  # Drop first residual (Kalman init artifact)
        resid = resid.reset_index(drop=True)
    except Exception as e:
        return {"error": f"Auto-ARIMA failed: {str(e)}"}, 500

    # ── ARCH LM test per lag (het_arch, same as _arch_lm_test) ───────────
    results = []
    for lag in lags:
        lag = int(lag)
        try:
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=lag)
            results.append({
                "lag"        : lag,
                "lm_stat"    : round(float(lm_stat),   4),
                "lm_pvalue"  : round(float(lm_pvalue), 4),
                "f_stat"     : round(float(f_stat),    4),
                "f_pvalue"   : round(float(f_pvalue),  4),
                "significant": bool(lm_pvalue < 0.05),
            })
        except Exception as e:
            results.append({
                "lag"        : lag,
                "lm_stat"    : None,
                "lm_pvalue"  : None,
                "f_stat"     : None,
                "f_pvalue"   : None,
                "significant": False,
                "error"      : str(e),
            })

    return {
        "results"    : results,
        "arima_order": list(order),
        "n_obs"      : len(y),
    }


@app.route("/download_results/<model_key>/<table_type>")
def download_results(model_key, table_type):
    """
    Download train/test/forecast table as Excel for a given model.
    table_type: 'train' | 'test' | 'forecast' | 'all'
    model_key : 'lstm' | 'gru' | 'rnn' | 'transformer' | 'ann' |
                'svr'  | 'rf'  | 'arima' | 'wavelet_ann' | 'comparison'
    """
    # ── 1. Collect tables ────────────────────────────────────────────────────
    tables = {}

    if model_key == "comparison":
        # Pull every model stored during comparison run
        model_keys = session.get("comparison_model_keys", [])
        
        # ✅ Build comparison metrics DataFrame
        metrics_rows = []
        for key in model_keys:
            model_data = session.get(f"comparison_{key}")
            if model_data:
                label = model_data.get("model_name", key.upper())
                metrics_rows.append({
                    "Model":       label,
                    "Train RMSE":  model_data.get("rmse_train"),
                    "Train MAE":   model_data.get("mae_train"),
                    "Train MAPE (%)": model_data.get("mape_train"),
                    "Test RMSE":   model_data.get("rmse_test"),
                    "Test MAE":    model_data.get("mae_test"),
                    "Test MAPE (%)":  model_data.get("mape_test"),
                    "Auto-Tuned":  "Yes" if model_data.get("auto_tuned") else "No",
                })
                tables[f"{label} – Train"] = pd.read_json(
                    StringIO(model_data["train_table"]), orient="split")
                tables[f"{label} – Test"]  = pd.read_json(
                    StringIO(model_data["test_table"]),  orient="split")

        if metrics_rows:
            # Sort by Test RMSE, metrics sheet goes first
            metrics_df = pd.DataFrame(metrics_rows).sort_values("Test RMSE").reset_index(drop=True)
            tables = {"Performance Metrics": metrics_df, **tables}

        # Forecasts (if generated)
        forecasts_json = session.get("comparison_forecasts", {})
        for fname, fjson in forecasts_json.items():
            tables[f"{fname.upper()} – Forecast"] = pd.read_json(
                StringIO(fjson), orient="split")

    else:
        # Single-model path
        stored = session.get(f"{model_key}_results")
        if not stored:
            # Try show_single_results path (full_results stored in training_progress)
            session_id = session.get("single_training_id")
            progress_data = get_progress(session_id)
            if session_id and progress_data:
                full = progress_data.get("full_results")
                if full:
                    stored = {
                        "train_table": full["train_table"],
                        "test_table":  full["test_table"],
                    }

        if not stored:
            flash("Results not found. Please run the model first.", "error")
            return redirect(url_for("models"))

        if table_type in ("train", "all"):
            tables["Train Predictions"] = pd.read_json(
                StringIO(stored["train_table"]), orient="split")
        if table_type in ("test", "all"):
            tables["Test Predictions"] = pd.read_json(
                StringIO(stored["test_table"]),  orient="split")
        if table_type in ("forecast", "all"):
            # Forecast table lives in model-specific session key
            forecast_key = f"{model_key}_forecast"
            fcast_json   = session.get(forecast_key)
            if fcast_json:
                tables["Forecast"] = pd.read_json(
                    StringIO(fcast_json), orient="split")

            # ✅ Build metrics DataFrame for single model
            meta = stored.get("meta", {})
            metrics_df = pd.DataFrame([
                {"Split": "Training", "RMSE": meta.get("rmse_train"), "MAE": meta.get("mae_train"), "MAPE (%)": meta.get("mape_train")},
                {"Split": "Testing",  "RMSE": meta.get("rmse_test"),  "MAE": meta.get("mae_test"),  "MAPE (%)": meta.get("mape_test")},
            ])
            tables = {"Performance Metrics": metrics_df, **tables}  # metrics sheet first
    if not tables:
        flash("No data available to download.", "error")
        return redirect(url_for("models"))

    # ── 2. Build Excel workbook ──────────────────────────────────────────────
    wb = openpyxl.Workbook()
    wb.remove(wb.active)   # remove default blank sheet

    HEADER_FILL  = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT  = Font(bold=True, color="FFFFFF", size=11)
    ALT_FILL     = PatternFill("solid", fgColor="D6E4F0")
    BORDER_SIDE  = Side(style="thin", color="B0BEC5")
    CELL_BORDER  = Border(left=BORDER_SIDE, right=BORDER_SIDE,
                          top=BORDER_SIDE,  bottom=BORDER_SIDE)

    for sheet_name, df in tables.items():
        # Strip all characters invalid in Excel sheet names
        import re
        safe_name = re.sub(r'[\[\]:\\/*?]', '', sheet_name)[:31].strip()   # Excel sheet name limit
        ws = wb.create_sheet(title=safe_name)

        # Header row
        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = ws.cell(row=1, column=col_idx, value=str(col_name))
            cell.font      = HEADER_FONT
            cell.fill      = HEADER_FILL
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = CELL_BORDER

        # Data rows
        for row_idx, row in enumerate(df.itertuples(index=False), start=2):
            fill = ALT_FILL if row_idx % 2 == 0 else PatternFill()
            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=row_idx, column=col_idx,
                               value=round(float(value), 4)
                               if isinstance(value, (float, int)) else value)
                cell.fill      = fill
                cell.alignment = Alignment(horizontal="center")
                cell.border    = CELL_BORDER

        # Auto-width
        for col_idx, col_name in enumerate(df.columns, start=1):
            max_len = max(
                len(str(col_name)),
                *[len(str(ws.cell(r, col_idx).value or ""))
                  for r in range(2, ws.max_row + 1)]
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 30)

        ws.freeze_panes = "A2"   # freeze header

    # ── 3. Stream to browser ─────────────────────────────────────────────────
    output   = BytesIO()
    wb.save(output)
    output.seek(0)

    filename = f"{model_key}_results_{int(time.time())}.xlsx"
    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/download_comparison_excel")
def download_comparison_excel():
    """Shortcut — download full comparison as Excel (all models, all tables)."""
    return redirect(url_for("download_results",
                            model_key="comparison", table_type="all"))


@app.route("/print_results/<model_key>")
def print_results(model_key):
    """
    Render a clean, print-friendly HTML page for a single model.
    Open in new tab → user can Ctrl+P or save as PDF from browser.
    """
    stored = session.get(f"{model_key}_results")

    # Fallback: try training_progress (right after show_single_results)
    if not stored:
        session_id = session.get("single_training_id")
        progress_data = get_progress(session_id)
        if session_id and progress_data:
            full = progress_data.get("full_results")
            if full:
                stored = {
                    "meta": {
                        "model_key":   model_key,
                        "model_name":  full.get("model_name", model_key.upper()),
                        "frequency":   session.get("frequency", "annual"),
                        "horizon":     session.get("horizon", 1),
                        "rmse_train":  full["rmse_train"],
                        "mae_train":   full["mae_train"],
                        "mape_train":  full["mape_train"],
                        "rmse_test":   full["rmse_test"],
                        "mae_test":    full["mae_test"],
                        "mape_test":   full["mape_test"],
                        "model_config": full.get("model_config", {}),
                        "auto_tuned":  full.get("auto_tuned", False),
                    },
                    "train_table": full["train_table"],
                    "test_table":  full["test_table"],
                }

    if not stored:
        flash("Results not found. Please run the model first.", "error")
        return redirect(url_for("models"))

    meta        = stored["meta"]
    train_table = pd.read_json(StringIO(stored["train_table"]), orient="split")
    test_table  = pd.read_json(StringIO(stored["test_table"]),  orient="split")

    # Forecast (optional)
    forecast_table = None
    fcast_json = session.get(f"{model_key}_forecast")
    if fcast_json:
        forecast_table = pd.read_json(StringIO(fcast_json), orient="split")

    return render_template(
        "print_results.html",
        meta=meta,
        train_table=train_table,
        test_table=test_table,
        forecast_table=forecast_table,
        model_key=model_key,
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

@app.route("/print_comparison")
def print_comparison():
    """
    Render a clean, print-friendly HTML page for all compared models.
    """
    model_keys = session.get("comparison_model_keys", [])
    comparison_results = {}
    for key in model_keys:
        model_data = session.get(f"comparison_{key}")
        if model_data:
            comparison_results[key] = model_data

    if not comparison_results:
        flash("No comparison data found. Please train models first.", "error")
        return redirect(url_for("models"))

    models_data = []
    for model_key, data in comparison_results.items():
        if data.get("train_table") is None:
            continue
        models_data.append({
            "key":        model_key,
            "name":       data["model_name"],
            "auto_tuned": data["auto_tuned"],
            "model_config": data.get("model_config", {}),
            "rmse_train": data["rmse_train"],
            "mae_train":  data["mae_train"],
            "mape_train": data["mape_train"],
            "rmse_test":  data["rmse_test"],
            "mae_test":   data["mae_test"],
            "mape_test":  data["mape_test"],
            "train_table": pd.read_json(StringIO(data["train_table"]), orient="split"),
            "test_table":  pd.read_json(StringIO(data["test_table"]),  orient="split"),
        })

    # Sort by test RMSE
    models_data.sort(key=lambda x: (x["rmse_test"] is None, x["rmse_test"] or float('inf')))

    # Forecasts (optional)
    forecasts = {}
    forecasts_json = session.get("comparison_forecasts", {})
    for fname, fjson in forecasts_json.items():
        forecasts[fname] = pd.read_json(StringIO(fjson), orient="split")

    return render_template(
        "print_comparison.html",
        models=models_data,
        forecasts=forecasts,
        horizon=session.get("comparison_horizon", 5),
        frequency=session.get("comparison_frequency", "annual"),
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )

# =====================================================
# STATISTICS DOWNLOAD & PRINT ROUTES
# =====================================================

@app.route("/download_statistics")
def download_statistics():
    """Download statistics results as a formatted Excel file."""
    results = session.get("statistics_results", {})
    selected_stats = session.get("selected_stats", {})

    if not results:
        flash("No statistics results found. Please generate statistics first.", "error")
        return redirect(url_for("summary_statistics"))

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
    ALT_FILL    = PatternFill("solid", fgColor="D6E4F0")
    BORDER_SIDE = Side(style="thin", color="B0BEC5")
    CELL_BORDER = Border(left=BORDER_SIDE, right=BORDER_SIDE,
                         top=BORDER_SIDE,  bottom=BORDER_SIDE)
    SECTION_FILL = PatternFill("solid", fgColor="764BA2")
    SECTION_FONT = Font(bold=True, color="FFFFFF", size=10)

    def write_sheet(ws, rows, headers):
        """Write a list of (label, value) rows with styled headers."""
        for col_idx, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col_idx, value=h)
            cell.font      = HEADER_FONT
            cell.fill      = HEADER_FILL
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = CELL_BORDER

        for row_idx, row in enumerate(rows, start=2):
            fill = ALT_FILL if row_idx % 2 == 0 else PatternFill()
            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.fill      = fill
                cell.alignment = Alignment(horizontal="left" if col_idx == 1 else "center")
                cell.border    = CELL_BORDER

        # Auto-width
        for col_idx in range(1, len(headers) + 1):
            max_len = max(
                len(str(ws.cell(r, col_idx).value or ""))
                for r in range(1, ws.max_row + 1)
            )
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 50)
        ws.freeze_panes = "A2"

    # ── Sheet 1: Summary Statistics ─────────────────────────────────────────
    summary = results.get("summary_stats", {})
    if summary:
        ws = wb.create_sheet("Summary Statistics")
        # summary is dict: {variable: {stat: value}}
        all_stats = []
        for var, stats in summary.items():
            for stat_name, stat_val in stats.items():
                all_stats.append((var, stat_name, stat_val))
        write_sheet(ws, all_stats, ["Variable", "Statistic", "Value"])

    def flatten_test_results(results_dict):
        """
        Handles two structures:
        A) {var: {metric: value}}                            → flat
        B) {var: {test_name: {metric: value, ...}}}          → nested (your case)
        """
        rows = []
        for var, res in results_dict.items():
            if not isinstance(res, dict):
                rows.append((var, "Result", str(res)))
                continue
            # Check if values are themselves dicts (nested)
            first_val = next(iter(res.values()), None)
            if isinstance(first_val, dict):
                # Structure B: {test_name: {metric: value}}
                for test_name, metrics in res.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            safe_val = float(value) if hasattr(value, '__float__') else str(value)
                            rows.append((var, test_name, metric, safe_val))
                    else:
                        rows.append((var, test_name, "Result", str(metrics)))
            else:
                # Structure A: {metric: value}
                for metric, value in res.items():
                    safe_val = float(value) if hasattr(value, '__float__') else str(value)
                    rows.append((var, "—", metric, safe_val))
        return rows

    # ── Sheet 2: Normality Tests ─────────────────────────────────────────────
    normality = results.get("normality") or results.get("normality_tests", {})
    if normality:
        ws = wb.create_sheet("Normality Test")
        rows = flatten_test_results(normality)
        write_sheet(ws, rows, ["Variable", "Test", "Metric", "Value"])

    # ── Sheet 3: Stationarity Tests ──────────────────────────────────────────
    stationarity = results.get("stationarity") or results.get("stationarity_tests", {})
    if stationarity:
        ws = wb.create_sheet("Stationarity Test")
        rows = flatten_test_results(stationarity)
        write_sheet(ws, rows, ["Variable", "Test", "Metric", "Value"])

    # ── Sheet 4: Trend Tests ─────────────────────────────────────────────────
    trend = results.get("trend") or results.get("trend_tests", {})
    if trend:
        ws = wb.create_sheet("Trend Tests")
        rows = flatten_test_results(trend)
        write_sheet(ws, rows, ["Variable", "Test", "Metric", "Value"])

    # ── Sheet 5: Non-Linearity Tests ─────────────────────────────────────────
    nonlinearity = results.get("nonlinearity") or results.get("nonlinearity_tests", {})
    if nonlinearity:
        ws = wb.create_sheet("Non-Linearity Test")
        rows = flatten_test_results(nonlinearity)
        write_sheet(ws, rows, ["Variable", "Test", "Metric", "Value"])

    if not wb.sheetnames:
        flash("No tabular statistics to download.", "error")
        return redirect(url_for("show_statistics_results"))

    output = BytesIO()
    wb.save(output)
    output.seek(0)

    filename = f"statistics_results_{int(time.time())}.xlsx"
    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/print_statistics")
def print_statistics():
    """Render a clean print-friendly page for statistics results."""
    results       = session.get("statistics_results", {})
    selected_stats = session.get("selected_stats", {})
    exog_cols     = session.get("exog_cols", [])
    target_col    = session.get("target_col", "Study Variable")

    if not results:
        flash("No statistics results found. Please generate statistics first.", "error")
        return redirect(url_for("summary_statistics"))

    return render_template(
        "print_statistics.html",
        results=results,
        selected_stats=selected_stats,
        exog_cols=exog_cols,
        target_col=target_col,
        is_multivariate=len(exog_cols) > 0,
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    )
@app.route("/residual_diagnostics/<model_key>")
def residual_diagnostics(model_key):
    import numpy as np
    from residual_diagnostics_route import run_residual_diagnostics, generate_residual_plot
 
    # Try training_progress first (right after training)
    session_id   = session.get("single_training_id")
    full_results = None
    progress_data = get_progress(session_id)
    if session_id and progress_data:
        full_results = progress_data.get("full_results")
 
    # Fallback: rebuild from session
    if not full_results:
        stored = session.get(f"{model_key}_results")
        if stored:
            full_results = stored.get("meta", {})
 
    residuals = None
    if full_results:
        residuals = full_results.get("residuals")
 
    if not residuals:
        flash("Residuals not available. Please retrain the model.", "warning")
        return redirect(url_for("select_models"))
 
    residuals_np = np.array(residuals, dtype=float)
    diag         = run_residual_diagnostics(residuals_np)
    plot_file    = generate_residual_plot(residuals_np, model_key, app.static_folder)
 
    return render_template(
        "residual_diagnostics.html",
        model_key       = model_key,
        model_name      = full_results.get("model_name", model_key.upper()),
        diag            = diag,
        plot_file       = plot_file,
        ts              = int(time.time()),
        from_comparison = False,      # ← single model: back goes to show_single_results
    )

@app.route("/residual_diagnostics_comparison/<model_key>")
def residual_diagnostics_comparison(model_key):
    import numpy as np
    from residual_diagnostics_route import run_residual_diagnostics, generate_residual_plot
 
    # Pull stored comparison result for this model
    model_data = session.get(f"comparison_{model_key}")
    if not model_data:
        flash("No results found for this model. Please retrain.", "warning")
        return redirect(url_for("show_comparison"))
 
    residuals = model_data.get("residuals")
    if not residuals:
        flash(f"Residuals not available for {model_key.upper()}. Please retrain.", "warning")
        return redirect(url_for("show_comparison"))
 
    residuals_np = np.array(residuals, dtype=float)
    diag         = run_residual_diagnostics(residuals_np)
    plot_file    = generate_residual_plot(residuals_np, f"cmp_{model_key}", app.static_folder)
 
    return render_template(
        "residual_diagnostics.html",
        model_key  = model_key,
        model_name = model_data.get("model_name", model_key.upper()),
        diag       = diag,
        plot_file  = plot_file,
        ts         = int(time.time()),
        from_comparison = True,   # flag so Back button goes to compare page
    )

@app.route("/forecast_result/<model_key>")
def show_forecast_result(model_key):
    """Dedicated forecast results page for single model"""
    forecast_json = session.get(f"{model_key}_forecast")
    if not forecast_json:
        flash("No forecast found. Please generate a forecast first.", "error")
        return redirect(url_for("select_models"))

    forecast_table = pd.read_json(StringIO(forecast_json), orient="split")

    stored = session.get(f"{model_key}_results")
    model_name = model_key.upper()
    if stored and stored.get("meta"):
        model_name = stored["meta"].get("model_name", model_key.upper())

    return render_template(
        "forecast_result.html",
        forecast_table=forecast_table,
        model_key=model_key,
        model_name=model_name,
        horizon=session.get("horizon", 1),
        frequency=session.get("frequency", "annual"),
        ts=int(time.time()),
    )
# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
