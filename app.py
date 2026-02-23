from flask import Flask, render_template, request, session, flash, redirect, url_for
import os
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
from models.ann_model import run_ann
from models.svr_model import run_svr
from models.rf_model import run_rf
from models.arima_model import run_arima
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
app.secret_key = "crop-yield-secret"   # REQUIRED for session storage

 #ADD SERVER-SIDE SESSION (REQUIRED!)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
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
training_progress = {}
progress_lock = threading.Lock()

def update_progress(session_id, data):
    """Update training progress for a session"""
    with progress_lock:
        if session_id not in training_progress:
            training_progress[session_id] = {
                'current_model': '',
                'current_step': '',
                'models_completed': 0,
                'total_models': 0,
                'progress_percent': 0,
                'logs': [],
                'status': 'running',
                'error': None
            }
        training_progress[session_id].update(data)
        
        # Keep only last 50 log entries
        if 'logs' in training_progress[session_id]:
            training_progress[session_id]['logs'] = training_progress[session_id]['logs'][-50:]

def get_progress(session_id):
    """Get current progress for a session"""
    with progress_lock:
        return training_progress.get(session_id, {})

def clear_progress(session_id):
    """Clear progress data for a session"""
    with progress_lock:
        if session_id in training_progress:
            del training_progress[session_id]

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
    
    # Manual parameters (used if auto-tune disabled)
    # After transformer manual params section:
    params["transformer_lags"] = safe_int(
    form.get("transformer_lags") or form.get("lags"), 3
    )
    params["d_model"] = safe_int(form.get("d_model"), 8)
    params["nhead"] = safe_int(form.get("nhead"), 2)
    params["dim_feedforward"] = safe_int(form.get("dim_feedforward"), 16)
    params["activation"] = form.get("activation", "gelu")
    
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
        params['svr_kernel_rbf'] = safe_bool(form.get('svr_kernel_rbf'))  # ✅ FIXED
        params['svr_kernel_poly'] = safe_bool(form.get('svr_kernel_poly'))  # ✅ FIXED
        params['svr_kernel_sigmoid'] = safe_bool(form.get('svr_kernel_sigmoid'))  # ✅ FIXED

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
    params["rf_n_estimators_max"] = safe_int(form.get("rf_n_estimators_max"), 500)
    params["rf_n_estimators_step"] = safe_int(form.get("rf_n_estimators_step"), 100)
    
    params["rf_max_depth_min"] = safe_int(form.get("rf_max_depth_min"), 2)
    params["rf_max_depth_max"] = safe_int(form.get("rf_max_depth_max"), 10)
    params["rf_max_depth_step"] = safe_int(form.get("rf_max_depth_step"), 2)
    params["rf_max_depth_include_none"] = safe_bool(form.get("rf_max_depth_include_none"))
    
    params["rf_min_samples_split_min"] = safe_int(form.get("rf_min_samples_split_min"), 2)
    params["rf_min_samples_split_max"] = safe_int(form.get("rf_min_samples_split_max"), 10)
    params["rf_min_samples_split_step"] = safe_int(form.get("rf_min_samples_split_step"), 2)
    
    params["rf_min_samples_leaf_min"] = safe_int(form.get("rf_min_samples_leaf_min"), 1)
    params["rf_min_samples_leaf_max"] = safe_int(form.get("rf_min_samples_leaf_max"), 4)
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
    params["max_features"] = safe_int(form.get("max_features"), 1)
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

@app.route("/upload_data", methods=["POST"])
def upload_data():
    # ✅ CLEAR STALE SESSION DATA FROM PREVIOUS UPLOADS
    for key in ["data", "exog_cols", "time_col", "target_col", 
                "comparison_model_keys", "comparison_results",
                "comparison_forecasts", "comparison_id"]:
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

    next_step = request.form.get("next_step", "statistics")
    if next_step == "models":
        return redirect(url_for("select_models"))
    else:
        return redirect(url_for("summary_statistics"))
        
    


# =====================================================
# REPLACE your existing /summary_statistics route
# with this one in app.py
# =====================================================

@app.route("/summary_statistics")
def summary_statistics():
    """
    Display the summary statistics selection page.
    Detects whether the uploaded dataset is multivariate (has Mean_T & Mean_Rain)
    and passes that flag to the template so it can conditionally show/hide
    correlation-based plots and CCF.
    """
    if "data" not in session:
        flash("Please upload data first.", "error")
        return redirect(url_for("models"))

    # ── Detect multivariate vs univariate ─────────────────────────────────────
    is_multivariate = False
    try:
        from io import StringIO
        data = pd.read_json(StringIO(session["data"]), orient="split", convert_dates=False)
        cols_lower = [c.lower() for c in data.columns]

      # ✅ NEW
        exog_cols = session.get("exog_cols", [])
        is_multivariate = len(exog_cols) > 0
        has_yield     = any("yield" in c for c in cols_lower)

        #✅ Fix — remove that line, the correct one is already above it:
        exog_cols = session.get("exog_cols", [])
        is_multivariate = len(exog_cols) > 0
    except Exception as e:
        print(f"Warning: could not detect data type for statistics page: {e}")

    return render_template(
        "summary_statistics.html",
        is_multivariate=is_multivariate,
    )


# =====================================================
# ALSO UPDATE /generate_statistics to pass
# trend_tests to generate_all_statistics
# =====================================================

@app.route("/generate_statistics", methods=["POST"])
def generate_statistics():
    """
    Generate selected statistics and tests, then display results.
    """
    data_json = session.get("data")
    if not data_json:
        flash("No data found. Please upload again.", "error")
        return redirect(url_for("models"))

    data = pd.read_json(StringIO(data_json), orient="split",convert_dates=False)

    summary_stats       = request.form.getlist("summary_stats")
    normality_tests     = request.form.getlist("normality_tests")
    stationarity_tests  = request.form.getlist("stationarity_tests")
    nonlinearity_tests  = request.form.getlist("nonlinearity_tests")
    trend_tests         = request.form.getlist("trend_tests")      # ← NEW
    plots               = request.form.getlist("plots")

    print("\n" + "=" * 80)
    print("🔍 FLASK ROUTE DEBUG")
    print("=" * 80)
    print(f"nonlinearity_tests from form : {nonlinearity_tests}")
    print(f"trend_tests from form        : {trend_tests}")
    print("=" * 80 + "\n")

    from statistics_generator import generate_all_statistics

    try:
        results = generate_all_statistics(
            data=data,
            summary_stats=summary_stats,
            normality_tests=normality_tests,
            stationarity_tests=stationarity_tests,
            nonlinearity_tests=nonlinearity_tests,
            trend_tests=trend_tests,           # ← NEW
            plots=plots,

            # ✅ Pass column names from session
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
            "trend_tests":        trend_tests,   # ← NEW
            "plots":              plots,
        }
        session.modified = True

        flash("Statistics generated successfully!", "success")
        return redirect(url_for("show_statistics_results"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error generating statistics: {str(e)}", "error")
        return redirect(url_for("summary_statistics"))

@app.route("/statistics_results")
def show_statistics_results():
    """
    Display the generated statistics results
    """
    # Get results from session
    results = session.get("statistics_results", {})
    selected_stats = session.get("selected_stats", {})
    
    if not selected_stats:
        flash("No statistics selected.", "error")
        return redirect(url_for("summary_statistics"))
    
    return render_template(
        "statistics_results.html",
        results=results,
        selected_stats=selected_stats,
        exog_cols=session.get("exog_cols", []),        # ✅ ADD
        is_multivariate=len(session.get("exog_cols", [])) > 0,        # ✅ ADD
    )


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
    
    return render_template("model_selection.html")


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
        'status': 'running'
    })
    
    # Start training in background thread
    def train_in_background():
        try:
            data = pd.read_json(StringIO(data_json), orient="split",convert_dates=False)
            
            # Update progress
            update_progress(session_id, {
                'current_step': 'Loading model architecture...',
                'logs': training_progress[session_id]['logs'] + [f'▶️ Training {model_name.upper()}...']
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
                with progress_lock:
                    training_progress[session_id]['full_results'] = result
                
                # Mark as complete
                update_progress(session_id, {
                    'status': 'complete',
                    'current_step': 'Training complete!',
                    'progress_percent': 100,
                    'models_completed': 1,
                    'logs': training_progress[session_id]['logs'] + [f'✅ {model_name.upper()} training complete!', '📊 Preparing results...']
                })
            else:
                raise Exception("Training failed")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            update_progress(session_id, {
                'status': 'error',
                'error': str(e),
                'logs': training_progress[session_id]['logs'] + [f'❌ ERROR: {str(e)}', error_details]
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

    if not session_id or session_id not in training_progress:
        flash("Training session not found.", "error")
        return redirect(url_for("models"))
    
    # Get results from training_progress
    full_results = training_progress[session_id].get('full_results')
    
    if not full_results:
        flash("Results not found.", "error")
        return redirect(url_for("models"))
    
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
        full_results.get("data_type") == "Univariate"
    )
    
    return render_template(
        "results.html",
        results=results_for_template,
        show_future_inputs=True,
        is_arima=(model_name == "arima"),
        is_univariate=is_univariate , # ✅ NEW FLAG
        exog_cols=session.get("exog_cols", [])
    )
##########################################################
#Training progress stream for SSE
#############################################################
@app.route("/training_progress_stream")
def training_progress_stream():
    """Server-Sent Events endpoint for training progress"""
    session_id = request.args.get('session_id', 'default')
    
    def generate():
        """Generate SSE stream"""
        last_data = None
        while True:
            current_data = get_progress(session_id)
            
            # Only send if data changed
            if current_data != last_data:
                yield f"data: {json.dumps(current_data)}\n\n"
                last_data = current_data.copy()
                
                # Stop streaming if complete or error
                if current_data.get('status') in ['complete', 'error']:
                    break
            
            time.sleep(0.5)  # Check every 500ms
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
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
                    'logs': training_progress[session_id]['logs'] + [f'\n▶️ Training {model_name.upper()} (Model {idx}/{total})']
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
                        'logs': training_progress[session_id]['logs'] + [f'✅ {model_name.upper()} training complete!']
                    })
                else:
                    # Model failed - log it but continue
                    update_progress(session_id, {
                        'logs': training_progress[session_id]['logs'] + [f'⚠️ {model_name.upper()} training failed - skipped']
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
                training_progress[session_id]['results'] = results.copy()

            # VERIFY IT WAS STORED
            print("=" * 80)
            print("🔍 VERIFIED STORAGE")
            print("=" * 80)
            print(f"Stored results keys: {list(training_progress[session_id]['results'].keys())}")
            print("=" * 80)
            # ✅✅✅ END OF NEW STORAGE BLOCK ✅✅✅

            
            # Mark as complete
            update_progress(session_id, {
                'status': 'complete',
                'current_model': 'All models trained!',
                'current_step': 'Redirecting to results...',
                'progress_percent': 100,
                'logs': training_progress[session_id]['logs'] + ['\n🎉 All models trained successfully!', '📊 Preparing results...']
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            update_progress(session_id, {
                'status': 'error',
                'error': str(e),
                'logs': training_progress[session_id]['logs'] + [f'\n❌ ERROR: {str(e)}', error_details]
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
            
        elif model_name == "svr":
            log(f"  → Loading SVR model...")
            # ✅ PASS log_callback HERE
            result = run_svr(data, params, horizon, frequency, mode="train", log_callback=log)
            
        elif model_name == "rf":
            log(f"  → Loading Random Forest model...")
            # ✅ PASS log_callback HERE
            result = run_rf(data, params, horizon, frequency, mode="train", log_callback=log)
            
        elif model_name == "arima":
            log(f"  → Loading ARIMA model...")
            # ✅ PASS log_callback HERE
            result = run_arima(data, params, horizon, frequency, mode="train", log_callback=log)
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
            "train_table": result["train_table"].to_json(orient="split"),
            "test_table": result["test_table"].to_json(orient="split"),
            "auto_tuned": result.get("auto_tuned", False),
            "model_config": result.get("model_config", {}),
            "data_type": result.get("data_type", "Multivariate"),  # ✅ ADD THIS LINE
             # ✅ ADD THIS LINE FOR ARIMA ONLY
            
            "trained_params": params if model_name == "arima" else None,
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
    if session_id and session_id in training_progress:
        fresh_results = training_progress[session_id].get('results')
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
        })
    
    # Sort by test RMSE (best first)
    models_data.sort(key=lambda x: x["rmse_test"])
    
    # Best model is the first one (lowest RMSE)
    best_model = models_data[0] if models_data else None
    
    #Detect univariate models
    has_multivariate = False
    has_univariate = False

    for model_key, model_data in comparison_results.items():
        data_type = model_data.get("data_type", "Multivariate")
        if data_type == "Multivariate":
            has_multivariate = True
        else:
            has_univariate = True
    
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
        all_univariate = (not has_multivariate), #ADD THIS FLAG TO INDICATE IF ALL MODELS ARE UNIVARIATE
        exog_cols=session.get("exog_cols", []),
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
        models_data.append({
            "key": model_key,
            "name": data["model_name"],
            "train_table": pd.read_json(StringIO(data["train_table"]), orient="split"),
            "test_table": pd.read_json(StringIO(data["test_table"]), orient="split"),
            "rmse_test": data["rmse_test"],
        })
    
    # Sort by RMSE
    models_data.sort(key=lambda x: x["rmse_test"])
    
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
                    
                elif model_name == "arima":
                    forecast_result = run_arima(
                        data=None,
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
                future_exog=None
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
    
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )
        
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
        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )

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
    
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )
        
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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )
    

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
    
            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )
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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )


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

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )

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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )


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

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )

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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )


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

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )

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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False
        )

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

            return render_template(
                "results.html",
                results=results,
                show_future_inputs=False,
                is_univariate=True
            )

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

        return render_template(
            "results.html",
            results=results,
            show_future_inputs=False,
        )

    elif model_name == "arima":
        # ✅ ADD THIS DEBUG LINE FIRST
        print(f"🔥 ARIMA ROUTE HIT - is_forecast_request will be: {request.form.get('arima_forecast_trigger', '').strip()}")
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
        # ✅✅✅ ADD THESE DEBUG LINES ✅✅✅
        print("=" * 80)
        print("🔍 DEBUG: ARIMA FORECAST SECTION")
        print("=" * 80)
        print(f"stored keys: {list(stored.keys())}")
        print(f"stored['trained_params'] exists: {'trained_params' in stored}")
    
        if 'trained_params' in stored:
            tp = stored['trained_params']
            print(f"trained_params type: {type(tp)}")
            if isinstance(tp, dict):
                print(f"trained_params keys: {list(tp.keys())}")
                print(f"  auto_arima: {tp.get('auto_arima')}")
                print(f"  arima_p: {tp.get('arima_p')}")
                print(f"  arima_d: {tp.get('arima_d')}")
                print(f"  arima_q: {tp.get('arima_q')}")
            else:
                print(f"trained_params value: {tp}")
        else:
            print("⚠️ trained_params NOT in stored!")
        print("=" * 80)
        # ✅✅✅ END DEBUG BLOCK ✅✅✅

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

        return render_template(
        "results.html",
        results=results,
        show_future_inputs=False
    )
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
        return {"error": "No data found in session."}, 400

    data = pd.read_json(StringIO(data_json), orient="split", convert_dates=False)
    exog_cols = session.get("exog_cols", [])
    horizon   = session.get("horizon", 1)

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
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
