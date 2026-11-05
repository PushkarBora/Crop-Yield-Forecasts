"""
Wavelet-RNN
===========
y -> causal wavelet decomposition [D1..DJ, AJ] (no look-ahead)
  -> one simple RNN neural network reads lags of every component (+ exogenous variables)
  -> predicts the change y(t) - y(t-1)
Engine: models/wavelet_engine.py.  Form fields start with "wrnn_" (auto_tune_wrnn).
"""
from models.wavelet_engine import run, extra_settings


def run_wavelet_rnn(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    s = extra_settings(params, "wrnn", "rnn")
    return run("wavelet_rnn", "rnn", "Wavelet-RNN", data, params, s,
               horizon, frequency, mode, log_callback)
