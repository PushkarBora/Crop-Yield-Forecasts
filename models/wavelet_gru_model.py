"""
Wavelet-GRU
===========
y -> causal wavelet decomposition [D1..DJ, AJ] (no look-ahead)
  -> one GRU neural network reads lags of every component (+ exogenous variables)
  -> predicts the change y(t) - y(t-1)
Engine: models/wavelet_engine.py.  Form fields start with "wgru_" (auto_tune_wgru).
"""
from models.wavelet_engine import run, extra_settings


def run_wavelet_gru(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    s = extra_settings(params, "wgru", "gru")
    return run("wavelet_gru", "gru", "Wavelet-GRU", data, params, s,
               horizon, frequency, mode, log_callback)
