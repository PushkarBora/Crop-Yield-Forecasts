"""
Wavelet-RF
==========
y -> causal wavelet decomposition [D1..DJ, AJ] (no look-ahead)
  -> one Random Forest reads lags of every component (+ exogenous variables)
  -> predicts the change y(t) - y(t-1)
Engine: models/wavelet_engine.py.  Form fields start with "wrf_" (auto_tune_wrf).
"""
from models.wavelet_engine import run, extra_settings


def run_wavelet_rf(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    s = extra_settings(params, "wrf", "rf")
    return run("wavelet_rf", "rf", "Wavelet-RF", data, params, s,
               horizon, frequency, mode, log_callback)
