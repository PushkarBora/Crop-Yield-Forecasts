"""
Wavelet-SVR
===========
y -> causal wavelet decomposition [D1..DJ, AJ] (no look-ahead)
  -> one Support Vector Regression reads lags of every component (+ exogenous variables)
  -> predicts the change y(t) - y(t-1)
Engine: models/wavelet_engine.py.  Form fields start with "wsvr_" (auto_tune_wsvr).
"""
from models.wavelet_engine import run, extra_settings


def run_wavelet_svr(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    s = extra_settings(params, "wsvr", "svr")
    return run("wavelet_svr", "svr", "Wavelet-SVR", data, params, s,
               horizon, frequency, mode, log_callback)
