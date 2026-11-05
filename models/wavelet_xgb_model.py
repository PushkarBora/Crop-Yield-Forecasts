"""
Wavelet-XGBoost
===============
y -> causal wavelet decomposition [D1..DJ, AJ] (no look-ahead)
  -> one XGBoost (gradient boosted trees) reads lags of every component (+ exogenous variables)
  -> predicts the change y(t) - y(t-1)
Engine: models/wavelet_engine.py.  Form fields start with "wxgb_" (auto_tune_wxgb).
"""
from models.wavelet_engine import run, extra_settings


def run_wavelet_xgb(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    s = extra_settings(params, "wxgb", "xgb")
    return run("wavelet_xgb", "xgb", "Wavelet-XGBoost", data, params, s,
               horizon, frequency, mode, log_callback)
