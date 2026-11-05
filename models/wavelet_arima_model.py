"""
Wavelet-ARIMA  (new)
====================
y -> causal MODWT-type decomposition [D1..DJ, AJ]
  -> ARMA(X)(p, q) on each detail Dj, ARIMA(X)(p, d, q) on the trend AJ
     (each exogenous variable is decomposed the same way; component k of y
      uses component k of the exogenous variables)
  -> component forecasts are added together.
Orders are chosen per component by AIC; the wavelet filter and level are
chosen on a validation slice of the training data. Engine: models/wavelet_engine.py

Form settings (templates/partials/wavelet_arima_config.html):
  auto_tune_warima, warima_wavelets, warima_min_level, warima_max_level,
  warima_max_p, warima_max_q, warima_manual_wavelet, warima_manual_level
"""
from models.wavelet_engine import run, resolve_filters, num


def run_wavelet_arima(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    auto = str(params.get("auto_tune_warima", True)).lower() in ("true", "1", "on")
    if auto:
        filters = resolve_filters(params.get("warima_wavelets") or ["haar", "db2", "db4", "sym8"])
        lv_min = max(1, num(params, "warima_min_level", 1, int))
        lv_max = max(lv_min, min(8, num(params, "warima_max_level", 3, int)))
        levels = list(range(lv_min, lv_max + 1))
    else:
        filters = resolve_filters([params.get("warima_manual_wavelet") or "haar"])[:1]
        levels = [max(1, min(8, num(params, "warima_manual_level", 2, int)))]

    settings = {
        "auto": auto, "filters": filters, "levels": levels, "lags": [3],
        "max_p": max(0, min(4, num(params, "warima_max_p", 2, int))),
        "max_q": max(0, min(4, num(params, "warima_max_q", 2, int))),
    }
    return run("wavelet_arima", "arima", "Wavelet-ARIMA", data, params, settings,
               horizon, frequency, mode, log_callback)
