"""
Wavelet-ANN  (fast version)
===========================
Same function name, inputs and outputs as before, so app.py and the forms
are unchanged. The work is done by models/wavelet_engine.py:

  y -> causal MODWT-type decomposition [D1..DJ, AJ]
    -> one ANN reads lags of every component (+ exogenous variables)
    -> predicts the change y(t) - y(t-1)

Form settings used (unchanged):
  auto_tune_wann, wann_wavelets, wann_min_level / wann_max_level,
  wann_lags_min / _max / _step,
  ann_hidden_layer_1_*, ann_hidden_layer_2_*  (architectures tried),
  ann_dropout_1_min, ann_weight_decay_min, ann_huber_delta_min,
  manual mode also: ann_learning_rate_min, ann_epochs_min, ann_patience_min
"""
from models.wavelet_engine import (run, base_settings, wavelet_levels_lags,
                                   int_range, spread)


def list_available_wavelets():
    import pywt
    return pywt.wavelist(kind="discrete")


def run_wavelet_ann(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = params or {}
    auto = bool(params.get("auto_tune_wann", True))
    levels, lags = wavelet_levels_lags(params)
    s = base_settings(params, auto, levels, lags)

    h1 = int_range(params, "ann_hidden_layer_1_min", "ann_hidden_layer_1_max",
                   "ann_hidden_layer_1_step", 8, 16, 8)
    h2 = int_range(params, "ann_hidden_layer_2_min", "ann_hidden_layer_2_max",
                   "ann_hidden_layer_2_step", 4, 8, 4)
    s["archs"] = ([(a, b) for a in spread(h1) for b in spread(h2)] if auto
                  else [(max(2, h1[0]), max(2, h2[0]))])

    return run("wavelet_ann", "ann", "Wavelet-ANN", data, params, s,
               horizon, frequency, mode, log_callback)
