"""
Wavelet-LSTM  (fast version)
============================
Same function name, inputs and outputs as before (app.py unchanged).
Engine: models/wavelet_engine.py — one LSTM reads all wavelet components
(+ exogenous variables) as a multichannel sequence.

Form settings used (unchanged; app.py maps wlstm_*_resolved into these):
  auto_tune_wlstm, wann_wavelets, wann_min_level / wann_max_level,
  wann_lags_*, ann_hidden_layer_1_* (= LSTM hidden size),
  wlstm_num_layers_*, ann_dropout_1_min, ann_learning_rate_min (manual) ...
"""
from models.wavelet_engine import (run, base_settings, wavelet_levels_lags,
                                   int_range, spread)


def _with_resolved(params):
    """If only the raw wlstm_*_resolved keys are present (legacy /predict path), map them."""
    p = dict(params or {})
    mapping = {
        "wann_wavelets": "wlstm_wavelets_resolved",
        "wann_min_level": "wlstm_min_level_resolved", "wann_max_level": "wlstm_max_level_resolved",
        "wann_lags_min": "wlstm_lags_min_resolved", "wann_lags_max": "wlstm_lags_max_resolved",
        "wann_lags_step": "wlstm_lags_step_resolved",
        "ann_hidden_layer_1_min": "wlstm_hs_min_resolved", "ann_hidden_layer_1_max": "wlstm_hs_max_resolved",
        "ann_hidden_layer_1_step": "wlstm_hs_step_resolved",
        "wlstm_num_layers_min": "wlstm_num_layers_min_resolved",
        "wlstm_num_layers_max": "wlstm_num_layers_max_resolved",
        "wlstm_num_layers_step": "wlstm_num_layers_step_resolved",
        "ann_dropout_1_min": "wlstm_dr_min_resolved", "ann_learning_rate_min": "wlstm_lr_min_resolved",
        "ann_weight_decay_min": "wlstm_wd_min_resolved", "ann_huber_delta_min": "wlstm_hd_min_resolved",
        "ann_epochs_min": "wlstm_ep_min_resolved", "ann_epochs_max": "wlstm_ep_max_resolved",
        "ann_patience_min": "wlstm_pa_min_resolved", "ann_patience_max": "wlstm_pa_max_resolved",
    }
    for plain, resolved in mapping.items():
        if resolved in p and p[resolved] is not None:
            p[plain] = p[resolved]
    return p


def run_wavelet_lstm(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = _with_resolved(params)
    auto = bool(params.get("auto_tune_wlstm", True))
    levels, lags = wavelet_levels_lags(params)
    s = base_settings(params, auto, levels, lags)

    hs = int_range(params, "ann_hidden_layer_1_min", "ann_hidden_layer_1_max",
                   "ann_hidden_layer_1_step", 8, 16, 8)
    nl = int_range(params, "wlstm_num_layers_min", "wlstm_num_layers_max",
                   "wlstm_num_layers_step", 1, 2, 1)
    nl = [max(1, min(3, n)) for n in nl]
    s["archs"] = ([(h, l) for h in spread(hs) for l in spread(nl)] if auto
                  else [(max(2, hs[0]), nl[0])])

    return run("wavelet_lstm", "lstm", "Wavelet-LSTM", data, params, s,
               horizon, frequency, mode, log_callback)
