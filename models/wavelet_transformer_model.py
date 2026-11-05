"""
Wavelet-Transformer  (fast version)
===================================
Same function name, inputs and outputs as before (app.py unchanged).
Engine: models/wavelet_engine.py — one small Transformer encoder reads all
wavelet components (+ exogenous variables) as a multichannel sequence.

Form settings used (unchanged; app.py maps wtransformer_*_resolved into these):
  auto_tune_wtransformer, wann_wavelets, wann_min_level / wann_max_level,
  wann_lags_*, ann_hidden_layer_1_* (= d_model), wtransformer_nhead_*,
  wtransformer_num_layers_*, ann_dropout_1_min, ann_learning_rate_min (manual) ...
"""
from models.wavelet_engine import (run, base_settings, wavelet_levels_lags,
                                   int_range, spread)


def _with_resolved(params):
    p = dict(params or {})
    mapping = {
        "wann_wavelets": "wtransformer_wavelets_resolved",
        "wann_min_level": "wtransformer_min_level_resolved", "wann_max_level": "wtransformer_max_level_resolved",
        "wann_lags_min": "wtransformer_lags_min_resolved", "wann_lags_max": "wtransformer_lags_max_resolved",
        "wann_lags_step": "wtransformer_lags_step_resolved",
        "ann_hidden_layer_1_min": "wtransformer_dm_min_resolved",
        "ann_hidden_layer_1_max": "wtransformer_dm_max_resolved",
        "ann_hidden_layer_1_step": "wtransformer_dm_step_resolved",
        "wtransformer_nhead_min": "wtransformer_nhead_min_resolved",
        "wtransformer_nhead_max": "wtransformer_nhead_max_resolved",
        "wtransformer_nhead_step": "wtransformer_nhead_step_resolved",
        "wtransformer_num_layers_min": "wtransformer_num_layers_min_resolved",
        "wtransformer_num_layers_max": "wtransformer_num_layers_max_resolved",
        "wtransformer_num_layers_step": "wtransformer_num_layers_step_resolved",
        "ann_dropout_1_min": "wtransformer_dr_min_resolved", "ann_learning_rate_min": "wtransformer_lr_min_resolved",
        "ann_weight_decay_min": "wtransformer_wd_min_resolved", "ann_huber_delta_min": "wtransformer_hd_min_resolved",
        "ann_epochs_min": "wtransformer_ep_min_resolved", "ann_epochs_max": "wtransformer_ep_max_resolved",
        "ann_patience_min": "wtransformer_pa_min_resolved", "ann_patience_max": "wtransformer_pa_max_resolved",
    }
    for plain, resolved in mapping.items():
        if resolved in p and p[resolved] is not None:
            p[plain] = p[resolved]
    return p


def run_wavelet_transformer(data, params, horizon, frequency, mode="train", log_callback=None, **_):
    params = _with_resolved(params)
    auto = bool(params.get("auto_tune_wtransformer", True))
    levels, lags = wavelet_levels_lags(params)
    s = base_settings(params, auto, levels, lags)

    dm = int_range(params, "ann_hidden_layer_1_min", "ann_hidden_layer_1_max",
                   "ann_hidden_layer_1_step", 8, 16, 8)
    nh = int_range(params, "wtransformer_nhead_min", "wtransformer_nhead_max",
                   "wtransformer_nhead_step", 1, 2, 1)
    nl = int_range(params, "wtransformer_num_layers_min", "wtransformer_num_layers_max",
                   "wtransformer_num_layers_step", 1, 2, 1)

    def fix(d, h, l):
        h = max(1, min(8, h))
        d = max(h, d)
        d = d + (-d % h)                # d_model must be divisible by nhead
        return (d, h, max(1, min(3, l)))

    if auto:
        archs = []
        for d in spread(dm):
            for h in spread(nh):
                for l in spread(nl, 1):
                    a = fix(d, h, l)
                    if a not in archs:
                        archs.append(a)
        s["archs"] = archs[:4]
    else:
        s["archs"] = [fix(dm[0], nh[0], nl[0])]

    return run("wavelet_transformer", "transformer", "Wavelet-Transformer", data, params, s,
               horizon, frequency, mode, log_callback)
