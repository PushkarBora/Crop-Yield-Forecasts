"""
Fast hyper-parameter search helpers shared by the ML and DL models.

Why the models were slow
------------------------
* Deep-learning models trained EVERY combination of the form ranges
  (e.g. 3 hidden × 3 layers × 4 dropout × 5 lr × 5 wd × 3 huber × 4 patience × 3 epochs
  = 32,400 trainings per lag value).
* Machine-learning models ran a 2–5-fold TimeSeriesSplit for every combination,
  but the CV score was only printed – the choice was made on the hold-out score.

What is done now (the user's ranges are still the search space)
---------------------------------------------------------------
* "epochs" and "patience" are upper limits for early stopping, so only their
  largest value is used (a longer budget with early stopping never loses the
  best epoch).
* If the remaining grid is larger than the budget, a fixed, reproducible
  sample of it is tested (random search – as good as a full grid in practice),
  always including the middle-of-range combination.
* Tiny networks train fastest on one CPU thread.
"""
import itertools
import math
import random

UPPER_LIMIT_KEYS = ("epochs", "patience")
DEFAULT_DL_BUDGET = 10          # trainings per lag value for deep-learning models
DEFAULT_ML_BUDGET = 40          # fits per lag value for machine-learning models


def use_single_thread():
    """Small models are much faster on one thread (avoids thread start-up cost)."""
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _middle(values):
    return values[(len(values) - 1) // 2]


def sample_grid(grid, budget, seed=42, upper_keys=UPPER_LIMIT_KEYS):
    """
    grid  : {name: [values, ...]}
    returns (list of dicts to try, total size of the original grid)
    """
    grid = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in grid.items()}
    grid = {k: (v if v else [None]) for k, v in grid.items()}
    full_total = math.prod(len(v) for v in grid.values())

    reduced = {}
    for k, v in grid.items():
        if k in upper_keys:
            nums = [x for x in v if isinstance(x, (int, float))]
            reduced[k] = [max(nums)] if nums else v[:1]
        else:
            reduced[k] = v
    keys = list(reduced)
    sizes = [len(reduced[k]) for k in keys]
    total = math.prod(sizes)

    if total <= budget:
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(reduced[k] for k in keys))]
        return combos, full_total

    rng = random.Random(seed)
    chosen = []
    seen = set()
    centre = tuple((len(reduced[k]) - 1) // 2 for k in keys)
    seen.add(centre)
    chosen.append(centre)
    while len(chosen) < budget:
        idx = tuple(rng.randrange(n) for n in sizes)
        if idx not in seen:
            seen.add(idx)
            chosen.append(idx)
    combos = [{k: reduced[k][i] for k, i in zip(keys, idx)} for idx in chosen]
    return combos, full_total


def dl_combos(grid, log=None, budget=DEFAULT_DL_BUDGET):
    """Combinations for a deep-learning auto-tune loop."""
    use_single_thread()
    combos, full_total = sample_grid(grid, budget)
    if log:
        if len(combos) < full_total:
            log(f"⚡ Fast search: testing {len(combos)} of {full_total} combinations "
                f"(epochs/patience = upper limits with early stopping)")
    return combos


def ml_combos(value_lists, log=None, budget=DEFAULT_ML_BUDGET):
    """
    value_lists : list of value lists in the same order the model unpacks them.
    Returns a list of tuples (like itertools.product would).
    """
    grid = {f"p{i}": list(v) for i, v in enumerate(value_lists)}
    combos, full_total = sample_grid(grid, budget, upper_keys=())
    out = [tuple(c[f"p{i}"] for i in range(len(value_lists))) for c in combos]
    if log and len(out) < full_total:
        log(f"⚡ Fast search: testing {len(out)} of {full_total} combinations per lag")
    return out
