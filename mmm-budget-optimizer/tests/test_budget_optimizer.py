import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, ROOT)

from fit import fit_channel  # noqa: E402
from optimize import optimize_budget  # noqa: E402
from response_curves import hill_response  # noqa: E402
from simulate import TRUE_PARAMS, simulate_history  # noqa: E402


def test_fit_recovers_scale_roughly():
    h = simulate_history()
    for channel, p in TRUE_PARAMS.items():
        est = fit_channel(h[channel]["spend"], h[channel]["revenue"])
        assert est["r2"] > 0.9
        assert abs(est["scale"] / p["scale"] - 1) < 0.25


def test_budget_is_fully_allocated():
    curves = {c: TRUE_PARAMS[c] for c in TRUE_PARAMS}
    alloc = optimize_budget(curves, total_budget=150_000)
    assert abs(sum(alloc.values()) - 150_000) < 1.0


def test_optimum_beats_equal_split():
    curves = {c: TRUE_PARAMS[c] for c in TRUE_PARAMS}
    budget = 150_000
    alloc = optimize_budget(curves, total_budget=budget)

    def revenue(a):
        return sum(float(hill_response(a[c], **{k: v for k, v in curves[c].items()})) for c in curves)

    equal = {c: budget / len(curves) for c in curves}
    assert revenue(alloc) >= revenue(equal) - 1e-6


def test_respects_max_spend():
    curves = {c: TRUE_PARAMS[c] for c in TRUE_PARAMS}
    alloc = optimize_budget(curves, total_budget=150_000, max_spend={"google": 40_000})
    assert alloc["google"] <= 40_000 + 1e-6
