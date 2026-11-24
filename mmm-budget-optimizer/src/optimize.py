"""Allocate a fixed total budget across channels to maximize modeled revenue."""
import numpy as np
from scipy.optimize import minimize

from response_curves import hill_response


def optimize_budget(curves: dict, total_budget: float, min_spend: dict | None = None,
                    max_spend: dict | None = None) -> dict:
    """curves: {channel: {"scale":..., "half_sat":..., "shape":...}}

    Returns {channel: optimal_spend} whose sum equals total_budget (within tolerance).
    """
    names = list(curves)
    n = len(names)
    min_spend = min_spend or {}
    max_spend = max_spend or {}
    lower = [min_spend.get(c, 0.0) for c in names]
    upper = [max_spend.get(c, total_budget) for c in names]

    def neg_revenue(x):
        total = 0.0
        for i, c in enumerate(names):
            p = curves[c]
            total += float(hill_response(x[i], p["scale"], p["half_sat"], p["shape"]))
        return -total

    x0 = np.full(n, total_budget / n)
    x0 = np.clip(x0, lower, upper)
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - total_budget}]
    result = minimize(neg_revenue, x0, method="SLSQP",
                      bounds=list(zip(lower, upper)), constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return {c: float(result.x[i]) for i, c in enumerate(names)}
