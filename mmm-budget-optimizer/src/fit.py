"""Fit response curves to channel-level spend and revenue history."""
import numpy as np
from scipy.optimize import curve_fit

from response_curves import hill_response


def fit_channel(spend: np.ndarray, revenue: np.ndarray) -> dict:
    """Estimate scale, half-saturation point and shape for one channel.

    Bounds keep the parameters positive. Shape is capped so the curve stays concave-ish.
    """
    spend = np.asarray(spend, dtype=float)
    revenue = np.asarray(revenue, dtype=float)
    p0 = [revenue.max() * 1.5, np.median(spend), 1.0]
    bounds = ([0, 1e-6, 0.3], [np.inf, np.inf, 3.0])
    params, cov = curve_fit(hill_response, spend, revenue, p0=p0, bounds=bounds, maxfev=20000)
    scale, half_sat, shape = params
    residuals = revenue - hill_response(spend, *params)
    r2 = 1 - np.sum(residuals**2) / np.sum((revenue - revenue.mean())**2)
    return {"scale": scale, "half_sat": half_sat, "shape": shape, "r2": r2}
