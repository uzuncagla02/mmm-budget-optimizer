"""Diminishing-returns response curves for paid media channels.

Model: revenue(spend) = scale * spend^shape / (half_sat^shape + spend^shape)
This is a Hill-type saturation curve. It rises quickly at low spend and flattens at high spend.
"""
import numpy as np


def hill_response(spend, scale, half_sat, shape):
    spend = np.asarray(spend, dtype=float)
    spend = np.clip(spend, 0, None)
    return scale * spend**shape / (half_sat**shape + spend**shape)


def marginal_response(spend, scale, half_sat, shape, eps=1e-3):
    """Numerical derivative: extra revenue from one more unit of spend."""
    return (hill_response(spend + eps, scale, half_sat, shape)
            - hill_response(spend, scale, half_sat, shape)) / eps
