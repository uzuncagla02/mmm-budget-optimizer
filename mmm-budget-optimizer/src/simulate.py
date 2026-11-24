"""Generate synthetic weekly spend and revenue with known saturation parameters.

Because the true parameters are known, the fitting code can be checked against them.
"""
import numpy as np

TRUE_PARAMS = {
    "meta":   {"scale": 180_000, "half_sat": 40_000, "shape": 1.1},
    "tiktok": {"scale": 110_000, "half_sat": 25_000, "shape": 0.9},
    "google": {"scale": 220_000, "half_sat": 55_000, "shape": 1.3},
}


def simulate_history(weeks: int = 104, noise_sd: float = 0.05, seed: int = 7) -> dict:
    from response_curves import hill_response

    rng = np.random.default_rng(seed)
    history = {}
    for channel, p in TRUE_PARAMS.items():
        spend = rng.uniform(5_000, 90_000, size=weeks)
        clean = hill_response(spend, p["scale"], p["half_sat"], p["shape"])
        revenue = clean * rng.normal(1, noise_sd, size=weeks)
        history[channel] = {"spend": spend, "revenue": revenue}
    return history
