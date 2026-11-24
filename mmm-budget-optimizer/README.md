# Marketing Budget Optimizer (Diminishing Returns Model)

A Python project that models how revenue responds to spend on each paid channel, then allocates a fixed budget to maximize modeled revenue.

> **Data note:** The included data is synthetic. Known parameters are used to generate it, which lets the fitting code be checked against the truth. No real campaign data is included.

## Why it matters
Spend does not return revenue in a straight line. The first dollars on a channel usually convert better than the last ones. This project fits a saturation curve per channel (Hill function) and finds the split that maximizes total modeled revenue.

## Method
1. **Response curve:** `revenue = scale × spend^shape / (half_sat^shape + spend^shape)`
2. **Fit:** `scipy.optimize.curve_fit` with bounds on each parameter.
3. **Allocate:** `scipy.optimize.minimize` (SLSQP) with a budget equality constraint and optional per-channel min and max spend.

## Structure
```
src/response_curves.py   # Hill curve and marginal response
src/fit.py               # Fit a curve to channel history
src/optimize.py          # Budget allocation
src/simulate.py          # Synthetic history with known parameters
tests/                   # Tests, including parameter recovery
```

## Run
```bash
pip install numpy scipy pytest
pytest tests/
```

## Example
```python
from simulate import simulate_history
from fit import fit_channel
from optimize import optimize_budget

history = simulate_history()
curves = {c: fit_channel(h["spend"], h["revenue"]) for c, h in history.items()}
print(optimize_budget(curves, total_budget=150_000, max_spend={"google": 60_000}))
```

## Limitations
- This is a single-period model. It ignores adstock (lagged effects), seasonality, and interactions between channels. A full media mix model would include those.
- Fitted curves are only reliable inside the spend range the data covers. Extrapolating beyond it is a guess.
