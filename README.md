# Bayesian-Regime-Shift-Detector

A from-scratch implementation of **Bayesian Online Changepoint Detection** for identifying regime shifts in noisy time-series data.

This project implements an online Bayesian filtering algorithm inspired by Adams and MacKay’s 2007 paper, *Bayesian Online Changepoint Detection*. The detector processes data sequentially and maintains a posterior distribution over the current **run length**, meaning the number of observations since the most recent changepoint.

The goal is not to predict prices or generate trading signals directly. Instead, the project focuses on a more general quantitative problem: detecting when a noisy time series may have shifted into a new statistical regime.

## Motivation

In quantitative finance and time-series modelling, observed performance can change for two very different reasons.

Sometimes a strategy, model, or market variable is simply fluctuating within its normal range. Other times, the underlying data-generating process has genuinely changed. For example:

* a trading strategy’s returns may deteriorate after a market regime shift;
* volatility may move into a new regime;
* execution costs may change after liquidity conditions worsen;
* a spread relationship may break down;
* a model’s live performance may drift away from its backtested behaviour.

Naive rolling-window methods can be unstable. A short rolling window reacts quickly but overfits to noise, while a long rolling window is smoother but reacts slowly. Bayesian Online Changepoint Detection provides a probabilistic alternative by tracking uncertainty over possible regime lengths.

## Model

Within each regime, observations are assumed to be independent draws from a Gaussian distribution with unknown mean and unknown variance:

```text
x_t ~ Normal(mu, sigma^2)
```

Both the mean and variance are treated as unknown. The model uses a **Normal-Inverse-Gamma** prior, parameterised by:

```text
mean0, confidence0, shape0, scale0
```

This prior is conjugate to the Gaussian likelihood with unknown mean and variance. As a result, the posterior predictive distribution for each possible run length is a **Student-t distribution**.

This is important because a Student-t predictive distribution accounts for uncertainty in both the mean and variance. A simpler plug-in approach that estimates the variance and then uses a Normal predictive distribution would underestimate uncertainty, making the detector too confident.

## How the algorithm works

At each new observation, the detector considers every currently possible run length.

For each possible run length, it asks:

```text
How likely is the new observation if the current regime has lasted this long?
```

The algorithm then splits probability mass into two cases:

1. **Growth case**
   The current regime continues, so the run length increases by one.

2. **Reset case**
   A changepoint occurs, so the run length resets to zero.

The prior probability of a changepoint is controlled by a constant hazard rate:

```text
hazard = 1 / expected_run_length
```

A larger `expected_run_length` makes the detector more conservative. A smaller value makes the detector more sensitive to possible changepoints.

## Important interpretation note

With a constant hazard rate, the posterior probability of run length zero is not the main detection signal. It will usually stay close to the hazard rate itself.

The more useful signal is the **most likely run length**.

For example, if the most likely run length has been increasing steadily and then suddenly falls from 100 to 3, the detector is effectively saying that the old regime no longer explains the data well and that a new regime may have started recently.

The implementation therefore includes:

```python
most_likely_run_length
```

and a helper function:

```python
find_changepoints(...)
```

which flags changepoints when the most likely run length drops sharply.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bayesian-regime-shift-detector.git
cd bayesian-regime-shift-detector
```

Install dependencies:

```bash
pip install numpy scipy
```

## Example usage

```python
import numpy as np

from changepoint_detector import (
    BayesianChangepointDetector,
    NormalInverseGammaPrior,
)

rng = np.random.default_rng(42)

# Synthetic series with a mean shift halfway through.
series = np.concatenate(
    [
        rng.normal(loc=0.0, scale=1.0, size=150),
        rng.normal(loc=2.5, scale=1.0, size=150),
    ]
)

prior = NormalInverseGammaPrior(
    mean0=0.0,
    confidence0=1.0,
    shape0=1.0,
    scale0=1.0,
)

detector = BayesianChangepointDetector(
    expected_run_length=100,
    prior=prior,
    max_tracked_run_length=250,
)

results = detector.run(series)

changepoints = BayesianChangepointDetector.find_changepoints(
    results["most_likely_run_length"],
    drop_threshold=10,
    min_distance=20,
)

print(changepoints)
```

## Output

The detector returns a dictionary containing:

```python
{
    "reset_probability": reset_probability,
    "short_run_probability": short_run_probability,
    "most_likely_run_length": most_likely_run_length,
    "run_length_history": run_length_history,
}
```

### `reset_probability`

The posterior probability assigned to run length zero.

With a constant hazard rate, this is not usually the best changepoint signal on its own. It is included for completeness, but the main practical signal is the movement of the run-length distribution.

### `short_run_probability`

The posterior probability that the current run length is short.

This is useful because, after a suspected changepoint, probability mass often shifts toward shorter run lengths.

### `most_likely_run_length`

The maximum a posteriori estimate of the current run length at each time step.

Sharp drops in this value suggest that the detector has reset its belief about the current regime.

### `run_length_history`

A stored history of the full run-length posterior distribution over time.

This can be used to build a heatmap visualisation of the detector’s belief over possible run lengths.

## Detecting changepoints

The helper method:

```python
BayesianChangepointDetector.find_changepoints(
    most_likely_run_length,
    drop_threshold=10,
    min_distance=20,
)
```

returns indices where the most likely run length drops sharply.

The `drop_threshold` parameter controls how large the fall must be before a changepoint is flagged.

The `min_distance` parameter prevents multiple nearby detections from being returned for the same regime shift.

## Why this project is relevant

This project demonstrates:

* Bayesian inference;
* online time-series modelling;
* conjugate priors;
* probabilistic filtering;
* Student-t posterior predictive distributions;
* regime shift detection;
* careful interpretation of noisy signals.

These are relevant skills for quantitative research, trading, risk modelling, and systematic strategy monitoring.

## Potential finance applications

This detector could be adapted to monitor:

* rolling strategy returns;
* realised volatility;
* rolling Sharpe ratios;
* spread behaviour;
* execution slippage;
* market-making fill quality;
* signal decay;
* live model performance versus backtest expectations.

The project is intentionally framed as a regime-monitoring tool rather than a standalone trading strategy.

## Limitations

This is a minimal educational implementation.

Current limitations:

* assumes scalar observations;
* assumes observations are independent within each regime;
* assumes a Gaussian regime model with unknown mean and variance;
* uses a constant hazard rate;
* does not model autocorrelation;
* does not include transaction costs, live data, or trading execution;
* does not include a visual dashboard yet.

## Possible extensions

Future improvements could include:

* log-space probability updates for better numerical stability;
* posterior heatmap visualisation;
* adaptive hazard functions;
* multivariate observations;
* Student-t or heavy-tailed observation models;
* application to real financial return series;
* comparison against rolling-window baselines;
* unit tests and performance benchmarks.

## Reference

Adams, R. P. and MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*.

arXiv: https://arxiv.org/abs/0710.3742
