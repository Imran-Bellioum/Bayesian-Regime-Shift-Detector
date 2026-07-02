"""
changepoint_detector.py
-----------------------

Bayesian Online Changepoint Detection (BOCPD), implemented from scratch
following Adams & MacKay (2007), "Bayesian Online Changepoint Detection".

The detector processes a scalar time series one observation at a time and
maintains a posterior probability distribution over the current run length:
the number of time steps since the most recent changepoint.

Model
-----
Within each regime, observations are assumed to be i.i.d. draws from a
Normal distribution with unknown mean and unknown variance.

The conjugate prior is Normal-Inverse-Gamma (NIG), parameterised here as:

    mean0, confidence0, shape0, scale0

Under this model, the posterior predictive distribution is a Student-t
distribution. This matters because using a plug-in Normal predictive with
a point-estimate variance would underestimate uncertainty and make the
detector overconfident.

Important interpretation note
-----------------------------
With a constant hazard rate, the posterior probability of run length zero
is not the main changepoint signal. It will usually be close to the hazard
rate itself. In practice, changepoints are identified from sharp drops in
the most likely run length, or from high posterior mass on short run lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass(frozen=True)
class NormalInverseGammaPrior:
    """
    Prior parameters for a Normal likelihood with unknown mean and variance.

    Parameters
    ----------
    mean0 : float
        Prior belief about the mean of a fresh regime.
    confidence0 : float
        Prior confidence in mean0. Higher values make the prior mean harder
        to move.
    shape0 : float
        Shape parameter of the Inverse-Gamma prior over variance.
    scale0 : float
        Scale parameter of the Inverse-Gamma prior over variance.
    """

    mean0: float = 0.0
    confidence0: float = 1.0
    shape0: float = 1.0
    scale0: float = 1.0

    def validate(self) -> None:
        """Validate that the prior parameters define a proper NIG prior."""
        if not np.isfinite(self.mean0):
            raise ValueError("mean0 must be finite.")

        if self.confidence0 <= 0 or not np.isfinite(self.confidence0):
            raise ValueError("confidence0 must be positive and finite.")

        if self.shape0 <= 0 or not np.isfinite(self.shape0):
            raise ValueError("shape0 must be positive and finite.")

        if self.scale0 <= 0 or not np.isfinite(self.scale0):
            raise ValueError("scale0 must be positive and finite.")


class BayesianChangepointDetector:
    """
    Bayesian Online Changepoint Detector for scalar Gaussian observations.

    Parameters
    ----------
    expected_run_length : float
        Prior expected regime length. Larger values make the detector more
        conservative. The constant hazard rate is 1 / expected_run_length.
    prior : NormalInverseGammaPrior, optional
        Prior belief about the mean and variance of a fresh regime.
    max_tracked_run_length : int or None, optional
        Optional cap on the largest tracked run length. This improves speed
        on long series, but is an approximation because probability mass
        beyond the cap is discarded and the remaining distribution is
        renormalised.
    """

    def __init__(
        self,
        expected_run_length: float = 250.0,
        prior: NormalInverseGammaPrior | None = None,
        max_tracked_run_length: int | None = None,
    ) -> None:
        if expected_run_length <= 1.0 or not np.isfinite(expected_run_length):
            raise ValueError("expected_run_length must be greater than 1 and finite.")

        if max_tracked_run_length is not None:
            if not isinstance(max_tracked_run_length, int) or max_tracked_run_length < 1:
                raise ValueError(
                    "max_tracked_run_length must be a positive integer or None."
                )

        self.expected_run_length = float(expected_run_length)
        self.prior = prior or NormalInverseGammaPrior()
        self.prior.validate()
        self.max_tracked_run_length = max_tracked_run_length

        self.reset()

    @property
    def hazard_rate(self) -> float:
        """Prior per-step probability of a changepoint."""
        return 1.0 / self.expected_run_length

    def reset(self) -> None:
        """
        Reset the detector state.

        Call this before running the detector on a fresh independent series.
        The run() method does this by default.
        """
        self.run_mean = np.array([self.prior.mean0], dtype=float)
        self.run_confidence = np.array([self.prior.confidence0], dtype=float)
        self.run_shape = np.array([self.prior.shape0], dtype=float)
        self.run_scale = np.array([self.prior.scale0], dtype=float)

        self.run_length_probs = np.array([1.0], dtype=float)
        self.time_step = 0
        self.run_length_history: list[np.ndarray] = [self.run_length_probs.copy()]

    def _predictive_density(self, x: float) -> np.ndarray:
        """
        Compute the Student-t posterior predictive density of x for every
        currently tracked run length.
        """
        degrees_of_freedom = 2.0 * self.run_shape

        predictive_scale = np.sqrt(
            self.run_scale
            * (self.run_confidence + 1.0)
            / (self.run_shape * self.run_confidence)
        )

        densities = stats.t.pdf(
            x,
            df=degrees_of_freedom,
            loc=self.run_mean,
            scale=predictive_scale,
        )

        # Avoid exact zeros from floating-point underflow.
        return np.maximum(densities, np.finfo(float).tiny)

    def update(self, x: float) -> np.ndarray:
        """
        Ingest one new observation and return the updated run-length posterior.

        Returns
        -------
        np.ndarray
            Probability distribution over run length 0, 1, 2, ...
        """
        if not np.isscalar(x) or not np.isfinite(float(x)):
            raise ValueError("x must be a finite scalar.")

        x = float(x)

        predictive_probs = self._predictive_density(x)

        # Existing regimes continue and their run lengths grow by one.
        growth_probs = (
            self.run_length_probs
            * predictive_probs
            * (1.0 - self.hazard_rate)
        )

        # A new regime begins, so run length resets to zero.
        reset_prob = np.sum(
            self.run_length_probs
            * predictive_probs
            * self.hazard_rate
        )

        new_run_length_probs = np.empty(len(growth_probs) + 1, dtype=float)
        new_run_length_probs[0] = reset_prob
        new_run_length_probs[1:] = growth_probs

        total_prob = new_run_length_probs.sum()

        if total_prob <= 0 or not np.isfinite(total_prob):
            new_run_length_probs[:] = 0.0
            new_run_length_probs[0] = 1.0
        else:
            new_run_length_probs /= total_prob

        # NIG posterior update for every currently tracked run length.
        updated_mean = (
            (self.run_confidence * self.run_mean + x)
            / (self.run_confidence + 1.0)
        )

        updated_confidence = self.run_confidence + 1.0
        updated_shape = self.run_shape + 0.5

        updated_scale = self.run_scale + (
            self.run_confidence * (x - self.run_mean) ** 2
        ) / (2.0 * (self.run_confidence + 1.0))

        # Prepend fresh-prior parameters for the new run length 0.
        self.run_mean = np.concatenate(([self.prior.mean0], updated_mean))
        self.run_confidence = np.concatenate(
            ([self.prior.confidence0], updated_confidence)
        )
        self.run_shape = np.concatenate(([self.prior.shape0], updated_shape))
        self.run_scale = np.concatenate(([self.prior.scale0], updated_scale))

        self.run_length_probs = new_run_length_probs

        # Optional run-length cap for speed.
        if (
            self.max_tracked_run_length is not None
            and len(self.run_length_probs) > self.max_tracked_run_length + 1
        ):
            keep = self.max_tracked_run_length + 1

            self.run_length_probs = self.run_length_probs[:keep]
            self.run_mean = self.run_mean[:keep]
            self.run_confidence = self.run_confidence[:keep]
            self.run_shape = self.run_shape[:keep]
            self.run_scale = self.run_scale[:keep]

            total_prob = self.run_length_probs.sum()

            if total_prob <= 0 or not np.isfinite(total_prob):
                self.run_length_probs[:] = 0.0
                self.run_length_probs[0] = 1.0
            else:
                self.run_length_probs /= total_prob

        self.time_step += 1
        self.run_length_history.append(self.run_length_probs.copy())

        return self.run_length_probs.copy()

    def run(
        self,
        series: ArrayLike,
        reset: bool = True,
        short_run_window: int = 10,
    ) -> dict[str, np.ndarray | list[np.ndarray]]:
        """
        Run the detector over a full series.

        Parameters
        ----------
        series : array-like
            One-dimensional numeric time series.
        reset : bool
            If True, reset the detector before processing the series.
        short_run_window : int
            Used to compute the posterior probability that the current run
            length is short. For example, short_run_window=10 stores the
            probability that the run length is between 0 and 9.

        Returns
        -------
        dict
            Dictionary containing:
              - reset_probability
              - short_run_probability
              - most_likely_run_length
              - run_length_history
        """
        series = np.asarray(series, dtype=float)

        if series.ndim != 1:
            raise ValueError("series must be a one-dimensional array.")

        if not np.all(np.isfinite(series)):
            raise ValueError("series must only contain finite values.")

        if short_run_window < 1:
            raise ValueError("short_run_window must be at least 1.")

        if reset:
            self.reset()

        reset_probability = np.zeros(len(series), dtype=float)
        short_run_probability = np.zeros(len(series), dtype=float)
        most_likely_run_length = np.zeros(len(series), dtype=int)

        for i, x in enumerate(series):
            probs = self.update(float(x))

            reset_probability[i] = probs[0]
            most_likely_run_length[i] = int(np.argmax(probs))

            window = min(short_run_window, len(probs))
            short_run_probability[i] = probs[:window].sum()

        return {
            "reset_probability": reset_probability,
            "short_run_probability": short_run_probability,
            "most_likely_run_length": most_likely_run_length,
            "run_length_history": self.run_length_history,
        }

    @staticmethod
    def find_changepoints(
        most_likely_run_length: ArrayLike,
        drop_threshold: int = 5,
        min_distance: int = 1,
    ) -> np.ndarray:
        """
        Convert a MAP run-length trace into discrete changepoint indices.

        A changepoint is flagged when the most likely run length drops sharply.
        For example, if the MAP run length falls from 80 to 2, the detector
        has effectively reset its belief about the current regime.

        Parameters
        ----------
        most_likely_run_length : array-like
            MAP run length at each time step.
        drop_threshold : int
            Minimum fall in MAP run length required to flag a changepoint.
        min_distance : int
            Minimum spacing between returned changepoints.

        Returns
        -------
        np.ndarray
            Array of changepoint indices.
        """
        if drop_threshold < 1:
            raise ValueError("drop_threshold must be at least 1.")

        if min_distance < 1:
            raise ValueError("min_distance must be at least 1.")

        run_lengths = np.asarray(most_likely_run_length, dtype=int)

        if run_lengths.ndim != 1:
            raise ValueError("most_likely_run_length must be one-dimensional.")

        candidate_points = np.where(np.diff(run_lengths) <= -drop_threshold)[0] + 1

        if len(candidate_points) == 0 or min_distance == 1:
            return candidate_points

        kept_points = [candidate_points[0]]

        for point in candidate_points[1:]:
            if point - kept_points[-1] >= min_distance:
                kept_points.append(point)

        return np.array(kept_points, dtype=int)


if __name__ == "__main__":
    # Minimal synthetic example.
    rng = np.random.default_rng(42)

    series = np.concatenate(
        [
            rng.normal(loc=0.0, scale=1.0, size=150),
            rng.normal(loc=2.5, scale=1.0, size=150),
        ]
    )

    detector = BayesianChangepointDetector(
        expected_run_length=100,
        max_tracked_run_length=250,
    )

    results = detector.run(series)

    changepoints = BayesianChangepointDetector.find_changepoints(
        results["most_likely_run_length"],
        drop_threshold=10,
        min_distance=20,
    )

    print("Detected changepoints:", changepoints)
