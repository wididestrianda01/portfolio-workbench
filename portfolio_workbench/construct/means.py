"""The mean-input axis: four answers to how far the estimated mean can be trusted, behind one
interface.

Every input returns the same thing (one mean vector over the sleeves, plus the diagnostics that say
how it was formed) and the constructor that consumes it never learns which input produced it. That
is what makes the axis an input choice rather than a code path, and it is why the four cells can be
compared with the optimiser held fixed.

**Shrinking estimates its own intensity.** The Bayes-Stein estimator pulls the sample mean toward the
minimum-variance portfolio's implied mean, and the intensity is not a parameter of this module: it is
estimated from the window, and it is returned so it can be printed per window. A reader expecting a
fixed shrinkage fraction will find a small one, and the small value is the window's own statement
that its sample mean, measured against its own dispersion, is not far enough from the target to be
worth shrinking hard. Nothing is chosen here, so nothing here can be tuned to flatter the cell.

**The view is stated, and so is its uncertainty.** Black-Litterman needs a prior, a view and the
view's uncertainty, and a cell that leaves any of the three implicit is reporting a number whose
content is unknown. The prior is the returns implied by the mandate's own policy holdings (the
reverse optimisation that defines the equilibrium in the method's own construction) at the same
risk-aversion weight mean-variance uses. The view is the window's sample mean, one view per sleeve.
The uncertainty is the sampling variance of that estimate, one over the observation count times the
sleeve's own variance, which states the belief in the sample mean only as far as its own standard
error reaches, rather than naming a chosen confidence.

**What the tau-over-omega ratio does and does not move.** With the prior's scaling set to one over
the observation count and the view's uncertainty to the same factor, the observation count cancels
and the posterior depends only on the ratio between how much the prior is trusted and how much the
view is. So a rerun at a different sample size is the same answer, and the sensitivity that matters
is the ratio, which is why the cell reports the base case and the ratio run beside it instead of a
single number whose provenance a reader cannot reconstruct.

**All four are sample-based, and the sample is the window.** Every input is estimated from the same
trailing window the covariance comes from, so the axis carries no look-ahead of its own; a mean taken
from somewhere else would move the estimation boundary in one cell and not the others.
"""

import numpy as np
import pandas as pd

from ..data import universe
from .families import RISK_AVERSION

# The prior's scaling, in units of one over the observation count. Kept as a named multiple rather
# than folded into the arithmetic so the ratio the sensitivity run varies is visible in the code.
TAU_SCALE = 1.0
# The view's uncertainty, in the same units, times the sleeve's own variance.
OMEGA_SCALE = 1.0
# The ratio run reported beside the base case, per the cell's declaration.
TAU_SENSITIVITY = (0.1, 1.0, 10.0)
# The posterior is computed from an inverse, and a singular prior covariance would silently produce a
# pseudo-inverse rather than an error.
CONDITION_LIMIT = 1e12


def _frame(returns, covariance):
    """The two inputs checked against each other: same sleeves, same order, no gap, invertible.

    A mean vector and a covariance that disagree about the sleeve order produce a posterior that is
    arithmetically sound and describes a different portfolio, and nothing downstream can tell. The
    check lives here rather than in each input because all four read the same two objects.
    """
    if not isinstance(covariance, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError("the mean inputs read labelled frames; a bare array carries no sleeve names")
    if list(returns.columns) != list(covariance.columns):
        raise ValueError("the return frame and the covariance disagree about the sleeves or their order")
    window = np.asarray(returns, dtype=float)
    if window.shape[0] < 2:
        raise ValueError("a mean needs at least two observations in its window")
    if np.isnan(window).any():
        raise ValueError("the window carries a missing value; a filled one would enter every mean built on it")
    return window, np.asarray(covariance, dtype=float)


def sample(returns, covariance, setting=None):
    """The window's own arithmetic mean. The input every other one is a modification of, and the one
    whose error mean-variance is accused of maximising."""
    window, _ = _frame(returns, covariance)
    vector = window.mean(axis=0)
    return pd.Series(vector, index=returns.columns, name="mean"), {"kind": "sample"}


def jorion(returns, covariance, setting=None):
    """The Bayes-Stein shrinkage of the sample mean toward the minimum-variance implied mean.

    The target is the mean the minimum-variance portfolio implies, `(1' S^-1 mu) / (1' S^-1 1)`,
    repeated across sleeves. The intensity is the estimated quantity above; it is returned rather
    than applied and forgotten, because a cell whose intensity runs near zero is reporting that the
    window's own dispersion test found little to shrink.
    """
    window, matrix = _frame(returns, covariance)
    observations, size = window.shape
    vector = window.mean(axis=0)
    inverse = np.linalg.inv(matrix)
    ones = np.ones(size)
    grand = float(ones @ inverse @ vector) / float(ones @ inverse @ ones)
    offset = vector - grand
    quadratic = float(offset @ inverse @ offset)
    intensity = (size + 2.0) / ((size + 2.0) + observations * quadratic)
    shrunk = (1.0 - intensity) * vector + intensity * grand
    return pd.Series(shrunk, index=returns.columns, name="mean"), {
        "kind": "jorion",
        "intensity": float(intensity),
        "target": float(grand),
        "unshrunk": pd.Series(vector, index=returns.columns),
    }


def posterior(covariance, prior, tau, views, view_returns, view_covariance):
    """The Black-Litterman posterior mean, in its precision form.

    `[(tau S)^-1 + P' W^-1 P]^-1 [(tau S)^-1 pi + P' W^-1 q]`, with the prior mean `pi`, the views
    `P`, their returns `q` and their covariance `W`. Written in this form rather than in the
    covariance form because it needs one solve instead of one inverse plus one solve, and the two
    forms are algebraically identical - the test suite checks this implementation against the
    covariance form, so the equivalence is verified on numbers rather than asserted here.
    """
    covariance = np.asarray(covariance, dtype=float)
    prior = np.asarray(prior, dtype=float).ravel()
    views = np.atleast_2d(np.asarray(views, dtype=float))
    view_returns = np.asarray(view_returns, dtype=float).ravel()
    view_covariance = np.asarray(view_covariance, dtype=float)
    if views.shape[1] != covariance.shape[0] or prior.shape[0] != covariance.shape[0]:
        raise ValueError("the prior and the views do not describe the covariance's sleeves")
    if view_returns.shape[0] != views.shape[0] or view_covariance.shape != (views.shape[0], views.shape[0]):
        raise ValueError("the view returns and their covariance do not match the view matrix")
    if float(np.linalg.cond(covariance)) > CONDITION_LIMIT:
        raise ValueError(
            f"the prior covariance has condition number {np.linalg.cond(covariance):.3e}; the posterior "
            f"would be computed from a matrix that is numerically rank deficient"
        )
    prior_precision = np.linalg.inv(tau * covariance)
    inverse_view_covariance = np.linalg.inv(view_covariance)
    view_precision = views.T @ inverse_view_covariance @ views
    return np.linalg.solve(
        prior_precision + view_precision,
        prior_precision @ prior + views.T @ inverse_view_covariance @ view_returns,
    )


def black_litterman(returns, covariance, setting=TAU_SCALE):
    """The posterior mean under the equilibrium prior and a sample-mean view.

    `setting` scales the prior's precision against the view's, in units of one over the observation
    count; the base case is one, and the cell's sensitivity run moves it by an order of magnitude
    either way. The view matrix is the identity - one view per sleeve - so the posterior is a
    precision-weighted blend of the mandate's own implied returns and the window's sample mean.
    """
    window, matrix = _frame(returns, covariance)
    observations, size = window.shape
    weights = np.array([universe.POLICY_WEIGHTS[name] for name in returns.columns], dtype=float)
    prior = RISK_AVERSION * (matrix @ weights)
    tau = float(setting) / observations
    view_covariance = OMEGA_SCALE * np.diag(np.diag(matrix)) / observations
    vector = window.mean(axis=0)
    blended = posterior(matrix, prior, tau, np.eye(size), vector, view_covariance)
    return pd.Series(blended, index=returns.columns, name="mean"), {
        "kind": "black_litterman",
        "tau": tau,
        "prior": pd.Series(prior, index=returns.columns),
        "view": pd.Series(vector, index=returns.columns),
        "view_volatility": np.sqrt(np.diag(view_covariance)),
        "setting": float(setting),
    }


def none(returns, covariance, setting=None):
    """No mean at all: the zero vector.

    It is not a neutral input. The mean-variance objective with a zero mean is pure variance under
    the constraint set, so the cell that consumes it is the family's no-mean form and its answer is
    decided by the constraints - which is the thing that cell exists to show.
    """
    window, _ = _frame(returns, covariance)
    return pd.Series(np.zeros(window.shape[1]), index=returns.columns, name="mean"), {"kind": "none"}


# The inputs behind one name each. The keys are what the cell registry refers to; an input missing
# from here cannot be run, which is what keeps a cell and its mean input from drifting apart.
MEANS = {
    "sample": sample,
    "jorion": jorion,
    "black_litterman": black_litterman,
    "none": none,
}
