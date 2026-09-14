"""The constructor families, behind one interface.

Each family takes a window's covariance, a window's mean vector and the window's returns, and
returns one labelled target weight vector under the package's constraint set. Every family is called
with the same three inputs - a family that ignores the mean ignores it rather than being called
differently - because the comparison varies one axis at a time and a second calling convention would
be a second axis.

Five things in here are decisions rather than mechanics.

**The risk-aversion weight is a stated convention, not a preference estimate.** Mean-variance needs
the coefficient trading expected return against variance, and practitioners are documented as unable
to state it: the reading behind this file records Northfield's note that most investors cannot
numerically express their mean-variance trade-off parameter, and its rule of thumb that the
parameter in the `mean - lambda * variance` form is "typically about one sixth". This file writes
the objective as `mu'w - (gamma/2) w'Sw`, where `gamma = 2 * lambda`, so the rule of thumb is the
constant below. It is held fixed across every mean-variance cell, so the mean input is the only
thing that moves between them.

**The mean arrives already formed.** Shrinking and viewing are the mean axis and live in their own
module; the constructor receives a vector. That is what makes the axis an input choice rather than a
code path, and it is why no family here knows which input produced its mean.

**ERC is one objective, run twice.** The bounded form is the same problem with the package's own
per-sleeve cap as the bound, so the difference between the two cells is the constraint and not the
solver. Both minimise the same sum of squared deviations of the sleeve risk contributions from their
common target, which is the definition of the portfolio rather than a proxy for it: a solution whose
contributions are unequal has not solved the problem, and the spread is reported so a caller can see
how far from it the answer sits.

**Tail construction is a linear program against the window's own tail.** The Rockafellar-Uryasev
formulation minimises a level plus the mean excess beyond it, which makes expected shortfall linear
in the weights, so the cell is solved exactly rather than by iterating on a quantile. Over a
60-month window at the five percent level the tail is three observations, so the cell measures the
tails of its own window rather than a tail property of the market; that is reported rather than
smoothed over.

**A rule-based family is projected onto the constraint set, and the projection is reported.**
Equal weight and the policy portfolio sit inside the cap by construction. Hierarchical risk parity
is a bisection with no bounds to set, so it is projected, and the projection changes it whenever a
cap binds - which the report prints, because a cell that stops being the method it is named after
is a result about the constraint set.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from ..data import universe
from . import constraints

# Northfield's rule of thumb for the trade-off parameter, in the `mean - lambda * variance` form,
# doubled into the `mu'w - (gamma/2) w'Sw` form this file writes.
MEAN_VARIANCE_TRADEOFF = 1.0 / 6.0
RISK_AVERSION = 2.0 * MEAN_VARIANCE_TRADEOFF
# The tail the CVaR cell constructs against: the worst five percent of the window's months.
CVAR_LEVEL = 0.05
# One solver's settings, shared, so that two families cannot differ in tolerance and produce a
# difference the axes do not name.
OPTIONS = {"maxiter": 800, "ftol": 1e-14}
# The tolerance a boundary optimum can certify. See the retry in `_solve`: the tighter setting is
# tried first and this one only takes over when the solver declines to certify a point it reached.
RETRY_FTOL = 1e-10
# How much worse than the tightened attempt the certified attempt may be. Relative, because an
# objective's scale is a property of the problem: a ratio of order one and a variance of order 1e-4
# cannot share one absolute tolerance.
OPTIMALITY_SLACK = 1e-8
# A departure below this is solver noise rather than a violated constraint; the identity checks in
# the test suite are tighter than it.
SOLVER_TOLERANCE = 1e-9
# The cap used where a cell declares the bounds lifted. One is the widest a single weight can be in
# a fully invested long-only book, so it binds nothing.
UNCAPPED = 1.0


def _columns(covariance):
    """The sleeve labels a family's answer must carry, which the covariance always carries.

    Every weight in the package is indexed by the sleeve map, so a weight vector that lost its
    labels would misalign against the returns without raising anywhere downstream. The covariance is
    the input that always has them, because every estimator in the risk module returns a labelled
    matrix, so the labels come from there and a bare array is refused rather than guessed at.
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("a weight vector cannot be labelled: the covariance carries no sleeve names")
    return list(covariance.columns)


def _attempt(objective, gradient, start, cap, block, options=None):
    """One solve of the constrained problem, at one tolerance."""
    return minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, cap)] * start.size,
        constraints=block,
        options=dict(OPTIONS if options is None else options),
    )


def _solve(objective, start, cap, columns, gradient=None, extra=()):
    """A constrained minimisation: fully invested, long-only, capped, with the answer made feasible.

    One wrapper for the optimised families, so the fixed constraint set is applied in one place. A
    family that assembled its own bounds could quietly drop the cap, and a cap applied by one family
    and not another turns a constraint difference into a difference the axes do not name. `gradient`
    is supplied only where the derivative is one line of arithmetic; where it is not, the solver's own
    difference quotient is used rather than a hand-derived expression whose sign error would return a
    plausible weight vector for a different problem.
    """
    start = constraints.bounded_simplex(np.asarray(start, dtype=float), cap=cap)
    block = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    block.extend(extra)
    first = _attempt(objective, gradient, start, cap, block)
    if first.success:
        return constraints.labelled(first.x, columns, cap=cap)
    # A ratio objective with several sleeves at their lower bound is an optimum the sequential
    # quadratic solver reaches and does not always certify: at that point its line search finds no
    # improving feasible direction and reports a failure the point does not deserve. The retry
    # loosens only the tolerance, and it is accepted only when it is no worse than the first attempt -
    # so the certificate cannot have been bought with a worse portfolio. The comparison is relative,
    # because an objective's absolute scale is a property of the problem and a ratio of order one
    # cannot share an absolute tolerance with a variance of order 1e-4.
    retry = _attempt(objective, gradient, start, cap, block, options=dict(OPTIONS, ftol=RETRY_FTOL))
    slack = OPTIMALITY_SLACK * (1.0 + abs(objective(first.x)))
    if retry.success and objective(retry.x) - objective(first.x) <= slack:
        return constraints.labelled(retry.x, columns, cap=cap)
    raise ValueError(f"the optimiser did not return a fully invested book: {first.message}")


def _inverse_volatility(covariance):
    """One over each sleeve's volatility, normalised: the starting point and the diagonal answer."""
    spread = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    weights = 1.0 / spread
    return weights / weights.sum()


def equal_weight(covariance, mean=None, returns=None, cap=constraints.CAP):
    """The bar: one over the sleeve count, the portfolio every estimation-based method must beat."""
    size = len(covariance)
    return constraints.labelled(np.full(size, 1.0 / size), _columns(covariance), cap=cap)


def policy(covariance, mean=None, returns=None, cap=constraints.CAP):
    """The mandate's strategic allocation, which is also the benchmark every cell is measured on."""
    weights = np.array([universe.POLICY_WEIGHTS[name] for name in covariance.index], dtype=float)
    return constraints.labelled(weights, _columns(covariance), cap=cap)


def mean_variance(covariance, mean=None, returns=None, cap=constraints.CAP, norm=None):
    """Mean-variance: maximise `mu'w - (gamma/2) w'Sw` under the constraint set.

    `norm` bounds the weight vector's Euclidean norm, and it is the family's no-mean form: with a
    zero mean the objective is pure variance and nothing but the constraint set decides the answer. At
    the tightest norm a fully invested long-only book admits - one over the square root of the sleeve
    count - the feasible set is that single point, so the answer is the equal-weight portfolio and it
    arrives with the mean entering nowhere. That is what the cell demonstrates, and it is a property
    of the constraint set rather than a claim about means.
    """
    matrix = np.asarray(covariance, dtype=float)
    vector = np.asarray(mean, dtype=float) if mean is not None else np.zeros(matrix.shape[0])
    if vector.shape != (matrix.shape[0],):
        raise ValueError(f"a {vector.shape} mean vector does not price {matrix.shape[0]} sleeves")
    gamma = RISK_AVERSION
    extra = ()
    if norm is not None:
        budget = float(norm)
        if budget * budget * matrix.shape[0] < 1.0 - SOLVER_TOLERANCE:
            raise ValueError(
                f"a norm budget of {budget:.6f} over {matrix.shape[0]} sleeves admits no fully "
                f"invested book; the bound is below the tightest the constraint set allows"
            )
        extra = ({"type": "ineq", "fun": lambda w, b=budget: b * b - float(w @ w)},)
    return _solve(
        lambda w: -(w @ vector - 0.5 * gamma * float(w @ matrix @ w)),
        np.full(matrix.shape[0], 1.0 / matrix.shape[0]),
        cap,
        _columns(covariance),
        gradient=lambda w: -(vector - gamma * (matrix @ w)),
        extra=extra,
    )


def minimum_variance(covariance, mean=None, returns=None, cap=constraints.CAP):
    """Minimum variance: covariance error amplified, with no mean anywhere in the problem."""
    matrix = np.asarray(covariance, dtype=float)
    return _solve(
        lambda w: float(w @ matrix @ w),
        np.full(matrix.shape[0], 1.0 / matrix.shape[0]),
        cap,
        _columns(covariance),
        gradient=lambda w: 2.0 * (matrix @ w),
    )


def maximum_diversification(covariance, mean=None, returns=None, cap=constraints.CAP):
    """Maximum diversification: the ratio of weighted volatility to portfolio volatility.

    The ratio is scale-free, so the fully invested constraint is what fixes the scale.
    """
    matrix = np.asarray(covariance, dtype=float)
    spread = np.sqrt(np.diag(matrix))

    def gradient(w):
        # The derivative of the ratio, written out because the ratio's finite-difference quotient
        # near a boundary optimum is noisy enough to stall the line search. `tests/test_identities.py`
        # checks it against a central difference on a planted covariance.
        variance = float(w @ matrix @ w)
        return -(spread / np.sqrt(variance) - (w @ spread) * (matrix @ w) / variance ** 1.5)

    return _solve(
        lambda w: -float(w @ spread) / np.sqrt(float(w @ matrix @ w)),
        np.full(matrix.shape[0], 1.0 / matrix.shape[0]),
        cap,
        _columns(covariance),
        gradient=gradient,
    )


def erc(covariance, mean=None, returns=None, cap=constraints.CAP):
    """Equal risk contribution: every sleeve contributing the same share of portfolio volatility.

    Solved by minimising the squared deviations of the contributions from their common target, which
    is the definition of the portfolio rather than a proxy for it. The spread of the contributions is
    not returned here; it is computed from the returned weights wherever it is reported, so a caller
    cannot read a spread that belongs to a different weight vector than the one it holds.
    """
    matrix = np.asarray(covariance, dtype=float)
    size = matrix.shape[0]

    def objective(w):
        sigma = np.sqrt(float(w @ matrix @ w))
        risk = w * (matrix @ w) / sigma
        return float(((risk - sigma / size) ** 2).sum())

    return _solve(objective, _inverse_volatility(matrix), cap, _columns(covariance))


def hierarchical_risk_parity(covariance, mean=None, returns=None, cap=constraints.CAP):
    """Hierarchical risk parity: cluster the sleeves, then bisect the capital down the tree.

    The distance is the usual one derived from correlation, the clustering is single linkage, and the
    allocation at each node splits by the two children's own variance, which makes it a rule of the
    covariance rather than an optimisation. Because it is a rule it has no bounds to set, so the
    constraint set is applied to its answer by projection; where the cap binds, the projection is
    what the cell actually traded and the report says so.
    """
    matrix = np.asarray(covariance, dtype=float)
    size = matrix.shape[0]
    if size == 1:
        return constraints.labelled(np.ones(1), _columns(covariance), cap=cap)
    spread = np.sqrt(np.diag(matrix))
    correlation = matrix / np.outer(spread, spread)
    correlation = np.clip(correlation, -1.0, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1.0 - correlation), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    order = list(leaves_list(linkage(squareform(distance, checks=False), method="single")))
    weights = np.ones(size)

    def cluster_variance(members):
        sub = matrix[np.ix_(members, members)]
        inverse = 1.0 / np.diag(sub)
        inverse = inverse / inverse.sum()
        return float(inverse @ sub @ inverse)

    def bisect(members):
        if len(members) < 2:
            return
        half = len(members) // 2
        left, right = members[:half], members[half:]
        left_variance, right_variance = cluster_variance(left), cluster_variance(right)
        total = left_variance + right_variance
        # A pair of clusters with no variance at all cannot be split by variance; the capital is
        # halved instead, which is the limit the formula approaches and the only defensible answer.
        alpha = 0.5 if total == 0 else 1.0 - left_variance / total
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        bisect(left)
        bisect(right)

    bisect(order)
    return constraints.labelled(weights, _columns(covariance), cap=cap)


def expected_shortfall(weights, returns, level=CVAR_LEVEL):
    """The window's realised expected shortfall of a book, as a positive share of NAV lost.

    The mean of the worst `ceil(level * observations)` months, computed directly from the window's
    own returns. It is written here rather than inferred from the linear program's objective so that
    the program has something independent to be checked against.
    """
    weights = np.asarray(weights, dtype=float)
    losses = -(np.asarray(returns, dtype=float) @ weights)
    count = max(int(np.ceil(level * len(losses))), 1)
    return float(np.sort(losses)[-count:].mean())


def mean_cvar(covariance, mean=None, returns=None, cap=constraints.CAP, level=CVAR_LEVEL):
    """Mean-CVaR: minimise expected shortfall at the stated level, by linear program.

    The Rockafellar-Uryasev substitution makes the objective linear: a level and the mean of the
    excess beyond it, minimised jointly over the weights. Its optimum equals the expected shortfall
    of the weights it returns, which is the identity the test suite checks against the direct
    computation above rather than against the solver's own report.
    """
    if returns is None:
        raise ValueError("mean-CVaR is constructed against the window's own returns, and none was given")
    window = np.asarray(returns, dtype=float)
    observations, size = window.shape
    if size != len(covariance):
        raise ValueError(f"{size} return columns do not match {len(covariance)} sleeves")
    # Variables: the weights, the level, then one excess per observation.
    objective = np.concatenate([np.zeros(size), np.ones(1), np.full(observations, 1.0 / (level * observations))])
    # For each month: level + excess must cover that month's loss, with the excess non-negative.
    block = np.hstack([-window, -np.ones((observations, 1)), -np.eye(observations)])
    bounds = [(0.0, cap)] * size + [(None, None)] + [(0.0, None)] * observations
    equality = np.zeros((1, size + 1 + observations))
    equality[0, :size] = 1.0
    result = linprog(
        objective,
        A_ub=block,
        b_ub=np.zeros(observations),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"the tail construction did not solve: {result.message}")
    return constraints.labelled(result.x[:size], _columns(covariance), cap=cap)


# The families behind one name each, in the order the grid registers them. The keys are the names the
# registry refers to; a family missing from here cannot be run, which is what keeps a cell and its
# constructor from drifting apart.
FAMILIES = {
    "equal_weight": equal_weight,
    "policy": policy,
    "mean_variance": mean_variance,
    "minimum_variance": minimum_variance,
    "maximum_diversification": maximum_diversification,
    "erc": erc,
    "hierarchical_risk_parity": hierarchical_risk_parity,
    "mean_cvar": mean_cvar,
}
