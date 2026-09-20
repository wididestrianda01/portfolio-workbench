"""The noise floor: the paired test, the bootstrap, and the bars a difference has to clear.

**Every comparison is a paired test on the difference of two monthly return series.** A single
strategy's annualised information ratio carries a standard error near 0.30 over the 131 months this
panel affords, so two cells compared by their separate ratios are indistinguishable by construction:
every method sits inside every other's interval. Pairing is what buys resolution, and it buys it only
because the cells are highly correlated: same universe, same months, long-only books that mostly
agree. The realised correlation is therefore measured and reported rather than assumed, because the
resolution quoted beside every verdict is a function of it.

**The resolution is quoted in the metric's own units, at the bar the row was decided at.** The paired
standard error of the annualised information-ratio difference is `sqrt(12/T) * sqrt(2 * (1 - rho))`
under the null, which is 0.098 at a correlation of 0.95 and 131 months, the figure the design's own
power calculation arrived at. The smallest difference a test detects at eighty percent power is that
test's **own** critical value plus the power quantile times the standard error, and a comparison row
decides at the family-wise bar of 2.9552, so the multiplier is 3.7968 and the resolution 0.36 here. The
nominal two-sided five percent would give 2.8016 and 0.27, and printing that beside a verdict the
family-wise bar produced describes a different test from the one that decided the row: it would claim a
finer resolution than the test has, which is the direction that flatters every "no difference detected"
and the one the noise floor exists to guard. Two multipliers coexist for that reason - the
detectable-alpha example in the attribution layer decides at the nominal two-sided level and keeps the
nominal one - and `detection` builds either from the bar it belongs to.

**The bar is the family-wise one, and the correlation-adjusted bar is printed beside it.** Across
sixteen cells the declared bar on the null statistic is the Bonferroni-equivalent
`Phi^-1(1 - 0.05/(2*16))` = 2.9552, taken at the two-sided level because the ladder compares the
**absolute** statistic: a cell is reported as *different* from the benchmark, and the direction is read
afterwards rather than being the hypothesis. The same number is the 95th percentile of the maximum of
sixteen independent absolute standard normals to within a quarter of a percent (2.9478 measured against
2.9552), and Bonferroni is the conservative one of the two. The one-sided quantile over the same cells
is 2.7344, the value this module shipped before the two readings were separated, and a bar that would
have held the family to 9.5% rather than to the 5% it is declared at. The cells are positively
correlated, so the effective number of tests is smaller and the honest bar would be lower; both numbers
are printed, and the conservative one decides. The haircut is **self-imposed from the literature** - it
sits in the same territory as the threshold that reading arrived at once search is accounted for - and
nothing regulatory requires it. Saying so is the point: a self-imposed bar presented as a requirement
would borrow an authority it does not have.

**A difference is reported only when every leg of the row's own claim passes, and the legs answer two
different questions.** The paired bar and the cell's own share of resamples in which its advantage keeps
its sign are statements about the cell against the benchmark; the expanding-window sign is a statement
about the protocol; and the leader's **rank retention** is a statement about the cells against each
other, that is, whether one of them is *uniquely* best. A cell failing a leg of its own claim is
published as **no difference detected**, with the resolution limit beside it. A leader that clears every
leg about its own advantage and fails the rank-retention rung is published as a refusal to choose
*between* cells rather than as an absence of an advantage, because those are two different findings and
only one of them is measured here.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..factors import components
from . import metrics

# The level the family-wise bar is declared at, stated once.
ALPHA = 0.05
# The power the design's resolution is quoted at, and its quantile on its own: a resolution is built on
# the bar its own test decides at, and the two tests in this package that quote one decide at different
# bars. Keeping the quantile separate is what lets both be built rather than one being reused.
POWER_LEVEL = 0.80
POWER_QUANTILE = float(norm.ppf(POWER_LEVEL))
# The two-sided 5% critical value plus the power quantile: the multiplier for a test that decides at the
# **nominal** level, which is the detectable-alpha example in the attribution layer. A comparison row
# decides at the family-wise bar and calls `detection` with that bar instead.
POWER = float(norm.ppf(1.0 - ALPHA / 2.0) + POWER_QUANTILE)
# The resample count and the retention floor the design fixed before any cell ran.
BOOTSTRAP_DRAWS = 2000
RANK_RETENTION_FLOOR = 0.80
# The one documented seed, read from the module that owns every draw in the package, so a bootstrap
# and a permutation null are reproducible from the same number.
SEED = components.SEED
# The tolerance a rerun is held to. Solvers differ in their last decimal, so bit-equality is not the
# bar and pretending otherwise would be a false acceptance criterion; the seed makes the draws
# reproducible, which is the part that can be exact.
RERUN_TOLERANCE = 1e-6


def family_wise_bar(tests, alpha=ALPHA):
    """The bar a null statistic must clear when `tests` hypotheses are tested at once.

    Bonferroni-equivalent and **two-sided**, because the statistic that is held against it is the
    absolute one: `haircut` decides on `abs(statistic)` and the ladder labels a cell "significantly
    behind" as well as "a candidate", so the hypothesis being tested is that the cell differs from the
    benchmark and the sign is read as a direction afterwards. The level therefore splits over both
    directions, which puts the quantile at `1 - alpha / (2 * tests)`.

    Reading the same family-wise level one-sidedly gives a *lower* bar - `Phi^-1(1 - alpha/tests)`,
    2.7344 over sixteen cells - and holding an absolute statistic against that number is what the
    two readings cannot share: it would reject under the null at 9.5% while the table is published as
    a 5% family-wise test. One quantile cannot serve both.
    """
    if tests < 1:
        raise ValueError(f"{tests} tests have no family-wise bar")
    return float(norm.ppf(1.0 - alpha / (2.0 * tests)))


def correlation_matrix(frame):
    """The realised correlation of the cells' monthly return series, published rather than assumed."""
    values = np.asarray(frame, dtype=float)
    if values.shape[1] < 2:
        raise ValueError("a pairwise correlation needs at least two cells")
    return pd.DataFrame(np.corrcoef(values, rowvar=False), index=frame.columns, columns=frame.columns)


def effective_tests(correlation):
    """Nyholt's effective number of independent tests, read off the realised correlation matrix.

    Sixteen cells evaluated on the same months are not sixteen independent tests, and the eigenvalue
    spread of their correlation matrix says how many they effectively are. This is the number behind
    the lower bar the design asks to be printed beside the conservative one - never instead of it.
    """
    matrix = np.asarray(correlation, dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    count = len(eigenvalues)
    if count < 2:
        return 1.0
    return float(max(1.0, 1.0 + (count - 1.0) * (1.0 - float(eigenvalues.var()) / count)))


def adjusted_bar(correlation, alpha=ALPHA):
    """The bar the realised correlation implies, which is lower than the declared one and is reported
    beside it: the declared bar stays the one that decides.

    The effective count is rounded to a whole number of tests, because that is what the bar's own
    arithmetic admits - and the estimate is an estimate: a fully degenerate matrix comes out just under
    two rather than at one, which is the estimator's slack rather than a second test.
    """
    return family_wise_bar(max(int(round(effective_tests(correlation))), 1), alpha=alpha)


def detection(bar, power=POWER_LEVEL):
    """The multiplier that turns a standard error into the smallest difference a test would detect.

    `bar` is the critical value the test's **own** decision is made at, and the answer is that bar plus
    the power quantile: a difference at the bar is detected with fifty percent probability, so eighty
    percent power costs the extra quantile. Building the number from the bar rather than from a constant
    is what keeps a row's resolution and its verdict statements about the same test - the comparison
    rows decide at the family-wise bar, the detectable-alpha example at the nominal two-sided one, and
    a resolution quoted at the other bar would describe a test that did not print the verdict beside it.
    """
    return float(bar + norm.ppf(power))


def paired(reference, alternative, benchmark, bar, periods=metrics.PERIODS_PER_YEAR):
    """One paired comparison: the difference of two annualised information ratios, the standard error
    the realised correlation gives it, and the resolution that follows.

    `reference` and `alternative` are the two monthly return series against the same benchmark, so the
    comparison of a cell against the benchmark passes the benchmark's own series as the alternative -
    its active series is identically zero and its ratio is zero by construction rather than undefined.
    Two series that are perfectly correlated leave no variance for the standard error, and the answer
    is then decided by whether the two books are the same book: the difference of their ratios is
    reported as zero when they are, and as unbounded when a perfectly correlated pair still differs.

    `bar` is required rather than defaulted, because the resolution this returns is a statement about
    the test that decides with it: a caller that does not know its own bar does not know what resolution
    it is reporting, and the default would silently be the nominal one for a row decided family-wise.
    """
    left = pd.Series(reference, dtype=float)
    right = pd.Series(alternative, dtype=float).reindex(left.index)
    if right.isna().any():
        raise ValueError("the two series being paired do not cover the same months")
    months = int(len(left))
    if months < 3:
        raise ValueError("a paired test needs at least three months")
    rolling = float(np.corrcoef(left, right)[0, 1])
    ratio = metrics.information_ratio(metrics.active(left, benchmark), periods) - metrics.information_ratio(
        metrics.active(right, benchmark), periods
    )
    error = float(np.sqrt(periods / months) * np.sqrt(2.0 * (1.0 - rolling)))
    # A correlation that reaches one to within floating-point accumulation leaves a standard error that
    # is zero in every sense a reader would accept, and dividing by it would report an unbounded
    # statistic from a rounding error rather than from a result.
    if error <= RERUN_TOLERANCE:
        statistic = 0.0 if abs(ratio) <= RERUN_TOLERANCE else math.copysign(math.inf, ratio)
    else:
        statistic = ratio / error
    return {
        "months": months,
        "correlation": rolling,
        "difference": ratio,
        "standard_error": error,
        "statistic": statistic,
        # The smallest difference this test would have detected at eighty percent power, printed
        # beside any verdict that reports no difference, and built from the bar this test decides at.
        # The statistic is reported in the units the bar is declared in rather than as a p-value: the
        # bar is a z, and a second scale for the same number is a place for the two to disagree.
        "bar": float(bar),
        "resolution": float(detection(bar) * error),
        "degenerate": bool(error <= RERUN_TOLERANCE),
    }


def rank_retention(frame, benchmark, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """How often the full-sample leader stays leader when the months are resampled, and what each cell's
    own advantage does under the same resampling.

    One resampled month index is drawn per resample and shared by every cell, so the correlation
    between cells survives the resampling: drawing each cell's months independently would break the
    very structure that makes the comparison paired and would report a leader stability nobody
    measured. Two statistics come out of it, and they answer different questions. The design's headline
    is the **rank retention** of the leader - the share of resamples in which the cell that led the full
    sample leads the resample - which is a statement about the *cells against each other*, that is,
    whether one of them is uniquely best. The per-cell statistic is the share of resamples in which that
    cell's information ratio **keeps the sign** of its full-sample one, which is the same question asked
    of the claim the cell's own row makes: a row is a statement about one cell against the benchmark, and
    an advantage that flips sign when the months are resampled has not been measured. A leader can fail
    the first while passing the second, and on a table of near-tied methods it usually does; reading the
    first as a failure of the advantage is reading a ranking as an absence.
    """
    active = pd.DataFrame(frame, dtype=float).sub(pd.Series(benchmark, dtype=float), axis=0)
    values = active.to_numpy()
    months, cells = values.shape
    if draws < 1:
        raise ValueError("a bootstrap needs at least one resample")
    generator = np.random.default_rng(seed)
    index = generator.integers(0, months, size=(draws, months))
    drawn = values[index]
    spread = drawn.std(axis=1, ddof=1)
    spread[spread == 0.0] = np.nan
    # One vectorised pass over every resample and every cell, rather than `metrics.information_ratio`
    # called per cell per draw: the annualisation is that function's, applied to the drawn matrix at
    # once, and the shared month index above is what keeps the cells correlated through it.
    ratios = drawn.mean(axis=1) / spread * np.sqrt(metrics.PERIODS_PER_YEAR)
    full = np.asarray(active.mean() / active.std(ddof=1) * np.sqrt(metrics.PERIODS_PER_YEAR), dtype=float)
    leader = int(np.nanargmax(full))
    drawn_leader = np.nanargmax(np.where(np.isnan(ratios), -np.inf, ratios), axis=1)
    order = np.argsort(np.where(np.isnan(ratios), np.inf, -ratios), axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(cells)[None, :].repeat(draws, axis=0), axis=1)
    keeps = np.sign(ratios) == np.sign(full)[None, :]
    return {
        "draws": int(draws),
        "seed": int(seed),
        "leader": str(frame.columns[leader]),
        "retention": float((drawn_leader == leader).mean()),
        "floor": RANK_RETENTION_FLOOR,
        "cells": {
            str(name): {
                "share_leader": float((drawn_leader == position).mean()),
                "median_rank": float(np.median(ranks[:, position]) + 1.0),
                "retention": float(keeps[:, position].mean()),
                "information_ratio": float(full[position]),
            }
            for position, name in enumerate(frame.columns)
        },
    }


def haircut(statistic, bar):
    """The multiple-testing haircut applied: whether the statistic clears the declared family-wise bar.

    The statement travels with the verdict because the bar is the project's own and not a rule anyone
    imposed on it: a self-imposed threshold reported as a requirement would borrow an authority the
    reading does not give it.
    """
    return {
        "bar": float(bar),
        "clears": bool(abs(float(statistic)) >= float(bar)),
        "statement": (
            "self-imposed from the literature, not a regulatory requirement: nothing located imposes a "
            "multiple-testing correction on portfolio research"
        ),
    }


def sign_holds(rolling_ratio, expanding_ratio):
    """Whether the advantage keeps its sign under the secondary protocol.

    The expanding run trades the same months with a longer estimate, so a sign that flips with the
    window length was a property of the window rather than of the method.
    """
    left, right = float(rolling_ratio), float(expanding_ratio)
    if abs(left) <= RERUN_TOLERANCE or abs(right) <= RERUN_TOLERANCE:
        return False
    return (left > 0.0) == (right > 0.0)
