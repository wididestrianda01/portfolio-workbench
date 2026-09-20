"""Hand-calculated identities, on hand-built or planted inputs.

Every check here is arithmetic that has to come out exactly, which is why the frames are planted
rather than loaded: if one of these fails, the code is wrong rather than the market.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.attribute import brinson
from portfolio_workbench.attribute import factor as factor_attribution
from portfolio_workbench.budget import euler
from portfolio_workbench.compare import registry, table as comparison
from portfolio_workbench.construct import constraints
from portfolio_workbench.construct import families as construct_families
from portfolio_workbench.construct import means as construct_means
from portfolio_workbench.data import universe
from portfolio_workbench.factors import components as cp
from portfolio_workbench.evaluate import metrics, statistics
from portfolio_workbench.factors import exposures, spanning, spine
from portfolio_workbench.risk import covariance

CALENDAR = pd.period_range("2000-01", periods=140, freq="M")
FACTORS = ["f1", "f2"]


def planted(alpha=0.0, beta=(1.0, -0.5), noise=0.0, seed=3, periods=140):
    """A sleeve frame whose exposures are known, so recovery is checked rather than assumed."""
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(rng.normal(0, 0.03, (periods, 2)), index=CALENDAR[:periods], columns=FACTORS)
    values = alpha + factors.to_numpy() @ np.asarray(beta) + rng.normal(0, noise, periods)
    return pd.DataFrame({"sleeve_a": values}, index=factors.index), factors


def test_the_exposure_regression_recovers_its_planted_values_and_the_split_sums():
    """The regression returns what was planted, and the factor and idiosyncratic parts add to the
    sleeve's own variance: the split is exact for variance, which is why shares of variance are what
    the report prints, the square roots not adding."""
    returns, factors = planted(alpha=0.004, beta=(1.5, -0.75))
    fit = exposures.regress(returns, factors)
    assert fit["alpha"][0] == pytest.approx(0.004, abs=1e-12)
    assert fit["beta"][:, 0] == pytest.approx([1.5, -0.75], abs=1e-12)
    assert fit["resid_var"][0] == pytest.approx(0.0, abs=1e-20)
    assert fit["r2"][0] == pytest.approx(1.0, abs=1e-12)

    noisy, noisy_factors = planted(noise=0.01)
    fit = exposures.regress(noisy, noisy_factors)
    total = float(noisy["sleeve_a"].var(ddof=1))
    assert fit["explained_var"][0] + fit["resid_var"][0] == pytest.approx(total, rel=1e-12)
    factor_part, idio_part = exposures.factor_split([1.0], fit["beta"], fit["factor_cov"], fit["resid_var"])
    assert factor_part + idio_part == pytest.approx(total, rel=1e-12)


def test_the_block_is_the_sleeve_arithmetic_it_declares():
    """The constructed block on round numbers, one row per series: a level that is the government
    block's average, a slope that is long minus short, and two spreads naming the sleeves they read."""
    months = pd.PeriodIndex(["2020-01", "2020-02"], freq="M")
    returns = pd.DataFrame(
        {"IBGL.AS": [0.04, -0.02], "IEGE.AS": [0.01, 0.01], "IEAC.AS": [0.05, -0.01], "IHYG.L": [0.09, 0.00]},
        index=months,
    )
    block = spine.constructed_block(returns)
    assert list(block.columns) == list(spine.CONSTRUCTED)
    assert float(block.loc[months[0], "government_level"]) == pytest.approx(0.025)
    assert float(block.loc[months[1], "term_slope"]) == pytest.approx(-0.03)
    assert float(block.loc[months[0], "credit"]) == pytest.approx(0.01)
    assert float(block.loc[months[1], "high_yield_excess"]) == pytest.approx(0.01)


def test_orthogonalising_the_block_leaves_the_fit_identical_and_identifies_the_loadings():
    """The parameterisation change, as an identity: the same window regressed on the declared block and
    on the block orthogonalised against itself gives the same residuals, the same alpha and the same
    share of variance, and the two loading vectors describe one fit. What changes is the design's
    collinearity, measured on its correlation matrix so the series' scales cannot flatter it, and that
    is the point of it."""
    rng = np.random.default_rng(4)
    months = pd.period_range("2010-01", periods=80, freq="M")
    level = rng.normal(0.001, 0.017, 80)
    quiet = rng.normal(0.0005, 0.002, 80)
    declared = pd.DataFrame(
        {
            "government_level": level,
            # The long government sleeve dominates the short one, so the slope is the level almost
            # exactly: the correlation the orthogonalisation exists to remove.
            "term_slope": 1.02 * level + 0.001 * quiet,
            "credit": -2.0 * level + rng.normal(0, 0.01, 80),
            "high_yield_excess": rng.normal(0.001, 0.013, 80),
        },
        index=months,
    )
    y = pd.DataFrame({"sleeve": 0.7 * level + 0.3 * declared["credit"] + rng.normal(0, 0.004, 80)}, index=months)
    orthogonal = exposures.orthogonalise(declared)

    plain = exposures.regress(y, declared)
    rotated = exposures.regress(y, orthogonal)
    # The two fits are the same fit, and they agree to the precision the worse-conditioned of the two
    # can reach: with the pair 0.9999 correlated, the declared-basis fit itself is computed at a
    # conditioning of 1e8, which is the same problem the coefficients have.
    assert plain["alpha"][0] == pytest.approx(rotated["alpha"][0], abs=1e-9)
    assert plain["r2"][0] == pytest.approx(rotated["r2"][0], rel=1e-6)
    assert plain["resid_var"][0] == pytest.approx(rotated["resid_var"][0], rel=1e-5)
    assert np.allclose(declared.to_numpy() @ plain["beta"][:, 0], orthogonal.to_numpy() @ rotated["beta"][:, 0])

    inflation = exposures.variance_inflation
    assert np.corrcoef(level, declared["term_slope"])[0, 1] > 0.999
    assert inflation(declared) > 100, "the declared pair leaves a coefficient unidentified"
    assert inflation(orthogonal) < 3, "orthogonalised, the block pinches nothing"
    assert np.all(np.abs(np.corrcoef(orthogonal.to_numpy(), rowvar=False) - np.eye(4)) < 0.2)


def test_the_spanning_statistic_reproduces_the_identity_and_ignores_a_rescaling():
    """Huberman–Kandel: an asset that is a portfolio of the benchmark, with weights summing to one,
    leaves no detectable intercept and loadings that sum to one. The statistic is invariant to any
    invertible transform of the test assets, and a planted intercept is rejected."""
    rng = np.random.default_rng(2)
    months = pd.period_range("2000-01", periods=180, freq="M")
    factors = pd.DataFrame(rng.normal(0, 0.03, (180, 2)), index=months, columns=["b0", "b1"])
    spanned = pd.DataFrame(
        factors.to_numpy() @ np.outer([0.5, 0.5], np.ones(3)) + rng.normal(0, 1e-5, (180, 3)),
        index=months,
        columns=["t0", "t1", "t2"],
    )
    result = spanning.grs(spanned, factors)
    assert result["rows_sum"] == pytest.approx(np.ones(3), abs=1e-3)
    assert result["p_value"] > 0.05, "a spanned benchmark must not be rejected"

    transform = np.array([[1.5, 0.4, 0.0], [0.2, -1.0, 0.7], [0.0, 0.3, 2.0]])
    moved = pd.DataFrame(spanned.to_numpy() @ transform, index=months, columns=spanned.columns)
    assert spanning.grs(moved, factors)["p_value"] == pytest.approx(result["p_value"], rel=1e-9)

    tilted = pd.DataFrame(rng.normal(0, 0.03, (180, 3)) + 0.01, index=months, columns=spanned.columns)
    assert spanning.grs(tilted, factors)["p_value"] < 1e-4


def test_the_factor_covariance_reconstructs_the_discarded_eigenvalues():
    """What the retained components leave out is exactly the sum of the eigenvalues that were not
    retained, and the share reported is that sum over the total. Checked against an independent
    decomposition of the same window, so the claim is arithmetic rather than an assertion."""
    rng = np.random.default_rng(5)
    columns = [f"s{position}" for position in range(5)]
    frame = pd.DataFrame(rng.normal(0, 0.02, (80, 5)), index=pd.period_range("2000-01", periods=80, freq="M"), columns=columns)
    _, report = covariance.factor_model(frame, count=2)
    values = frame.to_numpy()
    standardised = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.corrcoef(standardised, rowvar=False)))[::-1]
    assert report["discarded_sum"] == pytest.approx(float(eigenvalues[2:].sum()), abs=1e-10)
    assert report["discarded_share"] == pytest.approx(float(eigenvalues[2:].sum()) / len(columns), abs=1e-12)
    assert report["residual_sum"] == pytest.approx(report["discarded_sum"], abs=1e-10)


def test_the_varimax_rotation_preserves_the_model():
    """Presentation only: the communalities, the reconstructed correlation matrix and the total
    variance are preserved exactly, so the rotated table is the same model seen differently. What it
    does change is which component a share of variance sits in, which is why the count rule runs on
    the unrotated eigenvalues."""
    rng = np.random.default_rng(7)
    factors = rng.normal(0, 0.03, (300, 3))
    frame = pd.DataFrame(
        factors @ rng.normal(0, 1, (3, 11)) + rng.normal(0, 0.01, (300, 11)),
        index=pd.period_range("2000-01", periods=300, freq="M"),
        columns=[f"s{position}" for position in range(11)],
    )
    loadings = cp.decompose(frame, draws=20)["loadings"]
    rotated = cp.varimax(loadings)
    assert np.allclose((rotated ** 2).sum(axis=1), (loadings ** 2).sum(axis=1), atol=1e-12)
    assert np.allclose(rotated @ rotated.T, loadings @ loadings.T, atol=1e-12)
    assert float(np.trace(rotated.T @ rotated)) == pytest.approx(float(np.trace(loadings.T @ loadings)), rel=1e-12)


def test_the_estimators_are_what_they_claim_to_be():
    """The sample estimator is the sample covariance; the Marchenko–Pastur edge is the closed form the
    sanity check pins; shrinkage averages toward a better-conditioned matrix, which is its purpose; and
    the time-series premium is the hand calculation."""
    rng = np.random.default_rng(11)
    columns = [f"s{position}" for position in range(5)]
    frame = pd.DataFrame(rng.normal(0, 0.02, (80, 5)), index=pd.period_range("2000-01", periods=80, freq="M"), columns=columns)
    assert np.allclose(covariance.sample(frame).to_numpy(), np.cov(frame.to_numpy(), rowvar=False, ddof=1))

    assert cp.mp_edge(11, 191) == pytest.approx(1.53759, rel=1e-4)
    assert cp.mp_edge(11, 60) == pytest.approx(2.03970, rel=1e-4)

    collinear = frame.copy()
    collinear[columns[-1]] = collinear[columns[0]] * 0.999999 + rng.normal(0, 1e-7, 80)
    shrunk, intensity = covariance.shrinkage(collinear)
    assert np.linalg.cond(covariance.sample(collinear)) > 1e6
    assert np.linalg.cond(shrunk) < 1e3
    assert 0.0 < intensity < 1.0

    series = pd.Series(np.linspace(-0.02, 0.04, 60), index=pd.period_range("2000-01", periods=60, freq="M"), name="f")
    stats = spanning.premiums(series.to_frame())["f"]
    assert stats["t"] == pytest.approx(float(series.mean() / (series.std(ddof=1) / np.sqrt(60))), rel=1e-12)


# ------------------------------------------------------------------ the construct layer

def diagonal(variances):
    """A labelled diagonal covariance, so a closed form can be written down for it."""
    names = list(variances)
    return pd.DataFrame(np.diag([variances[name] for name in names]), index=names, columns=names)


def test_the_constraint_set_is_a_capped_simplex_rather_than_a_clip():
    """Clipping a weight at the cap and renormalising leaves the vector over the cap it just enforced.
    The projection fills the cap exactly and spreads what it removed by the room the other sleeves have
    left; a cap the sleeve count cannot satisfy is refused rather than returned as a book that cannot
    exist."""
    assert constraints.bounded_simplex(np.array([0.9, 0.05, 0.05]), cap=0.5) == pytest.approx([0.5, 0.25, 0.25], abs=1e-12)
    assert constraints.bounded_simplex(np.array([0.1, 0.3, 0.6]), cap=0.6) == pytest.approx([0.1, 0.3, 0.6], abs=1e-12)
    assert constraints.bounded_simplex(np.array([-0.2, 1.2]), cap=1.0) == pytest.approx([0.0, 1.0], abs=1e-12)
    with pytest.raises(ValueError, match="cannot be satisfied"):
        constraints.bounded_simplex(np.array([0.5, 0.5]), cap=0.35)


def test_the_band_bounds_drift_and_the_trade_is_scaled_to_the_turnover_cap():
    """The band is the rule that stops the book churning on noise, and it does not bound turnover: a
    trade twice the cap is scaled toward the target, lands on a convex combination of two feasible
    books, and reports the cap as binding. The cost is then charged on traded notional, which is twice
    the one-way turnover, and the establishment trade is the whole book once."""
    current = pd.Series([0.5, 0.3, 0.2], index=list("abc"))
    target = pd.Series([0.505, 0.10, 0.395], index=list("abc"))
    banded = constraints.banded(target, current, band=0.01)
    assert banded["a"] == pytest.approx(0.5), "a move inside the band is not a trade"

    rebalance = constraints.rebalance(target, current, band=0.01, cap=0.05)
    assert rebalance["binding"] is True
    assert rebalance["turnover"] == pytest.approx(0.05, abs=1e-12)
    assert rebalance["weights"].sum() == pytest.approx(1.0, abs=1e-12)
    assert rebalance["untaken"] > 0.0, "the trade that was not taken is reported, not dropped"
    assert list(rebalance["weights"].index) == list("abc")

    assert constraints.cost(0.05, bp=10.0) == pytest.approx(0.0001, abs=1e-15)
    # A path arrives as a labelled series, and the one definition has to hand it back labelled: the
    # sensitivity run reads the charge against the months it was traded in.
    path = pd.Series([0.05, 0.02], index=pd.period_range("2020-01", periods=2, freq="M"))
    charged = constraints.cost(path, bp=10.0)
    assert list(charged.index) == list(path.index)
    assert float(charged.iloc[0]) == pytest.approx(0.0001, abs=1e-15)
    assert constraints.establishment(pd.Series([0.5, 0.5], index=list("ab")))["cost"] == pytest.approx(0.001, abs=1e-15)


def test_every_constructor_returns_a_labelled_book_the_constraint_set_allows():
    """One interface, one constraint set: each family returns a fully invested, long-only book inside
    the cap, labelled in the covariance's own order, so a reordering cannot misalign it against the
    returns."""
    rng = np.random.default_rng(3)
    names = list(universe.TICKERS)
    frame = pd.DataFrame(rng.normal(0, 0.02, (80, len(names))), columns=names)
    matrix = pd.DataFrame(np.cov(frame, rowvar=False, ddof=1), index=names, columns=names)
    vector = frame.mean()
    for name, family in construct_families.FAMILIES.items():
        book = family(matrix, vector, frame)
        assert list(book.index) == names, f"{name} lost the sleeve map's order"
        assert float(book.sum()) == pytest.approx(1.0, abs=1e-12), f"{name} is not fully invested"
        assert float(book.min()) >= 0.0, f"{name} went short"
        assert float(book.max()) <= constraints.CAP + 1e-12, f"{name} breached the cap"


def test_minimum_variance_and_maximum_diversification_match_their_closed_forms():
    """On a diagonal covariance both optima are known in closed form: minimum variance weights each
    sleeve by the reciprocal of its variance, maximum diversification by the reciprocal of its
    volatility. The second is the reason the family is called diversification - the ratio is scale
    free, so the fully invested constraint is what fixes its scale."""
    matrix = diagonal({"a": 0.04, "b": 0.01, "c": 0.09})
    inverse_variance = (1.0 / np.diag(matrix))
    inverse_volatility = (1.0 / np.sqrt(np.diag(matrix)))
    assert construct_families.minimum_variance(matrix, cap=1.0).to_numpy() == pytest.approx(
        inverse_variance / inverse_variance.sum(), abs=1e-7
    )
    assert construct_families.maximum_diversification(matrix, cap=1.0).to_numpy() == pytest.approx(
        inverse_volatility / inverse_volatility.sum(), abs=1e-7
    )


def test_the_diversification_gradient_is_the_derivative_of_the_ratio_it_optimises():
    """The objective is left to the solver's own difference quotient everywhere it is well behaved, and
    written out here only because the quotient near a boundary optimum stalls the line search. A
    gradient that is not the derivative returns a plausible book for a different problem, so it is
    checked against a central difference on the same planted covariance."""
    rng = np.random.default_rng(8)
    names = ["a", "b", "c", "d"]
    frame = pd.DataFrame(rng.normal(0, 0.02, (80, 4)), columns=names)
    matrix = np.cov(frame, rowvar=False, ddof=1)
    spread = np.sqrt(np.diag(matrix))
    ratio = lambda w: float(w @ spread) / np.sqrt(float(w @ matrix @ w))
    gradient = lambda w: -(spread / np.sqrt(float(w @ matrix @ w))
                           - (w @ spread) * (matrix @ w) / float(w @ matrix @ w) ** 1.5)
    point = np.full(4, 0.25)
    step = 1e-7
    difference = np.array([
        (ratio(point + step * np.eye(4)[position]) - ratio(point - step * np.eye(4)[position])) / (2 * step)
        for position in range(4)
    ])
    assert gradient(point) == pytest.approx(-difference, abs=1e-8)


def test_equal_risk_contribution_equalises_the_contributions_and_they_sum_to_volatility():
    """The portfolio is defined by its contributions rather than by an objective that approximates
    them, so the check is that the contributions the returned book produces are equal, and that they
    sum to the book's volatility - the additivity the whole risk-budget layer rests on."""
    matrix = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.01]],
        index=list("abc"),
        columns=list("abc"),
    )
    def contributions(book, values):
        volatility = float(np.sqrt(book @ values @ book))
        return book * (values @ book) / volatility, volatility

    values = matrix.to_numpy()
    book = construct_families.erc(matrix, cap=1.0).to_numpy()
    spread, volatility = contributions(book, values)
    assert spread.max() - spread.min() < 1e-6 * volatility, "the contributions are the portfolio, not a proxy for it"
    assert float(spread.sum()) == pytest.approx(volatility, rel=1e-10)

    # With the cap binding the equal-contribution point is unreachable, so the most that can be asked
    # of the constrained solve is that it is at least as even as the book it started from.
    bounded = construct_families.erc(matrix, cap=0.4).to_numpy()
    even, _ = contributions(np.full(3, 1 / 3), values)
    capped, _ = contributions(bounded, values)
    assert capped.max() - capped.min() <= even.max() - even.min() + 1e-12

    plain = diagonal({"a": 0.04, "b": 0.01, "c": 0.09})
    inverse_volatility = 1.0 / np.sqrt(np.diag(plain))
    assert construct_families.erc(plain, cap=1.0).to_numpy() == pytest.approx(
        inverse_volatility / inverse_volatility.sum(), abs=1e-7
    )


def test_hierarchical_risk_parity_splits_by_cluster_variance():
    """With every sleeve alike the bisection has nothing to split on and returns the equal-weight book
    exactly. With two clusters of equal within-cluster variance, the top-level split is the hand
    calculation: each side takes the other side's cluster variance over the sum."""
    alike = diagonal({name: 0.04 for name in "abcd"})
    assert construct_families.hierarchical_risk_parity(alike).to_numpy() == pytest.approx(np.full(4, 0.25), abs=1e-9)

    rng = np.random.default_rng(9)
    months = pd.period_range("2000-01", periods=120, freq="M")
    quiet = rng.normal(0, 0.01, (120, 2)) @ np.array([[1.0, 0.8], [0.8, 1.0]]) ** 0.5
    loud = rng.normal(0, 0.04, (120, 2)) @ np.array([[1.0, 0.8], [0.8, 1.0]]) ** 0.5
    frame = pd.DataFrame(np.hstack([quiet, loud]), index=months, columns=["q0", "q1", "l0", "l1"])
    matrix = pd.DataFrame(np.cov(frame, rowvar=False, ddof=1), index=frame.columns, columns=frame.columns)
    book = construct_families.hierarchical_risk_parity(matrix, cap=1.0)

    def cluster_variance(block):
        block = np.asarray(block)
        sub = np.cov(block, rowvar=False, ddof=1)
        inverse = 1.0 / np.diag(sub)
        return float((inverse / inverse.sum()) @ sub @ (inverse / inverse.sum()))

    quiet_variance, loud_variance = cluster_variance(quiet), cluster_variance(loud)
    expected_quiet = loud_variance / (quiet_variance + loud_variance)
    assert float(book[["q0", "q1"]].sum()) == pytest.approx(expected_quiet, abs=1e-9)
    assert float(book[["l0", "l1"]].sum()) == pytest.approx(1.0 - expected_quiet, abs=1e-9)
    assert float(book.min()) > 0.0


def test_the_tail_program_reproduces_the_shortfall_it_minimises():
    """Expected shortfall is computed directly from the window's months, so the program has something
    independent to be checked against: on a planted window the direct computation is a hand value, and
    the optimiser's book cannot have a worse tail than any other feasible book."""
    months = pd.period_range("2000-01", periods=100, freq="M")
    one = pd.DataFrame({"a": np.concatenate([[-0.20], np.full(99, 0.01)])}, index=months)
    assert construct_families.expected_shortfall(np.ones(1), one, level=0.05) == pytest.approx(0.032, abs=1e-12)

    window = pd.DataFrame(
        {"a": np.concatenate([[-0.20], np.full(99, 0.005)]), "b": np.full(100, 0.005)},
        index=months,
    )
    matrix = pd.DataFrame(np.cov(window.to_numpy(), rowvar=False, ddof=1), index=["a", "b"], columns=["a", "b"])
    book = construct_families.mean_cvar(matrix, None, window, cap=1.0)
    assert float(book["b"]) == pytest.approx(1.0, abs=1e-6), "the tail is in one sleeve, so the programme leaves it"
    chosen = construct_families.expected_shortfall(book.to_numpy(), window.to_numpy(), level=0.05)
    draws = np.random.default_rng(1).dirichlet(np.ones(2), size=200)
    tails = [construct_families.expected_shortfall(candidate, window.to_numpy(), level=0.05) for candidate in draws]
    assert chosen <= min(tails) + 1e-12
    assert chosen < construct_families.expected_shortfall(np.full(2, 0.5), window.to_numpy(), level=0.05)


def test_the_no_mean_form_is_decided_by_the_constraint_set():
    """With a zero mean and the tightest norm a fully invested long-only book admits, the feasible set
    is a single point, so the equal-weight portfolio arrives without the mean entering anywhere. That
    is the family's no-mean cell, and it is a statement about the constraint set rather than about
    means."""
    rng = np.random.default_rng(10)
    names = list(universe.TICKERS)
    frame = pd.DataFrame(rng.normal(0, 0.02, (80, len(names))), columns=names)
    matrix = pd.DataFrame(np.cov(frame, rowvar=False, ddof=1), index=names, columns=names)
    budget = 1.0 / len(names) ** 0.5
    book = construct_families.mean_variance(matrix, np.zeros(len(names)), frame, norm=budget)
    assert book.to_numpy() == pytest.approx(np.full(len(names), 1.0 / len(names)), abs=1e-6)
    with pytest.raises(ValueError, match="admits no fully invested book"):
        construct_families.mean_variance(matrix, np.zeros(len(names)), frame, norm=0.1)


def test_black_litterman_reproduces_the_posterior_form_and_its_two_limits():
    """The module writes the posterior in its precision form; the test computes the covariance form,
    which is the algebraically equivalent expression, so the equivalence is verified on numbers. The
    two limits are the method's own claims: infinite view uncertainty returns the prior, and zero
    uncertainty reproduces the view exactly."""
    matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
    prior = np.array([0.02, 0.03])
    views = np.array([[1.0, -1.0]])
    view_returns = np.array([0.01])
    tau = 0.05
    for view_variance in (0.0004, 0.01, 0.005):
        view_covariance = np.array([[view_variance]])
        written = construct_means.posterior(matrix, prior, tau, views, view_returns, view_covariance)
        # pi + tau*Sigma*P' (P tau Sigma P' + Omega)^-1 (q - P pi), the same posterior in another form
        scaled = tau * matrix
        expected = prior + scaled @ views.T @ np.linalg.inv(views @ scaled @ views.T + view_covariance) @ (
            view_returns - views @ prior
        )
        assert written == pytest.approx(expected.ravel(), abs=1e-10)

    assert construct_means.posterior(matrix, prior, tau, views, view_returns, np.array([[1e12]])) == pytest.approx(prior, abs=1e-8)
    tight = construct_means.posterior(matrix, prior, tau, views, view_returns, np.array([[1e-14]]))
    assert (views @ tight).item() == pytest.approx(0.01, abs=1e-6)


def test_the_shrunk_mean_is_pulled_toward_the_target_and_has_a_smaller_error():
    """The estimator's own claim, planted: the sample mean of a window carries noise, the Bayes-Stein
    mean pulls it toward the mean the minimum-variance portfolio implies, and the pull leaves it
    closer to the truth. The intensity is returned rather than applied and forgotten, because it is
    estimated from the window and a small value is a statement about the window."""
    rng = np.random.default_rng(11)
    names = list(universe.TICKERS)
    truth = np.full(len(names), 0.004)
    window = pd.DataFrame(rng.normal(truth, 0.03, (60, len(names))), columns=names)
    matrix = pd.DataFrame(np.cov(window, rowvar=False, ddof=1), index=names, columns=names)
    vector, report = construct_means.jorion(window, matrix)
    plain = window.mean()
    assert 0.0 < report["intensity"] < 1.0
    assert float(((vector - truth) ** 2).sum()) < float(((plain - truth) ** 2).sum())
    assert float(vector.sub(report["target"]).abs().sum()) < float(plain.sub(report["target"]).abs().sum())
    assert float(vector.sub(plain).abs().sum()) == pytest.approx(
        report["intensity"] * float(plain.sub(report["target"]).abs().sum()), rel=1e-9
    )


def test_the_grid_is_pre_registered_at_twenty_runs_before_any_of_them_runs():
    """The count is the design's, not the table's: sixteen distinct cells, two perturbation runs and
    two repeats of the secondary protocol. A cell added later is an amendment to the registry, and a
    duplicated identifier is two runs' manifests keyed to one name."""
    declared = registry.validate()
    assert declared["pre_registered"] == 20
    assert (declared["cells"], declared["perturbed"], declared["repeated"]) == (16, 2, 2)
    assert len(set(declared["runs"])) == len(declared["runs"])
    stages = {run["stage"] for run in registry.RUNS}
    assert stages == {"A", "B", "C"}, "every stage of the design carries at least one run"
    # The pair read beside each other to report what the bounds do, and its uncapped half: both have to
    # be cells of the grid rather than runs the report goes looking for.
    assert set(registry.BOUNDS_PAIRS) <= {spec["id"] for spec in registry.CELLS}
    assert set(registry.BOUNDS_PAIRS.values()) <= {spec["id"] for spec in registry.CELLS}


# --------------------------------------------------------------- the evaluation harness


def planted_pair(periods=131, rho=0.95, seed=11):
    """Two monthly series at a planted correlation, on the out-of-sample calendar, with a zero
    benchmark so the active series is the series itself."""
    rng = np.random.default_rng(seed)
    index = pd.period_range("2015-09", periods=periods, freq="M")
    one = rng.normal(0, 0.03, periods)
    two = rho * one + np.sqrt(1.0 - rho ** 2) * rng.normal(0, 0.03, periods)
    flat = pd.Series(np.zeros(periods), index=index)
    return pd.Series(one, index=index), pd.Series(two, index=index), flat


def test_the_paired_test_quotes_its_resolution_at_the_bar_it_decides_at():
    """The design's numbers, re-derived from the paired statistic: the standard error of the annualised
    information-ratio difference is `sqrt(12/T) * sqrt(2(1-rho))` on the realised correlation, which is
    0.098 at a correlation of 0.95 over 131 months, and the smallest difference this test detects at
    eighty percent power is that standard error times the bar the row decided at plus the power
    quantile.

    The bar is the family-wise 2.9552 here, so the resolution is 0.36; the nominal two-sided 1.96 would
    give 0.27, and the assertion below is that the second is the finer of the two. A row whose verdict
    was printed by the family-wise bar may not quote the resolution of a test that decides at the
    nominal one, because the resolution's whole job is to qualify that verdict.
    """
    one, two, flat = planted_pair()
    bar = statistics.family_wise_bar(16)
    result = statistics.paired(two, one, flat, bar)
    realised = float(np.corrcoef(two, one)[0, 1])

    assert result["correlation"] == pytest.approx(realised, abs=1e-12)
    assert realised == pytest.approx(0.95, abs=0.03), "the plant has to land where the design's table is"
    assert result["standard_error"] == pytest.approx(
        np.sqrt(12 / len(one)) * np.sqrt(2 * (1 - realised)), abs=1e-12
    )
    assert result["standard_error"] == pytest.approx(0.098, rel=0.10)
    assert result["statistic"] == pytest.approx(result["difference"] / result["standard_error"], abs=1e-12)
    assert result["bar"] == pytest.approx(bar, abs=1e-12)
    assert result["resolution"] == pytest.approx(statistics.detection(bar) * result["standard_error"], abs=1e-12)
    assert result["resolution"] == pytest.approx(0.36, rel=0.10)
    assert statistics.detection(bar) == pytest.approx(bar + statistics.POWER_QUANTILE, abs=1e-12)
    assert statistics.detection(bar) > statistics.POWER, "the deciding bar is the conservative one"

    identical = statistics.paired(one, one.copy(), flat, bar)
    assert identical["degenerate"] is True and identical["statistic"] == 0.0
    assert statistics.sign_holds(-0.4, -0.7) is True and statistics.sign_holds(-0.4, 0.7) is False


def test_the_family_wise_bar_is_the_two_sided_level_and_the_adjusted_one_is_lower():
    """The bar the design fixes: sixteen cells tested at once put the null statistic's threshold at
    2.9552, the two-sided Bonferroni level, which is also the 95th percentile of the maximum of sixteen
    independent **absolute** standard normals. The one-sided quantile over the same cells is 2.7344, and
    the two readings cannot share a value: holding an absolute statistic against the one-sided number
    runs the family at 9.5% while the table is published as a 5% family-wise test. Positive correlation
    between the cells lowers the honest bar, and both are reported rather than only the flattering one."""
    bar = statistics.family_wise_bar(16)
    rng = np.random.default_rng(20260912)
    drawn = rng.standard_normal((200000, 16))
    signed = float(np.percentile(drawn.max(axis=1), 95))
    absolute = float(np.percentile(np.abs(drawn).max(axis=1), 95))

    assert bar == pytest.approx(2.9552, abs=1e-3)
    assert absolute == pytest.approx(bar, abs=0.05), "the drawn two-sided maximum is the bar"
    # The two readings, held against each other: a one-sided bar at twice the level **is** the one-sided
    # quantile at this level, which is the number an absolute statistic must not be read against.
    assert signed == pytest.approx(statistics.family_wise_bar(16, alpha=0.10), abs=0.01)
    assert statistics.family_wise_bar(16) > signed, "the bar is the stricter of the two readings"
    assert statistics.family_wise_bar(1) == pytest.approx(1.96, abs=1e-3), "one test is the two-sided 5% value"
    assert statistics.family_wise_bar(20) > bar, "more cells cannot buy a weaker bar"

    independent = np.eye(16)
    assert statistics.effective_tests(independent) == pytest.approx(16, abs=1e-9)
    correlated = 0.9 * np.ones((16, 16)) + 0.1 * np.eye(16)
    assert 1.0 < statistics.effective_tests(correlated) < 16.0, "correlated cells are not sixteen tests"
    assert 1.0 <= statistics.effective_tests(np.ones((16, 16))) < 2.0, (
        "a fully degenerate matrix is one test to within the estimator's own slack"
    )
    assert statistics.adjusted_bar(correlated) < bar, "the flattering bar is lower, which is why both print"
    assert statistics.adjusted_bar(np.eye(16)) == pytest.approx(bar, abs=1e-3)


def test_the_haircut_is_stated_as_self_imposed_and_the_retention_separates_a_leader_from_noise():
    """The two challenge diagnostics. The haircut is the family-wise bar, and the statement that it is
    self-imposed travels with it, because a self-imposed threshold reported as a requirement would
    borrow an authority the reading does not give it. The bootstrap resamples the months once and
    shares the draw across the cells, so a planted advantage keeps its rank and a panel of noise does
    not."""
    haircut = statistics.haircut(3.0, statistics.family_wise_bar(16))
    assert haircut["clears"] is True
    assert statistics.haircut(2.0, statistics.family_wise_bar(16))["clears"] is False
    assert "self-imposed" in haircut["statement"] and "not a regulatory requirement" in haircut["statement"]

    rng = np.random.default_rng(5)
    index = pd.period_range("2015-09", periods=131, freq="M")
    benchmark = pd.Series(rng.normal(0.002, 0.02, len(index)), index=index)
    noise = pd.DataFrame(
        {f"n{position}": benchmark + rng.normal(0, 0.01, len(index)) for position in range(6)}, index=index
    )
    noise["strong"] = benchmark + 0.006 + rng.normal(0, 0.002, len(index))
    measured = statistics.rank_retention(noise, benchmark, draws=400)
    assert measured["leader"] == "strong"
    assert measured["retention"] >= statistics.RANK_RETENTION_FLOOR
    assert measured["cells"]["strong"]["retention"] >= statistics.RANK_RETENTION_FLOOR
    assert measured["seed"] == cp.SEED, "the one documented seed is the one the draws come from"

    coin_flip = statistics.rank_retention(noise.drop(columns=["strong"]), benchmark, draws=400)
    assert coin_flip["retention"] < statistics.RANK_RETENTION_FLOOR, "noise cannot retain a ranking"
    rerun = statistics.rank_retention(noise, benchmark, draws=400)
    assert rerun["retention"] == measured["retention"], "the same seed reproduces the same resamples"


def test_the_metric_block_is_the_arithmetic_it_claims():
    """The primary metric, the drawdown, the cost sensitivity and the declared sub-periods, on inputs
    whose answers are known: the information ratio is the mean over the deviation annualised by the
    square root of twelve, the cost is charged on traded notional at the per-side rate, and the four
    sub-periods partition the out-of-sample window exactly."""
    index = pd.period_range("2015-09", periods=131, freq="M")
    values = pd.Series(np.sin(np.arange(131)), index=index) / 100.0
    assert metrics.information_ratio(values) == pytest.approx(values.mean() / values.std(ddof=1) * np.sqrt(12), abs=1e-12)
    assert metrics.information_ratio(pd.Series(np.zeros(131), index=index)) == 0.0
    assert metrics.drawdown(pd.Series([0.10, -0.20, 0.05])) == pytest.approx(-0.20, abs=1e-12)
    assert metrics.tracking_error(values) == pytest.approx(values.std(ddof=1) * np.sqrt(12), abs=1e-12)

    gross = pd.Series(np.linspace(-0.01, 0.02, 131), index=index)
    turnover = pd.Series(np.linspace(0.0, 0.04, 131), index=index)
    report = metrics.sensitivity(gross, turnover, pd.Series(np.zeros(131), index=index), bps=(5.0, 40.0))
    for rate in (5.0, 40.0):
        charged = gross - 2.0 * turnover * rate / 1e4
        assert report[rate]["net_cumulative"] == pytest.approx(float((1 + charged).prod() - 1), abs=1e-12)
        assert report[rate]["cost_annualised"] == pytest.approx(
            float(turnover.mean() * 2.0 * rate / 1e4 * 12), abs=1e-12
        )
    assert report[40.0]["cost_annualised"] > report[5.0]["cost_annualised"]

    windows = metrics.sub_periods(values)
    assert sum(entry["months"] for entry in windows.values()) == 131
    assert [entry["months"] for entry in windows.values()] == [52, 24, 24, 31]
    outside = metrics.sub_periods(pd.Series(np.ones(12), index=pd.period_range("1990-01", periods=12, freq="M")))
    assert all(entry["months"] == 0 and entry["information_ratio"] is None for entry in outside.values())


def test_a_rerun_agrees_within_the_stated_tolerance_and_a_moved_number_does_not():
    """A rerun is held to a stated tolerance rather than to bit-equality, because solvers differ in
    their last decimal and pretending otherwise would be a false acceptance criterion. The check has to
    fire on a number that moved for a reason and to pass on floating-point noise."""
    table = {
        "a": {"information_ratio": 0.544, "net_cumulative": 1.3339},
        "bar": {"family_wise": 2.9552, "adjusted": 2.5758},
    }
    assert comparison.reproduces(table, table)["agrees"] is True
    tiny = {"a": {"information_ratio": 0.544 + 1e-9, "net_cumulative": 1.3339}, "bar": table["bar"]}
    assert comparison.reproduces(table, tiny)["agrees"] is True
    moved = {"a": {"information_ratio": 0.5441, "net_cumulative": 1.3339}, "bar": table["bar"]}
    verdict = comparison.reproduces(table, moved)
    assert verdict["agrees"] is False and verdict["at"] == "a.information_ratio"
    assert verdict["tolerance"] == statistics.RERUN_TOLERANCE


def test_a_recommendation_claim_carries_the_bar_three_ingredients():
    """The fourth acceptance bar is bar 2 plus the haircut plus the constraint-binding frequency plus a
    positive cost-adjusted advantage, so a row that clears every rung records the claim with all four
    rather than leaving it as a verdict word. The binding frequency is the ingredient a reader cannot
    reconstruct from the rest: a book that is mostly the cap is a result about the constraint set, and
    the perturbations exist to show it. A row that is not a claim carries nothing rather than zeroes."""
    row = {
        "cell": "minimum_variance",
        "verdict": comparison.CANDIDATE,
        "paired_benchmark": {"statistic": 2.9},
        "haircut": {"bar": 2.9552, "statement": "self-imposed from the literature"},
        "information_ratio": 0.42,
        "cap_binding_frequency": 3.21,
        "cap_binding_steps": 131,
        "months": 131,
    }
    claim = comparison.claimed(row)
    assert claim["statistic"] == 2.9 and claim["bar"] == 2.9552
    assert claim["information_ratio_net"] == 0.42 and claim["cost_bp"] == constraints.COST_BP
    assert claim["cap_binding_frequency"] == 3.21 and claim["cap_binding_steps"] == 131
    assert claim["steps"] == 131 and "self-imposed" in claim["statement"]

    assert comparison.claimed({**row, "verdict": comparison.BEHIND}) is None
    assert comparison.claimed({**row, "verdict": comparison.NO_DIFFERENCE}) is None


def test_the_brinson_identity_holds_and_only_the_allocation_level_differs_from_bhb():
    """Two sleeves whose returns differ between the two books, because that is the only case in which
    the selection and interaction terms exist at all. The identity is the claim: allocation plus
    selection plus interaction is the active return exactly. Brinson-Hood-BeeBower is computed beside it
    for the one difference it actually carries - its allocation is not measured net of the benchmark's
    own return, so the two allocation columns differ by the weight deviation times that return while
    their totals agree term by term."""
    weights_p = pd.Series({"a": 0.50, "b": 0.30, "c": 0.20})
    weights_b = pd.Series({"a": 0.40, "b": 0.35, "c": 0.25})
    returns_p = pd.Series({"a": 0.020, "b": 0.050, "c": -0.010})
    returns_b = pd.Series({"a": 0.010, "b": 0.040, "c": 0.000})

    decomposition = brinson.single_period(weights_p, weights_b, returns_p, returns_b)
    active = float(weights_p @ returns_p) - float(weights_b @ returns_b)
    assert active == pytest.approx(0.005, abs=1e-15)
    assert decomposition["active"] == pytest.approx(active, abs=1e-15)
    assert sum(decomposition["totals"].values()) == pytest.approx(active, abs=1e-15)
    assert decomposition["residual"] == pytest.approx(0.0, abs=1e-15)
    assert decomposition["benchmark_total"] == pytest.approx(0.018, abs=1e-15)
    assert decomposition["totals"]["allocation"] == pytest.approx(-0.001, abs=1e-15)
    assert decomposition["totals"]["selection"] == pytest.approx(0.005, abs=1e-15)
    assert decomposition["totals"]["interaction"] == pytest.approx(0.001, abs=1e-15)

    # The two published forms share one interaction term and one set of totals; the only column that
    # moves is allocation, by exactly the weight deviation times the benchmark's own return.
    shift = (weights_p - weights_b) * decomposition["benchmark_total"]
    assert (decomposition["bhb_allocation"] - decomposition["allocation"]).to_numpy() == pytest.approx(
        shift.to_numpy(), abs=1e-15
    )
    assert decomposition["bhb_total"] == pytest.approx(decomposition["active"], abs=1e-15)
    assert float(decomposition["other"].sum()) == pytest.approx(decomposition["totals"]["interaction"])


def test_the_currency_lines_sum_and_measuring_allocation_in_euro_would_double_count():
    """One euro sleeve and one translating sleeve, with a stated move in the currency. The four lines
    sum to the active return exactly, selection is absent by structure, and the reason the allocation
    is measured in local currency is demonstrated rather than asserted: allocation taken on the euro
    return is already the whole active return, so adding a currency line to it counts the translation
    twice - which is invisible, because the lines still sum."""
    months = pd.period_range("2020-01", periods=3, freq="M")
    local = pd.DataFrame({"eur": [0.020, 0.010, -0.030], "sek": [0.050, -0.030, 0.010]}, index=months)
    translation = pd.DataFrame(
        {"eur": np.zeros(3), "sek": [0.040, -0.020, 0.030]}, index=months
    )
    split = {"local": local, "translation": translation, "euro": local + translation + local * translation}
    book = pd.DataFrame({"eur": [0.60, 0.50, 0.40], "sek": [0.40, 0.50, 0.60]}, index=months)
    benchmark = pd.Series({"eur": 0.50, "sek": 0.50})

    block = brinson.allocation_only(book, benchmark, split, pd.Series(0.0, index=months))
    deviation = book.sub(benchmark, axis=1)
    for month in months:
        lines = block["lines"].loc[month]
        assert lines[list(brinson.LINES)].sum() == pytest.approx(lines["active"], abs=1e-15)
        assert lines["residual"] == pytest.approx(0.0, abs=1e-15)
        # The translation is carried by the currency line and its cross term, and by nothing else.
        assert lines["currency"] == pytest.approx(
            float((deviation.loc[month] * translation.loc[month]).sum()), abs=1e-15
        )

    # Allocation taken on the euro return is the whole active return already; adding the currency line
    # to it then overstates the total by the translation the allocation has absorbed.
    euro_allocation = (deviation * split["euro"]).sum(axis=1)
    assert euro_allocation.to_numpy() == pytest.approx(block["lines"]["active"].to_numpy(), abs=1e-15)
    double_counted = euro_allocation + block["lines"]["currency"]
    assert abs(double_counted - block["lines"]["active"]).max() == pytest.approx(
        abs(block["lines"]["currency"]).max(), abs=1e-15
    )
    assert abs(block["lines"]["currency"]).max() > 0.0

    # The level the linking is measured against belongs to the frame the package reports in, and the
    # regression is that it is the benchmark's own: the euro leg, less the accrued cash rate, which is
    # what the grid's `gross` and `net` series are. Assembled from the local leg beside a euro active
    # series it still reconciles month by month - the lines sum either way - and leaves `gross` a level
    # belonging to no frame, which is the error the cost line reads directly.
    book_euro = (book * split["euro"]).sum(axis=1)
    benchmark_euro = (split["euro"] * benchmark).sum(axis=1)
    assert block["benchmark"].to_numpy() == pytest.approx(benchmark_euro.to_numpy(), abs=1e-15)
    assert brinson.decompose(book, benchmark, split, pd.Series(0.0, index=months))["gross"].to_numpy() == (
        pytest.approx(book_euro.to_numpy(), abs=1e-15)
    )

    # With a cash rate both levels are the excess ones, and the cost line then subtracts the charge the
    # caller's own net series carries rather than the difference between two accounts of one year.
    rate = pd.Series(0.001, index=months)
    excess = brinson.decompose(book, benchmark, split, rate)
    assert excess["gross"].to_numpy() == pytest.approx(book_euro.to_numpy() - 0.001, abs=1e-15)
    assert (excess["gross"] - excess["benchmark"]).to_numpy() == pytest.approx(
        excess["lines"]["active"].to_numpy(), abs=1e-15
    )
    charged = brinson.decompose(book, benchmark, split, rate, net=excess["gross"] - 0.0002)
    assert charged["cost"]["mean_monthly"] == pytest.approx(-0.0002, abs=1e-15)


def test_carino_reproduces_the_published_case_and_both_methods_land_on_the_compounded_excess():
    """The published two-month example first - coefficients 0.976 and 1.015, whole-run 0.991, linked
    allocation 0.80% - and then a self-consistent case, where the three lines do sum to the active
    return in every period. There both methods land on the compounded difference of the two growth
    rates rather than near it, which is the property the reconciliation rests on, and they land on it
    while distributing it differently: their per-line readings differ by more than the residual."""
    months = pd.period_range("2020-01", periods=2, freq="M")
    portfolio = pd.Series([0.03, -0.01], index=months)
    benchmark = pd.Series([0.02, -0.02], index=months)
    effects = pd.DataFrame({"allocation": [0.005, 0.003]}, index=months)

    published = brinson.carino(portfolio, benchmark, effects)
    assert published["coefficients"].to_numpy() == pytest.approx([0.9756, 1.0152], abs=1e-4)
    assert published["whole"] == pytest.approx(0.991, abs=1e-3)
    assert float(published["linked"]["allocation"]) == pytest.approx(0.008, abs=1e-4)

    lines = pd.DataFrame(
        {"allocation": [0.006, 0.004], "currency": [0.002, 0.001], "interaction": [0.002, 0.005]},
        index=months,
    )
    assert lines.sum(axis=1).to_numpy() == pytest.approx((portfolio - benchmark).to_numpy(), abs=1e-15)
    linked = brinson.linking(portfolio, benchmark, lines)
    assert linked["target"] == pytest.approx(
        float((1 + portfolio).prod() - (1 + benchmark).prod()), abs=1e-15
    )
    assert linked["carino"]["total"] == pytest.approx(linked["target"], abs=1e-15)
    assert linked["menchero"]["total"] == pytest.approx(linked["target"], abs=1e-15)
    assert abs(linked["residual"]) <= linked["tolerance"] * abs(linked["target"])
    assert linked["agreement"] > 100.0 * abs(linked["residual"])

    # A book that reproduces the benchmark leaves a 0/0 in both formulae; both take their limit and
    # return nothing rather than a NaN.
    flat = pd.Series(np.zeros(3), index=pd.period_range("2020-01", periods=3, freq="M"))
    zeros = pd.DataFrame({"allocation": np.zeros(3)}, index=flat.index)
    assert brinson.carino(flat, flat, zeros)["total"] == 0.0
    assert brinson.menchero(flat, flat, zeros)["total"] == 0.0


def test_the_euler_contributions_sum_to_volatility_and_a_var_decomposition_does_not():
    """Two assets and a covariance whose answer is known by hand: the contributions sum to the
    volatility exactly, which is the condition the budget is read under. Then the failing case that
    justifies refusing the other measure, on a window with one planted tail month: the conditional
    contributions a VaR report prints sum to the expected shortfall and exceed the value at risk, so
    reading them as VaR contributions overstates the quantity they decompose."""
    covariance = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=["a", "b"], columns=["a", "b"])
    weights = pd.Series({"a": 0.6, "b": 0.4})
    sigma = float(np.sqrt(weights @ covariance @ weights))
    table = euler.contributions(weights, covariance)
    assert table["contribution"].sum() == pytest.approx(sigma, rel=1e-15)
    assert table["share"].sum() == pytest.approx(1.0, rel=1e-15)
    assert table["marginal"].to_numpy() == pytest.approx((covariance @ weights).to_numpy() / sigma, abs=1e-15)
    assert euler.additivity(weights, covariance)["relative"] <= euler.ADDITIVITY_TOLERANCE

    months = pd.period_range("2000-01", periods=20, freq="M")
    window = np.zeros((20, 2))
    window[0, 0] = -0.20
    window[1:, 0] = 0.005
    window[:, 1] = 0.005
    tail = euler.tail_contributions(window, np.array([0.5, 0.5]), level=0.05)
    assert tail["tail_months"] == 1
    assert tail["sum"] == pytest.approx(tail["tail_mean"], abs=1e-15), "the contributions add to the tail mean"
    refused = euler.var_refusal(window, np.array([0.5, 0.5]), level=0.05)
    assert refused["contributions_sum"] == pytest.approx(refused["expected_shortfall"], abs=1e-15)
    assert refused["contributions_sum"] > refused["value_at_risk"], "a VaR decomposition does not add up"
    assert refused["overstatement"] == pytest.approx(
        refused["expected_shortfall"] - refused["value_at_risk"], abs=1e-15
    )


def test_the_factor_attribution_reconstructs_planted_returns_on_the_basis_it_fitted():
    """Three sleeves whose returns are exactly six factors plus an alpha, and a factor frame carrying
    the block's own column names, so no sleeve is the block's construction and every coefficient is
    estimated. The decomposition then has to reproduce the planted returns: the coefficients were fitted
    on the block net of its predecessors, the traded month is put on that basis by the window's own map,
    and a residual that is not zero is the basis having been lost between the two."""
    rng = np.random.default_rng(11)
    months = pd.period_range("2005-01", periods=90, freq="M")
    columns = ["Mkt-RF", "HML", *list(spine.CONSTRUCTED)]
    factors = pd.DataFrame(rng.normal(0.001, 0.02, (90, len(columns))), index=months, columns=columns)
    truth = {"s1": (0.001, [1.0, 0.2, 0.5, 0.3, -0.2, 0.1]), "s2": (-0.002, [0.3, -0.5, 1.0, 0.0, 0.4, 0.2]),
             "s3": (0.0, [0.1, 0.9, -0.3, 0.6, 0.1, -0.4])}
    returns = pd.DataFrame(
        {sleeve: alpha + factors.to_numpy() @ np.asarray(beta) for sleeve, (alpha, beta) in truth.items()},
        index=months,
    )
    fit = exposures.rolling(returns, factors, months)
    assert fit["determined"][0] == [], "no sleeve here is the block's own construction"
    assert len(fit["traded"]) == 90 - exposures.WINDOW

    book = pd.DataFrame(
        {"s1": 0.5, "s2": 0.3, "s3": 0.2}, index=fit["traded"]
    )
    composition = factor_attribution.decomposition(book, returns, fit, factors)
    assert composition["residual"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert composition["explained"].to_numpy() == pytest.approx(composition["active"].to_numpy(), abs=1e-15)
    assert composition["total"].to_numpy() == pytest.approx(composition["active"].to_numpy(), abs=1e-15)
    # The contribution of every factor is its exposure times its return, on the basis the fit used.
    assert composition["factors"].to_numpy().sum() == pytest.approx(
        float(composition["explained"].sum() - composition["alpha"].sum()), abs=1e-15
    )

    risk = factor_attribution.risk_attribution(book, fit, factors)
    assert risk["total_tracking_error"] == pytest.approx(
        np.sqrt(risk["variance"]["total"].mean() * metrics.PERIODS_PER_YEAR), abs=1e-15
    )
    assert risk["factor_share"] == pytest.approx(1.0, abs=1e-9), "a model with no residual noise is all factor"


def test_the_cross_view_residual_is_zero_by_construction_and_fires_on_a_misaligned_view():
    """Both views decompose the same active return, so their gap is zero and the check is a definition
    rather than a tolerance. It is kept because it fires the moment one view reads a different month,
    which is the failure a passing identity is supposed to rule out - and a view shifted by one month
    has to fail it."""
    index = pd.period_range("2020-01", periods=12, freq="M")
    holding = pd.Series(np.linspace(-0.01, 0.02, 12), index=index)
    view = factor_attribution.cross_view(holding, holding)
    assert view["worst"] == 0.0

    shifted = holding.shift(1).fillna(0.0)
    moved = factor_attribution.cross_view(holding, shifted)
    assert moved["reconciles"] is False and moved["worst"] > 0.0
    assert moved["relative"] > moved["tolerance"]

    # A view that stops short of the other's months is refused rather than read as a zero residual for
    # the missing ones.
    with pytest.raises(ValueError):
        factor_attribution.cross_view(holding, holding.iloc[:-1])


def test_ex_ante_and_ex_post_tracking_error_are_the_two_objects_they_name():
    """Two assets with round numbers: the ex-ante figure is the covariance form annualised, the ex-post
    figure is the realised dispersion of the active series, and the difference between them is reported
    as its own number rather than left to a reader to subtract. An active book with no tracking error
    reports a ratio of nothing rather than a zero."""
    covariance = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["a", "b"], columns=["a", "b"])
    active_weights = pd.Series({"a": 0.2, "b": -0.2})
    ex_ante = euler.ex_ante_tracking_error(active_weights, covariance)
    assert ex_ante == pytest.approx(np.sqrt(0.04 * 0.04 + 0.01 * 0.04) * np.sqrt(12), rel=1e-15)

    index = pd.period_range("2020-01", periods=24, freq="M")
    active = pd.Series(np.sin(np.arange(24)) / 100.0, index=index)
    ex_post = euler.ex_post_tracking_error(active)
    assert ex_post == pytest.approx(float(active.std(ddof=1) * np.sqrt(metrics.PERIODS_PER_YEAR)), rel=1e-12)
    quality = euler.forecast_quality(ex_ante, ex_post)
    assert quality["difference"] == pytest.approx(ex_post - ex_ante, abs=1e-15)
    assert quality["ratio"] == pytest.approx(ex_post / ex_ante, rel=1e-12)
    assert euler.forecast_quality(0.0, ex_post)["ratio"] is None
