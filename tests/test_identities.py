"""Hand-calculated identities, on hand-built or planted inputs.

Every check here is arithmetic that has to come out exactly, which is why the frames are planted
rather than loaded: if one of these fails, the code is wrong rather than the market.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.compare import registry
from portfolio_workbench.construct import constraints
from portfolio_workbench.construct import families as construct_families
from portfolio_workbench.construct import means as construct_means
from portfolio_workbench.data import universe
from portfolio_workbench.factors import components as cp
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
