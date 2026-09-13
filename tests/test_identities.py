"""Hand-calculated identities, on hand-built or planted inputs.

Every check here is arithmetic that has to come out exactly, which is why the frames are planted
rather than loaded: if one of these fails, the code is wrong rather than the market.
"""

import numpy as np
import pandas as pd
import pytest

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
