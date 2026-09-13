"""Spanning: the identity it must reproduce, the results it must refuse, and the illustration."""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.factors import spanning

MONTHS = pd.period_range("2000-01", periods=180, freq="M")


def planted(assets=3, benchmarks=2, seed=2, intercept=0.0, noise=0.0):
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(
        rng.normal(0, 0.03, (len(MONTHS), benchmarks)),
        index=MONTHS,
        columns=[f"b{position}" for position in range(benchmarks)],
    )
    weights = np.full(benchmarks, 1.0 / benchmarks)
    values = intercept + factors.to_numpy() @ np.outer(weights, np.ones(assets))
    values = values + rng.normal(0, noise, values.shape)
    test_assets = pd.DataFrame(values, index=MONTHS, columns=[f"t{position}" for position in range(assets)])
    return test_assets, factors


def test_a_spanned_set_is_not_rejected_and_its_loadings_sum_to_one():
    """The Huberman-Kandel identity, with the estimation noise every real fit carries: an asset that
    is a portfolio of the benchmark leaves a row of loadings summing to one and no detectable
    intercept. The statistic is F-distributed under a spanned null, so it sits near one rather than
    at zero - a zero would mean no residual at all, which is the refused case below."""
    test_assets, factors = planted(noise=1e-5)
    result = spanning.grs(test_assets, factors)
    assert result["rows_sum"] == pytest.approx(np.ones(len(test_assets.columns)), abs=1e-3)
    assert result["p_value"] > 0.05, "a spanned benchmark must not be rejected"


def test_an_exactly_reproduced_test_asset_is_refused_rather_than_reported():
    """No noise, no residual, nothing to invert: the arithmetic has no answer here and a zero would
    read as a spanning verdict."""
    test_assets, factors = planted()
    with pytest.raises(ValueError, match="cannot be inverted"):
        spanning.grs(test_assets, factors)


def test_a_planted_intercept_is_rejected():
    test_assets, factors = planted(intercept=0.01, noise=0.001)
    result = spanning.grs(test_assets, factors)
    assert result["p_value"] < 1e-4
    assert np.abs(result["alpha"]).min() > 0.005


def test_the_statistic_is_invariant_to_an_invertible_transform_of_the_test_assets():
    """A property of the test, not of the data: rescaling or rotating the test assets cannot change
    the answer, which is why the components' scale is a presentation choice."""
    test_assets, factors = planted(intercept=0.004, noise=0.002)
    transform = np.array([[1.5, 0.4, 0.0], [0.2, -1.0, 0.7], [0.0, 0.3, 2.0]])
    rotated = pd.DataFrame(test_assets.to_numpy() @ transform, index=MONTHS, columns=test_assets.columns)
    plain = spanning.grs(test_assets, factors)
    moved = spanning.grs(rotated, factors)
    assert moved["statistic"] == pytest.approx(plain["statistic"], rel=1e-9)
    assert moved["p_value"] == pytest.approx(plain["p_value"], rel=1e-9)


def test_a_degenerate_residual_covariance_is_refused():
    """One test asset that is a copy of another leaves a residual covariance that cannot be
    inverted, and the statistic is undefined rather than large."""
    test_assets, factors = planted(assets=2, noise=1e-4)
    degenerate = test_assets.copy()
    degenerate["t1"] = degenerate["t0"] * (1.0 + 1e-15)
    with pytest.raises(ValueError, match="cannot be inverted"):
        spanning.grs(degenerate, factors)


def test_the_component_portfolios_are_built_from_raw_returns_not_demeaned_scores():
    """A demeaned series has mean zero by construction, which would hand the test an intercept that
    is an artefact of the demeaning. The portfolios carry the returns' own mean."""
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        0.004 + rng.normal(0, 0.02, (200, 6)),
        index=pd.period_range("2000-01", periods=200, freq="M"),
        columns=[f"s{position}" for position in range(6)],
    )
    portfolios = spanning.component_module.component_portfolios(frame, count=2)
    assert float(portfolios.mean().abs().min()) > 1e-4

    demeaned = portfolios - portfolios.mean()
    named = pd.DataFrame(rng.normal(0, 0.02, (200, 4)), index=frame.index, columns=list("abcd"))
    assert spanning.grs(portfolios, named)["statistic"] != pytest.approx(
        spanning.grs(demeaned, named)["statistic"], rel=1e-6
    )


def test_the_time_series_premium_is_the_hand_calculation():
    factors = pd.DataFrame(
        {"f": np.linspace(-0.02, 0.04, 60)},
        index=pd.period_range("2000-01", periods=60, freq="M"),
    )
    stats = spanning.premiums(factors)["f"]
    assert stats["t"] == pytest.approx(
        float(factors["f"].mean() / (factors["f"].std(ddof=1) / np.sqrt(60))), rel=1e-12
    )
    assert stats["months"] == 60


def test_the_cross_sectional_illustration_reports_its_own_identification():
    """Eleven sleeves against eleven second-stage parameters fits every month exactly. The label
    rests on that computed number rather than on a caveat in prose."""
    rng = np.random.default_rng(8)
    months = pd.period_range("2000-01", periods=120, freq="M")
    returns = pd.DataFrame(rng.normal(0, 0.01, (120, 11)), index=months)
    factors = pd.DataFrame(rng.normal(0, 0.01, (120, 10)), index=months)
    result = spanning.fama_macbeth(returns, factors)
    assert result["identified"] is True
    assert result["residual_dof"] == 0
