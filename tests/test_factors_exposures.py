"""The rolling regression: what it recovers, what it refuses, and the identities it satisfies."""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.factors import exposures

CALENDAR = pd.period_range("2000-01", periods=140, freq="M")
FACTORS = ["f1", "f2"]


def planted(alpha=0.0, beta=(1.0, -0.5), noise=0.0, seed=3, periods=140):
    """A sleeve frame whose exposures are known, so recovery can be checked rather than assumed."""
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(rng.normal(0, 0.03, (periods, 2)), index=CALENDAR[:periods], columns=FACTORS)
    returns = pd.DataFrame(
        alpha + factors.to_numpy() @ np.asarray(beta) + rng.normal(0, noise, periods),
        index=factors.index,
        columns=["sleeve_a"],
    )
    return returns, factors


def test_the_regression_recovers_the_planted_exposures():
    returns, factors = planted(alpha=0.004, beta=(1.5, -0.75))
    fit = exposures.regress(returns, factors)
    assert fit["alpha"][0] == pytest.approx(0.004, abs=1e-12)
    assert fit["beta"][:, 0] == pytest.approx([1.5, -0.75], abs=1e-12)
    assert fit["resid_var"][0] == pytest.approx(0.0, abs=1e-20)
    assert fit["r2"][0] == pytest.approx(1.0, abs=1e-12)


def test_the_factor_and_idiosyncratic_parts_sum_to_the_sleeve_variance():
    """The split is exact for variance, which is why the report prints shares of variance: the two
    square roots do not add. Checked on the sample variance of the same window."""
    returns, factors = planted(noise=0.01)
    fit = exposures.regress(returns, factors)
    total = float(returns["sleeve_a"].var(ddof=1))
    assert fit["explained_var"][0] + fit["resid_var"][0] == pytest.approx(total, rel=1e-12)
    factor_part, idio_part = exposures.factor_split([1.0], fit["beta"], fit["factor_cov"], fit["resid_var"])
    assert factor_part + idio_part == pytest.approx(total, rel=1e-12)


def test_windows_end_before_the_month_they_trade_and_span_the_declared_calendar():
    calendar = pd.period_range("2010-09", "2026-07", freq="M")
    steps = list(exposures.windows(calendar))
    assert len(steps) == len(calendar) - exposures.WINDOW
    for traded, window in steps:
        assert window[-1] == traded - 1, "the estimate at the close of t cannot read t's own bar"
        assert len(window) == exposures.WINDOW
    assert steps[0][0] == pd.Period("2015-09", freq="M"), "the decided first out-of-sample month"


def test_a_window_with_a_hole_in_it_is_refused_rather_than_regressed():
    calendar = pd.period_range("2010-09", "2026-07", freq="M")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, (len(calendar), 1)), index=calendar, columns=["sleeve_a"])
    factors = pd.DataFrame(rng.normal(0, 0.01, (len(calendar), 2)), index=calendar, columns=FACTORS)
    fit = exposures.rolling(returns, factors, calendar)
    assert fit["n_obs"][0] == exposures.WINDOW, "a complete frame leaves no absence to excuse"
    partial = exposures.rolling(returns.iloc[1:], factors.iloc[1:], calendar)
    assert partial["n_obs"][0] == exposures.MIN_OBS, "the panel's own first bar carries no return"

    with pytest.raises(ValueError, match="different months"):
        exposures.rolling(returns.iloc[1:], factors, calendar)

    holed = returns.copy()
    holed.loc[pd.Period("2012-03", freq="M"), "sleeve_a"] = float("nan")
    with pytest.raises(ValueError, match="2012-03"):
        exposures.rolling(holed, factors, calendar)


def test_a_fixed_loading_run_differs_where_the_exposure_drifts():
    """The contrast exists because the study is about estimation error: a sleeve whose exposure
    moves has an alpha the fixed run cannot see, and the difference is the measurement."""
    rng = np.random.default_rng(5)
    calendar = pd.period_range("2000-01", periods=140, freq="M")
    factor = pd.DataFrame({"f1": rng.normal(0, 0.03, 140)}, index=calendar)
    drifting = np.where(np.arange(140) < 70, 1.0, 3.0)
    returns = pd.DataFrame(
        {"sleeve_a": factor["f1"].to_numpy() * drifting + rng.normal(0, 0.001, 140)}, index=calendar
    )
    fit = exposures.rolling(returns, factor, calendar)
    fixed = exposures.fixed_alpha(returns, factor, fit["beta"], calendar)
    difference = (fit["alpha"]["sleeve_a"] - fixed["sleeve_a"]).abs()
    assert float(difference.max()) > 1e-3, "a drifting exposure must show up as a difference"
    estimated = fit["beta"]["f1"]["sleeve_a"]
    assert float(estimated.max() - estimated.min()) > 1.5, "and the loading path is where it shows up"


def test_a_sleeve_spanned_by_the_factors_is_named_rather_than_estimated():
    """The sleeve is an exact combination of the factors, so its exposures are the definition
    restated. The run must say so rather than report a mechanical coefficient as a measurement."""
    rng = np.random.default_rng(9)
    factors = pd.DataFrame(rng.normal(0, 0.02, (120, 2)), index=CALENDAR[:120], columns=FACTORS)
    spanned = factors @ np.array([1.0, -0.25])
    returns = pd.DataFrame({"spanned": spanned, "own": spanned + rng.normal(0, 0.01, 120)})
    fit = exposures.rolling(returns, factors, factors.index)
    assert exposures.spanned_sleeves(fit) == ["spanned"]
