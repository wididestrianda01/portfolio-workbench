"""The split boundaries: what the walk-forward may see, and what is refused rather than solved.

Each check here is about a line rather than a value: the month an estimate may not read, the month a
window may not be missing, the sign a component carries between refits, the movement that falsifies a
count, and the inputs whose statistic would be arithmetic on nothing.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.factors import components as cp
from portfolio_workbench.factors import exposures, spanning, spine
from portfolio_workbench.risk import covariance

CALENDAR = pd.period_range("2010-09", "2026-07", freq="M")


def planted(periods=191, seed=1, series=11):
    """A sleeve frame with a named set beside it: the published spine, then the block's four series."""
    rng = np.random.default_rng(seed)
    months = CALENDAR[:periods]
    frame = pd.DataFrame(
        rng.normal(0, 0.01, (periods, series)),
        index=months,
        columns=[f"s{position}" for position in range(series)],
    )
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (periods, 2 + len(exposures.BLOCK))),
        index=months,
        columns=["Mkt-RF", "SMB", *exposures.BLOCK],
    )
    return frame, factors


def test_the_walk_forward_window_ends_before_the_month_it_trades():
    """The estimate formed at the close of t cannot read t's own bar, the calendar the windows are cut
    on is the panel's own, and the panel's first bar has no return, so the earliest window carries one
    observation fewer and no other window is excused."""
    steps = list(exposures.windows(CALENDAR))
    assert len(steps) == len(CALENDAR) - exposures.WINDOW
    for traded, window in steps:
        assert window[-1] == traded - 1
        assert len(window) == exposures.WINDOW
    assert steps[0][0] == pd.Period("2015-09", freq="M"), "the decided first out-of-sample month"

    returns, factors = planted()
    fit = exposures.rolling(returns, factors, CALENDAR)
    assert set(fit["n_obs"]) == {exposures.WINDOW}, "a complete frame leaves no absence to excuse"
    partial = exposures.rolling(returns.iloc[1:], factors.iloc[1:], CALENDAR)
    assert partial["n_obs"][0] == exposures.MIN_OBS, "the panel's first bar carries no return"
    assert set(partial["n_obs"][1:]) == {exposures.WINDOW}


def test_a_window_with_a_hole_is_refused_while_the_panel_open_is_excused():
    """One observation may be absent and only at the front; an interior gap is a data fault, and the
    two frames a regression consumes have to cover the same months or the fit compares wrong rows."""
    returns, factors = planted()
    exposures.rolling(returns, factors, CALENDAR)                              # the frame is complete

    holed = returns.copy()
    holed.loc[pd.Period("2012-03", freq="M"), "s0"] = float("nan")
    with pytest.raises(ValueError, match="2012-03"):
        exposures.rolling(holed, factors, CALENDAR)

    with pytest.raises(ValueError, match="different months"):
        exposures.rolling(returns.iloc[1:], factors, CALENDAR)

    with pytest.raises(ValueError, match="below the"):
        exposures.rolling(returns.iloc[2:], factors.iloc[2:], CALENDAR)


def test_the_sign_convention_is_reapplied_at_every_refit():
    """Eigenvectors are defined up to sign, so without the convention a component can invert between
    two adjacent refits and everything read off it flips with it."""
    rng = np.random.default_rng(4)
    loadings = rng.normal(0, 1, (6, 2))
    oriented = cp.orient(loadings)
    for column in range(oriented.shape[1]):
        assert oriented[:, column][np.argmax(np.abs(oriented[:, column]))] > 0
    assert np.allclose(np.abs(oriented), np.abs(loadings)), "orienting is a sign change, nothing more"


def test_the_count_rule_recovers_planted_structure_and_invents_none_in_noise():
    """The rule's two failure modes, planted on both sides: it must find the common factors that are
    there, and it must not report the eigenvalues of a noise panel as structure."""
    rng = np.random.default_rng(7)
    months = pd.period_range("2000-01", periods=300, freq="M")
    common = rng.normal(0, 0.03, (300, 3)) @ rng.normal(0, 1, (3, 11)) + rng.normal(0, 0.01, (300, 11))
    assert cp.decompose(pd.DataFrame(common, index=months, columns=[f"s{p}" for p in range(11)]), draws=60)["components"] == 3

    noise = pd.DataFrame(rng.normal(0, 1, (300, 11)), index=months, columns=[f"s{p}" for p in range(11)])
    decomposition = cp.decompose(noise, draws=60)
    assert decomposition["components"] <= 1
    assert decomposition["eigenvalues"][0] < decomposition["threshold"][0] * 1.05


def test_the_falsification_boundary_is_a_move_of_more_than_one():
    """A count that jumps by more than one component between adjacent steps, in more than a quarter of
    them, is a rule choosing the answer as much as measuring it: the fallback uses the fixed count and
    reports both. A move of exactly one is not a jump."""
    steady = cp.decide([2, 2, 3, 3, 2, 2])
    assert steady["falsified"] is False and steady["move_share"] == 0.0
    assert list(steady["counts"]) == [2, 2, 3, 3, 2, 2]

    jumping = cp.decide([1, 3, 1, 3, 1, 3])
    assert jumping["falsified"] is True
    assert set(jumping["counts"]) == {cp.PREREGISTERED_K}
    assert list(jumping["mechanical"]) == [1, 3, 1, 3, 1, 3]


def test_a_degenerate_input_is_refused_rather_than_solved():
    """Four ways an answer would be arithmetic on nothing: a sleeve the frame does not carry, a factor
    set with a hole in it, a test asset that is a copy of another, and a benchmark that reproduces one
    exactly. Each is refused with its reason instead of returning a number."""
    months = pd.PeriodIndex(["2020-01", "2020-02"], freq="M")
    sleeves = pd.DataFrame(
        {"IBGL.AS": [0.04, -0.02], "IEGE.AS": [0.01, 0.01], "IEAC.AS": [0.05, -0.01], "IHYG.L": [0.09, 0.00]},
        index=months,
    )
    with pytest.raises(ValueError, match="IHYG.L"):
        spine.constructed_block(sleeves.drop(columns=["IHYG.L"]))

    published = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": [0.001, 0.001]}, index=months)
    holed = published.copy()
    holed.loc[months[1], "Mkt-RF"] = float("nan")
    with pytest.raises(ValueError, match="missing factor value"):
        spine.named_set(holed, spine.constructed_block(sleeves))

    rng = np.random.default_rng(2)
    factors = pd.DataFrame(rng.normal(0, 0.03, (180, 2)), index=pd.period_range("2000-01", periods=180, freq="M"))
    assets = pd.DataFrame(
        factors.to_numpy() @ np.outer([0.5, 0.5], np.ones(2)) + rng.normal(0, 1e-4, (180, 2)),
        index=factors.index,
        columns=["t0", "t1"],
    )
    duplicated = assets.copy()
    duplicated["t1"] = duplicated["t0"] * (1.0 + 1e-15)
    with pytest.raises(ValueError, match="cannot be inverted"):
        spanning.grs(duplicated, factors)

    exact = pd.DataFrame(factors.to_numpy() @ np.outer([0.5, 0.5], np.ones(2)), index=factors.index, columns=["t0", "t1"])
    with pytest.raises(ValueError, match="cannot be inverted"):
        spanning.grs(exact, factors)

    with pytest.raises(ValueError, match="repeated instrument"):
        covariance.sample(pd.DataFrame(rng.normal(0, 0.01, (30, 3)), columns=["a", "a", "b"]))
