"""The split boundaries: what the walk-forward may see, and what is refused rather than solved.

Each check here is about a line rather than a value: the month an estimate may not read, the month a
window may not be missing, the sign a component carries between refits, the movement that falsifies a
count, and the inputs whose statistic would be arithmetic on nothing.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.compare import grid, registry
from portfolio_workbench.data import panel
from portfolio_workbench.evaluate import walkforward
from portfolio_workbench.construct import constraints
from portfolio_workbench.data import universe
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


# --------------------------------------------------------------- the construct boundary

CALENDAR_FULL = pd.period_range("2010-09", "2026-07", freq="M")


def planted_panel(periods=75, seed=12):
    """A planted sleeve frame carrying the real sleeve names, so the grid's rules can be exercised
    without the frozen snapshot: the policy weights and the sleeve map are what the runner reads."""
    rng = np.random.default_rng(seed)
    names = list(universe.TICKERS)
    months = CALENDAR_FULL[:periods]
    frame = pd.DataFrame(rng.normal(0.002, 0.02, (periods, len(names))), index=months, columns=names)
    return frame, months


def test_the_expanding_protocol_trades_the_same_months_and_still_cannot_look_ahead():
    """The secondary protocol answers whether a result is an artefact of a fixed window length, so it
    has to trade the same months with a longer estimate. The boundary is unchanged: an estimate formed
    at the close of t-1 cannot read t's own bar, and the window grows from the decided length."""
    rolling = list(exposures.windows(CALENDAR))
    expanding = list(exposures.expanding_windows(CALENDAR))
    assert [traded for traded, _ in expanding] == [traded for traded, _ in rolling]
    assert expanding[0][1][0] == CALENDAR[0] and len(expanding[0][1]) == exposures.WINDOW
    assert len(expanding[-1][1]) == len(CALENDAR) - 1
    for traded, window in expanding:
        assert window[-1] == traded - 1


def test_every_measured_rebalance_respects_the_constraint_set_on_a_planted_panel():
    """The rules the grid applies between a target and a book: the band, the cap it must not exceed,
    and the funding adjustment that closes the band's residual. Read off the run itself rather than
    off the rules in isolation, because the failure this defends against is a rule that is correct
    alone and applied in the wrong order."""
    returns, months = planted_panel()
    for identifier in ("minimum_variance", "hierarchical_risk_parity", "mean_cvar"):
        spec = next(run for run in registry.RUNS if run["id"] == identifier)
        result = grid.run_cell(spec, returns, months)
        books = result["weights"]
        assert len(books) == len(months) - exposures.WINDOW
        assert float(books.to_numpy().min()) >= -1e-12
        for _, book in books.iterrows():
            assert float(book.sum()) == pytest.approx(1.0, abs=1e-9), f"{identifier} left a book not fully invested"
            assert float(book.max()) <= constraints.CAP + 1e-12, f"{identifier} breached the per-sleeve cap"
            assert float(book.min()) >= 0.0, f"{identifier} went short"
        measured = result["turnover"].iloc[1:]
        assert len(measured) == len(books) - 1
        assert float(measured.max()) <= constraints.TURNOVER_CAP + 1e-12, f"{identifier} breached the turnover cap"
        assert result["summary"]["establishment"]["cost"] > 0.0, "the establishment trade is reported per cell"
        assert float(result["turnover"].iloc[0]) == 0.0, "the funded book is not a measured rebalance"


def test_the_turnover_cap_binding_is_recorded_with_the_trade_it_capped():
    """A cap that binds silently is a rule applied without a record, and a flag that reports binding
    while the trade sits under the cap is a record of nothing. The two are read together: a flagged
    rebalance trades exactly the cap, an unflagged one trades less, and neither exceeds it."""
    returns, months = planted_panel()
    for identifier in ("minimum_variance", "mean_cvar", "maximum_diversification"):
        spec = next(run for run in registry.RUNS if run["id"] == identifier)
        result = grid.run_cell(spec, returns, months)
        for binding, turnover in zip(result["turnover_binding"].iloc[1:], result["turnover"].iloc[1:]):
            assert float(turnover) <= constraints.TURNOVER_CAP + 1e-12, f"{identifier} breached the cap"
            if binding:
                assert float(turnover) == pytest.approx(constraints.TURNOVER_CAP, abs=1e-12), (
                    f"{identifier} reports the cap binding without trading it"
                )
            else:
                assert float(turnover) < constraints.TURNOVER_CAP + 1e-12


def test_the_cost_is_charged_to_the_net_series_on_traded_notional():
    """The convention the layer reports on: gross is the book's own return and net is gross less two
    times the one-way turnover at the per-side rate, with the establishment trade outside both because
    it is funded before the measured window opens. Computed the other way round - the cost added to the
    gross - every cell's net series is the cost-free one and its cost arrives as a gain, which reads
    the cost question backwards: the number separating the cells is the same order as what this charges.
    """
    returns, months = planted_panel()
    for identifier in ("minimum_variance", "mean_cvar", "mean_variance_sample"):
        spec = next(run for run in registry.RUNS if run["id"] == identifier)
        result = grid.run_cell(spec, returns, months)
        raw = (result["weights"].to_numpy() * returns.loc[result["traded"]].to_numpy()).sum(axis=1)
        charged = 2.0 * result["turnover"].to_numpy() * constraints.COST_BP / 1e4

        assert np.allclose(result["gross"], raw), f"{identifier}: the gross series is not the book's return"
        assert np.allclose(result["net"], raw - charged), f"{identifier}: the cost is not charged to net"
        assert charged[0] == 0.0, "the funded book is not a measured rebalance and pays no cost"
        assert float(result["summary"]["net_cumulative"]) < float(result["summary"]["gross_cumulative"])


def test_the_estimation_error_diagnostics_are_read_off_the_target_path():
    """Weight concentration is one of the two proxies for estimation error, so it is measured before
    the no-trade band and the turnover cap move the book away from its target: read off the traded
    books it answers a question about the trading rules instead, and the cell's concentration then moves
    with the band rather than with the estimator under test. Both paths are reported, so the gap between
    them is visible as what the rules cost rather than hidden inside the number."""
    returns, months = planted_panel()
    for identifier in ("minimum_variance", "mean_variance_sample"):
        spec = next(run for run in registry.RUNS if run["id"] == identifier)
        result = grid.run_cell(spec, returns, months)
        target_count = float(np.mean(1.0 / (result["targets"].to_numpy() ** 2).sum(axis=1)))
        traded_count = float(np.mean(1.0 / (result["weights"].to_numpy() ** 2).sum(axis=1)))

        assert result["summary"]["concentration"] == pytest.approx(target_count), (
            f"{identifier}: the reported concentration is not the target path's"
        )
        assert result["summary"]["concentration_traded"] == pytest.approx(traded_count)
        assert target_count != pytest.approx(traded_count), (
            f"{identifier}: the two paths agree, so this plant cannot tell them apart"
        )


# --------------------------------------------------------------- the walk-forward engine


def test_the_engine_records_every_step_and_refuses_a_planted_look_ahead():
    """The boundary is a record checked on the whole run rather than an intention held step by step,
    so it can be shown to have teeth: a window that reaches the month it trades, a window that stops
    short of the last bar that month could have read, a row missing anywhere but the panel's own front
    bar, an availability that does not follow from the bar it names, and a window answering two
    different component counts are each refused with the month named."""
    records = walkforward.steps(CALENDAR)
    for record in records:
        record["observations"] = record["months"] - 1 if record["window_start"] == CALENDAR[0] else record["months"]
    report = walkforward.assert_no_look_ahead(records, CALENDAR)
    assert report["steps"] == len(CALENDAR) - exposures.WINDOW
    assert report["first_traded"] == "2015-09" and report["last_traded"] == "2026-07"
    assert all(record["available_from"] == panel.available_from(record["window"])[-1] for record in records)
    assert all(record["gated_by"] == record["traded"] - 1 for record in records)

    reaching = list(records)
    reaching[3] = {**reaching[3], "window_end": reaching[3]["traded"]}
    with pytest.raises(ValueError, match="inside its own estimation window"):
        walkforward.assert_no_look_ahead(reaching, CALENDAR)

    short = list(records)
    short[3] = {**short[3], "window_end": short[3]["traded"] - 2}
    with pytest.raises(ValueError, match="stops at"):
        walkforward.assert_no_look_ahead(short, CALENDAR)

    holed = list(records)
    holed[3] = {**holed[3], "observations": holed[3]["months"] - 1}
    with pytest.raises(ValueError, match="starts at"):
        walkforward.assert_no_look_ahead(holed, CALENDAR)

    misgated = list(records)
    misgated[3] = {**misgated[3], "available_from": misgated[3]["available_from"] - pd.Timedelta(days=1)}
    with pytest.raises(ValueError, match="became readable"):
        walkforward.assert_no_look_ahead(misgated, CALENDAR)

    for record in records:
        record["components"] = 2
    assert walkforward.assert_no_look_ahead(records, CALENDAR)["components"] == [2]


def test_the_engine_refuses_a_window_that_reaches_its_own_month_as_it_hands_the_rows_out():
    """The cheap half of the check runs before an optimiser is called: a mis-cut step cannot first
    produce a plausible book and be caught afterwards. The window's rows are cut by the factor layer's
    own rule, so an estimate reads exactly what the exposures read for the same month - the panel's
    first bar excepted, which carries no return."""
    rolls = list(exposures.windows(CALENDAR))
    returns, factors = planted()
    frame = pd.DataFrame(np.asarray(returns), index=CALENDAR[: len(returns)], columns=returns.columns)
    step = {
        "traded": rolls[0][0],
        "window": rolls[0][1],
        "window_start": rolls[0][1][0],
        "window_end": rolls[0][1][-1],
    }
    assert len(walkforward.block(frame, step)) == exposures.WINDOW
    assert len(walkforward.block(frame.iloc[1:], step)) == exposures.MIN_OBS, "the front bar carries no return"
    with pytest.raises(ValueError, match="below the"):
        walkforward.block(frame.iloc[2:], step)

    broken = {**step, "window": pd.period_range(step["window_start"], step["traded"], freq="M"),
              "window_end": step["traded"]}
    with pytest.raises(ValueError, match=str(step["traded"])):
        walkforward.block(frame, broken)
