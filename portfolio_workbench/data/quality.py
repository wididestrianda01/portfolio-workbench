"""The quality gate: eight stops and four warnings, as rules rather than as surprises.

A stop means the panel is not the panel the results are keyed to, and it raises. A
warning means the number is usable but qualified, and it prints beside the number it
qualifies - never in a footnote, because a qualification a reader has to go looking for
is a qualification that gets dropped.

The eighth stop, a manifest mismatch, is enforced by `manifest.verify` before any frame
is built, since it is a property of the files rather than of the loaded table.
"""

import pandas as pd

from .universe import WINDOW_END, WINDOW_START

# A monthly move beyond this is a data fault rather than a market event at these sizes:
# the widest single-month move any of these sleeves has posted is a fraction of it.
# A level at or below zero is a split or a denomination error, not a price.
IMPOSSIBLE_MOVE = 0.60

# Consecutive unchanged monthly closes. A line whose NAV has not moved in half a year is
# not being priced; the threshold clears the slowest genuine sleeve here (a money-market
# line, whose NAV still drifts every month) so that a warning means a fault.
STALE_RUN = 6

# Cumulative divergence between the feed's adjusted close and the total return rebuilt
# from unadjusted close plus distributions. The widest divergence among the lines that
# reconcile is 0.006 against a bound of 0.02, and the headroom is deliberate: the check
# exists to catch a line whose adjustment ignores its distributions entirely, which
# diverges by whole percentage points. XACT-NORDEN.ST diverges by 2.155 because its raw
# dividend field reports events its adjusted close does not reflect, and the warning is
# how that reaches the reader instead of being averaged away.
TR_DIVERGENCE = 0.02


class DataStop(Exception):
    """A quality stop fired. The panel is refused, not patched."""

    def __init__(self, rule, detail):
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


def _months(frame):
    return pd.PeriodIndex(frame["period_month"], freq="M")


# ------------------------------------------------------------------------------- stops
def stop_missing_columns(frame, required, name):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataStop("missing column", f"{name} lacks {missing}; the file shape has changed")


def stop_malformed_dates(frame, name="panel"):
    if frame["date"].isna().any():
        bad = frame.loc[frame["date"].isna()]
        raise DataStop("malformed date", f"{name} has unparsable dates at rows {bad.index[:3].tolist()}")
    labels = pd.DatetimeIndex(_months(frame).to_timestamp(how="start"))
    off_grid = frame.loc[labels != pd.DatetimeIndex(frame["date"])]
    if len(off_grid):
        first = off_grid.iloc[0]
        raise DataStop(
            "malformed date",
            f"{name} carries a bar dated {first['date']:%Y-%m-%d} in {first['instrument']}, "
            f"which is not a month-start label; the bar is not a monthly bar",
        )


def stop_missing_bars(frame):
    """A month absent inside a series' own range, or the same month twice."""
    for instrument, block in frame.groupby("instrument"):
        months = pd.PeriodIndex(block["period_month"].drop_duplicates(), freq="M").sort_values()
        if len(months) != len(block):
            duplicated = block["period_month"][block["period_month"].duplicated()].iloc[0]
            raise DataStop("missing bar", f"{instrument} carries {duplicated:%Y-%m} twice")
        gaps = pd.period_range(months[0], months[-1], freq="M").difference(months)
        if len(gaps):
            raise DataStop(
                "missing bar",
                f"{instrument} has no bar for {[str(g) for g in gaps[:3]]} "
                f"between {months[0]} and {months[-1]}",
            )


def stop_late_start(frame, start=WINDOW_START):
    """A line whose history begins after the window opens is excluded, never padded in.

    Padding would fabricate returns at the front of the out-of-sample window, which is
    the one place a fabricated return is least visible and most damaging. History
    reaching further back than the window is not an error: the window trims it.
    """
    started = frame.groupby("instrument")["period_month"].min()
    late = {str(i): f"{m:%Y-%m}" for i, m in started.items() if pd.Period(m, freq="M") > pd.Period(start, freq="M")}
    if late:
        raise DataStop(
            "line starting late",
            f"{late} begin after the declared window opens at {start}; excluding them is a "
            f"universe decision recorded in the manifest, not a silent trim",
        )


def stop_too_fresh(frame, as_of):
    """A bar whose month had not ended when the snapshot was taken.

    It is excluded by rule on the load path; this stop proves it stayed excluded, because
    a partially formed month entering the panel is invisible in every statistic
    downstream of it.
    """
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        # Bar labels are naive month starts; the snapshot's creation stamp is UTC. The
        # comparison is on the calendar date the snapshot was taken.
        cutoff = cutoff.tz_localize(None)
    fresh = frame[frame["available_from"] > cutoff]
    if len(fresh):
        raise DataStop(
            "too-fresh bar",
            f"{len(fresh)} rows carry bars available only from {fresh['available_from'].min():%Y-%m-%d}, "
            f"after the snapshot was taken ({cutoff:%Y-%m-%d}); the newest is "
            f"{fresh['period_month'].max():%Y-%m}",
        )


def stop_dividend_blind(frame, facts):
    """A line the issuer says distributes, with no distribution event in the whole series.

    Its adjusted close then equals its close, so it is a price series wearing a
    total-return label. Nothing downstream can repair that, and the failure is silent.
    """
    for instrument, block in frame.groupby("instrument"):
        policy = str(facts.get(instrument, {}).get("income_policy", "")).lower()
        if policy.startswith("distributing") and float(block["dividend"].fillna(0.0).sum()) == 0.0:
            raise DataStop(
                "distributing line with no distribution",
                f"{instrument} is declared '{policy}' but carries no distribution event over "
                f"{len(block)} months; its adjusted close is not a total-return series",
            )


def stop_implausible_prices(frame):
    for instrument, block in frame.groupby("instrument"):
        block = block.sort_values("period_month")
        if (block["close"] <= 0).any() or (block["adj_close"] <= 0).any():
            row = block.loc[(block["close"] <= 0) | (block["adj_close"] <= 0)].iloc[0]
            raise DataStop(
                "implausible level",
                f"{instrument} shows a level of {row['close']} at {row['period_month']:%Y-%m}",
            )
        moves = block["adj_close"].pct_change()
        if (moves.abs() > IMPOSSIBLE_MOVE).any():
            worst = moves.abs().idxmax()
            raise DataStop(
                "implausible move",
                f"{instrument} moves {moves[worst]:+.1%} into {block.loc[worst, 'period_month']:%Y-%m}, "
                f"beyond the {IMPOSSIBLE_MOVE:.0%} plausibility bound",
            )


# ---------------------------------------------------------------------------- warnings
def warn_tr_divergence(frame, tolerance=TR_DIVERGENCE):
    """The feed's adjustment against total return rebuilt from close plus distributions."""
    out = []
    for instrument, block in frame.groupby("instrument"):
        block = block.sort_values("period_month")
        feed = float((1.0 + block["adj_close"].pct_change().fillna(0.0)).prod() - 1.0)
        rebuilt = float(
            ((block["close"] + block["dividend"].fillna(0.0)) / block["close"].shift(1)).fillna(1.0).prod() - 1.0
        )
        if abs(feed - rebuilt) > tolerance:
            out.append(
                f"WARNING recomputed-TR divergence: {instrument} adjusted close implies {feed:+.2%} "
                f"cumulative, close plus distributions implies {rebuilt:+.2%}; the series is used as "
                f"published and the divergence travels with it"
            )
    return out


def warn_stale_line(frame, run=STALE_RUN):
    out = []
    for instrument, block in frame.groupby("instrument"):
        closes = block.sort_values("period_month")["adj_close"].to_numpy()
        longest, current = 1, 1
        for previous, value in zip(closes, closes[1:]):
            current = current + 1 if value == previous else 1
            longest = max(longest, current)
        if longest >= run:
            out.append(
                f"WARNING thin liquidity: {instrument} shows {longest} consecutive unchanged monthly "
                f"closes; treated as a stale line"
            )
    return out


def warn_factor_trim(price_months, factor_months):
    """The factor file's newest month is the panel's end. Say what it costs."""
    prices = pd.PeriodIndex(price_months, freq="M")
    factors = pd.PeriodIndex(factor_months, freq="M")
    lost = prices.difference(factors)
    if len(lost):
        return [
            f"WARNING factor month trims the panel: prices reach {prices.max()}, factors stop at "
            f"{factors.max()}; {len(lost)} price months ({lost.min()}..{lost.max()}) are outside the "
            f"joined panel"
        ]
    return []


def warn_extra_distributions(frame, facts):
    out = []
    for instrument, block in frame.groupby("instrument"):
        events = block[block["dividend"].fillna(0.0) > 0]
        policy = str(facts.get(instrument, {}).get("income_policy", "")).lower()
        if not len(events):
            continue
        if policy.startswith("accumulating") or "no income" in policy:
            out.append(
                f"WARNING unexpected distribution: {instrument} is declared '{policy}' yet carries "
                f"{len(events)} distribution event(s), first {events['period_month'].min():%Y-%m}"
            )
            continue
        per_year = events.groupby(events["period_month"].dt.year).size()
        cadence = policy.split("(")[-1].strip(")")
        allowed = {"quarterly": 4, "semi-annual": 2, "annual": 1}.get(cadence, 12)
        if int(per_year.max()) > allowed:
            out.append(
                f"WARNING extra distribution events: {instrument} is declared '{policy}' but pays "
                f"{int(per_year.max())} times in {int(per_year.idxmax())}; a corporate action is the "
                f"usual cause"
            )
    return out


def run(frame, facts, as_of, factor_months=None):
    """Every frame-level rule, in the order that fails fastest.

    Malformed dates are checked where each file is parsed instead of here: after the
    files are concatenated the offending file can no longer be named, and a stop that
    cannot say which line is bad is a stop nobody can act on.
    """
    stop_missing_bars(frame)
    stop_late_start(frame)
    stop_too_fresh(frame, as_of)
    stop_dividend_blind(frame, facts)
    stop_implausible_prices(frame)
    warnings = warn_tr_divergence(frame) + warn_stale_line(frame) + warn_extra_distributions(frame, facts)
    if factor_months is not None:
        months = pd.PeriodIndex(frame["period_month"].drop_duplicates(), freq="M")
        warnings += warn_factor_trim(months, factor_months)
    return warnings


def assert_within_window(frame, end=WINDOW_END):
    """The window is declared; a frame running past it is refused rather than trimmed."""
    months = pd.PeriodIndex(frame["period_month"].drop_duplicates(), freq="M")
    if len(months) and months.max() > pd.Period(end, freq="M"):
        raise DataStop(
            "bar beyond the declared window",
            f"the panel runs to {months.max()} but the window closes at {end}",
        )
    return frame
