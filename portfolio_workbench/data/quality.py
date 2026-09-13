"""The quality gate: eight stops and four warnings, as rules rather than as surprises.

A stop means the panel is not the panel the results are keyed to, and it raises. A
warning means the number is usable but qualified, and it prints beside the number it
qualifies - never in a footnote, because a qualification a reader has to go looking for
is a qualification that gets dropped.

The eighth stop, a manifest mismatch, is enforced by `manifest.verify` before any frame
is built, since it is a property of the files rather than of the loaded table.
"""

import re

import pandas as pd

from . import panel
from .universe import SLEEVES, WINDOW_END, WINDOW_START

# A monthly move beyond the bound stated for that sleeve is a data fault rather than a
# market event at these sizes, and the bound travels per sleeve because the sleeves do not
# have the same tails. The widest single-month move measured on the frozen snapshot is
# real estate +32.4% (2013-03), gold +16.1%, euro equity -14.2%, high-yield credit -12.4%,
# long government +10.2%, short government 0.5% and cash 0.3%; each bound below sits well
# clear of its own sleeve's worst, so a bound fires on a split or a denomination error
# rather than on March 2020. A level at or below zero is never a price.
ABSURD_MOVE = {
    "real_estate": 1.00,
    "gold": 0.50,
    "equity_eu": 0.45,
    "credit_hy": 0.40,
    "equity_nordic": 0.35,
    "equity_dev": 0.35,
    "gov_long": 0.30,
    "inflation_linked": 0.20,
    "credit_ig": 0.20,
    "gov_short": 0.05,
    "cash": 0.05,
}

# A line the sleeve map does not name has no sleeve to read a bound from; it falls back
# to the single bound that served every line before the map existed. Unnamed lines reach
# the gate only from a hand-built fixture, never from a snapshot, which names its universe.
ABSURD_MOVE_DEFAULT = 0.60

# Fund size below which a sleeve is thin enough that its trading cost is not the flat
# per-side rate the cost model charges. The floor sits between the two lines the universe
# audit named as small and the next one up, and that gap is narrow: IBGL.AS reports EUR
# 0.94bn and IEGE.AS EUR 1.30bn against a floor of EUR 1.35bn, while IWDP.AS reports USD
# 1.62bn and XACT-NORDEN.ST SEK 17,092m, about EUR 1.41bn and EUR 1.54bn at the snapshot's
# own rates. Sizes arrive in the fund's own denomination, so a non-euro figure is converted
# at the snapshot's newest month-end rate rather than at a rate assumed today, and the
# converted figure is printed beside the warning so a close call is visible as one.
LIQUIDITY_FLOOR_EUR = 1.35e9
FUND_SIZE = re.compile(r"([A-Z]{3})\s+([\d.]+)(bn|m|k)")
SIZE_UNITS = {"bn": 1e9, "m": 1e6, "k": 1e3}

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


def income_policy(facts, instrument):
    """The issuer's declared income policy, and whether it means the line pays out.

    The field is free text in the manifest - "distributing (semi-annual)", "accumulating",
    "no income" - so which lines pay out is decided here, once, and both dividend rules
    read that answer rather than parsing the string for themselves. An accumulating line
    and a metal line are both non-payers, and the two rules must agree on that.
    """
    policy = str(facts.get(instrument, {}).get("income_policy", "")).lower()
    return policy, policy.startswith("distributing")


def fund_size_eur(reported, rates):
    """The issuer's reported fund size in euro, or None where it is not published.

    The manifest carries the figure as the issuer writes it - "EUR 0.94bn", "SEK 17,092m",
    "n/a" - so it is parsed rather than assumed. A size the parser cannot read and a
    currency the snapshot carries no rate for both return None, and the caller reports
    those lines as unscreened rather than letting them pass the floor by default.
    """
    match = FUND_SIZE.fullmatch(str(reported).replace(",", "").strip())
    if match is None:
        return None
    currency, amount, unit = match.groups()
    rate = 1.0 if currency == "EUR" else rates.get(currency)
    return None if rate is None else float(amount) * SIZE_UNITS[unit] * rate


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
        first, last, count, gaps = panel.month_span(block["period_month"])
        if count != len(block):
            duplicated = block["period_month"][block["period_month"].duplicated()].iloc[0]
            raise DataStop("missing bar", f"{instrument} carries {duplicated:%Y-%m} twice")
        if len(gaps):
            raise DataStop(
                "missing bar",
                f"{instrument} has no bar for {[str(g) for g in gaps[:3]]} "
                f"between {first} and {last}",
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

    It is excluded by rule on the load path; this stop proves the snapshot never carried
    it, because a partially formed month entering the panel is invisible in every
    statistic downstream of it.
    """
    moment = panel.cutoff(as_of)
    fresh = frame[frame["available_from"] > moment]
    if len(fresh):
        raise DataStop(
            "too-fresh bar",
            f"{len(fresh)} rows carry bars available only from {fresh['available_from'].min():%Y-%m-%d}, "
            f"after the snapshot was taken ({moment:%Y-%m-%d}); the newest is "
            f"{fresh['period_month'].max():%Y-%m}",
        )


def stop_dividend_blind(frame, facts):
    """A line the issuer says distributes, with no distribution event in the whole series.

    Its adjusted close then equals its close, so it is a price series wearing a
    total-return label. Nothing downstream can repair that, and the failure is silent.
    """
    for instrument, block in frame.groupby("instrument"):
        policy, pays_out = income_policy(facts, instrument)
        if pays_out and float(block["dividend"].fillna(0.0).sum()) == 0.0:
            raise DataStop(
                "distributing line with no distribution",
                f"{instrument} is declared '{policy}' but carries no distribution event over "
                f"{len(block)} months; its adjusted close is not a total-return series",
            )


def stop_implausible_prices(frame):
    """A non-positive level, or a single-month move beyond the bound for that sleeve."""
    for instrument, block in frame.groupby("instrument"):
        block = block.sort_values("period_month")
        if (block["close"] <= 0).any() or (block["adj_close"] <= 0).any():
            row = block.loc[(block["close"] <= 0) | (block["adj_close"] <= 0)].iloc[0]
            raise DataStop(
                "implausible level",
                f"{instrument} shows a level of {row['close']} at {row['period_month']:%Y-%m}",
            )
        sleeve = SLEEVES.get(instrument, (None, None))[0]
        bound = ABSURD_MOVE.get(sleeve, ABSURD_MOVE_DEFAULT)
        moves = block["adj_close"].pct_change()
        if (moves.abs() > bound).any():
            worst = moves.abs().idxmax()
            raise DataStop(
                "implausible move",
                f"{instrument} moves {moves[worst]:+.1%} into {block.loc[worst, 'period_month']:%Y-%m}, "
                f"beyond the {bound:.0%} bound stated for the {sleeve or 'unnamed'} sleeve",
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


def warn_thin_liquidity(frame, facts, rates, floor=LIQUIDITY_FLOOR_EUR):
    """A line whose fund is smaller than the floor stated for this universe.

    The floor is on the issuer's reported fund size, which is what the universe audit
    could actually read, not on a traded notional nobody here observes. Sizes are converted
    at the snapshot's own rate so that a line is not screened on the currency it happens to
    be quoted in. A line whose size the issuer does not publish is reported as unscreened:
    it must not pass a floor it was never measured against.
    """
    thin, unscreened = [], []
    for instrument in sorted(frame["instrument"].unique()):
        if instrument not in facts:
            # The FX legs are not funds and the manifest describes no issuer for them.
            continue
        reported = facts[instrument].get("fund_size")
        size = fund_size_eur(reported, rates)
        if size is None:
            unscreened.append(f"{instrument} ({reported or 'no size recorded'})")
        elif size < floor:
            converted = "" if reported.startswith("EUR") else f", EUR {size / 1e9:.2f}bn at the snapshot's rate"
            thin.append(
                f"WARNING thin liquidity: {instrument} reports {reported}{converted}, below "
                f"the EUR {floor / 1e9:.2f}bn floor; the flat per-side cost rate understates "
                f"what trading this sleeve costs"
            )
    if unscreened:
        thin.append(
            f"WARNING liquidity not screened: {', '.join(unscreened)} publish no fund size, "
            f"so the floor cannot be applied to them"
        )
    return thin


def warn_factor_trim(price_months, factor_months):
    """The factor file's newest month is the panel's end. Say what it costs."""
    prices = pd.PeriodIndex(price_months, freq="M")
    factors = pd.PeriodIndex(factor_months, freq="M")
    lost = prices.difference(factors)
    if len(lost):
        return [
            (
                f"WARNING factor month trims the panel: prices reach {prices.max()}, factors stop at "
                f"{factors.max()}; {len(lost)} price months ({lost.min()}..{lost.max()}) are outside the "
                f"joined panel"
            )
        ]
    return []


def warn_extra_distributions(frame, facts):
    out = []
    for instrument, block in frame.groupby("instrument"):
        events = block[block["dividend"].fillna(0.0) > 0]
        if not len(events):
            continue
        policy, pays_out = income_policy(facts, instrument)
        if not pays_out:
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


def run(frame, facts, as_of, factor_months=None, rates=None):
    """Every frame-level rule, in the order that fails fastest.

    Malformed dates are checked where each file is parsed instead of here: after the
    files are concatenated the offending file can no longer be named, and a stop that
    cannot say which line is bad is a stop nobody can act on.

    `rates` carries EUR per unit of each currency the issuer facts are quoted in, taken
    from the snapshot so that the liquidity floor is applied on one basis.
    """
    stop_missing_bars(frame)
    stop_late_start(frame)
    stop_too_fresh(frame, as_of)
    stop_dividend_blind(frame, facts)
    stop_implausible_prices(frame)
    warnings = (
        warn_tr_divergence(frame)
        + warn_thin_liquidity(frame, facts, rates or {})
        + warn_extra_distributions(frame, facts)
    )
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
