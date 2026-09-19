"""The as-of rule and the returned series.

A monthly bar is labelled with the month's first day while its close is the month's
last close. Nothing in the package may read such a bar before the first day of the
following month, and the only way to read the panel is through `as_of`, so the
embargo is a property of the access path rather than a rule each caller remembers.
"""

import pandas as pd

from . import external
from .universe import FX_QUOTES, PANEL_START, TICKERS, WINDOW_END


def available_from(months):
    """The moment a month-labelled bar becomes readable: the first day of the month after it.

    One definition for every reader of the rule. The as-of filter applies it to the rows
    a snapshot carries and the walk-forward engine applies it to the months a window is
    cut on, and the two have to agree on which bars a given moment exposes - a second copy
    of this arithmetic would let the gate a reader is told about differ from the gate the
    estimate was formed under.
    """
    return (pd.PeriodIndex(months, freq="M") + 1).to_timestamp(how="start")


def add_availability(frame, date_col="date"):
    """Attach `period_month` and `available_from` to a dated frame.

    `period_month` is the label the analytics join on; `available_from` is the first
    instant the bar may be used. The two are deliberately different columns: joining on
    the label alone is the look-ahead error this module exists to prevent.
    """
    out = frame.copy()
    months = pd.PeriodIndex(out[date_col], freq="M")
    out["period_month"] = months.to_timestamp(how="start")
    out["available_from"] = available_from(months)
    return out


def cutoff(when):
    """The moment a reader stands at, on the calendar rather than in UTC.

    Bar labels are naive month starts while a snapshot's creation stamp carries a UTC
    offset, so the comparison is made on the calendar date. One definition, because the
    as-of filter and the too-fresh stop must agree on which bars a given moment exposes.
    """
    moment = pd.Timestamp(when)
    return moment.tz_localize(None) if moment.tzinfo is not None else moment


def as_of(frame, when, column="available_from"):
    """Rows a reader standing at `when` could have seen. Rows are never dropped here
    for being unended - that is the quality gate's business; this is only the filter."""
    return frame[frame[column] <= cutoff(when)]


def available_months(index, when):
    """The existing months a reader standing at `when` could have seen.

    The same rule as `as_of`, for the legs that carry no `available_from` column because
    they are month-labelled already: the factor library and the accrued overnight rate.
    A month is readable from the first day of the next one, so a reader standing inside
    month M sees up to M-1. Applied by the loader to every leg, because a leg that
    skipped it would be the one look-ahead path into the package that the availability
    rule exists to close.
    """
    months = pd.PeriodIndex(index, freq="M")
    return months[months < cutoff(when).to_period("M")]


def wide(frame, value="adj_close"):
    """Long table to one column per instrument, indexed by period month."""
    out = frame.pivot(index="period_month", columns="instrument", values=value)
    return out.sort_index()


def total_return(prices_wide):
    """Total return from the feed's adjusted close.

    The adjustment is the feed's own and is used as given: a total-return series is
    never rebuilt by hand, because a hand-built one hides the fact that it was built.
    Where the adjustment is wrong the recomputation check reports it.
    """
    return prices_wide / prices_wide.shift(1) - 1.0


def recomputed_total_return(close_wide, dividend_wide):
    """Total return rebuilt from the unadjusted close plus distributions.

    Used only as a cross-check against the feed's adjustment. A line whose adjusted
    close ignores its distributions diverges from this series immediately, which is
    how a dividend-blind line is caught mechanically rather than by eye.
    """
    return (close_wide + dividend_wide.fillna(0.0)) / close_wide.shift(1) - 1.0


def sleeve_order(frame):
    """The sleeve map's order where the frame carries the whole universe, the frame's own otherwise.

    Every weight vector in the package is indexed by `TICKERS`, so a frame that came back in another
    order would misalign against them silently. A partial frame is left alone rather than reordered,
    so a fixture keeps the order it declared.
    """
    names = list(frame.columns)
    order = [name for name in TICKERS if name in names]
    return frame[order] if len(order) == len(names) else frame


def currency_split(prices, fx, value="adj_close"):
    """The panel's returns split into the legs the attribution decomposes, and the euro return they
    multiply into.

    A line quoted in another currency contributes three things: its return in its own denomination,
    the return of holding that currency against the euro, and their product, which is the cross term
    the decomposition names separately rather than folding into either. All three are returned
    together, because the holding-based attribution decomposes along exactly this line and a caller
    that rebuilt one leg from the other two would be reconstructing the input of the decomposition
    it is meant to check.

    The translation is the RECIPROCAL of the currency leg's own move, never the move itself: the rate
    is quoted as units of the foreign currency per euro, so a euro that buys more of it loses on a
    foreign return, and taking the leg directly would invert the currency basis while leaving the euro
    return it is checked against untouched - the shape of error that never shows as a failed identity.
    An instrument quoting in euro translates by exactly zero, and the frame says so rather than
    carrying a rate it does not apply.
    """
    # The panel's first bar has no predecessor, so its return does not exist. Dropped rather than
    # filled: a leg cannot translate a return that is not there, and a filled zero would enter every
    # window containing it.
    local = total_return(wide(prices, value)).iloc[1:]
    fx_returns = total_return(wide(fx, "close")).iloc[1:]
    translation = pd.DataFrame(0.0, index=local.index, columns=local.columns)
    euro = local.copy()
    for instrument, quote in prices.groupby("instrument")["currency"].first().items():
        if quote == "EUR":
            continue
        leg_name = next((name for name, currency in FX_QUOTES.items() if currency == quote), None)
        if leg_name is None or leg_name not in fx_returns.columns:
            raise ValueError(f"{instrument} is quoted in {quote} and the snapshot carries no {quote} leg")
        translation[instrument] = 1.0 / (1.0 + fx_returns[leg_name]) - 1.0
        euro[instrument] = external.eur_translate(local[[instrument]], fx_returns[leg_name])[instrument]
    # The pivot labels rows with the month's first day; the factor spine and the joined panel are
    # labelled with the month itself. One representation across every analytics frame, so a join
    # between them cannot silently come back empty.
    frames = {"local": local, "translation": translation, "euro": euro}
    for frame in frames.values():
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    return {name: sleeve_order(frame) for name, frame in frames.items()}


def excess(euro, risk_free):
    """The euro leg in excess of the euro overnight rate, on one calendar.

    Named separately from `eur_excess_returns` because the loader splits one snapshot once and needs
    both the legs and the excess frame: recomputing the split for the second of them would be the same
    arithmetic twice, and a caller that rebuilt the euro leg from the other two would be reconstructing
    the input of the decomposition it is meant to check.

    The cash rate is subtracted, because every factor in the model is an excess return or a spread and
    a left-hand side measured gross would carry the cash rate into the intercept. The month the panel's
    first bar cannot produce is dropped as a structural absence while any other missing month is
    refused: a filled zero is a fabricated observation, and it would enter every window containing it.
    """
    rate = risk_free.copy()
    # The split's frames are labelled with the month itself, so the cash rate is put on the same
    # calendar rather than the return frame being moved onto the rate's: one representation, no
    # timestamp round-trip in the middle of a subtraction.
    rate.index = pd.PeriodIndex(rate.index, freq="M")
    rate = rate.reindex(euro.index)
    if rate.isna().any():
        raise ValueError(f"the risk-free series does not cover {list(rate.index[rate.isna()][:3])}")
    frame = euro.sub(rate, axis=0)

    uncomputed = frame.index[frame.isna().any(axis=1)]
    structural = len(uncomputed) == 1 and uncomputed[0] == frame.index[0]
    if len(uncomputed) and not structural:
        raise ValueError(f"the return frame holds months it cannot compute: {list(uncomputed[:3])}")
    frame = frame.loc[~frame.isna().any(axis=1)]
    return sleeve_order(frame)


def eur_excess_returns(prices, fx, risk_free, value="adj_close"):
    """The sleeve frame every factor result is measured on: EUR total returns in excess of the
    euro overnight rate.

    The translation into euro is the split `currency_split` performs - one definition, so the frame the
    factors are fitted on and the frame the attribution decomposes cannot disagree about what a
    translation is - and the excess is then `excess`'s own arithmetic rather than a second statement of
    it here.
    """
    return excess(currency_split(prices, fx, value=value)["euro"], risk_free)


def joined_months(price_months, factor_months, risk_free_months):
    """Hard intersection of the three legs. No padding, no forward fill.

    A late-starting leg shortens the panel for everyone and is reported as the number
    of months it costs, because a padded series would put a fabricated return into the
    out-of-sample window.
    """
    index = pd.PeriodIndex(price_months, freq="M")
    for other in (factor_months, risk_free_months):
        index = index.intersection(pd.PeriodIndex(other, freq="M"))
    return index.sort_values()


def month_span(months):
    """One series' own coverage: first month, last month, month count, months missing.

    Computed in one place because the coverage report and the missing-bar stop would
    otherwise each decide for themselves what a missing month is, and the two reports
    would then disagree about the same panel.
    """
    unique = pd.PeriodIndex(months, freq="M").drop_duplicates().sort_values()
    full = pd.period_range(unique[0], unique[-1], freq="M")
    return unique[0], unique[-1], len(unique), full.difference(unique)


def coverage_months(frame):
    """First and last month present per instrument, and the months missing between them."""
    report = {}
    for instrument, block in frame.groupby("instrument"):
        first, last, count, missing = month_span(block["period_month"])
        report[instrument] = {
            "first": str(first),
            "last": str(last),
            "months": count,
            "missing": [str(m) for m in missing],
        }
    return report


def window(frame, start=PANEL_START, end=WINDOW_END, column="period_month"):
    """Trim to the declared panel window, and say how much was trimmed."""
    months = pd.PeriodIndex(frame[column], freq="M")
    keep = (months >= pd.Period(start, freq="M")) & (months <= pd.Period(end, freq="M"))
    before, after = len(frame), int(keep.sum())
    return frame[keep], before - after


def window_months(index, start=PANEL_START, end=WINDOW_END):
    """A month-labelled index trimmed to the declared window.

    The priced legs are trimmed by `window`, which reads a `period_month` column; the factor and rate
    legs are indexed by the month itself and carry no such column, so the same rule is applied to the
    index. Without it a leg reaches whatever its publisher has published since the panel's last
    completed month - on this snapshot a ten-day September accrual in the cash leg, which is a month no
    book was held for, in a leg the join hides rather than excludes.
    """
    months = pd.PeriodIndex(index, freq="M")
    return months[(months >= pd.Period(start, freq="M")) & (months <= pd.Period(end, freq="M"))]
