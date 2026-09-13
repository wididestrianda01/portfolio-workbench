"""The as-of rule and the returned series.

A monthly bar is labelled with the month's first day while its close is the month's
last close. Nothing in the package may read such a bar before the first day of the
following month, and the only way to read the panel is through `as_of`, so the
embargo is a property of the access path rather than a rule each caller remembers.
"""

import pandas as pd

from .universe import PANEL_START, WINDOW_END


def add_availability(frame, date_col="date"):
    """Attach `period_month` and `available_from` to a dated frame.

    `period_month` is the label the analytics join on; `available_from` is the first
    instant the bar may be used. The two are deliberately different columns: joining on
    the label alone is the look-ahead error this module exists to prevent.
    """
    out = frame.copy()
    months = pd.PeriodIndex(out[date_col], freq="M")
    out["period_month"] = months.to_timestamp(how="start")
    out["available_from"] = (months + 1).to_timestamp(how="start")
    return out


def as_of(frame, when, column="available_from"):
    """Rows a reader standing at `when` could have seen. Rows are never dropped here
    for being unended - that is the quality gate's business; this is only the filter."""
    return frame[frame[column] <= pd.Timestamp(when)]


def drop_unended(frame, when, column="available_from"):
    """Remove bars whose month had not ended at `when`. The rule, not a check."""
    return as_of(frame, when, column)


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


def coverage_months(frame):
    """First and last month present per instrument, and the months missing between them."""
    report = {}
    for instrument, block in frame.groupby("instrument"):
        months = pd.PeriodIndex(block["period_month"].drop_duplicates(), freq="M").sort_values()
        full = pd.period_range(months[0], months[-1], freq="M")
        report[instrument] = {
            "first": str(months[0]),
            "last": str(months[-1]),
            "months": len(months),
            "missing": [str(m) for m in full.difference(months)],
        }
    return report


def window(frame, start=PANEL_START, end=WINDOW_END, column="period_month"):
    """Trim to the declared panel window, and say how much was trimmed."""
    months = pd.PeriodIndex(frame[column], freq="M")
    keep = (months >= pd.Period(start, freq="M")) & (months <= pd.Period(end, freq="M"))
    before, after = len(frame), int(keep.sum())
    return frame[keep], before - after
