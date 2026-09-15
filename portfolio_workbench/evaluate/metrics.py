"""The metric block: what every cell is reported on, and the one metric that decides.

**The information ratio against the policy benchmark is primary.** It is benchmark-relative, which is
where the reading behind this effort found the construction choice to matter most; it is denominated
against the same policy portfolio the mandate states; and it follows from the mandate rather than from
a preference formed after the numbers were seen. Everything else is reported **beside** it and never
swapped in for it - realised volatility, tracking error, turnover, drawdown, constraint-binding
frequency, the estimation-error diagnostics and the cost sensitivity. A cell that loses on the
primary metric and wins on one of these has lost.

**The benchmark is costless, so every active figure is policy-relative.** The policy portfolio is
rebalanced to its fixed weights every month and charged nothing, which is a stated convention rather
than a finding: a strategic allocation is not billed for its own rebalancing, and charging it would
make every active cost look smaller by comparison.

**The cost sensitivity is arithmetic on the turnover path, not a second run.** The gross series is the
book's own return and the cost is a function of the turnover the cell already reports, so another rate
is a subtraction from a series the run holds. Re-running the grid at forty basis points would recompute
every solve to change one multiplication, and would produce a second weight path that differs only in
the last decimal.

**The sub-periods are fixed by calendar rather than by outcome.** They are declared here, before the
numbers are read, precisely so that no split can be chosen to flatter a method; and they are
**descriptive only** - never the headline, never the basis of a recommendation. A sub-period result
that disagrees with the full-sample one is a fact about the period, not a result about the method.
"""

import numpy as np
import pandas as pd

from ..construct import constraints

# The annualisation factor every rate in this module carries, and the period count the information
# ratio is scaled by. Stated once, because a metric annualised twelve times and a metric annualised
# by the square root of twelve cannot be compared.
PERIODS_PER_YEAR = 12

# The sub-periods, declared by date rather than by realised drawdowns: the zero-rate stretch, the
# pandemic dislocation and its recovery, the inflation and rates shock, and the recent regime.
SUB_PERIODS = (
    ("zero_rate", "2015-09", "2019-12"),
    ("dislocation", "2020-01", "2021-12"),
    ("inflation_shock", "2022-01", "2023-12"),
    ("recent", "2024-01", "2026-07"),
)


def information_ratio(active, periods=PERIODS_PER_YEAR):
    """The annualised ratio of mean active return to its own volatility.

    A series that is identically zero returns zero rather than an undefined ratio: the benchmark's
    active return against itself is zero, and so is a book that reproduces the benchmark, and both are
    results about the design rather than missing numbers.
    """
    values = np.asarray(active, dtype=float)
    if len(values) < 2:
        raise ValueError("an information ratio needs at least two months")
    spread = float(values.std(ddof=1))
    if spread == 0.0:
        return 0.0
    return float(values.mean() / spread * np.sqrt(periods))


def tracking_error(active, periods=PERIODS_PER_YEAR):
    """The annualised volatility of the active return series."""
    return float(np.asarray(active, dtype=float).std(ddof=1) * np.sqrt(periods))


def volatility(returns, periods=PERIODS_PER_YEAR):
    """The annualised volatility of the cell's own return series."""
    return float(np.asarray(returns, dtype=float).std(ddof=1) * np.sqrt(periods))


def drawdown(returns):
    """The deepest fall from a running peak of the compounded series, as a negative share."""
    values = np.asarray(returns, dtype=float)
    if not len(values):
        raise ValueError("a drawdown needs a return series")
    curve = np.cumprod(1.0 + values)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def active(returns, benchmark):
    """The cell's monthly return in excess of the benchmark's, on one index."""
    series = pd.Series(returns, dtype=float).reindex(pd.Series(benchmark, dtype=float).index)
    if series.isna().any():
        raise ValueError("the benchmark does not cover every month the cell returns")
    return series - pd.Series(benchmark, dtype=float)


def sensitivity(gross, turnover, benchmark, bps=constraints.COST_SENSITIVITY_BP):
    """The same path at another per-side rate: the cost is a function of the turnover already traded,
    so a rate is a subtraction rather than a rerun.

    The ranking's survival at the highest multiple is a separate statement from the level at the
    decided one, because the thin lines in this universe are what the higher rates actually test.
    """
    gross = pd.Series(gross, dtype=float)
    turnover = pd.Series(turnover, dtype=float).reindex(gross.index)
    report = {}
    for rate in bps:
        charged = gross - constraints.cost(turnover, bp=rate)
        report[float(rate)] = {
            "cost_annualised": float(constraints.cost(turnover.mean(), bp=rate) * PERIODS_PER_YEAR),
            "net_cumulative": float((1.0 + charged).prod() - 1.0),
            "information_ratio": information_ratio(active(charged, benchmark)),
        }
    return report


def sub_periods(active_series, declared=SUB_PERIODS):
    """The metric inside each declared calendar window, descriptive only.

    Read off the active series rather than off the cell's own returns, because the question a
    sub-period answers here is whether the cell's advantage existed in it.
    """
    series = pd.Series(active_series, dtype=float)
    months = pd.PeriodIndex(series.index, freq="M")
    report = {}
    for name, start, end in declared:
        inside = series[(months >= pd.Period(start, freq="M")) & (months <= pd.Period(end, freq="M"))]
        report[name] = {
            "months": int(len(inside)),
            "span": f"{start}..{end}",
            "information_ratio": information_ratio(inside) if len(inside) > 2 else None,
            "mean_active_annualised": float(inside.mean() * PERIODS_PER_YEAR) if len(inside) else None,
        }
    return report


def block(result, benchmark):
    """One run's metric block: the primary metric, everything reported beside it, and the diagnostics.

    The traded path and the target path are both reported for the two estimation-error proxies, as
    the runner records them: concentration and weight movement read off the target path are properties
    of the estimator under test, and the same numbers read off the traded path answer a question about
    the no-trade band instead.
    """
    summary = result["summary"]
    net, gross = result["net"], result["gross"]
    turnover = result["turnover"].reindex(net.index)
    difference = active(net, benchmark)
    return {
        "cell": result["id"],
        "stage": result["spec"]["stage"],
        "months": int(len(net)),
        "information_ratio": information_ratio(difference),
        "tracking_error": tracking_error(difference),
        "volatility": volatility(net),
        "mean_active_annualised": float(difference.mean() * PERIODS_PER_YEAR),
        "max_drawdown": drawdown(net),
        "net_cumulative": float((1.0 + net).prod() - 1.0),
        "gross_cumulative": summary["gross_cumulative"],
        "turnover_annualised": summary["turnover_annualised"],
        "cost_annualised": summary["cost_annualised"],
        "cap_binding_frequency": summary["cap_binding_mean"],
        "cap_binding_steps": summary["cap_binding_steps"],
        "turnover_cap_binding_steps": summary["turnover_cap_binding_steps"],
        "weight_stability": summary["target_movement"],
        "concentration": summary["concentration"],
        "concentration_traded": summary["concentration_traded"],
        "largest_weight": summary["max_single_weight"],
        "largest_weight_traded": summary["max_single_weight_traded"],
        "establishment_cost": summary["establishment"]["cost"],
        "cost_sensitivity": sensitivity(gross, turnover, benchmark),
        "sub_periods": sub_periods(difference),
    }
