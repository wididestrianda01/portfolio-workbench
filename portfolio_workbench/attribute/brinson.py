"""The holding-based decomposition: Brinson-Fachler, the currency dimension, and linking.

**The decomposition is allocation-only, structurally, and it says so rather than printing zeros.**
Brinson-Fachler (1985, JPM 11(3), 73-76) splits the active return into allocation, selection and
interaction. Its selection term is the benchmark's weight applied to the difference between the
portfolio's sleeve return and the benchmark's, and in this universe those are the same number: the
policy benchmark holds the *same instrument* for each sleeve, so `r_p,i = r_b,i` and selection is
identically zero. The workbook therefore states that the term does not exist here, with the reason,
rather than carrying a zero column - a column of zeros in a performance table reads as a market
finding, and this one is a property of how the benchmark was built.

**The currency dimension carries the interaction instead.** The euro return of a sleeve is
`(1 + local)(1 + translation) - 1`, so the active return decomposes along exactly that line into
allocation measured on **local-currency** returns, a currency effect that is the active weight applied
to the translation, and the cross product of the two, which is named as its own line rather than
folded into either. The lines then sum to the active return exactly, and measuring allocation on euro
returns while also adding a currency line would count the translation twice - which is why the
ordering is stated here and not left to the reader. Two limits travel with it: the dimension is
**quotation-currency, not look-through**, since an unhedged USD holding inside a euro-quoted ETF moves
the fund's NAV with nothing in this data to separate it; and only the SEK-quoted sleeve shows a
translation at all.

**"Reconciles exactly" is a linking property once more than one period is joined.** The single-period
identity is arithmetic; the compounded one is not, because the effects do not compound with the
returns that produced them. Cariño (1999) is the reported method and Menchero (2000) the cross-check:
both are implemented, both are held against the compounded geometric excess, and the difference
between them is reported rather than resolved away. Cariño's coefficient is one logarithm per period
and is exact by construction; Menchero's is a constant for the run plus a period adjustment carrying
the compounding residual in proportion to each period's own active return. They land on the same total
and distribute it differently, which is exactly why the second is worth printing.

**Costs are a named line and the benchmark is costless by convention.** A policy benchmark is not
billed for its own rebalancing, so gross reconciles to it; the portfolio is charged the per-side rate
on traded notional and net equals gross minus cost. The cost is never merged into selection, where it
would read as a stock-picking result.
"""

import numpy as np
import pandas as pd

from ..compare import grid as grid_module
from ..construct import constraints
from ..data import loader, panel

# The papers this module implements, cited where their difference shows. 1985 is Brinson-Fachler, the
# allocation term the report uses; 1986 is Brinson-Hood-BeeBower, which the hand-checked case runs
# alongside because the two are routinely confused and their difference is not the one usually claimed.
BRINSON_FACHLER = "Brinson & Fachler (1985), JPM 11(3), 73-76"
BRINSON_HOOD_BEEBOWER = "Brinson, Hood & Beebower (1986), FAJ 42(4), 39-44"
CARINO = "Carino (1999), Journal of Performance Measurement 3(4)"
MENCHERO = "Menchero (2000), Journal of Performance Measurement, Fall, 36-42"

# The lines a period's active return is split into, in the order the report prints them. Selection is
# absent in this universe and is deliberately not a member: a line that is identically zero cannot be
# told apart from a line whose computation is broken.
LINES = ("allocation", "currency", "interaction")

# Why the selection term is absent, carried as a value rather than left to each caller's prose, so the
# workbook and the report cannot come to disagree about the reason.
SELECTION_ABSENT = (
    "the selection term does not exist here: the policy benchmark holds the same instrument for each "
    "sleeve, so the portfolio's sleeve return and the benchmark's are the same number and the term is "
    "identically zero. Stated rather than printed as a zero column, which would read as a finding"
)

# The cost convention in one sentence, quoted by the report rather than rebuilt from the module that
# applies it.
COST_CONVENTION = "the per-side rate on traded notional; the benchmark is costless by convention"

# Where a quotient stops being computable rather than meaningful: Cariño's scaling factor tends to
# 1/(1+R_b) as a period's active return goes to zero, and Menchero's period adjustment tends to zero as
# the run's active returns all vanish. Both limits are taken rather than divided out, because a month
# whose book reproduces the benchmark leaves a genuine 0/0 in both formulae.
COEFFICIENT_FLOOR = 1e-12

# The tolerance the linked effects are held to against the compounded excess. It is a floating-point
# bound rather than a modelling one: both methods land on the compounded difference identically, so
# anything above accumulation is a broken decomposition rather than one method's error. The scale
# carries a floor because a relative bound around a zero excess is not a bound at all.
LINKING_TOLERANCE = 1e-9
LINKING_SCALE_FLOOR = 1e-6


def single_period(portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
    """One period's Brinson-Fachler decomposition, with the Brinson-Hood-BeeBower form beside it.

    Three effects per sleeve whose totals add to the period's active return exactly. Allocation is the
    weight difference applied to the sleeve's benchmark return **net of the benchmark's own total**,
    which is what makes the term benchmark-relative: a sleeve held above its benchmark weight in a
    market that did badly contributes a negative allocation. Selection is the benchmark's weight
    applied to the return difference. Interaction is the product of the two differences.

    Nothing here is specific to this universe - the two return vectors are separate arguments - which is
    what lets the hand-checked case be a case rather than a restatement of `allocation_only`. The second
    form returned is Brinson-Hood-BeeBower as it is normally reported: its allocation is the weight
    difference applied to the sleeve's return with **no** benchmark-relative adjustment, and its
    interaction is left as the residual the paper calls "other" rather than named as an effect. The two
    forms' totals agree term by term, because the weight deviations sum to zero: what differs is the
    level of the allocation column, and that is the whole of what the contrast shows.
    """
    w_p = pd.Series(portfolio_weights, dtype=float)
    index = w_p.index
    w_b = pd.Series(benchmark_weights, dtype=float).reindex(index)
    r_p = pd.Series(portfolio_returns, dtype=float).reindex(index)
    r_b = pd.Series(benchmark_returns, dtype=float).reindex(index)
    if w_b.isna().any() or r_p.isna().any() or r_b.isna().any():
        raise ValueError("the two weight vectors and the two return vectors must cover the same sleeves")

    total_b = float(w_b @ r_b)
    deviation = w_p - w_b
    allocation = deviation * (r_b - total_b)
    selection = w_b * (r_p - r_b)
    interaction = deviation * (r_p - r_b)
    active = float(w_p @ r_p) - total_b
    totals = {
        "allocation": float(allocation.sum()),
        "selection": float(selection.sum()),
        "interaction": float(interaction.sum()),
    }
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "bhb_allocation": deviation * r_b,
        "other": interaction,
        "benchmark_total": total_b,
        "totals": totals,
        "active": active,
        "residual": float(sum(totals.values())) - active,
        "bhb_total": float((deviation * r_b).sum() + totals["selection"] + totals["interaction"]),
        "source": BRINSON_FACHLER,
        "contrast": BRINSON_HOOD_BEEBOWER,
    }


def allocation_only(book, benchmark_weights, split):
    """Every month's lines for one book, from its weight path and the panel's currency split.

    The portfolio holds one weight per sleeve and the benchmark holds the policy weights, so the weight
    deviation is the only thing that varies: allocation is that deviation applied to the sleeve's
    local-currency return measured against the benchmark's own local total, currency is the deviation
    applied to the translation, and interaction is the deviation applied to their product. Their sum is
    the euro active return, and that is checked here rather than left to the report, because a
    decomposition that does not add up is not a decomposition.

    Allocation is measured on local returns for the reason stated in the module: measuring it in euro
    and then adding a currency line counts the translation twice, and the double count is invisible -
    the lines still sum, they simply attribute a currency move to allocation.
    """
    book = book.reindex(columns=split["local"].columns)
    months = book.index.intersection(split["local"].index)
    if len(months) == 0:
        raise ValueError("the weight path and the currency split share no month")
    weights = book.reindex(months)
    departure = float((weights.sum(axis=1) - 1.0).abs().max())
    if departure > constraints.FEASIBILITY_TOLERANCE:
        raise ValueError(
            f"the book departs from full investment by {departure:.3e} in one month; the decomposition "
            f"is written for two fully invested books and would read the shortfall as a sleeve's weight"
        )
    local, translation, euro = (split[name].reindex(months) for name in ("local", "translation", "euro"))
    benchmark = pd.Series(benchmark_weights, dtype=float).reindex(local.columns)
    if benchmark.isna().any():
        raise ValueError("the benchmark does not cover every sleeve the book holds")

    deviation = weights.sub(benchmark, axis=1)
    allocation = deviation.mul(local.sub(local @ benchmark, axis=0), axis=0)
    currency = deviation.mul(translation, axis=0)
    interaction = deviation.mul(local * translation, axis=0)
    lines = pd.DataFrame({name: frame.sum(axis=1) for name, frame in
                          (("allocation", allocation), ("currency", currency), ("interaction", interaction))})
    lines["active"] = (deviation * euro).sum(axis=1)
    lines["residual"] = lines[list(LINES)].sum(axis=1) - lines["active"]
    worst = float(lines["residual"].abs().max())
    if worst > LINKING_TOLERANCE * max(float(lines["active"].abs().max()), LINKING_SCALE_FLOOR):
        raise ValueError(f"the decomposition leaves a residual of {worst:.3e} against the active return")
    return {
        "months": pd.PeriodIndex(months, freq="M"),
        "allocation": allocation,
        "currency": currency,
        "interaction": interaction,
        "lines": lines,
        "benchmark": pd.Series(local @ benchmark, index=months),
        "selection": 0.0,
        "selection_absent": SELECTION_ABSENT,
        "source": BRINSON_FACHLER,
    }


def cost_line(gross, net):
    """The cost the book paid, as its own line rather than a subtraction somewhere inside selection.

    Two numbers, because they answer different questions: the per-month mean is what the rebalances
    cost at the decided rate, and the compounded drag is what the net series lost against the gross
    over the run. They are not the same number and neither is derivable from the other once returns
    compound, which is why both are reported.
    """
    gross = pd.Series(gross, dtype=float)
    net = pd.Series(net, dtype=float).reindex(gross.index)
    if net.isna().any():
        raise ValueError("the net series does not cover every month the gross series does")
    difference = net - gross
    return {
        "mean_monthly": float(difference.mean()),
        "annualised": float(difference.mean() * 12),
        "drag": float((1.0 + net).prod() - (1.0 + gross).prod()),
        "convention": COST_CONVENTION,
    }


def _quotient(numerator, denominator, limit, floor=COEFFICIENT_FLOOR):
    """A quotient whose limit is taken where the denominator vanishes.

    Both linking methods divide by a period's active return, which is legitimately zero whenever a book
    reproduces the benchmark in that month; the limit is what the ratio tends to, and it is used over
    every row whose denominator is small rather than only where it is exactly zero, because a quotient
    computed from a denominator of 1e-14 is noise amplified.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    computable = np.abs(denominator) > floor
    return np.where(computable, numerator / np.where(computable, denominator, 1.0), limit)


def carino(portfolio, benchmark, effects):
    """Cariño's linking: one logarithmic coefficient per period, one for the run, and the ratio.

    The coefficient is `(ln(1+R_p) - ln(1+R_b)) / (R_p - R_b)` per period, and the same expression over
    the run's compounded returns for the whole; the linked effect is the ratio of the two applied to
    each period's effect. It is exact rather than approximate: the coefficients convert each period's
    arithmetic active return into its log-equivalent contribution, and logarithms of products are sums,
    so the linked effects land on the compounded difference and not near it. That property is what makes
    "the identity reconciles" a claim a reader can check rather than a tolerance to negotiate.
    """
    portfolio = pd.Series(portfolio, dtype=float)
    benchmark = pd.Series(benchmark, dtype=float).reindex(portfolio.index)
    effects = effects.reindex(portfolio.index)
    active = portfolio - benchmark
    coefficients = _quotient(np.log1p(portfolio) - np.log1p(benchmark), active, limit=1.0 / (1.0 + benchmark))
    total_p = float((1.0 + portfolio).prod() - 1.0)
    total_b = float((1.0 + benchmark).prod() - 1.0)
    whole = _quotient(np.log1p(total_p) - np.log1p(total_b), total_p - total_b, limit=1.0 / (1.0 + total_b))
    linked = effects.mul(coefficients / whole, axis=0).sum()
    return {
        "linked": linked,
        "coefficients": pd.Series(coefficients, index=portfolio.index),
        "whole": float(whole),
        "total": float(linked.sum()),
        "target": total_p - total_b,
        "method": CARINO,
    }


def menchero(portfolio, benchmark, effects):
    """Menchero's linking: one constant for the run plus a period adjustment carrying the residual.

    The constant `M` is the run's average active return over the difference between the two compounded
    growth rates taken to the same root, and the adjustment `a_t` hands each period a share of what `M`
    leaves unaccounted, in proportion to that period's own active return. The two together preserve the
    total exactly and distribute it differently from Cariño: where Cariño weights every period by the
    logarithm its returns imply, Menchero keeps a common level and corrects it, so a run with one
    violent month and otherwise flat effects lands the same total with the two methods pointing at
    different months.

    The limit is used where the run's own active return vanishes: `M` tends to `(1+R_p)^((n-1)/n)`
    there and the period adjustment tends to zero, which is the case a book reproducing the benchmark
    reaches every month.
    """
    portfolio = pd.Series(portfolio, dtype=float)
    benchmark = pd.Series(benchmark, dtype=float).reindex(portfolio.index)
    effects = effects.reindex(portfolio.index)
    active = (portfolio - benchmark).to_numpy()
    periods = len(active)
    if periods == 0:
        raise ValueError("linking needs at least one period")
    total_p = float((1.0 + portfolio).prod() - 1.0)
    total_b = float((1.0 + benchmark).prod() - 1.0)
    constant = float(_quotient(
        (total_p - total_b) / periods,
        (1.0 + total_p) ** (1.0 / periods) - (1.0 + total_b) ** (1.0 / periods),
        limit=(1.0 + total_p) ** ((periods - 1.0) / periods),
    ))
    adjustment = _quotient(total_p - total_b - constant * active.sum(), float(active @ active), limit=0.0) * active
    factors = constant + adjustment
    linked = effects.mul(factors, axis=0).sum()
    return {
        "linked": linked,
        "factors": pd.Series(factors, index=portfolio.index),
        "constant": constant,
        "total": float(linked.sum()),
        "target": total_p - total_b,
        "method": MENCHERO,
    }


def linking(portfolio, benchmark, effects, tolerance=LINKING_TOLERANCE):
    """Both methods on the same effects, with the residual and their disagreement reported beside them.

    The residual is measured against the compounded difference of the two growth rates, which is the
    object both methods are constructed to land on, and it travels with its scale so a reader can see
    whether a residual is a tolerance or a broken decomposition.
    """
    reported = carino(portfolio, benchmark, effects)
    crossed = menchero(portfolio, benchmark, effects)
    target = reported["target"]
    gap = reported["linked"] - crossed["linked"]
    return {
        "carino": reported,
        "menchero": crossed,
        "target": target,
        "residual": reported["total"] - target,
        "relative": abs(reported["total"] - target) / max(abs(target), LINKING_SCALE_FLOOR),
        "reconciles": bool(abs(reported["total"] - target) <= tolerance * max(abs(target), LINKING_SCALE_FLOOR)),
        "agreement": float(gap.abs().max()),
        "per_line_difference": gap,
        "tolerance": tolerance,
        "benchmark_cost": 0.0,
    }


def decompose(book, benchmark_weights, split, net=None):
    """One cell's whole attribution: the monthly lines, the linked totals, and the cost line.

    The linking runs on the three lines and on nothing else, because those are the effects that sum to
    the active return; the cost is carried beside them as its own line, so gross reconciles to the
    benchmark and net equals gross minus cost without the cost entering a decomposition it does not
    belong to. Each line is also linked sleeve by sleeve on the same coefficients, because the
    coefficients belong to the period rather than to the line: the sleeve totals then sum to the linked
    total, and the workbook can say which sleeve moved an effect rather than only how big it was.
    """
    block = allocation_only(book, benchmark_weights, split)
    benchmark = block["benchmark"]
    gross = block["lines"]["active"] + benchmark
    result = {
        **block,
        "gross": gross,
        "linked": linking(gross, benchmark, block["lines"][list(LINES)]),
        "linked_by_sleeve": {
            name: carino(gross, benchmark, block[name])["linked"] for name in LINES
        },
    }
    if net is not None:
        result["cost"] = cost_line(gross, net)
    return result


def report(cells, document, count=6):
    """The attribution's own report: every cell's linked lines, the leader's sleeves, the currency and
    the cost, each beside the residual that says whether it reconciles."""
    months = cells[0]["months"]
    print(f"[attrib] snapshot {document['snapshot_id']}: {len(cells)} runs, {months.min()}..{months.max()} "
          f"({len(months)} months)")
    print(f"[attrib] {BRINSON_FACHLER}, arithmetic; {CARINO} reported, {MENCHERO} as the cross-check")
    print(f"[attrib] allocation is measured on LOCAL-currency returns, so selection here is absent by "
          f"structure - {cells[0]['selection_absent']}")
    print(f"[attrib] cost: {constraints.COST_BP:.0f} bp per side on traded notional - twice the one-way "
          f"turnover - charged to the net series; {COST_CONVENTION}")
    print(f"[attrib] {'cell':<34s}{'allocation':>11s}{'currency':>10s}{'interaction':>12s}"
          f"{'gross excess':>13s}{'residual':>11s}{'carino-menchero':>17s}")
    for cell in cells:
        linked = cell["linked"]
        totals = {name: float(linked["carino"]["linked"][name]) for name in LINES}
        print(f"[attrib] {cell['id']:<34s}{totals['allocation']:>+11.4%}{totals['currency']:>+10.4%}"
              f"{totals['interaction']:>+12.4%}{linked['target']:>+13.4%}{linked['residual']:>+11.2e}"
              f"{linked['agreement']:>17.2e}")
    worst = max(cells, key=lambda cell: abs(cell["linked"]["residual"]))
    print(f"[attrib] the residual is largest on {worst['id']}: {worst['linked']['residual']:+.2e} against a "
          f"compounded excess of {worst['linked']['target']:+.4%}, relative {worst['linked']['relative']:.1e} "
          f"against a tolerance of {worst['linked']['tolerance']:.0e}, so every run reconciles: "
          f"{all(cell['linked']['reconciles'] for cell in cells)}")
    carried = max(cells, key=lambda cell: abs(float(cell["linked"]["carino"]["linked"]["currency"])))
    print(f"[attrib] the largest currency line is {carried['id']}'s at "
          f"{float(carried['linked']['carino']['linked']['currency']):+.4%} compounded: one sleeve is quoted "
          f"in SEK and every other translation sits inside a euro-quoted fund's own return, where this data "
          f"cannot separate it - the dimension is quotation-currency, not look-through")
    leader = max(cells, key=lambda cell: float(cell["linked"]["target"]))
    print(f"[attrib] {leader['id']}, whose compounded gross excess is the largest, sleeve by sleeve over "
          f"{len(leader['months'])} months - linked allocation, currency and interaction in return points:")
    ranked = leader["linked_by_sleeve"]["allocation"].abs().sort_values(ascending=False).index[:count]
    for sleeve in ranked:
        print(f"[attrib]   {sleeve:<16s} allocation "
              f"{float(leader['linked_by_sleeve']['allocation'][sleeve]):>+10.4%}  currency "
              f"{float(leader['linked_by_sleeve']['currency'][sleeve]):>+10.4%}  interaction "
              f"{float(leader['linked_by_sleeve']['interaction'][sleeve]):>+10.4%}")
    if "cost" in leader:
        print(f"[attrib] {leader['id']}'s cost over the same months: {abs(leader['cost']['annualised']):.4%}/yr "
              f"on the mean, {abs(leader['cost']['drag']):.4%} compounded, against the benchmark's zero by "
              f"convention - the line is a named drag and is never merged into selection")
    return cells


def main(root=None):
    """Load the snapshot, run the declared grid, and attribute every book against the policy weights."""
    document = loader.load_panel(root)
    split = panel.currency_split(document["prices"], document["fx"])
    grid = grid_module.run_grid(document)
    benchmark_weights = grid_module.policy_weights(grid["returns"].columns)
    cells = []
    for result in grid["results"]:
        block = decompose(result["weights"], benchmark_weights, split, net=result["net"])
        block["id"] = result["id"]
        block["stage"] = result["spec"]["stage"]
        cells.append(block)
    report(cells, document)
    for cut in grid["cuts"]:
        print(f"[attrib] {cut['id']}: cut from the grid, so there is nothing to attribute: {cut['reason']}")
    for line in document["warnings"]:
        print(f"[attrib] {line}")
    return {"cells": cells, "split": split, "grid": grid, "document": document}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
