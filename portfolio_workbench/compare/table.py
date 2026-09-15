"""The comparison table: the pre-registered rows, each carrying its verdict.

The table is the deliverable the whole comparison is read from, so what it may and may not say is
fixed here rather than decided per row.

**Two testing families, and nothing else is tested.** Each cell against the policy benchmark - the
mandate's own question - and each cell against equal weight, which is the bar the literature uses. The
cell-versus-cell matrix is published descriptively and **never tested**: 120 pairs tested at five
percent is where a false positive is born, and the table is the place a reader would take one from.

**A verdict is a ladder, and each rung is a leg the design fixed.** A cell is reported as different
only when it clears the declared family-wise bar - which is also the multiple-testing haircut, since
one test of one statistic serves both - keeps rank in at least eighty percent of bootstrap resamples,
and holds its sign under the expanding-window protocol. Anything else prints **no difference detected**
with the resolution limit beside it. A cell whose book reproduces equal weight is a declared negative
result rather than a small difference, and a cell that clears every leg while sitting behind the
benchmark is a difference detected rather than a candidate.

**The resolution limit travels with the verdict.** Every row carries the smallest information-ratio
difference this test would have detected at eighty percent power, so a reader can see whether "no
difference detected" means the difference is absent or means it is smaller than the panel can
resolve. That distinction is the whole point of measuring the noise floor, and a table that printed
verdicts without it would be making the second statement while appearing to make the first.

**The expanding-window leg is only available where a repeat was declared**, which is the two cells
the mandate's question turns on. The table says so on the row rather than leaving a blank that reads
as a pass.
"""

import numpy as np
import pandas as pd

from ..construct import constraints
from ..evaluate import metrics, statistics
from . import grid as grid_module
from . import registry

# The columns of the row print, in order, and the width each is given.
COLUMNS = (
    ("cell", 34, "s"),
    ("st", 2, "s"),
    ("IR", 8, ".3f"),
    ("vol/yr", 8, ".2%"),
    ("TE/yr", 8, ".2%"),
    ("z vs pol", 9, ".2f"),
    ("res", 6, ".2f"),
    ("ret", 6, ".0%"),
    ("expand", 7, "s"),
    ("verdict", 0, "s"),
)

# The verdict vocabulary, in the order the ladder applies it. The first three name rows that are not
# claims about a method - the two baselines and a repeat of a cell already in the table - and the rest
# are the acceptance bars of the design, in the order it fixes them.
IS_BENCHMARK = "the policy benchmark itself"
IS_EQUAL_WEIGHT = "the equal-weight bar itself"
IS_EXPANDING = "the expanding repeat of the rolling cell"
REPRODUCES = "negative result: the book reproduces equal weight"
NO_DIFFERENCE = "no difference detected"
BEHIND = "significantly behind the benchmark once the cost is charged"
RESAMPLING_LOST = "no difference detected: the advantage does not survive the bootstrap resampling"
SIGN_LOST = "no difference detected: the sign does not hold under the expanding window"
LEG_NOT_RUN = "different on the paired test; the expanding leg was not run for this cell"
CANDIDATE = "recommendation candidate"

# The verdicts the design says are published as negative results rather than as small differences.
NEGATIVE = (NO_DIFFERENCE, REPRODUCES, RESAMPLING_LOST, SIGN_LOST)


def _pad(value, width, spec, align=">"):
    """One cell of the row print. A string is padded rather than formatted, because a verdict and a
    percentage share a column width in the declaration above but not a format code."""
    if not width:
        return f"  {value}"
    if isinstance(value, str):
        return f"{value:{align}{width}}"
    return format(value, f"{align}{width}{spec}")


def cell_ids():
    """The sixteen distinct cells, which is the count the family-wise bar is declared over."""
    return tuple(spec["id"] for spec in registry.CELLS)


def verdict(row):
    """The ladder the design fixed, applied to one row.

    The first rung **is** the haircut: the multiple-testing bar and the paired bar are the same test on
    the same statistic at this design, so they are one rung and not two, and the statement that the bar
    is self-imposed travels with the row's own haircut record. The bootstrap then has to leave the
    advantage standing, the expanding window has to keep its sign where such a run exists, and only
    then is the direction read: a cell that is significantly behind the benchmark is a difference
    detected and not a candidate, and its advantage is measured after the decided cost rather than
    before it, because the primary metric is net.

    The bar that decides is the declared conservative one, and it is the one the row's haircut record
    holds. The correlation-adjusted bar is not a second chance: a cell that clears only the lower bar is
    still reported as no difference detected, and the row's own statistic is printed beside both so a
    reader can weigh it.
    """
    if row["cell"] == "policy":
        return IS_BENCHMARK
    if row["cell"] == "equal_weight":
        return IS_EQUAL_WEIGHT
    if row["is_repeat"]:
        return IS_EXPANDING
    if row["reproduces_equal_weight"]:
        return REPRODUCES
    if not row["haircut"]["clears"]:
        return NO_DIFFERENCE
    if row["retention"] is not None and row["retention"] < statistics.RANK_RETENTION_FLOOR:
        return RESAMPLING_LOST
    if row["is_leader"] and row["leader_retention"] < statistics.RANK_RETENTION_FLOOR:
        return RESAMPLING_LOST
    # The direction is read before the expanding leg so that every cell sitting behind the benchmark
    # says so in the same words; whether such a cell's sign survives the second protocol is a
    # question about a ranking it is not in.
    if row["information_ratio"] <= 0.0:
        return BEHIND
    if row["expanding_sign"] is None:
        return LEG_NOT_RUN
    if not row["expanding_sign"]:
        return SIGN_LOST
    return CANDIDATE


def claimed(row):
    """The four ingredients a recommendation claim is made of, carried on the row itself.

    The design writes bar 3 as *bar 2 plus the haircut plus constraint-binding frequency plus a
    positive cost-adjusted advantage at the decided rate*, so the claim is recorded with all four
    rather than left as a verdict word: the statistic against the bar it cleared, the advantage the
    net series measures after the decided cost, and how often the per-sleeve cap was actually binding
    on the book. The last of those is the one a reader cannot reconstruct from the rest - a cell whose
    book is mostly the bound is a result about the constraint set, and the perturbations exist to show
    it - so a claim that did not travel with it would be silent about the ingredient most likely to
    withdraw it. Rows that are not claims carry None rather than zeroes.
    """
    if row["verdict"] != CANDIDATE:
        return None
    return {
        "statistic": row["paired_benchmark"]["statistic"],
        "bar": row["haircut"]["bar"],
        "information_ratio_net": row["information_ratio"],
        "cost_bp": constraints.COST_BP,
        "cap_binding_frequency": row["cap_binding_frequency"],
        "cap_binding_steps": row["cap_binding_steps"],
        "steps": row["months"],
        "statement": row["haircut"]["statement"],
    }


def rows(grid, benchmark):
    """Every pre-registered run as one row: the metric block, the two paired families, the bootstrap
    retention, the haircut and the verdict.

    The bootstrap and the correlation matrix are computed over the sixteen distinct cells rather than
    over the preconditioned runs, because a perturbation run and an expanding repeat are the same
    method revisited and counting them twice would both change the bar and double a cell's weight in
    the ranking.
    """
    nets = pd.DataFrame({result["id"]: result["net"] for result in grid["results"]})
    cells = [identifier for identifier in cell_ids() if identifier in nets.columns]
    bar = statistics.family_wise_bar(len(cells))
    correlation = statistics.correlation_matrix(nets[cells])
    adjusted = statistics.adjusted_bar(correlation)
    retention = statistics.rank_retention(nets[cells], benchmark)
    equal_weight = nets["equal_weight"]
    repeated = {spec["id"][: -len("_expanding")]: spec["id"] for spec in registry.REPEATED}
    sheet = []
    for result in grid["results"]:
        row = metrics.block(result, benchmark)
        row["paired_benchmark"] = statistics.paired(result["net"], benchmark, benchmark)
        row["paired_equal_weight"] = statistics.paired(result["net"], equal_weight, benchmark)
        measured = retention["cells"].get(result["id"], {})
        row["retention"] = measured.get("retention")
        row["share_leader"] = measured.get("share_leader")
        row["median_rank"] = measured.get("median_rank")
        row["is_leader"] = result["id"] == retention["leader"]
        row["leader_retention"] = retention["retention"]
        row["is_repeat"] = result["id"] in repeated.values()
        row["reproduces_equal_weight"] = bool(
            np.abs(result["net"].to_numpy() - equal_weight.to_numpy()).max() <= statistics.RERUN_TOLERANCE
        )
        row["haircut"] = statistics.haircut(row["paired_benchmark"]["statistic"], bar)
        row["expanding_child"] = repeated.get(result["id"])
        row["expanding_sign"] = None
        sheet.append(row)
    by_id = {row["cell"]: row for row in sheet}
    for identifier, child in repeated.items():
        if identifier in by_id and child in by_id:
            by_id[identifier]["expanding_sign"] = statistics.sign_holds(
                by_id[identifier]["information_ratio"], by_id[child]["information_ratio"]
            )
    for row in sheet:
        row["verdict"] = verdict(row)
        row["claim"] = claimed(row)
    return {
        "rows": sheet,
        "bar": bar,
        "adjusted_bar": adjusted,
        "effective_tests": statistics.effective_tests(correlation),
        "correlation": correlation,
        "retention": retention,
        "cells": cells,
    }


def metric_table(sheet):
    """The numbers a rerun has to reproduce, keyed by cell.

    Only numbers: the verdicts are prose derived from them, and a rerun is compared on the quantities
    a solver's last decimal can move. The manifest and the snapshot id are what make two runs the same
    run; this is what makes them the same numbers.
    """
    table = {}
    for row in sheet["rows"]:
        table[row["cell"]] = {
            "information_ratio": row["information_ratio"],
            "tracking_error": row["tracking_error"],
            "volatility": row["volatility"],
            "max_drawdown": row["max_drawdown"],
            "net_cumulative": row["net_cumulative"],
            "turnover_annualised": row["turnover_annualised"],
            "cost_annualised": row["cost_annualised"],
            "weight_stability": row["weight_stability"],
            "paired_benchmark_statistic": row["paired_benchmark"]["statistic"],
            "paired_equal_weight_statistic": row["paired_equal_weight"]["statistic"],
        }
    table["bar"] = {"family_wise": sheet["bar"], "adjusted": sheet["adjusted_bar"]}
    table["retention"] = {"leader": sheet["retention"]["leader"], "retention": sheet["retention"]["retention"]}
    return table


def reproduces(left, right, tolerance=statistics.RERUN_TOLERANCE):
    """Whether two runs' metric tables agree within the stated tolerance.

    Relative, because an information ratio near zero and a cumulative return near one cannot share an
    absolute bound; the tolerance itself is stated rather than implied, since a rerun's equality is a
    claim the acceptance fixture makes. A value that is not a number is compared by equality rather
    than by tolerance: the leader's own name is part of what a rerun reproduces, and a name that moved
    is a different result whatever the decimals say.
    """
    worst, where, scale = 0.0, None, 1.0

    def walk(a, b, path=""):
        nonlocal worst, where, scale
        if isinstance(a, dict):
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, str) or not isinstance(a, (int, float, np.floating)):
            if a != b:
                worst, where, scale = float("inf"), path.lstrip("."), 1.0
            return
        gap = abs(float(a) - float(b))
        room = max(abs(float(a)), abs(float(b)), 1e-3)
        if gap / room > worst:
            worst, where, scale = gap / room, path.lstrip("."), room

    walk(left, right)
    return {"agrees": bool(worst <= tolerance), "worst": float(worst), "at": where, "tolerance": tolerance, "scale": scale}


def report(sheet, document):
    """The table, the bars, the recommendation claims with the ingredients bar 3 names, the correlation
    between cells, the sub-periods and the negative results."""
    print(
        f"[table] snapshot {document['snapshot_id']}: {len(sheet['rows'])} pre-registered runs, "
        f"{len(sheet['cells'])} distinct cells, {sheet['retention']['draws']} bootstrap resamples under seed "
        f"{sheet['retention']['seed']}"
    )
    print(
        "[table] two testing families only: every cell against the policy benchmark, and every cell against "
        f"equal weight; the {len(sheet['cells']) * (len(sheet['cells']) - 1) // 2}-pair cell matrix is printed "
        "descriptively below and never tested"
    )
    print(
        f"[table] the family-wise bar over {len(sheet['cells'])} cells is |z| >= {sheet['bar']:.4f} "
        f"(Bonferroni-equivalent, and the 95th percentile of the maximum of {len(sheet['cells'])} independent "
        f"standard normals); the realised correlation implies {sheet['effective_tests']:.1f} effective tests, at "
        f"which the bar would be {sheet['adjusted_bar']:.4f} - both are printed, and the conservative one decides"
    )
    print(
        "[table] the multiple-testing haircut is self-imposed from the literature and not a regulatory "
        "requirement: nothing located imposes a multiple-testing correction on portfolio research"
    )
    header = "".join(
        _pad(name, width, spec, align="<" if position == 0 else ">") for position, (name, width, spec) in enumerate(COLUMNS)
    )
    print(f"[table] {header}")
    for row in sheet["rows"]:
        paired, retention = row["paired_benchmark"], row["retention"]
        values = {
            "cell": row["cell"],
            "st": row["stage"],
            "IR": row["information_ratio"],
            "vol/yr": row["volatility"],
            "TE/yr": row["tracking_error"],
            "z vs pol": paired["statistic"],
            "res": paired["resolution"],
            "ret": f"{retention:.0%}" if retention is not None else "--",
            "expand": "yes" if row["expanding_sign"] else ("no" if row["expanding_sign"] is False else "-"),
            "verdict": row["verdict"],
        }
        print("[table] " + "".join(_pad(values[name], width, spec) for name, width, spec in COLUMNS))
    print(
        "[table] res is the smallest information-ratio difference this test would have detected at eighty "
        "percent power; every 'no difference detected' above is qualified by it rather than left to read as "
        "an absence"
    )
    print(
        f"[table] the leader on the full sample is {sheet['retention']['leader']}, retaining rank in "
        f"{sheet['retention']['retention']:.1%} of resamples against a floor of {sheet['retention']['floor']:.0%}; "
        f"ret is each cell's own share of resamples in which its advantage keeps its sign"
    )
    _print_claims(sheet)
    _print_correlation(sheet)
    _print_diagnostics(sheet)
    _print_sub_periods(sheet)
    return sheet


def _print_claims(sheet):
    """The recommendation claims, each printed with the ingredients the design requires beside it.

    Bar 3 is bar 2 plus the haircut plus the constraint-binding frequency plus a positive
    cost-adjusted advantage at the decided rate, so a claim is printed with the statistic it cleared
    the bar with, the advantage its net series measured, and how often the cap was binding on the book
    it actually held. The statement that the bar is self-imposed travels in the row's claim record;
    printing it beside every claim would repeat one sentence as if it were a finding.
    """
    claims = [row for row in sheet["rows"] if row["claim"]]
    if not claims:
        print("[table] no row cleared every rung, so no recommendation is claimed on this run")
        return
    for row in claims:
        claim = row["claim"]
        print(
            f"[table] recommendation claim {row['cell']}: z {claim['statistic']:+.2f} against the declared "
            f"bar {claim['bar']:.4f}, information ratio {claim['information_ratio_net']:+.3f} net of the "
            f"{claim['cost_bp']:.0f} bp decision, per-sleeve cap binding on {claim['cap_binding_steps']} of "
            f"{claim['steps']} steps"
        )


def _print_correlation(sheet):
    """The realised cell-to-cell correlation, published so the bar can be recomputed rather than
    trusted, and tested nowhere."""
    correlation = sheet["correlation"]
    print("[table] the realised correlation of the cells' monthly returns, descriptive and untested:")
    for name in correlation.index:
        values = " ".join(f"{correlation.loc[name, other]:>6.2f}" for other in correlation.columns)
        print(f"[table]   {name:<34s}{values}")


def _print_diagnostics(sheet):
    """The estimation-error diagnostics and the cost, beside the verdicts they qualify.

    Weight stability is the second of the two proxies for estimation error and it needs no ground
    truth: a cell whose optimal weights move when the estimation window shifts one month is exploiting
    estimation error whatever its information ratio. It is reported as the movement the cell's target
    path makes between adjacent months, against the cross-cell median, and the weight a fixed-weight
    book moves - none - so a reader can see the number is estimation error rather than drift.
    """
    movement = sorted(row["weight_stability"] for row in sheet["rows"])
    median = movement[len(movement) // 2]
    print(
        f"[table] estimation-error diagnostics: target-path movement between adjacent months, median "
        f"{median:.2%} across the runs, against 0.00% for a fixed-weight book"
    )
    for row in sheet["rows"]:
        print(
            f"[table]   {row['cell']:<34s} movement {row['weight_stability']:>6.2%}  concentration "
            f"{row['concentration']:>5.2f} on the target path against {row['concentration_traded']:>5.2f} traded  "
            f"largest weight {row['largest_weight']:.3f} against {row['largest_weight_traded']:.3f}  "
            f"turnover {row['turnover_annualised']:>6.2%}/yr  cost {row['cost_annualised']:>5.2%}/yr  "
            f"cap binding {row['cap_binding_frequency']:>5.2f}  drawdown {row['max_drawdown']:>7.2%}"
        )
    print(
        "[table] the cost sensitivity, information ratio at 5, 20 and 40 bp per side, beside the decided "
        "10 bp; the ranking's survival at the top multiple is a separate statement from the level at the base"
    )
    for row in sheet["rows"]:
        rates = "  ".join(
            f"{rate:>4.0f}bp {row['cost_sensitivity'][rate]['information_ratio']:>7.3f}"
            for rate in sorted(row["cost_sensitivity"])
        )
        print(f"[table]   {row['cell']:<34s} {rates}")


def _print_sub_periods(sheet):
    """The declared sub-periods, descriptive only: never the headline, never the basis of a
    recommendation."""
    names = [name for name, _, _ in metrics.SUB_PERIODS]
    spans = "  ".join(f"{name} {row['span']}" for name, row in sheet["rows"][0]["sub_periods"].items())
    print(f"[table] sub-periods fixed by calendar, descriptive only: {spans}")
    for row in sheet["rows"]:
        values = "  ".join(
            f"{name} {row['sub_periods'][name]['information_ratio']:>7.3f}"
            if row["sub_periods"][name]["information_ratio"] is not None
            else f"{name} {'--':>7s}"
            for name in names
        )
        print(f"[table]   {row['cell']:<34s} {values}")


def negative_results(sheet):
    """The cells the design says are published as negative results rather than as small differences."""
    return [
        {"cell": row["cell"], "verdict": row["verdict"]}
        for row in sheet["rows"]
        if row["verdict"] in NEGATIVE
    ]


def main(grid=None, document=None):
    """Run the grid if it is not given, then print the comparison table."""
    from ..data import loader

    if document is None:
        document = loader.load_panel()
    if grid is None:
        grid = grid_module.run_grid(document)
    benchmark = grid_module.benchmark(grid["returns"], grid["months"])
    sheet = rows(grid, benchmark)
    report(sheet, document)
    published = negative_results(sheet)
    print(f"[table] the {len(published)} cells published as negative results, each also carrying a verdict above:")
    for entry in published:
        print(f"[table]   {entry['cell']:<34s} {entry['verdict']}")
    return sheet


if __name__ == "__main__":
    main()
