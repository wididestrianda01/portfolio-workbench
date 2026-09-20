"""The comparison table: the pre-registered rows, each carrying its verdict.

The table is the deliverable the whole comparison is read from, so what it may and may not say is
fixed here rather than decided per row.

**Two testing families, and nothing else is tested.** Each cell against the policy benchmark (the
mandate's own question) and each cell against equal weight, which is the bar the literature uses. The
cell-versus-cell matrix is published descriptively and **never tested**: 120 pairs tested at five
percent is where a false positive is born, and the table is the place a reader would take one from.

**A verdict is a ladder, and each rung is a leg the design fixed.** A cell is reported as different
only when it clears the declared family-wise bar (which is also the multiple-testing haircut, since
one test of one statistic serves both), keeps its own advantage's sign in at least eighty percent of
bootstrap resamples, and holds that sign under the expanding-window protocol. Anything else prints
**no difference detected** with the resolution limit beside it. A cell whose book reproduces equal
weight is a declared negative result rather than a small difference, and a cell that clears every leg
while sitting behind the benchmark is a difference detected rather than a candidate.

**Two of those legs ask whether one cell is better than the benchmark, and one asks whether it is
uniquely best.** The paired bar and the cell's own sign retention are the first kind. The leader's
**rank retention** - the share of resamples in which it stays ahead of the other fifteen - is the
second, and on a table of near-tied methods it fails where the advantage holds. That row is refused a
recommendation in words that say no unique cell was found, not that the advantage was absent: two
different findings, and the table's columns carry both.

**The resolution limit travels with the verdict.** Every row carries the smallest information-ratio
difference a test of that row's own bar would have detected at eighty percent power - the family-wise
bar, since that is what decides the row - so a reader can see whether "no difference detected" means
the difference is absent or means it is smaller than the panel can resolve. That distinction is the
whole point of measuring the noise floor, and a table that printed verdicts without it would be making
the second statement while appearing to make the first.

**The expanding-window leg is only available where a repeat was declared**, which is the two cells
the mandate's question turns on. The table says so on the row rather than leaving a blank that reads
as a pass.
"""

import numpy as np
import pandas as pd

from ..construct import constraints
from ..evaluate import metrics, statistics
from ..factors import exposures
from . import registry

# The columns of the row print, in order: the name, the width the print gives it, the format code the
# print applies and the alignment. The format is carried beside the column rather than applied where
# the row is built, because the workbook writes the same value under its own format and a number is
# only formatted once, at the point it is shown to a reader.
COLUMNS = (
    ("cell", 34, "s", "<"),
    ("st", 2, "s", ">"),
    ("IR", 8, ".3f", ">"),
    ("vol/yr", 8, ".2%", ">"),
    ("TE/yr", 8, ".2%", ">"),
    ("z vs pol", 9, ".2f", ">"),
    ("res", 6, ".2f", ">"),
    ("ret", 6, ".0%", ">"),
    ("expand", 7, "s", ">"),
    ("verdict", 0, "s", ">"),
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
NOT_UNIQUE = (
    "the advantage clears every bar, and no cell is uniquely best: the leader's rank is not retained "
    "in bootstrap resamples"
)
SIGN_LOST = "no difference detected: the sign does not hold under the expanding window"
LEG_NOT_RUN = "different on the paired test; the expanding leg was not run for this cell"
CANDIDATE = "recommendation candidate"

# The verdicts the design says are published as negative results rather than as small differences. The
# rank-refusal belongs here: it is a result the table publishes rather than a candidate it promotes, and
# a reader who takes it for an absence would be reading the ranking as the advantage.
NEGATIVE = (NO_DIFFERENCE, REPRODUCES, RESAMPLING_LOST, NOT_UNIQUE, SIGN_LOST)


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

    **The two bootstrap statistics get two verdicts, because they are two questions.** A cell whose own
    advantage loses its sign under resampling has not measured an advantage, and that keeps the word
    `no difference detected`. The leader whose *rank* is not retained has measured an advantage and has
    not measured a unique one; it is refused a recommendation because the design's acceptance rule
    requires every leg, but it is refused in words that say which leg failed, since a cell reported as
    an absence when the absence is in the ranking is the one reading this ladder exists to prevent.
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
        return NOT_UNIQUE
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
        row["paired_benchmark"] = statistics.paired(result["net"], benchmark, benchmark, bar)
        row["paired_equal_weight"] = statistics.paired(result["net"], equal_weight, benchmark, bar)
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



# The table is read as two stacked blocks rather than as one wide sheet, and the split is the question
# a number answers: the first block carries the return and risk metrics the ranking is read from, the
# second carries what the choice cost and how much of the answer is estimation error. The design fixed
# that shape because a single eight-column table was wide rather than tall, and the cost columns read
# as though they qualified the returns rather than as a separate question.
RETURNS_BLOCK = "returns and risk"
COST_BLOCK = "cost and diagnostics"


def protocol():
    """The protocol every number in the table is produced under.

    The window is read off the module that owns it rather than restated here, because a second copy of
    sixty is how a report comes to describe a model it no longer runs.
    """
    return (
        f"rolling {exposures.WINDOW}-month estimation, monthly refit, out-of-sample months only, "
        f"cost {constraints.COST_BP:.0f} bp per side on traded notional"
    )


def provenance(sheet, document):
    """What every number in the table travels with: which snapshot, under which protocol, over which
    runs and cells, net of which cost multiple, at which bar.

    A number taken out of the table or out of the workbook without these cannot be checked, and this is
    the deliverable a reader takes numbers from, so the block is printed above the first table and
    written into every sheet of the workbook rather than stated once in prose.
    """
    sensitivity = sorted(sheet["rows"][0]["cost_sensitivity"])
    return {
        "snapshot": document.snapshot_id,
        "protocol": protocol(),
        "runs": f"{len(sheet['rows'])} pre-registered runs, {len(sheet['cells'])} distinct cells",
        "months traded": f"{sheet['rows'][0]['months']} out-of-sample months, monthly refit",
        "cost multiple": (
            f"{constraints.COST_BP:.0f} bp per side on traded notional, sensitivity at "
            + ", ".join(f"{rate:.0f} bp" for rate in sensitivity)
        ),
        "bootstrap": f"{sheet['retention']['draws']} resamples under seed {sheet['retention']['seed']}",
        "bar": (
            f"|z| >= {sheet['bar']:.4f} over {len(sheet['cells'])} cells; the realised correlation implies "
            f"{sheet['effective_tests']:.1f} effective tests, at which it would be {sheet['adjusted_bar']:.4f}, "
            "and the conservative one decides"
        ),
    }


def negative_results(sheet):
    """The cells the design says are published as negative results rather than as small differences."""
    return [
        {"cell": row["cell"], "verdict": row["verdict"]}
        for row in sheet["rows"]
        if row["verdict"] in NEGATIVE
    ]


def _column(name, width=0, spec="s", align=">"):
    """One column declaration: what to call it, how wide the print draws it, how to format the value."""
    return (name, width, spec, align)


def _table(title, columns, rows, header=True):
    """One titled table: the title, its column declarations and the raw rows.

    Raw values rather than formatted strings, because the print and the workbook want different things
    from the same number - a column width and a format code against a native cell - and formatting at
    the point of display is what keeps the two from disagreeing about what the value is. A table of
    many narrow numeric columns suppresses the header in the print and says so in its title instead,
    because sixteen column names cannot be drawn above six-character cells.
    """
    return {"title": title, "columns": tuple(columns), "rows": [list(row) for row in rows], "header": header}


def _row_values(row):
    """One pre-registered run as the row both readers carry."""
    paired, retention = row["paired_benchmark"], row["retention"]
    return [
        row["cell"],
        row["stage"],
        row["information_ratio"],
        row["volatility"],
        row["tracking_error"],
        paired["statistic"],
        paired["resolution"],
        retention,
        "yes" if row["expanding_sign"] else ("no" if row["expanding_sign"] is False else "-"),
        row["verdict"],
    ]


def _diagnostic_columns():
    return (
        _column("cell", 34, "s", "<"),
        _column("movement", 9, ".2%"),
        _column("concentration", 14, ".2f"),
        _column("concentration traded", 14, ".2f"),
        _column("largest weight", 14, ".3f"),
        _column("largest traded", 14, ".3f"),
        _column("turnover/yr", 11, ".2%"),
        _column("cost/yr", 9, ".2%"),
        _column("cap binding", 11, ".2f"),
        _column("drawdown", 10, ".2%"),
    )


def _claim_notes(sheet):
    """The recommendation claims, each with the ingredients bar 3 names, or the statement that the run
    made none.

    Bar 3 is bar 2 plus the haircut plus the constraint-binding frequency plus a positive cost-adjusted
    advantage at the decided rate, so a claim travels with the statistic it cleared the bar with, the
    advantage its net series measured, and how often the cap was binding on the book it actually held.
    The last of those is the one a reader cannot reconstruct from the rest, and a claim silent about it
    would be silent about the ingredient most likely to withdraw it.
    """
    claims = [row for row in sheet["rows"] if row["claim"]]
    if not claims:
        return ["no row cleared every rung, so no recommendation is claimed on this run"]
    return [
        f"recommendation claim {row['cell']}: z {row['claim']['statistic']:+.2f} against the declared bar "
        f"{row['claim']['bar']:.4f}, information ratio {row['claim']['information_ratio_net']:+.3f} net of the "
        f"{row['claim']['cost_bp']:.0f} bp decision, per-sleeve cap binding on {row['claim']['cap_binding_steps']} "
        f"of {row['claim']['steps']} steps"
        for row in claims
    ]


def block_tables(sheet):
    """The two stacked blocks as titled tables of raw rows.

    Both readers are rendered from this one structure, so the workbook cannot carry a number the table
    does not print or print one the workbook does not carry. The notes belong to the block rather than
    to a table inside it where they qualify the block as a whole - the res line, the leader's retention
    and the haircut's provenance are statements about the reading, not about one row.
    """
    rates = sorted(sheet["rows"][0]["cost_sensitivity"])
    correlation = sheet["correlation"]
    periods = [name for name, _, _ in metrics.SUB_PERIODS]
    rows = sheet["rows"]
    movement = sorted(row["weight_stability"] for row in rows)
    pairs = len(sheet["cells"]) * (len(sheet["cells"]) - 1) // 2
    return (
        {
            "block": RETURNS_BLOCK,
            "tables": (
                _table("the pre-registered runs", COLUMNS, [_row_values(row) for row in rows]),
                _table(
                    "the cells published as negative results, each also carrying its verdict above",
                    (_column("cell", 34, "s", "<"), _column("verdict")),
                    [[row["cell"], row["verdict"]] for row in negative_results(sheet)],
                ),
                _table(
                    "the realised correlation of the cells' monthly returns, descriptive and untested; rows "
                    "and columns are in the same order as the runs above, and the workbook carries the names",
                    (_column("cell", 34, "s", "<"),)
                    + tuple(_column(name, 6, ".2f") for name in sheet["cells"]),
                    [
                        [name] + [correlation.loc[name, other] for other in correlation.columns]
                        for name in correlation.index
                    ],
                    header=False,
                ),
                _table(
                    "sub-periods fixed by calendar, descriptive only: never the headline, never the basis "
                    "of a recommendation",
                    (_column("cell", 34, "s", "<"),) + tuple(_column(name, 8, ".3f") for name in periods),
                    [
                        [row["cell"]] + [row["sub_periods"][name]["information_ratio"] for name in periods]
                        for row in rows
                    ],
                ),
            ),
            "notes": (
                f"two testing families only: every cell against the policy benchmark, and every cell against "
                f"equal weight; the {pairs}-pair cell matrix above is descriptive and never tested",
                "res is the smallest information-ratio difference this test would have detected at eighty "
                "percent power; every 'no difference detected' is qualified by it rather than left to read as "
                "an absence",
                f"the leader on the full sample is {sheet['retention']['leader']}, retaining rank in "
                f"{sheet['retention']['retention']:.1%} of resamples against a floor of "
                f"{sheet['retention']['floor']:.0%}; ret is each cell's own share of resamples in which its "
                "advantage keeps its sign",
                "the multiple-testing haircut is self-imposed from the literature and not a regulatory "
                "requirement: nothing located imposes a multiple-testing correction on portfolio research",
            )
            + tuple(_claim_notes(sheet)),
        },
        {
            "block": COST_BLOCK,
            "tables": (
                _table(
                    "estimation-error diagnostics and the cost, beside the verdicts they qualify; the median "
                    f"target-path movement across the runs is {movement[len(movement) // 2]:.2%} against 0.00% "
                    "for a fixed-weight book",
                    _diagnostic_columns(),
                    [
                        [
                            row["cell"],
                            row["weight_stability"],
                            row["concentration"],
                            row["concentration_traded"],
                            row["largest_weight"],
                            row["largest_weight_traded"],
                            row["turnover_annualised"],
                            row["cost_annualised"],
                            row["cap_binding_frequency"],
                            row["max_drawdown"],
                        ]
                        for row in rows
                    ],
                ),
                _table(
                    "the cost sensitivity, information ratio at each per-side multiple beside the decided one; "
                    "the ranking's survival at the top multiple is a separate statement from the level at the base",
                    (_column("cell", 34, "s", "<"),)
                    + tuple(_column(f"{rate:.0f} bp", 8, ".3f") for rate in rates),
                    [
                        [row["cell"]]
                        + [row["cost_sensitivity"][rate]["information_ratio"] for rate in rates]
                        for row in rows
                    ],
                ),
            ),
            "notes": (),
        },
    )


def _print_table(table):
    """One table: its title, its column header and its rows, under the layer's own tag.

    A column is drawn no narrower than its own name when the name is shown, because a name wider than
    its column overflows into the next one and two numbers with no gap between them read as one; the
    last column carries no width at all and is separated from the one before it by two spaces.
    """
    columns = table["columns"]
    widths = [
        max(width, len(name) + 1) if (table["header"] and width) else width
        for name, width, spec, align in columns
    ]
    print(f"[table] {table['title']}")
    if table["header"]:
        header = "".join(
            _pad(name, width, "s", align) if width else f"  {name}"
            for (name, declared, spec, align), width in zip(columns, widths)
        )
        print(f"[table] {header}")
    for row in table["rows"]:
        cells = (
            _pad("--" if value is None else value, width, spec, align)
            for (name, declared, spec, align), width, value in zip(columns, widths, row)
        )
        print("[table] " + "".join(cells))


def report(sheet, document):
    """The provenance block, then the two stacked blocks: the return and risk metrics first, then the
    cost and the estimation-error diagnostics, each table printed with the notes that qualify it."""
    for name, value in provenance(sheet, document).items():
        print(f"[table] {name}: {value}")
    for position, block in enumerate(block_tables(sheet), start=1):
        print(f"[table] block {position} - {block['block']}")
        for table in block["tables"]:
            _print_table(table)
        for note in block["notes"]:
            print(f"[table] {note}")
    return sheet


def main(document=None, grid=None):
    """Analyse the snapshot unless a reading is given, then print the comparison table.

    The grid is passed through to the analysis rather than run here, so a caller that has just run it
    is not charged for a second run and the table is read off the same run everything else is.
    """
    from ..data import loader
    from ..study import analyse

    analysis = analyse(loader.load_panel() if document is None else document, grid=grid)
    report(analysis.sheet, analysis.document)
    return analysis.sheet


if __name__ == "__main__":
    main()
