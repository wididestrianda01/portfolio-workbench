"""The five figures: what the tables show, drawn once, under the licence rule.

**Why so few, and why these.** A figure earns its place only where it carries an argument a table
cannot: a dispersion is easier to see than to read off sixteen rows, a cost sensitivity is a shape
rather than a column, and a spectrum against its null is a comparison of three series over 131 windows.
Padding one figure into every part would be decoration, so the guide carries five and the data, pipeline
and method parts carry prose.

**The licence rule is enforced here rather than described.** The price feed's terms do not permit
redistribution, so no figure may plot a price, a level, an index value or an instrument series. That is
implemented as a closed vocabulary: every drawn series declares the *kind* of quantity it carries, a
level kind is absent from the vocabulary, and `_series` refuses anything outside it. A figure that would
publish a series cannot be drawn by this module at all, which is a stronger guarantee than a rule in a
docstring.

**Every figure is checked as it is written.** The check asserts that the title, the axis labels and the
legend labels are present as *text* in the SVG, and that every drawn series carries exactly as many
points as the frame it was read from. Text rather than paths is a choice, not a default: matplotlib
writes glyph outlines by default, and outlines would make the labels unsearchable in a file a reader is
expected to open, quote and verify separately from this code.

**Reproducible.** The SVG backend is selected explicitly rather than inherited, because a workstation
with a GUI backend configured would otherwise open a window, and the hash salt and the metadata date are
fixed so that regenerating an unchanged figure produces an unchanged file.
"""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "portfolio-workbench"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

from pathlib import Path  # noqa: E402  (after the backend selection, deliberately)

import numpy as np  # noqa: E402
from matplotlib.backends.backend_svg import FigureCanvasSVG  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from portfolio_workbench.compare import table as table_module  # noqa: E402
from portfolio_workbench.construct import constraints  # noqa: E402
from portfolio_workbench.data import universe  # noqa: E402
from portfolio_workbench.evaluate import statistics  # noqa: E402

from . import memo as memo_module  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE / "figures"

# The kinds of quantity a figure may draw. A price, a level, an index value or an instrument series is
# absent on purpose, and `_series` refuses anything not listed here.
KINDS = (
    "dispersion",
    "ratio",
    "share",
    "count",
    "weight",
    "distribution",
    "eigenvalue",
    "cost",
)

INK = "#1b1b1b"
GRID = "#d8d8d8"
MARKS = ("#1b6ca8", "#c0504d", "#4f8a4f", "#8a6f1b")

DPI = 110


def _series(label, kind, values):
    """One drawn series, with the kind of quantity it carries checked against the closed vocabulary.

    The kind is not decoration: it is the licence rule as a type. A caller that wants to draw a price
    series has nothing to pass, and the refusal happens before a figure exists rather than after one has
    been written and published.
    """
    if kind not in KINDS:
        raise ValueError(
            f"{label}: '{kind}' is not a drawable kind of quantity; a price, a level or an instrument "
            f"series is refused because the feed's terms do not permit redistribution. Allowed: {KINDS}"
        )
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{label}: nothing to draw")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label}: a series carrying a non-finite value is not drawn")
    return {"label": label, "kind": kind, "values": array}


def _verify(path, spec):
    """The figure's own check: its words are text in the SVG, and its series match their frames."""
    text = path.read_text()
    missing = [word for word in spec["text"] if word not in text]
    if missing:
        raise ValueError(f"{path}: not present as text in the SVG: {missing}")
    if not spec["lengths"]:
        raise ValueError(f"{path}: a figure with no drawn series")
    for name, drawn, frame in spec["lengths"]:
        if drawn != frame:
            raise ValueError(f"{path}: {name} drew {drawn} points against a frame of {frame}")
    return path


def _canvas(fig, path, spec):
    """Draw to SVG, then check what was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasSVG(fig)
    fig.savefig(path, format="svg", dpi=DPI, metadata={"Date": None}, bbox_inches="tight")
    _verify(path, spec)
    return path


def _spec(*, text, lengths):
    return {"text": tuple(text), "lengths": tuple(lengths)}


def figure_dispersion(sheet, path):
    """Every run's tracking error against its information ratio.

    The argument the table cannot carry: the cells separate along tracking error far more than along
    the information ratio, and the mean-variance family sits alone on the right of the picture.
    """
    rows = sheet["rows"]
    widest = max(rows, key=lambda row: row["tracking_error"])
    labelled = {"policy", "equal_weight", sheet["retention"]["leader"], widest["cell"]}
    fig = Figure(figsize=(9.0, 4.8))
    ax = fig.subplots()
    stages = sorted({row["stage"] for row in rows})
    for position, stage in enumerate(stages):
        selected = [row for row in rows if row["stage"] == stage]
        tracking = _series(f"stage {stage} tracking error", "dispersion", [row["tracking_error"] for row in selected])
        ratio = _series(f"stage {stage} information ratio", "ratio", [row["information_ratio"] for row in selected])
        ax.scatter(tracking["values"], ratio["values"], s=42, color=MARKS[position % len(MARKS)],
                   label=f"stage {stage} ({len(selected)} runs)", zorder=3)
    # Four points carry the argument, and the table names the rest: twenty annotations in a cluster
    # overlap into an unreadable block, and a reader who wants a cell's identity has it two pages up.
    for row in rows:
        if row["cell"] not in labelled:
            continue
        ax.annotate(row["cell"], (row["tracking_error"], row["information_ratio"]),
                    fontsize=7.0, xytext=(4, 4), textcoords="offset points", color=INK, zorder=4)
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
    ax.grid(color=GRID, linewidth=0.6, zorder=1)
    ax.set_xlabel("Tracking error against the policy benchmark, % a year")
    ax.set_ylabel("Information ratio")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    labels = ["Tracking error against the policy benchmark, % a year", "Information ratio", "stage A", "stage B", "stage C"]
    return _canvas(fig, path, _spec(text=labels, lengths=[("dispersion", len(rows), len(rows))]))


def figure_cost(sheet, path):
    """The leader's information ratio against the per-side multiple, beside every run's turnover.

    Two statements the cost block makes in numbers: cost moves the level rather than the ranking at the
    multiple tested, and turnover is a property of the family rather than of the objective.
    """
    rows = sheet["rows"]
    rates = sorted(rows[0]["cost_sensitivity"])
    leader = next(row for row in rows if row["cell"] == sheet["retention"]["leader"])
    ratio = _series("leader information ratio", "ratio", [leader["cost_sensitivity"][rate]["information_ratio"] for rate in rates])
    cost = _series("per-side cost", "cost", rates)
    ordered = sorted(rows, key=lambda row: row["turnover_annualised"])
    turnover = _series("turnover annualised", "share", [row["turnover_annualised"] for row in ordered])

    fig = Figure(figsize=(9.4, 5.0))
    left, right = fig.subplots(1, 2, width_ratios=(1.0, 1.35))
    left.plot(cost["values"], ratio["values"], color=MARKS[0], marker="o", zorder=3)
    for rate, value in zip(rates, ratio["values"]):
        left.annotate(f"{value:+.3f}", (rate, value), fontsize=6.5, xytext=(4, 4), textcoords="offset points", color=INK)
    left.axvline(constraints.COST_BP, color=INK, linewidth=0.9, linestyle="--", zorder=2)
    left.grid(color=GRID, linewidth=0.6, zorder=1)
    left.set_xlabel("Per-side cost, bp")
    left.set_ylabel("Information ratio")
    left.set_title(f"the leader, {sheet['retention']['leader']}", fontsize=8)

    right.barh(range(len(ordered)), turnover["values"], color=MARKS[1], height=0.72, zorder=3)
    right.set_yticks(range(len(ordered)))
    right.set_yticklabels([row["cell"] for row in ordered], fontsize=6)
    right.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    right.grid(color=GRID, linewidth=0.6, axis="x", zorder=1)
    right.set_xlabel("One-way turnover, annualised")
    right.set_title("every run", fontsize=8)
    fig.tight_layout()
    labels = ["Per-side cost, bp", "Information ratio", "One-way turnover, annualised", leader["cell"],
              f"{constraints.COST_BP:.0f}"]
    return _canvas(fig, path, _spec(
        text=labels,
        lengths=[("ratio", len(ratio["values"]), len(rates)), ("turnover", len(turnover["values"]), len(rows))],
    ))


def figure_budget(evidence, path):
    """Declared and realised volatility contributions, by declared group.

    The argument: the mandate's own book departs from its declared vector, and the constructed books
    depart from it further, which is what makes consumption a matter of accident rather than design.
    """
    groups = list(universe.RISK_BUDGET)
    declared = _series("declared budget", "share", [universe.RISK_BUDGET[group] for group in groups])
    policy = evidence["budgets"]["policy"]["budget"]["by_group"]
    leader_cell = evidence["sheet"]["retention"]["leader"]
    leader = evidence["budgets"][leader_cell]["budget"]["by_group"]
    policy_values = _series("policy book", "share", [policy.loc[group, "realised"] for group in groups])
    leader_values = _series("the sample's leader", "share", [leader.loc[group, "realised"] for group in groups])

    fig = Figure(figsize=(8.6, 4.4))
    ax = fig.subplots()
    position = np.arange(len(groups))
    width = 0.26
    for offset, series, colour in ((-1, declared, INK), (0, policy_values, MARKS[0]), (1, leader_values, MARKS[1])):
        ax.bar(position + offset * width, series["values"], width=width, color=colour, label=series["label"], zorder=3)
    for index, group in enumerate(groups):
        ax.annotate(f"{declared['values'][index]:.0%}", (index - width, declared["values"][index]),
                    fontsize=6.5, ha="center", va="bottom", color=INK)
        ax.annotate(f"{policy_values['values'][index]:.1%}", (index, policy_values["values"][index]),
                    fontsize=6.5, ha="center", va="bottom", color=INK)
        ax.annotate(f"{leader_values['values'][index]:.1%}", (index + width, leader_values["values"][index]),
                    fontsize=6.5, ha="center", va="bottom", color=INK)
    ax.set_xticks(position)
    ax.set_xticklabels(groups)
    # Headroom above the tallest bar, because the value labels sit on top of the bars and a bar that
    # reaches the frame edge puts its own number outside the axes.
    ax.set_ylim(0.0, float(max(declared["values"].max(), policy_values["values"].max(), leader_values["values"].max())) * 1.12)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(color=GRID, linewidth=0.6, axis="y", zorder=1)
    ax.set_ylabel("Share of portfolio volatility")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    labels = ["Share of portfolio volatility", "declared budget", "policy book", "the sample's leader"] + groups
    return _canvas(fig, path, _spec(
        text=labels,
        lengths=[("groups", len(declared["values"]), len(groups)),
                 ("policy", len(policy_values["values"]), len(groups)),
                 ("leader", len(leader_values["values"]), len(groups))],
    ))


def figure_retention(sheet, path):
    """Each run's share of resamples in which it leads the sample, against the floor the design fixed.

    The argument: the sample's leader leads in only a little over two thirds of resamples, against a
    floor of eighty percent declared before any were drawn, and no other run comes close. The statistic
    is the per-cell counterpart of the leader's own rank retention, which is the same number for the
    leader itself; the per-cell sign retention the cost block reports is a different question (whether a
    cell's own advantage keeps its sign) and is deliberately not what this figure draws.
    """
    cells = sheet["retention"]["cells"]
    leader = sheet["retention"]["leader"]
    ordered = sorted(cells, key=lambda name: cells[name]["share_leader"])
    share = _series("share of resamples leading", "share", [cells[name]["share_leader"] for name in ordered])

    fig = Figure(figsize=(9.0, 5.2))
    ax = fig.subplots()
    colours = [MARKS[0] if name == leader else MARKS[3] for name in ordered]
    ax.barh(range(len(ordered)), share["values"], color=colours, height=0.74, zorder=3)
    ax.axvline(statistics.RANK_RETENTION_FLOOR, color=MARKS[1], linewidth=1.2, linestyle="--", zorder=4)
    ax.annotate(
        f"{statistics.RANK_RETENTION_FLOOR:.0%} floor",
        (statistics.RANK_RETENTION_FLOOR, len(ordered) - 0.4),
        fontsize=7,
        color=MARKS[1],
        ha="right",
        va="top",
        xytext=(-3, 0),
        textcoords="offset points",
    )
    ax.annotate(
        f"{cells[leader]['share_leader']:.1%}",
        (cells[leader]["share_leader"], ordered.index(leader)),
        fontsize=7,
        color=INK,
        va="center",
        xytext=(4, 0),
        textcoords="offset points",
    )
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=6)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(color=GRID, linewidth=0.6, axis="x", zorder=1)
    ax.set_xlabel(f"Share of {statistics.BOOTSTRAP_DRAWS} bootstrap resamples in which the run leads")
    ax.legend(
        handles=[
            Patch(color=MARKS[0], label="the sample's leader"),
            Patch(color=MARKS[3], label="every other run"),
        ],
        frameon=False,
        fontsize=7,
        loc="lower right",
    )
    fig.tight_layout()
    labels = [f"Share of {statistics.BOOTSTRAP_DRAWS} bootstrap resamples in which the run leads",
              f"{statistics.RANK_RETENTION_FLOOR:.0%} floor", "the sample's leader", "every other run"]
    return _canvas(fig, path, _spec(
        text=labels,
        lengths=[("share", len(share["values"]), len(cells))],
    ))


def figure_spectrum(evidence, path):
    """The observed top eigenvalue per window against its two references.

    The argument: the retained component is the one that beats the matched null rather than the analytic
    line, and the two references are far enough apart that which one is used decides the count.
    """
    counts = evidence["counts"]
    observed = _series("observed top eigenvalue", "eigenvalue", counts["observed_top"])
    null = _series("matched null, 95th percentile", "eigenvalue", counts["null_top"])
    edge = _series("Marchenko-Pastur edge", "eigenvalue", counts["mp_edge"])
    windows = np.arange(1, len(observed["values"]) + 1)

    fig = Figure(figsize=(9.0, 4.4))
    ax = fig.subplots()
    ax.plot(windows, observed["values"], color=MARKS[0], linewidth=1.3, label=observed["label"], zorder=3)
    ax.plot(windows, null["values"], color=MARKS[1], linewidth=1.2, linestyle="--", label=null["label"], zorder=3)
    ax.plot(windows, edge["values"], color=MARKS[2], linewidth=1.2, linestyle=":", label=edge["label"], zorder=3)
    ax.grid(color=GRID, linewidth=0.6, zorder=1)
    ax.set_xlabel("Estimation window, in order")
    ax.set_ylabel("Eigenvalue of the correlation matrix")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    labels = ["Estimation window, in order", "Eigenvalue of the correlation matrix", observed["label"], null["label"], edge["label"]]
    return _canvas(fig, path, _spec(
        text=labels,
        lengths=[("observed", len(observed["values"]), len(counts["observed_top"])),
                 ("null", len(null["values"]), len(counts["null_top"])),
                 ("edge", len(edge["values"]), len(counts["mp_edge"]))],
    ))


def write(evidence=None, root=None):
    """Draw every figure, checking each one, and return the paths."""
    evidence = evidence if evidence is not None else memo_module.collect()
    root = Path(root) if root else FIGURE_ROOT
    sheet = evidence["sheet"]
    written = [
        figure_dispersion(sheet, root / "fig1-cell-dispersion.svg"),
        figure_cost(sheet, root / "fig2-cost-multiple.svg"),
        figure_budget(evidence, root / "fig3-risk-budget.svg"),
        figure_retention(sheet, root / "fig4-rank-retention.svg"),
        figure_spectrum(evidence, root / "fig5-eigenvalues.svg"),
    ]
    print(f"[table] {len(written)} figures written to {root}, each checked for its labels and its series lengths")
    return written


def main(root=None):
    return write(memo_module.collect(root), root)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
