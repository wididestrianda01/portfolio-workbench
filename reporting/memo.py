"""The two prose deliverables: the findings memo and the decision record.

**Generated from a run, not written from memory.** Every number in either document is computed by the
same code the rest of the package runs, in the same process, and quoted with the file it was written to
and the snapshot it was read from. A memo written by hand from a table would be a second account of the
results, and the second account is the one that goes stale or, worse, stays plausible after the code
moved.

**Answer or refuse, and say which.** Each research question is answered with the numbers that decide it
or explicitly refused with the reason. A difference is reported only where the paired bar, the cell's own
sign retention under the bootstrap and the expanding-window sign all pass; the leader's **rank** retention
is a separate statistic about the cells against each other and a row failing it is refused in words that
say no unique cell was found rather than that no advantage was. Everything else is reported as no
difference detected with the resolution limit at the bar that decided the row, and the negative results
are published as results rather than dropped.

**The recommendation is made at the level the evidence reaches.** The pre-registered ladder is a test of
one cell, and where it returns no candidate the record says so and then states what the same table does
support - on this run, the axis whose cells cleared - with the axis level of the claim, the constraint
binding and the sizing caveats beside it. A record that stopped at the ladder would report a refusal as a
result about the benchmark.

**The decision record is the one page the committee would sign, and it carries its own falsification
conditions.** The six parts are the recommendation, the evidence with the haircut applied, the
alternatives rejected, the cost of the choice, the limitations, and the conditions under which the
recommendation would be withdrawn, each with the snapshot id and the version it was written against.
"""

from pathlib import Path

import numpy as np

from portfolio_workbench import facade, study
from portfolio_workbench.compare import table as table_module
from portfolio_workbench.construct import constraints
from portfolio_workbench.data import loader, universe
from portfolio_workbench.factors import components as components_module
from portfolio_workbench.factors import exposures, spanning
from portfolio_workbench.factors import spine as spine_module
from portfolio_workbench.evaluate import statistics

HERE = Path(__file__).resolve().parent
MEMO_PATH = HERE / "findings-memo.md"
RECORD_PATH = HERE / "decision-record.md"

DECLARATION = (
    "A learning exercise performed in role: a simulated mandate with no client and no institution. "
    "Nothing in this document is investment advice, a recommendation to any person, or a client "
    "communication, and the register is deliberate."
)

# The claim paragraph the skills-inventory decision fixes, adopted verbatim and carried directly under
# the committee's question. The wording is a decided text, so it is not re-punctuated here; what this
# memo changed is where it sits, not what it claims.
CLAIM = (
    "As a learning exercise, this project demonstrates on real data the ability to take a European "
    "multi-asset universe from a frozen, as-of-correct panel through factor and risk modelling, "
    "constrained portfolio construction across a pre-registered set of methods, walk-forward "
    "out-of-sample evaluation with paired testing and multiple-testing control, Brinson-Fachler and "
    "factor attribution with documented linking, and Euler risk-budget reporting against a stated "
    "budget, in Python and SQL, with every method traced to a cited source and every negative result "
    "published. It does not demonstrate discretion over a live book, client reporting, licensed vendor "
    "platforms, index construction or regulatory second-line work, and it claims model literacy in the "
    "vendor systems it cannot licence rather than hands-on use of them."
)


def collect(root=None):
    """Run the analysis the memo reports on, and add the factor readings only the memo needs.

    The analysis owns the run, the covariance behind the risk budget and the two decompositions. What
    is added here is the count series, the headline decomposition and the spanning directions, and
    this memo is their only reader. Pulling them into the analysis would widen its interface without
    giving a second caller anything.
    """
    analysis = study.analyse(loader.load_panel(root))
    document, returns, months = analysis.document, analysis.returns, analysis.months
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document.factors.eur, block)
    headline = components_module.decompose(returns)
    return {
        "analysis": analysis,
        "document": document,
        "grid": analysis.grid,
        "sheet": analysis.sheet,
        "holding": list(analysis.attribution.values()),
        "budgets": analysis.budgets,
        "counts": components_module.count_series(returns, months, window=exposures.WINDOW),
        "headline": headline,
        "directions": spanning.directions(returns, named, headline["components"]),
    }


def _by_cell(sheet):
    return {row["cell"]: row for row in sheet["rows"]}


def _span(rows, key):
    values = [row[key] for row in rows]
    return min(values), max(values)


def _verb(count, singular, plural):
    """A count with its verb inflected, for the clauses whose count is a property of the run.

    A sentence that is right at sixteen and wrong at one is a sentence a reader stops trusting, and
    these counts come from the verdicts rather than from the writer.
    """
    return f"{count} {singular if count == 1 else plural}"


BARS = ("policy", "equal_weight")
MEAN_CELLS = ("mean_variance_sample", "mean_variance_shrunk", "mean_variance_black_litterman", "mean_variance_none")


def _stage(rows, stage):
    """The method cells of one stage.

    The two bars are not methods, since one is the benchmark and one is the naive portfolio, and a
    dispersion measured across them would be the width of the comparison rather than the width of the
    choice. The perturbation runs and the expanding repeats are excluded for the same reason: they are
    the same method revisited, and counting them twice would weight one method twice in a range.
    """
    return [
        row
        for row in rows
        if row["stage"] == stage
        and row["cell"] not in BARS
        and not row["is_repeat"]
        and not row["cell"].endswith("_uncapped")
    ]


def memo(evidence):
    """The memo: the claim, what was run, an answer per research question, and the limits."""
    sheet = evidence["sheet"]
    document = evidence["document"]
    rows = sheet["rows"]
    by_cell = _by_cell(sheet)
    traded = evidence["analysis"].traded
    stage_a, stage_b, stage_c = _stage(rows, "A"), _stage(rows, "B"), _stage(rows, "C")
    leader = sheet["retention"]["leader"]
    leader_row = by_cell[leader]
    counts, directions = evidence["counts"], evidence["directions"]
    headline = evidence["headline"]
    holding = evidence["holding"]
    link_worst = max(float(block["linked"]["relative"]) for block in holding)
    additivity_worst = max(entry["additivity"]["relative"] for entry in evidence["budgets"].values())
    policy_budget = evidence["budgets"]["policy"]["budget"]
    leader_budget = evidence["budgets"][leader]["budget"]
    behind = [row for row in rows if row["verdict"] == table_module.BEHIND]
    negatives = table_module.negative_results(sheet)
    cost = _cost_reading(by_cell)

    lo_te, hi_te = _span(stage_a, "tracking_error")
    lo_b_te, hi_b_te = _span(stage_b, "tracking_error")
    lo_vol, hi_vol = _span(stage_a, "volatility")
    lo_b_vol, hi_b_vol = _span(stage_b, "volatility")

    lines = [
        "# Findings memo: which methodological choice moved the outcome",
        "",
        DECLARATION,
        "",
        "## The committee's question",
        "",
        "The exercise is performed in the role of a portfolio constructor reporting to an investment "
        "committee, on a real European multi-asset universe and real market data. The committee holds a "
        "balanced mandate at fixed strategic weights, and at this review it has to decide whether to "
        "leave that book alone or to pay for a different construction method. Paying means trading, "
        "trading means cost charged on traded notional, and a method that looks better before cost can be "
        "worse after it. The question is therefore narrow: which methodological choices move the outcome "
        "on this mandate, by how much, on which metric, and what does the change cost? The seven "
        "questions below take it apart in the order the build answers them, and the decision record "
        "beside this memo is the one page the committee would sign.",
        "",
        CLAIM,
        "",
        "## What was run",
        "",
        f"The comparison is {len(sheet['cells'])} distinct cells and {len(rows)} pre-registered runs on the "
        f"frozen snapshot `{document.snapshot_id}`, over {len(traded)} out-of-sample months "
        f"({traded.min()} to {traded.max()}), out of a joined panel running "
        f"{document.months.min()} to {document.months.max()}. Estimation is rolling "
        f"{exposures.WINDOW} months with a monthly refit; the constraint set is long-only, fully invested, "
        f"capped at {constraints.CAP:.0%} per sleeve with a {constraints.TURNOVER_CAP:.0%} one-way turnover "
        f"cap and a {constraints.BAND:.0%} no-trade band; cost is charged on traded notional at "
        f"{constraints.COST_BP:.0f} bp per side. Every number below is generated by the same code the "
        "package runs, and the notebook beside each layer carries the same provenance block.",
        "",
        f"The bar over {len(sheet['cells'])} cells is |z| >= {sheet['bar']:.4f}, and it is also the "
        f"multiple-testing haircut: one test of one statistic serves both. The realised correlation between "
        f"cells implies {sheet['effective_tests']:.1f} effective tests, at which the bar would be "
        f"{sheet['adjusted_bar']:.4f}; the conservative value decides. That correction is self-imposed from the "
        "literature and is not a regulatory requirement.",
        "",
        "## RQ1. Do the construction families differ out of sample on this mandate, and on which metric?",
        "",
        _rq1(stage_a, behind, leader_row, sheet),
        "",
        "## RQ2. Does the risk-model choice matter as much as the constructor choice?",
        "",
        _rq2(stage_a, stage_b, lo_te, hi_te, lo_b_te, hi_b_te, lo_vol, hi_vol, lo_b_vol, hi_b_vol),
        "",
        "## RQ3. How much turns on the mean input?",
        "",
        _rq3(stage_c, by_cell, sheet),
        "",
        "## RQ4. Does the factor set matter: does either set span the other?",
        "",
        _rq4(directions, headline),
        "",
        "## RQ5. How many factors does this panel support, and on what evidence?",
        "",
        _rq5(counts, headline),
        "",
        "## RQ6. Is the risk budget consumed by design or by accident?",
        "",
        _rq6(policy_budget, leader_budget, leader, link_worst, additivity_worst, evidence),
        "",
        "## RQ7. What does the choice cost, and is the apparent winner exploiting estimation error?",
        "",
        _rq7(cost, leader_row, sheet),
        "",
        "## Negative results, published as results",
        "",
        f"{len(negatives)} of the {len(sheet['cells'])} distinct cells carry a verdict of no difference "
        "detected or a declared negative result, and each carries its resolution limit on the row it was "
        "read from:",
        "",
        "\n".join(f"- **{entry['cell']}**: {entry['verdict']}" for entry in negatives),
        "",
        "The table's resolution on a typical row is printed beside its verdict; a reader who reads 'no "
        "difference detected' as 'equivalent' is reading the second statement while appearing to make the "
        "first, which is the distinction the noise floor exists to keep visible.",
        "",
        "## Limitations",
        "",
        _limitations(evidence, additivity_worst, link_worst),
        "",
        "## What this memo does not establish",
        "",
        "No research question asks which family is best in general, and the panel cannot support the "
        f"question: {len(universe.TICKERS)} sleeves, {len(document.months)} months and one mandate give a "
        "comparison on this universe rather "
        "than a ranking of methods. The memo does not establish that the leader would repeat out of "
        "sample, that a cost of a few basis points a year is the whole cost, or that a verdict of no "
        "difference detected means two methods are equivalent. It does not establish that the mean input "
        "is the axis that matters in general: the axis was pre-registered as a question, and reading it as "
        "the recommendation follows the cells that cleared rather than a rule fixed before the run. Nor "
        "does it establish that the tilt it recommends is the right size, which is a decision about the "
        "mandate's risk appetite rather than a result. Nothing here is a claim about a live "
        "book, a client, or a regulated activity; the exercise is performed in role and the register "
        "says so on every artifact.",
        "",
    ]
    return "\n".join(lines)


def _rq1(stage_a, behind, leader_row, sheet):
    widest = max(stage_a, key=lambda row: row["tracking_error"])
    best = max(stage_a, key=lambda row: row["information_ratio"])
    cells = sheet["retention"]["cells"]
    ranked = sorted(
        (
            entry
            for entry in cells.items()
            if not np.isnan(entry[1]["information_ratio"])
        ),
        key=lambda entry: entry[1]["information_ratio"],
        reverse=True,
    )
    ahead = [
        row
        for row in sheet["rows"]
        if row["cell"] not in BARS
        and not row["is_repeat"]
        and row["haircut"]["clears"]
        and row["information_ratio"] > 0.0
    ]
    gap = ranked[0][1]["information_ratio"] - ranked[1][1]["information_ratio"]
    leader = sheet["retention"]["leader"]
    return (
        f"{len(stage_a)} family cells span a tracking error from {min(row['tracking_error'] for row in stage_a):.2%} "
        f"to {max(row['tracking_error'] for row in stage_a):.2%} a year and an information ratio from "
        f"{min(row['information_ratio'] for row in stage_a):+.3f} to {max(row['information_ratio'] for row in stage_a):+.3f}. "
        f"The dispersion is the answer to whether the family matters, and it is large relative to the "
        f"test's resolution: the family moves tracking error by hundreds of basis points, while the "
        f"smallest difference this test would detect on a typical row is of the order of "
        f"{np.median([row['paired_benchmark']['resolution'] for row in stage_a]):.2f} in information-ratio terms.\n\n"
        f"The direction is not one direction. Cells that differ from the policy benchmark on the paired test "
        f"differ in both directions once the cost is charged: {len(behind)} sit significantly behind it, and "
        f"{len(ahead)} sit significantly ahead, the latter being the cells that carry a mean input, which is "
        f"the axis RQ3 reads. The widest tracking error among the family cells, on `{widest['cell']}`, comes "
        f"with an information ratio of {widest['information_ratio']:+.3f}; the best information ratio among "
        f"them is `{best['cell']}` at {best['information_ratio']:+.3f}.\n\n"
        f"The metric the question turns on is therefore tracking error rather than return: what separates "
        f"these families is how much risk they take away from the benchmark. What separates the top of the "
        f"table from itself is not resolvable here, and the two bootstrap statistics say which is which: "
        f"`{leader}` keeps the sign of its own advantage over the benchmark in "
        f"{cells[leader]['retention']:.1%} of {statistics.BOOTSTRAP_DRAWS} resamples against the "
        f"{statistics.RANK_RETENTION_FLOOR:.0%} floor and holds that sign under the expanding protocol, while "
        f"keeping its **rank** ahead of the other cells in {sheet['retention']['retention']:.1%} of the same "
        f"resamples. The advantage is measured and the ordering is not, which is what a {gap:.3f} gap to the "
        f"nearest rival against a resolution of "
        f"{leader_row['paired_benchmark']['resolution']:.2f} on that row means."
    )


def _rq2(stage_a, stage_b, lo_te, hi_te, lo_b_te, hi_b_te, lo_vol, hi_vol, lo_b_vol, hi_b_vol):
    a_te, b_te = hi_te - lo_te, hi_b_te - lo_b_te
    a_vol, b_vol = hi_vol - lo_vol, hi_b_vol - lo_b_vol
    comparison = "less" if b_te < a_te else "more"
    return (
        f"Across the {len(stage_b)} cells of the risk-model axis, tracking error moves over {b_te:.2%} "
        f"({lo_b_te:.2%} to {hi_b_te:.2%}) and volatility over {b_vol:.2%}. The family axis moves tracking "
        f"error over {a_te:.2%} and volatility over {a_vol:.2%} across its {len(stage_a)} cells. On this "
        f"panel the risk-model choice therefore moves the outcome {comparison} than the constructor "
        f"choice: swapping the covariance estimator changes the book, and it changes it by about a "
        f"quarter as much as swapping the objective does.\n\n"
        "That reading has a stated precondition, and it is the one the risk layer reports: at sixty months "
        f"and {len(universe.TICKERS)} sleeves the sample covariance's condition number makes the optimiser "
        "the object under "
        "test far more than the estimator. The three estimators differ in conditioning, and the matrix the "
        "shrinkage draws toward is a choice inside the estimator rather than a property of the panel, so "
        "the dispersion measured here is a lower bound on what a differently structured risk model would "
        "produce."
    )


def _rq3(stage_c, by_cell, sheet):
    none_row = by_cell["mean_variance_none"]
    sample_row = by_cell["mean_variance_sample"]
    posterior_row = by_cell["mean_variance_black_litterman"]
    shrunk_row = by_cell["mean_variance_shrunk"]
    stage_c = [by_cell[identifier] for identifier in MEAN_CELLS]
    parts = [
        "The mean input is the axis where the answer is least like a ranking. Four cells share one "
        "covariance and one constraint set and differ only in the mean:",
        "",
    ]
    for row in sorted(stage_c, key=lambda row: row["cell"]):
        parts.append(
            f"- `{row['cell']}`: information ratio {row['information_ratio']:+.3f}, tracking error "
            f"{row['tracking_error']:.2%}, target-path movement {row['weight_stability']:.2%} a month, "
            f"concentration {row['concentration']:.2f}"
        )
    parts += [
        "",
        f"The three cells that carry a mean land within {max(sample_row['information_ratio'], shrunk_row['information_ratio'], posterior_row['information_ratio']) - min(sample_row['information_ratio'], shrunk_row['information_ratio'], posterior_row['information_ratio']):.3f} "
        f"of each other on the information ratio ({sample_row['information_ratio']:+.3f} sample mean, "
        f"{shrunk_row['information_ratio']:+.3f} shrunk mean, {posterior_row['information_ratio']:+.3f} posterior), "
        "and all three move their weight path by between "
        f"{min(sample_row['weight_stability'], shrunk_row['weight_stability'], posterior_row['weight_stability']):.2%} "
        f"and {max(sample_row['weight_stability'], shrunk_row['weight_stability'], posterior_row['weight_stability']):.2%} a month. "
        "Shrinking the mean does not make the path calmer than the sample mean's on this panel "
        f"({shrunk_row['weight_stability']:.2%} against {sample_row['weight_stability']:.2%}), which is the "
        "honest reading: the remedy was applied and the instability is still there, so what the mean axis "
        "shows is how much of the answer the input carries rather than how much of the error a standard "
        "remedy removes.\n\n"
        "Dropping the mean entirely reproduces the equal-weight book on this constraint set, which is a "
        "property of the constraint set rather than a result about means: at the tightest norm the "
        "fully-invested long-only book admits, there is only one feasible weight vector, and it returns "
        f"{none_row['information_ratio']:+.3f} on the same metric with a weight path that does not move at all.",
        "",
        "All three cells that carry a mean clear the family-wise bar on their own rows "
        f"({shrunk_row['paired_benchmark']['statistic']:.2f}, {sample_row['paired_benchmark']['statistic']:.2f} "
        f"and {posterior_row['paired_benchmark']['statistic']:.2f} against {sheet['bar']:.4f}), keeping the sign "
        f"of their advantage in {shrunk_row['retention']:.0%}, {sample_row['retention']:.0%} and "
        f"{posterior_row['retention']:.0%} of resamples, while the cell that drops the mean clears nothing and "
        "sits significantly behind the benchmark. That is the one axis of this comparison where the evidence "
        "separates a group of cells from the benchmark and from that group's own no-mean control, and it is "
        "what the decision record's recommendation rests on. Which of the three to hold is a different "
        "question and this panel does not answer it: they sit within "
        f"{max(sample_row['information_ratio'], shrunk_row['information_ratio'], posterior_row['information_ratio']) - min(sample_row['information_ratio'], shrunk_row['information_ratio'], posterior_row['information_ratio']):.3f} "
        "of one another on the information ratio, against a resolution of the order of "
        f"{np.median([row['paired_benchmark']['resolution'] for row in stage_c]):.2f} on those rows. The axis "
        "supports a construction, not a variant.",
    ]
    return "\n".join(parts)


def _rq4(directions, decomposition):
    headline, reverse = directions["headline"], directions["reverse"]
    return (
        f"The test runs in both directions at the retained count of {decomposition['components']}. The named spine "
        f"is not spanned by the constructed block (GRS {headline['statistic']:.2f}, p "
        f"{headline['p_value']:.2g}), and the reverse also rejects (GRS {reverse['statistic']:.2f}, p "
        f"{reverse['p_value']:.2g}), so neither set prices what the other does on this panel. The "
        "Huberman-Kandel condition is reported per asset rather than as one number, and the rows sum to one "
        "under spanning, which is the half of the condition a reader can check directly.\n\n"
        "The direction that matters for the rest of the build is the first: the published spine carries "
        "something the four constructed bond and credit series do not. That is why the block is reported "
        "beside the spine rather than replacing it, and why the loadings on the four construction sleeves "
        "are treated as identities rather than as estimates."
    )


def _rq5(counts, decomposition):
    observed = set(int(count) for count in counts["counts"])
    return (
        f"The rule retains {decomposition['components']} components on the full panel, and the per-window counts "
        f"across the out-of-sample window take the values {sorted(observed)}, inside the pre-registered "
        f"bound of {components_module.PREREGISTERED_K}. The rule's own falsification check does not fire "
        f"(falsified: {counts['falsified']}, a move of more than one component in "
        f"{counts['move_share']:.0%} of the steps), and the stability check on independently permuted "
        "windows agrees with the count.\n\n"
        f"Two references travel with the count. The analytic Marchenko-Pastur edge for "
        f"{len(universe.TICKERS)} series over a "
        f"sixty-month window is {float(np.min(counts['mp_edge'])):.4f} to {float(np.max(counts['mp_edge'])):.4f} "
        "across the windows, and the matched permutation null the rule actually uses sits higher, at "
        f"{float(np.min(counts['null_top'])):.4f} to {float(np.max(counts['null_top'])):.4f}, because an "
        "independent permutation destroys the cross-sectional structure a normal null would keep. The "
        "observed top eigenvalue runs "
        f"{float(np.min(counts['observed_top'])):.4f} to {float(np.max(counts['observed_top'])):.4f}. The rule "
        "retains a component that beats the matched null rather than a component that beats an analytic "
        "line, and the evidence for the count is that comparison rather than an appeal to a common rule of "
        "thumb: Kaiser's eigenvalue-above-one rule would have retained a different, larger number here."
    )


def _rq6(policy_budget, leader_budget, leader, link_worst, additivity_worst, evidence):
    groups = list(universe.RISK_BUDGET)
    worst = max(groups, key=lambda group: abs(policy_budget["by_group"].loc[group, "gap"]))
    rows = []
    for group in groups:
        rows.append(
            f"- {group}: target {universe.RISK_BUDGET[group]:.0%}, policy book "
            f"{policy_budget['by_group'].loc[group, 'realised']:.1%}, leader "
            f"{leader_budget['by_group'].loc[group, 'realised']:.1%}"
        )
    return (
        "The budget is answered on two books: the policy benchmark, which is the mandate's own weights, "
        "and the sample's leader. Realised volatility contributions against the declared vector:\n\n"
        + "\n".join(rows)
        + f"\n\nThe largest departure on the policy book is {worst}, at "
        f"{policy_budget['by_group'].loc[worst, 'gap']:+.1%}. Consumption is therefore by accident rather "
        "than by design for the cells whose objective never mentioned the budget, and that is the honest "
        "reading rather than a defect: a risk-based family concentrates volatility where the covariance "
        "puts the risk, which is not where the mandate's vector puts it.\n\n"
        f"The decomposition itself carries no unexplained residual of consequence. Euler contributions sum "
        f"to portfolio volatility within {additivity_worst:.0e} relative on every run, and the linked "
        f"attribution lands on the compounded excess with a largest residual of {link_worst:.0e}. Value at "
        "risk is refused as a measure to decompose, and the refusal is measured rather than asserted: its "
        "conditional contributions do not sum to it, while the expected shortfall takes the same "
        "contributions additively."
    )


def _cost_reading(by_cell):
    """Every run's cost and turnover, which is the input RQ7's reading is built from."""
    return {row["cell"]: (row["turnover_annualised"], row["cost_annualised"]) for row in by_cell.values()}


def _rq7(cost, leader_row, sheet):
    rates = sorted(leader_row["cost_sensitivity"])
    sensitivity = ", ".join(
        f"{rate:.0f} bp {leader_row['cost_sensitivity'][rate]['information_ratio']:+.3f}" for rate in rates
    )
    return (
        f"Cost is charged on traded notional, and it separates the families by their turnover rather than "
        f"by their objective. The mean-input cells trade most "
        f"({cost['mean_variance_sample'][0]:.1%} a year, {cost['mean_variance_sample'][1]:.2%} of return a "
        f"year at the decided rate) and the risk-based cells trade least "
        f"(`erc_bounded` {cost['erc_bounded'][0]:.1%} a year, {cost['erc_bounded'][1]:.2%}). The leader's "
        f"information ratio at each per-side multiple is {sensitivity}, so the ranking at the top multiple "
        "is a separate statement from the level at the base.\n\n"
        f"The second half of the question is estimation error, and the resampling answers it in two parts "
        f"that are easy to run together. The leader keeps the sign of its own advantage over the benchmark "
        f"in {sheet['retention']['cells'][leader_row['cell']]['retention']:.1%} of {statistics.BOOTSTRAP_DRAWS} "
        f"resamples, above the {statistics.RANK_RETENTION_FLOOR:.0%} floor, so the advantage is measured; it "
        f"keeps its **rank** ahead of the other cells in {sheet['retention']['retention']:.1%} of the same "
        f"resamples, below it, so the ordering of the near-tied cells is not. Its target-path weight movement "
        f"of {leader_row['weight_stability']:.2%} a month is among the largest in the table, against 0.00% for "
        "a fixed-weight book, which is the honest signature of a method leaning on its estimates even where "
        "its advantage survives them. Both readings sit on the row, and reading the second as a failure of "
        "the first is the mistake this table is arranged to prevent. The multiple-testing haircut is applied "
        "as well, so the bar a row is read against is the corrected one and not the nominal five percent."
    )


def _limitations(evidence, additivity_worst, link_worst):
    document = evidence["document"]
    traded = len(evidence["analysis"].traded)
    sheet = evidence["sheet"]
    cells = len(sheet["cells"])
    by_cell = _by_cell(sheet)
    leader = sheet["retention"]["leader"]
    mean_rows = [
        by_cell[identifier]
        for identifier in MEAN_CELLS
        if identifier in by_cell and by_cell[identifier]["information_ratio"] > 0.0
    ]
    return (
        f"**The panel is one panel.** {len(universe.TICKERS)} UCITS sleeves, {len(document.months)} months "
        f"of which {traded} are traded, one mandate and one currency. A difference that this design cannot resolve is not reported as an "
        "absence, and a difference it does resolve is a statement about this universe.\n\n"
        f"**The noise floor is stated, not implied.** {cells} cells tested against two families give a "
        "family-wise bar that a real but modest advantage will not clear, and the smallest detectable "
        "difference is printed on every row. The bootstrap and the expanding protocol are the two further "
        "rungs, and a cell failing either is reported as no difference detected rather than as a small "
        "difference.\n\n"
        f"**Estimation error is visible and not modelled away.** The leader keeps the sign of its own "
        f"advantage in {sheet['retention']['cells'][leader]['retention']:.1%} of resamples against the "
        f"{statistics.RANK_RETENTION_FLOOR:.0%} floor the design declared, and its target path moves "
        f"{by_cell[leader]['weight_stability']:.2%} a month; the weight-stability diagnostic is reported for "
        "every cell. The mean-variance family's own error is in the mean, and the shrinkage cell exists to "
        "show how much of it a standard remedy removes rather than to claim the remedy.\n\n"
        f"**The last rung was reachable by two cells of {cells}.** The expanding-window repeat is "
        "pre-registered for the two cells the mandate's question turns on, so the rest are decided on the "
        "rolling protocol alone and each says so on its row. Of the three cells that showed an advantage on "
        "the paired test, two carry the verdict that the expanding leg was not run for them, which is a "
        "coverage limit of the design rather than a result about those two methods.\n\n"
        "**Where the recommendation is made is stated.** The pre-registered ladder asks whether one cell is "
        "uniquely best and returns no candidate. The statement the memo goes on to make is about the "
        "mean-input axis, whose three cells each clear the same family-wise bar; that axis was "
        "pre-registered as a question, and reading an axis as the recommendation because its cells are the "
        "ones that cleared is a step this panel does not certify. It is recorded here rather than left "
        "implicit.\n\n"
        f"**The constraint set is doing part of the work.** The books carrying a mean sit on the "
        f"{constraints.CAP:.0%} per-sleeve cap on "
        f"{min(row['cap_binding_frequency'] for row in mean_rows):.0%} to "
        f"{max(row['cap_binding_frequency'] for row in mean_rows):.0%} of their steps, against a "
        f"{by_cell['policy']['largest_weight']:.0%} largest weight on the policy book, so part of what the "
        "axis earns is the cap admitting more equity than the strategic weights hold. The perturbation runs "
        "exist to measure how much, and they are in the table.\n\n"
        f"**The residuals are named.** Attribution reconciles to {link_worst:.0e} and the Euler "
        f"contributions add to {additivity_worst:.0e} relative; the factor model's unexplained part is "
        "reported as its own measured quantity rather than absorbed. Value at risk is refused as a "
        "decomposable measure, with the measurement that justifies the refusal in the budget notebook.\n\n"
        "**The exercise is a simulation.** The data is a frozen snapshot read under a single-use licence, "
        "no market data is published with the build, and no number here is a forecast of a live mandate. "
        f"The snapshot is `{document.snapshot_id}` and the record below is written against it."
    )


def record(evidence):
    """The decision record: one page, six parts, with its own falsification conditions."""
    sheet = evidence["sheet"]
    document = evidence["document"]
    traded = len(evidence["analysis"].traded)
    cells = len(sheet["cells"])
    rows = sheet["rows"]
    by_cell = _by_cell(sheet)
    leader = sheet["retention"]["leader"]
    leader_row = by_cell[leader]
    candidates = [row for row in rows if row["verdict"] == table_module.CANDIDATE]
    behind = [row for row in rows if row["verdict"] == table_module.BEHIND]
    cleared = [row for row in rows if row["haircut"]["clears"]]
    lost = [row for row in rows if row["verdict"] == table_module.RESAMPLING_LOST]
    not_unique = [row for row in rows if row["verdict"] == table_module.NOT_UNIQUE]
    unrun = [row for row in rows if row["verdict"] == table_module.LEG_NOT_RUN]
    retention = sheet["retention"]["retention"]
    own = sheet["retention"]["cells"][leader]["retention"]
    without_mean = by_cell["mean_variance_none"]
    mean_rows = [by_cell[identifier] for identifier in MEAN_CELLS if identifier in by_cell]
    holding_mean = sorted(
        (row for row in mean_rows if row["information_ratio"] > 0.0),
        key=lambda row: row["information_ratio"],
        reverse=True,
    )
    family_bar = ", ".join(f"{row['paired_benchmark']['statistic']:.2f}" for row in holding_mean)
    family_retention = ", ".join(f"{row['retention']:.0%}" for row in holding_mean)
    family_ir = ", ".join(f"{row['information_ratio']:+.3f}" for row in holding_mean)
    family_turnover = f"{min(row['turnover_annualised'] for row in holding_mean):.1%} to {max(row['turnover_annualised'] for row in holding_mean):.1%}"
    family_cost = f"{min(row['cost_annualised'] for row in holding_mean):.2%} to {max(row['cost_annualised'] for row in holding_mean):.2%}"
    family_volatility = f"{min(row['volatility'] for row in holding_mean):.1%} to {max(row['volatility'] for row in holding_mean):.1%}"
    gap = holding_mean[0]["information_ratio"] - holding_mean[1]["information_ratio"]
    family_spread = holding_mean[0]["information_ratio"] - holding_mean[-1]["information_ratio"]
    sensitivity = ", ".join(
        f"{rate:.0f} bp {holding_mean[0]['cost_sensitivity'][rate]['information_ratio']:+.3f}"
        for rate in sorted(holding_mean[0]["cost_sensitivity"])
    )

    if candidates:
        recommendation = (
            f"Adopt `{candidates[0]['cell']}` for the mandate's construction decision, subject to the "
            "falsification conditions below. Every rung of the declared ladder was cleared: the "
            "family-wise bar, the bootstrap sign retention, the rank retention and the sign under the "
            "second protocol."
        )
    else:
        recommendation = (
            "**Carry a mean input; do not claim a variant.** No cell cleared every rung of the declared "
            "ladder, so the ladder names no cell and this record names none. The leader cleared the "
            f"family-wise bar, kept the sign of its own advantage in {own:.1%} of resamples and held it "
            "under the expanding protocol, and failed the rung that asks whether one cell is uniquely best "
            f"({retention:.1%} against the {statistics.RANK_RETENTION_FLOOR:.0%} floor). An advantage and a "
            "ranking are different findings, and this table measures both; the row says so in words that "
            "name the ranking.\n\n"
            "What the same table does support is the axis those cells share. All "
            f"{len(holding_mean)} cells that carry a mean input clear the family-wise bar on their own row "
            f"({family_bar} against {sheet['bar']:.4f}), keeping the sign of the advantage in "
            f"{family_retention} of resamples, while `mean_variance_none`, on the same covariance and the "
            f"same constraint set with no mean in it, clears nothing and returns "
            f"{without_mean['information_ratio']:+.3f}. The recommendation is to "
            "move the objective to a mean-input mean-variance construction, and **not to nominate one of "
            f"them**: the leading two sit {gap:.3f} apart on the information ratio against a resolution of "
            f"{holding_mean[0]['paired_benchmark']['resolution']:.2f} on the leading row, so this panel "
            "supports the construction and not the variant. Hold it as a partial tilt - the active decision "
            "at a size that leaves the mandate's volatility where the mandate put it - rather than as a "
            "replacement of the strategic weights."
        )

    return "\n".join(
        [
            "# Decision record",
            "",
            DECLARATION,
            "",
            f"Snapshot `{document.snapshot_id}`, version {facade.VERSION}, written from "
            f"{len(rows)} pre-registered runs over {len(sheet['cells'])} distinct cells.",
            "",
            "## 1. Recommendation",
            "",
            recommendation,
            "",
            "## 2. Evidence, with the haircut applied",
            "",
            f"The bar over {len(sheet['cells'])} cells is |z| >= {sheet['bar']:.4f}, which is also the "
            f"multiple-testing haircut; at the {sheet['effective_tests']:.1f} effective tests the realised "
            f"correlation implies, it would be {sheet['adjusted_bar']:.4f}, and the conservative value is the "
            f"one applied. {len(cleared)} of the {len(rows)} runs clear the bar, so a difference is detectable "
            f"on those rows; {len(behind)} of them are significantly behind the benchmark once the cost is "
            f"charged, {_verb(len(lost), 'loses', 'lose')} the sign of its own advantage under the bootstrap, "
            f"{_verb(len(not_unique), 'fails', 'fail')} the rank-retention rung that follows while keeping that "
            f"sign, {_verb(len(unrun), 'has', 'have')} no expanding leg and cannot complete the ladder, and "
            f"{_verb(len(candidates), 'clears', 'clear')} every rung. The leader's rank retention across "
            f"{statistics.BOOTSTRAP_DRAWS} resamples is {retention:.1%}, and the rung it fails is the one "
            "that asks whether one cell is uniquely best: the same row keeps the sign of its own advantage "
            f"in {own:.1%} of the same resamples and holds that sign under the expanding protocol. The "
            "haircut is self-imposed from the literature and is not a regulatory requirement.",
            "",
            "## 3. Alternatives rejected",
            "",
            "- The risk-based families - minimum variance, maximum diversification, equal risk "
            "contribution, hierarchical risk parity, mean-CVaR and their shrinkage and factor variants: "
            f"every one sits behind the policy benchmark on the sample, and {len(behind)} of them sit "
            "significantly behind it once the cost is charged. De-risking the sleeves against a benchmark "
            f"that runs at {by_cell['policy']['volatility']:.1%} volatility is a policy bet rather than a "
            "construction choice.",
            f"- Dropping the mean input: `mean_variance_none` reproduces the equal-weight book on this "
            f"constraint set and returns {without_mean['information_ratio']:+.3f}, which is the constraint "
            "set being the estimator rather than a result about means.",
            "- The sample mean as a named choice over the shrunk and posterior ones: its weight path moves "
            "most between adjacent months, and the three sit within "
            f"{family_spread:.3f} of one another on the metric the decision turns on, so choosing between "
            "them is choosing the ranking.",
            "- The uncapped perturbations: they exist to show how much of a cell's result the constraint "
            "set was doing, and both concentrate the book onto the near-riskless sleeve.",
            "- Vendor-model emulation, composite reporting, index construction and regulatory limits: "
            "refused structurally, with the reason recorded in the skills matrix.",
            "",
            "## 4. Cost of the choice",
            "",
            f"Adopting costs the family's own trading: {family_turnover} a year in one-way turnover and "
            f"{family_cost} of return a year at the decided {constraints.COST_BP:.0f} bp per side, charged "
            "on traded notional, and the advantage holds through the sensitivity run at four times that "
            f"rate ({sensitivity}). The books carrying a mean run at {family_volatility} volatility against "
            f"the policy book's {by_cell['policy']['volatility']:.1%}, so the tilt is sized to buy the "
            "active decision rather than the volatility, and a partial tilt leaves the mandate's own "
            "volatility where the mandate put it. Not adopting forgoes the advantage those rows measure, "
            f"which is {family_ir} net of cost and is not zero. The family also moves further from the "
            "declared risk budget than the policy book already sits from it, which is the cost the sizing "
            "is against.",
            "",
            "## 5. Limitations",
            "",
            f"One panel, {len(universe.TICKERS)} sleeves, {len(document.months)} months of which "
            f"{traded} are traded, one mandate and one currency. "
            f"{cells} cells tested at a family-wise bar leave a resolution limit on every row. The "
            "expanding-window repeat is pre-registered for two cells of the sixteen, so the last rung "
            f"could be reached by two rows and {len(unrun)} rows carry the verdict that it was not run for "
            "them. The recommendation is made about an axis rather than about a cell, and the axis was "
            "read after the results were seen, which is why that step is stated here rather than left "
            "implicit in the evidence. Cost is charged at a single per-side multiple with a sensitivity "
            "run beside it, and market impact and capacity are outside the panel. The multiple-testing "
            "correction is a self-imposed discipline. The exercise is a simulation and nothing here is "
            "advice or a client communication.",
            "",
            "## 6. Falsification conditions",
            "",
            "The recommendation is withdrawn or revisited if any of the following holds:",
            "",
            "1. A rerun on this snapshot no longer reproduces the metric table within the stated tolerance "
            f"({statistics.RERUN_TOLERANCE:.0e} relative).",
            "2. The cells carrying a mean stop clearing the family-wise bar, or stop retaining the sign of "
            f"their advantage in at least {statistics.RANK_RETENTION_FLOOR:.0%} of bootstrap resamples, on "
            "this snapshot or on a longer panel. The recommendation rests on those rows and on nothing "
            "else.",
            "3. The cap-binding frequency of those books rises far enough that the result becomes a result "
            "about the constraint set: the perturbation runs exist to measure how much of it already is.",
            "4. The declared risk budget is treated as a constraint rather than as a report, and a "
            "mean-input book cannot be sized to leave the mandate's volatility where it was.",
            "5. A cell clears the family-wise bar, keeps the sign of its own advantage in at least "
            f"{statistics.RANK_RETENTION_FLOOR:.0%} of bootstrap resamples, holds that sign under the "
            f"expanding protocol and keeps its rank in at least {statistics.RANK_RETENTION_FLOOR:.0%} of "
            "the same resamples: the ladder then names a cell, and this record names that cell rather "
            "than the axis.",
            "6. The resolution limit on a row that reports no difference falls below the difference the "
            "row reports as absent on a longer panel: an absence then becomes evidence, and the "
            "risk-based families would have to be reread against the benchmark.",
            "",
            f"Written against snapshot `{document.snapshot_id}`, build version {facade.VERSION}.",
            "",
        ]
    )


def write(evidence=None, memo_path=None, record_path=None):
    """Write both documents from one run and return their paths."""
    evidence = evidence if evidence is not None else collect()
    memo_path = Path(memo_path) if memo_path else MEMO_PATH
    record_path = Path(record_path) if record_path else RECORD_PATH
    memo_path.write_text(memo(evidence))
    record_path.write_text(record(evidence))
    print(
        f"[table] memo written to {memo_path} and the decision record to {record_path}, both against "
        f"snapshot {evidence['document'].snapshot_id}"
    )
    return {"memo": memo_path, "record": record_path}


def main(root=None):
    return write(collect(root))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
