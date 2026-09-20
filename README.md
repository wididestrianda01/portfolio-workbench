# portfolio-workbench

A methodology comparison for European multi-asset portfolio construction, run out of sample on one
frozen panel. A committee holding a balanced mandate at fixed strategic weights asks a question it asks
at every review: would a different construction method have produced a better book once cost is
charged? This repository answers it by running the candidates side by side on one universe and one
mandate, and reporting which choices moved the outcome, on which metric, and at what cost.

*A learning exercise performed in role: a simulated mandate with no client and no institution.
Nothing in this repository is investment advice, a recommendation to any person, or a client
communication.*

## What this does

A mandate is a set of weights, a universe and a benchmark, and construction is the act of choosing
those weights. Practitioners choose between families that ask different questions of the same inputs:
minimum variance asks which book has the smallest variance, equal risk contribution asks which makes
every sleeve's contribution to volatility equal, maximum diversification asks which gets the most
diversification out of the covariance, and mean-variance asks which trades expected return against
variance at a stated risk aversion. Each family consumes a different set of inputs, and the literature
is unusually united on one point: the mean is the input carrying the most estimation error.

The decision in front of a committee is narrower than a search for the best method. It can leave the
book alone, in which case it trades nothing and pays nothing, or it can adopt a constructed book, in
which case it pays the cost of getting there and of staying there. That is why the question here is not
which method is best in general. It is which methodological choices move the outcome on this mandate,
by how much, on which metric, and what the change costs.

What comes out is a comparison of sixteen cells on one universe, one mandate and one currency, and the
comparison is designed so that a difference has to earn its way past a cost charge, a family-wise bar
and two further rungs before it is reported as one. The deliverable is a table of verdicts rather than
a ranking, and the negative results are published beside the positive ones. The register is deliberate:
the exercise is performed in role, on public data, and the declaration above travels on every artifact
the repository publishes.

## How the comparison is built

Seven layers, each with the reason it is shaped the way it is. The learning guide
(`reporting/learning-guide.md`) walks through all of them with their citations and the alternatives
they displaced; this section is the map.

**The data contract.** Three sources feed one frozen snapshot: the European Central Bank's Data Portal
for the overnight rate chain, spliced from EONIA to the euro short-term rate at the transition with the
overlap measured rather than assumed; the Fama/French archives for the developed-market factor legs;
and a public price feed for the fund bars, read with the feed's adjusted close as the total-return
proxy. The snapshot carries a manifest of instruments, file hashes, row counts and factor vintages, and
the loader verifies all of it on every read and fails closed when it does not match. The rule doing the
most work is the as-of rule: a month's bar becomes readable on the first day of the following month,
and every join is gated by it. Without that gate an estimation window built at the end of March reads
March's own return, which is look-ahead bias of one month, and a performance table would not reveal it.

**The factor layer.** The eleven sleeves are funds rather than firms, so the security-level style and
industry attributes a commercial risk model would use are not available. What is available is a set of
published index legs, called the named spine, and the sleeves' own construction, assembled into a
constructed block of level, slope, credit and high-yield lines. The block's arithmetic is deliberately
trivial: its loadings on four sleeves are identities rather than estimates, which is what makes those
four checkable instead of merely fitted. Exposures are estimated per window by ordinary least squares
and refitted monthly, with the block orthogonalised against itself so that each later loading is the
clean reading. The statistical family is principal components, and the count is decided by a rule
declared before it was applied: a component is retained when its eigenvalue beats the 95th percentile
of a matched permutation null built from the panel's own marginals. Principal components extract
directions of common variation; they do not select factors and they do not identify which directions
are priced, so the count comes from that external criterion rather than from the extraction.

**The risk layer.** Three covariance estimators sit behind one interface: the sample covariance, linear
shrinkage toward a structured target at an intensity the estimator estimates, and the covariance
implied by the retained components. What that axis varies is the conditioning of the matrix handed to
the optimiser, and it is reported as such. There is no true covariance on this panel to rank the
estimators against, and a sample covariance is by construction the best fit to its own window, so an
accuracy ranking would rank whichever estimator it was computed from first.

**The construction layer.** Four mean inputs and the family list are crossed under one constraint set:
long-only, fully invested, capped at 35% per sleeve, with a 1% no-trade band per sleeve and a 5%
one-way turnover cap at each rebalance. The mean inputs are the sample mean, a shrunk mean, a
Black-Litterman posterior whose prior is implied by the mandate's own weights, and no mean at all. The
no-mean cell is the control: it isolates how much of each other cell's result the solver and the
constraints were producing on their own. Cost is charged on *traded* notional at 10 bp per side, so a
rebalance is billed for what it moves rather than for the book it holds, and a method that trades rarely
is not penalised for holding a large book.

**The evaluation layer.** Every cell is estimated on a rolling sixty months with a monthly refit and
trades only after its window ends. Comparing cells by their separate estimates would be useless at this
length: one strategy's annualised ratio carries an interval wide enough that nearly every method sits
inside every other's. Every comparison is therefore a paired test
on the difference of two monthly return series, which works because the cells share a universe, a set
of months and long-only books and are consequently highly correlated. A row is called a difference only
when the paired test clears the family-wise bar over the sixteen declared cells, the cell keeps its rank
in at least 80% of bootstrap resamples, and its sign holds under an expanding-window protocol. Failing
any rung publishes the row as no difference detected, with the smallest difference the test would have
caught printed beside it, so an absence is never read as an equivalence.

**Attribution and the risk budget.** The active return is decomposed twice, on the holdings in the
Brinson-Fachler form and through the fitted factor exposures, and the two views are never summed
because they read the same return in different bases. Allocation-only is a structural fact rather than
a simplification: the benchmark holds the same instrument in each sleeve, so the selection term does
not exist, and the currency dimension carries the interaction instead. The risk budget reads the
realised volatility contributions against the vector the mandate declared, using Euler contributions,
and value at risk is refused as a decomposition with the measurement that justifies the refusal, since
its conditional contributions do not sum to it.

**One run, one object.** `portfolio_workbench/study.py` assembles a snapshot's run and every reading
taken off it, so the comparison table, the workbook, the findings memo and the learning guide cannot
describe different runs. Nine call sites used to assemble it for themselves, and two of them disagreed
about the months a covariance was read over.

## What is being compared

Sixteen distinct cells, from twenty pre-registered runs, on one universe and one mandate. A cell is a
construction family, a covariance estimator, a mean input and a protocol, fixed before any of them
runs. The axes move one at a time: the family axis varies the objective at one covariance and one
default mean, the risk-model axis varies the estimator across the two most covariance-dependent
families, and the mean axis varies the input inside mean-variance. Two perturbation runs lift the
per-sleeve cap and change nothing else, so a result that came from the constraint set rather than from
the method is visible as such, and two repeats re-run the cells the mandate's question turns on under
the expanding protocol. The count is declared in the code before any of them runs, because a search
over sixteen methods whose size is reported after the results are seen is a search whose size was
chosen by the results.

Two rows are not methods. One is the policy benchmark, the mandate's own weights, which is what every
cell is measured against; the other is equal weight, which is the bar the literature uses and which
several of the constructed books reproduce on this constraint set.

## What it found

**No cell cleared every rung of the declared ladder, so the negative result is the finding.** The
sample's leader cleared the family-wise bar and then failed the rank-retention rung, keeping its rank
in only 67.5% of bootstrap resamples against the 80% floor declared in advance. Eight of the runs that
do clear the bar sit significantly behind the policy benchmark once cost is charged. Every cell that
carries a verdict of no difference detected, and every cell declared a negative result, is published
with that verdict and its resolution limit on the row.

**What separates the families is tracking error rather than return.** The family cells span a tracking
error of several percentage points a year, and that dispersion is large relative to what the test can
resolve, which makes the family the axis that matters most on this panel. The direction is not the one
a ranking would suggest: the cells that differ from the benchmark differ by taking more risk away from
it, not by earning more.

**The mean axis shows how much the input carries.** The three mean-carrying cells land close to one
another on the information ratio and all three move their weight path substantially month to month.
Shrinking the mean does not calm that path on this panel, which is the honest reading of a standard
remedy: it was applied, and the instability is still there.

**The risk budget is consumed by accident rather than by design.** The mandate's own book departs from
the volatility split it declares, and the constructed books depart further, because a risk-based family
concentrates volatility wherever the covariance puts it and no cell's objective mentioned the budget.
That is a property of the mandate as much as of the methods, and it is why the vector is declared in
the data layer rather than inferred afterwards.

## How to read the result

**A verdict of no difference detected is not a statement of equivalence.** The smallest difference the
test would have detected is printed on every row, and a reader who reads the absence as equality is
making the second statement while appearing to make the first. Where a difference is reported, it is
reported with the three conditions that produced it.

**The leader's advantage has the two signatures of estimation error.** An advantage that does not
survive resampling, and a weight path that moves when the estimation window shifts by a single month,
are what a method exploiting the sample looks like. Both are present here, and the weight-stability
diagnostic is reported for every cell so that a reader can see it rather than take it on trust.

**The residuals are named rather than absorbed.** The attribution reconciles after linking, the Euler
contributions sum to portfolio volatility, and the factor model's unexplained part is reported as its
own measured quantity. A decomposition that hides its residual cannot be checked.

**One panel bounds everything.** Eleven sleeves, one mandate and one currency: a difference this design
cannot resolve is not reported as an absence, and a difference it does resolve is a statement about
this universe rather than about construction in general. The memo's limitations section states the
bound in full, and the decision record states the conditions under which the recommendation would be
withdrawn.

## The conclusion

**Retain the policy benchmark and change nothing.** The chosen option is the one already held, so its
cost is nil, and the alternative was to pay a construction cost to move to a book whose advantage the
data does not support. The practical value of the exercise runs in two directions: it quantifies how
much of a methodological choice survives when cost, multiple testing and resampling are charged against
it, which is a more useful quantity than a ranking of in-sample fits; and it is a worked example of the
whole chain, from a licensed data source through a look-ahead-free panel, a documented risk model,
constrained construction, a paired out-of-sample evaluation and two decompositions, with every number
traceable to the module that computed it.

The recommendation is withdrawn or revisited on any of five conditions: a rerun that no longer
reproduces the metric table within the stated tolerance; a cell that clears the bar, keeps its rank in
at least 80% of resamples and holds its sign under the expanding protocol; a longer panel on which an
absent difference remains absent while the resolution limit falls below it; a revised cost multiple
that removes the leader's advantage; or a change to the constraint set that would make a family's
result a result about the constraints.

Nothing here is a claim about a live book, a client or a regulated activity. The exercise is a
simulation performed in role, and the numbers quoted above are the few that define the outcome: the
full set is generated from the run by the documents in the next section, and those are pinned to their
generators by the test suite, so a number cannot drift from the code that computed it.

## The universe and the window

Eleven UCITS ETF sleeves (developed and European and Nordic equity, short and long government,
investment-grade and high-yield credit, inflation-linked, gold, real estate, and a euro cash line),
priced in EUR except where the fund's own denomination is SEK. The joined panel runs 2010-09 to
2026-07, 191 months; the out-of-sample window is 131 months, 2015-09 to 2026-07. Two sleeve candidates
were dropped for reasons recorded beside the universe: one for near-collinearity with an existing
line, one because its feed is dividend-blind and therefore not a total-return series.

## Layout

```
portfolio_workbench/
  data/        the snapshot contract, the loader, the quality gate, coverage, SQL
  factors/     the factor spine, exposures, the component count, span tests
  risk/        the covariance estimators and the factor covariance
  construct/   the mean inputs, the greedy families, the constraint set
  evaluate/    the walk-forward engine, the metric block, the test statistics
  compare/     the cell registry, the grid, the comparison table
  attribute/   Brinson-Fachler allocation and the factor attribution
  budget/      the Euler risk decomposition
  study.py     one snapshot assembled into one run, with every reading off it
  facade.py    the consumer boundary: the contract plus five analytics entry points
reporting/     the output surface: workbook, notebooks, memo, guide, figures, source map, skills matrix
notebooks/     one executed notebook per entry point
tests/         the suite
```

Dependencies point inward, and nothing in the analytics imports `reporting`; the acceptance fixture
enforces that over the import graph.

## Running it

The package is imported from the repository root rather than installed, so there is no build step and
no environment to activate. Running the analytics needs numpy, pandas, scipy and scikit-learn. The
output surface adds openpyxl for the workbook, nbformat and nbclient to regenerate the notebooks,
matplotlib for the figures, and pytest to run the suite.

```bash
python3 -m pytest -q                             # the suite, offline and deterministic

python3 -m portfolio_workbench.study             # one snapshot, one run, one covariance
python3 -m portfolio_workbench.compare.grid      # every declared cell, then the comparison table
python3 -m portfolio_workbench.facade            # the contract and the five entry points
python3 -m portfolio_workbench.data.coverage     # the panel's shape and coverage
```

Every layer module is runnable the same way and prints its own report; each states its assumptions,
the resolution limit of the number it prints, and what the number does not establish. Only the grid
writes anything, and it writes outside version control.

## What it publishes

```bash
python3 -m reporting.workbook        # the Excel export, into .data/runs/<snapshot>/
python3 -m reporting.notebooks       # writes notebooks/ and executes every one
python3 -m reporting.memo            # the findings memo and the decision record
python3 -m reporting.guide           # the learning guide, written from the same run
python3 -m reporting.figures         # the figures the guide and the notebooks reference
python3 -m reporting.skills_matrix   # the coverage matrix and its check
```

The findings memo (`reporting/findings-memo.md`) answers the seven research questions in order, and the
decision record (`reporting/decision-record.md`) is the one page a committee would sign. Both are
generated from the code that owns the numbers they quote, so neither can drift from the run it
describes. The learning guide (`reporting/learning-guide.md`) is generated from the same run and reads
the exercise end to end: where the data comes from, what each layer does and why it is shaped that way,
what the results say, and what they do not.

## Data

**No market data ships with this repository, and none can be added to it.** The frozen snapshot, the run
manifests and the workbook export are all excluded by path and by extension. A reader who wants to
reproduce the run must fetch the sources and assemble a snapshot that satisfies the table contract: one
row per instrument-month carrying `period_month` and `available_from`, plus a manifest whose hashes and
row counts are verified on every load.

The three sources are the ECB Data Portal for the overnight rate chain, spliced from EONIA to the
euro short-term rate at the 2019-10-01 transition with the spread measured over the overlap rather
than assumed; the Fama/French factor archives for the developed-market factors; and a public price
feed for the fund bars. The fetch script is deliberately not part of the package, because it is a
one-off retrieval rather than a build step.

The data conditions that bind the published output follow from those sources. ECB statistics are free
to reuse on condition that the source is quoted and the series is not modified, and any
transformation, whether a growth rate or an adjustment, is stated explicitly. The Fama/French
archives carry a copyright line and no published licence; they are used as research inputs and no
factor series is redistributed here. The price feed is the binding constraint: its terms do not
permit redistribution, which is why the snapshot stays local and why the derived statistics in the
memo are published while no price, level or instrument series is. The figures the guide draws hold to
the same line. Each one plots a dispersion, a share, a count, a weight vector or a distribution
computed from the run, the module refuses to write a figure that would carry a per-instrument level,
and they are tracked in the repository rather than excluded like the workbook, because what they carry
is derived statistics rather than any part of the panel.

Point the loader at a snapshot with `WORKBENCH_SNAPSHOT` or by passing a directory to any entry point.

## Tests

`python3 -m pytest -q` runs offline against a synthetic, shape-matched snapshot for the identity and
boundary checks, and against the frozen snapshot for the end-to-end acceptance run. The suite asserts
what the modules claim: hand-calculated identities and closed forms, the walk-forward boundary, the
as-of rule, every stop and warning fired on a planted input, the constraint rules, and the direction
of the import graph. Two checks take most of the runtime, because each runs the whole stack once, and
the acceptance fixture also reads the three generated documents against their generators, the guide's
glossary against its own body, and the figures off disk for their presence and their licence posture.

## Vocabulary

Three words are deliberately overloaded, and the code says which it means:

- **manifest**: the snapshot's own file, which the loader verifies, or the per-run record under
  `.data/runs/`.
- **window**: the declared panel span, the sixty-month estimation window, or the traded out-of-sample
  months.
- **run**: one cell of the grid, or the act of running all of them.

## Licence

MIT. See `LICENSE`. The licence covers the code; it does not extend to any market data, which is not
distributed here, and it makes no claim about the terms of the data sources themselves.
