# portfolio-workbench

A methodology comparison for European multi-asset portfolio construction, run out of sample on one
frozen panel. The question is not which method is best; it is which methodological choice moves the
outcome, on which metric, and at what cost.

*A learning exercise performed in role: a simulated mandate with no client and no institution.
Nothing in this repository is investment advice, a recommendation to any person, or a client
communication.*

## What is being compared

Sixteen distinct cells, from twenty pre-registered runs, on one universe and one mandate. A cell is a
construction family, a covariance estimator, a mean input and a protocol, fixed before any of them
runs. The families cover the risk-based objectives, the mean-variance family across four mean inputs,
and the no-mean and perturbation cases; the axes are read separately so that the family choice, the
risk-model choice and the mean choice can be told apart.

Each cell is estimated on a rolling sixty months with a monthly refit, and trades only inside the
out-of-sample window. The constraint set is long-only, fully invested, capped at 35% per sleeve, with
a 5% one-way turnover cap and a 1% no-trade band. Cost is charged on *traded* notional at 10 bp per
side, so a rebalance is billed for what it moves rather than for the book it holds.

The result is a comparison table, not a ranking. A row is called a difference only when the paired
test, the family-wise bar, the bootstrap rank retention and the sign under a second protocol all
agree; every other row is published as a negative result with the resolution limit printed beside it.
No cell clears every rung of the ladder, so the decision record retains the policy benchmark and
publishes the negative result as the finding — eight of the runs that clear the bar sit significantly
behind that benchmark once cost is charged, and the leader's advantage survives only 67.5% of
bootstrap resamples against an 80% floor.

## The universe and the window

Eleven UCITS ETF sleeves — developed and European and Nordic equity, long and short government,
investment-grade and high-yield credit, inflation-linked, gold, real estate, and a euro cash line —
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
reporting/     the output surface: workbook, notebooks, memo, source map, skills matrix
notebooks/     one executed notebook per entry point
tests/         the suite
```

Dependencies point inward, and nothing in the analytics imports `reporting`; the acceptance fixture
enforces that over the import graph.

## Running it

The package is imported from the repository root rather than installed, so there is no build step and
no environment to activate. Only numpy, pandas, scipy and scikit-learn are needed to run it, plus
nbformat and nbclient to regenerate the notebooks and pytest to run the suite.

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
python3 -m reporting.skills_matrix   # the coverage matrix and its check
```

The findings memo (`reporting/findings-memo.md`) answers the seven research questions in order. It
and the decision record (`reporting/decision-record.md`) are generated from the code that owns the
numbers they quote, so a document cannot drift from the run it describes.

## Data

**No market data ships with this repository, and none can be added to it.** The frozen snapshot, the
run manifests and the workbook export are all excluded by path and by extension. A reader who wants to
reproduce the run must fetch the sources themselves and assemble a snapshot that satisfies the table
contract: one row per instrument-month carrying `period_month` and `available_from`, plus a manifest
whose hashes and row counts are verified on every load.

The three sources are the ECB Data Portal for the overnight rate chain, spliced from EONIA to the euro
short-term rate at the 2019-10-01 transition with the spread measured over the overlap rather than
assumed; the Fama/French factor archives for the developed-market factors; and a public price feed for
the fund bars. The fetch script is deliberately not part of the package, because it is a one-off
retrieval rather than a build step.

The data conditions that bind the published output follow from those sources. ECB statistics are free
to reuse on condition that the source is quoted and the series is not modified, and any transformation
— a growth rate, an adjustment — is stated explicitly. The Fama/French archives carry a copyright line
and no published licence; they are used as research inputs and no factor series is redistributed here.
The price feed is the binding constraint: its terms do not permit redistribution, which is why the
snapshot stays local and why the derived statistics in the memo are published while no price, level or
series is.

Point the loader at a snapshot with `WORKBENCH_SNAPSHOT` or by passing a directory to any entry point.

## Tests

`python3 -m pytest -q` runs offline against a synthetic, shape-matched snapshot for the identity and
boundary checks, and against the frozen snapshot for the end-to-end acceptance run. The suite asserts
what the modules claim: hand-calculated identities and closed forms, the walk-forward boundary, the
as-of rule, every stop and warning fired on a planted input, the constraint rules, and the direction
of the import graph. Two checks take most of the runtime, because each runs the whole stack once.

## Vocabulary

Three words are deliberately overloaded, and the code says which it means:

- **manifest** — the snapshot's own file, which the loader verifies, or the per-run record under
  `.data/runs/`.
- **window** — the declared panel span, the sixty-month estimation window, or the traded out-of-sample
  months.
- **run** — one cell of the grid, or the act of running all of them.

## Licence

MIT. See `LICENSE`. The licence covers the code; it does not extend to any market data, which is not
distributed here, and it makes no claim about the terms of the data sources themselves.
