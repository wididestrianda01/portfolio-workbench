"""The analysis: one snapshot, analysed once, with one covariance behind every reading of it.

**Why it exists.** The grid, the benchmark, the comparison table, the attribution and the risk budget
are five readings of one run, and each entry point that prints them used to assemble that run for
itself. Nine call sites made nine assemblies, and an assembly is where a convention lives: the sample
covariance the risk budget decomposes was read over the panel's 190 returns at one call site and over
the 131 traded months at another, so two published documents stated different numbers for the same
book, and nothing could catch it because no module owned the quantity they both claimed.

**What it owns.** Which months a reading is taken over, which covariance it is taken against, and the
order the run is assembled in. The covariance is the sample covariance over the traded months, taken
from the estimator module rather than re-derived here: the risk budget describes the books the run
produced, so the months those books were held are the months their covariance is read over. The
estimator is named in the report this module prints, because a budget read against a different
covariance from the one the cell was built on is a statement about two objects and not one.

**What it deliberately does not own.** The mandate. The window, the universe, the policy weights and
the constraint set are this build's decisions for one panel and one mandate and stay in the modules
that declare them; a consumer with a different mandate supplies its own document. Nor does it own the
modules' own vocabularies: the count series, the spanning directions and the headline decomposition
belong to the one caller that reports them, and pulling them in here would widen this interface
without giving a single further caller anything.

**Where it sits.** Above the layers and below the output surface: `reporting/` reads the analysis and
nothing here reads `reporting/`. The entry points import this module inside their own functions rather
than at module level, because this module imports the layers those entry points live in - the same
deferral the comparison table already uses to reach the loader.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .attribute import brinson
from .budget import euler
from .compare import grid as grid_module
from .compare import table as table_module
from .data import loader, panel
from .evaluate import metrics
from .risk import covariance as covariance_module


def traded_months(grid):
    """The out-of-sample window: the months the engine trades, read off the run rather than re-cut.

    Read from the first run rather than from a second call to the walk-forward engine, so the months
    every reading is taken over are the months the cells were actually measured on. A grid that
    produced no run at all has no window to report, and refuses rather than reading an empty frame as
    an answer.
    """
    if not grid["results"]:
        raise ValueError("the grid produced no run, so the analysis has no traded months to read over")
    return grid["results"][0]["traded"]


def risk_budgets(grid, window, covariance, policy, benchmark):
    """Every run's consumption of the declared risk budget, and the policy book's consumption.

    The policy book is included because it is the mandate's own weights and therefore the reading the
    others are compared against. It carries no forecast: a fixed-weight book has no active series
    against which an ex-ante estimate could have been tested.

    One record per book, built here rather than at each reporting call site, because two call sites
    each assembling their own record is how the covariance came to be read two different ways.
    """
    def record(identifier, book, active=None):
        weights = book.mean()
        ex_ante = float(np.mean([
            euler.ex_ante_tracking_error(book.loc[month] - policy, covariance) for month in book.index
        ]))
        return {
            "id": identifier,
            "contributions": euler.path_contributions(book, covariance),
            "budget": euler.budget(weights, covariance),
            "diversification": euler.diversification(weights, covariance),
            "additivity": euler.additivity(weights, covariance),
            "forecast": (
                euler.forecast_quality(ex_ante, euler.ex_post_tracking_error(active))
                if active is not None
                else None
            ),
            "refusal": euler.var_refusal(window, weights),
        }

    budgets = {
        result["id"]: record(result["id"], result["weights"], metrics.active(result["net"], benchmark))
        for result in grid["results"]
    }
    policy_path = pd.DataFrame([policy] * len(benchmark), index=benchmark.index)
    budgets["policy"] = record("policy", policy_path)
    return budgets


def analyse(document, grid=None):
    """One snapshot, analysed: the grid and everything read off it, under one covariance.

    A caller that has just run the grid hands it in rather than making the analysis run it a second
    time. The returns frame and the calendar come from that run rather than from a second derivation of
    the panel's own legs, so the frame the budget is read against is the frame the cells earned their
    returns on. The attribution is keyed by cell id, which is the name every other reading of a run is
    keyed by and the name a run's manifest is written under; each block carries its own id, so a
    consumer holding one block can still name it.
    """
    grid = grid_module.run_grid(document) if grid is None else grid
    returns, months = grid["returns"], grid["months"]
    benchmark = grid_module.benchmark(returns, months)
    sheet = table_module.rows(grid, benchmark)
    traded = traded_months(grid)
    window = returns.reindex(traded)
    covariance = covariance_module.sample(window)
    policy = grid_module.policy_weights(returns.columns)
    split = panel.currency_split(document["prices"], document["fx"])
    return SimpleNamespace(
        document=document,
        returns=returns,
        months=months,
        grid=grid,
        benchmark=benchmark,
        sheet=sheet,
        covariance=covariance,
        attribution={
            result["id"]: {
                **brinson.decompose(
                    result["weights"], policy, split, document["risk_free"]["monthly"], net=result["net"]
                ),
                "id": result["id"],
            }
            for result in grid["results"]
        },
        budgets=risk_budgets(grid, window, covariance, policy, benchmark),
    )


def main(root=None):
    """What was assembled, over which months, against which covariance.

    The analysis prints the shape of the run rather than a result: the numbers live in the modules
    behind it, each with its own report. What is printed here is the part every one of those reports
    shares and none of them states - the window, the estimator and the book set their readings are
    taken over - so a reader comparing two reports can see that the two were read against one object.

    The tag is `[table]` by the print protocol's nearest-tag rule: the output is an inventory, and the
    package has no tag for one of those.
    """
    analysis = analyse(loader.load_panel(root))
    traded = analysis.grid["results"][0]["traded"]
    print(f"[table] analysis: {len(analysis.months)} panel months, {len(traded)} traded "
          f"({traded.min()}..{traded.max()}), {len(analysis.grid['results'])} runs and "
          f"{len(analysis.grid['cuts'])} cut")
    print(f"[table] covariance: sample, over the traded months, {analysis.covariance.shape[0]} sleeves; "
          f"the estimator every risk budget below is read against")
    print(f"[table] readings: {len(analysis.sheet['rows'])} table rows over "
          f"{len(analysis.sheet['cells'])} distinct cells, {len(analysis.attribution)} attributed books, "
          f"{len(analysis.budgets)} budgeted books")
    for line in analysis.document["warnings"]:
        print(f"[table] {line}")
    return analysis


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
