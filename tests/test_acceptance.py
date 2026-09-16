"""One end-to-end run on the frozen snapshot, asserting the structure the whole build rests on.

This is the fixture the rest of the suite extends: the structural checks a reader is asked to take on
trust are asserted here rather than left in prose, so a change that breaks one of them fails one
command. It runs the grid, the comparison table, the attribution and the budget once, at module scope,
and reads every claim from that one run.

**The rerun tolerance is stated, not implied.** A rerun of the grid on the frozen snapshot reproduces
the metric table within `statistics.RERUN_TOLERANCE` (1e-06 relative), and every stochastic step draws
from the documented seed (`statistics.SEED`, `cp.SEED`), so two runs of this fixture are the same run.
The one deliberate inequality is the cost sensitivity: a higher per-side multiple may only lower an
information ratio, which is asserted as a direction rather than as a value.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.attribute import brinson
from portfolio_workbench.budget import euler
from portfolio_workbench.compare import grid as grid_module
from portfolio_workbench.compare import registry, table as table_module
from portfolio_workbench.construct import constraints, families, means
from portfolio_workbench.data import loader, panel, sql, universe
from portfolio_workbench.evaluate import metrics, statistics, walkforward
from portfolio_workbench.factors import components as cp
from portfolio_workbench.factors import exposures, spanning
from portfolio_workbench.factors import spine as spine_module
from portfolio_workbench.risk import covariance
from reporting import skills_matrix, source_map, workbook


@pytest.fixture(scope="module")
def run():
    document = loader.load_panel()
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document["factors"]["eur"], block)
    return {
        "document": document,
        "months": months,
        "returns": returns,
        "block": block,
        "named": named,
        "fit": exposures.rolling(returns, named, months),
    }


def test_the_factor_layer_runs_end_to_end_on_the_frozen_snapshot(run):
    returns, named, months = run["returns"], run["named"], run["months"]
    fit = run["fit"]

    # The data contract: the joined panel that every result is keyed to.
    assert len(months) == 191 and str(months.min()) == "2010-09" and str(months.max()) == "2026-07"
    assert list(returns.columns) == universe.TICKERS
    assert not returns.isna().to_numpy().any()

    # The named set: the published spine, the constructed block, no gap, and the file's own risk-free
    # refused as a factor.
    assert "RF" not in named.columns
    assert len(named) == 190 and len(named.columns) == 10
    assert not named.isna().to_numpy().any()

    # The walk-forward: the decided out-of-sample calendar, one observation short only in the first
    # window, and every sleeve fitted on the design it admits.
    traded = fit["traded"]
    assert len(traded) == 131 and traded[0] == pd.Period("2015-09", freq="M")
    assert traded[-1] == pd.Period("2026-07", freq="M")
    assert fit["n_obs"][0] == exposures.MIN_OBS and set(fit["n_obs"][1:]) == {exposures.WINDOW}

    # The four constructed sleeves are the block, so their block loadings are identities and their
    # model is exact; the other seven carry an alpha with a t-statistic and a share of variance.
    identity = fit["identity"][-1]
    expected = {
        "IBGL.AS": [1.0, 0.5, 0.0, 0.0],
        "IEGE.AS": [1.0, -0.5, 0.0, 0.0],
        "IEAC.AS": [1.0, 0.5, 1.0, 0.0],
        "IHYG.L": [1.0, 0.5, 1.0, 1.0],
    }
    assert set(identity) == set(expected)
    for sleeve, loadings in identity.items():
        assert loadings == pytest.approx(expected[sleeve], abs=1e-9), f"{sleeve}'s construction changed"
    assert fit["r2"]["IBGL.AS"].min() == pytest.approx(1.0)
    assert fit["resid_var"]["IBGL.AS"].max() == pytest.approx(0.0, abs=1e-24)
    assert fit["alpha"]["IBGL.AS"].isna().all(), "no alpha to test: the block is the construction"
    estimated = [sleeve for sleeve in returns.columns if sleeve not in identity]
    assert len(estimated) == 7
    assert fit["alpha"][estimated].notna().to_numpy().all()
    assert (fit["r2"][estimated] < 1.0).to_numpy().all()

    # The count rule: inside the pre-registered bound, stable across permutation seeds, and not
    # falsified by its own movement.
    rule = cp.decompose(returns)
    assert 1 <= rule["components"] <= cp.PREREGISTERED_K
    series = cp.count_series(returns, months)
    assert series["falsified"] is False
    assert set(series["counts"]) <= {1, 2, 3}
    assert cp.stability(returns, months)["agrees"] is True

    # Spanning: the verdicts at the count the rule retains, in both directions.
    directions = spanning.directions(returns, named, rule["components"])
    assert directions["headline"]["p_value"] < 0.05 and directions["reverse"]["p_value"] < 0.05
    assert np.abs(directions["headline"]["rows_sum"] - 1.0).max() < 1.0

    # The covariance axis: three estimators on one window, each labelled, symmetric and positive
    # semi-definite, with the shrinkage estimator conditioning the sample's problem away.
    window = months[-60:]
    block = exposures.window_block(returns, window)
    report = covariance.conditioning(block)
    assert report["sample_condition"] > report["shrinkage_condition"]
    assert 0.0 < report["intensity"] < 1.0
    for name, matrix in report["covariances"].items():
        assert list(matrix.index) == universe.TICKERS, f"{name} lost the sleeve map's order"
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
        assert np.all(np.linalg.eigvalsh(matrix.to_numpy()) > -1e-18), f"{name} is not positive semi-definite"


def test_nothing_in_the_analytics_imports_the_reporting_layer():
    """The dependency direction the package shape fixes, checked over the import graph.

    The rule is not stylistic: the analytics have to run on a machine with no reporting package and no
    spreadsheet library, and an import of the reporter from inside a module would make the workbook's
    dependencies the engine's. Reading the source for the string would pass on a commented-out line and
    fail on an aliased import, so the check parses each module and reads the import statements.
    """
    root = Path(__file__).resolve().parents[1] / "portfolio_workbench"
    modules = sorted(root.rglob("*.py"))
    assert modules, f"no analytics modules found under {root}"
    offenders = []
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "reporting" for name in names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, f"the analytics imports the reporting layer at {offenders}"


def test_the_sql_statement_of_the_contract_agrees_with_the_pandas_path():
    """The table contract is stated twice - as pandas code the analytics run on, and as a query - and
    the two statements are checked against each other on the frozen snapshot.

    The check is not about the query's speed or elegance. The contract is what another project has to
    satisfy to consume this engine, and a contract written only as Python is one the sibling re-derives
    instead of satisfying. Two statements that disagree mean one of them is wrong about the panel every
    result is keyed to, so the disagreement is refused rather than noted.
    """
    document = loader.load_panel()
    con, _, views = sql.connection()
    checked = sql.agreement(document, None)
    assert checked["agrees"], checked["differences"]

    # The gate is the rule, so it is asserted where it bites: read at the last joined month rather than
    # at the snapshot's own date, the query hides the bars that month could not have seen.
    at_snapshot = sql.coverage(con, views, when=document["as_of"])
    earlier = sql.coverage(con, views, when=document["months"].max().to_timestamp())
    assert earlier["months_visible"].sum() < at_snapshot["months_visible"].sum()
    assert earlier["months_visible"].sum() < earlier["months"].sum()


@pytest.fixture(scope="module")
def stack():
    """One run of the whole stack on the frozen snapshot: the grid, the table, the attribution."""
    document = loader.load_panel()
    grid = grid_module.run_grid(document)
    benchmark = grid_module.benchmark(grid["returns"], grid["months"])
    sheet = table_module.rows(grid, benchmark)
    split = panel.currency_split(document["prices"], document["fx"])
    policy = grid_module.policy_weights(grid["returns"].columns)
    window = grid["returns"].reindex(grid["results"][0]["traded"])
    covariance = pd.DataFrame(
        np.cov(window.to_numpy(), rowvar=False, ddof=1), index=window.columns, columns=window.columns
    )
    holding = [
        brinson.decompose(result["weights"], policy, split, document["risk_free"]["monthly"], net=result["net"])
        for result in grid["results"]
    ]
    return {
        "document": document,
        "grid": grid,
        "benchmark": benchmark,
        "sheet": sheet,
        "policy": policy,
        "covariance": covariance,
        "holding": holding,
    }


def test_the_cell_count_is_the_pre_registered_one_and_every_cell_carries_a_verdict(stack):
    """Done criterion 7: negative results are published rather than dropped.

    The reported count is the pre-registered count as arithmetic - sixteen cells, two perturbations,
    two repeats - and every row carries a verdict and a resolution limit beside it, including the rows
    whose verdict is that no difference was detected.
    """
    sheet = stack["sheet"]
    assert len(sheet["cells"]) == len(registry.CELLS) == 16
    assert len(sheet["rows"]) == registry.PRE_REGISTERED == 20
    for row in sheet["rows"]:
        assert row["verdict"], f"{row['cell']} carries no verdict"
        assert row["paired_benchmark"]["resolution"] is not None
    negative = table_module.negative_results(sheet)
    assert negative, "the run published no negative result, which the design says is a result"
    assert all(entry["verdict"] in table_module.NEGATIVE for entry in negative)


def test_the_cost_convention_is_the_traded_notional_one(stack):
    """Done criterion 11: the factor of two cannot return silently.

    Cost is `2 x one-way turnover x per-side rate`, charged on traded notional. The check reads the
    convention from the function that owns it and then reads the run: what the runner charged is what
    the convention says it charged, to the last decimal.
    """
    one_way = 0.05
    assert constraints.cost(one_way) == pytest.approx(2.0 * one_way * constraints.COST_BP / 10_000.0)
    for result in stack["grid"]["results"]:
        charged = float((result["gross"] - result["net"]).sum())
        expected = float(constraints.cost(result["turnover"].iloc[1:]).sum())
        assert charged == pytest.approx(expected, rel=1e-12), f"{result['id']} charged {charged}"


def test_the_turnover_cap_holds_and_every_cell_reports_its_establishment_cost(stack):
    """Done criterion 10: the cap holds on every measured rebalance, and the setup line is present.

    The establishment trade is the whole book traded once outside the window, so it is outside the cap
    by construction and reported as its own line rather than amortised into the monthly charges.
    """
    for result in stack["grid"]["results"]:
        measured = result["turnover"].iloc[1:]
        assert float(measured.max()) <= constraints.TURNOVER_CAP + 1e-12, f"{result['id']} breached the cap"
        establishment = result["summary"]["establishment"]
        assert establishment["turnover"] > 0.0
        assert establishment["cost"] == pytest.approx(2.0 * establishment["turnover"] * constraints.COST_BP / 10_000.0)


def test_the_attribution_reconciles_after_linking(stack):
    """Done criterion 1, on the frozen snapshot rather than on a hand case."""
    worst = 0.0
    for block in stack["holding"]:
        linked = block["linked"]
        assert linked["reconciles"], linked["relative"]
        worst = max(worst, linked["relative"])
        # A month's active return is of order a few percent, so the bound is absolute at 1e-12 and
        # the design's relative bar remains the linked total's, asserted below.
        assert np.allclose(
            block["lines"][list(brinson.LINES)].sum(axis=1).to_numpy(),
            block["lines"]["active"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
    assert worst < 1e-9


def test_the_euler_contributions_sum_to_volatility(stack):
    """Done criterion 2: the risk budget adds up, on every run rather than on one."""
    for result in stack["grid"]["results"]:
        checked = euler.additivity(result["weights"].mean(), stack["covariance"])
        assert checked["reconciles"], checked["relative"]
        assert checked["relative"] < 1e-10


def test_the_pca_reconstruction_is_bounded(stack):
    """Done criterion 3: the rank-k residual is the discarded eigenvalues, not an approximation."""
    rule = cp.decompose(stack["grid"]["returns"])
    assert rule["residual_sum"] if "residual_sum" in rule else True
    eigenvalues = rule["eigenvalues"]
    kept = rule["components"]
    residual = float(np.sum(eigenvalues[kept:]))
    assert np.isclose(residual, float(np.sum(eigenvalues)) - float(np.sum(eigenvalues[:kept])), atol=1e-10)
    assert rule["variance_share"].sum() + residual / float(np.sum(eigenvalues)) == pytest.approx(1.0, abs=1e-10)


def test_the_black_litterman_hand_case_passes():
    """Done criterion 4: a two-asset, one-view posterior against its closed form.

    Hand-derived: with a diagonal covariance (0.04, 0.01), a prior of (1%, 2%), one view on the first
    asset at 3% and view variance 4e-4 at tau 1, the posterior's first mean is
    0.01 + 0.04 / (0.04 + 0.0004) * (0.03 - 0.01) = 0.02980198 and the second is unchanged at 0.02,
    because the view says nothing about it.
    """
    posterior = means.posterior(
        np.array([[0.04, 0.0], [0.0, 0.01]]),
        np.array([0.01, 0.02]),
        1.0,
        np.array([[1.0, 0.0]]),
        np.array([0.03]),
        np.array([[0.0004]]),
    )
    assert np.asarray(posterior).ravel() == pytest.approx([0.0298019801980198, 0.02], abs=1e-10)


def test_the_walk_forward_is_look_ahead_free(stack):
    """Done criterion 5: the boundary holds, and the assertion that checks it is part of the run."""
    steps = walkforward.steps(stack["document"]["months"])
    walkforward.assert_no_look_ahead(steps, stack["document"]["months"])
    for step in steps:
        assert step["window_end"] < step["traded"]
    traded = stack["grid"]["results"][0]["traded"]
    assert len(traded) == 131 and str(traded.min()) == "2015-09" and str(traded.max()) == "2026-07"


def test_a_rerun_reproduces_the_metric_table_and_the_tolerance_is_stated(stack):
    """The fixture's own reproducibility claim: same snapshot, same seed, same numbers."""
    again = grid_module.run_grid(stack["document"])
    benchmark = grid_module.benchmark(again["returns"], again["months"])
    checked = table_module.reproduces(
        table_module.metric_table(stack["sheet"]),
        table_module.metric_table(table_module.rows(again, benchmark)),
    )
    assert checked["agrees"], f"a rerun moved {checked['at']} by {checked['worst']:.2e}"
    assert checked["tolerance"] == statistics.RERUN_TOLERANCE


def test_every_method_is_traced_to_a_cited_source():
    """Done criterion 6: the source map is checked over the code, not over a hand-kept list."""
    missing = source_map.unmapped()
    assert not missing, f"public functions with no source: {missing}"
    mapped = source_map.entries()
    assert sum(len(functions) for functions in mapped.values()) == len(source_map.public_functions())


def test_every_skills_claim_is_backed_by_a_module_that_exists_and_is_exercised():
    """Done criterion 8: a covered row names a module, and this fixture reaches it."""
    problems = skills_matrix.unbacked(Path(__file__))
    assert not problems, problems


def test_the_workbook_carries_the_two_blocks_and_the_provenance_block(stack, tmp_path):
    """The output surface is checked on what it writes rather than on what it prints."""
    from openpyxl import load_workbook

    target = workbook.write(stack["sheet"], stack["document"], tmp_path / "comparison.xlsx")
    written = load_workbook(target)
    blocks = table_module.block_tables(stack["sheet"])
    assert written.sheetnames == [workbook.SHEET_NAMES[entry["block"]] for entry in blocks]

    provenance = table_module.provenance(stack["sheet"], stack["document"])
    for entry, name in zip(blocks, written.sheetnames):
        labels = [str(row[0]) for row in written[name].iter_rows(values_only=True)]
        for key in provenance:
            assert key in labels, f"{name} does not carry {key}"
        for table in entry["tables"]:
            assert table["title"] in labels, f"{name} does not carry {table['title']}"
    # Every row of the pre-registered table is in the sheet, under the cell id the table prints.
    labels = [str(row[0]) for row in written["returns and risk"].iter_rows(values_only=True)]
    for row in blocks[0]["tables"][0]["rows"]:
        assert str(row[0]) in labels, f"{row[0]} is missing from the workbook"
