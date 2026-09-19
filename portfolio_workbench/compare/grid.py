"""The grid runner: every declared run over the out-of-sample months, with its protocol recorded.

One pass per run. In each month the estimate uses the trailing window the availability rule allows,
the constructor turns it into a target, the trading rules turn the target into a book, and the book
earns the following month's return. Every step is kept - the weights, the target, the trade, the
cost and the binding - because a mean cannot show whether a cap bound once or every month, and the
estimation-error diagnostics are properties of the path rather than of its average.

Five things in here are decisions rather than mechanics.

**The first book is the cell's own first target, and the trade that funds it is reported separately.**
Starting every method from the policy portfolio would put a transition trade inside the measured
window, and on this panel that trade is more than eleven times the turnover cap: the cap would be
broken at inception, by construction, before the method had done anything. Each cell is therefore
funded from cash at its own first target, the measured window holds no transition trade at all, and
the establishment cost is reported as its own line - outside the window and outside the cap, visible
rather than averaged away.

**The diagnostics are read off the target path, not off the traded path.** Weight stability and
concentration are the estimation-error proxies, so they are measured before the no-trade band and
the turnover cap smooth them: a cell whose raw targets swing is exploiting estimation error whatever
its realised weights look like. The traded path is reported beside them, and the gap between the two
is what the trading rules cost.

**The benchmark is the policy portfolio, rebalanced to its fixed weights every month and costless.**
It is a policy benchmark rather than a market index, so tracking error measured against it is
policy-relative, and it is charged nothing: a strategic allocation is not billed for its own
rebalancing. Charging it would make every active cost look smaller by comparison, which is a
flattery nothing here has earned.

**A run that cannot fill its row is reported as a cut, not raised.** A method whose constraint set is
infeasible on some window, or whose solve fails, is a result about this panel; the grid records the
reason and continues, so one cell cannot remove the table. The reason travels with the run, beside
the cells that did fill their row, and never as a blank row.

**Every run writes a manifest keyed to the snapshot.** The manifest is written outside version
control because it is derived from the licence-encumbered snapshot, and it is written at all because
a weight path whose protocol, threshold and cost convention are not recorded beside it can be read
but not checked.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..construct import constraints, families, means
from ..data import loader, universe
from ..evaluate import walkforward
from ..factors import exposures
from ..risk import covariance as covariance_module
from . import registry

HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = HERE.parent.parent / ".data" / "runs"


def _sample(block):
    """The window's sample covariance, with no diagnostic of its own to report."""
    return covariance_module.sample(block), {}


def _shrinkage(block):
    """Linear shrinkage, whose intensity is a statement about the window and is reported."""
    matrix, intensity = covariance_module.shrinkage(block)
    return matrix, {"intensity": float(intensity)}


def _factor(block):
    """The PCA factor covariance, which reports the count the rule retained for this window."""
    return covariance_module.factor_model(block)


# The estimator names the registry declares, mapped to the functions that produce them. Every one
# returns the same pair - a labelled matrix and a report - so a cell's diagnostics do not depend on
# which estimator it happened to declare.
ESTIMATORS = {"sample": _sample, "shrinkage": _shrinkage, "factor": _factor}


def window_covariance(name, block, cache=None, key=None):
    """One window's covariance from the declared estimator, cached by window where a cache is given.

    A grid multiplies the same window across many runs, and the factor estimator's count rule alone
    makes three hundred permutation draws per window; without the cache the runner would spend its
    time rebuilding an object it already holds.
    """
    if name not in ESTIMATORS:
        raise ValueError(f"no covariance estimator is declared under {name!r}")
    if cache is not None and key is not None and key in cache:
        return cache[key]
    value = ESTIMATORS[name](block)
    if cache is not None and key is not None:
        cache[key] = value
    return value


def policy_weights(columns):
    """The policy weights in the sleeve frame's own order, which every return series is indexed by."""
    return pd.Series([universe.POLICY_WEIGHTS[name] for name in columns], index=columns, name="policy")


def benchmark(returns, months):
    """The policy portfolio at its fixed weights through the out-of-sample months, costless.

    The traded months come from the engine rather than from a second cut of the calendar, so the
    benchmark is measured on exactly the months the cells trade; a benchmark on a different month set
    would make every active series a comparison of two calendars.
    """
    weights = policy_weights(returns.columns)
    traded = [step["traded"] for step in walkforward.steps(months)]
    return pd.Series([float(weights @ returns.loc[month]) for month in traded], index=pd.PeriodIndex(traded, freq="M"), name="policy")


def run_cell(spec, returns, months, cache=None):
    """One declared run: its weight path, its return series, its diagnostics and its establishment line."""
    family = families.FAMILIES[spec["family"]]
    mean_input = means.MEANS[spec["mean"]] if spec["mean"] is not None else None
    records = list(walkforward.steps(months, spec["protocol"]))
    path, estimator_intensities, counts = [], [], []
    mean_reports, sensitivity = [], {}

    for position, step in enumerate(records):
        block, taken = walkforward.block(returns, step)
        traded = step["traded"]
        key = (spec["estimator"], spec["protocol"], str(step["window_start"]), str(step["window_end"]))
        matrix, covariance_report = window_covariance(spec["estimator"], block, cache=cache, key=key)
        # The record is completed by the module that owns it, in place, so the boundary check below
        # reads the same records the run was driven by.
        records[position] = step = walkforward.observed(step, taken, covariance_report.get("components"))
        if covariance_report.get("intensity") is not None:
            estimator_intensities.append(covariance_report["intensity"])
        if covariance_report.get("components") is not None:
            counts.append(covariance_report["components"])
        vector, mean_report = (None, None) if mean_input is None else mean_input(block, matrix, spec["base_setting"])
        if mean_report is not None:
            mean_reports.append(mean_report)
        target = family(matrix, vector, block, cap=spec["cap"], **spec["family_options"])
        for setting in spec["settings"] or ():
            if setting == spec["base_setting"]:
                continue
            alternative = family(
                matrix,
                mean_input(block, matrix, setting)[0],
                block,
                cap=spec["cap"],
                **spec["family_options"],
            )
            # The sensitivity is reported in weight space, because the question is how much of the
            # answer is the ratio rather than how much of the mean is: a ratio that moves the weights
            # a fraction of a percent has not decided anything.
            sensitivity.setdefault(str(setting), []).append(float((alternative - target).abs().max()))

        if position == 0:
            book = target.copy()
            traded_turnover, binding, step_cost, movement = 0.0, False, 0.0, None
            establishment = constraints.establishment(target)
        else:
            rebalance = constraints.rebalance(target, path[-1]["book"], limit=spec["cap"])
            book = rebalance["weights"]
            traded_turnover, binding = rebalance["turnover"], rebalance["binding"]
            step_cost = constraints.cost(traded_turnover)
            movement = float((target - path[-1]["target"]).abs().sum() / 2.0)

        realised = float(book @ returns.loc[traded])
        # The book earns its market return and the trade is paid for out of it, so the cost is
        # charged to the net series and the gross series is the book's own return. Adding it to the
        # gross instead credits every cell with its own turnover, and the cost-adjusted series the
        # acceptance bar reads then pays nothing at all - on this panel a cell's annual cost is the
        # same order as the differences the comparison exists to detect.
        #
        # One step's own numbers in one record: a set of lists appended in step order cannot be told
        # apart once one of them is appended twice, and every path below is read against the others by
        # position. The funded step carries no movement, which is what separates the cells' rebalances
        # from their one establishment trade.
        path.append(
            {
                "traded": traded,
                "target": target,
                "book": book,
                "turnover": traded_turnover,
                "cap_binding": int((book > spec["cap"] - constraints.CAP_TOLERANCE).sum()),
                "turnover_binding": binding,
                "movement": movement,
                "gross": realised,
                "net": realised - step_cost,
            }
        )

    index = pd.PeriodIndex([step["traded"] for step in path], freq="M")
    weights = pd.DataFrame([step["book"] for step in path], index=index)
    target_path = pd.DataFrame([step["target"] for step in path], index=index)
    gross_series = pd.Series([step["gross"] for step in path], index=index, name=spec["id"])
    net_series = pd.Series([step["net"] for step in path], index=index, name=spec["id"])
    turnover_series = pd.Series([step["turnover"] for step in path], index=index)
    movements = [step["movement"] for step in path if step["movement"] is not None]
    cap_binding = [step["cap_binding"] for step in path]
    turnover_binding = [step["turnover_binding"] for step in path]
    intensity_values = [report["intensity"] for report in mean_reports if "intensity" in report]
    mean_intensity_path = (
        pd.Series(intensity_values, index=index[-len(intensity_values):], name="mean_intensity")
        if intensity_values
        else None
    )
    # The run's boundary, checked on the record once the path is built: every window's last month
    # precedes the month it trades, and the count beside each step is the one the step's own window
    # decided. The check is here rather than inside the loop so that a run is validated as a whole -
    # a boundary broken at one step is a statement about the run, not about that step.
    walk = walkforward.assert_no_look_ahead(records, months)
    summary = {
        "steps": len(index),
        "measured_rebalances": len(movements),
        "first_traded": str(index[0]),
        "last_traded": str(index[-1]),
        "turnover_mean": float(turnover_series.mean()),
        "turnover_annualised": float(turnover_series.mean() * 12),
        "cost_annualised": float(constraints.cost(turnover_series.mean()) * 12),
        "cap_binding_mean": float(np.mean(cap_binding)),
        "cap_binding_steps": int(sum(value > 0 for value in cap_binding)),
        "turnover_cap_binding_steps": int(sum(turnover_binding)),
        # The estimation-error proxies are read off the target path, and the traded path is reported
        # beside each of them: the band and the turnover cap move a book away from its target, so a
        # concentration measured on the books answers a question about the trading rules rather than
        # about the estimator under test, and the gap between the two is what the rules cost.
        "concentration": float(np.mean(1.0 / (target_path ** 2).sum(axis=1))),
        "concentration_traded": float(np.mean(1.0 / (weights ** 2).sum(axis=1))),
        "max_single_weight": float(target_path.to_numpy().max()),
        "max_single_weight_traded": float(weights.to_numpy().max()),
        "target_movement": float(np.mean(movements)) if movements else 0.0,
        "establishment": establishment,
        "gross_cumulative": float((1.0 + gross_series).prod() - 1.0),
        "net_cumulative": float((1.0 + net_series).prod() - 1.0),
        "volatility_annualised": float(net_series.std(ddof=1) * np.sqrt(12)),
        "estimator_intensity": float(np.mean(estimator_intensities)) if estimator_intensities else None,
        "components": sorted(set(counts)) if counts else None,
        "mean_intensity": float(np.mean(intensity_values)) if intensity_values else None,
        "sensitivity": {setting: float(np.mean(values)) for setting, values in sensitivity.items()},
    }
    return {
        "id": spec["id"],
        "spec": spec,
        "traded": index,
        "weights": weights,
        "targets": target_path,
        "turnover": turnover_series,
        "gross": gross_series,
        "net": net_series,
        "mean_intensity": mean_intensity_path,
        "cap_binding": pd.Series(cap_binding, index=index),
        "turnover_binding": pd.Series(turnover_binding, index=index),
        "walk_report": walk,
        "summary": summary,
    }


def run_grid(document, specs=None, cache=None):
    """Every declared run in registry order, with a run that cannot fill its row recorded as a cut."""
    returns = document.returns
    months = document.months
    cache = {} if cache is None else cache
    results, cuts = [], []
    for spec in registry.RUNS if specs is None else specs:
        try:
            results.append(run_cell(spec, returns, months, cache=cache))
        except ValueError as failure:
            cuts.append({"id": spec["id"], "spec": spec, "reason": str(failure)})
    return {"results": results, "cuts": cuts, "returns": returns, "months": months}


def manifest(result, document):
    """The per-run record: what was run, on which protocol, threshold and convention.

    A weight path without this beside it cannot be reproduced. The same cell, estimator and mean
    input produce different numbers under a different cost multiple, a different band or a different
    component rule, and nothing in the path says which was used.
    """
    spec, summary = result["spec"], result["summary"]
    return {
        "cell": spec["id"],
        "stage": spec["stage"],
        "snapshot": document.snapshot_id,
        "as_of": str(document.as_of),
        "protocol": {
            "kind": spec["protocol"],
            "window_months": exposures.WINDOW,
            "refit": "monthly",
            "rebalance": "monthly",
            "band": constraints.BAND,
            "first_traded": summary["first_traded"],
            "last_traded": summary["last_traded"],
            "steps": summary["steps"],
            "measured_rebalances": summary["measured_rebalances"],
            "funding": "the cell's own first target, from cash, outside the measured window",
        },
        "axes": {
            "family": spec["family"],
            "estimator": spec["estimator"],
            "mean_input": spec["mean"],
            "mean_setting": spec["base_setting"] if spec["mean"] == "black_litterman" else None,
            "family_options": spec["family_options"],
            "sensitivity_settings": list(spec["settings"]) if spec["settings"] else None,
        },
        "thresholds": {
            "per_sleeve_cap": spec["cap"],
            "turnover_cap": constraints.TURNOVER_CAP,
            "band": constraints.BAND,
            "components": summary["components"],
            "tail_level": families.CVAR_LEVEL,
        },
        "conventions": {
            "sign": "the largest absolute loading of each component made positive, reapplied at every refit",
            "cost_multiple": "2 x one-way turnover x the per-side rate, on traded notional",
            "per_side_bp": constraints.COST_BP,
            "cost_sensitivity_bp": list(constraints.COST_SENSITIVITY_BP),
            "mean_variance_tradeoff": families.MEAN_VARIANCE_TRADEOFF,
            "black_litterman": black_litterman_convention(spec),
            "seed": components_seed(),
        },
        "walk_forward": {
            "windows": result["walk_report"]["windows"],
            "sleeve_months_read": result["walk_report"]["observations"],
            "components_per_window": result["walk_report"]["components"],
            "gate": result["walk_report"]["gate"],
            "checked": "no window reaches the month it trades, and one window carries one component count",
        },
        "summary": {key: value for key, value in summary.items() if key != "establishment"},
        "establishment": summary["establishment"],
    }


def black_litterman_convention(spec):
    """The mean input's own inputs, recorded for the cells that consume one.

    Black-Litterman reports a number that is a function of a prior, a view and the view's
    uncertainty; a run whose manifest does not state all three cannot be checked or reproduced, and a
    reader cannot tell from the weight path which of them moved.
    """
    if spec["mean"] != "black_litterman":
        return None
    return {
        "prior": "the returns implied by the policy weights, reverse-optimised at the mean-variance trade-off",
        "view": "one view per sleeve, at the window's sample mean",
        "view_uncertainty": "the sleeve's own variance over the observation count, scaled by OMEGA_SCALE",
        "tau": float(spec["base_setting"]) / exposures.WINDOW,
        "tau_scale": float(spec["base_setting"]),
        "omega_scale": means.OMEGA_SCALE,
        "sensitivity_tau_scale": list(spec["settings"] or ()),
    }


def components_seed():
    """The one documented seed, read from the module that owns every draw in the package."""
    from ..factors import components

    return components.SEED


def write_manifests(document, results, root=None):
    """One JSON record per run, under a directory named for the snapshot the runs are keyed to.

    Written outside version control rather than beside the code: the records are derived from the
    frozen snapshot, whose licence posture forbids shipping anything derived from it. A directory
    keyed by snapshot id is also what makes a rerun against a different snapshot visible rather than
    silently comparable.
    """
    directory = Path(root) if root is not None else DEFAULT_RUN_ROOT / document.snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for result in results:
        path = directory / f"{result['id']}.json"
        path.write_text(json.dumps(manifest(result, document), indent=2, default=str) + "\n")
        written.append(path)
    return written


def report(grid, document):
    """The grid's own summary: one line per run, the diagnostics beneath it, then the table."""
    print(f"[construct] snapshot {document.snapshot_id}; {len(registry.CELLS)} cells, "
          f"{len(registry.PRIMARY)} runs on the primary protocol, {registry.PRE_REGISTERED} pre-registered")
    print(f"[construct] constraint set: long-only, fully invested, cap {constraints.CAP:.0%}, band "
          f"{constraints.BAND:.0%} per sleeve, one-way turnover cap {constraints.TURNOVER_CAP:.0%} per measured "
          f"rebalance, {constraints.COST_BP:.0f} bp per side charged on traded notional")
    for result in grid["results"]:
        summary = result["summary"]
        print(f"[construct] {result['id']:<34s} {summary['steps']} steps "
              f"{summary['first_traded']}..{summary['last_traded']}  turnover "
              f"{summary['turnover_annualised']:.2%}/yr  cap binding on {summary['cap_binding_steps']} of "
              f"{summary['steps']} steps  turnover cap bound {summary['turnover_cap_binding_steps']} of "
              f"{summary['measured_rebalances']} rebalances  establishment "
              f"{summary['establishment']['cost'] * 1e4:.2f} bp")
        details = []
        details.append(
            f"effective sleeve count {summary['concentration']:.2f} on the target path, "
            f"{summary['concentration_traded']:.2f} traded, largest weight "
            f"{summary['max_single_weight']:.3f} against {summary['max_single_weight_traded']:.3f}"
        )
        if summary["components"] is not None:
            details.append(f"retained components {summary['components']}")
        if summary["estimator_intensity"] is not None:
            details.append(f"estimator intensity {summary['estimator_intensity']:.4f}")
        if summary["mean_intensity"] is not None:
            per_window = result["mean_intensity"]
            spread = (
                f"per window {per_window.min():.4f}..{per_window.max():.4f}" if per_window is not None else ""
            )
            details.append(f"mean shrinkage intensity {summary['mean_intensity']:.4f} {spread} beside the unshrunk mean")
        if summary["sensitivity"]:
            details.append("ratio sensitivity " + ", ".join(f"{key}: {value:.4f}" for key, value in summary["sensitivity"].items()))
        if details:
            print(f"[construct]     {result['id']}: " + "; ".join(details))
    for cut in grid["cuts"]:
        print(f"[construct] {cut['id']:<34s} cut after {len(grid.get('results', []))} runs: {cut['reason']}")
    # The engine's own verdict, printed once for the whole grid rather than per run: the boundary is a
    # property of the protocol every cell runs under, and a per-run line would repeat one sentence
    # twenty times while saying nothing about the cell.
    walks = [result["walk_report"] for result in grid["results"]]
    print(
        f"[construct] the walk-forward boundary held on all {len(walks)} runs: every window's last month "
        f"precedes the month it trades ({walks[0]['first_traded']}..{walks[0]['last_traded']}), "
        f"{walks[0]['gate']}, and each window carries one component count"
    )
    # The cap is read beside the cell it binds rather than found further down a table: the question it
    # answers is what the bound was doing, and that is a comparison of two rows - the two perturbation
    # pairs, and the two ERC cells the design asks to be read against each other.
    for identifier, uncapped_id in (*registry.PERTURBATION_PAIRS.items(), *registry.BOUNDS_PAIRS.items()):
        base = next((result for result in grid["results"] if result["id"] == identifier), None)
        lifted = next((result for result in grid["results"] if result["id"] == uncapped_id), None)
        if base is None or lifted is None:
            continue
        print(f"[construct] the cap's effect, {identifier} against its uncapped pair: turnover "
              f"{base['summary']['turnover_annualised']:.2%} -> {lifted['summary']['turnover_annualised']:.2%}, "
              f"effective sleeve count {base['summary']['concentration']:.2f} -> {lifted['summary']['concentration']:.2f}, "
              f"largest weight {base['summary']['max_single_weight']:.3f} -> {lifted['summary']['max_single_weight']:.3f}, "
              f"volatility {base['summary']['volatility_annualised']:.2%} -> {lifted['summary']['volatility_annualised']:.2%}")
    print(f"[table] {'cell':<34s} {'turnover/yr':>11s} {'cost/yr':>8s} {'capbind':>8s} {'turnbind':>9s} "
          f"{'conc':>6s} {'move':>7s} {'grosscum':>9s} {'vol/yr':>7s}")
    for result in grid["results"]:
        summary = result["summary"]
        print(f"[table] {result['id']:<34s} {summary['turnover_annualised']:>9.2%} {summary['cost_annualised']:>8.2%} "
              f"{summary['cap_binding_mean']:>8.2f} {summary['turnover_cap_binding_steps']:>9d} "
              f"{summary['concentration']:>6.2f} {summary['target_movement']:>6.2%} "
              f"{summary['gross_cumulative']:>9.2%} {summary['volatility_annualised']:>7.2%}")
    for cut in grid["cuts"]:
        print(f"[table] {cut['id']:<34s} cut: {cut['reason']}")
    return grid


def main(root=None, out=None, specs=None):
    """Load the snapshot, run the declared grid, print what came back, and write the manifests."""
    registry.validate()
    document = loader.load_panel(root)
    grid = run_grid(document, specs=specs)
    policy = benchmark(grid["returns"], grid["months"])
    report(grid, document)
    print(f"[construct] benchmark (policy weights, monthly, costless): {len(policy)} months, cumulative "
          f"{float((1 + policy).prod() - 1):+.2%}")
    written = write_manifests(document, grid["results"], root=out)
    print(f"[construct] {len(written)} run manifests written under {written[0].parent if written else None}")
    from . import table

    table.main(document=document, grid=grid)
    return grid


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
