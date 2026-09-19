"""The Euler risk budget, the condition it adds up under, and the refusal that condition forces.

**Components sum to total risk when the risk measure is homogeneous of degree one and the
decomposition is the Euler one.** That is the condition, stated rather than assumed, and it is why the
budget is denominated in **percentage contribution to volatility**: volatility is homogeneous of degree
one and its gradient decomposition is exact, so the shares decompose the total rather than merely
resembling its parts. Expected shortfall satisfies the same condition.

**Value at risk does not, so VaR contributions are refused.** The refusal is verified rather than
asserted: `var_refusal` computes the conditional contributions a VaR report normally prints and shows
what they actually sum to. They sum to the expected shortfall, not to the value at risk - one is the
mean of the tail and the other is its edge - so a report presenting them as VaR contributions
overstates the quantity it claims to decompose. The failing case is kept as a documented negative, and
the positive case falls out of the same computation.

**Marginal contributions are reported and incremental ones are not.** The marginal contribution is the
gradient `d(sigma)/d(w_i)`, a property of the portfolio; an incremental contribution is the change in
risk from adding a sleeve, which depends on where the sleeve is added and in what order the others
arrived. Reporting an order-dependent number beside an order-independent one invites a reader to read
the first as the second, so the second appears only in the hand-checked case.

**The budget is read against the vector the mandate declared, on the out-of-sample window's own
covariance.** The covariance is held fixed across the run rather than re-estimated monthly, because the
question the budget answers is where the risk *went*: a moving covariance would make the consumption
and the declared target two moving objects that never meet. What moves is the book, which is the thing
under test. The declared vector is fixed in the data layer before construction, which is what makes
"consumed by design or by accident" a measurable quantity rather than a claim.

**Ex-ante against ex-post tracking error, with the difference as its own result.** The ex-ante number
comes from the covariance and the book; the ex-post number is the realised dispersion of the active
return. Their difference is a forecast-quality metric and is reported as its own line rather than left
for a reader to subtract.
"""

import numpy as np
import pandas as pd

from ..construct import families
from ..data import loader, universe
from ..evaluate import metrics

# The level the tail case is computed at: the worst five percent of the window's months, read from the
# constructor that constructs against it rather than restated, so the budget and the mean-CVaR cell
# speak of the same tail.
DEFAULT_LEVEL = families.CVAR_LEVEL

# The additivity residual is arithmetic - the Euler decomposition is an identity for a homogeneous
# measure - so the tolerance is for floating-point accumulation rather than for a modelling
# approximation, and the scale carries a floor because a relative bound around a zero volatility is not
# a bound at all.
ADDITIVITY_TOLERANCE = 1e-10
ADDITIVITY_SCALE_FLOOR = 1e-12


def _aligned(weights, covariance):
    """A weight vector and a covariance put in one order, or refused.

    A covariance carrying its own sleeve labels is reindexed onto the weights, because the two arrive
    from different modules and a silent positional mismatch would attribute one sleeve's risk to
    another. An unlabelled matrix is taken as it comes, which is what the hand-checked cases pass.
    """
    weights = pd.Series(weights, dtype=float)
    if isinstance(covariance, pd.DataFrame):
        weights = weights.reindex(covariance.index)
        if weights.isna().any():
            raise ValueError("the covariance does not cover every sleeve the book holds")
        matrix = covariance.to_numpy()
    else:
        matrix = np.asarray(covariance, dtype=float)
    if len(weights) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{len(weights)} weights do not match a {matrix.shape} covariance")
    return weights, matrix


def contributions(weights, covariance):
    """Every sleeve's Euler contribution to the book's volatility, its share of it, and its gradient.

    `RC_i = w_i (S w)_i / sigma`, which sums to `sigma` exactly because volatility is homogeneous of
    degree one: the identity is a property of the measure rather than an approximation to be tuned.
    The share and the contribution are reported together because a share without its level states a
    proportion and says nothing about size.
    """
    weights, matrix = _aligned(weights, covariance)
    variance = float(weights @ matrix @ weights)
    if variance <= 0.0:
        raise ValueError("a book with no variance has no volatility to decompose")
    sigma = float(np.sqrt(variance))
    marginal = matrix @ weights.to_numpy() / sigma
    contribution = weights.to_numpy() * marginal
    return pd.DataFrame(
        {"contribution": contribution, "share": contribution / sigma, "marginal": marginal},
        index=weights.index,
    )


def additivity(weights, covariance, tolerance=ADDITIVITY_TOLERANCE):
    """Whether the contributions sum to the volatility, with the residual and its scale.

    Run numerically rather than assumed, because the acceptance test would otherwise silently assume
    the condition it exists to check: a decomposition that stopped adding up would read as a budget
    disagreeing with itself rather than as a measure that does not decompose this way.
    """
    weights, matrix = _aligned(weights, covariance)
    table = contributions(weights, covariance)
    sigma = float(np.sqrt(weights @ matrix @ weights))
    residual = float(table["contribution"].sum()) - sigma
    relative = abs(residual) / max(abs(sigma), ADDITIVITY_SCALE_FLOOR)
    return {
        "volatility": sigma,
        "sum": float(table["contribution"].sum()),
        "residual": residual,
        "relative": relative,
        "reconciles": bool(relative <= tolerance),
        "tolerance": tolerance,
        "measure": "volatility, homogeneous of degree one",
    }


def budget(weights, covariance, target=universe.RISK_BUDGET):
    """The book's realised contribution shares beside the vector the mandate declared.

    The gap is reported per group rather than as one norm, because "consumed by design or by accident"
    is a question about which group took more than its share, and a distance between two vectors cannot
    say that. A group the book does not hold appears with a realised share of zero and its full target
    as the gap, which is the honest reading of a declared budget nothing was spent on.
    """
    table = contributions(weights, covariance)
    groups = [universe.INSTRUMENT_GROUP[sleeve] for sleeve in table.index]
    realised = pd.Series(table["share"].to_numpy(), index=groups).groupby(level=0).sum()
    rows = {}
    for group, share in target.items():
        held = float(realised.get(group, 0.0))
        rows[group] = {"target": float(share), "realised": held, "gap": held - float(share)}
    return {
        "by_sleeve": table,
        "by_group": pd.DataFrame(rows).T,
        "largest_gap": max(rows, key=lambda group: abs(rows[group]["gap"])),
        "undeclared": sorted(set(realised.index) - set(target)),
        "target": dict(target),
    }


def path_contributions(book, covariance, groups=universe.INSTRUMENT_GROUP):
    """One book's contribution shares month by month, and the consumption they average to.

    The covariance is held fixed across the path on purpose; what moves is the book. The group frame is
    an aggregation of the same shares rather than a second computation, so the sleeve table and the
    budget cannot disagree about the same month.
    """
    rows, volatility = {}, []
    for month in book.index:
        table = contributions(book.loc[month], covariance)
        rows[month] = table["share"]
        volatility.append(float(np.sqrt(book.loc[month] @ covariance.to_numpy() @ book.loc[month])))
    shares = pd.DataFrame(rows).T
    grouped = shares.T.groupby(by=[groups[sleeve] for sleeve in shares.columns]).sum().T
    return {
        "shares": shares,
        "mean_share": shares.mean(),
        "by_group": grouped,
        "mean_group": grouped.mean(),
        "volatility_annualised": float(np.mean(volatility) * np.sqrt(metrics.PERIODS_PER_YEAR)),
    }


def diversification(weights, covariance):
    """The book's diversification ratio, read from the constructor that maximises it rather than rebuilt.

    Above one whenever the sleeves do not move together, exactly one for a single sleeve, and a number
    about the covariance rather than about the book - which is what makes it worth printing beside a
    budget that is a statement about the book.
    """
    return families.diversification_ratio(weights, covariance)


def ex_ante_tracking_error(active_weights, covariance, periods=metrics.PERIODS_PER_YEAR):
    """The tracking error the covariance and the book imply, annualised.

    Denominated against the same covariance the contributions read, so the two numbers in the report
    describe one estimate rather than two.
    """
    active, matrix = _aligned(active_weights, covariance)
    variance = float(active @ matrix @ active)
    if variance < 0.0:
        raise ValueError("a negative variance is a covariance that is not positive semi-definite")
    return float(np.sqrt(variance * periods))


def ex_post_tracking_error(active_returns, periods=metrics.PERIODS_PER_YEAR):
    """The realised dispersion of the active return, annualised, on the evaluation module's definition."""
    return metrics.tracking_error(active_returns, periods=periods)


def forecast_quality(ex_ante, ex_post):
    """The two tracking errors and the difference between them, which is the result rather than a
    footnote: an ex-ante estimate that ran below the realised dispersion understated the risk the book
    took, and the shortfall is a property of the estimate rather than of the market."""
    return {
        "ex_ante": float(ex_ante),
        "ex_post": float(ex_post),
        "difference": float(ex_post) - float(ex_ante),
        "ratio": float(ex_post) / float(ex_ante) if ex_ante else None,
        "statement": "ex-post minus ex-ante: positive means the estimate understated the realised dispersion",
    }


def tail_contributions(returns, weights, level=DEFAULT_LEVEL):
    """The conditional contributions a tail report prints: each sleeve's weight times the mean of its
    return over the months the book lost more than its own quantile.

    They are additive - but to the **expected shortfall**, because the mean of the tail's returns is the
    tail mean by construction and the weighted sum of the tail means is the book's own tail mean. The
    identity is exact, which is what lets the next function show what is wrong with reading them as
    value-at-risk contributions.
    """
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    losses = -(returns @ weights)
    threshold = float(np.quantile(losses, 1.0 - level))
    tail = losses >= threshold
    if not tail.any():
        raise ValueError(f"no month is beyond the {level:.0%} quantile of this window")
    contribution = weights * -(returns[tail].mean(axis=0))
    return {
        "threshold": threshold,
        "tail_months": int(tail.sum()),
        "contribution": contribution,
        "sum": float(contribution.sum()),
        "tail_mean": float(losses[tail].mean()),
    }


def var_refusal(returns, weights, level=DEFAULT_LEVEL):
    """The documented negative behind the refusal: VaR contributions do not add up, and this measures it.

    The conditional contributions sum to the expected shortfall of the same window while the value at
    risk is the quantile itself - the edge of the tail rather than its mean - so the sum exceeds the VaR
    by the tail's own dispersion beyond the threshold. A report presenting them as VaR contributions
    would overstate the quantity it claims to decompose, and the magnitude is printed rather than
    asserted because a refusal justified by the wrong number is no better than an assertion.

    Expected shortfall passes the same computation, which is the positive half of the same case: the
    contributions *are* additive for it, because it is homogeneous of degree one and the value at risk is
    not. No VaR contribution is reported anywhere in this package.
    """
    conditional = tail_contributions(returns, weights, level=level)
    return {
        "value_at_risk": conditional["threshold"],
        "contributions_sum": conditional["sum"],
        "expected_shortfall": conditional["tail_mean"],
        "tail_months": conditional["tail_months"],
        "overstatement": conditional["sum"] - conditional["threshold"],
        "worse": "the value at risk is not homogeneous of degree one, so its contributions do not add to it",
        "homogeneous": "volatility and expected shortfall decompose this way; the value at risk does not",
        "refused": "no VaR contribution is reported: the decomposition is valid only for a homogeneous measure",
    }


def report(cells, document, policy, default=None):
    """The budget report: consumption against the declared vector, per run, with the condition checked
    numerically and the refused measure's failing case printed beside it."""
    months = cells[0]["contributions"]["shares"].index
    print(f"[risk] snapshot {document.snapshot_id}: the budget is read over {len(months)} out-of-sample "
          f"months {months.min()}..{months.max()}, one covariance held across the run")
    print(f"[risk] the declared vector is fixed in the data layer before construction: "
          f"{', '.join(f'{group} {share:.0%}' for group, share in universe.RISK_BUDGET.items())} of total "
          f"volatility, which is the denomination the mandate itself states")
    print(f"[risk] components sum to total risk when the measure is homogeneous of degree one and the "
          f"decomposition is the Euler one - true for volatility and expected shortfall, false for the "
          f"value at risk: checked numerically to {ADDITIVITY_TOLERANCE:.0e} relative on every run")
    # The columns are the declared vector's own keys, in its own order and under its own names: a group
    # renamed or added in the data layer moves this table, instead of leaving it printing four names that
    # no longer key it. The keys are printed as declared rather than abbreviated, so a reader can match a
    # column to the vector quoted two lines above.
    budget_columns = [(group, max(len(group) + 2, 9)) for group in universe.RISK_BUDGET]
    print(f"[risk] {'run':<34s}"
          + "".join(f"{group:>{width}s}" for group, width in budget_columns)
          + f"{'vol/yr':>9s}{'div ratio':>11s}{'additivity':>12s}")
    for cell in cells:
        groups = cell["budget"]["by_group"]
        print(f"[risk] {cell['id']:<34s}"
              + "".join(f"{groups.loc[group, 'realised']:>{width}.1%}" for group, width in budget_columns)
              + f"{cell['contributions']['volatility_annualised']:>9.2%}{cell['diversification']:>11.2f}"
              f"{cell['additivity']['residual']:>12.2e}")
    worst = max(cells, key=lambda cell: cell["budget"]["by_group"]["gap"].abs().max())
    group = worst["budget"]["largest_gap"]
    row = worst["budget"]["by_group"].loc[group]
    print(f"[risk] the largest departure from the declared vector is {worst['id']}'s {group} at "
          f"{row['gap']:+.1%} against a target of {row['target']:.0%}, so the budget is consumed by accident "
          f"wherever a run departs: the vector was declared against the policy book, whose own consumption is "
          f"{', '.join(f'{name} {policy["budget"]["by_group"].loc[name, "realised"]:.1%}' for name in universe.RISK_BUDGET)} "
          f"at {policy['contributions']['volatility_annualised']:.2%}/yr and a diversification ratio of "
          f"{policy['diversification']:.2f}")
    heaviest = max(cells, key=lambda cell: abs(cell["additivity"]["residual"]))
    print(f"[risk] the additivity residual is the largest on {heaviest['id']} at "
          f"{heaviest['additivity']['residual']:.2e} against volatilities of order "
          f"{np.mean([cell['contributions']['volatility_annualised'] for cell in cells]):.2%}/yr: every run "
          f"reconciles - {all(cell['additivity']['reconciles'] for cell in cells)}")
    leader = default if default is not None else cells[0]
    quality = leader["forecast"]
    ratio = "undefined" if quality["ratio"] is None else f"{quality['ratio']:.2f}"
    print(f"[risk] tracking error on {leader['id']}: ex-ante {quality['ex_ante']:.2%}/yr from the covariance "
          f"against ex-post {quality['ex_post']:.2%}/yr realised, a difference of "
          f"{quality['difference']:+.2%}/yr (ratio {ratio}) - {quality['statement']}")
    refusal = leader["refusal"]
    print(f"[risk] the refused measure, measured on {leader['id']}: the conditional contributions a VaR "
          f"report prints sum to {refusal['contributions_sum']:.4%} while the value at risk itself is "
          f"{refusal['value_at_risk']:.4%}, overstating it by {refusal['overstatement']:.4%} over "
          f"{refusal['tail_months']} tail months - {refusal['worse']}; {refusal['homogeneous']}")
    print(f"[risk] {refusal['refused']}")
    return cells


def main(root=None):
    """Load the snapshot, analyse it once, and read every book's consumption against the declared budget.

    The budget is read against the analysis's covariance, which the analysis names in its own report:
    a budget read against a covariance the cell was not built on is a statement about two objects.
    """
    from ..study import POLICY_BOOK, analyse

    analysis = analyse(loader.load_panel(root))
    cells = [book for identifier, book in analysis.budgets.items() if identifier != POLICY_BOOK]
    policy = analysis.budgets[POLICY_BOOK]
    heaviest = max(cells, key=lambda cell: cell["contributions"]["volatility_annualised"])
    report(cells, analysis.document, policy, default=heaviest)
    for cut in analysis.grid["cuts"]:
        print(f"[risk] {cut['id']}: cut from the grid, so it carries no budget: {cut['reason']}")
    for line in analysis.document.warnings:
        print(f"[risk] {line}")
    return {"cells": cells, "policy": policy, "covariance": analysis.covariance, "grid": analysis.grid}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
