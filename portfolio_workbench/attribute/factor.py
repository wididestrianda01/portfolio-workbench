"""Factor attribution: the explanation layer beside the holding-based one, and the gap named.

**Two decompositions of one number, and neither is ever added to the other.** The holding-based
Brinson decomposition of `attribute/brinson.py` is what the committee reads: it reconciles to the
benchmark by construction and speaks the mandate's language. This is the explanation layer - *why* the
allocation effects came out as they did - and it decomposes the **same** active return into factor
exposures times factor returns, plus alpha, plus residual. The only legitimate sum in the system is
that total, so the two views are printed side by side and never summed; the difference between them is
a single named cross-view residual, and the word "reconciles" is used only where that residual is
inside a stated tolerance.

**The residual is a definition check, not a modelling error.** Both views decompose the same series, so
the cross-view residual is zero by construction and the case proves the definition rather than the
arithmetic - which is worth having precisely because it fires when one view is silently reading a
different month set, a different weight path or a different return frame. It is reported as a number
with its scale beside it, for the same reason every other residual in this package is.

**The coefficients are read on the basis they were fitted in, and the basis travels with the month.**
The block is fitted net of its predecessors inside each window, so the loadings on its columns belong
to the orthogonalised series rather than the declared ones. Attributing a traded month therefore needs
that month on the same basis, and the window's own map is the only object that carries a basis across
the boundary between the window and the month it is applied to. The map is a linear functional once the
window is fixed, exactly as a loading is, so applying it to a month outside the window is the same kind
of extrapolation the attribution already makes with every coefficient.

**The four construction sleeves are the block, so their return is attributed to the block in full.**
Their loadings on the published spine are a summary of the same return expressed in another basis - a
sensitivity, and a useful one - not a second decomposition of it; adding both would count their return
twice. Their identity loadings are moved onto the orthogonal basis here so that every sleeve in the
report is read against the same series, and their residual is exactly zero because their model is.

**Risk attribution is factor-based only**, because Brinson is a return decomposition with no risk
dimension: the factor and idiosyncratic parts of tracking error are read off the same
`factors.exposures.factor_split` the factor layer reports, so the split the budget quotes and the split
this module quotes cannot come apart.

**The fund decomposition is a worked example and its skill claim is refused.** One book is read as
though it were an external manager's track record and its residual tested against the family-wise bar.
The finding is stated before the analysis: on 131 months with a few factors, a residual t-statistic
cannot clear that bar for a fund, so the method is demonstrated and the claim is not made.
"""

import numpy as np
import pandas as pd

from ..compare import grid as grid_module, registry
from ..data import loader
from ..evaluate import metrics, statistics
from ..factors import exposures, spine as spine_module
from . import brinson

# The gap between the two views, and the tolerance below which "reconciles" may be used. Both are
# decompositions of one series, so anything above accumulation is one of them reading a different
# month, weight path or return frame rather than a modelling error.
CROSS_VIEW_TOLERANCE = 1e-9
CROSS_VIEW_SCALE_FLOOR = 1e-6

# The block, read from the module that declares it rather than restated: the attribution and the
# regression have to agree about which columns were orthogonalised or the basis they are read on
# differs between them.
BLOCK = exposures.BLOCK


def design(fit, factors):
    """Every month's coefficients and the series they are read against, on one basis.

    Returns a coefficient frame and a value frame per factor, both month by sleeve, so the
    decomposition is `sum over i of a_i,t * coefficient[i,f,t] * value[i,f,t]` summed over the factors.
    The values are frames rather than one vector per factor because the block's series differ by
    sleeve: a construction sleeve is read against the block in its declared basis, everything else
    against the same series net of its predecessors.

    Three things here are decisions rather than mechanics. The orthogonal map is the window's, and it
    is applied to the traded month - the coefficients belong to the window and the month they are
    applied to is not in it. The construction sleeves' spine coefficients are zeroed and their identity
    loadings are solved into the orthogonal basis, because their return is the block and a spine
    loading is a second reading of it rather than a second part of it. And every frame is reindexed onto
    the fit's own traded months, so a month cannot enter the attribution that the fit did not cover.
    """
    block = list(BLOCK)
    months = pd.PeriodIndex(fit["traded"], freq="M")
    spine_columns = [column for column in factors.columns if column not in block]
    sleeves = list(fit["beta"][factors.columns[0]].columns)
    coefficients = {column: pd.DataFrame(index=months, columns=sleeves, dtype=float) for column in factors.columns}
    values = {column: pd.DataFrame(index=months, columns=sleeves, dtype=float) for column in factors.columns}
    declared = pd.DataFrame(index=months, columns=block, dtype=float)
    orthogonal = pd.DataFrame(index=months, columns=block, dtype=float)
    maps = {}
    for position, month in enumerate(months):
        window = pd.period_range(fit["window_first"][position], fit["window_last"][position], freq="M")
        window_block = exposures.window_block(factors[block], window)
        maps[month] = exposures.orthogonal_transform(window_block)
        declared.loc[month] = factors.loc[month, block].to_numpy()
        orthogonal.loc[month] = factors.loc[month, block].to_numpy() @ maps[month]
        for column in factors.columns:
            coefficients[column].loc[month] = fit["beta"][column].loc[month].to_numpy()
            values[column].loc[month] = float(
                orthogonal.loc[month, column] if column in block else factors.loc[month, column]
            )
        for sleeve, loadings in fit["identity"][position].items():
            for column in spine_columns:
                coefficients[column].loc[month, sleeve] = 0.0
            converted = np.linalg.solve(maps[month], np.asarray(loadings, dtype=float))
            for place, column in enumerate(block):
                coefficients[column].loc[month, sleeve] = converted[place]
    return {
        "months": months,
        "sleeves": sleeves,
        "factors": list(factors.columns),
        "coefficients": coefficients,
        "values": values,
        "declared": declared,
        "orthogonal": orthogonal,
        "maps": maps,
        "alpha": fit["alpha"].reindex(months).fillna(0.0),
    }


def design_covariance(fit, factors):
    """Each window's own covariance of the regressors the coefficients were fitted against.

    The covariance the risk split reads has to be the one of the series the loadings belong to, which
    is why it is taken from the design matrix rather than from the named factors: the block's columns
    are net of their predecessors inside the window, and a covariance of the declared series would be
    the covariance of a different set of regressors.
    """
    block = list(BLOCK)
    spine_columns = [column for column in factors.columns if column not in block]
    order = [*spine_columns, *block]
    out = {}
    for position, month in enumerate(pd.PeriodIndex(fit["traded"], freq="M")):
        window = pd.period_range(fit["window_first"][position], fit["window_last"][position], freq="M")
        spine = exposures.window_block(factors[spine_columns], window)
        orthogonal = exposures.orthogonalise(exposures.window_block(factors[block], window))
        design = pd.concat([spine, orthogonal], axis=1)[order]
        out[month] = pd.DataFrame(np.cov(design.to_numpy(), rowvar=False, ddof=1), index=order, columns=order)
    return out


def decomposition(active_weights, returns, fit, factors, model=None):
    """The active return split into factor contributions, alpha and residual, month by month.

    The residual is the part of the realised active return the fitted model does not explain in that
    month. It is computed as the remainder rather than estimated, which is what makes the identity
    exact; its realised size is then compared against the variance the windows estimated, in the
    report, because in-sample residual variance and out-of-sample residual size are different objects
    and only the second is a result.
    """
    model = design(fit, factors) if model is None else model
    months, sleeves = model["months"], model["sleeves"]
    weights = active_weights.reindex(index=months, columns=sleeves)
    if weights.isna().to_numpy().any():
        raise ValueError("the weight path does not cover every month and sleeve the fit covers")
    contributions = pd.DataFrame(
        {
            column: (weights * model["coefficients"][column] * model["values"][column]).sum(axis=1)
            for column in model["factors"]
        }
    )
    alpha = (weights * model["alpha"]).sum(axis=1)
    realised = (weights * returns.reindex(index=months, columns=sleeves)).sum(axis=1)
    explained = contributions.sum(axis=1) + alpha
    return {
        "months": months,
        "factors": contributions,
        "explained": explained,
        "alpha": alpha,
        "active": realised,
        "residual": realised - explained,
        "total": realised,
        "residual_estimate": (weights ** 2 * fit["resid_var"].reindex(index=months, columns=sleeves)).sum(axis=1),
        "model": model,
    }


def cross_view(holding, factor_view, tolerance=CROSS_VIEW_TOLERANCE):
    """The gap between the two views of one active return, as a named residual with its scale.

    Each view is taken complete - the factor view including the part of the return its model does not
    explain - so the gap is zero by construction and is kept for that reason: it is the check that
    fires the moment one view reads a different month set, weight path or return frame, and a check
    that cannot fire is worth less than one nobody would think to write. The unexplained part is *not*
    absorbed into the gap: it is reported as its own measured quantity, because how much of an active
    return a ten-factor model leaves unexplained is a result and not a reconciliation term.
    """
    holding = pd.Series(holding, dtype=float)
    factor_view = pd.Series(factor_view, dtype=float).reindex(holding.index)
    if factor_view.isna().any():
        raise ValueError("the factor view does not cover every month the holding view does")
    residual = holding - factor_view
    scale = float(holding.abs().max())
    return {
        "residual": residual,
        "worst": float(residual.abs().max()),
        "mean": float(residual.mean()),
        "scale": scale,
        "relative": float(residual.abs().max()) / max(scale, CROSS_VIEW_SCALE_FLOOR),
        "reconciles": bool(float(residual.abs().max()) <= tolerance * max(scale, CROSS_VIEW_SCALE_FLOOR)),
        "tolerance": tolerance,
        "statement": "one residual per period between two decompositions of the same active return",
    }


def risk_attribution(active_weights, fit, factors, model=None, covariances=None):
    """The factor and idiosyncratic parts of the tracking error each window's estimate implies.

    Read off `exposures.factor_split`, which is the one definition of the split in the package, so the
    budget's ex-ante number and this one are the same arithmetic. Reported as annualised volatility,
    because the split of a variance is exact and the split of its square root is not: the two parts
    printed here are the square roots of the parts that sum.
    """
    model = design(fit, factors) if model is None else model
    covariances = design_covariance(fit, factors) if covariances is None else covariances
    months, sleeves = model["months"], model["sleeves"]
    weights = active_weights.reindex(index=months, columns=sleeves)
    if weights.isna().to_numpy().any():
        raise ValueError("the weight path does not cover every month and sleeve the fit covers")
    rows = {}
    order = list(covariances[months[0]].columns)
    for month in months:
        beta = np.vstack([model["coefficients"][column].loc[month].to_numpy() for column in order])
        factor_part, idio_part = exposures.factor_split(
            weights.loc[month].to_numpy(),
            beta,
            covariances[month].to_numpy(),
            fit["resid_var"].loc[month].to_numpy(),
        )
        rows[month] = {"factor_part": factor_part, "idio_part": idio_part}
    frame = pd.DataFrame(rows).T
    frame["total"] = frame["factor_part"] + frame["idio_part"]
    total = float(frame["total"].mean())
    # A book that reproduces the benchmark has no tracking error to split, and the share of nothing is
    # not zero - it is undefined, and reporting it as zero would read as a model that explains none of
    # an empty quantity.
    return {
        "months": months,
        "variance": frame,
        "factor_tracking_error": float(np.sqrt(frame["factor_part"].mean() * metrics.PERIODS_PER_YEAR)),
        "idio_tracking_error": float(np.sqrt(frame["idio_part"].mean() * metrics.PERIODS_PER_YEAR)),
        "total_tracking_error": float(np.sqrt(total * metrics.PERIODS_PER_YEAR)),
        "factor_share": float(frame["factor_part"].mean() / total) if total > 0.0 else None,
        "split": "factors.exposures.factor_split, on the out-of-sample months' own windows",
    }


def fund_example(series, factors, bar):
    """One book read as though it were an external manager's track record, and the claim refused.

    A fund's residual alpha is tested against the family-wise bar, and the sample's own detection limit
    is reported beside it: the smallest alpha this many months could have detected at eighty percent
    power. The finding is stated in advance of the number - on 131 months with a few factors the bar is
    out of reach for a fund - so what the example shows is the method and the resolution limit rather
    than a verdict about any sleeve.

    It is a worked example and not a module for two reasons that are worth stating: it adds no
    machinery the factor layer does not already carry, and nothing in the research questions needs it.

    The three refusals are returned rather than raised, and no caller catches anything: a track record
    that never moves, one too short for the regression, and a factor frame that does not cover its
    months. Each is an answer about the book being read, and a caller that swallowed whatever the
    regression raised as one of them would be reporting a broken fit as a finding about a manager.
    """
    values = pd.Series(series, dtype=float)
    # A track record that never moves has no residual to test and no standard error to divide by; the
    # refusal is explicit rather than a NaN, because a NaN t-statistic would print as a number.
    if float(values.std(ddof=1)) == 0.0:
        return {"refused": "the track record does not vary, so there is no residual to test",
                "months": int(len(values))}
    months = pd.PeriodIndex(values.index, freq="M").intersection(pd.PeriodIndex(factors.index, freq="M"))
    if len(months) < exposures.MIN_OBS:
        return {"refused": f"{len(months)} months cannot support the {len(factors.columns)}-factor regression",
                "months": int(len(values))}
    y = pd.DataFrame({"fund": values.reindex(months)})
    frame = factors.reindex(months)
    if frame.isna().to_numpy().any():
        return {"refused": "the factor frame does not cover the fund's months", "months": int(len(values))}
    fit = exposures.regress(y, frame)
    alpha, t_alpha = float(fit["alpha"][0]), float(fit["t_alpha"][0])
    standard_error = abs(alpha / t_alpha) if t_alpha else float("nan")
    detectable = statistics.POWER * standard_error
    return {
        "months": len(months),
        "factors": list(frame.columns),
        "alpha_monthly": alpha,
        "alpha_annualised": alpha * metrics.PERIODS_PER_YEAR,
        "t_alpha": t_alpha,
        "residual_volatility_annualised": float(np.sqrt(fit["resid_var"][0]) * np.sqrt(metrics.PERIODS_PER_YEAR)),
        "r2": float(fit["r2"][0]),
        "bar": bar,
        "detectable_alpha_monthly": detectable,
        "detectable_alpha_annualised": detectable * metrics.PERIODS_PER_YEAR,
        "clears": bool(abs(t_alpha) >= bar),
        "claim": (
            "no skill claim is made: a residual t-statistic must clear the family-wise bar, and the "
            "sample's own detection limit is reported so a reader can see that failing it is a limit of "
            "131 months rather than a fact about the sleeve"
        ),
    }


def report(cells, document, fit, holding, count=5):
    """Both views side by side, the risk split, and the fund example with its claim refused."""
    months = holding[0]["decomposition"]["months"]
    print(f"[attrib] snapshot {document.snapshot_id}: {len(fit['traded'])} refits "
          f"{months.min()}..{months.max()} on {len(holding[0]['decomposition']['factors'].columns)} factors")
    print(f"[attrib] two views of one active return, never added to each other: the holding-based "
          f"Brinson total from {brinson.BRINSON_FACHLER}, and this factor decomposition as the explanation")
    # The holding-based column is the sum of the monthly lines, unlinked, because the cross-view residual
    # beside it is a monthly identity: linking is what turns those lines into a compounded total, and the
    # linked total is the one `attribute/brinson.py` reports. Two numbers under one name would be read as
    # a disagreement between the views, so the column says which of the two it is.
    print(f"[attrib] {'cell':<34s}{'brinson-sum':>12s}{'factors':>11s}{'alpha':>10s}{'unexplained':>13s}"
          f"{'cross-view':>12s}")
    for cell, composition in zip(cells, holding):
        view, parts = composition["cross_view"], composition["decomposition"]
        print(f"[attrib] {cell['id']:<34s}"
              f"{float(cell['lines'][list(brinson.LINES)].sum(axis=1).sum()):>+12.4%}"
              f"{float(parts['factors'].sum().sum()):>+11.4%}"
              f"{float(parts['alpha'].sum()):>+10.4%}"
              f"{float(parts['residual'].sum()):>+13.4%}"
              f"{view['worst']:>12.2e}")
    worst = max(holding, key=lambda composition: composition["cross_view"]["worst"])
    print(f"[attrib] the cross-view residual is largest on {worst['id']} at "
          f"{worst['cross_view']['worst']:.2e} against a scale of {worst['cross_view']['scale']:.4%} "
          f"(relative {worst['cross_view']['relative']:.1e}, tolerance {worst['cross_view']['tolerance']:.0e}): "
          f"zero by construction, since both views decompose the same series, and kept because it fires "
          f"the moment one of them reads a different month set - reconciles on every run: "
          f"{all(composition['cross_view']['reconciles'] for composition in holding)}")
    leader = max(cells, key=lambda cell: float(cell["linked"]["carino"]["total"]))
    composition = next(entry for entry, cell in zip(holding, cells) if cell is leader)
    ranked = composition["decomposition"]["factors"].sum().abs().sort_values(ascending=False).index[:count]
    print(f"[attrib] {leader['id']}, factor by factor over {len(months)} months (summed contributions, "
          f"then alpha and the unexplained residual):")
    for column in ranked:
        print(f"[attrib]   {column:<22s} {float(composition['decomposition']['factors'][column].sum()):>+10.4%}")
    print(f"[attrib]   {'alpha':<22s} {float(composition['decomposition']['alpha'].sum()):>+10.4%}   "
          f"{'residual':<22s} {float(composition['decomposition']['residual'].sum()):>+10.4%}   "
          f"realised residual volatility "
          f"{float(composition['decomposition']['residual'].std(ddof=1) * np.sqrt(metrics.PERIODS_PER_YEAR)):.2%}/yr "
          f"against the windows' estimate of "
          f"{float(np.sqrt(composition['decomposition']['residual_estimate'].mean()) * np.sqrt(metrics.PERIODS_PER_YEAR)):.2%}/yr")
    risk = composition["risk"]
    share = "undefined, the book holding no tracking error" if risk["factor_share"] is None else f"{risk['factor_share']:.1%} factor"
    print(f"[attrib] {leader['id']}'s tracking error, factor-based only, split by the one definition in the "
          f"package: factor part {risk['factor_tracking_error']:.2%}/yr and idiosyncratic part "
          f"{risk['idio_tracking_error']:.2%}/yr against a total of {risk['total_tracking_error']:.2%}/yr "
          f"({share}), read off {risk['split']}")
    example = composition["fund"]
    if "refused" in example:
        print(f"[attrib] the fund decomposition is not run on {leader['id']}: {example['refused']}")
        return {"cells": cells, "holding": holding, "fit": fit}
    print(f"[attrib] the fund decomposition, as a worked example only: {leader['id']} read as an external "
          f"manager's track record over {example['months']} months gives alpha "
          f"{example['alpha_annualised']:+.2%}/yr at t {example['t_alpha']:+.2f} against "
          f"{len(example['factors'])} factors, residual volatility "
          f"{example['residual_volatility_annualised']:.2%}/yr, R2 {example['r2']:.2f}")
    print(f"[attrib] the family-wise bar is {example['bar']:.4f} and the smallest alpha this sample could "
          f"have detected at eighty percent power is {example['detectable_alpha_annualised']:.2%}/yr: "
          f"{example['claim']}")
    for line in document.warnings:
        print(f"[attrib] {line}")
    return {"cells": cells, "holding": holding, "fit": fit}


def main(root=None):
    """Load the snapshot, fit the factor model, and put the two views side by side on one analysis."""
    from ..study import analyse

    analysis = analyse(loader.load_panel(root))
    document, returns, months = analysis.document, analysis.returns, analysis.months
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document.factors.eur, block)
    fit = exposures.rolling(returns, named, months)
    model = design(fit, named)
    covariances = design_covariance(fit, named)

    cells = [analysis.attribution[result["id"]] for result in analysis.grid["results"]]
    benchmark_weights = grid_module.policy_weights(returns.columns)
    bar = statistics.family_wise_bar(len(registry.CELLS))
    holding = []
    for result, holding_block in zip(analysis.grid["results"], cells):
        active_weights = result["weights"].sub(benchmark_weights, axis=1)
        composition = decomposition(active_weights, returns, fit, named, model=model)
        active = metrics.active(result["net"], analysis.benchmark)
        holding.append(
            {
                "id": result["id"],
                "decomposition": composition,
                # The cross-view residual holds the factor decomposition against the holding-based one
                # on the same active series, which is what makes two views of one return a check rather
                # than two accounts of it.
                "cross_view": cross_view(
                    holding_block["lines"][list(brinson.LINES)].sum(axis=1), composition["total"]
                ),
                "risk": risk_attribution(active_weights, fit, named, model=model, covariances=covariances),
                "fund": fund_example(active, named, bar),
            }
        )
    report(cells, document, fit, holding)
    return {"cells": cells, "holding": holding, "fit": fit, "named": named, "returns": returns,
            "grid": analysis.grid}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
