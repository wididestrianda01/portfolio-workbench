"""Spanning, tested in both directions, and the premia the factors are credited with.

The evidential core of the factor work is not which set explains more variance - it is whether
one set is needed at all once the other is present. So the two sets are tested against each
other: the **named set as benchmark with the statistical components as test assets** is the
headline, because it asks whether the published and constructed factors describe this panel,
and the reverse direction prints beside it rather than being dropped, because a one-directional
answer to a two-directional question is a selection of the result.

The test form is Huberman and Kandel (1987) in the spanning reading of their 1987 result: regress
each test asset
on the benchmark set, and the benchmark spans the test assets when, jointly, **all intercepts
are zero** and **the rows of the loadings sum to one**. Gibbons, Ross and Shanken (1989)
supplies the finite-sample F test for the joint-intercepts half. Both halves are reported.

Two limits travel with the result and neither is a footnote. The rows-sum-to-one half is a
statement about a *fully invested* benchmark: it says that a portfolio of the benchmark factors
reproduces the test asset with no residual, and one condition for that is that the exposures add
up to the whole asset. This benchmark mixes an excess-market series with zero-cost spread series
- the momentum, term, credit and high-yield legs invest nothing - so the condition is reported
as computed and its failure is not read as a spanning verdict on its own; the GRS half carries
the verdict. And **GRS assumes iid normal residuals**, which monthly ETF returns violate in the
tails, so the statistic travels with the assumption rather than resting on it silently.

Nothing here is evidence about European UCITS ETFs in general, and no published head-to-head of
statistical against fundamental risk models on this universe exists to replicate. The result is
a statement about these eleven sleeves over this window.
"""

import numpy as np
from scipy.stats import f as f_distribution

from ..data import loader, panel
from . import components as component_module
from . import exposures
from . import spine as spine_module

# A residual covariance this ill-conditioned cannot support the inverse the statistic is built
# on. The bound is a numerical floor, not a finding about the data.
CONDITION_LIMIT = 1e10
# Stated before any of these numbers existed. 3.0 is the general multiple-testing bar the
# literature settled on; the cross-sectional and time-series figures are the higher bars the
# decision carried for the two ways a premium can look significant.
PREMIUM_BARS = {"general": 3.0, "cross_sectional": 3.4, "time_series": 3.8}


def grs(test_assets, factors):
    """The GRS statistic, its p-value, and the rows-sum-to-one half of the spanning condition.

    The covariances are maximum-likelihood scaled, which is the scaling the F distribution is
    derived for: a degrees-of-freedom correction here would give a p-value belonging to a
    different sample size than the one printed beside it.
    """
    y = np.asarray(test_assets, dtype=float)
    f = np.asarray(factors, dtype=float)
    if np.isnan(y).any() or np.isnan(f).any():
        raise ValueError("the spanning sample carries a missing value; a filled one would enter the statistic")
    observations, assets = y.shape
    benchmarks = f.shape[1]
    if observations - assets - benchmarks <= 0:
        raise ValueError(
            f"{observations} observations cannot test {assets} assets against {benchmarks} benchmarks: the "
            f"statistic's F distribution has no degrees of freedom left"
        )

    design = np.column_stack([np.ones(observations), f])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    alpha, beta = coef[0], coef[1:]
    residual = y - design @ coef
    sigma = residual.T @ residual / observations
    centred = f - f.mean(axis=0)
    omega = centred.T @ centred / observations
    mu = f.mean(axis=0)

    condition = float(np.linalg.cond(sigma))
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        # A zero residual covariance - test assets the benchmark reproduces exactly - arrives here
        # as a non-finite condition number. It is the most degenerate case of the same fault, not a
        # statistic of zero, so it is refused in the same place.
        raise ValueError(
            f"the residual covariance cannot be inverted (condition number {condition}): a test asset is an "
            f"exact combination of the others, or the benchmark reproduces one exactly, and the statistic is "
            f"undefined"
        )
    statistic = (observations - assets - benchmarks) / assets * float(
        alpha @ np.linalg.inv(sigma) @ alpha
    ) / (1.0 + float(mu @ np.linalg.inv(omega) @ mu))
    return {
        "statistic": statistic,
        "p_value": float(f_distribution.sf(statistic, assets, observations - assets - benchmarks)),
        "assets": assets,
        "benchmarks": benchmarks,
        "observations": observations,
        "alpha": alpha,
        "beta": beta,
        # The Huberman-Kandel half: one row per test asset, and one under spanning for a fully
        # invested benchmark. Returned per asset rather than as a single number, because which
        # asset fails the condition is the part a reader can act on.
        "rows_sum": beta.sum(axis=0),
        "condition": float(np.linalg.cond(sigma)),
        "factor_sharpe": float(np.sqrt(mu @ np.linalg.inv(omega) @ mu)),
    }


def directions(returns, named, count):
    """Both spanning directions for one component count.

    The component series enter as portfolios of the same excess returns the named set is
    measured on, so neither direction has been rescaled into significance.
    """
    portfolios = component_module.component_portfolios(returns, count=count)
    common = returns.index.intersection(named.index)
    return {
        "headline": grs(portfolios.loc[common], named.loc[common]),
        "reverse": grs(named.loc[common], portfolios.loc[common]),
        "components": count,
        "months": common,
        "portfolios": portfolios.loc[common],
    }


def premiums(factors):
    """Each factor's mean against its own time-series t-statistic.

    Reported for every factor in the named set, including the ones whose t-statistic sits
    nowhere near a bar: the point of stating the thresholds in advance is that the ones that do
    not clear them stay visible.
    """
    out = {}
    for column in factors.columns:
        series = factors[column]
        error = float(series.std(ddof=1) / np.sqrt(len(series)))
        out[column] = {
            "mean": float(series.mean()),
            "t": float(series.mean() / error) if error else float("nan"),
            "months": len(series),
        }
    return out


def fama_macbeth(returns, factors):
    """The two-pass cross-sectional regression, as a labelled illustration.

    Carried because a premium claim is a cross-sectional claim, and refusing to run it would
    hide the question rather than answer it. With eleven sleeves and eleven second-stage
    parameters the cross-section is **exactly identified**: every month fits perfectly, the
    residual degrees of freedom are zero, and the premia are an algebraic readout of the sample
    rather than estimates of anything. The identification is computed and returned, so the label
    rests on a number. The t-statistics use Fama-MacBeth standard errors and are read as an
    illustration, which is all this procedure can be on a panel this size.
    """
    common = returns.index.intersection(factors.index)
    y = np.asarray(returns.loc[common], dtype=float)
    beta = exposures.regress(returns.loc[common], factors.loc[common])["beta"]
    design = np.column_stack([np.ones(len(returns.columns)), beta.T])
    dof = len(returns.columns) - design.shape[1]
    slopes = np.linalg.lstsq(design, y.T, rcond=None)[0]
    errors = slopes.std(axis=1, ddof=1) / np.sqrt(slopes.shape[1])
    names = ["intercept", *list(factors.columns)]
    return {
        "premia": {
            name: {
                "mean": float(slopes[position].mean()),
                "t": float(slopes[position].mean() / errors[position]) if errors[position] else float("nan"),
            }
            for position, name in enumerate(names)
        },
        "residual_dof": int(dof),
        "identified": dof <= 0,
        "months": int(slopes.shape[1]),
    }


def main(root=None):
    document = loader.load_panel(root)
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    named = spine_module.named_set(document["factors"]["eur"], spine_module.constructed_block(returns))

    rule = component_module.decompose(returns)
    # The counts the loop runs. The family size below is this loop's own, and both directions are tested
    # at each count, so a count added to the run cannot leave the reported family behind.
    counts = sorted({rule["components"], component_module.PREREGISTERED_K})
    for count in counts:
        run = directions(returns, named, count)
        label = "the count the rule retains" if count == rule["components"] else "the pre-registered count"
        headline, reverse = run["headline"], run["reverse"]
        print(f"[factor] spanning over {len(run['months'])} months at {count} component(s), {label}; iid-normal "
              f"residuals assumed, and monthly returns violate that in the tails")
        print(f"    headline: the named set ({headline['benchmarks']} factors) against the components "
              f"({headline['assets']} assets) - GRS {headline['statistic']:.3f}, p {headline['p_value']:.4f}, "
              f"joint intercepts {'rejected' if headline['p_value'] < 0.05 else 'not rejected'}")
        print(f"    reverse: the components ({reverse['benchmarks']}) against the named set "
              f"({reverse['assets']} assets) - GRS {reverse['statistic']:.3f}, p {reverse['p_value']:.4f}, "
              f"printed beside the headline rather than dropped")
        rows = headline["rows_sum"]
        print(f"    rows-sum-to-one half, headline direction: the loadings' row sums run {rows.min():+.2f}.."
              f"{rows.max():+.2f} against 1.00 under spanning, largest deviation "
              f"{np.abs(rows - 1.0).max():.2f}; the components are fully invested portfolios, so this is a "
              f"statement about net exposure")
        reverse_rows = reverse["rows_sum"]
        print(f"    rows-sum-to-one half, reverse direction: {reverse_rows.min():+.2f}..{reverse_rows.max():+.2f}, "
              f"reported as computed - the test assets there are the named factors themselves, and the "
              f"zero-cost spread series among them are not the fully invested case the condition is stated for")
    verdicts = 2 * len(counts)
    print(f"[factor] the verdict family is {verdicts} joint tests over {len(counts)} count(s), the named set "
          f"against the components and back - {rule['components']} test assets in one direction and "
          f"{len(named.columns)} in the other - so Bonferroni at 5% needs p < {0.05 / verdicts:.4f}")

    premium = premiums(named)
    clearing = [name for name, stats in premium.items() if abs(stats["t"]) >= PREMIUM_BARS["time_series"]]
    strongest = max(premium, key=lambda name: abs(premium[name]["t"]))
    print(f"[factor] time-series premia against the ex-ante {PREMIUM_BARS['time_series']} bar: {len(clearing)} of "
          f"{len(premium)} clear it ({', '.join(clearing) if clearing else 'none'}); the strongest is {strongest} "
          f"at |t| = {abs(premium[strongest]['t']):.2f}")
    for name, stats in premium.items():
        print(f"    {name:<20s} {stats['mean']:+.3%}/month over {stats['months']} months, t {stats['t']:+.2f}")

    cross = fama_macbeth(returns, named)
    bars = ", ".join(f"{key} {value}" for key, value in PREMIUM_BARS.items())
    print(f"[factor] cross-sectional illustration: {len(returns.columns)} sleeves against "
          f"{len(named.columns) + 1} second-stage parameters leaves {cross['residual_dof']} residual degrees of "
          f"freedom, so the stage is "
          f"{'exactly identified and its premia are an algebraic readout' if cross['identified'] else 'thinly identified'}")
    print(f"[factor] the bars were stated ex ante ({bars}); Fama-MacBeth here is an illustration with its limits "
          f"computed, never asset-pricing evidence from this panel")
    print("[factor] describes this panel only: one eleven-sleeve universe, one window, and no published "
          "head-to-head on this universe to replicate")
    for line in document["warnings"]:
        print(f"[factor] {line}")
    return {"rule": rule, "premium": premium, "cross": cross, "returns": returns, "named": named}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
