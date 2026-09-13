"""The risk-model axis: three estimators of one object behind one interface.

The comparison varies the covariance estimator with the constructor held fixed, and what that
axis measures is **the optimiser's numerical behaviour**, never which estimator is more
accurate. Nothing here estimates accuracy: there is no true covariance to compare against on
this panel, and a sample covariance is by construction the best fit *to its own window*, so an
in-sample accuracy ranking would rank the estimator it was computed from first.

Three choices are made once for all three estimators so that the axis compares estimators and
not conventions. They are all returned on the **correlation-free covariance scale in the
sleeve map's order**, with the labels travelling with the matrix. They all use the same
degrees-of-freedom convention: the shrinkage estimator is published maximum-likelihood scaled
and is rescaled here, because a comparison between an estimator that divides by T and one that
divides by T-1 measures the division rather than the estimator. And the factor covariance is
built on the correlation scale and de-standardised at the end, so the count that decides it is
the count the rule decided rather than one a volatile sleeve would have driven.

The reconstruction identity is checked rather than asserted: the residual variances left after
the retained components are the discarded eigenvalues, summed. That is arithmetic, not an
empirical claim, and a violation would mean the factor covariance is not the decomposition it
is described as.

This module reads the component model from `factors/`, so `risk/` depends on it: the PCA factor
covariance *is* the statistical family's covariance, and re-extracting the components here to
avoid the import would be the same model built twice, which is how two modules come to disagree
about what a component is. The layout's dependency arrows name `risk/ -> data`; the estimator
this axis actually varies needs `factors` as well.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..data import loader, panel, universe
from ..factors import components as component_module
from ..factors.exposures import windows

# The reconstruction identity is arithmetic; the tolerance is for floating-point accumulation,
# not for a modelling approximation.
RECONSTRUCTION_TOLERANCE = 1e-10


def _ordered(values, columns):
    """A covariance carrying its own labels, checked for the size and names the labels imply.

    Every weight vector in the package is indexed by the sleeve map, so a covariance whose labels
    did not travel with it would misalign weights against returns without raising anywhere
    downstream: an optimiser would return a plausible vector for the wrong instruments and the
    resulting return series would look like a result. The labels are therefore attached here,
    once, rather than reconstructed by each caller.
    """
    columns = list(columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"the returns frame carries a repeated instrument: {columns}")
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(columns), len(columns)):
        raise ValueError(f"a {matrix.shape} matrix cannot be labelled by {len(columns)} instruments")
    return pd.DataFrame(matrix, index=columns, columns=columns)


def sample(returns):
    """The window's sample covariance, the estimator every other one is a modification of."""
    x = np.asarray(returns, dtype=float)
    if len(x) < 2:
        raise ValueError("a covariance needs at least two observations")
    return _ordered(np.cov(x, rowvar=False, ddof=1), returns.columns)


def shrinkage(returns):
    """Linear shrinkage, and the intensity it chose.

    The intensity is the answer to "how much of the sample is noise", so it is returned beside
    the matrix rather than kept inside it: a cell whose intensity runs near one is telling the
    reader that the sample carried almost no information, which is a result about the window and
    not a defect of the estimator.
    """
    x = np.asarray(returns, dtype=float)
    if len(x) < 2:
        raise ValueError("a covariance needs at least two observations")
    estimator = LedoitWolf().fit(x)
    # Ledoit-Wolf is maximum-likelihood scaled where the sample estimator here is not; matching
    # the convention keeps the conditioning comparison about the estimator rather than the divisor.
    rescaled = estimator.covariance_ * len(x) / (len(x) - 1)
    return _ordered(rescaled, returns.columns), float(estimator.shrinkage_)


def factor_model(returns, count=None, draws=component_module.DRAWS, seed=component_module.SEED):
    """The factor covariance: the retained components plus a diagonal residual.

    The components are extracted from the window's correlation matrix, so the retained count is
    the one the count rule decided for that window rather than one a high-variance sleeve would
    have forced. The residual is diagonal by construction, which is the model's assumption and
    the reason its reconstruction is exact on the correlation scale: the row communalities plus
    the residual variances reproduce the diagonal exactly.
    """
    x = np.asarray(returns, dtype=float)
    eigenvalues, vectors = component_module.eigen_structure(x)
    if count is None:
        count = component_module.decompose(returns, draws=draws, seed=seed)["components"]
    count = max(int(count), 0)
    # The loadings come from the one function that defines what a loading is - the oriented
    # eigenvector scaled by the square root of its own eigenvalue - so the sign convention
    # cannot be forgotten in a second copy of that arithmetic.
    loadings = component_module.component_loadings(eigenvalues, vectors, count)
    residual = 1.0 - (loadings ** 2).sum(axis=1)
    if (residual < -RECONSTRUCTION_TOLERANCE).any():
        raise ValueError(
            "a sleeve's communality exceeds one, so the factor covariance is not positive semi-definite"
        )
    discarded = float(eigenvalues[count:].sum())
    if abs(discarded - float(residual.sum())) > RECONSTRUCTION_TOLERANCE * max(len(residual), 1):
        raise ValueError(
            f"the residual variances sum to {residual.sum():.3e} where the discarded eigenvalues sum to "
            f"{discarded:.3e}; the factor covariance is not the decomposition it is reported as"
        )
    correlation = loadings @ loadings.T + np.diag(residual)
    spread = x.std(axis=0, ddof=1)
    covariance = correlation * spread[:, None] * spread[None, :]
    return _ordered(covariance, returns.columns), {
        "components": count,
        "discarded_share": discarded / len(residual),
        "residual_sum": float(residual.sum()),
        "discarded_sum": discarded,
    }


def conditioning(returns, count=None):
    """All three estimators on one window: what each does to the optimiser, and how they differ.

    The condition number is the quantity a constructor actually feels: a near-singular covariance
    makes the inverse-dominated families (minimum variance, maximum diversification) amplify
    estimation error in their weights, while the no-inversion families are exposed to it through
    the return series instead. No accuracy is claimed or measurable from these numbers.
    """
    plain = sample(returns)
    shrunk, intensity = shrinkage(returns)
    factored, report = factor_model(returns, count=count)
    return {
        "sample_condition": float(np.linalg.cond(plain)),
        "shrinkage_condition": float(np.linalg.cond(shrunk)),
        "factor_condition": float(np.linalg.cond(factored)),
        "intensity": intensity,
        "covariances": {"sample": plain, "shrinkage": shrunk, "factor": factored},
        **report,
    }


def main(root=None):
    document = loader.load_panel(root)
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])

    traded, estimation = list(windows(months))[-1]
    block = returns.reindex(estimation).dropna(how="all")
    report = conditioning(block)
    print(f"[risk] snapshot {document['snapshot_id']}, the window behind the last traded month "
          f"{traded}: {estimation[0]}..{estimation[-1]}, {len(block)} observations × {block.shape[1]} sleeves")
    print(f"[risk] sample covariance: condition number {report['sample_condition']:,.0f}")
    print(f"[risk] linear shrinkage: condition number {report['shrinkage_condition']:,.1f} at intensity "
          f"{report['intensity']:.3f} - the intensity is how much of the sample the estimator judged to be "
          f"noise, and it is a statement about the window rather than a defect")
    print(f"[risk] factor covariance: condition number {report['factor_condition']:,.1f} on {report['components']} "
          f"component(s), discarding {report['discarded_share']:.2%} of the correlation variance "
          f"({report['discarded_sum']:.6f} against a residual sum of {report['residual_sum']:.6f})")
    for name, covariance in report["covariances"].items():
        if list(covariance.index) != universe.TICKERS:
            raise ValueError(f"the {name} covariance lost the sleeve map's order")
        if not np.allclose(covariance.to_numpy(), covariance.to_numpy().T):
            raise ValueError(f"the {name} covariance is not symmetric")
    print("[risk] all three carry the sleeve map's order and are symmetric; the axis measures the optimiser's "
          "numerical behaviour, and no estimator is ranked for accuracy, which this panel cannot establish")
    print(f"[risk] the three diagnostics this window produces for the estimator axis: sample conditioning "
          f"{report['sample_condition']:,.0f}, shrunk {report['shrinkage_condition']:,.0f}, factor "
          f"{report['factor_condition']:,.0f}")
    for line in document["warnings"]:
        print(f"[risk] {line}")
    return {"report": report, "window": (traded, estimation), "returns": returns}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
