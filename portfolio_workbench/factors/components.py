"""The statistical family: principal components of the panel's own returns, and the rule that
decides how many of them exist.

The count is decided by an external criterion stated in advance, never by out-of-sample
performance: choosing the count on the results it produces and then reporting it is circular.
Here the criterion is a **matched empirical null**. For each estimation window the null is
built by giving every series its own independent random circular shift, which preserves each
series' values, its marginal distribution and its autocorrelation, and destroys only the
alignment between series - the cross-correlation the test needs a null for. A plain
permutation of each series would also destroy the series' own serial dependence, making the
null less correlated in time than the panel and the threshold too low, which is the wrong
direction for a rule that decides how many factors to trust.

Two reference points are printed beside the mechanical count. The **Marchenko-Pastur edge**
`(1 + sqrt(N/T))^2` is the analytic bulk edge for an uncorrelated panel and is a cross-check on
the simulation rather than the threshold, because monthly returns are not normal and the edge
assumes they are. A **pre-registered fixed count** of three is reported as well: it was fixed before the rule was
applied, so a mechanical count that disagrees with it is a result to report rather than a number
to overwrite.

Both of these are counts on a correlation matrix, so every share of variance prints beside the
**pure-noise share** for the same panel: at eleven series and this many observations the first
two components carry about a quarter of the variance with no common structure at all, and a
share read against zero would make that quarter look like a finding.

Extraction is extraction. The components are directions of common variation in this panel's
covariance. They are not selected factors, they are not identified as priced, and their labels
are descriptive: nothing here says a component *is* a term-structure factor or a market factor.
Selection required the criterion above, and the criterion is about variance, not about premia.

**A panel this small undercounts.** Eigenvalue-based rules on a few dozen observations understate
the number of directions a longer sample would support, so the count here is a decision about this
panel's usable dimension and never a count of the factors in the market: the number is reported
with the panel it was computed on, and no result may describe it as "the number of factors".
"""

import numpy as np
import pandas as pd

from ..data import loader, panel
from . import spine as spine_module
from .exposures import WINDOW, window_block, windows

# The documented seed. Every permutation draw in the package comes from it, so a rerun of a
# reported count reproduces the count rather than a nearby one.
SEED = 20260912
DRAWS = 200
PERCENTILE = 95
# Fixed before the rule was applied, and reported beside the mechanical count whatever it says.
PREREGISTERED_K = 3
# The falsification fixed before the rule was run: a count that jumps by more than one component
# between adjacent steps, in more than this share of steps, is a rule too unstable to use, and the
# fixed count takes over with both reported.
MOVE_SHARE = 0.25
STABILITY_SEEDS = 5


def mp_edge(n, t):
    """The Marchenko-Pastur upper bulk edge for an uncorrelated panel of n series over t rows."""
    return float((1.0 + np.sqrt(n / t)) ** 2)


def _correlation(window):
    x = np.asarray(window, dtype=float)
    if x.shape[1] < 2:
        raise ValueError("a correlation matrix needs at least two series")
    if np.isnan(x).any():
        # A NaN reaching the decomposition raises from inside numpy with nothing named; refused
        # here in this package's vocabulary instead.
        raise ValueError("the window carries a missing value; a filled one would enter every component built on it")
    spread = x.std(axis=0, ddof=1)
    if (spread == 0).any():
        raise ValueError("a series in the window has no variance; its correlation is undefined")
    return np.corrcoef(x, rowvar=False)


def permutation_null(window, draws=DRAWS, seed=SEED):
    """The matched null: one independent random circular shift per series, per draw.

    Returns the eigenvalues of each draw's correlation matrix, descending, one row per draw.
    The shifts are drawn from the documented seed, so the null is a reproducible object rather
    than a fresh random one each time a report is printed.
    """
    x = np.asarray(window, dtype=float)
    rows, columns = x.shape
    rng = np.random.default_rng(seed)
    out = np.empty((draws, columns))
    for draw in range(draws):
        # Offsets run from one to rows-1: a zero shift would leave a series aligned with itself
        # and the null would keep a sliver of the very correlation it is meant to remove.
        shifted = np.column_stack([np.roll(x[:, j], int(rng.integers(1, rows))) for j in range(columns)])
        out[draw] = np.linalg.eigvalsh(_correlation(shifted))[::-1]
    return out


def threshold(null, percentile=PERCENTILE):
    """The retention threshold per component: the null's percentile for that same component.

    Component-wise rather than one number for the whole matrix: the second component of a null
    panel is not distributed like the first, and a single threshold would either retain noise in
    the low components or discard structure in the high ones.
    """
    return np.percentile(null, percentile, axis=0)


def retained(eigenvalues, null, percentile=PERCENTILE):
    """How many observed components beat the null's threshold for their own position."""
    limits = threshold(null, percentile)
    return int((np.asarray(eigenvalues) > limits).sum())


def orient(loadings):
    """The sign convention: the largest absolute loading of each component is made positive.

    Eigenvectors are defined up to sign, so without this a component can invert between two
    adjacent refits and everything read off it - an exposure, an attribution - flips with it.
    Applied at every refit rather than once, because the refit is where the sign drifts.
    """
    loadings = np.array(loadings, dtype=float, copy=True)
    for column in range(loadings.shape[1]):
        vector = loadings[:, column]
        if vector[np.argmax(np.abs(vector))] < 0:
            loadings[:, column] = -vector
    return loadings


def component_loadings(eigenvalues, vectors, count):
    """The first `count` components as loadings on the correlation scale, signed.

    One place, because a loading is only ever the oriented eigenvector scaled by the square root
    of its eigenvalue, and a second copy of that is a second place for the sign convention to be
    forgotten. The factor covariance and the decomposition both read the components through here,
    so they cannot disagree about what a component is.
    """
    return orient(vectors[:, :count]) * np.sqrt(eigenvalues[:count])


def eigen_structure(window):
    """The window's correlation matrix decomposed, eigenvalues descending.

    The correlation rather than the covariance, because the refusals here are about *counts of
    common variation* and a covariance matrix would let one volatile sleeve decide them. The
    Marchenko-Pastur reference is stated on the same scale, so the comparison is like for like.
    """
    eigenvalues, vectors = np.linalg.eigh(_correlation(window))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], vectors[:, order]


def decompose(window, components=None, draws=DRAWS, seed=SEED):
    """The correlation matrix's eigen-decomposition, the loadings, and the shares.

    `components` fixes the count; omitted, the count comes from the rule applied to this window
    alone, which is what keeps selection inside the trailing window and unable to see the month it
    will be used to trade. The component *series* are not returned here: a standardised score has
    mean zero by construction, and a series with an invented mean is not a return, so anything
    needing the components as returns takes them from `component_portfolios`.
    """
    eigenvalues, vectors = eigen_structure(window)
    null = permutation_null(window, draws=draws, seed=seed)
    count = retained(eigenvalues, null) if components is None else int(components)
    count = max(count, 0)
    vectors = orient(vectors[:, :count])
    loadings = component_loadings(eigenvalues, vectors, count)
    return {
        "eigenvalues": eigenvalues,
        "loadings": loadings,
        "vectors": vectors,
        "components": count,
        "threshold": threshold(null),
        "null": null,
        "variance_share": eigenvalues[:count] / len(eigenvalues),
        "noise_share": np.median(null[:, :count], axis=0) / len(eigenvalues),
    }


def component_portfolios(returns, count=None, draws=DRAWS, seed=SEED):
    """The components as fully invested portfolios of the sleeve excess returns.

    A component is defined only up to scale, and the weights implied by standardised series are
    enormous: a bond sleeve at one percent monthly volatility carries a weight of order a
    hundred. The scale is therefore fixed here so that the weights sum to one, which makes the
    series a portfolio whose net exposure means what a reader expects, and makes the spanning
    condition - a statement about net exposure - readable. The series is built from the raw
    excess returns and never from a standardised or demeaned one: a demeaned series has mean
    zero by construction, which would hand the spanning test an intercept that is an artefact of
    the demeaning rather than a finding. Weights that net to zero are refused rather than divided
    through, because a zero-investment position has no net exposure to state.
    """
    decomposition = decompose(returns, components=count, draws=draws, seed=seed)
    weights = decomposition["vectors"] / np.asarray(returns.std(ddof=1)).reshape(-1, 1)
    totals = weights.sum(axis=0)
    if np.abs(totals).min() < 1e-8:
        raise ValueError(
            "a component's weights net to zero, so it cannot be written as a fully invested portfolio; the "
            "spanning condition is stated for portfolios whose weights sum to one"
        )
    weights = weights / totals
    return pd.DataFrame(
        np.asarray(returns) @ weights,
        index=returns.index,
        columns=[f"component_{position + 1}" for position in range(decomposition["components"])],
    )


def varimax(loadings, iterations=100, tolerance=1e-9):
    """Kaiser's varimax rotation, used once as a labelled presentation.

    Presentation only. The unrotated components are the model, because the rotation redistributes
    variance across components while preserving the total and the reconstructed correlation matrix,
    and the count rule was applied to the unrotated eigenvalues. An oblique rotation is refused for
    the same reason in the other direction: it would make the components correlated, and the
    reconstruction the factor covariance rests on assumes they are not. Row communalities and the
    total variance are preserved exactly, which is what makes the rotated table the same model seen
    differently rather than a second one.
    """
    loadings = np.asarray(loadings, dtype=float)
    rows, columns = loadings.shape
    scale = np.sqrt((loadings ** 2).sum(axis=1, keepdims=True))
    normalised = loadings / scale
    rotation = np.eye(columns)
    criterion = 0.0
    for _ in range(iterations):
        transformed = normalised @ rotation
        gradient = normalised.T @ (
            transformed ** 3 - transformed @ np.diag(np.diag(transformed.T @ transformed)) / rows
        )
        left, singular, right = np.linalg.svd(gradient)
        rotation = left @ right
        if abs(singular.sum() - criterion) < tolerance:
            break
        criterion = float(singular.sum())
    return (normalised @ rotation) * scale


def decide(counts):
    """The counts, the falsification rule applied, and what the rule decided.

    Split out because the rule is arithmetic on a series of counts. Tested only through a full
    walk-forward it would be a rule whose boundary nobody can exercise, and the boundary is exactly
    what the falsification turns on: a move of more than one component, in more than a quarter of
    the steps. A move of exactly one is not a move.
    """
    counts = np.asarray(counts)
    moves = np.abs(np.diff(counts)) > 1
    share = float(moves.mean()) if len(moves) else 0.0
    falsified = share > MOVE_SHARE
    return {
        "mechanical": counts,
        "counts": np.full_like(counts, PREREGISTERED_K) if falsified else counts,
        "falsified": falsified,
        "move_share": share,
    }


def count_series(returns, months, window=WINDOW, draws=DRAWS, seed=SEED):
    """The rule applied in every window, and the falsification fixed before the counts were seen.

    The count is decided inside each window from that window's own permutation null. If it moves by
    more than one component between adjacent steps in more than a quarter of steps, the fixed count
    takes over and both are reported: a rule that unstable would be choosing the answer as much as
    measuring it.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    counts, edges, limits, tops, traded, correlations = [], [], [], [], [], []
    for traded_month, window_months in windows(months, window):
        block = window_block(returns, window_months)
        if len(block) < block.shape[1] + 1:
            raise ValueError(f"the window ending {window_months[-1]} holds {len(block)} returns for its series")
        correlation = _correlation(block)
        eigenvalues = np.sort(np.linalg.eigvalsh(correlation))[::-1]
        null = permutation_null(block, draws=draws, seed=seed)
        counts.append(retained(eigenvalues, null))
        edges.append(mp_edge(block.shape[1], len(block)))
        limits.append(threshold(null)[0])
        tops.append(float(eigenvalues[0]))
        correlations.append(float(np.median(np.abs(correlation[np.triu_indices_from(correlation, 1)]))))
        traded.append(traded_month)

    counts = np.asarray(counts)
    return {
        "traded": pd.PeriodIndex(traded, freq="M"),
        **decide(counts),
        "mp_edge": np.asarray(edges),
        "null_top": np.asarray(limits),
        "observed_top": np.asarray(tops),
        "mean_abs_correlation": np.asarray(correlations),
    }


def stability(returns, months, window=WINDOW, draws=DRAWS):
    """The count from repeated permutation draws on the last window, and whether they agree.

    A count that flips with the seed is not a count. The tolerance is stated rather than assumed:
    every seed must return the same count, and a disagreement is reported rather than averaged.
    """
    traded_month, window_months = list(windows(pd.PeriodIndex(months, freq="M").sort_values(), window))[-1]
    block = window_block(returns, window_months)
    eigenvalues = np.sort(np.linalg.eigvalsh(_correlation(block)))[::-1]
    counts = [
        retained(eigenvalues, permutation_null(block, draws=draws, seed=SEED + offset))
        for offset in range(STABILITY_SEEDS)
    ]
    return {"month": traded_month, "counts": counts, "agrees": len(set(counts)) == 1}


def _component_names(loadings, sleeve_names, count):
    """A descriptive label per component: the sleeves it loads on most heavily.

    Descriptive by construction. The label names where the weight sits, not what the component
    is, because a name would assert an economic identity this panel cannot establish.
    """
    names = []
    for column in range(count):
        vector = loadings[:, column]
        order = np.argsort(-np.abs(vector))[:3]
        names.append(" ".join(f"{sleeve_names[i]}{vector[i]:+.2f}" for i in order))
    return names


def main(root=None):
    document = loader.load_panel(root)
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    named = spine_module.named_set(document["factors"]["eur"], spine_module.constructed_block(returns))

    full = decompose(returns, components=None)
    count = full["components"]
    print(f"[factor] snapshot {document['snapshot_id']}, {returns.shape[0]} returns × {returns.shape[1]} sleeves")
    print(f"[factor] full panel: Marchenko-Pastur edge {mp_edge(returns.shape[1], returns.shape[0]):.4f}, "
          f"permutation null 95th percentile of the top eigenvalue {full['threshold'][0]:.4f}, observed top "
          f"{full['eigenvalues'][0]:.3f}; the matched null sits above the analytic edge, which is what keeping "
          f"the panel's own marginals rather than a normal's does")
    print(f"[factor] the rule retains {count} component(s); the pre-registered fixed count is {PREREGISTERED_K} "
          f"and the eigenvalues run {', '.join(f'{value:.2f}' for value in full['eigenvalues'][:6])}")
    for position in range(count):
        print(f"    component {position + 1}: variance share {full['variance_share'][position]:.2%} against a "
              f"pure-noise share of {full['noise_share'][position]:.2%} at the same N and T")
    noise_two = float(np.median(full['null'][:, :2].sum(axis=1)) / returns.shape[1])
    print(f"[factor] the first two components together: {float(full['eigenvalues'][:2].sum() / returns.shape[1]):.2%} "
          f"of variance against {noise_two:.2%} under pure noise")

    series = count_series(returns, months)
    unique = sorted(set(series["mechanical"].tolist()))
    print(f"[factor] walk-forward, {len(series['mechanical'])} windows: the mechanical count takes "
          f"{', '.join(str(value) for value in unique)}; it moves by more than one component in "
          f"{series['move_share']:.1%} of steps, against the {MOVE_SHARE:.0%} that falsifies the rule")
    if series["falsified"]:
        print(f"[factor] the rule is falsified on this panel: the fixed count {PREREGISTERED_K} is used and the "
              f"mechanical counts are reported beside it")
    print(f"[factor] in-window references: Marchenko-Pastur edge {series['mp_edge'].min():.2f}.."
          f"{series['mp_edge'].max():.2f}, null 95th percentile {series['null_top'].min():.2f}.."
          f"{series['null_top'].max():.2f}, observed top eigenvalue {series['observed_top'].min():.2f}.."
          f"{series['observed_top'].max():.2f}, mean |off-diagonal correlation| "
          f"{series['mean_abs_correlation'].min():.2f}..{series['mean_abs_correlation'].max():.2f}")
    stable = stability(returns, months)
    print(f"[factor] count stability on the last window ({stable['month']}) across {STABILITY_SEEDS} permutation "
          f"seeds: {stable['counts']} - {'all agree' if stable['agrees'] else 'the count moves with the seed'}")

    rotated = varimax(full["loadings"])
    print("[factor] labelled presentation only, varimax-rotated once; the unrotated components are the model:")
    for position, label in enumerate(_component_names(rotated, list(returns.columns), count)):
        print(f"    rotated {position + 1}: {label}")
    print("[factor] these are directions of common variation in this panel's covariance: extraction, not "
          "selection, and no component is claimed to be priced. Selection needed the criterion stated above, "
          "which is about variance.")
    print(f"[factor] a panel this short undercounts the directions a longer sample would support, so {count} is "
          f"this panel's usable dimension over {returns.shape[0]} months, never the number of factors in the market")
    for line in document["warnings"]:
        print(f"[factor] {line}")
    return {"full": full, "series": series, "stability": stable, "returns": returns, "named": named}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
