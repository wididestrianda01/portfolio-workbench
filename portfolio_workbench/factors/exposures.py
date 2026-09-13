"""Rolling exposures, alpha, and the factor/idiosyncratic split of return variance.

Every sleeve's EUR excess return is regressed on the named factor set in a trailing window,
and the window is refitted every month. Four things in here are decisions rather than
mechanics.

The window is stated in **month labels**, not in return observations: it is the sixty months
ending in the month before the traded month, which is all the availability rule allows, since
a bar for month M is readable from the first day of M+1. The calendar the windows are cut on
is the panel's own, so the out-of-sample run keeps the decided length; the panel's first bar
has no return, so the earliest window carries fifty-nine observations where later ones carry
sixty, and the count prints per window rather than being assumed.

Loadings **roll**. The study is about estimation error, and fixed loadings would hide the
quantity under study, so a fixed-loading run is kept only as a contrast whose size is itself
the measurement of loading drift.

The variance split is exact for variance, not for volatility: with an intercept the fitted
and residual sums of squares partition the total exactly, so the two parts add to the
sleeve's own variance to machine precision. Volatility is the square root of each part and
does not add, which is why shares of variance are what the report prints.

**Four sleeves are spanned by the block by construction.** The constructed block is built from
`IBGL.AS`, `IEGE.AS`, `IEAC.AS` and `IHYG.L`, and with level the government block's average
and slope its difference, each of those four is an exact linear combination of the factors:
`IBGL = level + slope/2`, `IEGE = level - slope/2`, `IEAC = level + slope/2 + credit`,
`IHYG = level + slope/2 + credit + high-yield excess`. Their exposures and alphas are
therefore algebraically determined rather than estimated, and the fitted residual is at
machine zero. That is a property of the block's construction rather than an accident of this
sample, so the report names those sleeves and every diagnostic is quoted on the sleeves that
are genuinely estimated. A reader must not take the bond sleeves' exposures as evidence about
bond returns: they are the definition restated.

**The block's level and slope are near-collinear, and the exposures say so.** With level the
government block's average and slope its difference, their correlation tends to one as the
short sleeve's variance shrinks against the long sleeve's, and on this panel the short sleeve
is the quiet one, so their correlation is 0.98 or higher in a sixty-month window. A single
coefficient on either is therefore not identified, while the pair is, and the report prints
the design's conditioning beside the exposures so that an unstable loading is read as a
collinear design rather than as a finding about bonds.
"""

import numpy as np
import pandas as pd

from ..data import loader, panel
from . import spine as spine_module

WINDOW = 60
# The panel's first bar carries no return, so the earliest window is one observation short.
# The floor is derived from the window rather than typed, so the two cannot drift apart.
MIN_OBS = WINDOW - 1
# A sleeve whose factors explain all but a rounding error of its variance in every window is
# spanned by the factor set rather than measured against it.
EXACT_R2 = 1.0 - 1e-9


def windows(months, window=WINDOW):
    """The estimation window behind each traded month, on the panel's own calendar.

    Yields `(traded month, window months)` for every month with a full trailing window. The
    estimate for month t uses months t-window .. t-1 and is applied to month t: an estimate
    formed at the close of t cannot read t's own bar, which is the walk-forward boundary the
    availability rule forces. Cutting the windows on the panel rather than on the return frame is
    what keeps the first traded month a full window after the panel opens, since the return frame
    begins one month later than the panel does; cutting them on the return frame would have moved
    the first traded month a month late and shortened the out-of-sample run by one.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    for position in range(window, len(months)):
        yield months[position], months[position - window : position]


def _require_observations(block, window_months):
    """Refuse a window with a hole in it, in the only place a hole is legitimate.

    Exactly one observation may be absent, it must be the window's first month, and only
    because the panel's first bar has no return to compute. Any other absent month is a gap
    in the panel, and a regression over a window with a gap is a regression on a fabricated
    panel: the missing row would have to be invented, and an invented row enters every result
    built on that window.
    """
    shortfall = len(window_months) - len(block)
    if len(block) < MIN_OBS:
        raise ValueError(
            f"the window {window_months[0]}..{window_months[-1]} holds {len(block)} observations, "
            f"below the {MIN_OBS} a {WINDOW}-month window can legitimately carry"
        )
    if shortfall > 1 or (shortfall == 1 and block.index[0] == window_months[0]):
        missing = window_months.difference(block.index)
        raise ValueError(
            f"the window {window_months[0]}..{window_months[-1]} is missing {list(missing)}; a gap "
            f"inside a window is a data fault, never a short history"
        )


def window_block(frame, window_months):
    """A frame's rows for one window, with the front-only absence rule applied.

    Sliced by label rather than by position so a window can never silently take the rows beside
    the ones it asked for, and shared by every module that cuts a window, so the two frames a
    regression consumes cannot drift apart and a partially missing row cannot reach a covariance
    as a NaN the caller never named.
    """
    block = frame.reindex(window_months).dropna(how="any")
    _require_observations(block, window_months)
    return block


def regress(returns, factors):
    """OLS of every sleeve on the factors, with an intercept, in one window.

    One design matrix for all sleeves at once: the exposures, their standard errors and the
    residual covariance of a window are properties of that window, and fitting each sleeve in
    its own call would let one of them be fitted on a different one.

    Two residual scalings are returned because they answer different questions and one cannot
    be recovered from the other afterwards: `resid_var` divides by the residual degrees of
    freedom so that the fitted and residual parts sum to the sleeve's sample variance, while
    `sigma2` divides by the observation count less the parameters, which is what the alpha
    standard error needs. Alphas here are **monthly** excess returns.
    """
    y = np.asarray(returns, dtype=float)
    x = np.asarray(factors, dtype=float)
    if np.isnan(y).any() or np.isnan(x).any():
        raise ValueError("the window carries a missing value; a filled one would enter every result built on it")

    design = np.column_stack([np.ones(len(x)), x])
    gram = design.T @ design
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        raise ValueError(
            f"a design of rank {np.linalg.matrix_rank(gram)} cannot identify {gram.shape[0]} "
            f"parameters: the factors are collinear in this window"
        )
    dof = len(design) - design.shape[1]
    if dof <= 0:
        raise ValueError(f"{len(design)} observations cannot estimate {design.shape[1]} parameters")

    gram_inv = np.linalg.inv(gram)
    coef = gram_inv @ design.T @ y
    fitted = design @ coef
    resid = y - fitted

    factor_cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    beta = coef[1:]
    explained_var = np.einsum("kn,kl,ln->n", beta, factor_cov, beta)
    resid_var = resid.var(axis=0, ddof=1)
    sigma2 = (resid ** 2).sum(axis=0) / dof
    alpha = coef[0]
    return {
        "alpha": alpha,
        "t_alpha": alpha / np.sqrt(sigma2 * gram_inv[0, 0]),
        "beta": beta,
        "sigma2": sigma2,
        "resid_var": resid_var,
        "explained_var": explained_var,
        "total_var": explained_var + resid_var,
        "r2": explained_var / (explained_var + resid_var),
        "factor_cov": factor_cov,
        "factor_mean": x.mean(axis=0),
        "n_obs": len(design),
        "dof": dof,
    }


def rolling(returns, factors, months, window=WINDOW):
    """The same regression in every window, as one frame per quantity.

    Frames are indexed by the traded month and columned by sleeve, so a series can be read
    across the out-of-sample calendar without re-running anything.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    betas = {column: [] for column in factors.columns}
    alpha, t_alpha, r2, resid, explained = [], [], [], [], []
    traded, firsts, lasts, counts = [], [], [], []
    for traded_month, window_months in windows(months, window):
        block = window_block(returns, window_months)
        factors_block = window_block(factors, window_months)
        if not block.index.equals(factors_block.index):
            raise ValueError(
                f"the sleeve frame and the factor frame cover different months in the window ending "
                f"{window_months[-1]}; a regression across two calendars compares the wrong rows"
            )
        fit = regress(block, factors_block)
        traded.append(traded_month)
        firsts.append(window_months[0])
        lasts.append(window_months[-1])
        counts.append(fit["n_obs"])
        alpha.append(fit["alpha"])
        t_alpha.append(fit["t_alpha"])
        r2.append(fit["r2"])
        resid.append(fit["resid_var"])
        explained.append(fit["explained_var"])
        for position, column in enumerate(factors.columns):
            betas[column].append(fit["beta"][position])

    index = pd.PeriodIndex(traded, freq="M")
    frames = lambda values: pd.DataFrame(values, index=index, columns=returns.columns)
    return {
        "traded": index,
        "window_first": pd.PeriodIndex(firsts, freq="M"),
        "window_last": pd.PeriodIndex(lasts, freq="M"),
        "n_obs": np.asarray(counts),
        "alpha": frames(alpha),
        "t_alpha": frames(t_alpha),
        "r2": frames(r2),
        "resid_var": frames(resid),
        "explained_var": frames(explained),
        "beta": {column: frames(values) for column, values in betas.items()},
    }


def conditioning(factors, months, window=WINDOW):
    """The factor design's conditioning per window, and the level/slope correlation.

    Reported beside the exposures because a coefficient read on its own is only a measurement
    where the design identifies it. Here the block's level and slope are the pair that does
    not: they are the government block's average and difference, so their correlation tends to
    one when one sleeve is much the quieter, which on this panel it is.
    """
    columns = list(factors.columns)
    pair = ("government_level", "term_slope")
    conditions, correlations = [], []
    for _, window_months in windows(months, window):
        x = np.asarray(window_block(factors, window_months), dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        conditions.append(float(np.linalg.cond(design)))
        if set(pair) <= set(columns):
            correlation = np.corrcoef(x, rowvar=False)
            correlations.append(float(correlation[columns.index(pair[0]), columns.index(pair[1])]))
    return np.asarray(conditions), np.asarray(correlations)


def spanned_sleeves(fit, tolerance=EXACT_R2):
    """The sleeves the factor set reproduces exactly, and so does not estimate.

    Named rather than annotated: a caller that wants the sleeves carrying information filters
    on this, and a reader comparing two runs is told which rows cannot move.
    """
    return [sleeve for sleeve in fit["r2"].columns if bool((fit["r2"][sleeve] > tolerance).all())]


def fixed_alpha(returns, factors, betas, months, window=WINDOW):
    """The contrast run: exposures held at the first window's, the intercept re-estimated.

    Holding the loadings fixed leaves the alpha as the only free parameter, so the gap between
    this and the rolling run is the part of the alpha that moving exposures accounts for. It is
    a diagnostic and never the reported estimate: the fixed run is the one that cannot see the
    quantity the study is about.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    columns = list(factors.columns)
    # One row per factor, one column per sleeve: the orientation the rolling betas are stored
    # in, so the two runs cannot be compared along the wrong axis.
    fixed = np.array([betas[column].iloc[0].to_numpy() for column in columns])
    out, traded = [], []
    for traded_month, window_months in windows(months, window):
        block = window_block(returns, window_months)
        y = np.asarray(block, dtype=float)
        x = np.asarray(window_block(factors, window_months), dtype=float)
        out.append(y.mean(axis=0) - fixed.T @ x.mean(axis=0))
        traded.append(traded_month)
    return pd.DataFrame(out, index=pd.PeriodIndex(traded, freq="M"), columns=returns.columns)


def loading_drift(betas, sleeves=None):
    """How far the loadings move between adjacent refits, per factor.

    Reported because it is the quantity the fixed-loading run suppresses: a factor whose
    exposure swings month to month is one the sample cannot pin down, whatever its average.
    """
    out = {}
    for column, frame in betas.items():
        step = (frame[sleeves] if sleeves is not None else frame).diff().abs()
        out[column] = {"mean": float(step.mean().mean()), "max": float(step.max().max())}
    return out


def factor_split(weights, beta, factor_cov, resid_var):
    """Portfolio return variance as `(factor part, idiosyncratic part)`.

    `weights` are active weights when the caller wants tracking error: the same two numbers are
    then the factor and idiosyncratic components of tracking error against the benchmark, which
    is the one definition the risk attribution and the budget both read. The idiosyncratic
    covariance is taken as diagonal, which is the residual structure the regression leaves; a
    non-diagonal one would be a second model and is not built here.
    """
    weights = np.asarray(weights, dtype=float)
    exposure = np.asarray(beta, dtype=float) @ weights
    factor_part = float(exposure @ np.asarray(factor_cov, dtype=float) @ exposure)
    idio_part = float((weights ** 2) @ np.asarray(resid_var, dtype=float))
    return factor_part, idio_part


def _run(returns, factors, months):
    fit = rolling(returns, factors, months)
    estimated = [sleeve for sleeve in fit["r2"].columns if sleeve not in spanned_sleeves(fit)]
    return fit, estimated, fixed_alpha(returns, factors, fit["beta"], months)


def main(root=None):
    document = loader.load_panel(root)
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document["factors"]["eur"], block)
    quoted = spine_module.named_set(document["factors"]["usd"], block)
    europe = spine_module.named_set(spine_module.cross_check(document), block)

    fit, estimated, fixed = _run(returns, named, months)
    traded = fit["traded"]
    print(f"[factor] snapshot {document['snapshot_id']}, {len(named.columns)} factors, {len(traded)} refits "
          f"{traded.min()}..{traded.max()} on the panel calendar {months.min()}..{months.max()}")
    print(f"[factor] window {WINDOW} month labels ending the month before the traded month; the first "
          f"window {fit['window_first'][0]}..{fit['window_last'][0]} carries {fit['n_obs'][0]} observations "
          f"(the panel's first bar has no return), later windows {fit['n_obs'].max()}")

    spanned = spanned_sleeves(fit)
    print(f"[factor] spanned by construction, so their exposures are the block's definition restated "
          f"rather than estimates: {', '.join(spanned)}")
    print("[factor] per sleeve, averaged over the refits: alpha, its t-statistic, the world exposure, and "
          "the share of variance the factors explain")
    for sleeve in returns.columns:
        if sleeve in spanned:
            print(f"    {sleeve:<16s} spanned by the block by construction: its exposure is the definition "
                  f"restated, and an alpha of zero over a machine-zero residual is not a test of anything")
            continue
        print(f"    {sleeve:<16s} alpha {fit['alpha'][sleeve].mean():+.4%}/month t "
              f"{fit['t_alpha'][sleeve].mean():+.2f}  beta(world) {fit['beta']['Mkt-RF'][sleeve].mean():+.2f}  "
              f"factors explain {fit['r2'][sleeve].mean():.1%} of variance")

    last = fit["traded"][-1]
    print(f"[factor] the frames are per window, not only the averages above; the newest refit trades "
          f"{last} on the window {fit['window_first'][-1]}..{fit['window_last'][-1]} "
          f"({fit['n_obs'][-1]} observations):")
    for sleeve in returns.columns:
        if sleeve in spanned:
            continue
        explained = float(fit["explained_var"][sleeve].iloc[-1])
        residual = float(fit["resid_var"][sleeve].iloc[-1])
        share = explained / (explained + residual) if (explained + residual) else float("nan")
        print(f"    {sleeve:<16s} alpha {fit['alpha'][sleeve].iloc[-1]:+.4%}/month "
              f"t {fit['t_alpha'][sleeve].iloc[-1]:+.2f}  beta(world) "
              f"{fit['beta']['Mkt-RF'][sleeve].iloc[-1]:+.2f}  split {share:.1%} factor / "
              f"{1.0 - share:.1%} idiosyncratic")

    drift = loading_drift(fit["beta"], estimated)
    worst = max(drift, key=lambda column: drift[column]["max"])
    conditions, correlations = conditioning(named, months)
    print(f"[factor] the design is collinear: the block's level and slope correlate "
          f"{correlations.min():.3f}..{correlations.max():.3f} across the windows, so a coefficient on "
          f"either is not identified on its own, and the design conditioning runs {conditions.min():.0f}.."
          f"{conditions.max():,.0f}")
    print(f"[factor] on the {len(estimated)} estimated sleeves, loadings roll: mean |step| by factor "
          f"{min(v['mean'] for v in drift.values()):.3f}..{max(v['mean'] for v in drift.values()):.3f}, the "
          f"largest single-month move {drift[worst]['max']:.2f} on {worst}, which is a collinear design "
          f"rather than a bond finding")
    difference = (fit["alpha"][estimated] - fixed[estimated]).abs()
    print(f"[factor] contrast, loadings held at the first window's: mean |alpha difference| "
          f"{difference.mean().mean():.4%}/month, worst {difference.max().max():.4%} on {difference.max().idxmax()} "
          f"- the part of the alpha that moving exposures accounts for")

    quoted_fit, _, _ = _run(returns, quoted, months)
    gap = (fit["alpha"][estimated] - quoted_fit["alpha"][estimated]).abs()
    identified = [column for column in fit["beta"] if column not in ("government_level", "term_slope")]
    exposure_gap = max(
        float((fit["beta"][column][estimated] - quoted_fit["beta"][column][estimated]).abs().max().max())
        for column in identified
    )
    print(f"[factor] the same run on the spine's quoted currency moves the alpha by {gap.mean().mean():.4%}/month "
          f"on average and the identified exposures by at most {exposure_gap:.2f} (the collinear level and slope "
          f"pair is excluded, since a translation cannot move a coefficient the design does not identify)")
    europe_fit, _, _ = _run(returns, europe, months)
    print(f"[factor] cross-check on the Europe cut, five factors rather than six and translated through the same "
          f"euro leg: mean |alpha| {europe_fit['alpha'][estimated].abs().mean().mean():.4%}/month against "
          f"{fit['alpha'][estimated].abs().mean().mean():.4%} on the Developed spine, both in euro - a different regional "
          f"cut carried beside the headline, never merged into it")
    for line in document["warnings"]:
        print(f"[factor] {line}")
    return {"eur": fit, "quoted": quoted_fit, "europe": europe_fit, "returns": returns, "named": named,
            "estimated": estimated, "spanned": spanned}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
