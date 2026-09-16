"""Rolling exposures, alpha, and the factor/idiosyncratic split of return variance.

Every sleeve's EUR excess return is regressed on the named factor set in a trailing window, and the
window is refitted every month. Five things in here are decisions rather than mechanics, and the
first two exist because the loadings are not identified without them.

**The window is stated in month labels, not in return observations.** It is the sixty months ending
in the month before the traded month, which is all the availability rule allows, since a bar for
month M is readable from the first day of M+1. The calendar the windows are cut on is the panel's
own, so the out-of-sample run keeps its decided length; the panel's first bar has no return, so the
earliest window carries fifty-nine observations where later ones carry sixty, and the count prints
per window rather than being assumed.

**The block is orthogonalised against itself inside each window.** Its four declared series are built
from four sleeves that move together: the level and the slope correlate 0.997 or higher in every
window on this panel, because the short government sleeve is far the quieter of the two, and the
credit spread sits at -0.96 against the level. A coefficient on either half of a pair like that is
not identified: the worst loading's variance is inflated by a factor of hundreds to thousands, where a
design whose regressors were unrelated would give one. Projecting the block onto itself in its declared order
leaves the level exactly as declared and gives the later series the cleaner reading - the slope net
of the level, the credit spread net of the term structure, high yield net of credit - with the
design's conditioning back at the spine's own. Nothing is centred, because a monthly factor return's
mean belongs to the factor model: centring would move the intercept and turn the reported alpha from
Jensen's into the sleeve's own average return. The transform is derived from the window alone, since
everything a window uses is inside it, and the fitted values, alphas, residual variances and shares
of variance are **identical** under either parameterisation - only the loadings and their standard
errors change.

**Four sleeves are the block, so the block is dropped from their design.** `IBGL.AS`, `IEGE.AS`,
`IEAC.AS` and `IHYG.L` are exact linear combinations of the block - `IBGL = level + slope/2`,
`IEGE = level - slope/2`, `IEAC = level + slope/2 + credit`, and `IHYG` adds the high-yield excess -
because the block is built from them. Regressing such a sleeve on a set containing its own
construction fits it exactly, reports every other loading as zero, and destroys the one exposure that
*is* estimable: its sensitivity to the published spine. Those sleeves are therefore regressed on the
spine alone, with the block part reported as the identity it is. Their full-model alpha is zero by
arithmetic rather than by evidence, so no alpha is reported for them, and their mean excess return
decomposes exactly into its spine part and its block part.

**Loadings roll, and a fixed-loading run is kept as the contrast.** The study is about estimation
error, and fixed loadings would hide the quantity under study; the size of the difference between the
two runs is itself the measurement of loading drift.

**The variance split is exact for variance, not for volatility.** With an intercept the fitted and
residual sums of squares partition the total exactly, so the two parts add to the sleeve's own
variance to machine precision. Volatility is the square root of each part and does not add, which is
why shares of variance are what the report prints.
"""

import numpy as np
import pandas as pd

from ..data import loader, panel
from . import spine as spine_module

WINDOW = 60
# The panel's first bar carries no return, so the earliest window is one observation short. The floor
# is derived from the window rather than typed, so the two cannot drift apart.
MIN_OBS = WINDOW - 1
# The block's declared series, in the order the orthogonalisation walks and the report names them.
BLOCK = tuple(spine_module.CONSTRUCTED)
# A sleeve the block reproduces this closely is the block's own construction rather than a sleeve the
# block explains. On this panel the four construction sleeves sit at 1e-31 and every other sleeve at
# 0.2 or above, so the line is not a judgement call.
SPAN_TOLERANCE = 1e-9
# The quantities every window's fit returns, in the order they are accumulated per sleeve.
FIT_QUANTITIES = ("alpha", "t_alpha", "r2", "resid_var", "explained_var")


def windows(months, window=WINDOW):
    """The estimation window behind each traded month, on the panel's own calendar.

    Yields `(traded month, window months)` for every month with a full trailing window. The estimate
    for month t uses months t-window .. t-1 and is applied to month t: an estimate formed at the close
    of t cannot read t's own bar, which is the walk-forward boundary the availability rule forces.
    Cutting the windows on the panel rather than on the return frame is what keeps the first traded
    month a full window after the panel opens, since the return frame begins one month later than the
    panel does; cutting them on the return frame would have moved the first traded month a month late
    and shortened the out-of-sample run by one.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    for position in range(window, len(months)):
        yield months[position], months[position - window : position]


def expanding_windows(months, window=WINDOW):
    """The same traded months, with every month since the panel opened in the estimate.

    The secondary protocol: the estimate for month t uses the months from the panel's first month
    through t-1, so the window grows rather than rolling, and the traded months are exactly the ones
    the rolling rule trades. The boundary is unchanged - an estimate formed at the close of t-1 still
    cannot read t's own bar - so a difference between the two protocols is a difference in window
    length rather than in the out-of-sample span.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    for position in range(window, len(months)):
        yield months[position], months[:position]


def _require_observations(block, window_months):
    """Refuse a window with a hole in it, in the only place a hole is legitimate.

    Exactly one observation may be absent, it must be the window's first month, and only because the
    panel's first bar has no return to compute. Any other absent month is a gap in the panel, and a
    regression over a window with a gap is a regression on a fabricated panel: the missing row would
    have to be invented, and an invented row enters every result built on that window.
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

    Sliced by label rather than by position so a window can never silently take the rows beside the
    ones it asked for, and shared by every module that cuts a window, so the two frames a regression
    consumes cannot drift apart and a partially missing row cannot reach a covariance as a NaN the
    caller never named.
    """
    block = frame.reindex(window_months).dropna(how="any")
    _require_observations(block, window_months)
    return block


def orthogonal_transform(block):
    """The window's own map from the block's declared series to the ones net of their predecessors.

    A linear map on the block's columns, returned as a matrix rather than applied and discarded,
    because the factor attribution has to put a month **outside** the window onto the basis the
    window's coefficients were fitted in: the coefficients belong to the window, the month they are
    applied to does not, and this map is the only object that carries the basis across that boundary.

    Each series is projected onto the predecessors already transformed, so the map is accumulated in
    the order the report names the block in and is unit upper-triangular over the declared series. The
    series are not centred, so the map carries no intercept and a month travels through it as a row.
    """
    x = np.asarray(block, dtype=float)
    gram = x.T @ x
    size = x.shape[1]
    transform = np.zeros((size, size))
    for position in range(size):
        basis = np.zeros(size)
        basis[position] = 1.0
        for earlier in range(position):
            vector = transform[:, earlier]
            coefficient = float(vector @ gram @ basis) / float(vector @ gram @ vector)
            basis = basis - coefficient * vector
        transform[:, position] = basis
    return transform


def orthogonalise(block):
    """The block re-expressed against itself, in its declared order, inside one window.

    Each series is projected onto the ones declared before it, so the level is left exactly as it was
    declared and every later series is carried net of its predecessors. The series are not centred: a
    monthly factor return's mean belongs to the factor model, and centring would move the intercept,
    which would quietly turn the reported alpha from Jensen's into the sleeve's average return.
    """
    values = np.asarray(block, dtype=float) @ orthogonal_transform(block)
    return pd.DataFrame(values, index=block.index, columns=list(block.columns))


def within_span(block, sleeve, tolerance=SPAN_TOLERANCE):
    """How far a sleeve is from being an exact combination of the block, and that combination.

    A sleeve the block reproduces to machine precision is not a sleeve the block explains: its
    loading is fixed by the construction, a regression including the block fits it exactly, and every
    other loading comes back as zero. The share is returned rather than a bare verdict so a caller can
    see the margin, and the loadings are the identity to report in place of an estimate.
    """
    y = np.asarray(sleeve, dtype=float)
    x = np.asarray(block, dtype=float)
    loadings = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ loadings
    spread = float(np.var(y, ddof=1))
    return loadings, (float(np.var(residual, ddof=1)) / spread if spread else float("inf"))


def regress(returns, factors):
    """OLS of every sleeve in a group on the factors, with an intercept, in one window.

    One design matrix per group: the exposures, their standard errors and the residual covariance of
    a window are properties of that window, and fitting each sleeve in its own call would let one of
    them be fitted on a different one. The two groups are the sleeves the block explains and the
    sleeves the block *is*, and each carries the same spine columns, which is what makes them
    comparable.

    Two residual scalings are returned because they answer different questions and one cannot be
    recovered from the other afterwards: `resid_var` divides by the residual degrees of freedom so
    that the fitted and residual parts sum to the sleeve's sample variance, while the standard errors
    need the observation count less the parameters. Alphas are **monthly** excess returns, measured
    against the factors as declared.
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
    resid = y - design @ coef

    factor_cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    beta = coef[1:]
    explained_var = np.einsum("kn,kl,ln->n", beta, factor_cov, beta)
    resid_var = resid.var(axis=0, ddof=1)
    sigma2 = (resid ** 2).sum(axis=0) / dof
    alpha = coef[0]
    return {
        "alpha": alpha,
        # One standard error per coefficient from the same inverse the coefficients came from, so a
        # loading the design cannot pin down arrives with the number that says so.
        "t_alpha": alpha / np.sqrt(sigma2 * gram_inv[0, 0]),
        "beta": beta,
        "t_beta": beta / np.sqrt(np.outer(np.diag(gram_inv)[1:], sigma2)),
        "resid_var": resid_var,
        "explained_var": explained_var,
        "r2": explained_var / (explained_var + resid_var),
        "factor_cov": factor_cov,
        "n_obs": len(design),
        "dof": dof,
    }


def _design(factors, block, window_months, sleeves):
    """One window's two designs, the block's orthogonalised form, and the sleeves the block is.

    Two designs rather than one, because a sleeve the block reproduces exactly cannot be regressed on
    it: the block is dropped for those sleeves and kept, orthogonalised, for the rest. Both carry the
    same spine columns, which is what makes the two groups comparable.
    """
    spine_columns = [column for column in factors.columns if column not in block]
    spine = window_block(factors.loc[:, spine_columns], window_months)
    declared = window_block(factors.loc[:, list(block)], window_months)
    _aligned(sleeves, spine, declared, window_months=window_months)
    determined = {}
    for sleeve in sleeves:
        loadings, share = within_span(declared, sleeves[sleeve])
        if share <= SPAN_TOLERANCE:
            determined[sleeve] = loadings
    return spine, orthogonalise(declared), determined


def _aligned(sleeves, *frames, window_months):
    """Every frame a window's fit consumes has to cover the same months, or nothing is fitted.

    The two frames arrive from different legs, and a factor frame that reaches further back than the
    sleeve frame would let a regression take its response from one calendar and its regressors from
    another. numpy would raise on the shapes or, worse, not raise at all.
    """
    for frame in frames:
        if not frame.index.equals(sleeves.index):
            raise ValueError(
                f"the sleeve frame and the factor frame cover different months in the window ending "
                f"{window_months[-1]}; a regression across two calendars compares the wrong rows"
            )


def _pivot(values, index, columns):
    """One long series of `(month, sleeve, value)` triples as a month-by-sleeve frame."""
    series = pd.Series(values)
    series.index = pd.MultiIndex.from_tuples(series.index, names=["traded", "sleeve"])
    return series.unstack("sleeve").reindex(index=index, columns=columns)


def rolling(returns, factors, months, block=BLOCK, window=WINDOW):
    """Every sleeve's regression in every window, on the design that sleeve admits.

    One frame per quantity, indexed by the traded month and columned by sleeve, plus a record of which
    design each sleeve used: a consumer of these results has to be able to tell an estimate from an
    identity, and the flag is carried rather than left to be inferred from a suspiciously round
    number. `beta` and `t_beta` are keyed by factor and hold the loadings a caller can use for
    arithmetic downstream; the block's entries for a determined sleeve are its identity, with no
    standard error beside them because there is nothing to test.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    columns = list(factors.columns)

    scalar = {name: {} for name in (*FIT_QUANTITIES, "spine_r2")}
    beta = {column: {} for column in columns}
    t_beta = {column: {} for column in columns}
    traded, firsts, lasts, counts, factor_means, estimated, determined, identity = [], [], [], [], [], [], [], []

    for traded_month, window_months in windows(months, window):
        sleeves = window_block(returns, window_months)
        spine, orthogonal, fixed = _design(factors, block, window_months, sleeves)
        design = pd.concat([spine, orthogonal], axis=1)
        groups = (
            ([sleeve for sleeve in returns.columns if sleeve not in fixed], design),
            ([sleeve for sleeve in returns.columns if sleeve in fixed], spine),
        )
        for group, matrix in groups:
            if not group:
                continue
            fit = regress(sleeves[group], matrix)
            for position, sleeve in enumerate(group):
                for name in FIT_QUANTITIES:
                    scalar[name][(traded_month, sleeve)] = fit[name][position]
                for place, column in enumerate(matrix.columns):
                    beta[column][(traded_month, sleeve)] = fit["beta"][place][position]
                    t_beta[column][(traded_month, sleeve)] = fit["t_beta"][place][position]

        # The block is these sleeves' own construction, so their model is exact by arithmetic: no
        # alpha to test, no residual, and the block part of their loadings is an identity. What the
        # spine alone explains is kept beside it, since that is the part that is estimated.
        for sleeve, loadings in fixed.items():
            scalar["spine_r2"][(traded_month, sleeve)] = scalar["r2"][(traded_month, sleeve)]
            scalar["alpha"][(traded_month, sleeve)] = float("nan")
            scalar["t_alpha"][(traded_month, sleeve)] = float("nan")
            scalar["r2"][(traded_month, sleeve)] = 1.0
            scalar["resid_var"][(traded_month, sleeve)] = 0.0
            scalar["explained_var"][(traded_month, sleeve)] = float(sleeves[sleeve].var(ddof=1))
            for place, column in enumerate(block):
                beta[column][(traded_month, sleeve)] = loadings[place]
                t_beta[column][(traded_month, sleeve)] = float("nan")

        traded.append(traded_month)
        firsts.append(window_months[0])
        lasts.append(window_months[-1])
        counts.append(len(sleeves))
        factor_means.append(design.mean().rename(traded_month))
        estimated.append([sleeve for sleeve in returns.columns if sleeve not in fixed])
        determined.append(sorted(fixed))
        identity.append(fixed)

    index = pd.PeriodIndex(traded, freq="M")
    return {
        "traded": index,
        "window_first": pd.PeriodIndex(firsts, freq="M"),
        "window_last": pd.PeriodIndex(lasts, freq="M"),
        "n_obs": np.asarray(counts),
        **{name: _pivot(values, index, returns.columns) for name, values in scalar.items()},
        "beta": {column: _pivot(values, index, returns.columns) for column, values in beta.items()},
        "t_beta": {column: _pivot(values, index, returns.columns) for column, values in t_beta.items()},
        "factor_mean": pd.DataFrame(factor_means),
        "estimated": estimated,
        "determined": determined,
        "identity": identity,
    }


def conditioning(factors, months, block=BLOCK, window=WINDOW):
    """How badly a loading is pinched, per window, declared against orthogonalised.

    Measured as the worst variance inflation factor in the design - the factor by which a coefficient's
    variance is multiplied relative to a design whose regressors are unrelated. A condition number was
    the wrong instrument twice over: it takes a view on the columns' scales as well as their
    directions, and a large one does not by itself say which coefficient suffers. Inflations of
    hundreds mean a coefficient is not identified; inflations of tens are ordinary factor-model
    correlation and are estimable.
    """
    spine_columns = [column for column in factors.columns if column not in block]
    spine_only, declared_inflation, orthogonal_inflation, correlations = [], [], [], []
    for _, window_months in windows(months, window):
        spine = window_block(factors.loc[:, spine_columns], window_months)
        declared = window_block(factors.loc[:, list(block)], window_months)
        spine_only.append(variance_inflation(spine))
        declared_inflation.append(variance_inflation(pd.concat([spine, declared], axis=1)))
        orthogonal_inflation.append(variance_inflation(pd.concat([spine, orthogonalise(declared)], axis=1)))
        correlations.append(float(np.corrcoef(declared[block[0]], declared[block[1]])[0, 1]))
    return {
        "spine_only": np.asarray(spine_only),
        "declared": np.asarray(declared_inflation),
        "orthogonal": np.asarray(orthogonal_inflation),
        "correlations": np.asarray(correlations),
    }


def variance_inflation(design):
    """The worst variance inflation factor in a design: `[corr(X)^-1]_jj`, largest over its columns.

    One for a design whose regressors are uncorrelated, and the multiple by which a coefficient's
    variance exceeds what that design would give. It is the scale-free statement of whether a loading
    is identified, which a condition number is not: the scale of a regressor cannot flatter it, and it
    names the size of the penalty rather than an abstract ratio.
    """
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return 1.0
    return float(np.diag(np.linalg.inv(np.corrcoef(x, rowvar=False))).max())


def fixed_alpha(returns, factors, betas, months, block=BLOCK, window=WINDOW):
    """The contrast run: exposures held at the first window's, the intercept re-estimated.

    Holding the loadings fixed leaves the alpha as the only free parameter, so the gap between this
    and the rolling run is the part of the alpha that moving exposures accounts for. It is a
    diagnostic and never the reported estimate: the fixed run is the one that cannot see the quantity
    the study is about. Each sleeve keeps the design it had, so the contrast compares like with like.
    """
    months = pd.PeriodIndex(months, freq="M").sort_values()
    first_month = next(iter(windows(months, window)))[0]
    out = {}
    for traded_month, window_months in windows(months, window):
        sleeves = window_block(returns, window_months)
        spine, orthogonal, fixed = _design(factors, block, window_months, sleeves)
        design = pd.concat([spine, orthogonal], axis=1)
        for sleeve in returns.columns:
            matrix = spine if sleeve in fixed else design
            anchor = np.asarray([float(betas[column].loc[first_month, sleeve]) for column in matrix.columns])
            out[(traded_month, sleeve)] = float(sleeves[sleeve].mean() - anchor @ matrix.mean().to_numpy())
    index = pd.PeriodIndex([month for month, _ in windows(months, window)], freq="M")
    return _pivot(out, index, returns.columns)


def loading_drift(betas, sleeves=None):
    """How far the loadings move between adjacent refits, per factor.

    Reported because it is the quantity the fixed-loading run suppresses: a factor whose exposure
    swings month to month is one the sample cannot pin down, whatever its average.
    """
    out = {}
    for column, frame in betas.items():
        step = (frame[sleeves] if sleeves is not None else frame).diff().abs()
        out[column] = {"mean": float(step.mean().mean()), "max": float(step.max().max())}
    return out


def factor_split(weights, beta, factor_cov, resid_var):
    """Portfolio return variance as `(factor part, idiosyncratic part)`.

    `weights` are active weights when the caller wants tracking error: the same two numbers are then
    the factor and idiosyncratic components of tracking error against the benchmark, which is the one
    definition the risk attribution and the budget both read. The idiosyncratic covariance is taken as
    diagonal, which is the residual structure the regression leaves; a non-diagonal one would be a
    second model and is not built here.
    """
    weights = np.asarray(weights, dtype=float)
    exposure = np.asarray(beta, dtype=float) @ weights
    factor_part = float(exposure @ np.asarray(factor_cov, dtype=float) @ exposure)
    idio_part = float((weights ** 2) @ np.asarray(resid_var, dtype=float))
    return factor_part, idio_part


def _identity_line(sleeve, loadings, block):
    terms = [f"{value:+.2f} {column}" for column, value in zip(block, loadings) if abs(value) > 1e-12]
    return f"{sleeve} = {' '.join(terms)}"


def main(root=None):
    document = loader.load_panel(root)
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document["factors"]["eur"], block)
    quoted = spine_module.named_set(document["factors"]["usd"], block)
    europe = spine_module.named_set(spine_module.cross_check(document), block)

    fit = rolling(returns, named, months)
    determined = fit["determined"][-1]
    estimated = fit["estimated"][-1]
    traded = fit["traded"]
    print(f"[factor] snapshot {document['snapshot_id']}, {len(named.columns)} factors, {len(traded)} refits "
          f"{traded.min()}..{traded.max()} on the panel calendar {months.min()}..{months.max()}")
    print(f"[factor] window {WINDOW} month labels ending the month before the traded month; the first window "
          f"{fit['window_first'][0]}..{fit['window_last'][0]} carries {fit['n_obs'][0]} observations (the "
          f"panel's first bar has no return), later windows {fit['n_obs'].max()}")

    report = conditioning(named, months)
    print(f"[factor] the block is orthogonalised against itself inside each window: as declared the level and "
          f"the slope correlate {report['correlations'].min():.3f}..{report['correlations'].max():.3f}, and the "
          f"worst loading's variance is inflated x{report['declared'].min():,.0f}..x{report['declared'].max():,.0f} "
          f"by the declared block - a coefficient on the level or the slope is not identified there - against "
          f"x{report['orthogonal'].min():,.0f}..x{report['orthogonal'].max():,.0f} once it is orthogonalised, "
          f"with x{report['spine_only'].min():,.0f}..x{report['spine_only'].max():,.0f} for the spine alone. What "
          f"is left is the published factors' own correlation, which is estimable; fitted values, alphas and "
          f"variance shares are identical either way, and only the loadings and their standard errors change")
    block_volatility = block.std(ddof=1)
    print(f"[factor] orthogonalising leaves the first series alone and makes the later ones quieter, so a block "
          f"loading is per unit of that series: declared volatilities "
          f"{', '.join(f'{column} {block_volatility[column]:.2%}' for column in block)}/month, and the "
          f"net-of-predecessors series carry less, which is why their loadings are larger and their t-statistics, "
          f"not their magnitudes, are the numbers to read")
    print("[factor] the block is these sleeves' own construction, so it is dropped from their design and its "
          "part of their loadings is an identity rather than an estimate, with no alpha to test:")
    for sleeve, loadings in fit["identity"][-1].items():
        print(f"    {_identity_line(sleeve, loadings, list(block))}")

    print("[factor] per sleeve, averaged over the refits: alpha and its t-statistic, the exposure to the world "
          "factor, and the share of variance the spine explains (a constructed sleeve) or the factors explain")
    for sleeve in returns.columns:
        if sleeve in determined:
            print(f"    {sleeve:<16s} no alpha: the model is exact          beta(world) "
                  f"{fit['beta']['Mkt-RF'][sleeve].mean():+.2f}  the spine alone explains "
                  f"{fit['spine_r2'][sleeve].mean():.1%} of its variance")
            continue
        print(f"    {sleeve:<16s} alpha {fit['alpha'][sleeve].mean():+.4%}/month t "
              f"{fit['t_alpha'][sleeve].mean():+.2f}  beta(world) {fit['beta']['Mkt-RF'][sleeve].mean():+.2f}  "
              f"factors explain {fit['r2'][sleeve].mean():.1%} of variance")

    print("[factor] the estimated sleeves' block loadings, net of the ones declared before them, averaged over "
          "the refits with the mean |t| beside each:")
    for sleeve in estimated:
        parts = "  ".join(
            f"{column} {fit['beta'][column][sleeve].mean():+.2f} (|t| "
            f"{fit['t_beta'][column][sleeve].abs().mean():.1f})"
            for column in block
        )
        print(f"    {sleeve:<16s} {parts}")

    last = traded[-1]
    print(f"[factor] the frames are per window, not only the averages above; the newest refit trades {last} on "
          f"the window {fit['window_first'][-1]}..{fit['window_last'][-1]} ({fit['n_obs'][-1]} observations):")
    for sleeve in returns.columns:
        if sleeve in determined:
            print(f"    {sleeve:<16s} no alpha: the model is exact          beta(world) "
                  f"{fit['beta']['Mkt-RF'][sleeve].iloc[-1]:+.2f}  the spine alone explains "
                  f"{fit['spine_r2'][sleeve].iloc[-1]:.1%} of its variance")
            continue
        explained, residual = float(fit["explained_var"][sleeve].iloc[-1]), float(fit["resid_var"][sleeve].iloc[-1])
        share = explained / (explained + residual) if (explained + residual) else float("nan")
        print(f"    {sleeve:<16s} alpha {fit['alpha'][sleeve].iloc[-1]:+.4%}/month t "
              f"{fit['t_alpha'][sleeve].iloc[-1]:+.2f}  beta(world) {fit['beta']['Mkt-RF'][sleeve].iloc[-1]:+.2f}  "
              f"split {share:.1%} factors / {1.0 - share:.1%} idiosyncratic")

    drift = loading_drift({column: fit["beta"][column][estimated] for column in block}, estimated)
    worst = max(drift, key=lambda column: drift[column]["max"])
    fixed = fixed_alpha(returns, named, fit["beta"], months)
    difference = (fit["alpha"][estimated] - fixed[estimated]).abs()
    print(f"[factor] on the {len(estimated)} estimated sleeves the block loadings move "
          f"{min(v['mean'] for v in drift.values()):.3f}..{max(v['mean'] for v in drift.values()):.3f} a month "
          f"on average, worst {drift[worst]['max']:.2f} on {worst}. The loadings are identified now rather "
          f"than 0.999 collinear; the net-slope one is the noisiest because the net slope is the quietest of "
          f"the four series, and its t-statistic is where that shows")
    print(f"[factor] contrast, loadings held at the first window's: mean |alpha difference| "
          f"{difference.mean().mean():.4%}/month, worst {difference.max().max():.4%} on {difference.max().idxmax()}")

    quoted_fit = rolling(returns, quoted, months)
    gap = (fit["alpha"][estimated] - quoted_fit["alpha"][estimated]).abs()
    print(f"[factor] the same run on the spine's quoted currency moves the alpha by {gap.mean().mean():.4%}/month "
          f"on average; the constructed block is euro either way, so the gap is what the translation carries")
    europe_fit = rolling(returns, europe, months)
    print(f"[factor] cross-check on the Europe cut, five factors rather than six and translated through the same "
          f"euro leg: mean |alpha| {europe_fit['alpha'][estimated].abs().mean().mean():.4%}/month against "
          f"{fit['alpha'][estimated].abs().mean().mean():.4%} on the Developed spine, both in euro - a different "
          f"regional cut carried beside the headline, never merged into it")
    for line in document["warnings"]:
        print(f"[factor] {line}")
    return {"eur": fit, "quoted": quoted_fit, "europe": europe_fit, "returns": returns, "named": named,
            "estimated": estimated, "determined": determined}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
