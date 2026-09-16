"""The source map: every public function in the analytics and the source it traces to.

The done criterion is that **every method is traced to a cited source**, and the way that becomes a
fact rather than a paragraph is this map plus the check over the code that the acceptance fixture runs:
`unmapped()` walks the analytics, enumerates every module-level function a reader could call, and
returns the ones with no entry here, so adding a method without stating where it comes from fails one
command rather than going unnoticed.

**Two kinds of source, and the map says which is which.** A method that is someone's published result
carries the citation, taken from the effort's own research dossiers rather than from memory - the
author, the year, the journal, and the DOI where the dossier records one. A method that is this
effort's own decision says so in those words and names the decision, because a decision presented as a
citation would be the more damaging of the two errors: it claims a literature for a choice the panel
and the mandate forced, and a reader who went looking for the paper would find nothing.

**Module-level entries are the default and per-function entries are the exceptions**, because most
modules implement one source through several functions (the eight quality stops are one set of rules,
the Cariño link is one paper) and repeating a citation eight times is how eight copies come to
disagree. A function whose source differs from its module's carries its own line.
"""

import ast
from pathlib import Path

from portfolio_workbench.attribute import brinson

HERE = Path(__file__).resolve().parent
ANALYTICS = HERE.parent / "portfolio_workbench"

# One source for every public function of a module, keyed by the module's path inside the package.
SHARED = {
    "attribute/brinson.py": (
        "the holding-based decomposition: Brinson & Fachler (1985), Journal of Portfolio Management 11(3), "
        "and Brinson, Hood & Beebower (1986), Financial Analysts Journal 42(4), reported allocation-only "
        "because one instrument per sleeve leaves no selection term to compute"
    ),
    "attribute/factor.py": (
        "the factor view of the same active return, read on the basis the coefficients were fitted in - a "
        "construction of this effort, not a published decomposition, and the basis convention is what makes "
        "it checkable against the holding-based total"
    ),
    "budget/euler.py": (
        "Euler's theorem applied to a risk measure homogeneous of degree one: the component contributions "
        "risk systems report as the risk budget"
    ),
    "compare/grid.py": (
        "the pre-registered cell grid of this effort: the risk-model axis, the protocol axis and the mean "
        "axis are declared before any of them runs, and the runner charges the mandate's cost convention"
    ),
    "compare/registry.py": (
        "the pre-registered count of this effort - sixteen distinct cells and twenty runs - declared in the "
        "design rather than chosen after the results"
    ),
    "compare/table.py": (
        "the verdict ladder of this effort's evaluation design, and the honesty furniture that travels with "
        "it: the resolution limit, the negative results and the provenance block"
    ),
    "construct/constraints.py": (
        "the mandate's constraint set and trading conventions of this effort: long-only, fully invested, a "
        "35% per-sleeve cap, a 1% absolute no-trade band and a 5% one-way turnover cap on traded notional"
    ),
    "construct/families.py": (
        "the constructor families: constrained mean-variance (Markowitz 1952), its analytic relatives and "
        "the risk-based family, each under the mandate's own constraint set"
    ),
    "construct/means.py": (
        "the mean-input axis: the sample mean, Bayes-Stein shrinkage and the Black-Litterman posterior, "
        "declared as an axis because the mean is the input the literature agrees carries the most error"
    ),
    "data/coverage.py": (
        "the coverage report of this effort: the joined panel's shape, its spans and the sleeves' own "
        "coverage, printed rather than summarised so a gap is visible before it is used"
    ),
    "data/external.py": (
        "the external legs: the ECB Data Portal's monthly series in the position it publishes them, the "
        "Fama/French archive's monthly returns, and the European Central Bank's daily/annual basis "
        "distinction with the /360 accrual it forces"
    ),
    "data/loader.py": (
        "the snapshot contract of this effort: a manifest plus one long-format file per leg, read as "
        "monthly bars keyed to the instrument and the period"
    ),
    "data/manifest.py": (
        "the manifest of this effort: what makes two runs the same run is the snapshot they read, so the "
        "instrument facts, the file hashes and the vintages are recorded rather than inferred"
    ),
    "data/panel.py": (
        "the table contract of this effort: one row per instrument-month carrying `period_month` and "
        "`available_from`, the as-of rule that gates every join, and the total-return proxy of the "
        "adjusted close"
    ),
    "data/sql.py": (
        "the table contract of this effort stated a second time as SQL: the same as-of rule and coverage "
        "report the pandas path carries, asserted against it on the frozen snapshot rather than adopted "
        "by it, because the contract another project satisfies is a table contract and a query states it "
        "in a form that can be read without reading Python"
    ),
    "data/quality.py": (
        "the eight stops and four warnings of this effort, each taken from the European UCITS universe "
        "audit's findings, plus the issuer facts the UCITS rules make checkable"
    ),
    "evaluate/metrics.py": (
        "standard performance arithmetic on monthly excess returns: the information ratio, tracking error "
        "and volatility at twelve periods a year, and the drawdown of the compounded series"
    ),
    "evaluate/statistics.py": (
        "the paired test, the bootstrap rank retention and the family-wise bar of this effort's evaluation "
        "design, with the multiple-testing haircut self-imposed from the literature (Harvey, Liu & Zhu 2016, "
        "Review of Financial Studies 29(1)) rather than presented as a regulatory requirement"
    ),
    "evaluate/walkforward.py": (
        "the out-of-sample protocol of this effort: rolling sixty-month estimation, monthly refit, and an "
        "assertion that every estimation window ends before the month it trades"
    ),
    "factors/components.py": (
        "principal components extraction over the sleeve correlation matrix, with the retention rule "
        "measured rather than assumed"
    ),
    "factors/exposures.py": (
        "the exposure regression of this effort: ordinary least squares per window on the named spine plus "
        "the constructed block, orthogonalised within the window so the block is net of the spine it is "
        "built from"
    ),
    "factors/spanning.py": (
        "the spanning test of Huberman & Kandel (1987), Journal of Finance 42(4), in the regression form "
        "that the second factor set adds nothing the first did not price"
    ),
    "factors/spine.py": (
        "the named spine and the constructed block of this effort: published index legs beside a block "
        "assembled from the sleeves' own income and credit lines, kept separate so a block loading is an "
        "identity rather than a finding"
    ),
    "risk/covariance.py": (
        "the risk-model axis of this effort: the sample covariance, linear shrinkage toward a structured "
        "target, and the factor-model covariance, each reported with the conditioning that decides which "
        "of them is usable at this ratio of series to observations"
    ),
}

# The functions whose source is not their module's, keyed by module and name.
OVERRIDES = {
    "attribute/brinson.py": {
        "single_period": (
            "the single-period arithmetic of Brinson & Fachler (1985) and Brinson, Hood & Beebower (1986): "
            "a weight deviation against the benchmark's own return"
        ),
        "carino": f"the reported link: {brinson.CARINO}",
        "menchero": f"the cross-check link: {brinson.MENCHERO}",
        "linking": (
            "the linking comparison itself: Cariño's smoothing factor against Menchero's single constant "
            "plus a period adjustment, with both landed on the compounded excess"
        ),
        "cost_line": (
            "the cost convention this effort pinned by execution: cost is charged on traded notional, "
            "`2 x one-way turnover x per-side rate`, so a factor of two cannot return silently"
        ),
    },
    "attribute/factor.py": {
        "fund_example": (
            "the worked decomposition of a fund's track record, whose skill claim is refused on the "
            "evaluation design's own power argument rather than on the fit"
        ),
        "cross_view": (
            "the cross-view identity of this effort: both views are taken complete, so the residual is zero "
            "by construction and fires only when one view reads a different month set"
        ),
    },
    "budget/euler.py": {
        "contributions": "Euler component contributions: `w_i d(sigma)/d(w_i)` in percentage of volatility",
        "additivity": (
            "the additivity condition of a measure homogeneous of degree one, checked numerically on every "
            "run rather than assumed"
        ),
        "var_refusal": (
            "the coherence failure of value at risk (Artzner, Delbaen, Eber & Heath 1999, Mathematical "
            "Finance 9(3)): its conditional contributions do not sum to it, which is measured here rather "
            "than asserted"
        ),
        "ex_ante_tracking_error": (
            "the ex-ante tracking error the risk model predicts, against the ex-post one the realised "
            "series measures - the forecast-quality pair of this effort's risk-budgeting design"
        ),
        "path_contributions": (
            "component contributions along the realised path of a book rather than at one date, which is "
            "what the risk budget is read against"
        ),
        "tail_contributions": (
            "the tail decomposition of expected shortfall, which does satisfy the additivity the value at "
            "risk does not"
        ),
    },
    "compare/grid.py": {
        "window_covariance": (
            "the risk-model axis read inside the estimation window: the sample covariance, Ledoit & Wolf "
            "shrinkage and the factor-model covariance of the component structure"
        ),
        "policy_weights": "the mandate's declared strategic weights, fixed before the build began",
        "black_litterman_convention": (
            "the Black-Litterman scaling convention of this effort: the prior's scaling as a multiple of "
            "1/observation count and the view's uncertainty over the same count, so the ratio is what "
            "moves and the count cancels"
        ),
        "components_seed": (
            "the permutation seed the count rule was declared under, fixed before the count was read"
        ),
    },
    "compare/table.py": {
        "verdict": (
            "the verdict ladder of the evaluation design: the family-wise bar, the bootstrap rank "
            "retention, the expanding-window sign, and the direction read after the cost is charged"
        ),
        "metric_table": (
            "the numbers a rerun has to reproduce, which is what makes two runs the same run rather than "
            "two prints of the same report"
        ),
        "reproduces": (
            "the rerun tolerance of this effort, stated in relative terms because an information ratio near "
            "zero and a cumulative return near one cannot share an absolute bound"
        ),
        "provenance": (
            "the provenance block of this effort: snapshot, protocol, cost multiple, bar - carried beside "
            "every number because that is the rule the effort's never-shorten list fixes"
        ),
        "block_tables": (
            "the two stacked blocks the prototype fixed: the return and risk metrics in the first, the cost "
            "and estimation-error diagnostics in the second"
        ),
    },
    "construct/families.py": {
        "maximum_diversification": (
            "the most diversified portfolio of Choueifaty & Coignard (2008), Journal of Portfolio "
            "Management 35(1): maximise the diversification ratio, the weighted average of the sleeves' "
            "volatilities over the portfolio's"
        ),
        "erc": (
            "equally weighted risk contributions, Maillard, Roncalli & Teiletche (2010), Journal of "
            "Portfolio Management 36(4)"
        ),
        "hierarchical_risk_parity": (
            "hierarchical risk parity, Lopez de Prado (2016), Journal of Portfolio Management 42(4): "
            "single-linkage clustering on correlation distance, quasi-diagonalisation, then recursive "
            "bisection"
        ),
        "mean_cvar": (
            "the mean-CVaR programme of Rockafellar & Uryasev (2000), Journal of Risk 2(3), as the linear "
            "programme that measures the tail of its own window"
        ),
        "minimum_variance": (
            "minimum variance, whose closed form against reciprocal variances is what the hand-checked "
            "identity case uses (Clarke, de Silva & Thorley 2006, Journal of Portfolio Management 33(1))"
        ),
        "mean_variance": (
            "constrained mean-variance, Markowitz (1952), Journal of Finance 7(1), at the stated risk-"
            "aversion weight"
        ),
    },
    "construct/means.py": {
        "jorion": (
            "Bayes-Stein shrinkage of the sample mean, Jorion (1986), Journal of Financial and Quantitative "
            "Analysis 21(3), DOI 10.2307/2331042"
        ),
        "black_litterman": (
            "the Black-Litterman posterior, Black & Litterman (1992), Financial Analysts Journal 48(5), in "
            "the form Walters (2011) derives"
        ),
        "none": (
            "the no-mean cell: the estimator is dropped and the constraint set decides the book, which is "
            "the control that separates a mean input's contribution from the solver's"
        ),
    },
    "construct/constraints.py": {
        "cost": (
            "the mandate's cost convention: a per-side rate charged on traded notional, so the charge is "
            "twice the one-way turnover"
        ),
        "banded": (
            "the no-trade band of the prototype's amended transition policy: a 1% absolute band per sleeve, "
            "with the residual it leaves closed by funding rather than left outside the sleeves"
        ),
        "establishment": (
            "the establishment cost of this effort: the whole book traded once outside the window, charged "
            "at the same rate and reported as its own line rather than amortised"
        ),
    },
    "data/external.py": {
        "splice_risk_free": (
            "the risk-free splice of this effort: EONIA and the euro short-term rate joined at the euro "
            "short-term rate's first observation, whose overlap was measured at +8.5 bp on 579 days"
        ),
        "accrue_monthly": (
            "the accrual rule the European Central Bank's published basis forces: its rate is annualised, so "
            "a month's return is the rate divided by 360 over that month's days rather than the rate "
            "compounded per period"
        ),
        "parse_french_zip": (
            "the Fama/French monthly archive and its two parser traps: the momentum column is WML, and the "
            "annual rows follow the monthly ones"
        ),
        "eur_translate": (
            "the translation leg of this effort: a sleeve's euro return is "
            "`(1 + local)(1 + translation) - 1`, so the currency line is a term rather than a residual"
        ),
    },
    "data/panel.py": {
        "available_from": (
            "the as-of rule of this effort: a month's bar is available only from the next month's first "
            "day, which is what makes the join look-ahead-free"
        ),
        "total_return": (
            "the total-return proxy of this effort: the adjusted close, with a distribution-blind series "
            "refused rather than patched"
        ),
        "eur_excess_returns": (
            "the panel's own series: euro total returns translated from the local leg, less the accrued "
            "cash rate"
        ),
    },
    "data/sql.py": {
        "as_of": (
            "the as-of gate as one predicate, `available_from <= when`: a bar labelled with a month is "
            "readable only from the first day of the following month, which is the difference between a "
            "monthly label and a monthly bar"
        ),
        "coverage": (
            "the coverage aggregation of this effort in SQL, with the months a moment can actually see "
            "beside the months carried"
        ),
        "agreement": (
            "the check that the two statements of the contract agree, which is what makes the query a "
            "statement of the interface rather than a second implementation of the panel"
        ),
        "returns": (
            "the assembly of the monthly euro excess return table of this effort, stated as one query: "
            "the price and currency legs as month-on-month ratios, the currency translation as "
            "`(1 + r) / (1 + fx) - 1` rather than a sum, the overnight rate compounded within the month "
            "at /360 on the month's own calendar days, and every leg under the same availability gate"
        ),
        "returns_agreement": (
            "the check that the query's table and the pandas path's agree cell by cell, which is what "
            "lets the statement stand as the contract's definition rather than as a paraphrase"
        ),
    },
    "data/quality.py": {
        "stop_dividend_blind": (
            "the distribution-blindness stop of this effort: a price series whose adjusted close never "
            "diverges from its close is refused rather than patched"
        ),
        "issuer_facts": (
            "the UCITS issuer facts the universe audit found checkable: fund size, income policy and the "
            "wrapper's own registration"
        ),
    },
    "evaluate/statistics.py": {
        "family_wise_bar": (
            "the family-wise bar over the declared cells: Bonferroni-equivalent, and the 95th percentile of "
            "the maximum of that many independent standard normals"
        ),
        "rank_retention": (
            "the bootstrap rank retention of the evaluation design, with the 80% floor declared before the "
            "resamples were drawn"
        ),
        "haircut": (
            "the multiple-testing haircut, self-imposed from the literature: nothing located imposes a "
            "multiple-testing correction on portfolio research"
        ),
    },
    "factors/components.py": {
        "mp_edge": (
            "the Marchenko-Pastur upper bulk edge, Marchenko & Pastur (1967), Matematicheskii Sbornik "
            "114(1): `(1 + sqrt(n/t))^2` on the standardised panel"
        ),
        "permutation_null": (
            "the matched empirical null of this effort: independent permutation of each series, so the null "
            "is the panel's own marginals rather than a simulation assumption"
        ),
        "retained": (
            "the retention rule of this effort, stated before it was applied: an eigenvalue beating the 95th "
            "percentile of the matched null, with the Kaiser eigenvalue-above-one rule rejected and the "
            "scree and parallel-analysis heuristics recorded in the dossier"
        ),
        "varimax": (
            "the varimax rotation, Kaiser (1958), Psychometrika 23(3), used only to orient an extracted "
            "basis and never to select one"
        ),
        "component_portfolios": (
            "the component portfolios of this effort: fully invested weights recovered from the component "
            "series, since a standardised score's own weights sum to zero"
        ),
        "stability": (
            "the stability check of this effort: the count re-decided on permuted windows, so a count that "
            "moves with the seed is refused by its own rule"
        ),
    },
    "factors/exposures.py": {
        "regress": (
            "ordinary least squares per window with the residual split reported: the factor and "
            "idiosyncratic parts of the tracking error"
        ),
        "orthogonalise": (
            "the block orthogonalised against the spine inside the same window, so a block loading is the "
            "sleeve's construction and not a fit"
        ),
        "variance_inflation": (
            "the variance inflation factor, the diagnostic for the collinearity the orthogonalisation exists "
            "to remove"
        ),
        "fixed_alpha": (
            "the fixed-alpha restriction as a diagnostic, reported because a panel of diversified index legs "
            "is expected to price near beta one on its own benchmark"
        ),
    },
    "factors/spanning.py": {
        "grs": (
            "the Gibbons, Ross & Shanken (1989), Econometrica 57(5), statistic reported beside the "
            "regression form"
        ),
        "premiums": (
            "the second-pass premium estimate and the errors-in-variables caveat that makes it a "
            "descriptive reading on this panel rather than a test"
        ),
        "fama_macbeth": (
            "the Fama & MacBeth (1973), Journal of Political Economy 81(3), two-pass estimate, reported as "
            "unsupported at this panel's size rather than as a result"
        ),
    },
    "factors/spine.py": {
        "named_set": (
            "the named spine of this effort: published index legs, with the archive's own risk-free column "
            "refused as a factor because it is the series every excess return is already taken against"
        ),
        "constructed_block": (
            "the constructed block of this effort: the sleeves' income and credit lines assembled so the "
            "block's loadings are identities"
        ),
    },
    "risk/covariance.py": {
        "shrinkage": (
            "linear shrinkage toward a structured target at the intensity the estimator itself estimates, "
            "Ledoit & Wolf (2004b), Journal of Multivariate Analysis 88(2), DOI 10.1016/S0047-259X(03)00096-4"
        ),
        "factor_model": (
            "the factor-model covariance of this effort, reading the component structure the count rule "
            "retained: `B F B' + D`, which is why the risk layer depends on the factor layer"
        ),
        "conditioning": (
            "the conditioning report of this effort, which is how the axis is read: at sixty months and "
            "eleven sleeves a sample covariance's condition number makes the optimiser the object under "
            "test far more than the estimator"
        ),
    },
}


def public_functions(root=None):
    """Every module-level public function in the analytics, as (module, name) pairs.

    The check enumerates the code rather than a hand-kept list, which is the difference between a rule
    and a paragraph: a method added tomorrow appears here without anyone remembering to add it, and its
    absence from the map is what fails.
    """
    root = Path(root) if root is not None else ANALYTICS
    found = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                found.append((relative, node.name))
    return found


def entries(root=None):
    """The map as module to function to source, with each module's default filled in."""
    mapped = {}
    for module, name in public_functions(root):
        mapped.setdefault(module, {})[name] = OVERRIDES.get(module, {}).get(name, SHARED.get(module))
    return mapped


def unmapped(root=None):
    """The public functions with no source, which is what the acceptance fixture asserts is empty."""
    missing = []
    for module, name in public_functions(root):
        source = OVERRIDES.get(module, {}).get(name, SHARED.get(module))
        if source is None:
            missing.append(f"{module}::{name}")
    return missing
