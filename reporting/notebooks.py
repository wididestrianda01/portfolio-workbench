"""The module notebooks: one per entry point, on the seven-section spine the prototype fixed.

**What a notebook is for.** The package records what it computes; a notebook records why the
computation is the way it is, what a reader may conclude from its output, and what the output does not
establish. The second and sixth sections are the reason the deliverable exists: a reader who wants the
reasoning should not have to reconstruct it from the code, and a reader who takes a number from the
table should meet the resolution limit beside it.

**The sections, in order, and nothing else:**

1. What this module does, and the source of every method in it
2. Why it works this way, including what was rejected
3. The data contract it consumes, and the as-of rule
4. The worked example on small numbers, with the identity checked
5. The real run: inputs, parameters, provenance block
6. Results, and how to read them, including the resolution limit and what a reader must not conclude
7. What this module does not establish

**Every cell is generated from the code, not transcribed from a session.** Section 2 is the module's
own header rendered verbatim, so the reasoning in a notebook cannot drift from the reasoning in the
module it documents; section 1's source lines come from the source map; section 5 runs the module's own
entry point as a subprocess and prints what it prints. A hand-written notebook would be a second
account of the same modules, and the second account is the one that goes stale.

**One notebook per entry point, and every module is covered by one.** The spine is per module, and a
module is a thing a reader can run: the entry points below. A module that exists only to be imported
appears in the notebook of the entry point that drives it, which is what `drives` records and what
section 1 prints, so no module of the package is left without a notebook that mentions it.

After writing, each notebook is executed with nbclient, so "every notebook runs end to end" is a fact
about the files on disk rather than an intention.
"""

import ast
import sys
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from . import source_map

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NOTEBOOK_ROOT = ROOT / "notebooks"
ANALYTICS = ROOT / "portfolio_workbench"

# The spine, in the order the prototype fixed it. Sections 2 and 6 are the two that make a notebook a
# record of judgement rather than a record of computation, and both are named in the deliverable.
SPINE = (
    "What this module does, and the source of every method in it",
    "Why it works this way, including what was rejected",
    "The data contract it consumes, and the as-of rule",
    "The worked example on small numbers, with the identity checked",
    "The real run: inputs, parameters, provenance block",
    "Results, and how to read them, including the resolution limit and what a reader must not conclude",
    "What this module does not establish",
)

# The declaration every artifact of the effort carries, in its own line rather than in a footnote.
DECLARATION = (
    "*A learning exercise performed in role: a simulated mandate with no client and no institution. "
    "Nothing in this notebook is investment advice, a recommendation, or a client communication.*"
)

# The cell that states where the run's numbers come from, printed in every notebook before the run.
PROVENANCE_CODE = '''
from portfolio_workbench.data import loader, universe

document = loader.load_panel()
months = document.months
print(f"snapshot {document.snapshot_id}, taken as of {document.as_of}")
print(f"panel {len(months)} months {months.min()}..{months.max()} across {len(universe.TICKERS)} sleeves")
print("manifest fields: " + ", ".join(sorted(document.manifest)))
'''

# The cell that runs the entry point, exactly as the repository's own instructions run it.
RUN_CODE = '''
import subprocess
import sys
from pathlib import Path

# The package is imported from the repository root, so the run needs the root as its working
# directory rather than wherever the kernel was started. Walked up from the kernel's own directory
# rather than written in at generation time: an absolute path here would name one workstation, and
# the notebook is a file every reader runs on their own.
root = next(
    parent for parent in [Path.cwd(), *Path.cwd().parents] if (parent / "portfolio_workbench").is_dir()
)

finished = subprocess.run(
    [sys.executable, "-m", "{entry}"], capture_output=True, text=True, cwd=root
)
print(finished.stdout)
assert finished.returncode == 0, finished.stderr
'''.strip()


def _cells_for(record):
    """The notebook's cells, section by section, built from the records and the code."""
    cells = [
        _md(
            f"# {record['title']}\n\n{DECLARATION}\n\n"
            f"**Entry point** `python3 -m {record['entry']}`\n\n"
            f"**Modules covered** {', '.join('`' + module + '`' for module in record['drives'])}\n\n"
            "_Generated from the code by `python3 -m reporting.notebooks`: the module headers below "
            "are read out of the modules themselves, and the run is the entry point's own output._"
        )
    ]
    cells.append(_md(f"## 1. {SPINE[0]}\n\n{record['does']}"))
    cells.append(
        _md(
            "**Sources.** Every public function of the modules this notebook covers, and what it "
            "traces to. The map is checked over the code by the acceptance fixture, so a method "
            "added without a source fails a command rather than going unnoticed.\n\n"
            + "\n".join(record["sources"])
        )
    )
    cells.append(
        _md(
            f"## 2. {SPINE[1]}\n\n"
            "_The module headers, verbatim: each records why the module is shaped the way it is, "
            "what was rejected, and the measurement that settled it. They are quoted here rather "
            "than restated, so the notebook cannot drift from the code._\n\n"
            + "\n\n".join(record["headers"])
        )
    )
    cells.append(_md(f"## 3. {SPINE[2]}\n\n{record['contract']}"))
    cells.append(
        _md(
            f"## 4. {SPINE[3]}\n\n{record['example_note']}\n\nThe cell below runs on numbers small "
            "enough to check by hand and asserts the identity, so a reader can see the arithmetic "
            "rather than take the module's word for it."
        )
    )
    cells.append(_code(record["example"]))
    cells.append(
        _md(
            f"## 5. {SPINE[4]}\n\n"
            "The provenance block is printed first, then the parameters this module decides under, "
            "then the entry point's own report. The report is the module's output rather than a "
            "transcription of it, so a number quoted from a notebook is the number the module prints."
        )
    )
    cells.append(_code(PROVENANCE_CODE + record["parameters"]))
    cells.append(_code(RUN_CODE.format(entry=record["entry"])))
    cells.append(_md(f"## 6. {SPINE[5]}\n\n{record['reading']}"))
    cells.append(_md(f"## 7. {SPINE[6]}\n\n{record['not_establish']}"))
    return cells


def _md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def _code(source):
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


def headers(drives):
    """The module headers of the covered modules, verbatim, with the file each came from."""
    parts = []
    for module in drives:
        path = ANALYTICS / module
        doc = ast.get_docstring(ast.parse(path.read_text())) or "No module header."
        parts.append(f"**`{module}`**\n\n{doc}")
    return parts


def sources(drives):
    """The source map's line for every public function of the covered modules."""
    mapped = source_map.entries()
    lines = []
    for module in drives:
        for name, source in sorted(mapped.get(module, {}).items()):
            lines.append(f"- `{module}::{name}` traces to {source}")
    return lines


def notebook(record):
    """One notebook, on the spine, from one record."""
    record = dict(record, headers=headers(record["drives"]), sources=sources(record["drives"]))
    built = nbformat.v4.new_notebook(cells=_cells_for(record))
    built.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return built


def build(records=None, root=None, execute=True, timeout=900):
    """Write every notebook, executing each one, and return the paths."""
    target = Path(root) if root is not None else NOTEBOOK_ROOT
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for record in records if records is not None else RECORDS:
        path = target / f"{record['slug']}.ipynb"
        built = notebook(record)
        if execute:
            client = NotebookClient(
                built,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": ROOT}},
            )
            client.execute()
        nbformat.write(built, path)
        print(f"[notebook] {path.name}: {len(built.cells)} cells, executed {bool(execute)}")
        written.append(path)
    return written


RECORDS = (
    {
        "slug": "01-data-coverage",
        "title": "The data layer: the frozen panel and its coverage",
        "entry": "portfolio_workbench.data.coverage",
        "drives": (
            "data/coverage.py",
            "data/loader.py",
            "data/manifest.py",
            "data/panel.py",
            "data/quality.py",
            "data/universe.py",
            "data/external.py",
        ),
        "does": (
            "The data layer turns three external sources into one monthly table: a frozen price "
            "snapshot of eleven UCITS sleeves, the European Central Bank's overnight rate, and the "
            "Fama/French factor archives. Every other layer reads that table and nothing else. The "
            "coverage entry point prints what arrived, what was dropped, and from which month each "
            "sleeve is priced."
        ),
        "contract": (
            "One row per instrument-month, carrying `period_month` and `available_from`, plus the "
            "manifest that keys the whole snapshot. **A month's bar is available only from the first "
            "day of the month after the month it describes**, so a decision dated to a month can use "
            "that month's bar only if the decision is taken after its close. The price file's adjusted "
            "close is the total-return proxy, and a series whose adjusted close never diverges from "
            "its close is refused rather than patched, because a distribution-blind series would "
            "understate every return it carries. Euro returns are the local leg translated at the "
            "month's rate, less the overnight rate accrued over the month's days at the annualised "
            "quote divided by 360."
        ),
        "example_note": (
            "The as-of rule is the whole of the look-ahead defence, so the worked example is that rule "
            "on three months: each bar becomes available the month after it, and a decision dated to "
            "March can see February's bar and not March's."
        ),
        "example": '''
import pandas as pd

from portfolio_workbench.data import panel

months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
frame = pd.DataFrame({"period_month": months, "available_from": panel.available_from(months)})

# Each bar is available from the first day of the following month.
assert list(frame["available_from"].dt.strftime("%Y-%m")) == ["2020-02", "2020-03", "2020-04"]

# A decision dated 2020-03-15 sees January's and February's bars, and not March's.
seen = panel.as_of(frame, pd.Timestamp("2020-03-15"))
assert list(seen["period_month"]) == list(months[:2])
print("bars visible on 2020-03-15:", [str(month) for month in seen["period_month"]])
''',
        "parameters": '''
from portfolio_workbench.data import panel, universe

print(f"sleeves {len(universe.TICKERS)}: {', '.join(universe.TICKERS)}")
print(f"panel window {universe.WINDOW_START}..{universe.WINDOW_END}, as-of taken on the manifest's own date")
''',
        "reading": (
            "The panel's shape is the resolution limit of every number downstream: the joined months the "
            "provenance cell above prints, of which the out-of-sample window is the part the cells trade, "
            "and eleven sleeves of which four are priced from the first month of the panel. A reader must not read a "
            "coverage gap as a verdict on the sleeve: the emerging-market sleeve is absent by "
            "decision rather than by data failure, and a sleeve that starts late is reported with its "
            "start rather than filled backwards. Nor is a thin liquidity warning a statement about "
            "the sleeve's tradability at size, which this panel cannot observe."
        ),
        "not_establish": (
            "Nothing here establishes anything about instruments outside the eleven sleeves, about "
            "intra-month timing, or about execution: the panel is monthly and the prices are closing "
            "marks. It does not establish that the sleeves are the right exposures for the mandate, "
            "and the coverage report is a description of what was read rather than a judgement on "
            "the source."
        ),
    },
    {
        "slug": "02-factor-spine",
        "title": "The factor spine: the published spine and the constructed block",
        "entry": "portfolio_workbench.factors.spine",
        "drives": ("factors/spine.py",),
        "does": (
            "The factor layer regresses each sleeve on a small set of factors. Those factors are two "
            "things kept apart: a published spine of market, size, value, profitability, investment "
            "and momentum legs, and a constructed block of four bond and credit series built from the "
            "panel's own sleeves. The entry point builds both and prints their coverage."
        ),
        "contract": (
            "The spine arrives as monthly euro total returns from the archive's own files, with the "
            "archive's risk-free column refused as a factor because every sleeve's return is already "
            "taken against it. The block is arithmetic on the panel: the government level is the "
            "average of the two government sleeves, the term slope is long minus short, credit is "
            "investment grade over government, and high-yield excess is high yield over investment "
            "grade. The named set is the inner join of the two, and a missing value anywhere in it is "
            "refused rather than filled."
        ),
        "example_note": (
            "The block is four lines of arithmetic, which is the point of constructing it: a load of "
            "one on the government level is the sleeve's own construction rather than a fitted "
            "estimate. The worked example builds the block from four hand-chosen sleeve returns."
        ),
        "example": '''
import pandas as pd

from portfolio_workbench.factors import spine

sleeves = pd.DataFrame(
    {"IBGL.AS": [0.02, 0.01], "IEGE.AS": [0.01, 0.005], "IEAC.AS": [0.025, 0.012], "IHYG.L": [0.03, 0.02]}
)
block = spine.constructed_block(sleeves)

assert list(block.columns) == ["government_level", "term_slope", "credit", "high_yield_excess"]
assert block.loc[0, "government_level"] == 0.5 * (0.02 + 0.01)
assert block.loc[0, "term_slope"] == 0.02 - 0.01
assert block.loc[0, "credit"] == 0.025 - 0.02
assert block.loc[0, "high_yield_excess"] == 0.03 - 0.025
print(block.round(4))
''',
        "parameters": '''
from portfolio_workbench.factors import spine

print("constructed series: " + ", ".join(spine.CONSTRUCTED))
print(f"block sleeves: {', '.join(spine.BLOCK_SLEEVES)}")
print(f"columns refused as factors: {', '.join(spine.NOT_A_FACTOR)}")
''',
        "reading": (
            "Two readings decide what the rest of the factor layer can claim. The spine is six "
            "published legs on a European multi-asset panel, so a weak GRS statistic is expected and "
            "is not evidence that the factors are unpriced elsewhere. The block's loadings on the "
            "four construction sleeves are identities, so a high R-squared there measures the "
            "construction rather than the model, and only the other seven sleeves carry an alpha a "
            "reader may interpret. The block is this effort's own assembly: a reader must not read it "
            "as a vendor factor."
        ),
        "not_establish": (
            "The spine does not establish that these six factors are the right ones, and the block "
            "does not establish a term-structure or credit model: it is four differences of sleeves, "
            "with no fitted structure behind it. Nothing here establishes that a loading is stable, "
            "since stability is a question for the rolling estimation and is answered there."
        ),
    },
    {
        "slug": "03-factor-exposures",
        "title": "The exposure regression: rolling loadings and the factor split",
        "entry": "portfolio_workbench.factors.exposures",
        "drives": ("factors/exposures.py",),
        "does": (
            "The exposure layer fits every sleeve on the named set, once per estimation window, and "
            "splits each sleeve's variance into the part its factors explain and the part they do "
            "not. The entry point prints the loadings, the identity check on the four constructed "
            "sleeves, the alphas with their t-statistics, and the factor and idiosyncratic shares of "
            "tracking error."
        ),
        "contract": (
            "One ordinary least squares fit per window per group of sleeves, with an intercept, on "
            "the factors as declared. The window is sixty months by decision and the first window is "
            "allowed to be shorter than the rest only where the panel begins, which is stated on the "
            "step rather than smoothed. Every frame is reindexed onto the fit's own traded months, so "
            "a month cannot enter a regression the window did not cover, and the residual split "
            "divides by the residual degrees of freedom so that the fitted and residual parts sum to "
            "the sleeve's own sample variance."
        ),
        "example_note": (
            "Ordinary least squares on a window with two factors and five months, where the series "
            "is generated from known loadings: the recovered coefficients are the planted ones, which "
            "is the check that the design matrix, the alignment and the residual split are the "
            "arithmetic they claim."
        ),
        "example": '''
import numpy as np
import pandas as pd

from portfolio_workbench.factors import exposures

months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03", "2020-04", "2020-05"], freq="M")
factors = pd.DataFrame({"market": [0.01, -0.02, 0.03, 0.00, 0.02], "credit": [0.005, 0.001, -0.004, 0.002, 0.0]}, index=months)
# A sleeve planted as 2 x market + 3 x credit + a constant, with no residual at all.
sleeve = 2.0 * factors["market"] + 3.0 * factors["credit"] + 0.001
fit = exposures.regress(pd.DataFrame({"sleeve": sleeve}), factors)

assert np.allclose(np.asarray(fit["beta"]).ravel(), [2.0, 3.0], atol=1e-12)
assert abs(float(np.asarray(fit["alpha"]).ravel()[0]) - 0.001) < 1e-12
assert float(np.asarray(fit["r2"]).ravel()[0]) == 1.0
print("recovered loadings:", np.round(np.asarray(fit["beta"]).ravel(), 12).tolist())
''',
        "parameters": '''
from portfolio_workbench.factors import exposures

print(f"estimation window {exposures.WINDOW} months, minimum observations {exposures.MIN_OBS}")
print(f"block: {', '.join(exposures.BLOCK)}")
''',
        "reading": (
            "The R-squared on the four construction sleeves is one by construction, so it is a check "
            "on the arithmetic and not a result; the seven other sleeves carry the interpretable "
            "alphas. At sixty-month windows and eleven sleeves the loadings are estimated with error, "
            "and the layer reports the variance inflation and the loading drift beside them rather "
            "than claiming stability. A reader must not read a non-zero alpha on a diversified index "
            "leg as skill: an index sleeve's alpha is a statement about the factor set's ability to "
            "price that market, and the layer says as much."
        ),
        "not_establish": (
            "Nothing here establishes that the factors are priced, that the loadings are constant "
            "outside the window, or that a low alpha means a well-priced market. The regression "
            "measures association within windows, and its fixed-alpha and variance-inflation "
            "diagnostics are reported rather than acted on."
        ),
    },
    {
        "slug": "04-factor-components",
        "title": "The statistical components, and how many this panel supports",
        "entry": "portfolio_workbench.factors.components",
        "drives": ("factors/components.py",),
        "does": (
            "The statistical family extracts directions of common variation from the sleeve "
            "correlation matrix and decides how many of them the panel supports. The entry point "
            "prints the eigenvalues against both references, the retention count per window, the "
            "component portfolios, and the stability check on the count."
        ),
        "contract": (
            "The decomposition runs inside the trailing sixty-month window, so the count a window "
            "retains cannot see the month it will be used to trade. The count rule is stated before "
            "it is applied: an eigenvalue beats the 95th percentile of a matched permutation null "
            "built by permuting each series independently, which keeps the panel's own marginals and "
            "its own autocorrelation structure rather than assuming normal noise. Extraction returns "
            "loadings and shares; component *series* are taken from the component portfolios, "
            "because a standardised score has mean zero by construction and a series with an "
            "invented mean is not a return."
        ),
        "example_note": (
            "The Marchenko-Pastur edge and the retention rule are both hand-checkable. The worked "
            "example computes the edge for eleven series over sixty rows from its closed form, and "
            "applies the rule to three eigenvalues against a three-column null."
        ),
        "example": '''
import numpy as np

from portfolio_workbench.factors import components

# The Marchenko-Pastur upper edge for n series over t rows, from its closed form.
edge = components.mp_edge(11, 60)
assert abs(edge - (1 + np.sqrt(11 / 60)) ** 2) < 1e-12
print(f"eleven series over sixty rows: edge {edge:.4f}")

# The rule counts observed components that beat the null's own threshold for their position.
null = np.array([[1.30, 1.10, 1.00], [1.20, 1.00, 0.90], [1.25, 1.05, 0.95]])
assert components.retained(np.array([3.00, 1.15, 0.50]), null) == 2
assert components.retained(np.array([1.00, 0.90, 0.80]), null) == 0
print("two of three observed components clear the matched null; a noise panel clears none")
''',
        "parameters": '''
from portfolio_workbench.factors import components

print(f"pre-registered count bound k <= {components.PREREGISTERED_K}, null percentile {components.PERCENTILE}")
print(f"permutation draws per window {components.DRAWS}, seed {components.SEED}")
''',
        "reading": (
            "The count is the answer to how many directions this panel supports, and its resolution "
            "limit is the gap between the null's threshold and the eigenvalue it is compared with: a "
            "panel whose eigenvalues sit near the edge gives a count that moves with the window. The "
            "stability check is printed for that reason. A reader must not read the extracted "
            "components as priced factors, and must not read the rule as selecting them: extraction "
            "finds directions of common variation, and selection needs an external criterion stated "
            "in advance, which here is the retention rule and the bound on the count."
        ),
        "not_establish": (
            "Nothing here establishes that the components are economically identified, that they are "
            "stable outside the sample, or that a count of two is the right number rather than the "
            "count this rule retains on this panel. The rule's own bound is a pre-registered "
            "constraint, not a finding about markets."
        ),
    },
    {
        "slug": "05-factor-spanning",
        "title": "Spanning: whether either factor set prices what the other does",
        "entry": "portfolio_workbench.factors.spanning",
        "drives": ("factors/spanning.py",),
        "does": (
            "Spanning asks whether one factor set adds anything the other does not already price. The "
            "entry point runs the test in both directions, with the GRS statistic and its p-value "
            "beside the regression form, and reports the second-pass premium estimate with its own "
            "caveat."
        ),
        "contract": (
            "The test assets are the sleeves, the factors are the named set, and the regression runs "
            "on the months both cover at the count the retention rule holds. Both directions are run "
            "on the same month set, so the two p-values are comparable. The premium estimate is "
            "descriptive: with eleven series and one out-of-sample window the second pass is "
            "underpowered, and the layer says so rather than reporting a t-statistic as a result."
        ),
        "example_note": (
            "Two planted cases show what the statistic reads. Test assets built from the factors plus "
            "independent noise leave the alphas at zero and the test keeps its size; the same assets "
            "given a constant 40 bp a month the factors do not carry are rejected outright. Assets "
            "exactly reproducible from the factors are *not* the case to check: both the alphas and "
            "their residual covariance sit at rounding level there, and the ratio of the two is "
            "numerical noise rather than a statistic."
        ),
        "example": '''
import numpy as np
import pandas as pd

from portfolio_workbench.factors import spanning

months = pd.period_range("2020-01", periods=60, freq="M")
rng = np.random.default_rng(11)
factors = pd.DataFrame(
    {"market": rng.normal(0.004, 0.03, 60), "credit": rng.normal(0.001, 0.01, 60)}, index=months
)
noise = rng.normal(0.0, 0.004, (60, 2))
spanned = pd.DataFrame(
    {
        "a": 0.9 * factors["market"] + 0.4 * factors["credit"] + noise[:, 0],
        "b": 1.1 * factors["market"] - 0.2 * factors["credit"] + noise[:, 1],
    },
    index=months,
)

# Under the null the test keeps its size: this seed does not reject.
assert spanning.grs(spanned, factors)["p_value"] > 0.05

# A constant 40 bp a month the factors do not carry is what the test exists to detect.
planted = spanned + 0.004
assert spanning.grs(planted, factors)["p_value"] < 0.05
print(f"null case p {spanning.grs(spanned, factors)['p_value']:.4f}, planted alpha p "
      f"{spanning.grs(planted, factors)['p_value']:.4f}")
''',
        "parameters": '''
from portfolio_workbench.factors import spanning

print("the test form: Huberman & Kandel (1987) read as a regression of the test assets on the factors")
''',
        "reading": (
            "The test is run at the retained count and on the months both legs cover, so its power is "
            "bounded by that count and by that span: a failure to reject is the expected outcome on a "
            "panel of diversified index sleeves and is not evidence that the sets are equivalent. "
            "Where the test does reject, the direction matters and both are printed. A reader must "
            "not read the second-pass premium estimate as a priced-factor result: at this panel's "
            "size the standard error dominates it, and the layer prints the estimate beside that "
            "statement rather than instead of it."
        ),
        "not_establish": (
            "Nothing here establishes that a spanning rejection is economically meaningful, or that "
            "the two sets are the same model. The test is about whether one set's information is "
            "contained in the other's on this sample, and the answer is sample-specific."
        ),
    },
    {
        "slug": "06-risk-covariance",
        "title": "The risk-model axis: three covariance estimators and their conditioning",
        "entry": "portfolio_workbench.risk.covariance",
        "drives": ("risk/covariance.py",),
        "does": (
            "The risk layer carries the estimators the construction cells are built on: the sample "
            "covariance, linear shrinkage toward a structured target, and the covariance implied by "
            "the retained component model. The entry point prints each estimator's condition number "
            "and the shrinkage intensity it chose, which is how the axis is read."
        ),
        "contract": (
            "One matrix per window over the eleven sleeves in the universe's own order, so a matrix "
            "cannot silently disagree with the weight vectors it is used with. Every estimator is "
            "symmetric and positive semi-definite, and the intensity the shrinkage chose is returned "
            "beside the matrix rather than hidden inside it. The factor-model estimator reads the "
            "component structure the retention rule keeps, which is why the risk layer depends on "
            "the factor layer."
        ),
        "example_note": (
            "A two-sleeve case with known variances: the sample estimator reproduces them exactly, "
            "shrinkage lands between the sample and its target, and shrinkage's condition number is "
            "the better of the two. The identity checked is that the estimators are the objects "
            "their names claim."
        ),
        "example": '''
import numpy as np
import pandas as pd

from portfolio_workbench.risk import covariance

months = pd.PeriodIndex([f"2020-{i + 1:02d}" for i in range(12)], freq="M")
rng = np.random.default_rng(3)
frame = pd.DataFrame({"a": rng.normal(0.0, 0.02, 12), "b": rng.normal(0.0, 0.01, 12)}, index=months)

sample = covariance.sample(frame)
assert np.allclose(np.diag(sample.to_numpy()), frame.var(ddof=1).to_numpy(), atol=1e-15)

report = covariance.conditioning(frame)
assert report["sample_condition"] > report["shrinkage_condition"]
assert 0.0 < report["intensity"] < 1.0
assert set(report["covariances"]) == {"sample", "shrinkage", "factor"}
print(f"sample condition {report['sample_condition']:.1f} against shrinkage {report['shrinkage_condition']:.1f}, "
      f"intensity {report['intensity']:.4f}")
''',
        "parameters": '''
from portfolio_workbench.risk import covariance

print("the axis: the sample covariance, linear shrinkage, and the factor-model covariance")
print(f"reconstruction tolerance {covariance.RECONSTRUCTION_TOLERANCE:.0e} on the discarded eigenvalues")
''',
        "reading": (
            "The condition number is what a constructor feels, and it is reported because at sixty "
            "months and eleven sleeves the sample covariance's condition number makes the optimiser "
            "the object under test far more than the estimator. Shrinkage's intensity is a result "
            "about the window: an intensity near one says the sample carried almost no information, "
            "which is not a defect of the estimator. No accuracy is claimed or measurable from these "
            "numbers, and a reader must not read a better-conditioned matrix as a more accurate one."
        ),
        "not_establish": (
            "Nothing here establishes that any estimator forecasts future covariance better than "
            "another. Nothing establishes the target the shrinkage is drawn toward, which is a "
            "choice inside the estimator rather than a finding of this layer, and nothing here "
            "addresses tail dependence or regime change beyond the window's own variability."
        ),
    },
    {
        "slug": "07-construction",
        "title": "The constructor families and the constraint set",
        "entry": "portfolio_workbench.construct.families",
        "drives": ("construct/families.py", "construct/constraints.py", "construct/means.py"),
        "does": (
            "The construction layer turns a covariance, a mean input and a constraint set into a "
            "weight vector, and does it for seven families plus their variants: equal weight, the "
            "policy book, mean-variance (sample, shrunk, Black-Litterman and no-mean), minimum "
            "variance, maximum diversification, equal risk contribution, hierarchical risk parity "
            "and mean-CVaR. The constraint set is the same one for every cell, so a difference "
            "between cells is the objective rather than the rules."
        ),
        "contract": (
            "Long-only, fully invested, a 35% per-sleeve cap, a 1% absolute no-trade band and a 5% "
            "one-way turnover cap **on traded notional**. The band is applied to the target and the "
            "residual it leaves is brought back inside the constraint set before the book is traded, "
            "because a banded book can otherwise sum to something other than one. Cost is "
            "`2 x one-way turnover x per-side rate`: the rate is charged on traded notional, and "
            "one-way turnover is half of it. The no-mean cell bounds the weight vector's norm, so "
            "that nothing but the constraint set decides its answer."
        ),
        "example_note": (
            "Minimum variance on a diagonal covariance has a closed form, so the worked example "
            "checks the solver against the reciprocal variances and then shows the cap binding; the "
            "constraint set is checked separately by the projection that clipping alone would get "
            "wrong."
        ),
        "example": '''
import numpy as np
import pandas as pd

from portfolio_workbench.construct import constraints, families

covariance = pd.DataFrame(np.diag([0.04, 0.01, 0.0025]), index=list("abc"), columns=list("abc"))

# The closed form: weights proportional to the reciprocal variances, which the solver reproduces
# when nothing binds.
loose = families.minimum_variance(covariance, cap=1.0)
expected = np.array([1 / 0.04, 1 / 0.01, 1 / 0.0025])
expected = expected / expected.sum()
assert np.allclose(loose.to_numpy(), expected, atol=1e-6)
print("uncapped closed form:", loose.round(4).to_dict())

# The mandate's 35% cap binds here, and the delivered book is the capped projection of it.
capped = families.minimum_variance(covariance)
assert abs(capped.max() - 0.35) < 1e-9, capped
print("capped book:", capped.round(4).to_dict())

# Clipping alone leaves the book short of one; the projection hands the removed weight back.
assert np.allclose(constraints.bounded_simplex(np.array([0.9, 0.05, 0.05]), cap=0.5), [0.5, 0.25, 0.25])
''',
        "parameters": '''
from portfolio_workbench.construct import constraints, families, means

print(f"cap {constraints.CAP:.0%}, no-trade band {constraints.BAND:.0%}, turnover cap {constraints.TURNOVER_CAP:.0%}, "
      f"cost {constraints.COST_BP:.0f} bp per side on traded notional")
from portfolio_workbench.compare import registry

print("mean-input cells: " + ", ".join(cell["mean"] or "none" for cell in registry.STAGE_C))
print(f"risk aversion: gamma {families.RISK_AVERSION}")
''',
        "reading": (
            "Weight concentration is the resolution limit here: at eleven sleeves a 35% cap binds "
            "often, and a cell whose book sits on the bound is a result about the constraint set "
            "rather than about its objective. The entry point prints how often the cap bound on each "
            "book and how much of each target was left untaken at the turnover cap for that reason. "
            "A reader must not read a tight weight vector as a confident one: minimum variance has no "
            "mean in it and is exposed to covariance error, while the mean-variance cells are exposed "
            "to mean error instead, and both are visible in the weight stability the table prints."
        ),
        "not_establish": (
            "Nothing here establishes that an objective is the right one for a mandate, or that a "
            "better-constrained book will perform better. The no-mean cell establishes only that the "
            "constraint set is the estimator in that case. Nothing here addresses implementation "
            "shortfall, market impact or the size of the trade relative to the sleeve's turnover, "
            "none of which this panel observes."
        ),
    },
    {
        "slug": "08-walk-forward",
        "title": "The walk-forward engine: what each month may see",
        "entry": "portfolio_workbench.evaluate.walkforward",
        "drives": ("evaluate/walkforward.py",),
        "does": (
            "The walk-forward engine decides, month by month, which observations a decision may use, "
            "and records the decision so that the boundary can be asserted rather than trusted. The "
            "entry point prints the step table with each window's span, its availability date and "
            "what gated it."
        ),
        "contract": (
            "Rolling sixty-month estimation with a monthly refit: a step trading month m is estimated "
            "on the sixty months ending at m-1, and every bar enters only from the first day of the "
            "month after the month it describes. The expanding variant trades the same months with a "
            "window that grows from the start of the panel. Every window's last month precedes every "
            "month it trades, and the assertion that checks it is part of the run rather than a "
            "comment."
        ),
        "example_note": (
            "The boundary is the whole claim, so the worked example builds the steps for a short "
            "panel and asserts it directly: the window ends the month before the month it trades, and "
            "the traded months are the panel's own tail."
        ),
        "example": '''
import pandas as pd

from portfolio_workbench.evaluate import walkforward

months = pd.period_range("2020-01", periods=70, freq="M")
steps = walkforward.steps(months, kind="rolling", window=60)

assert len(steps) == 10
assert steps[0]["traded"] == pd.Period("2025-01", freq="M")
assert steps[0]["window_end"] == pd.Period("2024-12", freq="M")
for step in steps:
    assert step["window_end"] < step["traded"], step
    assert step["available_from"] > step["window_end"].to_timestamp(how="end")
print(f"{len(steps)} steps, first trading {steps[0]['traded']} on a window ending {steps[0]['window_end']}")
''',
        "parameters": '''
from portfolio_workbench.evaluate import walkforward

from portfolio_workbench.factors import exposures

print(f"protocols {tuple(walkforward.PROTOCOLS)}, refit {walkforward.REFIT}")
print(f"estimation window {exposures.WINDOW} months, out-of-sample months only")
''',
        "reading": (
            "The engine's output is a decision about what a month may see, and its resolution limit "
            "is the panel: the months the engine trades are what the sixty-month estimation requirement "
            "leaves of the joined calendar. The planted break in the acceptance fixture is the evidence "
            "that the assertion fires rather than merely existing. A reader must not read the first "
            "window's shorter observation count as a data gap: the panel simply starts there, and the "
            "step records it."
        ),
        "not_establish": (
            "Nothing here establishes that the protocol matches how a manager would actually trade. "
            "It fixes a monthly decision with a monthly refit and no intra-month action, and it does "
            "not model the delay between a decision and its execution beyond the one-month "
            "availability rule."
        ),
    },
    {
        "slug": "09-brinson",
        "title": "The holding-based attribution: allocation, currency and the link",
        "entry": "portfolio_workbench.attribute.brinson",
        "drives": ("attribute/brinson.py",),
        "does": (
            "The attribution layer explains each cell's active return from its holdings. The "
            "holding-based view splits the active return into allocation, a currency line and their "
            "cross product, links the monthly effects so that they sum to the compounded excess, and "
            "names the cost line. The entry point prints each cell's lines, the link's residual and "
            "the cost the grid actually charged."
        ),
        "contract": (
            "One weight per sleeve against the policy benchmark's weights, on the panel's own "
            "currency split. A sleeve's euro return is `(1 + local)(1 + translation) - 1`, so "
            "allocation is measured on **local-currency** returns and the currency move is its own "
            "line: measuring allocation in euro and then adding a currency line counts the "
            "translation twice, and the double count is invisible because the lines still sum. "
            "Selection and interaction do not exist here and are stated as structural rather than "
            "printed as zeros: the policy benchmark holds the same instrument for each sleeve, so "
            "`r_p,i = r_b,i` for every sleeve."
        ),
        "example_note": (
            "The Brinson arithmetic on two sleeves and one month, with the three lines summing to the "
            "active return by construction, and the level the link is measured against taken in the "
            "frame the rest of the package reports in."
        ),
        "example": '''
import numpy as np

from portfolio_workbench.attribute import brinson

portfolio_weights = np.array([0.60, 0.40])
benchmark_weights = np.array([0.50, 0.50])
portfolio_returns = np.array([0.02, 0.01])
benchmark_returns = np.array([0.01, 0.02])

single = brinson.single_period(portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns)
assert abs(single["totals"]["allocation"] + single["totals"]["interaction"] - single["active"]) < 1e-15
# Allocation uses the benchmark's own return as the reference level, which is what makes the sum
# identify the active return rather than a mixture of two framings.
assert abs(single["totals"]["allocation"] - (0.10 * (0.01 - 0.015) + -0.10 * (0.02 - 0.015))) < 1e-15
assert single["source"] == brinson.BRINSON_FACHLER
print("allocation", round(single["totals"]["allocation"], 6), "interaction", round(single["totals"]["interaction"], 6),
      "active", round(single["active"], 6))
''',
        "parameters": '''
from portfolio_workbench.attribute import brinson

print(f"reported form: {brinson.BRINSON_FACHLER}")
print(f"cross-check form: {brinson.BRINSON_HOOD_BEEBOWER}")
print(f"link: {brinson.CARINO}")
''',
        "reading": (
            "The linked effects sum to the compounded geometric excess, and the residual is reported "
            "rather than absorbed, so a reader can see how much of the answer the link carries. "
            "Allocation-only is a statement about this universe, not about the method: with one "
            "instrument per sleeve there is no security selection to measure. The cost line is the "
            "charge the grid applied, so a difference between cells after cost is a difference the "
            "account has already paid for. A reader must not read the currency line as a decision a "
            "manager took: it is the translation the sleeve was exposed to."
        ),
        "not_establish": (
            "Nothing here establishes that allocation was a deliberate exposure rather than a drift "
            "the benchmark rule produced. It does not measure security selection, because the "
            "universe has none to measure, and it says nothing about the sleeves' internal holdings."
        ),
    },
    {
        "slug": "10-factor-attribution",
        "title": "The factor view, and the residual between the two views",
        "entry": "portfolio_workbench.attribute.factor",
        "drives": ("attribute/factor.py",),
        "does": (
            "The factor view explains the same active return the holding-based view decomposes, as "
            "the sum of the passive exposures the book carries, the active loadings it took, and its "
            "alpha. The entry point prints both views side by side with the residual between them, "
            "and refuses a skill claim for the worked fund example on the stated power argument."
        ),
        "contract": (
            "Coefficients are read on the basis they were fitted in: the block net of its predecessors "
            "inside the window, with the window's own map carrying that basis to a month outside it. "
            "Both views are taken **complete**, so the cross-view residual is zero by construction "
            "and fires only when one view reads a different month set. The unexplained part is "
            "reported as its own measured quantity rather than absorbed into that residual."
        ),
        "example_note": (
            "The cross-view residual is an identity, so the worked example checks it directly: two "
            "views of one return that agree produce a zero residual, and a misaligned view makes it "
            "fire."
        ),
        "example": '''
import pandas as pd

from portfolio_workbench.attribute import factor

# Two views that decompose the same active series agree exactly; one that reads a different month
# set does not, which is the only thing this residual is allowed to detect.
months = pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
holding = pd.Series([0.002, -0.001, 0.003], index=months)

# Two views that decompose the same active series agree exactly.
agreement = factor.cross_view(holding, holding)
assert agreement["worst"] == 0.0 and agreement["reconciles"] is True

# A view that reads a different month set makes the residual fire, which is the only thing it detects.
misaligned = pd.Series([0.002, -0.001, 0.000], index=months)
assert factor.cross_view(holding, misaligned)["reconciles"] is False
print("aligned views gap", agreement["worst"], "; a misaligned view fires at", factor.cross_view(holding, misaligned)["worst"])
''',
        "parameters": '''
from portfolio_workbench.attribute import factor

print("the two views are never added to each other; they explain one active return and are compared")
''',
        "reading": (
            "The residual between the views is zero by construction, so its value is not the reading: "
            "what a reader reads is the unexplained part, which is the model's out-of-sample residual "
            "and is measured rather than hidden. Alpha on a cell is a residual after the factors, and "
            "its resolution limit is the detectable-alpha bar the worked example prints. A reader "
            "must not read the alpha as skill: the fund example's own alpha is refused precisely "
            "because the panel's power cannot separate it from zero."
        ),
        "not_establish": (
            "Nothing here establishes that the factor set explains the active return in a causal "
            "sense, or that alpha is stable out of sample. The two views are two accounts of one "
            "series, not two independent confirmations, and the unexplained part is a limit of the "
            "model rather than a market fact."
        ),
    },
    {
        "slug": "11-risk-budget",
        "title": "The risk budget: Euler contributions against what was declared",
        "entry": "portfolio_workbench.budget.euler",
        "drives": ("budget/euler.py",),
        "does": (
            "The budget layer decomposes each cell's risk by Euler contributions and reads the "
            "result against the vector the mandate declared before construction: 55% equity, 20% "
            "government, 15% credit, 10% real and cash. The entry point prints the realised shares "
            "beside the target, the additivity residual, and the ex-ante against ex-post tracking "
            "error."
        ),
        "contract": (
            "Euler contributions in percentage of volatility, weighted by the covariance the cell was "
            "built on, aggregated into the declared budget groups in the data layer's own order. "
            "Additivity is checked numerically on every run rather than assumed, and value at risk is "
            "refused as a measure to decompose: its conditional contributions do not sum to it, which "
            "is measured rather than asserted."
        ),
        "example_note": (
            "For two sleeves with a diagonal covariance the contributions are hand-checkable: sleeve "
            "a's share is its own variance over the portfolio's, and the two sum to the portfolio "
            "volatility."
        ),
        "example": '''
import numpy as np

from portfolio_workbench.budget import euler

weights = np.array([0.6, 0.4])
covariance = np.array([[0.04, 0.0], [0.0, 0.01]])
table = euler.contributions(weights, covariance)

volatility = float(np.sqrt(weights @ covariance @ weights))
assert abs(table["contribution"].sum() - volatility) < 1e-15
assert abs(table["share"].sum() - 1.0) < 1e-15
assert abs(table.loc[0, "share"] - 0.36 * 0.04 / volatility ** 2) < 1e-15
print(f"volatility {volatility:.6f}, shares {table['share'].round(4).tolist()}")
''',
        "parameters": '''
from portfolio_workbench.budget import euler
from portfolio_workbench.data import universe

print(f"declared budget: {universe.RISK_BUDGET}")
print(f"additivity tolerance {euler.ADDITIVITY_TOLERANCE:.0e} relative, checked on every run")
''',
        "reading": (
            "The gap between realised and target is the answer to whether the budget was consumed by "
            "design or by accident, and it is a property of the constructor: a risk-based family "
            "concentrates volatility where the covariance says the risk is, which may not be where "
            "the budget says it should be. The resolution limit is the window's own covariance, so a "
            "gap measured on one estimation window moves with the next. A reader must not read a "
            "realised share near its target as evidence that the constructor targeted it, since a "
            "book can land near a target for reasons the objective never mentioned."
        ),
        "not_establish": (
            "Nothing here establishes that the declared vector is the right budget for the mandate, "
            "or that a realised share far from it is a failure: it is a description of the book the "
            "objective produced. Nothing here decomposes tail risk additively beyond expected "
            "shortfall, and the value-at-risk refusal is a statement about additivity rather than "
            "about the measure's usefulness in other roles."
        ),
    },
    {
        "slug": "12-cell-grid",
        "title": "The pre-registered grid: twenty runs and what each one changes",
        "entry": "portfolio_workbench.compare.grid",
        "drives": ("compare/grid.py", "compare/registry.py"),
        "does": (
            "The grid is the runner: twenty pre-registered runs across three stages, each stage "
            "moving one axis. Stage A varies the constructor family, stage B the risk model, and "
            "stage C the mean input, with two perturbation runs and two expanding-protocol repeats "
            "beside them. The entry point runs them all, prints what each produced, and writes one "
            "manifest per run."
        ),
        "contract": (
            "The cell list is declared before any cell runs and is validated at import time, so the "
            "pre-registered count is a fact about the code. Each cell trades the same out-of-sample "
            "months on the same constraint set and the same cost convention, and each writes a manifest "
            "keyed to the snapshot it read. The runner charges cost on traded notional, step by step, "
            "along the path the book actually took."
        ),
        "example_note": (
            "The grid's own claim is that its cell list was fixed before the runs, so the worked "
            "example is that declaration read back: the count, the stage membership, and the "
            "perturbation pairing."
        ),
        "example": '''
from portfolio_workbench.compare import registry

assert registry.PRE_REGISTERED == 20
assert registry.validate()["cells"] == 16
assert len(registry.STAGE_A) + len(registry.STAGE_B) + len(registry.STAGE_C) == 16
assert set(registry.PERTURBATION_PAIRS) <= {cell["id"] for cell in registry.CELLS}
assert set(registry.PERTURBATION_PAIRS.values()) == {cell["id"] for cell in registry.PERTURBED}
assert set(registry.BOUNDS_PAIRS.values()) <= {cell["id"] for cell in registry.CELLS}
assert len(registry.REPEATED) == 2
print("stage A", len(registry.STAGE_A), "stage B", len(registry.STAGE_B), "stage C", len(registry.STAGE_C),
      "perturbed", len(registry.PERTURBED), "repeated", len(registry.REPEATED))
''',
        "parameters": '''
from portfolio_workbench.compare import grid, registry

print(f"pre-registered runs {registry.PRE_REGISTERED}, distinct cells {len(registry.CELLS)}")
print(f"run manifests are written under the snapshot's own directory, {grid.shown_path(grid.DEFAULT_RUN_ROOT)}")
''',
        "reading": (
            "The grid produces the series the comparison is read from, and its resolution limit is "
            "the cell count: the distinct cells tested against two families leave a family-wise "
            "bar that a real but modest advantage will not clear. A reader must not read a cell's "
            "rank in the table as a recommendation: the ranking is descriptive, and a recommendation "
            "requires every rung of the ladder including the bootstrap and the second protocol, "
            "which the comparison table applies. The perturbation runs are comparisons, not "
            "recommendations, and they exist to show how much of a cell's result the constraint set "
            "was doing."
        ),
        "not_establish": (
            "Nothing here establishes that the twenty runs exhaust the design space: they are the "
            "pre-registered cells, and a family outside the list is simply untested. Nothing here "
            "establishes anything about a live implementation, and the manifests record what ran "
            "rather than asserting that a rerun will match."
        ),
    },
    {
        "slug": "14-data-sql",
        "title": "The data contract and the panel's own table, as queries",
        "entry": "portfolio_workbench.data.sql",
        "drives": ("data/sql.py",),
        "does": (
            "The panel is read by pandas, and the same work is stated as SQL over the snapshot's own "
            "files: the as-of join, the coverage report, and the assembly of the monthly euro excess "
            "return table the factor model is fitted on. The entry point runs all three and prints "
            "whether each agrees with the pandas path, which they are asserted to do on the frozen "
            "snapshot."
        ),
        "contract": (
            "The same contract, in the query's arithmetic: `period_month` is `date_trunc('month', date)` "
            "and `available_from` is that month plus one, so a bar labelled with a month becomes readable "
            "on the first day of the next. The priced legs are read as one long table by union, the gate "
            "is `available_from <= the moment the reader stands at`, and the coverage query reports the "
            "months carried beside the months visible at that moment. The pandas path stays the authority "
            "the analytics run on; the query is checked against it. The third statement is the table "
            "itself rather than its columns: the price and currency legs reduced to month-on-month "
            "ratios, the currency translation as `(1 + r) / (1 + fx) - 1` rather than a sum, the "
            "overnight rate compounded within the month at /360 on the month's own calendar days, and "
            "every leg under the same gate and window."
        ),
        "example_note": (
            "The rule's arithmetic is the thing worth checking twice, so the worked example states it in "
            "both languages on the same three dated bars and asserts that the two agree."
        ),
        "example": '''
import duckdb
import pandas as pd

from portfolio_workbench.data import panel

bars = pd.DataFrame({"date": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15"])})
pandas_rule = panel.available_from(pd.PeriodIndex(bars["date"], freq="M"))

connection = duckdb.connect()
connection.register("bar", bars)
queried = connection.execute(
    "SELECT date_trunc('month', bar.date) + INTERVAL 1 MONTH AS available_from FROM bar"
).fetch_df()

assert list(pd.to_datetime(queried["available_from"])) == list(pandas_rule)
print("both statements put a month's bar in the hands of the reader from", pandas_rule[0].date())
''',
        "parameters": '''
print("priced legs are read as one long table by union; the factor archives are not part of it")
print("the gate is available_from <= the moment read, and nothing else")
''',
        "reading": (
            "Two statements of one contract that agree are worth more than one statement, because the "
            "second can be read by someone who does not read Python and can be pointed at another "
            "project's loader. What the query does not carry is any of the quality gate: the eight stops "
            "and four warnings live in the pandas path, and a reader who took the query as the contract "
            "would satisfy the shape of the table without the checks on its content. The assembly's "
            "agreement is a tolerance rather than an identity, because the two paths compound a "
            "month's rate in a different order: on the frozen snapshot the largest month-instrument "
            "difference is zero, and the stated tolerance is what another snapshot's rerun would be "
            "read against."
        ),
        "not_establish": (
            "Nothing here establishes that the snapshot is usable, which is the quality gate's work and "
            "not the contract's. Nothing here establishes that a second project's file layout matches: "
            "what is fixed is the columns and the availability rule, and a sibling satisfies it by "
            "writing those columns rather than by reusing these queries."
        ),
    },
    {
        "slug": "13-comparison-table",
        "title": "The comparison table: verdicts, the noise floor and the cost",
        "entry": "portfolio_workbench.compare.table",
        "drives": ("compare/table.py", "evaluate/metrics.py", "evaluate/statistics.py"),
        "does": (
            "The comparison table turns the grid's series into the deliverable: two stacked blocks, "
            "each row a pre-registered run with its metric block, its two paired tests, its "
            "bootstrap rank retention, its resolution limit and its verdict. The second block "
            "carries what the choice cost and how much of the answer is estimation error."
        ),
        "contract": (
            "Only two families are tested: every cell against the policy benchmark and every cell "
            "against equal weight. The 120-pair cell matrix is published descriptively and never "
            "tested. A verdict is a ladder: the family-wise bar, which is also the multiple-testing "
            "haircut because one test serves both, the bootstrap rank retention at or above 80%, and "
            "the sign under the expanding protocol where such a run exists. Everything else prints "
            "**no difference detected** with the resolution limit beside it, and every number "
            "travels with the provenance block: the snapshot, the protocol, the cell id and the cost "
            "multiple."
        ),
        "example_note": (
            "The bars are numbers before anything is run, so the worked example computes them: the "
            "family-wise bar over the pre-registered cells, the haircut it produces, and the resolution "
            "a paired test of two identical series reports."
        ),
        "example": '''
import numpy as np
from scipy.stats import norm

from portfolio_workbench.compare import registry
from portfolio_workbench.evaluate import statistics

# The bar over the pre-registered cells, read from the cell list rather than typed in, and held to the
# identity the function claims: the two-sided normal quantile at the family-wise level, which is the
# reading the absolute statistic is decided under.
cells = len(registry.CELLS)
bar = statistics.family_wise_bar(cells)
assert abs(bar - norm.ppf(1.0 - statistics.ALPHA / (2 * cells))) < 1e-12, bar
assert statistics.haircut(3.0, bar)["clears"] is True
assert statistics.haircut(2.0, bar)["clears"] is False

# Two identical books leave no variance for the standard error, and the test says so rather than
# dividing by a rounding error.
same = np.array([0.001, -0.002, 0.003, 0.0005])
paired = statistics.paired(same, same, np.zeros(4))
assert paired["degenerate"] is True and paired["statistic"] == 0.0
print(f"bar over {cells} cells {bar:.4f}; identical series report a degenerate paired test")
''',
        "parameters": '''
from portfolio_workbench.evaluate import metrics, statistics

print(f"rank retention floor {statistics.RANK_RETENTION_FLOOR:.0%}, bootstrap draws "
      f"{statistics.BOOTSTRAP_DRAWS}, seed {statistics.SEED}, alpha {statistics.ALPHA}")
print(f"annualisation {metrics.PERIODS_PER_YEAR} periods, sub-periods {[name for name, _, _ in metrics.SUB_PERIODS]}")
''',
        "reading": (
            "The resolution limit is the line that makes the table readable: **no difference "
            "detected** means the difference is smaller than this panel can resolve, not that the "
            "methods are equivalent, and the smallest detectable information-ratio difference is "
            "printed on every row. The bootstrap's rank retention is the second bar, because a cell "
            "whose advantage does not keep its rank across resamples is the leader of this sample "
            "rather than of the strategy. A reader must not read the table as a recommendation about "
            "portfolio construction in general: it reports which methodological choice moved which "
            "metric on one European multi-asset panel out of sample, and the negative results are "
            "published as results."
        ),
        "not_establish": (
            "Nothing here establishes which family is best beyond this panel and this mandate, and "
            "the table refuses that question rather than answering it weakly. It does not establish "
            "that a cost of a few basis points a year is the whole cost, since market impact and "
            "capacity are outside the panel. It does not establish that the leader would repeat out "
            "of sample, which is the question the bootstrap and the expanding protocol exist to "
            "qualify."
        ),
    },
    {
        "slug": "15-the-analysis",
        "title": "The analysis: one snapshot, one run, one covariance",
        "entry": "portfolio_workbench.study",
        "drives": ("study.py",),
        "does": (
            "The analysis is not a method: it is the one place a snapshot's run and the readings taken off "
            "it are assembled. Nine call sites used to do that for themselves - the five entry points, the "
            "memo, the workbook, the grid's own report and the acceptance fixture - and an assembly is "
            "where a convention lives, which is how the risk budget came to be read against a covariance "
            "taken over a different set of months from the one the cells were built on. The entry point "
            "prints the shape of the run rather than a result: the numbers live in the modules behind it, "
            "each with its own report."
        ),
        "contract": (
            "The analysis consumes the document the loader returns and adds no contract of its own: the "
            "table contract, the as-of rule and the panel window belong to `data/`, and this module "
            "inherits them by reading what the loader handed it. What it does add is a constraint on the "
            "readings - every one of them is taken over the months the run traded, and against one sample "
            "covariance over those months - so a report naming a different window is describing a "
            "different object rather than a second view of this one."
        ),
        "example_note": (
            "The analysis's own identity is that its window is the run's window rather than a second cut of "
            "the calendar, and that it refuses a grid with no run instead of reading an empty frame as a "
            "window of zero months. The cell plants both cases; the covariance itself is checked where it "
            "belongs, in the acceptance fixture, which holds the analysis's covariance against the "
            "estimator module's own."
        ),
        "example": '''
import pandas as pd

from portfolio_workbench import study

# The window every reading is taken over is read off the run rather than re-cut, so a grid that traded
# three months is analysed over those three and not over whatever the calendar would have offered.
planted = {"results": [{"traded": pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")}]}
assert [str(month) for month in study.traded_months(planted)] == ["2020-01", "2020-02", "2020-03"]

# A grid that produced no run has no window, and the analysis refuses rather than reading the absence
# as a window of zero months.
try:
    study.traded_months({"results": []})
except ValueError as refusal:
    print("refused:", refusal)
else:
    raise AssertionError("a grid with no run should not yield a window")
''',
        "parameters": '''
from portfolio_workbench.compare import registry
from portfolio_workbench.data import universe

print(f"sleeves {len(universe.TICKERS)}: {', '.join(universe.TICKERS)}")
print(f"declared panel window {universe.WINDOW_START}..{universe.WINDOW_END}")
print(f"cells {len(registry.CELLS)} over {registry.PRE_REGISTERED} pre-registered runs")
print("covariance every reading is taken against: the sample covariance over the traded months")
''',
        "reading": (
            "The analysis is what makes two reports comparable: the window, the estimator and the book set "
            "their readings share are assembled in one place and stated by its entry point, so a reader "
            "holding the comparison table and the risk budget is holding two readings of one run. A reader "
            "must not read it as a result, because it computes nothing the modules behind it do not, and "
            "must not read it as the mandate either - the panel window, the universe, the policy weights "
            "and the constraint set are this build's decisions for one panel, and a consumer with a "
            "different mandate supplies its own document."
        ),
        "not_establish": (
            "Nothing here establishes anything about the panel, the methods or the cost convention: those "
            "are the modules' own subjects, and each states what it does not establish in its own "
            "notebook. It does not establish that the traded months are the right months to trade, which "
            "is the evaluation design's claim rather than a property of this module, and it makes no claim "
            "about the consumer boundary, which deliberately carries none of these decisions."
        ),
    },
    {
        "slug": "16-the-consumer-boundary",
        "title": "The consumer boundary: the data contract and the five analytics entry points",
        "entry": "portfolio_workbench.facade",
        "drives": ("facade.py",),
        "does": (
            "The boundary is the whole of what a consumer of this engine adopts: the data contract it "
            "satisfies, and the five entry points it then calls - factor exposures, risk models, "
            "construction, evaluation, and attribution with the Euler risk budget. The module holds no "
            "arithmetic of its own. It binds each entry point to the layer modules that do the work, so "
            "the names a consumer imports are the modules this repository runs and not a second copy of "
            "them, and its entry point prints that inventory rather than a report of results."
        ),
        "contract": (
            "The contract is the engine's own, unchanged by the boundary: one row per instrument-month "
            "carrying `period_month` and `available_from`, a manifest the loader verifies and fails "
            "closed on, and the as-of rule that a bar becomes readable on the first day of the month "
            "after the month it is labelled with. Nothing optional is added to it here, and no default "
            "is applied on the consumer's behalf - the window, the universe, the policy weights and the "
            "constraint set are this build's decisions for one panel and one mandate, so a consumer "
            "with a different mandate passes its own frames and its own weights into the entry points "
            "rather than inheriting these."
        ),
        "example_note": (
            "The boundary has no arithmetic, so the identity worth checking is object identity: a group "
            "member must be the layer's own module rather than a copy, which is what makes the promise "
            "that a module moved inside a layer costs one edit here rather than an edit in every "
            "consumer. The cell checks that, and checks that the groups are the five entry points and "
            "the contract the module declares."
        ),
        "example": '''
from portfolio_workbench import facade
from portfolio_workbench.budget import euler
from portfolio_workbench.risk import covariance

declared = {"contract", "factor_exposures", "risk_models", "construction", "evaluation", "attribution"}
assert {name for name, _ in facade.GROUPS} == declared

# Identity, not equality: a copy of a module would be the second account of it that this design
# exists to avoid.
assert facade.attribution.euler is euler
assert facade.risk_models.covariance is covariance
assert facade.attribution.euler.__name__ == "portfolio_workbench.budget.euler"

calls = sum(len(vars(group)) for _, group in facade.GROUPS)
print(f"{len(facade.GROUPS)} groups carrying {calls} modules, version {facade.VERSION}")
print("identity holds for every member: " + ", ".join(
    f"{name} {len(vars(group))}" for name, group in facade.GROUPS))
''',
        "parameters": '''
from portfolio_workbench import facade

print(f"build version {facade.VERSION}, the revision the repository tags this commit at")
print("entry points: " + ", ".join(name for name, _ in facade.GROUPS))
print("no window, universe, policy weights or constraint set is applied by the boundary")
''',
        "reading": (
            "The boundary is the surface a consumer pins: a revision tagged in this repository, with "
            "the contract above it and five named entry points behind it. A reader must not read the "
            "grouping as a completeness claim about portfolio methods - it is the order this build "
            "exercises them in, and the risk budget sits inside the attribution group because the two "
            "decompose the same realised series. A reader must not read the version constant as a "
            "guarantee either: the repository tag pins the commit, and a consumer that imports the layer "
            "modules directly rather than through these names has pinned nothing."
        ),
        "not_establish": (
            "Nothing here establishes that a consumer satisfies the contract by importing this module: "
            "the contract is a table shape, and whether another project's files carry it is decided by "
            "its own run of the quality gate and by comparing its returns with the pandas path. Nothing "
            "here establishes that the five groups are the five a different mandate needs, and nothing "
            "here makes any claim about results - the numbers live in the modules behind these names, "
            "each with its own notebook stating what it does not establish."
        ),
    },
)


def main(records=None, root=None, execute=True):
    """Write every notebook under `notebooks/`, executing each, and return the paths."""
    written = build(records, root, execute)
    print(f"[notebook] {len(written)} notebooks written to {NOTEBOOK_ROOT}")
    return written



if __name__ == "__main__":
    argv = sys.argv[1:]
    main(execute="--no-execute" not in argv)
