"""The skills-coverage matrix, as rows a check can read, and the check itself.

The matrix is a decision of this effort: it was built by reading 36 postings, the professional bodies'
outlines and the vendor documentation, and it states what the finished project exercises, at what
depth, per skill. Read as a table in a document it is a claim; read as rows and checked against the
code it is a fact about the build.

**A claim is a module, and the depth is honest.** Depth `A` means applied on real data, `D` exercised
once as a demonstration, `R` read about only and `-` not exercised. A row at `A` or `D` names the
module that does the work, and the check refuses a claim whose module does not exist or that the
acceptance fixture never runs. A row at `R` or `-` names no module, because a skill read about is not
a skill exercised, and the check refuses a claim that tries to have it both ways.

**Two rows are deliberate negatives.** Index methodology and the vendor model families sit at literacy
depth and say so: the first has no object because the benchmark is a policy portfolio, and the second
cannot be demonstrated without a licence this effort does not hold. They are in the matrix so that a
reader sees what the build does not claim, which is the whole reason the matrix exists rather than a
feature list.
"""

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANALYTICS = ROOT / "portfolio_workbench"

APPLIED = "A"
DEMONSTRATED = "D"
READ_ABOUT = "R"
NOT_EXERCISED = "-"

# One row per skill: what the posting literature asks for, the module of this build that does the work,
# and the depth the build reaches. `module` is None exactly where nothing is claimed.
ROWS = (
    {
        "skill": "Attribution and performance measurement",
        "module": "portfolio_workbench.attribute.brinson",
        "depth": APPLIED,
        "exercised_by": "Brinson-Fachler allocation-only, Carino linking with the Menchero cross-check, "
        "and the factor view beside it in attribute/factor.py",
    },
    {
        "skill": "Tracking error and active risk",
        "module": "portfolio_workbench.evaluate.metrics",
        "depth": APPLIED,
        "exercised_by": "the metric block, the factor and idiosyncratic split of tracking error, and "
        "ex-ante against ex-post in budget/euler.py",
    },
    {
        "skill": "Factor models and risk models",
        "module": "portfolio_workbench.factors.exposures",
        "depth": APPLIED,
        "exercised_by": "the named spine, the constructed term and credit block, the PCA count rule, and "
        "the risk-model axis in risk/covariance.py",
    },
    {
        "skill": "Tail risk and correlation",
        "module": "portfolio_workbench.construct.families",
        "depth": APPLIED,
        "exercised_by": "the mean-CVaR cell on expected shortfall, the correlation structure throughout, "
        "and value at risk computed once as a declared negative in budget/euler.py",
    },
    {
        "skill": "Portfolio construction and optimisation with constraints",
        "module": "portfolio_workbench.construct.constraints",
        "depth": APPLIED,
        "exercised_by": "mean-variance and its variants, minimum variance, maximum diversification, ERC "
        "bounded and unbounded, hierarchical risk parity, mean-CVaR, bounds and the turnover cap",
    },
    {
        "skill": "Benchmark construction",
        "module": "portfolio_workbench.compare.grid",
        "depth": APPLIED,
        "exercised_by": "the policy benchmark: its weights, its rebalancing rule and its drift against "
        "the no-trade band",
    },
    {
        "skill": "Covariance estimation",
        "module": "portfolio_workbench.risk.covariance",
        "depth": APPLIED,
        "exercised_by": "the sample covariance, Ledoit-Wolf shrinkage and the factor-model covariance, "
        "each reported with its conditioning",
    },
    {
        "skill": "Risk budgeting",
        "module": "portfolio_workbench.budget.euler",
        "depth": APPLIED,
        "exercised_by": "Euler contributions, additivity checked numerically on every run, realised "
        "against the declared budget, and the tracking-error contributions",
    },
    {
        "skill": "Shrinkage estimation",
        "module": "portfolio_workbench.construct.means",
        "depth": APPLIED,
        "exercised_by": "Bayes-Stein shrinkage of the mean as the mean-input axis, and linear shrinkage "
        "toward a structured target on the covariance",
    },
    {
        "skill": "Python and SQL",
        "module": "portfolio_workbench.data.sql",
        "depth": APPLIED,
        "exercised_by": "the package, plus the table contract and the panel's own table stated as "
        "DuckDB SQL over the snapshot's files: the as-of join, the coverage report, and the assembly of "
        "the monthly euro excess return frame - currency translation, /360 cash accrual, window and gate "
        "- asserted against the pandas path cell by cell",
    },
    {
        "skill": "Excel",
        "module": "reporting.workbook",
        "depth": DEMONSTRATED,
        "exercised_by": "the workbook export of the comparison table, written from the same structure the "
        "print renders",
    },
    {
        "skill": "Index methodology: weighting schemes and float adjustment",
        "module": None,
        "depth": READ_ABOUT,
        "exercised_by": "nothing: the benchmark is a policy portfolio with fixed weights, so weighting "
        "schemes and float adjustment have no object here",
    },
    {
        "skill": "VBA",
        "module": None,
        "depth": NOT_EXERCISED,
        "exercised_by": "nothing: a legacy layer, and no seat this effort targets screens on it",
    },
    {
        "skill": "Bloomberg and SimCorp Dimension",
        "module": None,
        "depth": READ_ABOUT,
        "exercised_by": "nothing: no licence, and the posting evidence does not expect prior experience "
        "with either",
    },
    {
        "skill": "Barra, RiskMetrics, Axioma, PORT, Aladdin, FactSet, Northfield",
        "module": None,
        "depth": READ_ABOUT,
        "exercised_by": "model literacy built from the vendors' own documentation, claimed as literacy "
        "and not as hands-on use",
    },
)


def modules():
    """The distinct modules the matrix claims, in the order the rows name them."""
    return [row["module"] for row in ROWS if row["module"]]


def imports(path):
    """The module paths one file imports, so 'exercised' can be read off the fixture rather than assumed."""
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            prefix = node.module
            found.add(prefix)
            found.update(f"{prefix}.{alias.name}" for alias in node.names)
    return found


def unbacked(fixture):
    """The claimed rows whose module does not exist or is not exercised by the acceptance fixture.

    Exercised means the fixture imports it, which is the narrowest reading that is still checkable: a
    claim is backed when the end-to-end run the fixture performs actually reaches the module, and a
    claim whose module nothing runs is a row of a table rather than a skill of a build.
    """
    imported = imports(fixture)
    problems = []
    for row in ROWS:
        claimed = row["depth"] in (APPLIED, DEMONSTRATED)
        if not claimed:
            if row["module"] is not None:
                problems.append(f"{row['skill']}: depth {row['depth']} claims a module")
            continue
        if row["module"] is None:
            problems.append(f"{row['skill']}: depth {row['depth']} claims no module")
            continue
        if not _exists(row["module"]):
            problems.append(f"{row['skill']}: {row['module']} does not exist")
        elif row["module"] not in imported:
            problems.append(f"{row['skill']}: {row['module']} is not exercised by the fixture")
    return problems


def _exists(name):
    relative = name.split(".")
    module = ANALYTICS / Path(*relative[1:-1]) / f"{relative[-1]}.py" if relative[0] == "portfolio_workbench" else None
    if module is not None and module.exists():
        return True
    return (ROOT / Path(*relative)).with_suffix(".py").exists()


def table():
    """The matrix as it is reported, with the depth spelled out rather than left as a letter."""
    spelled = {APPLIED: "applied on real data", DEMONSTRATED: "exercised as a demonstration",
               READ_ABOUT: "read about only", NOT_EXERCISED: "not exercised"}
    return [
        {
            "skill": row["skill"],
            "depth": spelled[row["depth"]],
            "exercised_by": row["exercised_by"],
            "module": row["module"] or "nothing claimed",
        }
        for row in ROWS
    ]


def main(fixture=None):
    """Print the matrix and whether every claim is backed, which is the skills-coverage check."""
    fixture = Path(fixture) if fixture else ROOT / "tests" / "test_acceptance.py"
    for row in table():
        print(f"[table] {row['skill']:<58s} {row['depth']:<28s} {row['module']}")
    problems = unbacked(fixture)
    print(
        f"[table] skills coverage: {len(modules())} claimed rows, all backed by a module the acceptance "
        f"fixture runs" if not problems else f"[table] skills coverage: {problems}"
    )
    return problems


if __name__ == "__main__":
    main()
