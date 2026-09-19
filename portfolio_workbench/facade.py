"""The engine's consumer boundary: the data contract and the five analytics entry points, in one import.

A consumer of this package satisfies the table contract - one row per instrument-month carrying
`period_month` and `available_from`, plus a manifest - and then calls the analytics through the names
below rather than through the layer modules directly. The indirection is not for its own sake: a module
moved inside a layer, a function renamed or a layer split then costs one edit in this file instead of an
edit in every consumer, and a consumer that pins the revision by tag gets a boundary whose contents it
can name before it imports anything.

**The five groups are the order a portfolio build exercises them in**: exposures, risk model,
construction, evaluation, and the decomposition of what came out. The Euler risk budget is grouped with
attribution rather than given a sixth name because the two read the same realised series into named
contributions, so a consumer that runs one almost always runs the other.

**What is deliberately absent.** No function is wrapped: a wrapper duplicates a signature that then
drifts from the module's own, or becomes a second place for a convention to be stated, and the module
headers are where those conventions live. No window, universe or mandate is applied either - those are
this build's own decisions, made for one panel and one mandate, and a consumer with a different mandate
supplies its own frames and its own policy weights. The boundary carries the machinery, not the choices.

**Version.** The repository tags the commit this boundary was released at, and `VERSION` names that
revision so a consumer that pinned the tag can read back which one it imported. The prose deliverables
carry the same constant rather than one of their own, so a document cannot cite a version the boundary
does not have.
"""

from types import SimpleNamespace

from .attribute import brinson, factor
from .budget import euler
from .construct import constraints, families, means
from .data import loader, manifest, panel, quality, sql
from .evaluate import metrics, statistics, walkforward
from .factors import components, exposures, spanning, spine
from .risk import covariance

VERSION = "1.0"

# The contract a consumer satisfies before calling anything else here: the manifest it writes, the
# loader that reads one back and fails closed on a mismatch, the as-of rule every join is gated on, the
# quality gate that decides whether the data may be used at all, and the same contract stated as a query.
contract = SimpleNamespace(manifest=manifest, loader=loader, panel=panel, quality=quality, sql=sql)

factor_exposures = SimpleNamespace(exposures=exposures, components=components, spanning=spanning, spine=spine)
risk_models = SimpleNamespace(covariance=covariance)
construction = SimpleNamespace(families=families, means=means, constraints=constraints)
evaluation = SimpleNamespace(walkforward=walkforward, metrics=metrics, statistics=statistics)
attribution = SimpleNamespace(brinson=brinson, factor=factor, euler=euler)

# The inventory, in the order the docstring lists the groups: one namespace per entry point, each
# carrying the modules that do its work. Read by `main` and by the acceptance fixture, so a group
# dropped from here or a module dropped from a group is a failed check rather than a quieter boundary.
GROUPS = (
    ("contract", contract),
    ("factor_exposures", factor_exposures),
    ("risk_models", risk_models),
    ("construction", construction),
    ("evaluation", evaluation),
    ("attribution", attribution),
)


def main():
    # The boundary prints under [table] by the print protocol's nearest-tag rule: the output is an
    # inventory, and the package has no tag for one of those.
    print(f"[table] consumer boundary version {VERSION}, {len(GROUPS) - 1} analytics entry points over "
          f"the data contract")
    for name, group in GROUPS:
        print(f"[table] {name}: {', '.join(sorted(vars(group)))}")
    return {"version": VERSION, "groups": {name: sorted(vars(group)) for name, group in GROUPS}}


if __name__ == "__main__":
    main()
