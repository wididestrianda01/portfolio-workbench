"""The pre-registered grid: what is run, on which axis value, and in what order.

This file is the registration. Every run is declared here, before any of them is executed, and the
count is fixed rather than emergent: a run added later is a dated amendment to this file and to the
record beside it, never a row that appeared in a table. The table's authority rests on the count
being stated in advance - a search over sixteen methods whose size is reported after the results are
seen is a search whose size was chosen by the results.

**Three stages, one axis moving in each.** Stage A varies the constructor family with the risk model
held at the sample covariance and the mean held at one default. Stage B varies the risk model with
the constructor held at the two families whose behaviour is most covariance-dependent, minimum
variance and equal risk contribution. Stage C varies the mean input inside mean-variance, the only
family here that consumes a mean. The constraint set is fixed across all three and perturbed twice.

**Sixteen distinct cells, eighteen runs, twenty pre-registered.** Stage B computes six runs and
contributes four new cells, because the sample-covariance pair repeats Stage A's minimum variance
and equal risk contribution. Stage C computes four and contributes three, because the shrunk mean is
already Stage A's mean-variance cell. Two further runs repeat the perturbation on the two families
the reading records as the ones constraints actually bite, and two repeat a pair of cells under the
secondary protocol, so a ranking that only exists under a 60-month window is visible as such.

**The two protocols repeat the same cells, not different ones.** The expanding run covers the same
out-of-sample months with each estimate using every month since the panel opened, which is the check
that a result is not an artefact of a fixed window length. The two cells chosen for it are the two the
mandate's own question turns on: the mean-consuming family's representative, which is the industry
default under review, and minimum variance, which is the risk-based family's representative and the
cell the risk-model stage is built around. Both are named here before either is run.

**The perturbation lifts the per-sleeve cap and changes nothing else.** It answers whether the
constraints were doing the work the covariance was supposed to do, so a second change alongside the
cap would answer a question about the pair.
"""

from ..construct import constraints
from ..construct.families import UNCAPPED
from ..construct.means import TAU_SCALE, TAU_SENSITIVITY
from ..data import universe

# The sleeve count the tightest norm budget is derived from, which is the sleeve map's own count.
SLEEVES = len(universe.TICKERS)
# The norm budget of the no-mean form: the smallest the fully invested constraint admits, so the
# feasible set is a single point and the constraint set decides the answer without the mean.
TIGHTEST_NORM = 1.0 / SLEEVES ** 0.5


def _cell(identifier, stage, family, estimator, mean=None, **extra):
    """One declared run, with every axis value named rather than defaulted at execution time."""
    spec = {
        "id": identifier,
        "stage": stage,
        "family": family,
        "estimator": estimator,
        "mean": mean,
        "cap": extra.pop("cap", constraints.CAP),
        "protocol": extra.pop("protocol", "rolling"),
        "norm": extra.pop("norm", None),
        "settings": extra.pop("settings", None),
        "base_setting": extra.pop("base_setting", TAU_SCALE),
    }
    if extra:
        raise ValueError(f"{identifier} carries undeclared axis values: {sorted(extra)}")
    # The keyword arguments the constructor reads. Only the no-mean form takes one, and it is the
    # constraint set rather than an axis value, so it travels in its own mapping instead of widening
    # every family's signature to accept a bound nobody else applies.
    spec["family_options"] = {} if spec["norm"] is None else {"norm": spec["norm"]}
    return spec


# Stage A: the constructor family moves; sample covariance and one default mean throughout.
STAGE_A = (
    _cell("equal_weight", "A", "equal_weight", "sample"),
    _cell("policy", "A", "policy", "sample"),
    _cell("mean_variance_shrunk", "A", "mean_variance", "sample", mean="jorion"),
    _cell("minimum_variance", "A", "minimum_variance", "sample"),
    _cell("maximum_diversification", "A", "maximum_diversification", "sample"),
    _cell("erc_unbounded", "A", "erc", "sample", cap=UNCAPPED),
    _cell("erc_bounded", "A", "erc", "sample"),
    _cell("hierarchical_risk_parity", "A", "hierarchical_risk_parity", "sample"),
    _cell("mean_cvar", "A", "mean_cvar", "sample"),
)

# Stage B: the risk model moves; the constructor is held at the two most covariance-dependent families.
STAGE_B = (
    _cell("minimum_variance_shrinkage", "B", "minimum_variance", "shrinkage"),
    _cell("minimum_variance_factor", "B", "minimum_variance", "factor"),
    _cell("erc_shrinkage", "B", "erc", "shrinkage"),
    _cell("erc_factor", "B", "erc", "factor"),
)

# Stage C: the mean input moves inside mean-variance. The black-litterman cell declares the ratio run
# it reports beside its base case, so the sensitivity is part of this cell rather than two more runs.
STAGE_C = (
    _cell("mean_variance_sample", "C", "mean_variance", "sample", mean="sample"),
    _cell(
        "mean_variance_black_litterman",
        "C",
        "mean_variance",
        "sample",
        mean="black_litterman",
        settings=TAU_SENSITIVITY,
        base_setting=TAU_SCALE,
    ),
    _cell("mean_variance_none", "C", "mean_variance", "sample", mean="none", norm=TIGHTEST_NORM),
)

CELLS = STAGE_A + STAGE_B + STAGE_C

# The perturbation: the per-sleeve cap lifted, everything else unchanged. The pairs are declared here
# so that the report can read each perturbation beside the cell it perturbs, and so that `validate`
# can check that the cap is the only thing that moved.
PERTURBATION_PAIRS = {
    "minimum_variance": "minimum_variance_uncapped",
    "maximum_diversification": "maximum_diversification_uncapped",
}
PERTURBED = (
    _cell("minimum_variance_uncapped", "A", "minimum_variance", "sample", cap=UNCAPPED),
    _cell("maximum_diversification_uncapped", "A", "maximum_diversification", "sample", cap=UNCAPPED),
)

# The secondary protocol, on the two cells the mandate's question turns on.
REPEATED = (
    _cell("mean_variance_shrunk_expanding", "A", "mean_variance", "sample", mean="jorion", protocol="expanding"),
    _cell("minimum_variance_expanding", "A", "minimum_variance", "sample", protocol="expanding"),
)

PRIMARY = CELLS + PERTURBED
RUNS = PRIMARY + REPEATED

# The count the table is reported against. Stated as arithmetic so a reader can check it rather than
# count a table: sixteen cells, two perturbation runs, two repeats.
PRE_REGISTERED = len(CELLS) + len(PERTURBED) + len(REPEATED)


def validate():
    """The declarations checked against each other, before anything is run.

    A duplicated identifier would key two runs' manifests to one name; a run whose family or mean
    input is not in the modules' registries would fail halfway through a grid rather than at its
    start; and the pre-registered count is arithmetic that has to agree with the list it counts.
    """
    from ..construct.families import FAMILIES
    from ..construct.means import MEANS

    identifiers = [run["id"] for run in RUNS]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"the grid declares a repeated run id: {identifiers}")
    for run in RUNS:
        if run["family"] not in FAMILIES:
            raise ValueError(f"{run['id']} names a family the construct layer does not carry: {run['family']}")
        if run["mean"] is not None and run["mean"] not in MEANS:
            raise ValueError(f"{run['id']} names a mean input the construct layer does not carry: {run['mean']}")
        if run["estimator"] not in ("sample", "shrinkage", "factor"):
            raise ValueError(f"{run['id']} names an estimator the risk layer does not carry: {run['estimator']}")
        if run["protocol"] not in ("rolling", "expanding"):
            raise ValueError(f"{run['id']} names a protocol the grid does not run: {run['protocol']}")
        if run["settings"] is not None and run["base_setting"] not in run["settings"]:
            raise ValueError(f"{run['id']} declares a sensitivity run that excludes its own base case")
    if PRE_REGISTERED != 20:
        raise ValueError(f"the pre-registered count is {PRE_REGISTERED}, not the twenty the design fixed")
    by_id = {run["id"]: run for run in RUNS}
    for base_id, lifted_id in PERTURBATION_PAIRS.items():
        base, lifted = by_id[base_id], by_id.get(lifted_id)
        if lifted is None:
            raise ValueError(f"the perturbation pair names a run the grid does not declare: {lifted_id}")
        moved = {
            key: (base[key], lifted[key])
            for key in base
            if key not in ("id", "cap") and base[key] != lifted[key]
        }
        if moved or lifted["cap"] != UNCAPPED:
            raise ValueError(
                f"{lifted_id} is meant to lift the per-sleeve cap and change nothing else; it moves {moved} "
                f"and its cap is {lifted['cap']}"
            )
    return {"runs": identifiers, "cells": len(CELLS), "perturbed": len(PERTURBED), "repeated": len(REPEATED), "pre_registered": PRE_REGISTERED}
