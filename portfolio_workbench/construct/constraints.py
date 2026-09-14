"""The fixed constraint set, and the two rules that turn a target into a trade.

Every constructor in the package returns a **target** weight vector under the same constraint set:
long-only, fully invested, and no sleeve above the per-sleeve cap. The set is declared here, beside
the rules that enforce it, rather than in the universe module: a constraint parameterises a
construction and not a universe, and a bound held away from the code that applies it is a bound
that drifts from the code that assumes it.

Four things in here have a plausible wrong version, and each is stated because this package's cost
and turnover numbers are read off them.

**A cap is not a clip.** Clipping weights at the cap and renormalising is the obvious way to enforce
it and it is wrong twice over: the renormalisation can push a sleeve back over the cap, and it
leaves the vector summing to something other than one by whatever the clip removed. The projection
here solves the constrained problem instead - it clips, redistributes the excess among the sleeves
still below the cap, and repeats until nothing is left to give. The feasible set is empty when the
cap times the sleeve count is below one, and that is refused rather than returned as a vector that
cannot exist.

**The no-trade band bounds drift, not trading.** A sleeve is reverted to its held weight when its
target has moved less than the band, which stops the book churning on estimation noise. It does
**not** bound turnover: eleven sleeves each drifting just under a one-percent band can still trade
several percent in a month, and on this panel the minimum-variance cell traded more than twice the
turnover cap with the band in place. The cap therefore needs its own rule, and both are applied
here, band first.

**The turnover cap scales the trade vector.** The alternative - constrain the optimiser against the
current weights - changes each constructor's objective and so changes the method under comparison,
which is exactly what the fixed constraint set exists to prevent. Scaling lands the book on a convex
combination of where it is and where the banded target sits; both ends are feasible, so the point
between them is, and the scaling is reported as binding rather than applied quietly. The trade it
gives up is not lost: the same target is re-tested next month against the same cap.

**Cost is charged on traded notional, which is twice the one-way turnover.** The per-side rate
applies to what is actually traded, and one-way turnover counts only the buys (or only the sells) of
a book that stays fully invested, so it is half the traded notional. Writing the rate beside the
one-way figure instead halves every cost in the project, which is the error this module states the
convention to prevent.

The establishment trade is not a measured rebalance and neither rule applies to it. Every cell is
funded from cash at its own first target, which trades the whole book once; that cost is reported
as its own line, outside the measured window and outside the cap, because a rule that cannot be
satisfied at inception is not a rule.
"""

import numpy as np
import pandas as pd

# The per-sleeve cap, as a share of NAV. It is the constraint the two constraint-sensitive cells are
# re-run without, and it is what the projection below enforces.
CAP = 0.35
# The absolute no-trade band per sleeve, applied to the target-to-current weight change.
BAND = 0.01
# The one-way turnover a measured rebalance may not exceed, as a share of NAV per rebalance.
TURNOVER_CAP = 0.05
# Basis points per side, charged on traded notional. The sensitivity set is run beside the base.
COST_BP = 10.0
COST_SENSITIVITY_BP = (5.0, 20.0, 40.0)
# Accumulation tolerance for the projection: the constraint set is arithmetic, so the tolerance is
# for floating-point accumulation rather than for a modelling approximation.
PROJECTION_TOLERANCE = 1e-12
# The bound beyond which a finished projection is a failure rather than a rounding: it is looser than
# the iteration tolerance on purpose, because the loop stops after one sweep per sleeve and the vector
# it leaves is feasible to within accumulation. A vector outside this is not the constraint set's
# answer at any tolerance, and reading it as one would put an infeasible book into the weight path.
FEASIBILITY_TOLERANCE = 1e-9
# Reading a weight as sitting at the cap. The projected answer reaches the cap to within
# floating-point accumulation, never exactly, so a report that tested equality would print the cap
# as unbound on every step of a capped cell. It lives here rather than at the report because the
# number is a property of the constraint set, which is what CAP declares.
CAP_TOLERANCE = 1e-9


def bounded_simplex(weights, cap=CAP):
    """A weight vector projected onto the capped simplex: non-negative, summing to one, each at or
    below the cap.

    Clipping alone leaves the sum short, and renormalising the clip puts weight back on the sleeves
    that were just capped. The projection therefore clips, then hands what it removed to the sleeves
    still below the cap in proportion to the room they have left, and repeats: each round either
    finishes or fills at least one sleeve to the cap, so it terminates in at most one round per
    sleeve. Ties are broken by the vector's own order, so a rerun of a reported weight vector is the
    same vector.
    """
    values = np.asarray(weights, dtype=float).ravel()
    count = values.size
    if count == 0:
        raise ValueError("an empty weight vector has no projection")
    if cap * count < 1.0 - PROJECTION_TOLERANCE:
        raise ValueError(
            f"a cap of {cap:.4f} over {count} sleeves cannot be satisfied by a fully invested book: "
            f"the constraint set is empty and no weight vector reports it as feasible"
        )
    out = np.clip(values, 0.0, cap)
    for _ in range(count + 1):
        deficit = 1.0 - float(out.sum())
        if abs(deficit) <= PROJECTION_TOLERANCE:
            return out
        room = cap - out
        open_room = room > PROJECTION_TOLERANCE
        if not open_room.any():
            break
        share = room[open_room] / room[open_room].sum()
        if deficit > 0:
            out[open_room] += deficit * share
        else:
            # Over the target: take the excess from the sleeves that hold weight, in proportion to
            # what they hold, so a vector that sums high comes back inside the simplex too.
            held = out > 0
            out[held] += deficit * (out[held] / out[held].sum())
        out = np.clip(out, 0.0, cap)
    if abs(out.sum() - 1.0) > FEASIBILITY_TOLERANCE:
        raise ValueError(
            f"the projection ended at {out.sum():.12f} rather than one; the vector cannot be made "
            f"feasible under a cap of {cap:.4f}"
        )
    return out


def labelled(values, columns, cap=None):
    """A weight vector carrying its own sleeve labels, projected onto the constraint set.

    Every weight the package trades is indexed by the sleeve map, so a vector that lost its labels
    would misalign against the returns silently and produce a plausible result for the wrong
    instruments. The projection runs here rather than at each constructor's exit, so a constructor
    cannot hand back a vector the constraint set forbids.
    """
    columns = list(columns)
    values = np.asarray(values, dtype=float).ravel()
    if values.size != len(columns):
        raise ValueError(f"{values.size} weights cannot be labelled by {len(columns)} sleeves")
    limit = CAP if cap is None else cap
    return pd.Series(bounded_simplex(values, cap=limit), index=columns, name="weight")


def banded(target, current, band=BAND):
    """The target after the no-trade band: a sleeve whose target moved less than the band is left.

    The comparison is absolute in weight space rather than relative to the holding, because the band
    is declared as an absolute one percent per sleeve. A sleeve held at zero and targeted at zero
    does not move and is untouched, which is the case the band exists for.
    """
    target = pd.Series(target, dtype=float)
    current = pd.Series(current, dtype=float).reindex(target.index)
    if current.isna().any():
        raise ValueError("the current book does not cover every sleeve the target names")
    moved = (target - current).abs() > band
    return target.where(moved, current)


def trade(target, current, cap=TURNOVER_CAP):
    """The trade from the current book to a banded target, scaled to respect the turnover cap.

    One-way turnover is the buys (equivalently the sells) as a share of NAV, which for two books that
    both sum to one is half the traded notional. When that exceeds the cap the whole trade vector is
    scaled toward zero: the resulting book is a convex combination of the current book and the
    banded target, so it inherits both ends' feasibility - non-negative, fully invested, within the
    cap - and it is the smallest correction that satisfies the cap this month. The trade that is not
    taken is reported rather than silently dropped; the same target returns next month.
    """
    target = pd.Series(target, dtype=float)
    current = pd.Series(current, dtype=float).reindex(target.index)
    wanted = target - current
    one_way = float(wanted.abs().sum() / 2.0)
    binding = one_way > cap
    scale = cap / one_way if binding and one_way > 0 else 1.0
    traded = wanted * scale
    turnover = float(traded.abs().sum() / 2.0)
    return {
        "weights": current + traded,
        "trade": traded,
        "turnover": turnover,
        "binding": bool(binding),
        "untaken": float((wanted - traded).abs().sum() / 2.0),
    }


def cost(turnover, bp=COST_BP):
    """The cost of a rebalance, in return units, from its one-way turnover.

    `2 x one-way turnover x the per-side rate`: the rate is charged on traded notional, and one-way
    turnover is half of it.
    """
    return 2.0 * float(turnover) * float(bp) / 1e4


def rebalance(target, current, band=BAND, cap=TURNOVER_CAP, limit=CAP):
    """One measured rebalance: the band applied to the target, then the turnover cap to the trade.

    The band leaves the book short of one whenever it reverts a sleeve - the reverted deviations do
    not cancel - so the banded target is brought back inside the constraint set before it is traded,
    and the adjustment that does it is returned rather than absorbed. Untouched it would be the one
    place in the package where a book could be less than fully invested by accident, and the residual
    would silently sit outside every sleeve.
    """
    banded_target = banded(target, current, band=band)
    funded = pd.Series(bounded_simplex(banded_target.to_numpy(), cap=limit), index=banded_target.index)
    result = trade(funded, current, cap=cap)
    result["target"] = banded_target
    result["funding"] = float((funded - banded_target).abs().sum())
    return result


def establishment(current):
    """The cost of funding a book from cash, and the turnover that cost is charged on.

    A cell starts at its own first target rather than at the policy portfolio, so the book is bought
    once from cash: traded notional is the whole of NAV and one-way turnover is therefore one half,
    which is the same convention `cost` charges every later rebalance. It is reported per cell
    because the measured window contains no transition trade only if this line exists beside it.
    """
    held = pd.Series(current, dtype=float)
    turnover = float(held.abs().sum() / 2.0)
    return {"turnover": turnover, "cost": cost(turnover)}
