# Decision record

A learning exercise performed in role: a simulated mandate with no client and no institution. Nothing in this document is investment advice, a recommendation to any person, or a client communication, and the register is deliberate.

Snapshot `2026-09-13` - version 1.0 - written from 20 pre-registered runs over 16 distinct cells.

## 1. Recommendation

**Retain the policy benchmark and change nothing.** No cell cleared every rung of the declared ladder. The leader cleared the family-wise bar and failed the rank-retention rung (67.5% against the 80% floor), so its advantage is the ranking of this sample rather than a property of the method, and the sample's dispersion across cells is large enough that a choice made on it would be a choice about estimation error. The negative result is the finding.

## 2. Evidence, with the haircut applied

The bar over 16 cells is |z| >= 2.7344, which is also the multiple-testing haircut; at the 5.3 effective tests the realised correlation implies, it would be 2.3263, and the conservative value is the one applied. 9 cells are significantly behind the benchmark once cost is charged; 16 cleared the bar, of which 1 then failed the rank-retention rung and 2 are reported as different on the paired test with no expanding leg run for them; and 0 cleared every rung. The leader's rank retention across 2000 resamples is 67.5%. The haircut is self-imposed from the literature and is not a regulatory requirement.

## 3. Alternatives rejected

- Every Stage A family other than the retained benchmark: each either failed a rung of the ladder or sat significantly behind the benchmark after cost.
- The sample-mean input: its weight path moves most between adjacent months, which is the signature of estimation error rather than of a decision.
- The uncapped perturbations: they exist to show how much of a cell's result the constraint set was doing, and both concentrate the book onto the near-riskless sleeve.
- Vendor-model emulation, composite reporting, index construction and regulatory limits: refused structurally, with the reason recorded in the skills matrix.

## 4. Cost of the choice

The chosen option is the one already held, so its cost is the ongoing policy book's own: turnover 0.00% a year and no transaction charge, against the mean-input cells' 45.1% a year and 0.09% of return a year charged on traded notional. Not adopting also forgoes the sample's dispersion: the practical cost of this decision is that no evidence was found for changing the book, and the exercise publishes that rather than manufacturing a winner.

## 5. Limitations

One panel, eleven sleeves, 191 months of which 131 are traded, one mandate and one currency. Sixteen cells tested at a family-wise bar leave a resolution limit on every row. Cost is charged at a single per-side multiple with a sensitivity run beside it, and market impact and capacity are outside the panel. The multiple-testing correction is a self-imposed discipline. The exercise is a simulation and nothing here is advice or a client communication.

## 6. Falsification conditions

The recommendation is withdrawn or revisited if any of the following holds:

1. A rerun on this snapshot no longer reproduces the metric table within the stated tolerance (1e-06 relative).
2. A cell clears the family-wise bar, keeps its rank in at least 80% of bootstrap resamples, and holds its sign under the expanding protocol.
3. The resolution limit on a row is smaller than the difference the row reports as absent, and the difference remains absent on a longer panel.
4. The cost multiple is revised upward far enough that a cell's advantage at the sensitivity run disappears, and the sensitivity was the only thing supporting it.
5. The constraint set changes: a cap or band that binds on most steps makes a family's result a result about the constraints, and the perturbation runs exist to show how much.

Written against snapshot `2026-09-13`, build version 1.0.
