# Decision record

A learning exercise performed in role: a simulated mandate with no client and no institution. Nothing in this document is investment advice, a recommendation to any person, or a client communication, and the register is deliberate.

Snapshot `2026-09-13`, version 1.2, written from 20 pre-registered runs over 16 distinct cells.

## 1. Recommendation

**Carry a mean input; do not claim a variant.** No cell cleared every rung of the declared ladder, so the ladder names no cell and this record names none. The leader cleared the family-wise bar, kept the sign of its own advantage in 95.9% of resamples and held it under the expanding protocol, and failed the rung that asks whether one cell is uniquely best (67.5% against the 80% floor). An advantage and a ranking are different findings, and this table measures both; the row says so in words that name the ranking.

What the same table does support is the axis those cells share. All 3 cells that carry a mean input clear the family-wise bar on their own row (4.05, 3.91, 3.37 against 2.9552), keeping the sign of the advantage in 96%, 95%, 90% of resamples, while `mean_variance_none`, on the same covariance and the same constraint set with no mean in it, clears nothing and returns -0.573. The recommendation is to move the objective to a mean-input mean-variance construction, and **not to nominate one of them**: the leading two sit 0.027 apart on the information ratio against a resolution of 0.51 on the leading row, so this panel supports the construction and not the variant. Hold it as a partial tilt - the active decision at a size that leaves the mandate's volatility where the mandate put it - rather than as a replacement of the strategic weights.

## 2. Evidence, with the haircut applied

The bar over 16 cells is |z| >= 2.9552, which is also the multiple-testing haircut; at the 5.3 effective tests the realised correlation implies, it would be 2.5758, and the conservative value is the one applied. 15 of the 20 runs clear the bar, so a difference is detectable on those rows; 8 of them are significantly behind the benchmark once the cost is charged, 0 lose the sign of its own advantage under the bootstrap, 1 fails the rank-retention rung that follows while keeping that sign, 2 have no expanding leg and cannot complete the ladder, and 0 clear every rung. The leader's rank retention across 2000 resamples is 67.5%, and the rung it fails is the one that asks whether one cell is uniquely best: the same row keeps the sign of its own advantage in 95.9% of the same resamples and holds that sign under the expanding protocol. The haircut is self-imposed from the literature and is not a regulatory requirement.

## 3. Alternatives rejected

- The risk-based families - minimum variance, maximum diversification, equal risk contribution, hierarchical risk parity, mean-CVaR and their shrinkage and factor variants: every one sits behind the policy benchmark on the sample, and 8 of them sit significantly behind it once the cost is charged. De-risking the sleeves against a benchmark that runs at 7.4% volatility is a policy bet rather than a construction choice.
- Dropping the mean input: `mean_variance_none` reproduces the equal-weight book on this constraint set and returns -0.573, which is the constraint set being the estimator rather than a result about means.
- The sample mean as a named choice over the shrunk and posterior ones: its weight path moves most between adjacent months, and the three sit within 0.152 of one another on the metric the decision turns on, so choosing between them is choosing the ranking.
- The uncapped perturbations: they exist to show how much of a cell's result the constraint set was doing, and both concentrate the book onto the near-riskless sleeve.
- Vendor-model emulation, composite reporting, index construction and regulatory limits: refused structurally, with the reason recorded in the skills matrix.

## 4. Cost of the choice

Adopting costs the family's own trading: 41.3% to 45.1% a year in one-way turnover and 0.08% to 0.09% of return a year at the decided 10 bp per side, charged on traded notional, and the advantage holds through the sensitivity run at four times that rate (5 bp +0.553, 20 bp +0.526, 40 bp +0.490). The books carrying a mean run at 10.4% to 11.7% volatility against the policy book's 7.4%, so the tilt is sized to buy the active decision rather than the volatility, and a partial tilt leaves the mandate's own volatility where the mandate put it. Not adopting forgoes the advantage those rows measure, which is +0.544, +0.517, +0.392 net of cost and is not zero. The family also moves further from the declared risk budget than the policy book already sits from it, which is the cost the sizing is against.

## 5. Limitations

One panel, 11 sleeves, 191 months of which 131 are traded, one mandate and one currency. 16 cells tested at a family-wise bar leave a resolution limit on every row. The expanding-window repeat is pre-registered for two cells of the sixteen, so the last rung could be reached by two rows and 2 rows carry the verdict that it was not run for them. The recommendation is made about an axis rather than about a cell, and the axis was read after the results were seen, which is why that step is stated here rather than left implicit in the evidence. Cost is charged at a single per-side multiple with a sensitivity run beside it, and market impact and capacity are outside the panel. The multiple-testing correction is a self-imposed discipline. The exercise is a simulation and nothing here is advice or a client communication.

## 6. Falsification conditions

The recommendation is withdrawn or revisited if any of the following holds:

1. A rerun on this snapshot no longer reproduces the metric table within the stated tolerance (1e-06 relative).
2. The cells carrying a mean stop clearing the family-wise bar, or stop retaining the sign of their advantage in at least 80% of bootstrap resamples, on this snapshot or on a longer panel. The recommendation rests on those rows and on nothing else.
3. The cap-binding frequency of those books rises far enough that the result becomes a result about the constraint set: the perturbation runs exist to measure how much of it already is.
4. The declared risk budget is treated as a constraint rather than as a report, and a mean-input book cannot be sized to leave the mandate's volatility where it was.
5. A cell clears the family-wise bar, keeps the sign of its own advantage in at least 80% of bootstrap resamples, holds that sign under the expanding protocol and keeps its rank in at least 80% of the same resamples: the ladder then names a cell, and this record names that cell rather than the axis.
6. The resolution limit on a row that reports no difference falls below the difference the row reports as absent on a longer panel: an absence then becomes evidence, and the risk-based families would have to be reread against the benchmark.

Written against snapshot `2026-09-13`, build version 1.2.
