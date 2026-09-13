"""The coverage report: what the frozen panel actually holds.

Run from the repository root:

    python3 -m portfolio_workbench.data.coverage
    python3 -m portfolio_workbench.data.coverage .data/panel/2026-09-13

It writes nothing and asserts nothing. Its job is to put the panel's shape, its
warnings and the cost of the joins in front of a reader before any result is quoted.
"""

import sys

from . import loader, panel


def main(root=None):
    document = loader.load_panel(root)
    prices, fx = document["prices"], document["fx"]
    coverage = panel.coverage_months(prices)
    months = document["months"]

    print(f"[data] snapshot {document['snapshot_id']}, taken {document['as_of']}")
    print(f"[data] {len(prices)} rows over {len(coverage)} instruments, "
          f"{prices['period_month'].nunique()} months of history")
    print(f"[data] joined panel {months.min()} -> {months.max()} = {len(months)} months "
          f"x {prices['instrument'].nunique()} sleeves")
    if document["dropped"]["prices"]:
        print(f"[data] {document['dropped']['prices']} rows fell outside the declared window")

    print("[data] coverage per instrument:")
    for instrument, cover in coverage.items():
        events = int((prices.loc[prices["instrument"] == instrument, "dividend"].fillna(0.0) > 0).sum())
        gap = f", {len(cover['missing'])} missing" if cover["missing"] else ""
        print(f"    {instrument:<16s} {cover['first']} -> {cover['last']}  {cover['months']:>3d} months"
              f"{gap}  distributions {events}")

    print(f"[data] FX series: {sorted(fx['instrument'].unique())}")
    factors = document["factors"]
    print(f"[data] factor spine {factors['usd'].index.min()} -> {factors['usd'].index.max()} "
          f"({len(factors['usd'])} months, cols {list(factors['usd'].columns)}), quoted in USD, "
          f"translated to EUR as a separate frame")
    print(f"[data] factor vintage: {factors['vintage']!r}")
    if factors["untranslated"]:
        print(f"[data] factor months before the euro reference rate begins are not translated: "
              f"{factors['untranslated']} (outside the panel, so no return is affected)")
    risk_free = document["risk_free"]
    rf = risk_free["monthly"].reindex(months)
    print(f"[data] risk-free: overnight series spliced at the benchmark transition, overlap "
          f"{risk_free['overlap_days']} days at {risk_free['basis_bp']:+.1f} bp")
    print(f"[data] risk-free accrued daily at /360: mean {rf.mean():.4%}/month over the panel, "
          f"min {rf.min():.4%}, max {rf.max():.4%}")
    print(f"[data] the trap: compounding the annualised rate as a per-period rate gives "
          f"{risk_free['naive_monthly']:.0%} per month")
    for line in document["warnings"]:
        print(f"[data] {line}")
    return document


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
