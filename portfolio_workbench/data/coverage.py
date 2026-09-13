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
    print(f"[data] joined panel {months.min()} → {months.max()} = {len(months)} months "
          f"× {prices['instrument'].nunique()} sleeves")
    if document["dropped"]["prices"]:
        print(f"[data] {document['dropped']['prices']} rows fell outside the declared window")

    print("[data] coverage per instrument:")
    for instrument, cover in coverage.items():
        events = int((prices.loc[prices["instrument"] == instrument, "dividend"].fillna(0.0) > 0).sum())
        gap = f", {len(cover['missing'])} missing" if cover["missing"] else ""
        print(f"    {instrument:<16s} {cover['first']} → {cover['last']}  {cover['months']:>3d} months"
              f"{gap}  distributions {events}")

    print(f"[data] FX series: {sorted(fx['instrument'].unique())}")
    factors = document["factors"]
    usd, eur = factors["usd"], factors["eur"]
    print(f"[data] factor spine {usd.index.min()} → {usd.index.max()} "
          f"({len(usd)} months, cols {list(usd.columns)}), quoted in USD, translated to EUR "
          f"as a separate frame")
    newest = eur.index.max()
    print(f"[data] the translated frame is not the quoted one: Mkt-RF {usd.loc[newest, 'Mkt-RF']:+.2%} "
          f"quoted against {eur.loc[newest, 'Mkt-RF']:+.2%} in EUR at a dollar rate of "
          f"{factors['fx_level'].loc[newest]:.4f}")
    print(f"[data] factor vintage: {factors['vintage']!r}")
    if factors["untranslated"]:
        print(f"[data] factor months with no euro leg are not translated: "
              f"{factors['untranslated']} (outside the panel, so no return is affected)")
    risk_free = document["risk_free"]
    rf = risk_free["monthly"].reindex(months)
    print(f"[data] risk-free: overnight series spliced at the benchmark transition, overlap "
          f"{risk_free['overlap_days']} days at {risk_free['basis_bp']:+.1f} bp")
    print(f"[data] risk-free accrued daily at /360: mean {rf.mean():.4%}/month over the panel, "
          f"min {rf.min():.4%}, max {rf.max():.4%}")
    for line in document["warnings"]:
        print(f"[data] {line}")
    return document


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
