"""The named factor set: a published spine, an internally constructed block, and the two
joined into the set every other factor result is measured against.

The spine is the Fama/French Developed five-factor model plus momentum, EUR-translated,
with the Europe set carried as a cross-check. It is equity-only by construction: the
library's bond, maturity and rating portfolio files no longer exist, so it cannot describe
this mandate on its own.

The non-equity half is therefore built here, from the panel's own sleeves, and declared
**constructed** rather than presented as a vendor factor. Constructed means each series is
a difference of the panel's own returns: nothing is estimated, and nothing is assumed
about a yield curve the snapshot does not carry. With two government sleeves the block's
level and slope are that block's sum and difference and nothing further is identifiable,
so these four series are the whole term/credit block rather than four choices among many.

The input is the excess-return frame, never a gross one. A difference of excess returns is
the difference of the gross returns, because the cash rate cancels exactly; only the level
carries the cash rate, and it should, since it stands in for the government block's own
excess return.
"""

import pandas as pd

from ..data import loader, panel, universe

LEVEL_SLEEVES = ("IBGL.AS", "IEGE.AS")
CONSTRUCTED = ("government_level", "term_slope", "credit", "high_yield_excess")

# The published file carries the risk-free rate it was built against as a column. The panel's
# excess returns are measured against the euro overnight rate, not the dollar one, so the
# column is read from the file and refused as a factor: a regression on it would put the
# dollar cash rate on the right-hand side of a euro left-hand side.
NOT_A_FACTOR = ("RF",)


def _need(returns, instruments):
    missing = [name for name in instruments if name not in returns.columns]
    if missing:
        raise ValueError(
            f"the constructed block reads {missing} and the frame does not carry them; the "
            f"block is built from this universe's own sleeves rather than from a vendor file"
        )


def constructed_block(returns):
    """The four constructed series, in the block's own column order.

    The level is the government block's common return, which for two sleeves is their
    average. The slope is long minus short, so a positive month is one in which the long
    end out-earned the short - the term premium with its sign stated rather than left to
    the reader. Credit is what investment grade pays over government, and high-yield excess
    is what high yield pays over investment grade: each names the two sleeves it reads, so
    a reader can check the construction against the universe rather than trusting a label.
    """
    _need(returns, LEVEL_SLEEVES + ("IEAC.AS", "IHYG.L"))
    return pd.DataFrame(
        {
            "government_level": 0.5 * (returns["IBGL.AS"] + returns["IEGE.AS"]),
            "term_slope": returns["IBGL.AS"] - returns["IEGE.AS"],
            "credit": returns["IEAC.AS"] - returns["IBGL.AS"],
            "high_yield_excess": returns["IHYG.L"] - returns["IEAC.AS"],
        }
    )


def named_set(spine, block):
    """The published spine joined with the constructed block, on the months both cover.

    An inner join: a month one leg cannot price has no row rather than a filled one, and a
    gap anywhere in the result is refused rather than carried into a regression window.
    """
    factors = spine[[column for column in spine.columns if column not in NOT_A_FACTOR]]
    joined = factors.join(block, how="inner")
    if joined.isna().to_numpy().any():
        raise ValueError("the joined named set carries a missing factor value; a filled one is an invention")
    return joined


def cross_check(document):
    """The Europe set, as the same five factors plus momentum, in its own quoted currency.

    It is the same library's regional cut of the same model, so a result that leans on the
    Developed set can be re-run against it. It is carried, never merged: two equity factor
    sets on the right-hand side of one regression would be near-collinear by construction.
    """
    europe = document["factors"]["europe_usd"]
    return europe[[column for column in europe.columns if column not in NOT_A_FACTOR]]


def main(root=None):
    document = loader.load_panel(root)
    returns = panel.eur_excess_returns(
        document["prices"], document["fx"], document["risk_free"]["monthly"]
    )
    # The sleeve order is the map's, and every weight vector in the package is indexed by
    # it; a frame that arrived in another order would misalign weights against returns
    # without raising anywhere downstream.
    if list(returns.columns) != universe.TICKERS:
        raise ValueError(f"the excess frame is not in the sleeve map's order: {list(returns.columns)}")

    block = constructed_block(returns)
    spine = document["factors"]["eur"]
    named = named_set(spine, block)

    print(f"[factor] snapshot {document['snapshot_id']}, sleeves in the map's order")
    print(f"[factor] sleeve frame {returns.shape[0]} months × {returns.shape[1]} sleeves, EUR total "
          f"returns less the overnight rate, {returns.index.min()}..{returns.index.max()}")
    print(f"[factor] published spine {spine.shape[0]} months × {spine.shape[1]} columns "
          f"({', '.join(spine.columns)}), EUR-translated; {', '.join(NOT_A_FACTOR)} is the file's own "
          f"risk-free and is dropped, not regressed on")
    print(f"[factor] the panel reaches {returns.index.min()}..{returns.index.max()}, so the named set "
          f"runs {named.index.min()}..{named.index.max()} = {len(named)} months × {named.shape[1]} factors")
    print(f"[factor] constructed block {len(block)} months × {block.shape[1]} series "
          f"({', '.join(block.columns)}), built from the panel's own sleeves")
    for column in CONSTRUCTED:
        series = block[column]
        print(f"    {column:<20s} mean {series.mean():+.3%}/month  vol {series.std():.2%}  "
              f"cumulative {float((1 + series).prod() - 1):+.1%}")
    print(f"[factor] named set {named.shape[0]} months × {named.shape[1]} factors, no missing value: "
          f"{', '.join(named.columns)}")
    print("[factor] the block is declared constructed: it is not vendor-supplied, and the spine that "
          "is published says nothing about bonds, credit or the term structure")
    for line in document["warnings"]:
        print(f"[factor] {line}")
    return {"returns": returns, "block": block, "spine": spine, "named": named, "document": document}


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
