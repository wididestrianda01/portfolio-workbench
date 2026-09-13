"""The constructed term and credit block, and the named set it forms with the spine."""

import pandas as pd
import pytest

from portfolio_workbench.factors import spine

MONTHS = pd.PeriodIndex(["2020-01", "2020-02"], freq="M")
SLEEVE_RETURNS = {
    "IBGL.AS": [0.04, -0.02],
    "IEGE.AS": [0.01, 0.01],
    "IEAC.AS": [0.05, -0.01],
    "IHYG.L": [0.09, 0.00],
}


def sleeves():
    return pd.DataFrame(SLEEVE_RETURNS, index=MONTHS)


def test_the_block_is_the_arithmetic_it_declares():
    """Hand-checked on round numbers, one row per series: a level that is the block's average, a
    slope that is long minus short, and two spreads that name the sleeves they read."""
    block = spine.constructed_block(sleeves())
    assert list(block.columns) == list(spine.CONSTRUCTED)
    assert float(block.loc[MONTHS[0], "government_level"]) == pytest.approx((0.04 + 0.01) / 2)
    assert float(block.loc[MONTHS[1], "term_slope"]) == pytest.approx(-0.02 - 0.01)
    assert float(block.loc[MONTHS[0], "credit"]) == pytest.approx(0.05 - 0.04)
    assert float(block.loc[MONTHS[1], "high_yield_excess"]) == pytest.approx(0.00 - (-0.01))


def test_a_sleeve_the_block_reads_and_the_frame_lacks_is_refused():
    with pytest.raises(ValueError, match="IHYG.L"):
        spine.constructed_block(sleeves().drop(columns=["IHYG.L"]))


def test_the_named_set_drops_the_files_own_risk_free_and_refuses_a_gap():
    """The published file carries the risk-free it was built against. This panel's excess returns
    are measured against the euro overnight rate, so the column is read and refused as a factor."""
    published = pd.DataFrame({"Mkt-RF": [0.01, 0.02], "RF": [0.001, 0.001]}, index=MONTHS)
    named = spine.named_set(published, spine.constructed_block(sleeves()))
    assert "RF" not in named.columns
    assert list(named.columns) == ["Mkt-RF", *spine.CONSTRUCTED]

    holed = published.copy()
    holed.loc[MONTHS[1], "Mkt-RF"] = float("nan")
    with pytest.raises(ValueError, match="missing factor value"):
        spine.named_set(holed, spine.constructed_block(sleeves()))
