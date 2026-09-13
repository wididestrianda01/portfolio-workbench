"""The quality gate, rule by rule, on planted frames.

Each test names the rule it plants and asserts the rule fires. A frame is built by hand
rather than loaded from a snapshot so that the plant is the only thing wrong with it.
"""

import pandas as pd
import pytest

from portfolio_workbench.data import quality

FACTS = {
    "AAA": {"income_policy": "distributing (quarterly)", "fund_size": "EUR 0.80bn"},
    "BBB": {"income_policy": "accumulating", "fund_size": "EUR 4.00bn"},
}
MONTHS = pd.period_range("2020-01", "2020-06", freq="M")


def planted_frame():
    rows = []
    for instrument, price, dividend in (("AAA", 100.0, 0.25), ("BBB", 50.0, 0.0)):
        for month in MONTHS:
            rows.append(
                {
                    "instrument": instrument,
                    "date": month.to_timestamp(how="start"),
                    "period_month": month.to_timestamp(how="start"),
                    "available_from": (month + 1).to_timestamp(how="start"),
                    "close": price,
                    "adj_close": price,
                    "dividend": dividend,
                    "currency": "EUR",
                }
            )
    return pd.DataFrame(rows)


def fire(rule, frame):
    """The rule named by the plant, and only that rule."""
    if rule == "missing column":
        quality.stop_missing_columns(frame, ("date", "close", "adj_close", "dividend"), "planted.csv")
    elif rule == "malformed date":
        quality.stop_malformed_dates(frame)
    elif rule == "missing bar":
        quality.stop_missing_bars(frame)
    elif rule == "distributing line with no distribution":
        quality.stop_dividend_blind(frame, FACTS)
    else:
        quality.stop_implausible_prices(frame)


@pytest.mark.parametrize(
    "plant,rule",
    [
        (lambda f: f.drop(columns=["adj_close"]), "missing column"),
        (lambda f: f.assign(date=f["date"].mask(f.index == 0, pd.NaT)), "malformed date"),
        (lambda f: f.assign(date=f["date"].mask(f.index == 3, f["date"].iloc[3] + pd.Timedelta(days=9))), "malformed date"),
        (lambda f: f.drop(index=[7]), "missing bar"),
        (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "missing bar"),
        (lambda f: f.assign(dividend=f["dividend"].mask(f["instrument"] == "AAA", 0.0)), "distributing line with no distribution"),
        (lambda f: f.assign(close=f["close"].mask(f.index == 2, -1.0)), "implausible level"),
        (lambda f: f.assign(adj_close=f["adj_close"].mask(f.index == 4, 400.0)), "implausible move"),
    ],
)
def test_each_stop_fires_on_its_plant(plant, rule):
    with pytest.raises(quality.DataStop) as stop:
        fire(rule, plant(planted_frame()))
    assert stop.value.rule == rule


def test_the_plausibility_bound_follows_the_sleeve():
    """A 10% month is an ordinary month in real estate and a fault in cash, so the bound is
    read from the sleeve map instead of applied to every line alike."""
    frame = planted_frame()
    jumped = frame["adj_close"].to_numpy()
    jumped[3] = jumped[2] * 1.10

    with pytest.raises(quality.DataStop) as stop:
        quality.stop_implausible_prices(frame.assign(instrument="XEON.DE", adj_close=jumped))
    assert stop.value.rule == "implausible move"
    assert "cash sleeve" in stop.value.detail

    quality.stop_implausible_prices(frame.assign(adj_close=jumped))   # unnamed line: the fallback bound


def test_too_fresh_bar_needs_a_snapshot_time():
    frame = planted_frame()
    quality.stop_too_fresh(frame, as_of="2026-09-13")          # all bars long closed
    with pytest.raises(quality.DataStop) as stop:
        quality.stop_too_fresh(frame, as_of="2020-03-15")      # March 2020 has not closed
    assert stop.value.rule == "too-fresh bar"
    assert "2020-03" in stop.value.detail


def test_each_warning_fires_on_its_plant():
    frame = planted_frame()

    # AAA reports EUR 0.80bn against the floor, BBB reports EUR 4.00bn; the screen reads
    # the issuer's reported size rather than the price series, so neither line can pass
    # or fail it by accident.
    thin = quality.warn_thin_liquidity(frame, FACTS, {})
    assert any("thin liquidity" in line and "AAA" in line for line in thin)
    assert not any("BBB" in line for line in thin)

    # A size the parser cannot read, and a size in a currency the snapshot carries no rate
    # for, are both reported as unscreened rather than passing the floor unmeasured.
    assert quality.fund_size_eur("SEK 17,092m", {"SEK": 1 / 11.5}) == pytest.approx(1.486e9, rel=1e-3)
    assert quality.fund_size_eur("n/a", {}) is None
    assert quality.fund_size_eur("USD 2bn", {}) is None

    extra = frame.copy()
    events = (extra["instrument"] == "AAA") & (extra["period_month"].dt.month <= 5)
    extra.loc[events, "dividend"] = 0.5                        # monthly payer wearing a quarterly label
    assert any("extra distribution events" in line for line in quality.warn_extra_distributions(extra, FACTS))

    unexpected = frame.copy()
    february = (unexpected["instrument"] == "BBB") & (unexpected["period_month"] == pd.Timestamp("2020-02-01"))
    unexpected.loc[february, "dividend"] = 0.5
    assert any("unexpected distribution" in line for line in quality.warn_extra_distributions(unexpected, FACTS))

    # The feed's adjustment ignoring the distribution: adjusted close tracks price alone.
    blind = frame.assign(adj_close=frame["close"], dividend=frame["dividend"].mask(frame["instrument"] == "AAA", 2.0))
    assert any("recomputed-TR divergence" in line for line in quality.warn_tr_divergence(blind))
    assert quality.warn_tr_divergence(frame) == [], "a consistent fixture diverges"

    trim = quality.warn_factor_trim(
        pd.period_range("2020-01", "2020-06", freq="M"), pd.period_range("2020-01", "2020-04", freq="M")
    )
    assert len(trim) == 1 and "factor month trims the panel" in trim[0] and "2 price months" in trim[0]
