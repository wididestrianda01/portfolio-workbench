"""The data rules: what the frozen panel is, and every rule that refuses it or qualifies it.

The eight stops and four warnings are planted one case each, because a rule with no fixture that
trips it is a rule that is not doing any work. The external legs are checked against how their
publishers actually write them, since three of this layer's traps were units or shapes rather than
prices.
"""

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import SEED, _french_zip, build

from portfolio_workbench.data import (
    external,
    loader,
    manifest,
    panel,
    quality,
    universe,
)

FACTS = {
    "AAA": {"income_policy": "distributing (quarterly)", "fund_size": "EUR 0.80bn"},
    "BBB": {"income_policy": "accumulating", "fund_size": "EUR 4.00bn"},
}
MONTHS = pd.period_range("2020-01", "2020-06", freq="M")


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One synthetic snapshot for the whole module: the plants that would change it get their own."""
    return build(tmp_path_factory.mktemp("snapshot") / "2026-09-13")


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


RULES = {
    "missing column": lambda frame: quality.stop_missing_columns(
        frame, ("date", "close", "adj_close", "dividend"), "planted.csv"
    ),
    "malformed date": quality.stop_malformed_dates,
    "missing bar": quality.stop_missing_bars,
    "distributing line with no distribution": lambda frame: quality.stop_dividend_blind(frame, FACTS),
    "implausible level": quality.stop_implausible_prices,
    "implausible move": quality.stop_implausible_prices,
}


def fire(rule, frame):
    """Run the rule a parametrised plant expects, keyed by the name the gate itself raises with.

    The key is that name, so the table below states each rule's identity once and the assertion on
    the raised rule is what couples the dispatch to the gate's vocabulary: a rule renamed in the
    gate fails here rather than leaving a plant that quietly dispatched on nothing.
    """
    RULES[rule](frame)


def snapshot_copy(frozen, tmp_path):
    """A byte copy of the frozen fixture, so a plant that doctors the snapshot leaves the original."""
    copy = tmp_path / "2026-09-13"
    copy.mkdir()
    for path in frozen.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(frozen)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return copy


def test_the_joined_panel_holds_the_declared_shape_and_the_as_of_rule(frozen):
    """The long table is the contract: one row per instrument-month, dated on the month's first day and
    available only from the next month's first day, joined across the three legs without padding. The
    loader is the only way in, so a bar the reader could not have seen is absent rather than refused by
    a later filter."""
    document = loader.load_panel(frozen)
    prices = document.prices

    assert prices["instrument"].nunique() == 11
    assert prices["period_month"].nunique() == 192
    assert len(document.months) == 191, "the factor leg costs one month"
    assert str(document.months.min()) == "2010-09"
    assert str(document.months.max()) == "2026-07"
    assert document.snapshot_id == "2026-09-13"
    assert prices["available_from"].dt.to_period("M").eq(prices["period_month"].dt.to_period("M") + 1).all()
    assert set(prices["currency"]) == {"EUR", "SEK"}

    view = loader.load_panel(frozen, as_of="2026-08-31")
    seen = set(view.prices["period_month"].dt.strftime("%Y-%m"))
    assert "2026-08" not in seen, "August's bar is not available during August"
    assert str(view.months.max()) == "2026-07"

    later = loader.load_panel(frozen, as_of="2026-09-01")
    assert "2026-08" in set(later.prices["period_month"].dt.strftime("%Y-%m"))

    # Every leg is gated, not only the price frame. The factor and rate legs carry no
    # `available_from` column, so a filter that only touched the panel would hand a reader
    # standing in 2016 the 2026 factor library and an accrued rate for a month that has not
    # happened - and the availability rule is documented as the only read path.
    earlier = loader.load_panel(frozen, as_of="2016-01-31")
    assert str(earlier.months.max()) == "2015-12"
    assert earlier.factors.usd.index.max() == pd.Period("2015-12", freq="M")
    assert earlier.factors.eur.index.max() == pd.Period("2015-12", freq="M")
    assert earlier.factors.europe_usd.index.max() == pd.Period("2015-12", freq="M")
    assert earlier.factors.fx_level.index.max() == pd.Period("2015-12", freq="M")
    assert earlier.risk_free.monthly.index.max() == pd.Period("2015-12", freq="M")
    assert earlier.risk_free.daily.index.max() <= pd.Timestamp("2016-01-31")


@pytest.mark.parametrize(
    "plant,rule",
    [
        (lambda f: f.drop(columns=["adj_close"]), "missing column"),
        (lambda f: f.assign(date=f["date"].mask(f.index == 0, pd.NaT)), "malformed date"),
        (
            lambda f: f.assign(date=f["date"].mask(f.index == 3, f["date"].iloc[3] + pd.Timedelta(days=9))),
            "malformed date",
        ),
        (lambda f: f.drop(index=[7]), "missing bar"),
        (lambda f: pd.concat([f, f.iloc[[0]]], ignore_index=True), "missing bar"),
        (
            lambda f: f.assign(dividend=f["dividend"].mask(f["instrument"] == "AAA", 0.0)),
            "distributing line with no distribution",
        ),
        (lambda f: f.assign(close=f["close"].mask(f.index == 2, -1.0)), "implausible level"),
        (lambda f: f.assign(adj_close=f["adj_close"].mask(f.index == 4, 400.0)), "implausible move"),
    ],
)
def test_each_frame_stop_fires_on_its_planted_frame(plant, rule):
    """Each of the eight frame rules raises with its own name, on a frame whose only fault is the plant."""
    with pytest.raises(quality.DataStop) as stop:
        fire(rule, plant(planted_frame()))
    assert stop.value.rule == rule


def test_the_plausibility_bound_follows_the_sleeve_and_a_fresh_bar_needs_the_snapshot_time():
    """A ten percent month is an ordinary month in real estate and a fault in cash, so the bound is read
    from the sleeve map rather than applied to every line alike; and a bar from a month that had not
    closed when the snapshot was taken is a fetch fault whoever is asking."""
    frame = planted_frame()
    jumped = frame["adj_close"].to_numpy()
    jumped[3] = jumped[2] * 1.10

    with pytest.raises(quality.DataStop) as stop:
        quality.stop_implausible_prices(frame.assign(instrument="XEON.DE", adj_close=jumped))
    assert stop.value.rule == "implausible move"
    assert "cash sleeve" in stop.value.detail
    quality.stop_implausible_prices(frame.assign(adj_close=jumped))          # unnamed: the fallback bound

    quality.stop_too_fresh(frame, as_of="2026-09-13")                        # every bar long closed
    with pytest.raises(quality.DataStop) as stop:
        quality.stop_too_fresh(frame, as_of="2020-03-15")
    assert stop.value.rule == "too-fresh bar"
    assert "2020-03" in stop.value.detail


@pytest.mark.parametrize(
    "kwargs,rule",
    [
        ({"late_start": "IWDP.AS"}, "line starting late"),
        ({"partial_month": True}, "too-fresh bar"),
        ({"gap": ("IEAC.AS", "2016-04")}, "missing bar"),
        ({"dividend_blind": "IBGL.AS"}, "distributing line with no distribution"),
    ],
)
def test_each_snapshot_stop_fires_on_its_planted_snapshot(tmp_path, kwargs, rule):
    """The four rules that are properties of the frozen files rather than of a frame: a line starting
    after the window opens, a bar whose month had not ended, a month missing inside a series, and a
    distributing line whose series carries no distribution at all."""
    root = build(tmp_path / "2026-09-13", **kwargs)
    with pytest.raises(quality.DataStop) as stop:
        loader.load_panel(root)
    assert stop.value.rule == rule


def test_the_manifest_refuses_a_tampered_snapshot(frozen, tmp_path):
    """A checksum, a row count and a missing file each stop the load, and a factor archive that no
    longer parses arrives in the same vocabulary rather than as a bare parse error."""
    copy = snapshot_copy(frozen, tmp_path)
    loader.load_panel(copy)

    tampered = json.loads((copy / "manifest.json").read_text())
    tampered["files"][0]["rows"] = tampered["files"][0]["rows"] - 1
    (copy / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(manifest.ManifestError, match="manifest says"):
        loader.load_panel(copy)

    (copy / "manifest.json").write_text((frozen / "manifest.json").read_text())
    victim = copy / "prices/IWDA.AS.csv"
    victim.write_text(victim.read_text() + "\n")
    with pytest.raises(manifest.ManifestError, match="sha256"):
        loader.load_panel(copy)

    (copy / "manifest.json").write_text((frozen / "manifest.json").read_text())
    (copy / "prices/IWDA.AS.csv").unlink()
    with pytest.raises(manifest.ManifestError, match="absent from the snapshot"):
        loader.load_panel(copy)

    empty = pd.DataFrame(columns=["Mkt-RF"], index=pd.PeriodIndex([], freq="M"))
    path = _french_zip(tmp_path / "Developed_5_Factors.zip", "Developed_5_Factors.csv", ["Mkt-RF"], empty, "vintage")
    with pytest.raises(manifest.ManifestError, match="no monthly rows"):
        manifest.measure(path)


def test_a_priced_line_the_manifest_does_not_describe_stops_the_gate():
    """The issuer facts are what the dividend-blind stop and the liquidity floor read, so an absent
    entry has to refuse the frame. Read as "does not distribute" it silences the first rule, and read
    as "not a fund" it silences the second, and both would then pass a panel neither had examined."""
    frame = planted_frame().assign(instrument="CCC")
    with pytest.raises(quality.DataStop) as stop:
        quality.stop_dividend_blind(frame, FACTS)
    assert stop.value.rule == "manifest mismatch"
    with pytest.raises(quality.DataStop) as stop:
        quality.warn_thin_liquidity(frame, FACTS, {})
    assert stop.value.rule == "manifest mismatch"


def test_every_month_labelled_leg_stops_at_the_declared_window(frozen):
    """The declared window is a rule for every leg, not only for the ones carrying a `period_month`
    column. The cash leg is accrued from whatever the rate publisher has published, which reaches into a
    month the panel's prices have not closed: the join hides that month rather than excluding it, and a
    leg the join hides is still a leg a report can print. The month is dropped, and the count of what the
    window cost is reported rather than absorbed."""
    document = loader.load_panel(frozen)
    end = pd.Period(universe.WINDOW_END, freq="M")
    assert document.risk_free.monthly.index.max() <= end
    assert document.factors.usd.index.max() <= end
    assert document.factors.eur.index.max() <= end
    assert document.factors.fx_level.index.max() <= end
    # The month is in the publisher's series and gone from the leg: the trim is the window's own rule
    # rather than the join's, so a report that prints the leg cannot print a month no book was held for.
    untrimmed = loader.load_risk_free(frozen, manifest.read(frozen))["monthly"]
    assert untrimmed.index.max() > end


def test_the_fx_legs_are_the_one_exemption_from_the_issuer_facts(frozen):
    """A level is not a fund, so the quoted-series map grants the exemption rather than the line's
    absence from the manifest doing it. Every priced line is described, which is what lets the gate
    read a policy and a size for each of them."""
    snapshot = manifest.read(frozen)
    priced = {entry["instrument"] for entry in snapshot["files"] if entry["role"] == "price"}
    assert priced == set(snapshot["instruments"]) == set(universe.TICKERS)
    assert not priced & set(universe.FX_QUOTES)
    assert quality.issuer_facts({}, "EURUSD=X") is None


def test_a_manifest_that_describes_no_issuer_refuses_the_snapshot(frozen, tmp_path):
    """The record's own completeness is checked with its checksums: a manifest carrying no issuer
    facts, and one carrying none for a single priced line, each stop the load rather than arriving at
    the gate as a rule that cannot fire."""
    copy = snapshot_copy(frozen, tmp_path)
    whole = manifest.read(frozen)

    without_block = {key: value for key, value in whole.items() if key != "instruments"}
    (copy / "manifest.json").write_text(json.dumps(without_block))
    with pytest.raises(manifest.ManifestError, match="carries no instruments"):
        loader.load_panel(copy)

    missing_line = json.loads(json.dumps(whole))
    del missing_line["instruments"]["IWDP.AS"]
    (copy / "manifest.json").write_text(json.dumps(missing_line))
    with pytest.raises(manifest.ManifestError, match="no issuer facts"):
        loader.load_panel(copy)


def test_each_warning_prints_beside_the_value_it_qualifies(frozen):
    """The four warnings, each on its own plant: the feed's adjustment diverging from a recomputation,
    a fund below the floor, an unexpected distribution event, and a factor month trimming the panel. A
    size the issuer does not publish is reported as unscreened rather than passing a floor it was never
    measured against."""
    frame = planted_frame()
    thin = quality.warn_thin_liquidity(frame, FACTS, {})
    assert any("thin liquidity" in line and "AAA" in line for line in thin)
    assert not any("BBB" in line for line in thin)
    assert quality.fund_size_eur("SEK 17,092m", {"SEK": 1 / 11.5}) == pytest.approx(1.486e9, rel=1e-3)
    assert quality.fund_size_eur("n/a", {}) is None
    assert quality.fund_size_eur("USD 2bn", {}) is None

    extra = frame.copy()
    events = (extra["instrument"] == "AAA") & (extra["period_month"].dt.month <= 5)
    extra.loc[events, "dividend"] = 0.5
    assert any("extra distribution events" in line for line in quality.warn_extra_distributions(extra, FACTS))

    unexpected = frame.copy()
    february = (unexpected["instrument"] == "BBB") & (unexpected["period_month"] == pd.Timestamp("2020-02-01"))
    unexpected.loc[february, "dividend"] = 0.5
    assert any("unexpected distribution" in line for line in quality.warn_extra_distributions(unexpected, FACTS))

    blind = frame.assign(adj_close=frame["close"], dividend=frame["dividend"].mask(frame["instrument"] == "AAA", 2.0))
    assert any("recomputed-TR divergence" in line for line in quality.warn_tr_divergence(blind))
    assert quality.warn_tr_divergence(frame) == [], "a consistent fixture diverges"

    trim = quality.warn_factor_trim(
        pd.period_range("2020-01", "2020-06", freq="M"), pd.period_range("2020-01", "2020-04", freq="M")
    )
    assert len(trim) == 1 and "factor month trims the panel" in trim[0] and "2 price months" in trim[0]

    warnings = loader.load_panel(frozen).warnings
    assert any("thin liquidity" in line and "IBGL.AS" in line for line in warnings)
    assert any("liquidity not screened" in line and "IMEU.AS" in line for line in warnings)


def test_the_manifest_records_the_vintage_and_the_currency_of_every_leg(frozen, tmp_path):
    """The snapshot states what it is keyed to: the factor library's vintage stamp is recorded rather
    than recovered by each load, and every leg carries the currency its instrument is quoted in,
    which is the label another project reads to satisfy the table contract. Both are re-derived on
    verify, so a manifest that overstates either is refused rather than believed."""
    snapshot = manifest.read(frozen)
    factors = [entry for entry in snapshot["files"] if entry["role"] == "factor"]
    assert factors and all(str(SEED) in entry["vintage"] for entry in factors), "the stamp is recorded"
    assert {e["instrument"]: e["currency"] for e in snapshot["files"] if e["role"] == "fx"} == universe.FX_QUOTES

    copy = snapshot_copy(frozen, tmp_path)
    loader.load_panel(copy)

    doctored = json.loads((copy / "manifest.json").read_text())
    next(entry for entry in doctored["files"] if entry["role"] == "factor")["vintage"] = "19990101"
    (copy / "manifest.json").write_text(json.dumps(doctored))
    with pytest.raises(manifest.ManifestError, match="vintage"):
        loader.load_panel(copy)


def test_the_external_legs_are_read_as_the_publishers_write_them(frozen, tmp_path):
    """Four traps this layer cost, each exercised rather than documented: the French archive carries an
    annual block below the monthly rows, the ECB rate is annualised so compounding it per period gives
    an absurd month, the two overnight series are spliced at the transition rather than anywhere, and
    the currency leg divides rather than multiplies. On the fixture the reference rate is flat, so the
    translated spine has to equal the quoted one exactly: a currency level reaching the translation
    instead scales every quote by the rate itself, which reads as a frame of hundred-percent months."""
    factors = loader.load_panel(frozen).factors
    assert factors.fx_level.nunique() == 1, "the fixture holds the reference rate flat"
    assert len(factors.eur) > 100
    pd.testing.assert_frame_equal(factors.eur, factors.usd.loc[factors.eur.index])

    frame = pd.DataFrame(
        [[0.5, 0.1, -0.2, 0.0, 0.1]] * 12,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        index=pd.period_range("2020-01", periods=12, freq="M"),
    )
    archive = _french_zip(
        tmp_path / "Developed_5_Factors.zip",
        "Developed_5_Factors.csv",
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        frame,
        "This file was created using the 20260912 database",
    )
    parsed = external.parse_french_zip(archive)
    assert len(parsed) == 12, "the annual rows below the monthly block must not be parsed"
    assert parsed.iloc[0]["Mkt-RF"] == pytest.approx(0.005), "the file quotes percent"
    assert "20260912" in parsed.attrs["vintage"]

    empty = _french_zip(tmp_path / "Empty.zip", "Empty.csv", ["Mkt-RF"], frame.iloc[:0], "vintage")
    with pytest.raises(external.SourceFormatError, match="no monthly rows"):
        external.parse_french_zip(empty)

    daily = pd.Series(3.2, index=pd.date_range("2020-01-01", "2020-03-31", freq="D"))
    monthly = external.accrue_monthly(daily)
    assert monthly.index[0] == pd.Period("2020-01", freq="M")
    assert 0.002 < float(monthly.iloc[0]) < 0.004, "a 3.2% annual rate is about 27 bp a month"
    naive = float(np.prod(1.0 + daily.to_numpy()[:21] / 100.0) - 1)
    assert naive > 0.9, "the annualised rate read as a per-period rate gives an absurd month"

    # The denominator, as a bound rather than a citation: accruing at /360 rather than /365 moves a
    # year's accrual by 365/360 - 1, about 1.4% of the rate - 4.5 bp at a 3.2% rate, under 2 bp at
    # the panel's own average. Overnight euro rates are quoted ACT/360, and this asserts the size of
    # the choice so that changing the denominator cannot pass as a rounding difference.
    year_360 = float(np.prod(1.0 + external.accrue_monthly(
        pd.Series(3.2, index=pd.date_range("2021-01-01", "2021-12-31", freq="D"))
    ).to_numpy()) - 1)
    year_365 = float((1.0 + 0.032 / 365) ** 365 - 1)
    assert 0.0004 < year_360 - year_365 < 0.0006

    two_years = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    spliced, basis_bp, overlap = external.splice_risk_free(pd.Series(3.200, index=two_years), pd.Series(3.115, index=two_years))
    assert basis_bp == pytest.approx(8.5, abs=1e-6)
    assert overlap == len(two_years)
    assert float(spliced.loc["2020-06-01"]) == pytest.approx(3.115)
    assert float(spliced.loc["2019-06-01"]) == pytest.approx(3.200)

    months = pd.PeriodIndex(["2020-01", "2020-02"], freq="M")
    quoted = pd.DataFrame({"Mkt-RF": [0.02, 0.0]}, index=months)
    fx = pd.Series([0.01, 0.0], index=months)
    translated = external.eur_translate(quoted, fx)
    assert float(translated["Mkt-RF"].iloc[0]) == pytest.approx(1.02 / 1.01 - 1)
    assert float(translated["Mkt-RF"].iloc[0]) < 0.02, "a euro that buys more dollars leaves less return"
    with pytest.raises(ValueError, match="does not cover"):
        external.eur_translate(quoted, fx.iloc[:1])

    levels = pd.Series([1.10, 1.16, 1.13], index=pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M"))
    returns = external.monthly_returns(levels)
    assert len(returns) == 2, "the first month has no predecessor and is dropped, not filled"
    assert float(returns.iloc[0]) == pytest.approx(1.16 / 1.10 - 1)


def test_the_excess_frame_translates_the_foreign_line_and_subtracts_the_cash_rate(frozen):
    """Hand-rebuilt from the same snapshot: a euro line is its own total return less the overnight rate,
    the one line quoted in another currency is translated by dividing by its own currency leg, and the
    frame comes back in the sleeve map's order, which every weight vector is indexed by."""
    document = loader.load_panel(frozen)
    prices, fx = document.prices, document.fx
    excess = panel.eur_excess_returns(prices, fx, document.risk_free.monthly)

    local = panel.total_return(panel.wide(prices, "adj_close")).iloc[1:]
    fx_returns = panel.total_return(panel.wide(fx, "close")).iloc[1:]
    translated = local.copy()
    translated["XACT-NORDEN.ST"] = (1.0 + local["XACT-NORDEN.ST"]) / (1.0 + fx_returns["EURSEK=X"]) - 1.0
    rate = document.risk_free.monthly
    rate.index = rate.index.to_timestamp(how="start")
    expected = translated.sub(rate.reindex(local.index), axis=0)
    expected.index = pd.PeriodIndex(expected.index, freq="M")
    untranslated = local.sub(rate.reindex(local.index), axis=0)
    untranslated.index = expected.index

    assert list(excess.columns) == universe.TICKERS
    assert not excess.isna().to_numpy().any()
    assert str(excess.index[0]) == "2010-10", "the panel's first bar has no predecessor to divide by"
    assert str(excess.index[-1]) == "2026-08"
    pd.testing.assert_frame_equal(excess, expected[list(universe.TICKERS)])
    assert not np.allclose(excess["XACT-NORDEN.ST"], untranslated["XACT-NORDEN.ST"], atol=1e-9)
