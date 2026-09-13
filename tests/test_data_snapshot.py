"""The snapshot contract: what the panel is, and what the loader refuses to accept."""

import json

import pandas as pd
import pytest
from synthetic import build

from portfolio_workbench.data import loader, manifest, panel, quality


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    return build(tmp_path_factory.mktemp("snapshot") / "2026-09-13")


def test_panel_shape_and_join(frozen):
    document = loader.load_panel(frozen)
    prices = document["prices"]

    assert prices["instrument"].nunique() == 11
    assert prices["period_month"].nunique() == 192, "the price window is 192 completed months"
    assert len(document["months"]) == 191, "the factor leg costs one month"
    assert str(document["months"].min()) == "2010-09"
    assert str(document["months"].max()) == "2026-07"
    assert document["snapshot_id"] == "2026-09-13"

    # The long table is the contract: every row is one instrument-month, dated on the
    # month's first day and available only from the next month's first day.
    assert (prices["available_from"].dt.day == 1).all()
    assert (prices["available_from"] > prices["period_month"]).all()
    assert prices["available_from"].dt.to_period("M").eq(prices["period_month"].dt.to_period("M") + 1).all()
    assert set(prices["currency"]) == {"EUR", "SEK"}


def test_as_of_rule_hides_the_month_that_has_not_closed(frozen):
    prices = loader.load_panel(frozen)["prices"]
    seen = lambda when: set(pd.PeriodIndex(panel.as_of(prices, when)["period_month"], freq="M").astype(str))

    assert "2026-08" not in seen("2026-08-31"), "August's bar is not available during August"
    assert "2026-08" in seen("2026-09-01"), "it becomes available on the first day of September"


def test_manifest_mismatch_is_refused(frozen, tmp_path):
    copy = tmp_path / "2026-09-13"
    copy.mkdir()
    for path in frozen.rglob("*"):
        if path.is_file():
            target = copy / path.relative_to(frozen)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    loader.load_panel(copy)   # the copy matches its manifest, so it loads

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


def test_late_starting_line_is_refused(tmp_path):
    root = build(tmp_path / "2026-09-13", late_start="IWDP.AS")
    with pytest.raises(quality.DataStop) as stop:
        loader.load_panel(root)
    assert stop.value.rule == "line starting late"
    assert "IWDP.AS" in stop.value.detail
    assert "excluding them" in stop.value.detail


def test_bar_from_an_unended_month_is_refused(tmp_path):
    root = build(tmp_path / "2026-09-13", partial_month=True)
    with pytest.raises(quality.DataStop) as stop:
        loader.load_panel(root)
    assert stop.value.rule == "too-fresh bar"
    assert "2026-09" in stop.value.detail


def test_missing_bar_is_refused(tmp_path):
    root = build(tmp_path / "2026-09-13", gap=("IEAC.AS", "2016-04"))
    with pytest.raises(quality.DataStop) as stop:
        loader.load_panel(root)
    assert stop.value.rule == "missing bar"
    assert "2016-04" in stop.value.detail
