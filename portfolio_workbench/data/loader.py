"""Reading the frozen snapshot.

The loader is the only way into the panel: it verifies the manifest, parses each file
into the long table the rest of the package consumes, runs the quality gate, and hands
back the joined legs. Anything reading a snapshot file directly bypasses the check,
which is why the fetch script writes its manifest through `manifest.describe` rather
than by hand.

The long table is the data contract: one row per instrument-month, carrying
`period_month`, `available_from`, the unadjusted close, the feed's adjusted close and the
distribution. It is what another project would have to satisfy to consume this engine.
"""

import os
from pathlib import Path

import pandas as pd

from . import external, manifest, panel, quality

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent / ".data" / "panel"
ENV_VAR = "WORKBENCH_SNAPSHOT"

REQUIRED_PRICE_COLUMNS = ("date", "close", "adj_close", "dividend")
REQUIRED_FX_COLUMNS = ("date", "close")

LONG_COLUMNS = ("instrument", "period_month", "available_from", "close", "adj_close", "dividend", "currency")


def snapshot_root(root=None):
    """The dated snapshot directory. An explicit path wins, then the environment, then the
    newest dated directory on disk - never a merge of two, because a panel assembled from
    two retrievals has no single snapshot id to key its results to."""
    if root is not None:
        return Path(root)
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    if not DEFAULT_ROOT.exists():
        raise FileNotFoundError(f"no frozen snapshot under {DEFAULT_ROOT}; set {ENV_VAR}")
    dated = sorted(path for path in DEFAULT_ROOT.iterdir() if path.is_dir())
    if not dated:
        raise FileNotFoundError(f"{DEFAULT_ROOT} holds no dated snapshot directory")
    return dated[-1]


def _read_frame(root, entry, required, kind):
    path = Path(root) / entry["path"]
    frame = pd.read_csv(path)
    quality.stop_missing_columns(frame, required, path.name)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["instrument"] = entry["instrument"]
    frame["currency"] = entry.get("currency", "EUR")
    if "dividend" not in frame.columns:
        frame["dividend"] = 0.0
    if "adj_close" not in frame.columns:
        # An FX series is a level, not a priced asset: it pays nothing, so its adjusted
        # close is its close and it carries the same table shape as a priced line.
        frame["adj_close"] = frame["close"]
    frame = panel.add_availability(frame)
    quality.stop_malformed_dates(frame, path.name)
    frame["kind"] = kind
    return frame[list(LONG_COLUMNS) + ["kind"]]


def _load_role(root, document, role, required, kind):
    frames = [
        _read_frame(root, entry, required, kind) for entry in document["files"] if entry.get("role") == role
    ]
    if not frames:
        raise quality.DataStop("missing file", f"the snapshot carries no file with role '{role}'")
    return pd.concat(frames, ignore_index=True)


def _role_path(document, role):
    for entry in document["files"]:
        if entry.get("role") == role:
            return entry["path"]
    raise quality.DataStop("missing file", f"the snapshot carries no file with role '{role}'")


def load_factors(root):
    """The fundamental spine, quoted and euro-translated, with the FX level behind it."""
    document = manifest.verify(root)
    frames = {
        entry["path"]: Path(root) / entry["path"]
        for entry in document["files"]
        if entry.get("role") == "factor"
    }
    if len(frames) < 2:
        raise quality.DataStop("missing file", "the snapshot carries fewer than two factor archives")

    def by_name(fragment):
        matches = [path for name, path in frames.items() if fragment in name]
        if not matches:
            raise quality.DataStop("missing file", f"no factor archive matching '{fragment}'")
        return external.parse_french_zip(matches[0])

    developed = by_name("Developed_5_Factors")
    momentum = by_name("Developed_MOM_Factor")
    europe = by_name("Europe_5_Factors")
    if "WML" in momentum.columns:
        momentum = momentum.rename(columns={"WML": "MOM"})
    usd = developed.join(momentum, how="inner")

    eurusd_daily = external.read_ecb_csv(Path(root) / _role_path(document, "ecb_eurusd"))
    fx_monthly = external.monthly_last(eurusd_daily)
    # The euro reference rate begins in 1990-11, before which the factor library still
    # publishes. Nothing in the panel reaches back that far, so the untranslatable months
    # are dropped and reported rather than filled with an assumed rate.
    covered = usd.index.intersection(fx_monthly.index)
    untranslated = usd.index.difference(covered)
    return {
        "usd": usd,
        "eur": external.eur_translate(usd.loc[covered], fx_monthly),
        "europe_usd": europe,
        "fx_level": fx_monthly,
        "untranslated": f"{untranslated.min()}..{untranslated.max()}" if len(untranslated) else None,
        "vintage": developed.attrs.get("vintage", "unknown"),
    }


def load_risk_free(root):
    """The overnight rate, spliced and accrued into monthly returns, with the basis."""
    document = manifest.verify(root)
    eonia = external.read_ecb_csv(Path(root) / _role_path(document, "ecb_eonia"))
    estr = external.read_ecb_csv(Path(root) / _role_path(document, "ecb_estr"))
    daily, basis_bp, overlap = external.splice_risk_free(eonia, estr)
    return {
        "daily": daily,
        "monthly": external.accrue_monthly(daily),
        "basis_bp": basis_bp,
        "overlap_days": overlap,
        "naive_monthly": external.naive_monthly(daily),
    }


def load_panel(root=None, as_of=None):
    """Every leg, verified and gated, ready for the analytics.

    Warnings travel with the panel rather than being printed here, so that whichever
    module reports a number prints the qualification beside that number.
    """
    root = snapshot_root(root)
    document = manifest.verify(root)
    as_of = as_of or document["created"]

    prices = _load_role(root, document, "price", REQUIRED_PRICE_COLUMNS, "price")
    fx = _load_role(root, document, "fx", REQUIRED_FX_COLUMNS, "fx")
    facts = document.get("instruments", {})
    factors = load_factors(root)
    risk_free = load_risk_free(root)

    # The gate runs on the panel as the snapshot holds it, before the declared window
    # trims anything: a bar that should never have been fetched is a fetch fault, and
    # trimming first would hide it behind the window.
    warnings = quality.run(prices, facts, as_of, factor_months=factors["usd"].index)
    warnings += quality.run(fx, {}, as_of)
    prices, dropped_prices = panel.window(prices)
    fx, dropped_fx = panel.window(fx)

    months = panel.joined_months(
        prices["period_month"].drop_duplicates(),
        factors["eur"].index,
        risk_free["monthly"].index,
    )
    return {
        "snapshot_id": document["snapshot_id"],
        "manifest": document,
        "facts": facts,
        "prices": prices,
        "fx": fx,
        "factors": factors,
        "risk_free": risk_free,
        "months": months,
        "warnings": warnings,
        "dropped": {"prices": dropped_prices, "fx": dropped_fx},
        "as_of": as_of,
    }
