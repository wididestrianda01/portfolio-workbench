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

from . import external, manifest, panel, quality, universe
from .universe import WINDOW_END

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


def load_factors(root, document):
    """The fundamental spine, quoted and euro-translated, with the FX level behind it.

    The verified manifest is passed in rather than re-derived: the whole snapshot is
    hashed on every load, and a leg that verified it again would hash every file once
    more for an answer its caller already holds.
    """
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
    fx_level = external.monthly_last(eurusd_daily)
    # The translation takes the currency leg as a RETURN, never as the quoted rate:
    # `monthly_last` returns the level a position is valued at, and a level reaching a
    # multiplicative translation scales every quote by the rate itself.
    fx_returns = external.monthly_returns(fx_level)
    # The ECB's dollar reference rate begins in 1999-01 and a return needs the month before
    # it, so the first euro month is untranslatable too; the factor library reaches back to
    # 1990-11. Nothing in the panel goes back that far, so the untranslatable months are
    # dropped and reported rather than filled with an assumed rate.
    covered = usd.index.intersection(fx_returns.index)
    untranslated = usd.index.difference(covered)
    return {
        "usd": usd,
        "eur": external.eur_translate(usd.loc[covered], fx_returns),
        "europe_usd": europe,
        "fx_level": fx_level,
        "untranslated": f"{untranslated.min()}..{untranslated.max()}" if len(untranslated) else None,
        "vintage": developed.attrs.get("vintage", "unknown"),
    }


def load_risk_free(root, document):
    """The overnight rate, spliced and accrued into monthly returns, with the basis."""
    eonia = external.read_ecb_csv(Path(root) / _role_path(document, "ecb_eonia"))
    estr = external.read_ecb_csv(Path(root) / _role_path(document, "ecb_estr"))
    daily, basis_bp, overlap = external.splice_risk_free(eonia, estr)
    return {
        "daily": daily,
        "monthly": external.accrue_monthly(daily),
        "basis_bp": basis_bp,
        "overlap_days": overlap,
    }


def _rates(fx):
    """EUR per unit of each currency the issuer facts are quoted in.

    Fund sizes arrive in the fund's own denomination while the liquidity floor is a euro
    figure, so they are converted at the snapshot's own newest month-end rate inside the
    declared window rather than at a rate assumed today. A currency the FX leg does not
    carry is absent from the table, and the line it belongs to is reported as unscreened.
    """
    if not len(fx):
        return {}
    month = fx.loc[fx["period_month"] <= pd.Period(WINDOW_END, freq="M").start_time, "period_month"].max()
    rates = {}
    for instrument, currency in universe.FX_QUOTES.items():
        rows = fx[(fx["instrument"] == instrument) & (fx["period_month"] == month)]
        if len(rows):
            rates[currency] = 1.0 / float(rows["close"].iloc[0])
    return rates


def load_panel(root=None, as_of=None):
    """Every leg, verified and gated, ready for the analytics.

    `as_of` is a reader's moment, and the legs come back as that reader could have seen
    them: the availability rule is enforced by the loader rather than left to every caller
    to re-apply, so there is no path into the package that skips it. The too-fresh stop is
    a different test and runs against the snapshot's own stamp, because a bar from a month
    that had not closed when the snapshot was taken is a fetch fault whoever is asking.

    Warnings travel with the panel rather than being printed here, so that whichever
    module reports a number prints the qualification beside that number.
    """
    root = snapshot_root(root)
    document = manifest.verify(root)
    taken = document["created"]

    prices = _load_role(root, document, "price", REQUIRED_PRICE_COLUMNS, "price")
    fx = _load_role(root, document, "fx", REQUIRED_FX_COLUMNS, "fx")
    # Read rather than defaulted: `verify` has refused any manifest without them, and an empty
    # mapping here would turn the dividend-blind stop and the liquidity floor into rules that
    # cannot fire, which is the failure the gate exists to prevent rather than to reproduce.
    facts = document["instruments"]
    factors = load_factors(root, document)
    risk_free = load_risk_free(root, document)

    # The gate runs on the panel as the snapshot holds it, before the declared window
    # trims anything: a bar that should never have been fetched is a fetch fault, and
    # trimming first would hide it behind the window.
    warnings = quality.run(prices, facts, taken, factor_months=factors["usd"].index, rates=_rates(fx))
    warnings += quality.run(fx, {}, taken)
    prices, dropped_prices = panel.window(prices)
    fx, dropped_fx = panel.window(fx)

    if as_of is not None:
        prices = panel.as_of(prices, as_of)
        fx = panel.as_of(fx, as_of)
        # The factor and rate legs carry no `available_from` column, so the same rule is
        # applied to their month labels. Every leg is gated, because a reader told that the
        # legs come back as they could have seen them has to be able to rely on it for the
        # leg they happen to read.
        kept = panel.available_months(factors["eur"].index, as_of)
        factors = {
            **factors,
            "usd": factors["usd"].loc[factors["usd"].index.intersection(kept)],
            "eur": factors["eur"].loc[factors["eur"].index.intersection(kept)],
            "europe_usd": factors["europe_usd"].loc[factors["europe_usd"].index.intersection(kept)],
            "fx_level": factors["fx_level"].loc[factors["fx_level"].index.intersection(kept)],
        }
        monthly = panel.available_months(risk_free["monthly"].index, as_of)
        risk_free = {
            **risk_free,
            "monthly": risk_free["monthly"].loc[monthly],
            "daily": risk_free["daily"][risk_free["daily"].index <= panel.cutoff(as_of)],
        }

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
        "as_of": as_of if as_of is not None else taken,
    }
