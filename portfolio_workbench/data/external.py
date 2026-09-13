"""External series: the central-bank risk-free chain, the FX leg and the factor archives.

Three traps live in this module and each cost an executed check to find, so they are
encoded as code with the reason beside them rather than left to the caller.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# The overnight benchmark migrated: EONIA is published up to the transition and the
# euro short-term rate takes over from that day. Splicing anywhere else mixes two
# rates with a structural spread between them.
RF_TRANSITION = "2019-10-01"
RF_SPREAD_BP = 8.5     # EONIA over the euro short-term rate, the published spread
ACCRUAL_DAYS = 360.0

FRENCH_QUOTE_CHARS = "latin-1"


def read_ecb_csv(path, column="OBS_VALUE"):
    """ECB Data Portal `csvdata`: one observation per date, no resampling."""
    frame = pd.read_csv(Path(path))
    if "TIME_PERIOD" not in frame.columns:
        raise ValueError(f"{Path(path).name}: no TIME_PERIOD column in the ECB export")
    frame["date"] = pd.to_datetime(frame["TIME_PERIOD"])
    series = frame.set_index("date")[column].astype(float).sort_index()
    return series[~series.index.duplicated(keep="first")]


def splice_risk_free(eonia, estr, transition=RF_TRANSITION):
    """EONIA before the transition, the euro short-term rate from it, and the spread
    between them measured over the overlap rather than assumed.

    The spread is a check, not a correction: both series are used as published, and the
    measured basis is reported so that a splice in the wrong place would show up as a
    step change in the monthly series rather than passing silently.
    """
    overlap = eonia.index.intersection(estr.index)
    if len(overlap) == 0:
        raise ValueError("the two overnight series do not overlap; the basis cannot be measured")
    basis_bp = float((eonia.reindex(overlap) - estr.reindex(overlap)).mean() * 100.0)
    parts = [eonia[eonia.index < pd.Timestamp(transition)], estr[estr.index >= pd.Timestamp(transition)]]
    joined = pd.concat(parts).sort_index()
    if joined.index.has_duplicates:
        joined = joined[~joined.index.duplicated(keep="last")]
    return joined, basis_bp, len(overlap)


def accrue_monthly(daily_percent):
    """Accrue an annualised overnight rate into monthly returns.

    The published figure is an ANNUALISED percentage. It is accrued over calendar days
    at /360 and compounded within the month, which is the only reading that matches the
    unit the publisher states. Compounding it as if it were a per-period rate produces a
    monthly risk-free return above 90% - the error class `naive_monthly` below exists to
    keep visible.
    """
    daily = daily_percent / 100.0
    monthly = (1.0 + daily / ACCRUAL_DAYS).resample("ME").prod() - 1.0
    monthly.index = monthly.index.to_period("M")
    return monthly


def naive_monthly(daily_percent, periods=21):
    """The counterfactual: the annualised rate treated as a daily rate.

    Kept as code so the report can print both numbers side by side; a reader who sees
    only the correct path has no way to tell a units error from a quiet market.
    """
    return float(np.prod(1.0 + daily_percent.to_numpy()[:periods] / 100.0) - 1.0)


def monthly_last(daily):
    """A daily level series reduced to month-end levels."""
    monthly = daily.resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    return monthly.dropna()


def parse_french_zip(path):
    """A Fama/French CSV archive, monthly rows only, in decimals.

    The files carry one provenance line, a header line whose first field is empty, the
    monthly rows, and then annual rows below them. The monthly and annual date fields are
    both numeric, so the parser cannot stop on a blank line: it stops on width. A
    six-digit first field is a month; the annual rows are four-digit and end the block.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0]).decode(FRENCH_QUOTE_CHARS)
    lines = raw.splitlines()
    if not lines:
        raise ValueError(f"{path.name}: empty archive member")
    vintage = lines[0].strip() if "created using" in lines[0] else "unknown"

    header_index = next((i for i, line in enumerate(lines) if line.strip().startswith(",")), None)
    if header_index is None:
        raise ValueError(f"{path.name}: no header line; the file shape has changed")
    header = ["date"] + [field.strip() for field in lines[header_index].split(",")[1:]]

    rows = []
    for line in lines[header_index + 1:]:
        first = line.split(",")[0].strip()
        if len(first) != 6 or not first.isdigit():
            break
        rows.append([field.strip() for field in line.split(",")][: len(header)])
    if not rows:
        raise ValueError(f"{path.name}: no monthly rows before the annual block")

    frame = pd.DataFrame(rows, columns=header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m")
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # -99.99 is the library's missing marker, not a return.
    frame = frame.replace(-99.99, np.nan).set_index("date").dropna(how="all")
    frame = frame / 100.0
    frame.index = frame.index.to_period("M")
    frame.attrs["vintage"] = vintage
    return frame


def eur_translate(factors, fx_monthly):
    """Translate factor returns quoted in a foreign currency into euro.

    The library's factor returns are USD. For a euro investor the translation is
    multiplicative: holding the foreign asset and the currency exposure compounds,
    it does not add. The translated block is kept as a separate, labelled frame so
    that a result cannot cite one and claim the other.
    """
    fx = fx_monthly.reindex(factors.index)
    if fx.isna().any():
        missing = factors.index[fx.isna()][:3].tolist()
        raise ValueError(f"the FX series does not cover the factor months; first gaps {missing}")
    return (1.0 + factors).mul(1.0 + fx, axis=0) - 1.0
