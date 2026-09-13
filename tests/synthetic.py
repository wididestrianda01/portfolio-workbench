"""A shape-matched synthetic snapshot.

The build is developed and gated against this before the licence-encumbered fetch, so
the data layer never blocks on the one act that carries a judgement. It is shape-matched
rather than realistic: four correlated equity sleeves, five bond sleeves of two distinct
volatilities, gold and cash, with dividends on the distributing lines only, so the block
structure of the covariance is the one the real panel has.

Every defect the quality gate catches has a plant here. A rule that no planted fixture
can trip is a rule that is not doing any work.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_workbench.data import manifest
from portfolio_workbench.data.universe import SLEEVES, TICKERS, WINDOW_END, WINDOW_START

SEED = 20260912
CREATED = "2026-09-13T09:00:00+00:00"

# The factor block spans more history than the price window and stops one month earlier,
# which is what the real library does; the join trims the difference.
FACTOR_START, FACTOR_END = "2008-01", "2026-07"
EQUITY_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MOMENTUM_FACTOR = ["WML"]

# The issuer facts as published, mirroring what the real manifest carries. The income
# policy is the field the dividend-blind stop reads, so it is declared rather than
# inferred from the series itself - inferring it would make the stop circular.
FACTS = {
    "IWDA.AS": ("IE00B4L5Y983", 0.0020, "accumulating", "physical (optimised)", "USD 152.9bn"),
    "IMEU.AS": ("IE00B4K48X80", 0.0012, "accumulating", "physical (optimised)", "n/a"),
    "XACT-NORDEN.ST": ("SE0001710914", 0.0016, "accumulating", "physical", "SEK 17,092m"),
    "IBGL.AS": ("IE00B1FZS913", 0.0015, "distributing (semi-annual)", "physical (sampled)", "EUR 0.94bn"),
    "IEGE.AS": ("IE00B3FH7618", 0.0007, "distributing (semi-annual)", "physical (sampled)", "EUR 1.30bn"),
    "IEAC.AS": ("IE00B3F81R35", 0.0009, "distributing (semi-annual)", "physical (sampled)", "EUR 13.2bn"),
    "IHYG.L": ("IE00B66F4759", 0.0050, "distributing (quarterly)", "physical (sampled)", "EUR 6.92bn"),
    "IBCI.AS": ("IE00B0M62X26", 0.0009, "accumulating", "physical (sampled)", "EUR 1.94bn"),
    "4GLD.DE": ("DE000A0S9GB0", 0.0, "no income", "physical metal", "n/a"),
    "IWDP.AS": ("IE00B1FZS350", 0.0059, "distributing (quarterly)", "physical", "USD 1.62bn"),
    "XEON.DE": ("LU0290358497", 0.0010, "accumulating", "synthetic (swap)", "n/a"),
}

BETA = {
    "equity_dev": (1.00, 0.00, 0.00),
    "equity_eu": (0.90, 0.00, 0.00),
    "equity_nordic": (0.85, 0.00, 0.00),
    "gov_long": (0.00, 1.60, 0.00),
    "gov_short": (0.00, 0.20, 0.00),
    "credit_ig": (0.05, 0.60, 1.00),
    "credit_hy": (0.35, 0.25, 1.55),
    "inflation_linked": (0.00, 0.90, 0.15),
    "gold": (0.05, 0.00, 0.00),
    "real_estate": (0.70, 0.25, 0.30),
    "cash": (0.00, 0.00, 0.00),
}
IDIO_VOL = {"gold": 0.045, "cash": 0.0006}
QUARTERLY = {"IWDP.AS", "IHYG.L"}
SEMI_ANNUAL = {"IBGL.AS", "IEGE.AS", "IEAC.AS"}

EONIA_LEVEL, ESTR_LEVEL = 3.200, 3.115     # the published EONIA/ESTR spread is 8.5 bp


def _csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def _french_zip(path, member, columns, frame, vintage, annual=True):
    """A Fama/French-shaped archive: provenance line, blank-led header, monthly rows,
    then an annual block that must not be parsed."""
    lines = [f"{vintage}, {len(frame)} monthly rows", "", "," + ",".join(columns)]
    for month, row in frame.iterrows():
        lines.append(f"{month.year}{month.month:02d}," + ",".join(f"{v:.2f}" for v in row))
    if annual:
        lines.append("")
        lines.append("," + ",".join(columns))
        for year, row in frame.groupby(frame.index.year).mean().iterrows():
            lines.append(f"{year}," + ",".join(f"{v * 12:.2f}" for v in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "\n".join(lines) + "\n")
    return path


def build(root, partial_month=False, late_start=None, dividend_blind=None, gap=None):
    """Write a complete frozen snapshot, with one plant per named defect."""
    root = Path(root)
    rng = np.random.default_rng(SEED)
    months = pd.period_range(WINDOW_START, WINDOW_END, freq="M")
    n = len(months)
    f_equity = rng.normal(0.006, 0.045, n)
    f_rates = rng.normal(0.001, 0.015, n)
    f_credit = rng.normal(0.001, 0.011, n)

    entries = []
    levels = {}
    for ticker in TICKERS:
        sleeve, currency = SLEEVES[ticker]
        beta = BETA[sleeve]
        returns = (
            beta[0] * f_equity
            + beta[1] * f_rates
            + beta[2] * f_credit
            + rng.normal(0, IDIO_VOL.get(sleeve, 0.012), n)
        )
        close = np.cumprod(1 + returns) * 100.0
        dividend = np.zeros(n)
        if ticker in QUARTERLY:
            quarterly = np.asarray(months.month % 3 == 0)
            dividend[quarterly] = close[quarterly] * 0.004
        if ticker in SEMI_ANNUAL:
            semi = np.asarray(np.isin(months.month, (6, 12)))
            dividend[semi] = close[semi] * 0.004
        if ticker == dividend_blind:
            dividend[:] = 0.0

        # The feed's adjusted close is reconstructed so that its month-on-month ratio is
        # exactly the total return; the recomputation check then agrees by construction and
        # any divergence it reports on real data is the feed's, not the fixture's.
        total = np.empty(n)
        total[0] = 0.0
        total[1:] = (close[1:] + dividend[1:]) / close[:-1] - 1.0
        adjusted = close[0] * np.cumprod(1.0 + total)

        frame = pd.DataFrame(
            {
                "date": months.to_timestamp(how="start").strftime("%Y-%m-%d"),
                "close": np.round(close, 4),
                "adj_close": np.round(adjusted, 4),
                "dividend": np.round(dividend, 4),
            }
        )
        if ticker == late_start:
            frame = frame.iloc[36:].reset_index(drop=True)
        if gap and gap[0] == ticker:
            month = pd.Period(gap[1], freq="M")
            keep = [i for i, m in enumerate(months) if m != month]
            frame = frame.iloc[keep].reset_index(drop=True)
        if partial_month:
            unended = pd.period_range(months[-1] + 1, months[-1] + 1, freq="M")
            extra = pd.DataFrame(
                {
                    "date": unended.to_timestamp(how="start").strftime("%Y-%m-%d"),
                    "close": [round(float(frame["close"].iloc[-1]) * 1.001, 4)],
                    "adj_close": [round(float(frame["adj_close"].iloc[-1]) * 1.001, 4)],
                    "dividend": [0.0],
                }
            )
            frame = pd.concat([frame, extra], ignore_index=True)

        relative = f"prices/{ticker}.csv"
        _csv(root / relative, ["date", "close", "adj_close", "dividend"], frame.to_numpy())
        entries.append(
            manifest.describe(
                root,
                relative,
                "price",
                "Yahoo Finance monthly bars (interval=1mo, auto_adjust=false, actions=true)",
                f"https://finance.yahoo.com/quote/{ticker}/history",
                CREATED,
                instrument=ticker,
                currency=currency,
            )
        )
        levels[ticker] = close[-1]

    for ticker, start in (("EURUSD=X", 1.05), ("EURSEK=X", 10.4)):
        walk = np.cumprod(1 + rng.normal(0, 0.012, n)) * start
        relative = f"fx/{ticker.replace('=', '_')}.csv"
        _csv(root / relative, ["date", "close"], np.column_stack([months.to_timestamp(how="start").strftime("%Y-%m-%d"), np.round(walk, 4)]))
        entries.append(
            manifest.describe(
                root, relative, "fx", "Yahoo Finance monthly bars", f"https://finance.yahoo.com/quote/{ticker}/history",
                CREATED, instrument=ticker, currency=ticker[-3:],
            )
        )

    daily = pd.date_range("2008-01-01", "2026-09-30", freq="D")
    for role, series, url in (
        ("ecb_eonia", pd.Series(EONIA_LEVEL, index=daily[daily < "2022-01-01"]),
         "https://data-api.ecb.europa.eu/service/data/EON/D.EONIA_TO.RATE?format=csvdata"),
        ("ecb_estr", pd.Series(ESTR_LEVEL, index=daily[daily >= "2019-10-01"]),
         "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata"),
        ("ecb_eurusd", pd.Series(1.10, index=daily),
         "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata"),
    ):
        relative = f"ecb/{role}.csv"
        rows = [(d.strftime("%Y-%m-%d"), f"{v:.4f}") for d, v in series.items()]
        _csv(root / relative, ["KEY", "TIME_PERIOD", "OBS_VALUE"], [("probe", d, v) for d, v in rows])
        entries.append(manifest.describe(root, relative, role, "ECB Data Portal csvdata", url, CREATED))

    factor_months = pd.period_range(FACTOR_START, FACTOR_END, freq="M")
    factor_rng = np.random.default_rng(SEED + 1)
    block = pd.DataFrame(
        {"Mkt-RF": factor_rng.normal(0.6, 4.4, len(factor_months))} | {
            column: factor_rng.normal(0.2, 2.2, len(factor_months)) for column in EQUITY_FACTORS[1:]
        },
        index=factor_months,
    )
    momentum = pd.DataFrame({"WML": factor_rng.normal(0.3, 3.1, len(factor_months))}, index=factor_months)
    for relative, member, frame, columns in (
        ("factors/Developed_5_Factors.zip", "Developed_5_Factors.csv", block, EQUITY_FACTORS),
        ("factors/Developed_MOM_Factor.zip", "Developed_MOM_Factor.csv", momentum, MOMENTUM_FACTOR),
        ("factors/Europe_5_Factors.zip", "Europe_5_Factors.csv", block, EQUITY_FACTORS),
    ):
        _french_zip(
            root / relative,
            member,
            columns,
            frame,
            "This file was created using the 20260912 database",
        )
        entries.append(
            manifest.describe(
                root, relative, "factor", "Kenneth R. French Data Library",
                f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{Path(relative).name}",
                CREATED,
            )
        )

    instruments = {
        ticker: {
            "isin": facts[0],
            "ter": facts[1],
            "income_policy": facts[2],
            "replication": facts[3],
            "fund_size": facts[4],
            "currency": SLEEVES[ticker][1],
        }
        for ticker, facts in FACTS.items()
    }
    manifest.build(root, entries, instruments, extra={"created": CREATED, "synthetic": True})
    return root
