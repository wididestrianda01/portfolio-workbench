"""The external leg: the factor archives, the FX translation and the risk-free splice."""

import pandas as pd
import pytest
from synthetic import _french_zip

from portfolio_workbench.data import external

MONTHLY_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]


def archive(tmp_path, months=("202001", "202002", "202003"), annual=True, name="Developed_5_Factors"):
    frame = pd.DataFrame(
        [[0.5, 0.1, -0.2, 0.0, 0.1]] * len(months),
        columns=MONTHLY_COLUMNS,
        index=pd.PeriodIndex(months, freq="M"),
    )
    path = tmp_path / f"{name}.zip"
    return _french_zip(
        path, f"{name}.csv", MONTHLY_COLUMNS, frame, "This file was created using the 20260912 database", annual=annual
    )


def test_parser_stops_at_the_annual_block(tmp_path):
    frame = external.parse_french_zip(archive(tmp_path, months=tuple(f"2020{m:02d}" for m in range(1, 13))))
    assert len(frame) == 12, "the annual rows below the monthly block must not be parsed"
    assert list(frame.columns) == MONTHLY_COLUMNS
    assert frame.iloc[0]["Mkt-RF"] == pytest.approx(0.005), "the file quotes percent"
    assert "20260912" in frame.attrs["vintage"]


def test_parser_refuses_an_archive_without_monthly_rows(tmp_path):
    path = archive(tmp_path, months=())
    with pytest.raises(ValueError, match="no monthly rows"):
        external.parse_french_zip(path)


def test_risk_free_is_accrued_not_compounded():
    daily = pd.Series(
        3.2, index=pd.date_range("2020-01-01", "2020-03-31", freq="D")
    )
    monthly = external.accrue_monthly(daily)
    assert monthly.index[0] == pd.Period("2020-01", freq="M")
    assert 0.002 < float(monthly.iloc[0]) < 0.004, "a 3.2% annual rate is about 27 bp a month"
    naive = external.naive_monthly(daily)
    assert naive > 0.9, "treating the annualised rate as a per-period rate gives an absurd month"


def test_splice_reports_the_basis_over_the_overlap():
    daily = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    eonia = pd.Series(3.200, index=daily)
    estr = pd.Series(3.115, index=daily)
    spliced, basis_bp, overlap = external.splice_risk_free(eonia, estr)
    assert basis_bp == pytest.approx(8.5, abs=1e-6)
    assert overlap == len(daily)
    assert float(spliced.loc["2020-06-01"]) == pytest.approx(3.115)
    assert float(spliced.loc["2019-06-01"]) == pytest.approx(3.200)


def test_translation_compounds_the_currency_leg():
    months = pd.PeriodIndex(["2020-01", "2020-02"], freq="M")
    factors = pd.DataFrame({"Mkt-RF": [0.02, 0.0]}, index=months)
    fx = pd.Series([0.01, 0.0], index=months)
    translated = external.eur_translate(factors, fx)
    assert float(translated["Mkt-RF"].iloc[0]) == pytest.approx(1.02 * 1.01 - 1)
    with pytest.raises(ValueError, match="does not cover"):
        external.eur_translate(factors, fx.iloc[:1])
