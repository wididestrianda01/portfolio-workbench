"""One end-to-end run on the frozen snapshot, asserting the structure the layer rests on.

This is the fixture the rest of the suite extends: the structural checks a reader is asked to
take on trust are asserted here rather than left in prose, so a change that breaks one of them
fails one command.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.data import loader, panel, universe
from portfolio_workbench.factors import components as cp
from portfolio_workbench.factors import exposures, spanning
from portfolio_workbench.factors import spine as spine_module
from portfolio_workbench.risk import covariance


@pytest.fixture(scope="module")
def run():
    document = loader.load_panel()
    months = document["months"]
    returns = panel.eur_excess_returns(document["prices"], document["fx"], document["risk_free"]["monthly"])
    block = spine_module.constructed_block(returns)
    named = spine_module.named_set(document["factors"]["eur"], block)
    return {
        "document": document,
        "months": months,
        "returns": returns,
        "block": block,
        "named": named,
        "fit": exposures.rolling(returns, named, months),
    }


def test_the_factor_layer_runs_end_to_end_on_the_frozen_snapshot(run):
    returns, named, months = run["returns"], run["named"], run["months"]
    fit = run["fit"]

    # The data contract: the joined panel that every result is keyed to.
    assert len(months) == 191 and str(months.min()) == "2010-09" and str(months.max()) == "2026-07"
    assert list(returns.columns) == universe.TICKERS
    assert not returns.isna().to_numpy().any()

    # The named set: the published spine, the constructed block, no gap, and the file's own risk-free
    # refused as a factor.
    assert "RF" not in named.columns
    assert len(named) == 190 and len(named.columns) == 10
    assert not named.isna().to_numpy().any()

    # The walk-forward: the decided out-of-sample calendar, one observation short only in the first
    # window, and every sleeve fitted on the design it admits.
    traded = fit["traded"]
    assert len(traded) == 131 and traded[0] == pd.Period("2015-09", freq="M")
    assert traded[-1] == pd.Period("2026-07", freq="M")
    assert fit["n_obs"][0] == exposures.MIN_OBS and set(fit["n_obs"][1:]) == {exposures.WINDOW}

    # The four constructed sleeves are the block, so their block loadings are identities and their
    # model is exact; the other seven carry an alpha with a t-statistic and a share of variance.
    identity = fit["identity"][-1]
    expected = {
        "IBGL.AS": [1.0, 0.5, 0.0, 0.0],
        "IEGE.AS": [1.0, -0.5, 0.0, 0.0],
        "IEAC.AS": [1.0, 0.5, 1.0, 0.0],
        "IHYG.L": [1.0, 0.5, 1.0, 1.0],
    }
    assert set(identity) == set(expected)
    for sleeve, loadings in identity.items():
        assert loadings == pytest.approx(expected[sleeve], abs=1e-9), f"{sleeve}'s construction changed"
    assert fit["r2"]["IBGL.AS"].min() == pytest.approx(1.0)
    assert fit["resid_var"]["IBGL.AS"].max() == pytest.approx(0.0, abs=1e-24)
    assert fit["alpha"]["IBGL.AS"].isna().all(), "no alpha to test: the block is the construction"
    estimated = [sleeve for sleeve in returns.columns if sleeve not in identity]
    assert len(estimated) == 7
    assert fit["alpha"][estimated].notna().to_numpy().all()
    assert (fit["r2"][estimated] < 1.0).to_numpy().all()

    # The count rule: inside the pre-registered bound, stable across permutation seeds, and not
    # falsified by its own movement.
    rule = cp.decompose(returns)
    assert 1 <= rule["components"] <= cp.PREREGISTERED_K
    series = cp.count_series(returns, months)
    assert series["falsified"] is False
    assert set(series["counts"]) <= {1, 2, 3}
    assert cp.stability(returns, months)["agrees"] is True

    # Spanning: the verdicts at the count the rule retains, in both directions.
    directions = spanning.directions(returns, named, rule["components"])
    assert directions["headline"]["p_value"] < 0.05 and directions["reverse"]["p_value"] < 0.05
    assert np.abs(directions["headline"]["rows_sum"] - 1.0).max() < 1.0

    # The covariance axis: three estimators on one window, each labelled, symmetric and positive
    # semi-definite, with the shrinkage estimator conditioning the sample's problem away.
    window = months[-60:]
    block = exposures.window_block(returns, window)
    report = covariance.conditioning(block)
    assert report["sample_condition"] > report["shrinkage_condition"]
    assert 0.0 < report["intensity"] < 1.0
    for name, matrix in report["covariances"].items():
        assert list(matrix.index) == universe.TICKERS, f"{name} lost the sleeve map's order"
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
        assert np.all(np.linalg.eigvalsh(matrix.to_numpy()) > -1e-18), f"{name} is not positive semi-definite"
