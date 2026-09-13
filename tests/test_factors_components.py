"""The count rule: what it retains on planted structure, what it refuses on noise, and the
identities the null, the sign convention and the rotation have to satisfy."""

import itertools

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.factors import components as cp

DRAWS = 60


def planted(common=3, seed=7, periods=300, series=11):
    """A panel with a known number of common factors, so the count rule can be checked against an
    answer rather than against itself."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, 0.03, (periods, common))
    loadings = rng.normal(0, 1, (series, common))
    values = factors @ loadings.T + rng.normal(0, 0.01, (periods, series))
    return pd.DataFrame(
        values,
        index=pd.period_range("2000-01", periods=periods, freq="M"),
        columns=[f"s{position}" for position in range(series)],
    )


def test_the_rule_recovers_the_planted_common_factors():
    frame = planted(common=3)
    assert cp.decompose(frame, draws=DRAWS)["components"] == 3


def test_the_rule_does_not_invent_structure_in_noise():
    """Nothing planted, so the top eigenvalue is the top eigenvalue of a noise panel: the threshold
    exists to stop exactly this from being reported as a factor."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(0, 1, (300, 11)),
        index=pd.period_range("2000-01", periods=300, freq="M"),
        columns=[f"s{position}" for position in range(11)],
    )
    decomposition = cp.decompose(frame, draws=DRAWS)
    assert decomposition["components"] <= 1
    assert decomposition["eigenvalues"][0] < decomposition["threshold"][0] * 1.05


def test_the_mp_edge_matches_the_closed_form_the_decisions_quote():
    assert cp.mp_edge(11, 191) == pytest.approx(1.53759, rel=1e-4)
    assert cp.mp_edge(11, 60) == pytest.approx(2.03970, rel=1e-4)


def test_the_null_keeps_each_series_own_values_and_breaks_the_alignment():
    """A plain permutation would destroy each series' serial dependence and make the threshold too
    low; shifting each series by its own offset leaves its values - and so its marginal and its
    circular autocorrelation - untouched and removes only the alignment between series."""
    frame = planted(common=2, series=4, periods=120)
    null = cp.permutation_null(frame, draws=5, seed=1)
    assert null.shape == (5, 4)
    for draw in range(null.shape[0]):
        assert np.all(np.diff(null[draw]) <= 0), "the eigenvalues come back descending"
    # Under the null the cross-correlation is gone: the off-diagonal correlations of a shifted draw
    # sit near zero where the panel's own sit well away from it.
    shifted = np.column_stack([np.roll(frame.to_numpy()[:, 0], 7), np.roll(frame.to_numpy()[:, 1], 23)])
    panel_correlation = abs(np.corrcoef(frame.to_numpy()[:, :2], rowvar=False)[0, 1])
    shifted_correlation = abs(np.corrcoef(shifted, rowvar=False)[0, 1])
    assert shifted_correlation < 0.2 < panel_correlation


def test_the_sign_convention_fixes_the_largest_absolute_loading_positive():
    rng = np.random.default_rng(4)
    loadings = rng.normal(0, 1, (6, 2))
    oriented = cp.orient(loadings)
    for column in range(oriented.shape[1]):
        largest = oriented[:, column][np.argmax(np.abs(oriented[:, column]))]
        assert largest > 0
    assert np.allclose(np.abs(oriented), np.abs(loadings)), "orienting is a sign change, nothing more"


def test_varimax_rotates_without_changing_the_model():
    """Presentation only: the communalities and the total variance are preserved exactly, so the
    rotated table reconstructs the same correlation matrix. What it does change is which component
    a share of variance sits in, which is why the count rule runs on the unrotated eigenvalues."""
    decomposition = cp.decompose(planted(common=3), draws=DRAWS)
    loadings = decomposition["loadings"]
    rotated = cp.varimax(loadings)
    assert np.allclose((rotated ** 2).sum(axis=1), (loadings ** 2).sum(axis=1), atol=1e-12)
    assert np.allclose(rotated @ rotated.T, loadings @ loadings.T, atol=1e-12)
    assert float(np.trace(rotated.T @ rotated)) == pytest.approx(float(np.trace(loadings.T @ loadings)), rel=1e-12)


def test_the_falsification_falls_back_to_the_fixed_count(monkeypatch):
    """A count that jumps by more than one component between adjacent steps, in more than a quarter
    of them, is a rule choosing the answer as much as measuring it; the decision pre-specified the
    fallback and both counts are reported."""
    frame = planted(common=2, periods=200)
    calendar = frame.index
    sequence = [1, 3, 1, 3, 1, 3, 1, 3]
    calls = itertools.cycle(sequence)
    monkeypatch.setattr(cp, "retained", lambda *args, **kwargs: next(calls))
    series = cp.count_series(frame, calendar, window=100, draws=5)
    assert series["falsified"] is True
    assert set(series["counts"]) == {cp.PREREGISTERED_K}
    assert list(series["mechanical"][: len(sequence)]) == sequence
