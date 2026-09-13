"""The risk-model axis: the three estimators agree on what they describe, and each is what it says."""

import numpy as np
import pandas as pd
import pytest

from portfolio_workbench.risk import covariance

MONTHS = pd.period_range("2000-01", periods=80, freq="M")
COLUMNS = [f"s{position}" for position in range(5)]


def returns(seed=5, periods=80, collinear=False, columns=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.02, (periods, len(columns or COLUMNS)))
    if collinear:
        values[:, -1] = values[:, 0] * 0.999999 + rng.normal(0, 1e-7, periods)
    index = MONTHS[:periods] if periods == len(MONTHS) else pd.period_range("2000-01", periods=periods, freq="M")
    return pd.DataFrame(values, index=index, columns=columns or COLUMNS)


def test_the_sample_estimator_is_the_sample_covariance():
    frame = returns()
    assert np.allclose(covariance.sample(frame).to_numpy(), np.cov(frame.to_numpy(), rowvar=False, ddof=1))


def test_every_estimator_carries_the_labels_it_was_given():
    """A covariance without its labels is a covariance that can align against the wrong weights, so
    each one takes the frame's order out with it - including when that order is unusual."""
    frame = returns(columns=["b", "a", "c"])
    shrunk, _ = covariance.shrinkage(frame)
    factored, _ = covariance.factor_model(frame, count=2)
    for matrix in (covariance.sample(frame), shrunk, factored):
        assert list(matrix.index) == ["b", "a", "c"]
        assert list(matrix.columns) == ["b", "a", "c"]
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)


def test_a_repeated_instrument_is_refused_rather_than_labelled():
    frame = returns(columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="repeated instrument"):
        covariance.sample(frame)


def test_shrinkage_conditions_what_the_sample_cannot_invert():
    """The axis exists because a sixty-month sample covariance is nearly singular at this N/T. The
    intensity is the estimator's own statement of how much of the sample it judged to be noise."""
    frame = returns(periods=20, collinear=True)
    plain = covariance.sample(frame)
    shrunk, intensity = covariance.shrinkage(frame)
    assert np.linalg.cond(plain) > 1e6
    assert np.linalg.cond(shrunk) < 1e3
    assert 0.0 < intensity < 1.0


def test_the_factor_covariance_reconstructs_the_discarded_eigenvalues():
    """Arithmetic rather than an empirical claim: what the retained components leave out is exactly
    the sum of the eigenvalues that were not retained, and the share reported is that sum over the
    total. Checked against an independent decomposition of the same window."""
    frame = returns()
    _, report = covariance.factor_model(frame, count=2)
    standardised = (frame.to_numpy() - frame.to_numpy().mean(axis=0)) / frame.to_numpy().std(axis=0, ddof=1)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.corrcoef(standardised, rowvar=False)))[::-1]
    assert report["discarded_sum"] == pytest.approx(float(eigenvalues[2:].sum()), abs=1e-10)
    assert report["discarded_share"] == pytest.approx(float(eigenvalues[2:].sum()) / len(COLUMNS), abs=1e-12)
    assert report["residual_sum"] == pytest.approx(report["discarded_sum"], abs=1e-10)


def test_the_conditioning_report_describes_all_three_on_one_window():
    frame = returns()
    report = covariance.conditioning(frame, count=2)
    assert report["components"] == 2
    assert report["sample_condition"] > report["shrinkage_condition"]
    assert set(report["covariances"]) == {"sample", "shrinkage", "factor"}
    for name, matrix in report["covariances"].items():
        assert list(matrix.index) == COLUMNS, f"{name} lost the order"
        assert np.all(np.linalg.eigvalsh(matrix.to_numpy()) > -1e-18), f"{name} is not positive semi-definite"
