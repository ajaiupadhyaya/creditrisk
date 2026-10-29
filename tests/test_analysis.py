import pandas as pd
import pytest
from run_analysis import ROOT, FEATURES, EXCLUDED, validate, temporal_split, stress_summary


def test_pre_outcome_allowlist():
    assert not set(FEATURES) & set(EXCLUDED + ['defaulted', 'loan_id'])


def test_cohorts_do_not_overlap():
    df = pd.read_csv(ROOT / 'loan_portfolio.csv')
    validate(df)
    train, test, cutoff = temporal_split(df)
    assert not (train & test).any()
    assert (train | test).all()
    assert pd.to_datetime(df.loc[train, 'origination_date']).max() < cutoff
    assert pd.to_datetime(df.loc[test, 'origination_date']).min() >= cutoff


def test_stress_is_bounded_and_monotone():
    df = pd.DataFrame({'ead':[100,200], 'pd_annual':[.1,.9], 'lgd':[.4,.9]})
    rows = stress_summary(df)
    assert rows[0]['expected_loss'] == pytest.approx(166)
    assert 0 <= rows[0]['loss_rate'] <= rows[1]['loss_rate'] <= rows[2]['loss_rate'] <= 1


def test_duplicate_ids_rejected():
    df = pd.read_csv(ROOT/'loan_portfolio.csv', nrows=3)
    df.loc[1, 'loan_id'] = df.loc[0, 'loan_id']
    with pytest.raises(ValueError, match='Duplicate'): validate(df)
