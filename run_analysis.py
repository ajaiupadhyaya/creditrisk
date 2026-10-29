"""Canonical portfolio analysis: raw synthetic loans → held-out evaluation.

Run: python run_analysis.py. Uses only pandas, numpy, scipy and scikit-learn.
Legacy notebooks and outputs are exploratory, not validated portfolio evidence.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent
NUMERIC = ['credit_score', 'ead', 'coupon_rate', 'leverage', 'interest_coverage', 'debt_to_equity', 'maturity_months']
CATEGORICAL = ['sector', 'loan_type', 'collateral', 'initial_rating']
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ['pd_annual', 'lgd', 'el', 'unexpected_loss', 'rwa', 'default_date', 'survival_months', 'recovery_rate', 'loss_given_default']


def validate(df):
    required = set(FEATURES + ['loan_id', 'origination_date', 'defaulted', 'pd_annual', 'lgd'])
    if required - set(df): raise ValueError(f'Missing columns: {required - set(df)}')
    if df.loan_id.duplicated().any(): raise ValueError('Duplicate loan IDs')
    if df.defaulted.isna().any() or not set(df.defaulted.unique()) <= {0, 1}: raise ValueError('Target must be binary')
    if (df.ead <= 0).any() or df.ead.isna().any(): raise ValueError('EAD must be positive')
    if not df.pd_annual.between(0, 1).all() or not df.lgd.between(0, 1).all(): raise ValueError('PD and LGD must be probabilities')
    pd.to_datetime(df.origination_date, errors='raise')


def temporal_split(df):
    dates = pd.to_datetime(df.origination_date)
    cutoff = dates.sort_values().iloc[int(len(df) * .8)]
    train, test = dates < cutoff, dates >= cutoff
    if df.loc[train, 'defaulted'].nunique() != 2 or df.loc[test, 'defaulted'].nunique() != 2:
        raise ValueError('Both temporal partitions must contain both outcomes')
    return train, test, cutoff


def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])


def stress_summary(df):
    """Illustrative one-year expected loss, based on supplied synthetic PD/LGD."""
    rows = []
    for name, pd_multiplier, lgd_add in [('Base', 1., 0.), ('Adverse', 1.5, .1), ('Severe', 2.5, .2)]:
        loss = df.ead * (df.pd_annual * pd_multiplier).clip(0, 1) * (df.lgd + lgd_add).clip(0, 1)
        rows.append(dict(scenario=name, pd_multiplier=pd_multiplier, lgd_add=lgd_add,
                         expected_loss=float(loss.sum()), loss_rate=float(loss.sum()/df.ead.sum())))
    return rows


def main():
    source = ROOT / 'loan_portfolio.csv'
    df = pd.read_csv(source)
    validate(df)
    train, test, cutoff = temporal_split(df)
    X, y = df[FEATURES], df.defaulted
    predictions = df.loc[test, ['loan_id', 'origination_date', 'defaulted']].copy()
    models = {'Logistic regression': LogisticRegression(C=1, max_iter=2000, random_state=42),
              'Random forest': RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=30, random_state=42, n_jobs=1)}
    results = []
    for name, model in models.items():
        pipeline = Pipeline([('preprocess', preprocessing()), ('model', model)])
        pipeline.fit(X.loc[train], y.loc[train])
        proba = pipeline.predict_proba(X.loc[test])[:, 1]
        predictions[name] = proba
        fpr, tpr, _ = roc_curve(y.loc[test], proba)
        actual, predicted = calibration_curve(y.loc[test], proba, n_bins=8, strategy='quantile')
        # Fixed-seed bootstrap measures holdout sampling uncertainty, not deployment risk.
        rng = np.random.default_rng(42)
        yt = y.loc[test].to_numpy()
        aucs = []
        for _ in range(200):
            sample = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[sample])) == 2: aucs.append(roc_auc_score(yt[sample], proba[sample]))
        results.append(dict(name=name, auc=float(roc_auc_score(yt, proba)),
                            auc_ci=np.quantile(aucs, [.025, .975]).tolist(),
                            average_precision=float(average_precision_score(yt, proba)),
                            brier=float(brier_score_loss(yt, proba)), log_loss=float(log_loss(yt, proba)),
                            roc=[dict(x=float(a), y=float(b)) for a,b in zip(fpr[::max(1,len(fpr)//80)], tpr[::max(1,len(tpr)//80)])] + [dict(x=1.,y=1.)],
                            calibration=[dict(predicted=float(a), actual=float(b)) for a,b in zip(predicted,actual)]))
    sector = df.groupby('sector').agg(loans=('loan_id','size'), exposure=('ead','sum'), default_rate=('defaulted','mean')).reset_index()
    summary = dict(schema_version=1, data_kind='Synthetic educational loan portfolio',
        source_file=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_attribution='Existing project describes a Kaggle dataset; original listing and redistribution license were not recorded. Raw rows are not included in the public site.',
        rows=len(df), exposure=float(df.ead.sum()), default_rate=float(y.mean()),
        train_rows=int(train.sum()), test_rows=int(test.sum()), cutoff=str(cutoff.date()),
        test_default_rate=float(y.loc[test].mean()), train_default_rate=float(y.loc[train].mean()),
        baseline_brier=float(brier_score_loss(y.loc[test], np.repeat(y.loc[train].mean(), test.sum()))),
        origination_start=str(df.origination_date.min()), origination_end=str(df.origination_date.max()),
        features=FEATURES, excluded=EXCLUDED, models=results, sectors=sector.to_dict('records'), stress=stress_summary(df),
        environment=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
        limitations=['Synthetic outcomes do not establish real-world predictive ability.',
            'Chronological split is by origination cohort. Outcome timestamps and observation horizons are not validated, so this is not a historical deployment backtest.',
            'The target is the supplied default flag across varied loan lifetimes, not a calibrated one-year probability of default.',
            'Stress results use supplied annual PD and LGD with illustrative shocks; they are separate from classifier scores.',
            'No model is selected or tuned on the holdout. Both prespecified models are reported.',
            'Bootstrap AUC intervals reflect resampling of the holdout only.'])
    out = ROOT / 'outputs' / 'validated'
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    predictions.to_csv(out/'holdout_predictions.csv', index=False)
    sector.to_csv(out/'sector_summary.csv', index=False)
    pd.DataFrame(summary['stress']).to_csv(out/'stress_summary.csv', index=False)
    lines=['# Credit risk: validated research results', '', f'{len(df):,} synthetic loans; {int(test.sum()):,} loans in the holdout beginning {cutoff.date()}.', '',
           '| Model | Holdout ROC AUC | Average precision | Brier score |', '|---|---:|---:|---:|']
    for r in results: lines.append(f"| {r['name']} | {r['auc']:.4f} | {r['average_precision']:.4f} | {r['brier']:.4f} |")
    lines += ['', '## Interpretation', '', f"The constant training-prevalence baseline has Brier score {summary['baseline_brier']:.4f}. Lower is better. Compare precision-recall performance with holdout prevalence {summary['test_default_rate']:.2%}.", '', '## Limitations', ''] + ['- '+x for x in summary['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    from report import write_report
    write_report(summary, out/'REPORT.html')
    print(json.dumps({k:summary[k] for k in ['rows','train_rows','test_rows','cutoff','baseline_brier']}, indent=2))
    for r in results: print(r['name'], 'AUC', round(r['auc'],4), 'Brier', round(r['brier'],4))

if __name__ == '__main__': main()
