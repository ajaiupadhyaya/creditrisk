"""
Private Credit Fragility Project — ML Distress Classifier
Gradient boosting model with SHAP interpretability.
Predicts probability of shadow default (PIK, LME, amendment, extension, hard default).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

from paths import ensure_data_dir, get_data_dir

np.random.seed(42)


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def engineer_features(df, current_sofr=0.043):
    """
    Build feature matrix from loan data.
    Features are designed to capture the three thesis pillars:
    1. Credit quality (ICR, leverage)
    2. PIK / coupon structure
    3. Macro sensitivity (rate, vintage)
    """
    X = pd.DataFrame()

    # --- Credit quality features ---
    df['all_in_rate'] = df['spread_bps'] / 10000 + current_sofr
    df['annual_interest'] = df['total_debt_mm'] * df['all_in_rate']
    df['icr'] = df['ebitda_mm'] / df['annual_interest'].replace(0, np.nan)

    X['icr']                = df['icr'].fillna(0).clip(0, 10)
    X['leverage_x']         = df['leverage_x'].clip(1, 12)
    X['spread_bps']         = df['spread_bps']
    X['ltv']                = df['ltv'].fillna(0.5)

    # --- Rate / vintage exposure ---
    X['sofr_at_orig']       = df['base_rate_at_orig']
    X['rate_shock']         = (current_sofr - df['base_rate_at_orig']).clip(-0.05, 0.55)
    X['vintage']            = df['vintage']
    X['is_2021_vintage']    = (df['vintage'] == 2021).astype(int)
    X['is_2020_or_earlier'] = (df['vintage'] <= 2020).astype(int)

    # --- Tenor / maturity features ---
    df['maturity_dt'] = pd.to_datetime(df['maturity_dt'])
    df['months_to_mat'] = (df['maturity_dt'] - pd.Timestamp.today()).dt.days / 30.44
    X['months_to_maturity'] = df['months_to_mat'].clip(-6, 120)
    X['in_maturity_wall']   = (df['months_to_mat'] < 18).astype(int)
    X['tenor_yrs']          = df['tenor_yrs']

    # --- Structure features ---
    X['pik_flag']           = (df['coupon_type'] == 'pik').astype(int)
    X['is_unitranche']      = (df['lien_position'] == 'unitranche').astype(int)
    X['is_second_lien']     = (df['lien_position'] == 'second_lien').astype(int)

    cov_map = {'none': 0, 'incurrence': 1, 'maintenance': 2}
    X['covenant_score']     = df['covenant_type'].map(cov_map).fillna(1)

    # --- Sector encoding ---
    sector_risk = {
        'Technology / SaaS': 0.14,
        'Healthcare Services': 0.09,
        'Business Services': 0.10,
        'Consumer & Retail': 0.13,
        'Industrials': 0.08,
        'Financial Services': 0.07,
        'Media & Entertainment': 0.15,
        'Education': 0.09,
        'Real Estate Services': 0.11,
    }
    X['sector_base_risk']   = df['sector'].map(sector_risk).fillna(0.10)

    # --- Non-linear interaction features ---
    X['icr_x_leverage']     = X['icr'] * X['leverage_x']
    X['pik_x_rate_shock']   = X['pik_flag'] * X['rate_shock']
    X['icr_x_maturity']     = X['icr'] * X['in_maturity_wall']
    X['leverage_sq']        = X['leverage_x'] ** 2
    X['log_icr']            = np.log1p(X['icr'].clip(0, 10))

    return X


def build_target(df):
    """
    Binary distress flag: 1 = any shadow or hard default, 0 = performing.
    This captures the 'true' default rate (Fitch methodology).
    """
    distress_statuses = {'pik_toggle', 'amended', 'extended', 'lme', 'default'}
    return df['status'].isin(distress_statuses).astype(int)


# ======================================================
# MODEL TRAINING & EVALUATION
# ======================================================

def train_and_evaluate(X, y, loan_df):
    print("\n=== ML Distress Classifier ===")
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]} | Distress rate: {y.mean():.1%}")

    # Model zoo
    models = {
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=10, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight='balanced', random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=1000, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n{'Model':<25} {'AUC':>8} {'AUC Std':>9}")
    print("-" * 45)
    best_model = None
    best_auc = 0

    for name, model in models.items():
        if name == 'Logistic Regression':
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        print(f"{name:<25} {scores.mean():>7.3f}  {scores.std():>8.4f}")

        if scores.mean() > best_auc:
            best_auc = scores.mean()
            best_model_name = name
            if name == 'Logistic Regression':
                best_model = model
                best_model.fit(X_scaled, y)
                best_X = X_scaled
            else:
                best_model = model
                best_model.fit(X, y)
                best_X = X.values

    print(f"\nBest model: {best_model_name} (AUC={best_auc:.3f})")

    # Feature importance for GBM
    gbm = models['Gradient Boosting']
    gbm.fit(X, y)
    importances = pd.Series(gbm.feature_importances_, index=X.columns)
    print("\n=== Top Feature Importances (GBM) ===")
    print(importances.sort_values(ascending=False).head(12).round(4).to_string())

    # Calibrated probability predictions
    gbm_cal = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                   learning_rate=0.04, subsample=0.8,
                                   min_samples_leaf=10, random_state=42),
        cv=3, method='isotonic'
    )
    gbm_cal.fit(X, y)
    proba = gbm_cal.predict_proba(X)[:, 1]

    # Save predictions
    results = loan_df[['loan_id','borrower_name','sector','vintage',
                        'principal_mm','status']].copy()
    results['distress_flag_actual'] = y.values
    results['distress_proba'] = proba.round(4)
    results['distress_decile'] = pd.qcut(proba, 10, labels=False, duplicates='drop') + 1
    results.to_csv(get_data_dir() / "ml_predictions.csv", index=False)

    # Decile lift table (key output for credit analysts)
    print("\n=== Decile Lift Table ===")
    decile_table = results.groupby('distress_decile').agg(
        count=('loan_id', 'count'),
        actual_distress=('distress_flag_actual', 'sum'),
        avg_proba=('distress_proba', 'mean'),
        notional_mm=('principal_mm', 'sum'),
    )
    decile_table['distress_rate'] = decile_table['actual_distress'] / decile_table['count']
    decile_table['lift'] = decile_table['distress_rate'] / y.mean()
    print(decile_table.round(3).to_string())

    return gbm, gbm_cal, importances, results


# ======================================================
# PD → CREDIT SPREAD MAPPING (market-implied pricing)
# ======================================================

def pd_to_spread_model(loan_df):
    """
    Simple mapping from model-estimated PD to implied credit spread.
    Useful for assessing whether current spreads adequately price risk.
    """
    print("\n=== PD-to-Spread Adequacy Analysis ===")
    print("(Are current spreads pricing the model-estimated default risk?)")

    # Load predictions
    try:
        preds = pd.read_csv(get_data_dir() / "ml_predictions.csv")
        df = loan_df.merge(preds[['loan_id','distress_proba']], on='loan_id')
    except FileNotFoundError:
        print("Run train_and_evaluate() first.")
        return

    lgd_assumption = 0.45
    df['implied_spread_bps'] = (df['distress_proba'] * lgd_assumption * 10000).round(0)
    df['actual_spread_bps'] = df['spread_bps']
    df['spread_adequacy_bps'] = df['actual_spread_bps'] - df['implied_spread_bps']

    summary = df.groupby('sector').agg(
        avg_actual_spread=('actual_spread_bps', 'mean'),
        avg_implied_spread=('implied_spread_bps', 'mean'),
        avg_adequacy=('spread_adequacy_bps', 'mean'),
    ).round(0)
    summary['underpriced'] = summary['avg_adequacy'] < 0
    print(summary.sort_values('avg_adequacy').to_string())

    print(f"\nOverall: {(df['spread_adequacy_bps'] < 0).mean():.1%} of loans appear underpriced for risk taken")
    print(f"Avg spread gap: {df['spread_adequacy_bps'].mean():.0f} bps")


# ======================================================
# MAIN
# ======================================================

if __name__ == '__main__':
    data_dir = ensure_data_dir()
    print("Loading data...")
    loan_df = pd.read_csv(data_dir / "loans.csv")

    X = engineer_features(loan_df)
    y = build_target(loan_df)

    gbm, gbm_cal, importances, results = train_and_evaluate(X, y, loan_df)
    pd_to_spread_model(loan_df)

    print(f"\nML module complete. Predictions saved to {get_data_dir() / 'ml_predictions.csv'}")
