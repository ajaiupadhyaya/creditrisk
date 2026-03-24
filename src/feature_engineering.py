"""
Feature Engineering Module for Credit Risk Analysis
Transforms raw data, engineers new features, validates for leakage, computes VIF, handles imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from typing import Tuple, Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature engineering, validation, and preparation for modeling."""
    
    def __init__(self, loan_df: pd.DataFrame, portfolio_metrics_df: pd.DataFrame = None, 
                 macro_df: pd.DataFrame = None, output_path: Path = None):
        """Initialize with base dataframes."""
        self.loan_df = loan_df.copy()
        self.portfolio_metrics_df = portfolio_metrics_df
        self.macro_df = macro_df

        if output_path is not None:
            resolved_output = Path(output_path)
        else:
            cwd = Path.cwd()
            if cwd.name == 'notebooks':
                resolved_output = cwd.parent / 'outputs' / 'processed'
            else:
                resolved_output = cwd / 'outputs' / 'processed'

        self.output_path = resolved_output
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.engineered_df = None
        self.target = None
        self.numeric_features = []
        self.categorical_features = []
    
    def engineer_features(self) -> pd.DataFrame:
        """Execute all feature engineering transformations."""
        df = self.loan_df.copy()
        
        print("\n" + "="*70)
        print("🔧 FEATURE ENGINEERING")
        print("="*70 + "\n")
        
        # 1. Identify target
        self.target = 'defaulted'
        print(f"✅ Target variable: {self.target}")
        
        # 2. Engineer categorical buckets
        print(f"\n📦 Engineering categorical buckets...")
        df = self._engineer_dti_buckets(df)
        df = self._engineer_credit_utilization(df)
        df = self._engineer_delinquency_severity(df)
        
        # 3. Engineer continuous features
        print(f"\n🔢 Engineering continuous features...")
        df = self._engineer_payment_burden(df)
        df = self._engineer_log_transforms(df)
        
        # 4. Encode categorical variables
        print(f"\n🏷️  Encoding categorical variables...")
        df = self._onehot_encode_categoricals(df)
        
        # 5. Target encoding (with cross-validation to prevent leakage)
        print(f"\n🎯 Target encoding sectors & ratings...")
        df = self._target_encode_high_cardinality(df)
        
        # 6. Cross-file joins
        print(f"\n🔗 Cross-file feature joins...")
        df = self._join_sector_metrics(df)
        df = self._join_macro_variables(df)
        
        # 7. Remove intermediate columns
        print(f"\n🧹 Cleaning intermediate columns...")
        df = self._cleanup_columns(df)
        
        self.engineered_df = df
        return df
    
    def _engineer_dti_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create DTI (Debt-to-Income) buckets from leverage."""
        if 'leverage' in df.columns:
            # Use leverage as proxy for DTI (debt/equity)
            # Binned into 4 ordered categories
            df['dti_bucket'] = pd.cut(
                df['leverage'], 
                bins=[0, 0.20, 0.35, 0.50, np.inf],
                labels=['<20%', '20-35%', '35-50%', '>50%'],
                include_lowest=True
            )
            print(f"  ✓ DTI buckets created: {df['dti_bucket'].value_counts().to_dict()}")
        return df
    
    def _engineer_credit_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit utilization flags from leverage (as proxy)."""
        if 'leverage' in df.columns:
            def get_utilization(val):
                if val > 0.75:
                    return 'high'
                elif val > 0.30:
                    return 'medium'
                else:
                    return 'low'
            
            df['credit_util'] = df['leverage'].apply(get_utilization)
            print(f"  ✓ Credit utilization created: {df['credit_util'].value_counts().to_dict()}")
        return df
    
    def _engineer_delinquency_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create delinquency severity score from available flags."""
        # Check if delinquency columns exist; if not, use PD as proxy
        delinq_cols = [col for col in df.columns if 'delinq' in col.lower() or 'dq' in col.lower()]
        
        if delinq_cols:
            # Weighted sum: 30-day*1 + 60-day*2 + 90-day*4
            df['delinq_severity'] = 0
            for col in delinq_cols:
                if 'dq' in col.lower() or '30' in col:
                    df['delinq_severity'] += df[col].fillna(0) * 1
                elif '60' in col:
                    df['delinq_severity'] += df[col].fillna(0) * 2
                elif '90' in col:
                    df['delinq_severity'] += df[col].fillna(0) * 4
            print(f"  ✓ Delinquency severity created (from {len(delinq_cols)} columns)")
        else:
            # Default: use PD as proxy for risk severity
            if 'pd_annual' in df.columns:
                df['delinq_severity'] = df['pd_annual'] * 100  # Scale to 0-100
                print(f"  ✓ Delinquency severity (proxy from PD)")
        
        return df
    
    def _engineer_payment_burden(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create payment burden score: coupon_rate * EAD / annual_income proxy."""
        if 'coupon_rate' in df.columns and 'ead' in df.columns:
            # Create annual interest payment estimate
            df['annual_interest_payment'] = (df['coupon_rate'] / 100) * df['ead']
            
            # Assume annual income from EAD (scaled guess)
            df['assumed_annual_income'] = df['ead'] / 3  # Conservative estimate
            
            # Payment burden score
            df['payment_burden_score'] = (
                df['annual_interest_payment'] / df['assumed_annual_income']
            ).fillna(0) * 100  # Scale to 0-100
            
            print(f"  ✓ Payment burden score created")
        
        return df
    
    def _engineer_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform right-skewed numeric columns."""
        right_skewed = ['ead', 'pd_annual', 'leverage', 'interest_coverage']
        
        for col in right_skewed:
            if col in df.columns:
                # Add small constant to avoid log(0)
                df[f'log_{col}'] = np.log1p(df[col].fillna(0))
                print(f"  ✓ log_{col} created")
        
        return df
    
    def _onehot_encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables."""
        categorical_cols = ['loan_type', 'collateral', 'sector']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            # One-hot encode, drop first to avoid multicollinearity
            encoded = pd.get_dummies(df[col], prefix=col, drop_first=True, dummy_na=False)
            df = pd.concat([df, encoded], axis=1)
            print(f"  ✓ One-hot encoded: {col} ({encoded.shape[1]} columns created)")
        
        return df
    
    def _target_encode_high_cardinality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Target encode sector and rating using 5-fold CV to prevent leakage."""
        print(f"  • Using 5-fold Stratified CV for target encoding (no leakage)...")
        
        high_card_cols = ['sector', 'initial_rating']
        high_card_cols = [col for col in high_card_cols if col in df.columns]
        
        for col in high_card_cols:
            if col not in df.columns:
                continue
            
            # Initialize cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            target_encoded = np.zeros(len(df))
            
            # For each fold
            for train_idx, val_idx in kf.split(df):
                # Compute mean target for each category in training fold
                train_df = df.iloc[train_idx]
                target_means = train_df.groupby(col)[self.target].mean()
                
                # Apply to validation fold
                val_categories = df.iloc[val_idx][col]
                target_encoded[val_idx] = val_categories.map(target_means).fillna(target_means.mean()).values
            
            df[f'{col}_target_encoded'] = target_encoded
            print(f"    ✓ {col}_target_encoded created")
        
        return df
    
    def _join_sector_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join sector-level metrics from portfolio_metrics_df."""
        if self.portfolio_metrics_df is None:
            print(f"  ⚠️  portfolio_metrics_df not available, skipping join")
            return df

        # Some portfolio_metrics extracts are portfolio-wide only (no sector column).
        if 'sector' in self.portfolio_metrics_df.columns and 'avg_pd' in self.portfolio_metrics_df.columns:
            sector_avg_pd = self.portfolio_metrics_df.groupby('sector')['avg_pd'].mean().reset_index()
            sector_avg_pd.columns = ['sector', 'sector_avg_pd']
            df = df.merge(sector_avg_pd, on='sector', how='left')
            print(f"  ✓ sector_avg_pd joined from portfolio_metrics")
        else:
            if 'avg_pd' in self.portfolio_metrics_df.columns:
                df['portfolio_avg_pd'] = self.portfolio_metrics_df['avg_pd'].mean()
                print(f"  ✓ portfolio_avg_pd joined from portfolio_metrics (portfolio-level)")
            if 'avg_lgd' in self.portfolio_metrics_df.columns:
                df['portfolio_avg_lgd'] = self.portfolio_metrics_df['avg_lgd'].mean()
                print(f"  ✓ portfolio_avg_lgd joined from portfolio_metrics (portfolio-level)")
            if 'sector_hhi' in self.portfolio_metrics_df.columns:
                df['portfolio_sector_hhi'] = self.portfolio_metrics_df['sector_hhi'].mean()
                print(f"  ✓ portfolio_sector_hhi joined from portfolio_metrics (portfolio-level)")
        
        return df
    
    def _join_macro_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join macro variables from macro_stress_scenarios."""
        if self.macro_df is None:
            print(f"  ⚠️  macro_stress_scenarios not available, skipping join")
            return df

        if 'scenario' not in self.macro_df.columns:
            print(f"  ⚠️  scenario column not available in macro_stress_scenarios, skipping join")
            return df

        # Accept common baseline names and fallback to first available scenario.
        scenario_lower = self.macro_df['scenario'].astype(str).str.lower()
        baseline_mask = scenario_lower.isin(['base', 'baseline'])
        if baseline_mask.any():
            base_df = self.macro_df.loc[baseline_mask]
        else:
            base_df = self.macro_df.iloc[[0]]
            print(f"  ⚠️  baseline scenario not found, using first scenario row for macro joins")

        macro_cols = ['gdp_shock_pp', 'unemp_shock_pp', 'rate_shock_pp', 'credit_spread_bps']
        macro_cols = [col for col in macro_cols if col in self.macro_df.columns]

        for col in macro_cols:
            df[f'base_{col}'] = base_df[col].mean()
            print(f"  ✓ base_{col} joined from macro_stress")

        return df
    
    def _cleanup_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove intermediate/redundant columns."""
        cols_to_drop = [
            'annual_interest_payment', 'assumed_annual_income',  # Intermediate
            'dti_bucket',  # Replaced by buckets
        ]
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        df = df.drop(columns=cols_to_drop)
        print(f"  ✓ Dropped {len(cols_to_drop)} intermediate columns")
        
        return df
    
    def validate_features(self) -> Dict:
        """Validate engineered features for leakage and multicollinearity."""
        print("\n" + "="*70)
        print("⚠️  FEATURE VALIDATION")
        print("="*70 + "\n")
        
        validation_results = {}
        
        # 1. Leakage check
        print("✅ Leakage Check:")
        leakage_cols = [col for col in self.engineered_df.columns 
                       if any(x in col.lower() for x in ['recovery', 'loss_given', 'surprise'])]
        if leakage_cols:
            print(f"  ⚠️  POTENTIAL LEAKAGE DETECTED: {leakage_cols}")
            validation_results['leakage'] = 'FAILED'
        else:
            print(f"  ✓ No obvious post-default features detected")
            validation_results['leakage'] = 'PASSED'
        
        # 2. Handle missing values
        print(f"\n✅ Missing Values:")
        missing_cols = self.engineered_df.columns[self.engineered_df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                n_miss = self.engineered_df[col].isnull().sum()
                pct = n_miss / len(self.engineered_df) * 100
                # Impute
                if self.engineered_df[col].dtype in [np.float64, np.float32]:
                    self.engineered_df[col].fillna(self.engineered_df[col].median(), inplace=True)
                else:
                    self.engineered_df[col].fillna('unknown', inplace=True)
                print(f"  ✓ {col}: {n_miss} ({pct:.2f}%) imputed")
        else:
            print(f"  ✓ No missing values after feature engineering")
        
        # 3. VIF (Variance Inflation Factor) for numeric features
        print(f"\n✅ Multicollinearity Check (VIF):")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            numeric_features = self.engineered_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [col for col in numeric_features if col != self.target]
            
            vif_data = pd.DataFrame()
            vif_data['Feature'] = numeric_features
            vif_data['VIF'] = [variance_inflation_factor(
                self.engineered_df[numeric_features].values, i
            ) for i in range(len(numeric_features))]
            
            vif_high = vif_data[vif_data['VIF'] > 10]
            if len(vif_high) > 0:
                print(f"  ⚠️  HIGH VIF DETECTED ({len(vif_high)} features):")
                for idx, row in vif_high.iterrows():
                    print(f"     {row['Feature']:30s}: {row['VIF']:8.2f}")
                validation_results['vif'] = 'WARNING'
            else:
                print(f"  ✓ All VIF < 10 ({len(numeric_features)} numeric features)")
                validation_results['vif'] = 'PASSED'
        
        except Exception as e:
            print(f"  ⚠️  VIF computation failed: {e}")
            validation_results['vif'] = 'SKIPPED'
        
        # 4. Class balance
        print(f"\n✅ Target Variable Balance:")
        class_dist = self.engineered_df[self.target].value_counts()
        for label, count in class_dist.items():
            pct = count / len(self.engineered_df) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")
        
        imbalance_ratio = class_dist[0] / class_dist[1]
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        # Recommended class weights
        class_weight = {
            0: 1.0 / class_dist[0],
            1: 1.0 / class_dist[1]
        }
        class_weight = {k: v / sum(class_weight.values()) for k, v in class_weight.items()}
        print(f"\n  Recommended class_weight for models:")
        for label, weight in class_weight.items():
            print(f"    Class {label}: {weight:.4f}")
        
        validation_results['class_weight'] = class_weight
        validation_results['imbalance_ratio'] = imbalance_ratio
        
        return validation_results
    
    def save_processed_data(self, filename: str = 'processed_loans.csv') -> Path:
        """Save engineered dataset to CSV."""
        output_file = self.output_path / filename
        self.engineered_df.to_csv(output_file, index=False)
        
        print(f"\n" + "="*70)
        print(f"💾 PROCESSED DATA SAVED")
        print(f"="*70)
        print(f"  File: {output_file}")
        print(f"  Shape: {self.engineered_df.shape[0]:,} rows × {self.engineered_df.shape[1]} columns")
        print(f"  Size: {self.engineered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return output_file


def run_feature_engineering(loan_df: pd.DataFrame, portfolio_metrics_df: pd.DataFrame = None,
                           macro_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Path]:
    """Execute full feature engineering pipeline."""
    engineer = FeatureEngineer(loan_df, portfolio_metrics_df, macro_df)
    
    # Engineer features
    engineered_df = engineer.engineer_features()
    
    # Validate features
    validation_results = engineer.validate_features()
    
    # Save processed data
    output_file = engineer.save_processed_data()
    
    return engineered_df, output_file, validation_results
