"""
Model Training & Evaluation Module for Credit Risk Analysis
Trains 3 PD models with 5-fold CV and generates comprehensive evaluation charts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_score, recall_score,
    f1_score, log_loss, brier_score_loss, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

import xgboost as xgb
import lightgbm as lgb
import shap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


class CreditRiskModelTrainer:
    """Train and evaluate credit risk PD models."""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, output_path: Path = None):
        """Initialize with feature matrix and target."""
        self.X = X.copy()
        self.y = y.copy()

        if output_path is not None:
            resolved_output = Path(output_path)
        else:
            cwd = Path.cwd()
            if cwd.name == 'notebooks':
                resolved_output = cwd.parent / 'outputs' / 'models'
            else:
                resolved_output = cwd / 'outputs' / 'models'

        self.output_path = resolved_output
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.models = {}
        self.cv_results = {}
        self.predictions = {}
        self.metrics_summary = None

    def _concat_predictions(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate fold-level predictions for a model."""
        preds = self.predictions.get(model_name, [])
        y_true_all = np.concatenate([p['y_true'] for p in preds])
        y_pred_proba_all = np.concatenate([p['y_pred_proba'] for p in preds])
        y_pred_all = np.concatenate([p['y_pred'] for p in preds])
        return y_true_all, y_pred_proba_all, y_pred_all

    def save_oof_predictions(self):
        """Save out-of-fold predictions for downstream diagnostics/dashboarding."""
        rows = []
        for model_name, preds in self.predictions.items():
            for fold_pred in preds:
                fold = fold_pred['fold']
                for y_true, y_proba, y_hat in zip(fold_pred['y_true'], fold_pred['y_pred_proba'], fold_pred['y_pred']):
                    rows.append({
                        'model': model_name,
                        'fold': fold,
                        'y_true': int(y_true),
                        'y_pred_proba': float(y_proba),
                        'y_pred': int(y_hat)
                    })

        oof_df = pd.DataFrame(rows)
        out_file = self.output_path / 'oof_predictions.csv'
        oof_df.to_csv(out_file, index=False)
        print(f"  ✓ Saved: {out_file.name}")
        return oof_df
    
    def train_logistic_regression(self) -> Dict:
        """Train Logistic Regression baseline model."""
        print("\n🤖 Training Logistic Regression...")
        
        # Calculate class weights
        class_weight = {0: 0.5, 1: 0.5}  # Balanced
        
        model = LogisticRegression(penalty='l2', C=0.1, class_weight=class_weight,
                                  max_iter=1000, n_jobs=-1, random_state=42)
        
        # Cross-validation with predictions
        fold_metrics = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred = model.predict(X_val_scaled)
            
            # Compute metrics
            fold_metrics.append(self._compute_fold_metrics(y_val, y_pred, y_pred_proba))
            fold_predictions.append({
                'fold': fold, 'y_true': y_val.values, 'y_pred_proba': y_pred_proba, 'y_pred': y_pred
            })
        
        self.models['logistic'] = model
        self.cv_results['logistic'] = fold_metrics
        self.predictions['logistic'] = fold_predictions
        
        # Print summary
        avg_auc = np.mean([m['auc_roc'] for m in fold_metrics])
        print(f"  ✓ Avg AUC across 5 folds: {avg_auc:.4f}")
        
        return {'model': model, 'metrics': fold_metrics}
    
    def train_xgboost(self) -> Dict:
        """Train XGBoost model."""
        print("\n🤖 Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalance
        n_neg = (self.y == 0).sum()
        n_pos = (self.y == 1).sum()
        scale_pos_weight = n_neg / n_pos
        
        # Hyperparameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        
        # Cross-validation
        fold_metrics = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Train
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            
            # Predict
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Metrics
            fold_metrics.append(self._compute_fold_metrics(y_val, y_pred, y_pred_proba))
            fold_predictions.append({
                'fold': fold, 'y_true': y_val.values, 'y_pred_proba': y_pred_proba, 'y_pred': y_pred
            })
        
        self.models['xgboost'] = model
        self.cv_results['xgboost'] = fold_metrics
        self.predictions['xgboost'] = fold_predictions
        
        avg_auc = np.mean([m['auc_roc'] for m in fold_metrics])
        print(f"  ✓ Avg AUC across 5 folds: {avg_auc:.4f}")
        
        return {'model': model, 'metrics': fold_metrics}
    
    def train_lightgbm(self) -> Dict:
        """Train LightGBM model."""
        print("\n🤖 Training LightGBM...")
        
        # Calculate scale_pos_weight
        n_neg = (self.y == 0).sum()
        n_pos = (self.y == 1).sum()
        scale_pos_weight = n_neg / n_pos
        
        params = {
            'objective': 'binary',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'n_estimators': 200,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        fold_metrics = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Train
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            # Predict
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            # Metrics
            fold_metrics.append(self._compute_fold_metrics(y_val, y_pred, y_pred_proba))
            fold_predictions.append({
                'fold': fold, 'y_true': y_val.values, 'y_pred_proba': y_pred_proba, 'y_pred': y_pred
            })
        
        self.models['lightgbm'] = model
        self.cv_results['lightgbm'] = fold_metrics
        self.predictions['lightgbm'] = fold_predictions
        
        avg_auc = np.mean([m['auc_roc'] for m in fold_metrics])
        print(f"  ✓ Avg AUC across 5 folds: {avg_auc:.4f}")
        
        return {'model': model, 'metrics': fold_metrics}
    
    def _compute_fold_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """Compute all evaluation metrics for a fold."""
        # Find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'gini': 2 * roc_auc_score(y_true, y_pred_proba) - 1,
            'ks': ks_2samp(y_pred_proba[y_true == 1], y_pred_proba[y_true == 0]).statistic,
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred_optimal),
            'precision': precision_score(y_true, y_pred_optimal),
            'recall': recall_score(y_true, y_pred_optimal),
            'optimal_threshold': optimal_threshold
        }
        
        return metrics
    
    def generate_metrics_table(self) -> pd.DataFrame:
        """Generate summary metrics table."""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON - METRICS TABLE")
        print("="*70 + "\n")
        
        metrics_list = []
        
        for model_name, fold_metrics in self.cv_results.items():
            # Average across folds
            avg_metrics = {
                'Model': model_name.upper(),
                'AUC-ROC': np.mean([m['auc_roc'] for m in fold_metrics]),
                'Gini': np.mean([m['gini'] for m in fold_metrics]),
                'KS Statistic': np.mean([m['ks'] for m in fold_metrics]),
                'Log Loss': np.mean([m['log_loss'] for m in fold_metrics]),
                'Brier Score': np.mean([m['brier_score'] for m in fold_metrics]),
                'F1 Score': np.mean([m['f1'] for m in fold_metrics])
            }
            metrics_list.append(avg_metrics)
        
        self.metrics_summary = pd.DataFrame(metrics_list)
        
        # Display as table
        print(self.metrics_summary.to_string(index=False))
        
        # Save to CSV
        csv_file = self.output_path / 'model_metrics.csv'
        self.metrics_summary.to_csv(csv_file, index=False)
        print(f"\n✅ Metrics saved to {csv_file}")
        
        return self.metrics_summary
    
    def plot_roc_curves(self, save=True) -> go.Figure:
        """Plot ROC curves for all 3 models."""
        print("\n📈 Generating ROC Curve...")
        
        fig = go.Figure()
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        colors = {'logistic': '#1f77b4', 'xgboost': '#ff7f0e', 'lightgbm': '#2ca02c'}
        
        for model_name, predictions in self.predictions.items():
            # Concatenate predictions from all folds
            y_true_all = np.concatenate([p['y_true'] for p in predictions])
            y_pred_proba_all = np.concatenate([p['y_pred_proba'] for p in predictions])
            
            # Compute ROC
            fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
            auc_score = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name.upper()} (AUC={auc_score:.4f})',
                line=dict(width=2, color=colors.get(model_name, '#1f77b4'))
            ))
        
        fig.update_layout(
            title='ROC Curve Comparison - All 3 Models',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='closest',
            height=600,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'roc_curve_comparison.html'))
            print("  ✓ Saved: roc_curve_comparison.html")
        
        return fig
    
    def plot_pr_curves(self, save=True) -> go.Figure:
        """Plot Precision-Recall curves."""
        print("📈 Generating Precision-Recall Curve...")
        
        fig = go.Figure()
        
        colors = {'logistic': '#1f77b4', 'xgboost': '#ff7f0e', 'lightgbm': '#2ca02c'}
        
        for model_name, predictions in self.predictions.items():
            y_true_all = np.concatenate([p['y_true'] for p in predictions])
            y_pred_proba_all = np.concatenate([p['y_pred_proba'] for p in predictions])
            
            # Compute PR
            precision, recall, _ = precision_recall_curve(y_true_all, y_pred_proba_all)
            ap = average_precision_score(y_true_all, y_pred_proba_all)
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name.upper()} (AP={ap:.4f})',
                line=dict(width=2, color=colors.get(model_name, '#1f77b4'))
            ))
        
        fig.update_layout(
            title='Precision-Recall Curve Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='closest',
            height=600,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'pr_curve_comparison.html'))
            print("  ✓ Saved: pr_curve_comparison.html")
        
        return fig

    def plot_calibration_curves(self, save=True) -> go.Figure:
        """Plot reliability diagram for all models."""
        print("📈 Generating Calibration Curve...")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2)
        ))

        colors = {'logistic': '#1f77b4', 'xgboost': '#ff7f0e', 'lightgbm': '#2ca02c'}
        for model_name in self.predictions.keys():
            y_true_all, y_pred_proba_all, _ = self._concat_predictions(model_name)
            prob_true, prob_pred = calibration_curve(y_true_all, y_pred_proba_all, n_bins=10, strategy='uniform')

            fig.add_trace(go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode='lines+markers',
                name=model_name.upper(),
                line=dict(width=3, color=colors.get(model_name, '#1f77b4')),
                marker=dict(size=7)
            ))

        fig.update_layout(
            title='Calibration Curve (Reliability Diagram)',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Observed Default Rate',
            template='plotly_white',
            height=600,
            hovermode='closest'
        )

        if save:
            fig.write_html(str(self.output_path / 'calibration_curve_comparison.html'))
            print("  ✓ Saved: calibration_curve_comparison.html")

        return fig

    def plot_feature_importance(self, save=True, top_n: int = 25) -> go.Figure:
        """Plot top feature importances from XGBoost model."""
        print("📈 Generating Feature Importance Chart...")

        if 'xgboost' not in self.models:
            raise ValueError("XGBoost model not trained. Run train_xgboost first.")

        model = self.models['xgboost']
        importances = model.feature_importances_
        feat_names = list(self.X.columns)

        fi = pd.DataFrame({'feature': feat_names, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False).head(top_n).copy()

        def categorize_feature(name: str) -> str:
            n = name.lower()
            if n.startswith('log_') or 'target_encoded' in n or 'payment_burden' in n or 'delinq' in n or 'credit_util' in n:
                return 'engineered'
            if 'gdp' in n or 'unemp' in n or 'spread' in n or 'rate_shock' in n or 'portfolio_' in n:
                return 'macro'
            if 'sector' in n:
                return 'segment'
            return 'borrower/loan'

        fi['category'] = fi['feature'].apply(categorize_feature)
        fi['importance_pct'] = fi['importance'] / fi['importance'].sum() * 100

        colors = {
            'borrower/loan': '#4C78A8',
            'engineered': '#F58518',
            'segment': '#54A24B',
            'macro': '#E45756'
        }

        fig = go.Figure()
        for cat in ['borrower/loan', 'engineered', 'segment', 'macro']:
            sub = fi[fi['category'] == cat]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Bar(
                x=sub['importance_pct'],
                y=sub['feature'],
                orientation='h',
                name=cat,
                marker=dict(color=colors[cat]),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            title=f'XGBoost Feature Importance (Top {top_n})',
            xaxis_title='Relative Importance (%)',
            yaxis_title='Feature',
            barmode='stack',
            template='plotly_white',
            height=800,
            legend_title='Feature Category'
        )
        fig.update_yaxes(autorange='reversed')

        if save:
            fig.write_html(str(self.output_path / 'feature_importance_top25.html'))
            fi.to_csv(self.output_path / 'feature_importance_top25.csv', index=False)
            print("  ✓ Saved: feature_importance_top25.html")
            print("  ✓ Saved: feature_importance_top25.csv")

        return fig

    def plot_score_distributions(self, save=True):
        """Plot score distributions for default vs non-default by model."""
        print("📈 Generating Score Distribution Plots...")

        for model_name in self.predictions.keys():
            y_true_all, y_pred_proba_all, _ = self._concat_predictions(model_name)
            score_df = pd.DataFrame({'y_true': y_true_all, 'score': y_pred_proba_all})

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(
                data=score_df[score_df['y_true'] == 0],
                x='score', bins=50, stat='density', color='#4C78A8',
                alpha=0.45, label='Non-Default', ax=ax
            )
            sns.histplot(
                data=score_df[score_df['y_true'] == 1],
                x='score', bins=50, stat='density', color='#E45756',
                alpha=0.45, label='Default', ax=ax
            )
            ax.set_title(f'Score Distribution - {model_name.upper()}')
            ax.set_xlabel('Predicted PD')
            ax.set_ylabel('Density')
            ax.legend()
            plt.tight_layout()

            if save:
                out = self.output_path / f'score_distribution_{model_name}.png'
                plt.savefig(str(out), dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved: {out.name}")
            plt.close()

    def plot_shap_summary_xgboost(self, save=True):
        """Generate SHAP summary beeswarm plot for XGBoost."""
        print("📈 Generating SHAP Summary Plot (XGBoost)...")

        if 'xgboost' not in self.models:
            print("  ⚠️  XGBoost model not available. Skipping SHAP plot.")
            return

        try:
            model = self.models['xgboost']
            sample_n = min(5000, len(self.X))
            X_sample = self.X.sample(sample_n, random_state=42)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
            plt.title('SHAP Summary - XGBoost (Top 25 Features)')
            plt.tight_layout()

            if save:
                out = self.output_path / 'shap_summary_xgboost.png'
                plt.savefig(str(out), dpi=200, bbox_inches='tight')
                print("  ✓ Saved: shap_summary_xgboost.png")
            plt.close()
        except Exception as e:
            print(f"  ⚠️  SHAP generation skipped due to: {e}")
    
    def plot_confusion_matrices(self, save=True):
        """Generate confusion matrices for all models."""
        print("📈 Generating Confusion Matrices...")
        
        for model_name, predictions in self.predictions.items():
            y_true_all = np.concatenate([p['y_true'] for p in predictions])
            y_pred_all = np.concatenate([p['y_pred'] for p in predictions])  # Using hard predictions
            
            # Confusion matrix
            cm = confusion_matrix(y_true_all, y_pred_all)
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                       xticklabels=['Non-Default', 'Default'],
                       yticklabels=['Non-Default', 'Default'])
            ax.set_title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_xlabel('Predicted', fontsize=12)
            plt.tight_layout()
            
            if save:
                filename = self.output_path / f'confusion_matrix_{model_name}.png'
                plt.savefig(str(filename), dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved: confusion_matrix_{model_name}.png")
            
            plt.close()
    
    def run_all_training(self):
        """Execute complete model training pipeline."""
        print("\n" + "="*70)
        print("🚀 CREDIT RISK PD MODEL TRAINING - 5-FOLD CROSS-VALIDATION")
        print("="*70)
        
        # Train all models
        self.train_logistic_regression()
        self.train_xgboost()
        self.train_lightgbm()
        
        # Generate evaluation
        self.generate_metrics_table()
        self.plot_roc_curves()
        self.plot_pr_curves()
        self.plot_calibration_curves()
        self.plot_feature_importance()
        self.plot_shap_summary_xgboost()
        self.plot_confusion_matrices()
        self.plot_score_distributions()
        self.save_oof_predictions()
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"\n📁 All outputs saved to: {self.output_path}")
