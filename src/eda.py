"""
Exploratory Data Analysis Module for Credit Risk Analysis
Generates publication-quality visualizations using Plotly and Seaborn.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import gaussian_kde


class CreditRiskEDA:
    """Comprehensive EDA for credit risk datasets."""
    
    def __init__(self, datasets: Dict, output_path: Path = None):
        """Initialize with datasets dictionary."""
        self.datasets = datasets
        self.output_path = output_path or Path("outputs/eda")
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== PORTFOLIO OVERVIEW ====================
    
    def portfolio_sunburst(self, save=True) -> go.Figure:
        """Sunburst: loan count & total EAD by sector -> rating."""
        df = self.datasets['loan_portfolio']
        
        # Aggregate data
        agg_data = df.groupby(['sector', 'initial_rating']).agg({
            'loan_id': 'count',
            'ead': 'sum'
        }).reset_index()
        agg_data.columns = ['sector', 'rating', 'count', 'ead']
        
        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=['All'] + agg_data['sector'].tolist() + agg_data.apply(lambda x: f"{x['sector']}-{x['rating']}", axis=1).tolist(),
            parents=[''] + [agg_data.loc[0, 'sector']] * agg_data['sector'].nunique() + agg_data['sector'].tolist(),
            values=[agg_data['count'].sum()] + [agg_data[agg_data['sector']==s]['count'].sum() for s in agg_data['sector'].unique()] + agg_data['count'].tolist(),
            marker=dict(colorscale='Turbo', cmid=2),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Portfolio Overview: Loan Count by Sector & Rating',
            height=700,
            font=dict(size=11)
        )
        
        if save:
            fig.write_html(str(self.output_path / 'portfolio_sunburst.html'))
            print("✅ Saved: portfolio_sunburst.html")
        
        return fig
    
    def loan_amount_distribution(self, save=True) -> go.Figure:
        """Histogram + KDE: loan_amount by default status."""
        df = self.datasets['loan_portfolio']
        
        # Use EAD as proxy for loan amount
        non_default = df[df['defaulted'] == 0]['ead']
        default = df[df['defaulted'] == 1]['ead']
        
        fig = go.Figure()
        
        # Histograms
        fig.add_trace(go.Histogram(
            x=non_default, name='Non-Default (0)',
            nbinsx=50, opacity=0.7, marker_color='#1f77b4',
            hovertemplate='EAD: %{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Histogram(
            x=default, name='Default (1)',
            nbinsx=50, opacity=0.7, marker_color='#d62728',
            hovertemplate='EAD: %{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Loan Amount (EAD) Distribution by Default Status',
            xaxis_title='Exposure at Default ($)',
            yaxis_title='Number of Loans',
            barmode='overlay',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'loan_amount_distribution.html'))
            print("✅ Saved: loan_amount_distribution.html")
        
        return fig
    
    def correlation_heatmap(self, save=True):
        """Seaborn: Pearson correlation matrix of numeric features."""
        df = self.datasets['loan_portfolio']
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['loan_id']][:15]  # Limit to 15 for readability
        
        # Compute correlation
        corr = df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'}, vmin=-1, vmax=1)
        plt.title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(str(self.output_path / 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            print("✅ Saved: correlation_heatmap.png")
        
        return plt.gcf()
    
    def default_rate_by_sector(self, save=True) -> go.Figure:
        """Grouped bar: default rate by sector with 95% CI."""
        df = self.datasets['loan_portfolio']
        
        # Calculate default rate and CI by sector
        sector_stats = df.groupby('sector').agg({
            'defaulted': ['sum', 'count', 'mean']
        }).reset_index()
        sector_stats.columns = ['sector', 'n_default', 'n_total', 'default_rate']
        
        # Calculate 95% CI using binomial proportion
        sector_stats['se'] = np.sqrt(sector_stats['default_rate'] * (1 - sector_stats['default_rate']) / sector_stats['n_total'])
        sector_stats['ci_lower'] = np.maximum(sector_stats['default_rate'] - 1.96 * sector_stats['se'], 0)
        sector_stats['ci_upper'] = np.minimum(sector_stats['default_rate'] + 1.96 * sector_stats['se'], 1)
        
        # Sort descending
        sector_stats = sector_stats.sort_values('default_rate', ascending=True)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=sector_stats['sector'],
            x=sector_stats['default_rate'] * 100,
            orientation='h',
            error_x=dict(
                type='data',
                array=((sector_stats['ci_upper'] - sector_stats['default_rate']) * 100).values,
                visible=True
            ),
            marker_color='#d62728',
            hovertemplate='<b>%{y}</b><br>Default Rate: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Default Rate by Sector (with 95% CI)',
            xaxis_title='Default Rate (%)',
            yaxis_title='Sector',
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        if save:
            fig.write_html(str(self.output_path / 'default_rate_by_sector.html'))
            print("✅ Saved: default_rate_by_sector.html")
        
        return fig
    
    # ==================== CREDIT RATINGS ====================
    
    def rating_transition_sankey(self, save=True) -> go.Figure:
        """Sankey: flow from rating -> to_rating -> defaulted."""
        df = self.datasets.get('credit_ratings')
        
        if df is None:
            print("⚠️  credit_ratings.csv not available for Sankey")
            return None
        
        # Simplified Sankey: from_rating -> defaulted status
        sankey_data = df.groupby(['from_rating', 'defaulted']).size().reset_index(name='count')
        
        # Prepare nodes and links
        unique_ratings = sankey_data['from_rating'].unique().tolist()
        default_labels = ['Survived', 'Defaulted']
        all_labels = unique_ratings + default_labels
        
        # Create mapping
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        
        source = [label_to_idx[r] for r in sankey_data['from_rating']]
        target = [label_to_idx[default_labels[int(d)]] for d in sankey_data['defaulted']]
        value = sankey_data['count'].tolist()
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_labels,
                color=['#1f77b4']*len(unique_ratings) + ['#2ca02c', '#d62728']
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        fig.update_layout(
            title='Credit Rating Transitions & Default Outcomes',
            height=600,
            font=dict(size=11)
        )
        
        if save:
            fig.write_html(str(self.output_path / 'rating_transition_sankey.html'))
            print("✅ Saved: rating_transition_sankey.html")
        
        return fig
    
    def interest_rate_by_rating(self, save=True):
        """Violin plot: interest rate distribution by rating."""
        df = self.datasets['loan_portfolio']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Violin plot
        sns.violinplot(data=df, x='initial_rating', y='coupon_rate', ax=ax, palette='Set2')
        
        # Overlay points
        sns.stripplot(data=df, x='initial_rating', y='coupon_rate', 
                     hue='defaulted', ax=ax, size=3, alpha=0.3,
                     palette={0: '#1f77b4', 1: '#d62728'})
        
        ax.set_title('Interest Rate Distribution by Credit Rating', fontsize=14, fontweight='bold')
        ax.set_xlabel('Credit Rating', fontsize=12)
        ax.set_ylabel('Coupon Rate (%)', fontsize=12)
        ax.legend(title='Defaulted', loc='upper left')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(str(self.output_path / 'interest_rate_by_rating.png'), dpi=300, bbox_inches='tight')
            print("✅ Saved: interest_rate_by_rating.png")
        
        return fig
    
    # ==================== TIME/VINTAGE ====================
    
    def vintage_curves(self, save=True) -> go.Figure:
        """Line chart: cumulative default rate curves by vintage."""
        df = self.datasets.get('vintage_analysis')
        
        if df is None:
            print("⚠️  vintage_analysis.csv not available")
            return None
        
        fig = go.Figure()
        
        # Get unique vintages
        vintages = df['vintage'].unique()
        colors = sns.color_palette('husl', len(vintages))
        
        for i, vintage in enumerate(sorted(vintages)):
            vintage_data = df[df['vintage'] == vintage].sort_values('months_on_books')
            
            fig.add_trace(go.Scatter(
                x=vintage_data['months_on_books'],
                y=vintage_data['cumulative_default_rate'] * 100,
                mode='lines+markers',
                name=f'Vintage {vintage}',
                line=dict(width=2),
                hovertemplate='<b>Vintage %{fullData.name}</b><br>MOB: %{x}<br>CDR: %{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title='Vintage Curves: Cumulative Default Rate by Origination Cohort',
            xaxis_title='Months on Books',
            yaxis_title='Cumulative Default Rate (%)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'vintage_curves.html'))
            print("✅ Saved: vintage_curves.html")
        
        return fig
    
    def vintage_heatmap(self, save=True):
        """Heatmap: vintage × months_on_books -> cumulative default rate."""
        df = self.datasets.get('vintage_analysis')
        
        if df is None:
            print("⚠️  vintage_analysis.csv not available")
            return None
        
        # Pivot table
        pivot = df.pivot_table(
            values='cumulative_default_rate',
            index='vintage',
            columns='months_on_books',
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, cmap='RdYlBu_r', annot=True, fmt='.1f',
                   cbar_kws={'label': 'Cumulative Default Rate (%)'})
        plt.title('Vintage Analysis: Cumulative Default Rate Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Months on Books', fontsize=12)
        plt.ylabel('Vintage Year', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(str(self.output_path / 'vintage_heatmap.png'), dpi=300, bbox_inches='tight')
            print("✅ Saved: vintage_heatmap.png")
        
        return plt.gcf()
    
    # ==================== MACRO STRESS ====================
    
    def macro_scenarios_detail(self, save=True) -> go.Figure:
        """Multi-line chart: macro variables by stress scenario."""
        df = self.datasets.get('macro_stress_scenarios')
        
        if df is None:
            print("⚠️  macro_stress_scenarios.csv not available")
            return None
        
        # Create subplots for each macro variable
        macro_vars = ['gdp_shock_pp', 'unemp_shock_pp', 'rate_shock_pp', 'credit_spread_bps']
        macro_vars = [v for v in macro_vars if v in df.columns]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{v.replace("_", " ").title()}' for v in macro_vars],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        scenarios = df['scenario'].unique()
        colors = {'base': '#1f77b4', 'adverse': '#ff7f0e', 'severely_adverse': '#d62728'}
        
        for i, var in enumerate(macro_vars, 1):
            row, col = (i-1)//2 + 1, (i-1)%2 + 1
            
            for scenario in scenarios:
                scenario_df = df[df['scenario'] == scenario].sort_values('sector')
                
                fig.add_trace(
                    go.Scatter(
                        x=scenario_df['sector'],
                        y=scenario_df[var],
                        mode='lines+markers',
                        name=scenario,
                        line=dict(width=2, color=colors.get(scenario, '#1f77b4')),
                        hovertemplate=f'<b>{scenario}</b><br>{var}: %{{y}}<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=700, title_text='Macro Stress Scenarios Detail', hovermode='x unified')
        
        if save:
            fig.write_html(str(self.output_path / 'macro_scenarios_detail.html'))
            print("✅ Saved: macro_scenarios_detail.html")
        
        return fig
    
    def run_all_visualizations(self):
        """Generate all EDA visualizations."""
        print("\n" + "="*70)
        print("📊 GENERATING EXPLORATORY DATA ANALYSIS VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("🎨 Portfolio Overview:")
        self.portfolio_sunburst()
        self.loan_amount_distribution()
        self.correlation_heatmap()
        self.default_rate_by_sector()
        
        print("\n🎨 Credit Ratings:")
        self.rating_transition_sankey()
        self.interest_rate_by_rating()
        
        print("\n🎨 Time/Vintage:")
        self.vintage_curves()
        self.vintage_heatmap()
        
        print("\n🎨 Macro Scenarios:")
        self.macro_scenarios_detail()
        
        print("\n" + "="*70)
        print(f"✅ ALL VISUALIZATIONS SAVED TO: {self.output_path}")
        print("="*70)
