"""
Stress Testing & Portfolio Risk Analytics Module
Applies macro scenarios to shift PD and compute Expected Loss (EL) analysis by sector.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


class StressTester:
    """Portfolio-level stress testing and risk analytics."""
    
    def __init__(self, loan_df: pd.DataFrame, macro_df: pd.DataFrame, 
                 predictions_df: pd.DataFrame = None, output_path: Path = None):
        """Initialize with loan, macro scenarios, and optional model predictions."""
        self.loan_df = loan_df.copy()
        self.macro_df = macro_df.copy()
        self.predictions_df = predictions_df  # Should contain predicted PD per loan

        if output_path is not None:
            resolved_output = Path(output_path)
        else:
            cwd = Path.cwd()
            if cwd.name == 'notebooks':
                resolved_output = cwd.parent / 'outputs' / 'stress'
            else:
                resolved_output = cwd / 'outputs' / 'stress'

        self.output_path = resolved_output
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.stress_results = {}
        self.scenario_names = []
    
    def apply_stress_scenarios(self, lgd: float = 0.45) -> pd.DataFrame:
        """Apply macro stress scenarios to compute stressed PD and EL."""
        print("\n" + "="*70)
        print("⚡ APPLYING STRESS SCENARIOS")
        print("="*70 + "\n")
        
        results = []
        
        # Get unique scenarios
        scenarios = self.macro_df['scenario'].unique()
        self.scenario_names = sorted(scenarios)
        
        print(f"📊 Processing {len(scenarios)} scenarios: {self.scenario_names}\n")
        
        for scenario in self.scenario_names:
            scenario_df = self.macro_df[self.macro_df['scenario'] == scenario]
            
            # For each loan, apply scenario
            stress_loans = self.loan_df.copy()
            
            # Extract PD multiplier or uplift from macro_stress_scenarios
            if {'pd_multiplier', 'sector'}.issubset(scenario_df.columns) and 'sector' in stress_loans.columns:
                sector_multiplier = scenario_df.groupby('sector')['pd_multiplier'].mean()
                stress_loans['pd_multiplier'] = stress_loans['sector'].map(sector_multiplier).fillna(scenario_df['pd_multiplier'].mean())
                stress_loans['pd_stressed'] = stress_loans['pd_annual'] * stress_loans['pd_multiplier']
            elif 'pd_multiplier' in scenario_df.columns:
                pd_multiplier = scenario_df['pd_multiplier'].mean()
                stress_loans['pd_stressed'] = stress_loans['pd_annual'] * pd_multiplier
            elif {'pd_uplift_pp', 'sector'}.issubset(scenario_df.columns) and 'sector' in stress_loans.columns:
                sector_uplift = scenario_df.groupby('sector')['pd_uplift_pp'].mean() / 100.0
                stress_loans['pd_uplift'] = stress_loans['sector'].map(sector_uplift).fillna((scenario_df['pd_uplift_pp'].mean() / 100.0))
                stress_loans['pd_stressed'] = stress_loans['pd_annual'] + stress_loans['pd_uplift']
            elif 'pd_uplift_pp' in scenario_df.columns:
                pd_uplift = scenario_df['pd_uplift_pp'].mean() / 100
                stress_loans['pd_stressed'] = stress_loans['pd_annual'] + pd_uplift
            else:
                # If no direct PD adjustment, use base PD
                stress_loans['pd_stressed'] = stress_loans['pd_annual']
            
            # Clamp PD to [0, 1]
            stress_loans['pd_stressed'] = stress_loans['pd_stressed'].clip(0, 1)
            
            # Calculate EL = PD × LGD × EAD with collateral-aware LGD if available.
            if 'collateral' in stress_loans.columns:
                secured_mask = stress_loans['collateral'].astype(str).str.lower().isin(['secured', 'yes', 'true', '1'])
                stress_loans['lgd_stressed'] = np.where(secured_mask, 0.25, 0.45)
            else:
                stress_loans['lgd_stressed'] = lgd

            stress_loans['el_dollars'] = stress_loans['pd_stressed'] * stress_loans['lgd_stressed'] * stress_loans['ead']
            stress_loans['scenario'] = scenario
            
            results.append(stress_loans)
            
            # Print scenario summary
            total_el = stress_loans['el_dollars'].sum()
            avg_pd = stress_loans['pd_stressed'].mean()
            print(f"✓ {scenario.upper():20s}: Total EL = ${total_el/1e6:>8.2f}M, Avg PD = {avg_pd*100:5.2f}%")
        
        self.stress_results = pd.concat(results, ignore_index=True)
        return self.stress_results

    def _pick_scenario_name(self, preferred: list[str]) -> str | None:
        """Return the first matching scenario name (case-insensitive)."""
        if not self.scenario_names:
            return None

        lower_to_original = {str(name).lower(): name for name in self.scenario_names}
        for name in preferred:
            if name in lower_to_original:
                return lower_to_original[name]
        return None
    
    def compute_sector_analysis(self) -> pd.DataFrame:
        """Compute EL by sector and scenario."""
        print("\n" + "-"*70)
        print("📊 SECTOR-LEVEL EXPECTED LOSS ANALYSIS")
        print("-"*70 + "\n")
        
        sector_analysis = self.stress_results.groupby(['sector', 'scenario']).agg({
            'el_dollars': 'sum',
            'ead': 'sum',
            'pd_stressed': 'mean',
            'loan_id': 'count'
        }).reset_index()
        
        sector_analysis.columns = ['sector', 'scenario', 'total_el', 'total_ead', 'avg_pd', 'n_loans']
        sector_analysis['el_pct_ead'] = (sector_analysis['total_el'] / sector_analysis['total_ead'] * 100).round(2)
        
        # Display top sectors by EL
        print("Top 5 Sectors by EL (Adverse Scenario):")
        adverse_sectors = sector_analysis[sector_analysis['scenario'] == 'adverse'].nlargest(5, 'total_el')
        for _, row in adverse_sectors.iterrows():
            print(f"  {row['sector']:15s}: EL=${row['total_el']/1e6:8.2f}M ({row['el_pct_ead']:5.2f}% of EAD)")
        
        return sector_analysis
    
    def plot_el_by_sector(self, sector_analysis: pd.DataFrame, save=True) -> go.Figure:
        """Grouped bar chart: EL by sector under different scenarios."""
        print("\n📈 Generating EL by Sector Chart...")
        
        # Pivot for grouped bar
        pivot_data = sector_analysis.pivot(index='sector', columns='scenario', values='total_el')
        pivot_data = pivot_data / 1e6  # Convert to millions
        
        # Reorder columns by severity
        preferred_order = ['base', 'baseline', 'mild', 'adverse', 'gfc_like', 'covid_like', 'severe', 'severely_adverse']
        col_order = [col for col in preferred_order if col in pivot_data.columns]
        if not col_order:
            col_order = list(pivot_data.columns)
        pivot_data = pivot_data[col_order]
        
        fig = go.Figure()
        
        colors = {
            'base': '#2ca02c',
            'baseline': '#2ca02c',
            'mild': '#17becf',
            'adverse': '#ff7f0e',
            'gfc_like': '#9467bd',
            'covid_like': '#8c564b',
            'severe': '#d62728',
            'severely_adverse': '#d62728'
        }
        
        for scenario in col_order:
            fig.add_trace(go.Bar(
                x=pivot_data.index,
                y=pivot_data[scenario],
                name=scenario.upper(),
                marker=dict(color=colors.get(scenario, '#1f77b4')),
                hovertemplate='<b>%{x}</b><br>' + scenario + ': $%{y:.2f}M<extra></extra>'
            ))
        
        fig.update_layout(
            title='Expected Loss by Sector - Stress Scenarios',
            xaxis_title='Sector',
            yaxis_title='Expected Loss ($ Millions)',
            barmode='group',
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'el_by_sector_stress.html'))
            print("  ✓ Saved: el_by_sector_stress.html")
        
        return fig
    
    def plot_el_waterfall(self, save=True) -> go.Figure:
        """Waterfall chart showing EL attribution to macro factors."""
        print("📈 Generating EL Waterfall Chart...")
        
        # Compare base vs adverse scenario
        base_name = self._pick_scenario_name(['base', 'baseline'])
        adverse_name = self._pick_scenario_name(['adverse', 'severe', 'severely_adverse'])

        if base_name is None or adverse_name is None:
            raise ValueError("Could not determine baseline/adverse scenarios for waterfall chart")

        base_total = self.stress_results[self.stress_results['scenario'] == base_name]['el_dollars'].sum()
        adverse_total = self.stress_results[self.stress_results['scenario'] == adverse_name]['el_dollars'].sum()
        
        el_increase = adverse_total - base_total
        
        # Create waterfall
        fig = go.Figure(go.Waterfall(
            x=[f'{base_name.title()} EL', 'PD Shock', f'{adverse_name.title()} EL'],
            y=[base_total/1e6, el_increase/1e6, adverse_total/1e6],
            text=[f'${base_total/1e6:.2f}M', f'+${el_increase/1e6:.2f}M', f'${adverse_total/1e6:.2f}M'],
            textposition='outside',
            connector=dict(line=dict(dash='solid')),
            decreasing=dict(marker=dict(color='#ff7f0e')),
            increasing=dict(marker=dict(color='#d62728')),
            totals=dict(marker=dict(color='#1f77b4'))
        ))
        
        fig.update_layout(
            title='Expected Loss Attribution - Base to Adverse Scenario',
            yaxis_title='Expected Loss ($ Millions)',
            height=500,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'el_waterfall_adverse.html'))
            print("  ✓ Saved: el_waterfall_adverse.html")
        
        return fig
    
    def plot_sector_risk_bubble(self, sector_analysis: pd.DataFrame, save=True) -> go.Figure:
        """Bubble chart: current vs stressed PD by sector."""
        print("📈 Generating Sector Risk Bubble Chart...")
        
        # Get baseline and stressed scenarios dynamically
        base_name = self._pick_scenario_name(['base', 'baseline'])
        adverse_name = self._pick_scenario_name(['adverse', 'severe', 'severely_adverse'])

        if base_name is None or adverse_name is None:
            raise ValueError("Could not determine baseline/adverse scenarios for sector bubble chart")

        base_data = sector_analysis[sector_analysis['scenario'] == base_name][['sector', 'avg_pd', 'total_ead']].copy()
        adverse_data = sector_analysis[sector_analysis['scenario'] == adverse_name][['sector', 'avg_pd']].copy()
        adverse_data.columns = ['sector', 'avg_pd_adverse']
        
        bubble_data = base_data.merge(adverse_data, on='sector')
        if bubble_data.empty:
            raise ValueError("No overlapping sectors between baseline and stressed scenarios for bubble chart")

        bubble_data['pd_uplift'] = (bubble_data['avg_pd_adverse'] - bubble_data['avg_pd']) * 100
        bubble_data['ead_millions'] = bubble_data['total_ead'] / 1e6
        
        fig = go.Figure(data=go.Scatter(
            x=bubble_data['avg_pd'] * 100,
            y=bubble_data['pd_uplift'],
            mode='markers',
            marker=dict(
                size=bubble_data['ead_millions'] / bubble_data['ead_millions'].max() * 50,
                color=np.arange(len(bubble_data)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sector')
            ),
            text=bubble_data['sector'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Base PD: %{x:.2f}%<br>' +
                         'PD Uplift: %{y:.2f}pp<br>' +
                         'Exposure: $%{marker.size:.0f}M<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sector Risk Heat: Base vs Adverse PD',
            xaxis_title='Base Case PD (%)',
            yaxis_title='PD Uplift under Adverse (percentage points)',
            height=600,
            template='plotly_white'
        )
        
        if save:
            fig.write_html(str(self.output_path / 'sector_risk_bubble.html'))
            print("  ✓ Saved: sector_risk_bubble.html")
        
        return fig
    
    def plot_el_heatmap(self, sector_analysis: pd.DataFrame, save=True):
        """Heatmap: sector × scenario EL matrix."""
        print("📈 Generating EL Heatmap...")
        
        # Pivot table
        pivot = sector_analysis.pivot(index='sector', columns='scenario', values='el_pct_ead')
        
        # Reorder columns
        col_order = ['base', 'baseline', 'mild', 'adverse', 'gfc_like', 'covid_like', 'severe', 'severely_adverse']
        selected_cols = [col for col in col_order if col in pivot.columns]
        if selected_cols:
            pivot = pivot[selected_cols]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='Reds', cbar_kws={'label': 'EL as % of EAD'},
                   ax=ax, linewidths=0.5)
        ax.set_title('Expected Loss Heatmap - Sector × Scenario', fontsize=14, fontweight='bold')
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        plt.tight_layout()
        
        if save:
            plt.savefig(str(self.output_path / 'el_heatmap_sector_scenario.png'), dpi=300, bbox_inches='tight')
            print("  ✓ Saved: el_heatmap_sector_scenario.png")
        
        return fig

    def plot_vintage_stress_overlay(self, vintage_df: pd.DataFrame, scenario_name: str = 'adverse', save=True) -> go.Figure:
        """Overlay stressed vintage curves on original cumulative default curves."""
        print("📈 Generating Vintage Stress Overlay...")

        if vintage_df is None or vintage_df.empty:
            raise ValueError("vintage_df is required for vintage stress overlay")

        base_name = self._pick_scenario_name(['base', 'baseline'])
        stress_name = self._pick_scenario_name([scenario_name.lower(), 'adverse', 'severe', 'severely_adverse'])
        if base_name is None or stress_name is None:
            raise ValueError("Could not determine baseline/stress scenario for vintage overlay")

        base_mult = self.macro_df[self.macro_df['scenario'] == base_name]['pd_multiplier'].mean()
        stress_mult = self.macro_df[self.macro_df['scenario'] == stress_name]['pd_multiplier'].mean()
        uplift_ratio = 1.0 if base_mult == 0 or np.isnan(base_mult) else (stress_mult / base_mult)

        fig = go.Figure()
        for vintage, group in vintage_df.groupby('vintage'):
            g = group.sort_values('months_on_books')
            base_curve = g['cumulative_default_rate']
            stressed_curve = np.clip(base_curve * uplift_ratio, 0, 1)

            fig.add_trace(go.Scatter(
                x=g['months_on_books'], y=base_curve * 100,
                mode='lines',
                name=f'{vintage} Base',
                line=dict(width=2)
            ))
            fig.add_trace(go.Scatter(
                x=g['months_on_books'], y=stressed_curve * 100,
                mode='lines',
                name=f'{vintage} {stress_name.title()}',
                line=dict(width=2, dash='dash')
            ))

        fig.update_layout(
            title=f'Vintage Cumulative Default Curves: Base vs {stress_name.title()}',
            xaxis_title='Months on Books',
            yaxis_title='Cumulative Default Rate (%)',
            template='plotly_white',
            hovermode='x unified',
            height=650
        )

        if save:
            fig.write_html(str(self.output_path / 'vintage_stress_overlay.html'))
            print("  ✓ Saved: vintage_stress_overlay.html")

        return fig
    
    def run_all_stress_testing(self, vintage_df: pd.DataFrame = None):
        """Execute complete stress testing pipeline."""
        print("\n" + "="*70)
        print("🔬 CREDIT RISK STRESS TESTING & PORTFOLIO ANALYTICS")
        print("="*70)
        
        # Apply stress scenarios
        self.apply_stress_scenarios()
        
        # Compute sector analysis
        sector_analysis = self.compute_sector_analysis()
        
        # Generate visualizations
        self.plot_el_by_sector(sector_analysis)
        self.plot_el_waterfall()
        self.plot_sector_risk_bubble(sector_analysis)
        self.plot_el_heatmap(sector_analysis)
        if vintage_df is not None:
            self.plot_vintage_stress_overlay(vintage_df)
        
        print("\n" + "="*70)
        print("✅ STRESS TESTING COMPLETE")
        print("="*70)
        print(f"\n📁 All outputs saved to: {self.output_path}")
        
        return sector_analysis
