"""
Private Credit Fragility Project — Credit Loss Models
Modules:
  1. Monte Carlo Expected Loss Simulation
  2. ICR Distribution & Stress Testing
  3. PIK Compounding & Terminal Impairment
  4. Maturity Wall Analysis
  5. Fragility Scoring Engine
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from paths import ensure_data_dir

np.random.seed(42)

# ======================================================
# MODULE 1: MONTE CARLO CREDIT LOSS SIMULATION
# ======================================================

def simulate_credit_losses(loan_df, scenario='base', n_sims=50000):
    """
    Correlated Monte Carlo simulation of credit losses across the portfolio.
    Models systematic (macro) and idiosyncratic (borrower) risk components.
    
    Key parameters calibrated to:
    - KBRA DLD default data (1.8% cash, 4.4% with LMEs)
    - Fitch private credit monitoring (9.2% true distress)
    - Historical LGD for private credit (35-65% by lien position)
    """
    SCENARIO_PARAMS = {
        'base': {
            'sys_pd':    0.045,    # Systematic PD (macro-driven)
            'idio_std':  0.015,    # Idiosyncratic PD vol
            'lgd_mean':  0.38,
            'lgd_std':   0.10,
            'rho_pd_lgd': 0.42,   # Correlation: PD & LGD spike together in downturns
            'sofr':       0.043,
        },
        'stress': {
            'sys_pd':    0.085,
            'idio_std':  0.025,
            'lgd_mean':  0.52,
            'lgd_std':   0.12,
            'rho_pd_lgd': 0.55,
            'sofr':       0.053,
        },
        'severe': {
            'sys_pd':    0.130,
            'idio_std':  0.035,
            'lgd_mean':  0.65,
            'lgd_std':   0.15,
            'rho_pd_lgd': 0.65,
            'sofr':       0.063,
        },
    }

    p = SCENARIO_PARAMS[scenario]
    notional = loan_df['principal_mm'].values
    total_notional = notional.sum()
    n_loans = len(loan_df)

    # Sector correlation matrix (simplified 3-factor model)
    sector_map = {s: i % 3 for i, s in enumerate(loan_df['sector'].unique())}
    factors = np.random.standard_normal((n_sims, 3))  # 3 macro factors

    results = []
    BATCH = 500
    for batch_start in range(0, n_sims, BATCH):
        batch_end = min(batch_start + BATCH, n_sims)
        batch_size = batch_end - batch_start
        F = factors[batch_start:batch_end]  # (batch, 3)

        # Loan-level systematic exposure (random loadings per loan)
        rng = np.random.RandomState(batch_start)
        loadings = rng.uniform(0.3, 0.7, (n_loans, 3))
        loadings /= loadings.sum(axis=1, keepdims=True)

        # Correlated PD draws (Vasicek single-factor extension)
        Z_pd  = (F @ loadings.T)           # systematic component (batch x loans)
        Z_idio = rng.standard_normal((batch_size, n_loans))  # idiosyncratic
        rho = 0.35
        Z_combined = np.sqrt(rho) * Z_pd + np.sqrt(1 - rho) * Z_idio

        # Translate to PD using normal copula
        pd_draws = norm.cdf(Z_combined) * p['idio_std'] + p['sys_pd']
        pd_draws = np.clip(pd_draws, 0, 1)

        # Correlated LGD (pro-cyclical — LGD worsens when PD spikes)
        Z_lgd = p['rho_pd_lgd'] * Z_combined + np.sqrt(1 - p['rho_pd_lgd']**2) * rng.standard_normal((batch_size, n_loans))
        lgd_draws = norm.cdf(Z_lgd) * p['lgd_std'] + p['lgd_mean']
        lgd_draws = np.clip(lgd_draws, 0.05, 0.95)

        # Default indicator
        default_mask = rng.random((batch_size, n_loans)) < pd_draws

        # Portfolio loss for each simulation
        losses = (default_mask * lgd_draws * notional).sum(axis=1)
        loss_rates = losses / total_notional
        results.append(loss_rates)

    loss_dist = np.concatenate(results)

    summary = {
        'scenario':          scenario,
        'el_mean':           loss_dist.mean(),
        'el_median':         np.median(loss_dist),
        'var_95':            np.percentile(loss_dist, 95),
        'var_99':            np.percentile(loss_dist, 99),
        'cvar_95':           loss_dist[loss_dist >= np.percentile(loss_dist, 95)].mean(),
        'cvar_99':           loss_dist[loss_dist >= np.percentile(loss_dist, 99)].mean(),
        'max_loss':          loss_dist.max(),
        'loss_distribution': loss_dist,
    }
    return summary


def run_all_scenarios(loan_df):
    print("Running Monte Carlo credit loss simulations (50k paths each)...")
    results = {}
    for scenario in ['base', 'stress', 'severe']:
        print(f"  Scenario: {scenario}...")
        results[scenario] = simulate_credit_losses(loan_df, scenario)

    print("\n=== Credit Loss Summary ===")
    print(f"{'Scenario':<10} {'EL Mean':>9} {'VaR 95%':>9} {'VaR 99%':>9} {'CVaR 99%':>10}")
    print("-" * 55)
    for scen, r in results.items():
        print(f"{scen:<10} {r['el_mean']:>8.2%}  {r['var_95']:>8.2%}  {r['var_99']:>8.2%}  {r['cvar_99']:>9.2%}")
    return results


# ======================================================
# MODULE 2: ICR DISTRIBUTION & RATE STRESS TESTING
# ======================================================

def analyze_icr_distribution(loan_df, sofr_scenarios=None):
    """
    Compute ICR distribution under multiple rate scenarios.
    ICR = EBITDA / (Total Debt × All-in Rate)
    """
    if sofr_scenarios is None:
        sofr_scenarios = {
            'Current (4.3%)':    0.043,
            'Stress (+100bps)':  0.053,
            'Severe (+200bps)':  0.063,
            'Rate Cut (-200bps)':0.023,
        }

    df = loan_df.copy()
    df = df[~df['status'].isin(['paid_off', 'written_off'])].copy()

    results = {}
    for label, sofr in sofr_scenarios.items():
        all_in = df['spread_bps'] / 10000 + sofr
        annual_int = df['total_debt_mm'] * all_in
        icr = df['ebitda_mm'] / annual_int.replace(0, np.nan)

        results[label] = {
            'sofr':             sofr,
            'avg_icr':          icr.mean(),
            'median_icr':       icr.median(),
            'pct_below_1x':     (icr < 1.0).mean(),
            'pct_below_1_5x':   (icr < 1.5).mean(),
            'pct_below_2x':     (icr < 2.0).mean(),
            'notional_below_1x': df.loc[icr < 1.0, 'principal_mm'].sum(),
            'notional_below_1_5x': df.loc[icr < 1.5, 'principal_mm'].sum(),
            'icr_series':       icr,
        }

    print("\n=== ICR Distribution Under Rate Scenarios ===")
    print(f"{'Scenario':<22} {'Avg ICR':>8} {'<1.0x':>8} {'<1.5x':>8} {'Notional <1.5x ($mm)':>22}")
    print("-" * 75)
    for label, r in results.items():
        print(f"{label:<22} {r['avg_icr']:>7.2f}x {r['pct_below_1x']:>7.1%}  {r['pct_below_1_5x']:>7.1%}  ${r['notional_below_1_5x']:>18,.0f}")
    return results


def sector_icr_heatmap(loan_df, current_sofr=0.043):
    """Sector-level ICR breakdown — identify concentration of stress."""
    df = loan_df.copy()
    df['all_in'] = df['spread_bps'] / 10000 + current_sofr
    df['annual_interest'] = df['total_debt_mm'] * df['all_in']
    df['icr'] = df['ebitda_mm'] / df['annual_interest'].replace(0, np.nan)
    df['distress_flag'] = df['status'].isin(['pik_toggle','amended','extended','lme','default'])

    sector_stats = df.groupby('sector').agg(
        loans=('loan_id', 'count'),
        total_notional=('principal_mm', 'sum'),
        avg_icr=('icr', 'mean'),
        median_icr=('icr', 'median'),
        pct_below_1x=('icr', lambda x: (x < 1.0).mean()),
        pct_below_1_5x=('icr', lambda x: (x < 1.5).mean()),
        avg_leverage=('leverage_x', 'mean'),
        distress_rate=('distress_flag', 'mean'),
        pik_rate=('coupon_type', lambda x: (x == 'pik').mean()),
    ).round(3).sort_values('avg_icr')

    print("\n=== Sector ICR Heatmap ===")
    print(sector_stats[['avg_icr','pct_below_1x','pct_below_1_5x','avg_leverage','distress_rate','pik_rate']].to_string())
    return sector_stats


# ======================================================
# MODULE 3: PIK COMPOUNDING ANALYSIS
# ======================================================

def model_pik_trajectory(principal, spread_bps, vintage, sofr_at_orig,
                           ebitda_at_orig, pik_toggle_yr=3, tenor=6,
                           current_sofr=0.043, ebitda_growth=0.03):
    """
    Model a loan that toggles from cash to PIK mid-life.
    Demonstrates how principal balance grows and leverage expands.
    """
    cash_rate = spread_bps / 10000 + sofr_at_orig
    pik_rate  = cash_rate + 0.02   # PIK premium typically 150-250bps

    balance = principal
    ebitda = ebitda_at_orig
    records = []

    for yr in range(1, tenor + 1):
        is_pik = yr >= pik_toggle_yr
        ebitda *= (1 + ebitda_growth)

        if is_pik:
            interest_cash = 0
            interest_pik  = balance * pik_rate
            balance      += interest_pik
        else:
            rate = spread_bps / 10000 + sofr_at_orig
            interest_cash = balance * rate
            interest_pik  = 0

        all_in = spread_bps / 10000 + current_sofr
        icr = ebitda / (balance * all_in) if balance > 0 else 0
        leverage = balance / ebitda if ebitda > 0 else 0

        records.append({
            'year': yr,
            'is_pik': is_pik,
            'balance_mm': round(balance, 2),
            'ebitda_mm': round(ebitda, 2),
            'leverage_x': round(leverage, 2),
            'icr': round(icr, 3),
            'cash_interest_mm': round(interest_cash, 2),
            'pik_interest_mm': round(interest_pik, 2),
            'pik_overhang_mm': round(balance - principal, 2),
        })

    return pd.DataFrame(records)


def analyze_pik_portfolio(loan_df):
    """Portfolio-level PIK analysis: phantom yield quantification."""
    pik_loans = loan_df[loan_df['coupon_type'] == 'pik'].copy()
    cash_loans = loan_df[loan_df['coupon_type'] == 'cash'].copy()

    sofr = 0.043
    pik_loans['all_in'] = pik_loans['spread_bps'] / 10000 + sofr
    pik_loans['reported_interest_mm'] = pik_loans['total_debt_mm'] * pik_loans['all_in']
    pik_loans['cash_collected_mm']    = 0   # cash = 0 for full PIK

    cash_loans['all_in'] = cash_loans['spread_bps'] / 10000 + sofr
    cash_loans['reported_interest_mm'] = cash_loans['total_debt_mm'] * cash_loans['all_in']
    cash_loans['cash_collected_mm']    = cash_loans['reported_interest_mm']

    total_reported = (pik_loans['reported_interest_mm'].sum() +
                      cash_loans['reported_interest_mm'].sum())
    total_cash = cash_loans['cash_collected_mm'].sum()
    phantom_yield = pik_loans['reported_interest_mm'].sum()

    print(f"\n=== PIK Phantom Yield Analysis ===")
    print(f"PIK loans: {len(pik_loans)} ({len(pik_loans)/len(loan_df):.1%} of book)")
    print(f"PIK notional: ${pik_loans['principal_mm'].sum():,.0f}mm")
    print(f"Total reported interest income: ${total_reported:,.0f}mm/yr")
    print(f"Cash interest actually collected: ${total_cash:,.0f}mm/yr")
    print(f"Phantom (PIK) income: ${phantom_yield:,.0f}mm/yr ({phantom_yield/total_reported:.1%} of total)")
    print(f"PIK overhang (unpaid principal growth): compounds at ~{pik_loans['pik_rate'].mean():.1%}/yr")

    return {
        'pik_loan_count': len(pik_loans),
        'pik_pct_book': len(pik_loans) / len(loan_df),
        'phantom_income_mm': phantom_yield,
        'phantom_pct_total': phantom_yield / total_reported,
    }


# ======================================================
# MODULE 4: MATURITY WALL ANALYSIS
# ======================================================

def analyze_maturity_wall(loan_df, current_date=None):
    """
    Identify refinancing risk concentration over 24-month horizon.
    """
    if current_date is None:
        current_date = pd.Timestamp.today()

    df = loan_df.copy()
    df['maturity_dt'] = pd.to_datetime(df['maturity_dt'])
    df['days_to_mat'] = (df['maturity_dt'] - current_date).dt.days
    df['months_to_mat'] = df['days_to_mat'] / 30.44

    window_18m = df[df['months_to_mat'] <= 18]
    window_24m = df[df['months_to_mat'] <= 24]

    # Refinancing cost shock for 2021 vintage (borrowed at ~5.5%, now 9-10%)
    df['orig_all_in'] = df['spread_bps'] / 10000 + df['base_rate_at_orig']
    df['refi_all_in'] = df['spread_bps'] / 10000 + 0.043
    df['annual_cost_increase_mm'] = df['total_debt_mm'] * (df['refi_all_in'] - df['orig_all_in'])

    quarter_wall = df[df['months_to_mat'] <= 24].groupby(
        df['maturity_dt'].dt.to_period('Q')
    ).agg(
        loans=('loan_id', 'count'),
        principal_mm=('principal_mm', 'sum'),
        avg_icr=('icr_current', 'mean'),
        distressed=('status', lambda x: x.isin(['pik_toggle','amended','extended','lme','default']).sum()),
        pik_count=('coupon_type', lambda x: (x == 'pik').sum()),
        cost_increase_mm=('annual_cost_increase_mm', 'sum'),
    ).reset_index()

    print("\n=== Maturity Wall: 24-Month Refinancing Risk ===")
    print(f"Loans maturing in 18 months: {len(window_18m)} | Notional: ${window_18m['principal_mm'].sum():,.0f}mm")
    print(f"Loans maturing in 24 months: {len(window_24m)} | Notional: ${window_24m['principal_mm'].sum():,.0f}mm")
    print(f"Already-distressed in wall:  {window_24m['status'].isin(['pik_toggle','amended','extended','lme','default']).sum()}")
    print(f"\nAnnual interest cost increase on 2021 vintage refinancing:")
    cost_shock = df[df['vintage'] == 2021]['annual_cost_increase_mm'].sum()
    print(f"  2021 vintage cohort: +${cost_shock:,.0f}mm/yr additional burden")
    print(f"\nQuarterly maturity schedule:")
    print(quarter_wall.to_string(index=False))
    return quarter_wall, window_24m


# ======================================================
# MODULE 5: LIQUIDITY STRESS TEST
# ======================================================

class SemiLiquidFundStress:
    """
    Models liquidity dynamics of semi-liquid private credit funds
    under redemption pressure scenarios.
    Calibrated to the Cliffwater / Ares / Morgan Stanley events of 2025-2026.
    """

    def __init__(self, name, nav_mm, liquid_buffer_pct=0.12, gate_pct=0.05,
                 secondary_haircut=0.15, fund_leverage=0.65):
        self.name           = name
        self.nav            = nav_mm
        self.liquid_buffer  = nav_mm * liquid_buffer_pct
        self.gate_pct       = gate_pct
        self.haircut        = secondary_haircut
        self.loan_book      = nav_mm * (1 - liquid_buffer_pct)
        self.total_debt     = nav_mm * fund_leverage
        self.equity_nav     = nav_mm

    def simulate(self, redemption_rate_qtrly, quarters=8):
        """Simulate NAV erosion under sustained redemption pressure."""
        history = []
        nav = self.nav
        liquid = self.liquid_buffer
        loan_book = self.loan_book

        for q in range(1, quarters + 1):
            requested = nav * redemption_rate_qtrly
            max_from_gate = nav * self.gate_pct
            available_cash = liquid

            # Honor up to gate, then draw from liquid buffer
            fulfilled_cash = min(requested, max_from_gate, available_cash)
            liquid -= fulfilled_cash
            gated_amount = max(0, requested - fulfilled_cash)

            # Forced secondary sales to replenish (at haircut)
            if liquid < nav * 0.03 and gated_amount > 0:
                sale_needed = min(gated_amount, loan_book * 0.10)
                net_proceeds = sale_needed * (1 - self.haircut)
                additional_fulfilled = min(net_proceeds, gated_amount)
                fulfilled_cash += additional_fulfilled
                loan_book -= sale_needed
                nav -= sale_needed * self.haircut   # crystallize haircut loss
                liquid += net_proceeds - additional_fulfilled

            nav -= fulfilled_cash
            liquid = max(liquid, 0)

            history.append({
                'quarter':      q,
                'nav_mm':       round(nav, 1),
                'liquid_mm':    round(liquid, 1),
                'liquid_pct':   round(liquid / nav if nav > 0 else 0, 3),
                'requested_mm': round(requested, 1),
                'fulfilled_mm': round(fulfilled_cash, 1),
                'gated_mm':     round(gated_amount, 1),
                'gated':        gated_amount > 0.1,
            })

        return pd.DataFrame(history)


def run_fund_stress_test(funds_df):
    """Run stress scenarios across fund universe."""
    redemption_scenarios = {
        'mild (5% qtrly)':     0.05,
        'elevated (8% qtrly)': 0.08,
        'severe (11% qtrly)':  0.11,   # Cliffwater/Ares observed level
        'extreme (15% qtrly)': 0.15,
    }

    print("\n=== Fund Liquidity Stress Test ===")
    for _, fund in funds_df.iterrows():
        if fund['fund_type'] not in ('semi_liquid', 'bdc'):
            continue
        f = SemiLiquidFundStress(
            name=fund['fund_name'],
            nav_mm=fund['nav_mm'],
            liquid_buffer_pct=fund['liquid_buffer_pct'],
            gate_pct=fund['redemption_gate_pct'],
        )
        print(f"\n{fund['fund_name']} (NAV: ${fund['nav_mm']:,}mm)")
        for label, rate in redemption_scenarios.items():
            hist = f.simulate(rate)
            gated_quarters = hist['gated'].sum()
            nav_erosion = (f.nav - hist['nav_mm'].iloc[-1]) / f.nav
            print(f"  {label}: {gated_quarters}/8 quarters gated | NAV erosion: {nav_erosion:.1%}")


# ======================================================
# MODULE 6: COMPOSITE FRAGILITY SCORE
# ======================================================

def compute_fragility_scores(loan_df, current_sofr=0.043):
    """
    Composite fragility score (0-100) per loan.
    Weights calibrated to empirical distress predictors.
    """
    df = loan_df.copy()

    # ICR component (0-30 pts) — most predictive
    df['all_in'] = df['spread_bps'] / 10000 + current_sofr
    df['icr'] = df['ebitda_mm'] / (df['total_debt_mm'] * df['all_in']).replace(0, np.nan)
    df['icr_score'] = np.clip(30 * (1 - df['icr'].clip(0, 2.5) / 2.5), 0, 30)

    # Leverage score (0-25 pts)
    df['lev_score'] = np.clip((df['leverage_x'] - 4) * 4.2, 0, 25)

    # PIK / coupon score (0-20 pts)
    df['pik_score'] = np.where(df['coupon_type'] == 'pik', 20,
                     np.where(df['status'] == 'pik_toggle', 14, 0))

    # Covenant score (0-15 pts)
    cov_map = {'none': 15, 'incurrence': 8, 'maintenance': 0}
    df['cov_score'] = df['covenant_type'].map(cov_map).fillna(8)

    # Maturity proximity (0-10 pts)
    df['maturity_dt'] = pd.to_datetime(df['maturity_dt'])
    df['months_to_mat'] = (df['maturity_dt'] - pd.Timestamp.today()).dt.days / 30.44
    df['mat_score'] = np.where(df['months_to_mat'] < 12, 10,
                     np.where(df['months_to_mat'] < 24, 6,
                     np.where(df['months_to_mat'] < 36, 3, 0)))

    df['fragility_score'] = (df['icr_score'] + df['lev_score'] +
                              df['pik_score'] + df['cov_score'] + df['mat_score'])

    df['risk_tier'] = pd.cut(df['fragility_score'],
                              bins=[0, 25, 45, 65, 100],
                              labels=['Low', 'Medium', 'High', 'Critical'])

    print("\n=== Portfolio Fragility Score Distribution ===")
    tier_summary = df.groupby('risk_tier').agg(
        count=('loan_id', 'count'),
        notional=('principal_mm', 'sum'),
        avg_score=('fragility_score', 'mean'),
    )
    print(tier_summary.round(1))
    print(f"\nWeighted avg fragility score: {(df['fragility_score'] * df['principal_mm']).sum() / df['principal_mm'].sum():.1f}/100")

    return df[['loan_id', 'sector', 'vintage', 'principal_mm', 'status',
               'icr', 'leverage_x', 'fragility_score', 'risk_tier',
               'icr_score', 'lev_score', 'pik_score', 'cov_score', 'mat_score']]


# ======================================================
# MAIN RUNNER
# ======================================================

if __name__ == '__main__':
    data_dir = ensure_data_dir()
    print("Loading loan book...")
    loan_df = pd.read_csv(data_dir / "loans.csv")
    funds_df = pd.read_csv(data_dir / "funds.csv")

    # Run all modules
    mc_results = run_all_scenarios(loan_df)
    icr_results = analyze_icr_distribution(loan_df)
    sector_icr  = sector_icr_heatmap(loan_df)
    pik_stats   = analyze_pik_portfolio(loan_df)

    # PIK example trajectory
    print("\n=== Example: 2021 Vintage PIK Loan Trajectory ===")
    traj = model_pik_trajectory(100, 575, 2021, 0.005, 16.7, pik_toggle_yr=3)
    print(traj[['year','is_pik','balance_mm','leverage_x','icr']].to_string(index=False))

    maturity_wall, wall_loans = analyze_maturity_wall(loan_df)
    run_fund_stress_test(funds_df)
    scores_df = compute_fragility_scores(loan_df)

    # Save scored output
    scores_df.to_csv(data_dir / "fragility_scores.csv", index=False)
    print("\nAll analyses complete. Results saved.")
