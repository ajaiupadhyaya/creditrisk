"""
Private Credit Fragility Project — Synthetic Dataset Generator
Generates a realistic loan book for analysis based on actual market parameters.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import random

from paths import ensure_data_dir

np.random.seed(42)
random.seed(42)

# -------------------------------------------------------
# MARKET PARAMETERS (calibrated to real market data)
# -------------------------------------------------------

SECTORS = {
    'Technology / SaaS':    0.22,
    'Healthcare Services':  0.18,
    'Business Services':    0.15,
    'Consumer & Retail':    0.10,
    'Industrials':          0.10,
    'Financial Services':   0.08,
    'Media & Entertainment':0.07,
    'Education':            0.05,
    'Real Estate Services': 0.05,
}

VINTAGE_WEIGHTS = {
    2019: 0.08,
    2020: 0.15,
    2021: 0.28,   # Peak vintage — highest stress
    2022: 0.22,
    2023: 0.18,
    2024: 0.09,
}

SOFR_BY_YEAR = {
    2019: 0.022,
    2020: 0.008,
    2021: 0.005,
    2022: 0.140,
    2023: 0.430,
    2024: 0.520,
}

SPONSORS = [
    'Apollo', 'Ares', 'Blackstone', 'KKR', 'Carlyle', 'Vista Equity',
    'Thoma Bravo', 'Francisco Partners', 'Silver Lake', 'Bain Capital',
    'Warburg Pincus', 'TPG', 'Advent', 'CD&R', 'Leonard Green'
]


def generate_loan_book(n=600):
    rows = []
    vintages = list(VINTAGE_WEIGHTS.keys())
    vintage_probs = list(VINTAGE_WEIGHTS.values())
    sectors = list(SECTORS.keys())
    sector_probs = list(SECTORS.values())

    for i in range(n):
        vintage = np.random.choice(vintages, p=vintage_probs)
        sector = np.random.choice(sectors, p=sector_probs)
        sofr_at_orig = SOFR_BY_YEAR[vintage]

        # Loan sizing: log-normal, mean ~$75mm
        principal = np.random.lognormal(np.log(75), 0.7)
        principal = np.clip(principal, 15, 600)

        # Spread — tighter at peak (2021), wider in stress regime
        base_spread = {'Technology / SaaS': 550, 'Healthcare Services': 575,
                       'Consumer & Retail': 625, 'Industrials': 600}.get(sector, 590)
        vintage_adj = {2019: 25, 2020: 0, 2021: -50, 2022: 25, 2023: 50, 2024: 60}.get(vintage, 0)
        spread_bps = int(np.random.normal(base_spread + vintage_adj, 60))
        spread_bps = np.clip(spread_bps, 400, 800)

        # Leverage — 2021 vintage was most aggressive
        base_lev = {2019: 5.2, 2020: 5.4, 2021: 6.4, 2022: 6.0, 2023: 5.6, 2024: 5.3}[vintage]
        lev_sector_adj = {'Technology / SaaS': 0.4, 'Consumer & Retail': -0.3,
                          'Healthcare Services': 0.1}.get(sector, 0)
        leverage_x = np.random.normal(base_lev + lev_sector_adj, 0.9)
        leverage_x = np.clip(leverage_x, 2.5, 10.5)

        # EBITDA: derive from leverage and principal
        ebitda = principal / leverage_x * np.random.uniform(0.85, 1.15)
        ebitda = max(ebitda, 3.0)
        total_debt = leverage_x * ebitda
        senior_debt = total_debt * np.random.uniform(0.75, 1.0)
        equity = total_debt * np.random.uniform(0.3, 0.6)

        # Tenor
        tenor = np.random.choice([4, 5, 6, 7], p=[0.10, 0.42, 0.35, 0.13])
        orig_dt = date(vintage, np.random.randint(1, 13), np.random.randint(1, 28))
        mat_dt = date(vintage + tenor, orig_dt.month, orig_dt.day)

        # Covenant type: market shifted to incurrence/none over time
        cov_probs_by_vintage = {
            2019: [0.55, 0.35, 0.10],
            2020: [0.45, 0.40, 0.15],
            2021: [0.30, 0.45, 0.25],
            2022: [0.28, 0.47, 0.25],
            2023: [0.25, 0.50, 0.25],
            2024: [0.22, 0.50, 0.28],
        }[vintage]
        covenant_type = np.random.choice(['maintenance', 'incurrence', 'none'], p=cov_probs_by_vintage)
        covenant_threshold = np.round(leverage_x * 1.15, 1) if covenant_type == 'maintenance' else None

        # Current ICR with today's SOFR (4.3%)
        current_sofr = 0.043
        all_in = spread_bps / 10000 + current_sofr
        annual_interest = total_debt * all_in
        icr = ebitda / annual_interest if annual_interest > 0 else 99

        # Status — probabilistic, correlated with ICR
        status = _assign_status(icr, vintage, sector, leverage_x)
        coupon_type, pik_rate = _assign_coupon(status, icr)

        # LTV (for secured loans)
        ltv = np.random.uniform(0.35, 0.75)
        lien = np.random.choice(['first_lien', 'second_lien', 'unitranche'],
                                 p=[0.55, 0.15, 0.30])

        # Recovery (lower for second lien, PIK, high leverage)
        base_recovery = {'first_lien': 0.68, 'unitranche': 0.60, 'second_lien': 0.35}[lien]
        pik_penalty = -0.08 if coupon_type == 'pik' else 0
        lev_penalty = max(0, (leverage_x - 6) * -0.03)
        recovery = np.clip(base_recovery + pik_penalty + lev_penalty + np.random.normal(0, 0.05), 0.10, 0.90)

        rows.append({
            'loan_id': i + 1,
            'borrower_name': f'Portfolio Co. {i+1:04d}',
            'origination_dt': orig_dt.isoformat(),
            'maturity_dt': mat_dt.isoformat(),
            'principal_mm': round(principal, 2),
            'spread_bps': spread_bps,
            'base_rate_at_orig': sofr_at_orig,
            'coupon_type': coupon_type,
            'pik_rate': round(pik_rate, 4),
            'sector': sector,
            'sponsor': np.random.choice(SPONSORS),
            'ebitda_mm': round(ebitda, 2),
            'total_debt_mm': round(total_debt, 2),
            'senior_debt_mm': round(senior_debt, 2),
            'equity_mm': round(equity, 2),
            'covenant_type': covenant_type,
            'covenant_threshold': covenant_threshold,
            'ltv': round(ltv, 4),
            'lien_position': lien,
            'status': status,
            'recovery_rate': round(recovery, 4),
            'icr_current': round(icr, 4),
            'leverage_x': round(leverage_x, 2),
            'all_in_rate': round(all_in, 4),
            'vintage': vintage,
            'tenor_yrs': tenor,
            'days_to_maturity': (mat_dt - date.today()).days,
        })

    return pd.DataFrame(rows)


def _assign_status(icr, vintage, sector, leverage_x):
    """Assign loan status based on credit quality signals."""
    # Base distress probability from ICR
    if icr < 0.8:
        distress_prob = 0.70
    elif icr < 1.0:
        distress_prob = 0.45
    elif icr < 1.25:
        distress_prob = 0.22
    elif icr < 1.5:
        distress_prob = 0.10
    elif icr < 2.0:
        distress_prob = 0.04
    else:
        distress_prob = 0.01

    # Vintage adjustment: 2021 cohort is most stressed
    distress_prob *= {2019: 0.8, 2020: 0.9, 2021: 1.5, 2022: 1.2, 2023: 0.7, 2024: 0.4}[vintage]
    distress_prob = min(distress_prob, 0.90)

    if np.random.random() > distress_prob:
        return 'current'

    # Within distressed, assign type
    r = np.random.random()
    if r < 0.18:
        return 'default'
    elif r < 0.35:
        return 'lme'
    elif r < 0.55:
        return 'amended'
    elif r < 0.72:
        return 'extended'
    else:
        return 'pik_toggle'


def _assign_coupon(status, icr):
    """Assign coupon type based on loan status."""
    if status == 'pik_toggle' or (status == 'amended' and icr < 1.2):
        pik_rate = np.random.uniform(0.10, 0.14)
        return 'pik', round(pik_rate, 4)
    elif status == 'current' and icr < 1.5 and np.random.random() < 0.08:
        pik_rate = np.random.uniform(0.08, 0.12)
        return 'pik', round(pik_rate, 4)
    return 'cash', 0.0


def generate_rate_scenarios():
    """Generate macro scenario table."""
    scenarios = []
    dates = pd.date_range('2024-01-01', '2027-12-31', freq='QS')

    sofr_paths = {
        0: ('base', [0.053, 0.053, 0.048, 0.043, 0.043, 0.038, 0.033, 0.030,
                     0.028, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]),
        1: ('stress', [0.053, 0.058, 0.063, 0.068, 0.070, 0.068, 0.065, 0.060,
                       0.055, 0.050, 0.045, 0.040, 0.038, 0.035, 0.033, 0.030]),
        2: ('severe', [0.053, 0.063, 0.073, 0.078, 0.080, 0.078, 0.073, 0.065,
                       0.055, 0.045, 0.040, 0.035, 0.033, 0.030, 0.028, 0.025]),
        3: ('rate_cut', [0.053, 0.043, 0.033, 0.023, 0.018, 0.015, 0.013, 0.012,
                         0.012, 0.012, 0.013, 0.015, 0.018, 0.020, 0.022, 0.025]),
    }

    oas_by_scenario = {'base': 380, 'stress': 520, 'severe': 720, 'rate_cut': 340}
    gdp_by_scenario = {'base': 0.024, 'stress': 0.005, 'severe': -0.025, 'rate_cut': 0.032}
    rec_prob = {'base': 0.12, 'stress': 0.38, 'severe': 0.75, 'rate_cut': 0.08}

    for sid, (sname, sofr_path) in sofr_paths.items():
        for j, dt in enumerate(dates[:16]):
            scenarios.append({
                'scenario_id': sid,
                'scenario_name': sname,
                'effective_dt': dt.date().isoformat(),
                'sofr': round(sofr_path[j], 4),
                'credit_spread_bps': oas_by_scenario[sname],
                'gdp_growth': gdp_by_scenario[sname],
                'recession_prob': rec_prob[sname],
            })

    return pd.DataFrame(scenarios)


def generate_funds():
    """Generate fund-level data."""
    return pd.DataFrame([
        {'fund_id': 1, 'fund_name': 'Ares Capital Corp', 'fund_type': 'bdc',
         'manager': 'Ares', 'nav_mm': 21400, 'liquid_buffer_pct': 0.08,
         'redemption_gate_pct': 0.0, 'leverage_ratio': 1.15, 'lp_lockup_yrs': 0,
         'retail_pct': 0.55, 'inception_dt': '2004-10-08'},
        {'fund_id': 2, 'fund_name': 'Blue Owl Corporate Lending', 'fund_type': 'semi_liquid',
         'manager': 'Blue Owl', 'nav_mm': 18700, 'liquid_buffer_pct': 0.12,
         'redemption_gate_pct': 0.05, 'leverage_ratio': 0.80, 'lp_lockup_yrs': 0.25,
         'retail_pct': 0.42, 'inception_dt': '2016-06-01'},
        {'fund_id': 3, 'fund_name': 'Cliffwater Corporate Lending Fund', 'fund_type': 'semi_liquid',
         'manager': 'Cliffwater', 'nav_mm': 16200, 'liquid_buffer_pct': 0.10,
         'redemption_gate_pct': 0.05, 'leverage_ratio': 0.65, 'lp_lockup_yrs': 0.25,
         'retail_pct': 0.60, 'inception_dt': '2019-03-15'},
        {'fund_id': 4, 'fund_name': 'Blackstone Private Credit', 'fund_type': 'semi_liquid',
         'manager': 'Blackstone', 'nav_mm': 52000, 'liquid_buffer_pct': 0.15,
         'redemption_gate_pct': 0.05, 'leverage_ratio': 0.55, 'lp_lockup_yrs': 0.25,
         'retail_pct': 0.35, 'inception_dt': '2017-01-01'},
        {'fund_id': 5, 'fund_name': 'Apollo Flagship Credit Fund', 'fund_type': 'closed_end',
         'manager': 'Apollo', 'nav_mm': 31000, 'liquid_buffer_pct': 0.05,
         'redemption_gate_pct': 0.0, 'leverage_ratio': 1.10, 'lp_lockup_yrs': 7,
         'retail_pct': 0.10, 'inception_dt': '2014-09-01'},
        {'fund_id': 6, 'fund_name': 'FS KKR Capital Corp', 'fund_type': 'bdc',
         'manager': 'FS/KKR', 'nav_mm': 15800, 'liquid_buffer_pct': 0.09,
         'redemption_gate_pct': 0.0, 'leverage_ratio': 1.20, 'lp_lockup_yrs': 0,
         'retail_pct': 0.48, 'inception_dt': '2007-12-21'},
        {'fund_id': 7, 'fund_name': 'Morgan Stanley Northaven', 'fund_type': 'semi_liquid',
         'manager': 'Morgan Stanley', 'nav_mm': 7600, 'liquid_buffer_pct': 0.11,
         'redemption_gate_pct': 0.05, 'leverage_ratio': 0.60, 'lp_lockup_yrs': 0.25,
         'retail_pct': 0.70, 'inception_dt': '2021-05-01'},
    ])


if __name__ == '__main__':
    data_dir = ensure_data_dir()
    print("Generating loan book (n=600)...")
    loans = generate_loan_book(600)
    loans.to_csv(data_dir / "loans.csv", index=False)
    print(f"  Loans: {len(loans)}, Total notional: ${loans['principal_mm'].sum():,.0f}mm")

    print("Generating rate scenarios...")
    scenarios = generate_rate_scenarios()
    scenarios.to_csv(data_dir / "rate_scenarios.csv", index=False)

    print("Generating fund data...")
    funds = generate_funds()
    funds.to_csv(data_dir / "funds.csv", index=False)

    # Summary stats
    print("\n=== Loan Book Summary ===")
    print(f"Status distribution:\n{loans['status'].value_counts()}")
    print(f"\nSector distribution:\n{loans['sector'].value_counts()}")
    print(f"\nICR stats:\n{loans['icr_current'].describe().round(2)}")
    print(f"\nPIK loans: {(loans['coupon_type']=='pik').sum()} ({(loans['coupon_type']=='pik').mean():.1%})")
    print(f"Distress rate (shadow): {loans['status'].isin(['pik_toggle','amended','extended','lme','default']).mean():.1%}")
    print(f"Cash default rate: {(loans['status']=='default').mean():.1%}")
