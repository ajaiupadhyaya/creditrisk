-- =============================================================================
-- PRIVATE CREDIT FRAGILITY PROJECT — DATABASE SCHEMA
-- Author: Research Team
-- Thesis: "Private credit appears stable but beneath the surface, liquidity
--          mismatch, macro trends, and rising defaults suggest hidden fragility"
-- =============================================================================

-- -------------------------------------------------------
-- CORE LOAN BOOK
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS loans (
    loan_id             SERIAL PRIMARY KEY,
    borrower_name       VARCHAR(100),
    origination_dt      DATE            NOT NULL,
    maturity_dt         DATE            NOT NULL,
    principal_mm        NUMERIC(15,2)   NOT NULL,   -- $mm
    spread_bps          INT             NOT NULL,   -- spread over base rate (bps)
    base_rate_at_orig   NUMERIC(6,4)    NOT NULL,   -- SOFR / LIBOR at origination
    coupon_type         VARCHAR(10)     CHECK (coupon_type IN ('cash','pik','toggle')),
    pik_rate            NUMERIC(6,4)    DEFAULT 0,
    sector              VARCHAR(60),
    sub_sector          VARCHAR(80),
    sponsor             VARCHAR(100),
    ebitda_mm           NUMERIC(15,2),
    total_debt_mm       NUMERIC(15,2),
    senior_debt_mm      NUMERIC(15,2),
    equity_mm           NUMERIC(15,2),
    covenant_type       VARCHAR(20)     CHECK (covenant_type IN ('maintenance','incurrence','none')),
    covenant_threshold  NUMERIC(6,4),               -- e.g. 6.5x leverage covenant
    ltv                 NUMERIC(5,4),
    lien_position       VARCHAR(20)     DEFAULT 'first_lien',
    status              VARCHAR(20)     CHECK (status IN
                            ('current','pik_toggle','amended','extended',
                             'lme','default','paid_off','written_off')),
    recovery_rate       NUMERIC(5,4),
    notes               TEXT
);

-- -------------------------------------------------------
-- MACRO / RATE SCENARIOS
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS rate_scenarios (
    scenario_id     INT,
    scenario_name   VARCHAR(30),        -- 'base','stress','severe','rate_cut'
    effective_dt    DATE,
    sofr            NUMERIC(6,4),
    credit_spread_bps INT,              -- HY index OAS, bps
    gdp_growth      NUMERIC(6,4),
    recession_prob  NUMERIC(5,4),
    PRIMARY KEY (scenario_id, effective_dt)
);

-- -------------------------------------------------------
-- QUARTERLY PORTFOLIO SNAPSHOTS (for time-series)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id     SERIAL PRIMARY KEY,
    loan_id         INT REFERENCES loans(loan_id),
    snap_dt         DATE,
    fair_value_mm   NUMERIC(15,2),      -- mark-to-model value
    accrued_pik_mm  NUMERIC(15,2) DEFAULT 0,
    current_balance NUMERIC(15,2),      -- principal + accrued PIK
    current_sofr    NUMERIC(6,4),
    all_in_rate     NUMERIC(6,4),       -- spread + current_sofr
    icr             NUMERIC(8,4),       -- interest coverage ratio
    leverage_x      NUMERIC(6,2),       -- total debt / EBITDA
    status          VARCHAR(20),
    pik_flag        BOOLEAN DEFAULT FALSE
);

-- -------------------------------------------------------
-- FUND STRUCTURE (vehicle-level liquidity tracking)
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS funds (
    fund_id         SERIAL PRIMARY KEY,
    fund_name       VARCHAR(100),
    fund_type       VARCHAR(30),        -- 'bdc','semi_liquid','closed_end','clо'
    manager         VARCHAR(100),
    nav_mm          NUMERIC(15,2),
    liquid_buffer_pct NUMERIC(5,4),     -- % of NAV in liquid assets
    redemption_gate_pct NUMERIC(5,4),   -- max quarterly redemption allowed
    leverage_ratio  NUMERIC(6,4),       -- fund-level debt / equity
    lp_lockup_yrs   NUMERIC(4,1),
    retail_pct      NUMERIC(5,4),       -- % retail vs institutional LPs
    inception_dt    DATE
);

-- -------------------------------------------------------
-- REDEMPTION EVENT LOG
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS redemption_events (
    event_id        SERIAL PRIMARY KEY,
    fund_id         INT REFERENCES funds(fund_id),
    event_dt        DATE,
    requested_mm    NUMERIC(15,2),
    fulfilled_mm    NUMERIC(15,2),
    gated_mm        NUMERIC(15,2),
    nav_pre_mm      NUMERIC(15,2),
    nav_post_mm     NUMERIC(15,2),
    secondary_sales_mm NUMERIC(15,2) DEFAULT 0,
    haircut_pct     NUMERIC(5,4) DEFAULT 0,
    notes           TEXT
);

-- -------------------------------------------------------
-- PIK ACCRUAL LEDGER
-- -------------------------------------------------------
CREATE TABLE IF NOT EXISTS pik_accrual (
    accrual_id      SERIAL PRIMARY KEY,
    loan_id         INT REFERENCES loans(loan_id),
    period_start    DATE,
    period_end      DATE,
    cash_interest_mm    NUMERIC(15,2),
    pik_interest_mm     NUMERIC(15,2),
    cumulative_pik_mm   NUMERIC(15,2),
    toggle_reason   VARCHAR(100)        -- why cash was toggled to PIK
);

-- -------------------------------------------------------
-- INDEXES FOR PERFORMANCE
-- -------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_loans_sector ON loans(sector);
CREATE INDEX IF NOT EXISTS idx_loans_status ON loans(status);
CREATE INDEX IF NOT EXISTS idx_loans_vintage ON loans(origination_dt);
CREATE INDEX IF NOT EXISTS idx_loans_maturity ON loans(maturity_dt);
CREATE INDEX IF NOT EXISTS idx_snapshots_loan ON portfolio_snapshots(loan_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(snap_dt);
