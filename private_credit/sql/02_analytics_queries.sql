-- =============================================================================
-- PRIVATE CREDIT FRAGILITY PROJECT — ANALYTICS QUERIES
-- Run against the schema in 01_schema.sql
-- =============================================================================

-- -------------------------------------------------------
-- 1. DEFAULT RATE RECONCILIATION
--    The 1.8% (cash default) vs 9.2% (true distress) gap
-- -------------------------------------------------------
WITH distress_classification AS (
    SELECT
        loan_id,
        principal_mm,
        status,
        CASE
            WHEN status = 'default'                       THEN 'cash_default'
            WHEN status IN ('pik_toggle','amended','extended','lme') THEN 'shadow_default'
            ELSE 'performing'
        END AS distress_class
    FROM loans
),
summary AS (
    SELECT
        distress_class,
        COUNT(*)                            AS loan_count,
        SUM(principal_mm)                   AS notional_mm,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_by_count,
        ROUND(SUM(principal_mm) * 100.0 / SUM(SUM(principal_mm)) OVER (), 2) AS pct_by_notional
    FROM distress_classification
    GROUP BY distress_class
)
SELECT * FROM summary
ORDER BY pct_by_notional DESC;

-- -------------------------------------------------------
-- 2. MATURITY WALL — 18-MONTH REFINANCING RISK
-- -------------------------------------------------------
SELECT
    DATE_TRUNC('quarter', maturity_dt)          AS maturity_quarter,
    COUNT(*)                                    AS loans_maturing,
    ROUND(SUM(principal_mm), 1)                 AS total_principal_mm,
    ROUND(AVG(spread_bps), 0)                   AS avg_spread_bps,
    COUNT(*) FILTER (WHERE status != 'current') AS already_distressed,
    ROUND(SUM(principal_mm)
        FILTER (WHERE coupon_type = 'pik'), 1)  AS pik_balance_mm
FROM loans
WHERE maturity_dt BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '24 months'
GROUP BY DATE_TRUNC('quarter', maturity_dt)
ORDER BY maturity_quarter;

-- -------------------------------------------------------
-- 3. ICR DISTRIBUTION BY SECTOR (stress thermometer)
-- -------------------------------------------------------
WITH icr_calc AS (
    SELECT
        l.loan_id,
        l.sector,
        l.origination_dt,
        l.principal_mm,
        l.ebitda_mm,
        l.total_debt_mm,
        l.spread_bps,
        -- Apply current SOFR (4.3%) to floating rate book
        (l.spread_bps / 10000.0 + 0.043)       AS all_in_rate,
        l.ebitda_mm / NULLIF(
            l.total_debt_mm * (l.spread_bps / 10000.0 + 0.043), 0
        )                                        AS icr,
        l.total_debt_mm / NULLIF(l.ebitda_mm, 0) AS leverage_x
    FROM loans l
    WHERE l.status NOT IN ('paid_off','written_off')
),
icr_buckets AS (
    SELECT
        sector,
        COUNT(*) FILTER (WHERE icr < 1.0)   AS icr_below_1x,
        COUNT(*) FILTER (WHERE icr >= 1.0
                           AND icr < 1.5)   AS icr_1x_to_1_5x,
        COUNT(*) FILTER (WHERE icr >= 1.5
                           AND icr < 2.0)   AS icr_1_5x_to_2x,
        COUNT(*) FILTER (WHERE icr >= 2.0)  AS icr_above_2x,
        COUNT(*)                            AS total_loans,
        ROUND(AVG(icr), 2)                  AS avg_icr,
        ROUND(AVG(leverage_x), 2)           AS avg_leverage_x,
        ROUND(SUM(principal_mm) FILTER (WHERE icr < 1.5)
            / SUM(principal_mm), 3)         AS pct_stressed_by_notional
    FROM icr_calc
    GROUP BY sector
)
SELECT
    sector,
    total_loans,
    avg_icr,
    avg_leverage_x,
    icr_below_1x,
    icr_1x_to_1_5x,
    icr_1_5x_to_2x,
    icr_above_2x,
    ROUND((icr_below_1x + icr_1x_to_1_5x)::NUMERIC / NULLIF(total_loans, 0), 3)
        AS stressed_pct,
    pct_stressed_by_notional
FROM icr_buckets
ORDER BY avg_icr ASC;

-- -------------------------------------------------------
-- 4. PIK CONCENTRATION BY VINTAGE (shadow default signal)
-- -------------------------------------------------------
SELECT
    EXTRACT(YEAR FROM origination_dt)       AS vintage,
    COUNT(*)                                AS total_loans,
    COUNT(*) FILTER (WHERE coupon_type = 'pik' OR status = 'pik_toggle')
                                            AS pik_loans,
    ROUND(SUM(principal_mm) FILTER (WHERE coupon_type = 'pik'), 1)
                                            AS pik_principal_mm,
    ROUND(AVG(spread_bps), 0)              AS avg_spread_bps,
    ROUND(
        COUNT(*) FILTER (WHERE coupon_type = 'pik' OR status = 'pik_toggle')
        * 100.0 / NULLIF(COUNT(*), 0), 1
    )                                       AS pik_rate_pct,
    ROUND(
        COUNT(*) FILTER (WHERE status IN
            ('pik_toggle','amended','extended','lme','default'))
        * 100.0 / NULLIF(COUNT(*), 0), 1
    )                                       AS true_distress_rate_pct
FROM loans
GROUP BY EXTRACT(YEAR FROM origination_dt)
ORDER BY vintage;

-- -------------------------------------------------------
-- 5. COVENANT QUALITY ANALYSIS
-- -------------------------------------------------------
SELECT
    covenant_type,
    COUNT(*)                                    AS loans,
    ROUND(SUM(principal_mm), 0)                 AS total_notional_mm,
    ROUND(AVG(spread_bps), 0)                   AS avg_spread_bps,
    ROUND(AVG(
        l.total_debt_mm / NULLIF(l.ebitda_mm, 0)
    ), 2)                                       AS avg_leverage_x,
    COUNT(*) FILTER (WHERE status IN
        ('pik_toggle','amended','extended','lme','default'))
                                                AS distressed_count,
    ROUND(
        COUNT(*) FILTER (WHERE status IN
            ('pik_toggle','amended','extended','lme','default'))
        * 100.0 / NULLIF(COUNT(*), 0), 1
    )                                           AS distress_rate_pct
FROM loans l
GROUP BY covenant_type;

-- -------------------------------------------------------
-- 6. FUND LIQUIDITY STRESS — GATE RISK DASHBOARD
-- -------------------------------------------------------
SELECT
    f.fund_name,
    f.fund_type,
    f.nav_mm,
    f.liquid_buffer_pct,
    ROUND(f.nav_mm * f.liquid_buffer_pct, 1)    AS liquid_buffer_mm,
    f.redemption_gate_pct,
    ROUND(f.nav_mm * f.redemption_gate_pct, 1)  AS max_quarterly_redemption_mm,
    -- Redemption stress: how many quarters until buffer depleted at 11% requests/qtr
    ROUND(
        (f.nav_mm * f.liquid_buffer_pct)
        / NULLIF(f.nav_mm * GREATEST(0.11 - f.redemption_gate_pct, 0), 0)
    , 1)                                         AS quarters_until_depletion_11pct_shock,
    f.retail_pct,
    ROUND(f.nav_mm * f.retail_pct, 0)           AS retail_exposure_mm
FROM funds f
ORDER BY quarters_until_depletion_11pct_shock ASC;

-- -------------------------------------------------------
-- 7. SECTOR CONCENTRATION (HHI) — SYSTEMIC RISK
-- -------------------------------------------------------
WITH sector_share AS (
    SELECT
        sector,
        SUM(principal_mm) AS sector_notional,
        SUM(SUM(principal_mm)) OVER () AS total_notional
    FROM loans
    WHERE status NOT IN ('paid_off','written_off')
    GROUP BY sector
),
hhi AS (
    SELECT
        sector,
        ROUND(sector_notional, 0)           AS notional_mm,
        ROUND(sector_notional / total_notional, 4) AS share,
        ROUND(POWER(sector_notional / total_notional, 2), 6) AS hhi_component
    FROM sector_share
)
SELECT
    sector,
    notional_mm,
    ROUND(share * 100, 1) AS share_pct,
    ROUND(SUM(hhi_component) OVER () * 10000, 0) AS portfolio_hhi
FROM hhi
ORDER BY share DESC;

-- -------------------------------------------------------
-- 8. RATE SENSITIVITY — SOFR SHOCK IMPACT ON ICR
-- -------------------------------------------------------
WITH base_icr AS (
    SELECT
        loan_id, sector, principal_mm, ebitda_mm, total_debt_mm, spread_bps,
        ebitda_mm / NULLIF(total_debt_mm * (spread_bps/10000.0 + 0.043), 0) AS icr_current,
        ebitda_mm / NULLIF(total_debt_mm * (spread_bps/10000.0 + 0.053), 0) AS icr_plus100,
        ebitda_mm / NULLIF(total_debt_mm * (spread_bps/10000.0 + 0.023), 0) AS icr_minus200
    FROM loans
    WHERE status NOT IN ('paid_off','written_off')
)
SELECT
    'Current (SOFR 4.3%)'  AS scenario,
    ROUND(AVG(icr_current), 2)  AS avg_icr,
    ROUND(COUNT(*) FILTER (WHERE icr_current < 1.0) * 100.0 / COUNT(*), 1) AS pct_below_1x,
    ROUND(COUNT(*) FILTER (WHERE icr_current < 1.5) * 100.0 / COUNT(*), 1) AS pct_below_1_5x
FROM base_icr
UNION ALL
SELECT
    'Stress (+100bps)',
    ROUND(AVG(icr_plus100), 2),
    ROUND(COUNT(*) FILTER (WHERE icr_plus100 < 1.0) * 100.0 / COUNT(*), 1),
    ROUND(COUNT(*) FILTER (WHERE icr_plus100 < 1.5) * 100.0 / COUNT(*), 1)
FROM base_icr
UNION ALL
SELECT
    'Rate Cut (-200bps)',
    ROUND(AVG(icr_minus200), 2),
    ROUND(COUNT(*) FILTER (WHERE icr_minus200 < 1.0) * 100.0 / COUNT(*), 1),
    ROUND(COUNT(*) FILTER (WHERE icr_minus200 < 1.5) * 100.0 / COUNT(*), 1)
FROM base_icr;

-- -------------------------------------------------------
-- 9. LME / AMENDMENT ACTIVITY TRACKER
-- -------------------------------------------------------
SELECT
    DATE_TRUNC('quarter', origination_dt)   AS cohort_quarter,
    COUNT(*) FILTER (WHERE status = 'lme')  AS lme_count,
    COUNT(*) FILTER (WHERE status = 'amended') AS amendments,
    COUNT(*) FILTER (WHERE status = 'extended') AS extensions,
    COUNT(*) FILTER (WHERE status = 'default') AS hard_defaults,
    COUNT(*)                                AS total_cohort,
    ROUND((
        COUNT(*) FILTER (WHERE status IN ('lme','amended','extended','default'))
    ) * 100.0 / NULLIF(COUNT(*), 0), 1)     AS total_distress_rate_pct
FROM loans
GROUP BY DATE_TRUNC('quarter', origination_dt)
ORDER BY cohort_quarter;

-- -------------------------------------------------------
-- 10. COMPOSITE FRAGILITY SCORE BY LOAN
-- -------------------------------------------------------
WITH scored AS (
    SELECT
        loan_id,
        borrower_name,
        sector,
        principal_mm,
        status,
        -- ICR component (0-25 pts, lower ICR = higher score)
        LEAST(25, GREATEST(0,
            25 * (1 - LEAST(
                l.ebitda_mm / NULLIF(l.total_debt_mm * (l.spread_bps/10000.0 + 0.043), 0)
                / 2.5, 1
            ))
        )) AS icr_score,
        -- Leverage component (0-25 pts)
        LEAST(25, GREATEST(0,
            (l.total_debt_mm / NULLIF(l.ebitda_mm, 0) - 4) * 4
        )) AS leverage_score,
        -- PIK component (0-20 pts)
        CASE WHEN coupon_type = 'pik' THEN 20
             WHEN status = 'pik_toggle' THEN 15
             ELSE 0 END AS pik_score,
        -- Covenant component (0-15 pts)
        CASE WHEN covenant_type = 'none'        THEN 15
             WHEN covenant_type = 'incurrence'  THEN 8
             ELSE 0 END AS covenant_score,
        -- Maturity proximity (0-15 pts)
        CASE WHEN maturity_dt < CURRENT_DATE + INTERVAL '12 months' THEN 15
             WHEN maturity_dt < CURRENT_DATE + INTERVAL '24 months' THEN 8
             ELSE 0 END AS maturity_score
    FROM loans l
    WHERE status NOT IN ('paid_off','written_off')
)
SELECT
    loan_id,
    borrower_name,
    sector,
    principal_mm,
    status,
    ROUND(icr_score + leverage_score + pik_score + covenant_score + maturity_score, 1)
        AS fragility_score,   -- 0 = safe, 100 = maximum fragility
    CASE
        WHEN (icr_score + leverage_score + pik_score + covenant_score + maturity_score) >= 65 THEN 'CRITICAL'
        WHEN (icr_score + leverage_score + pik_score + covenant_score + maturity_score) >= 45 THEN 'HIGH'
        WHEN (icr_score + leverage_score + pik_score + covenant_score + maturity_score) >= 25 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_tier
FROM scored
ORDER BY fragility_score DESC;
