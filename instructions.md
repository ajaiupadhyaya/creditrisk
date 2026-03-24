You are a senior credit risk data scientist. I have a credit risk analysis project 
in this workspace with the following files:

- credit_ratings.csv — credit rating/scoring data
- loanportfolio.csv — 50,000 loans across 10 sectors
- macro_stress_scenarios.csv — macroeconomic stress testing inputs
- portfolio_metrics.csv — aggregated portfolio-level KPIs
- vintage_analysis.csv — cohort/vintage performance over time

Your job: 
1. Read every CSV and print shape, dtypes, null counts, and the first 5 rows for each
2. Identify the likely target variable (default flag, loan status, charge-off, etc.)
3. Identify all features that map to: PD drivers, LGD drivers, EAD drivers, 
   sector/segment columns, date/vintage columns, and macro linkage columns
4. Flag any data quality issues: class imbalance ratio, suspicious nulls, 
   dtype mismatches, duplicate loan IDs
5. Output a structured data dictionary as a markdown table for each file

Do not write any models yet. Audit only.


Now run a full exploratory data analysis. Use Plotly for interactive charts and 
Seaborn/Matplotlib for static publication-quality charts. Save all charts to an 
/outputs/eda/ folder. Create the following visualizations:

PORTFOLIO OVERVIEW (loanportfolio.csv):
- Plotly sunburst: loan count and total exposure by sector → loan grade/rating
- Plotly histogram with KDE overlay: loan amount distribution, colored by default status
- Seaborn heatmap: Pearson correlation matrix of all numeric features, 
  annotated, diverging RdBu palette
- Plotly grouped bar: default rate by sector, sorted descending, 
  with error bars showing 95% CI

CREDIT RATINGS (credit_ratings.csv):
- Plotly sankey diagram: flow from credit rating bucket → loan status outcome
- Seaborn violin plot: interest rate distribution by rating grade, 
  overlaid with individual data points (stripplot)

TIME/VINTAGE (vintage_analysis.csv):
- Plotly line chart: cumulative default rate curves by vintage year cohort, 
  each cohort a separate line, on the same axis (classic vintage curve chart)
- Seaborn heatmap: vintage × months-on-book default rate matrix 
  (rows = origination year, columns = MOB, values = cum default rate %)

MACRO STRESS (macro_stress_scenarios.csv):
- Plotly multi-line chart: each stress scenario as a line, 
  key macro variables on separate subplots (unemployment, GDP, rates)

All charts must have: proper axis labels, titles, source annotations, 
color scales that are colorblind-safe. Plotly charts must use 
plotly.graph_objects (not express) for full control.


Engineer the following credit risk features and save to a processed dataframe. 
Document every transformation in comments:

FROM LOAN DATA:
- DTI buckets: <20%, 20-35%, 35-50%, >50% (labeled, not just numeric)
- Loan-to-income ratio (loan_amount / annual_income)
- Payment burden score: installment / (annual_income / 12)
- Delinquency severity score: weighted sum of 30/60/90 day delinquency flags
- Credit utilization flags: high (>75%), medium (30-75%), low (<30%)
- Log-transform all right-skewed numeric columns (loan_amount, income, etc.)
- One-hot encode: purpose, home_ownership, verification_status
- Target encode: sector/grade columns using 5-fold cross-val to prevent leakage

CROSS-FILE FEATURES:
- Join portfolio_metrics.csv: sector-level default rate as a feature on each loan row
  (this encodes sector risk context per loan)
- Join macro_stress_scenarios.csv: attach the base scenario macro vars 
  to each loan by origination_date if date columns exist

VALIDATION:
- Check for target leakage (any feature derived from post-default info)
- Print VIF scores for all numeric features — flag anything above 10
- Show class balance of target variable and print recommended class_weight param

Save final processed dataframe to /outputs/processed_loans.csv



Train a credit risk PD (Probability of Default) model. Use the processed dataframe 
from outputs/processed_loans.csv. Build the following pipeline:

MODELS TO TRAIN (all using 5-fold stratified CV):
1. Logistic Regression (L2, C=0.1) — baseline / regulatory interpretable
2. XGBoost classifier — primary model
3. LightGBM classifier — speed/accuracy comparison

EVALUATION — for each model generate these charts and save to /outputs/models/:

PLOTLY:
- ROC curve with AUC — all 3 models on one chart, color-coded
- Precision-Recall curve — all 3 models, annotated with average precision
- Calibration curve (reliability diagram) — predicted probability vs actual default rate
- Feature importance bar chart (top 25 features) — horizontal bars, 
  colored by feature category (borrower/loan/macro/engineered)
- SHAP summary plot for XGBoost (beeswarm) — use shap library, 
  save as static PNG at 200dpi

SEABORN:
- Confusion matrix heatmap at optimal threshold (Youden's J) for each model
- Score distribution: overlapping histograms of predicted probability 
  for defaults vs non-defaults, both models, showing separation

METRICS TABLE:
Print a pandas DataFrame with: AUC-ROC, Gini coefficient, KS statistic, 
Log Loss, Brier Score, F1 at optimal threshold — for all 3 models

Use class_weight='balanced' for logistic regression, 
scale_pos_weight for XGBoost/LightGBM based on class imbalance ratio.

Now run portfolio-level credit risk analytics using the trained XGBoost model 
and macro_stress_scenarios.csv. 

STRESS TESTING:
- Apply each macro scenario (base, adverse, severely adverse) by adjusting 
  PD predictions using a macro sensitivity multiplier derived from 
  macro_stress_scenarios.csv variables
- For each scenario, compute: expected loss (EL = PD × LGD × EAD), 
  total portfolio EL in $, EL by sector
- Assume LGD = 0.45 for unsecured, 0.25 for secured (flag if collateral 
  column exists in data)

VISUALIZATIONS (save to /outputs/stress/):

PLOTLY:
- Grouped bar chart: EL by sector under base vs adverse vs severely adverse — 
  side by side, color-coded by scenario severity
- Waterfall chart: starting from base EL, show incremental EL contribution 
  by each macro variable shift (tornado/waterfall style)
- Bubble chart: each sector as a bubble — x=current default rate, 
  y=stressed default rate uplift, bubble size=exposure $, 
  color=sector — include hover tooltips with full sector detail

SEABORN:
- Heatmap: sector × scenario EL matrix, values as % of exposure, 
  annotated with exact numbers, red gradient

VINTAGE STRESS:
- Using vintage_analysis.csv, plot how each vintage cohort's cumulative 
  default curve shifts under the adverse scenario
- Overlay on the original vintage curves from EDA (use Plotly, same style)


Create a single-file interactive HTML dashboard using Plotly Dash or 
a standalone Plotly HTML export that combines:

1. Portfolio summary: total exposure, WA interest rate, WA LTV, default rate KPIs as cards
2. Sector breakdown: sunburst + default rate bar (from EDA)
3. Model performance: ROC + PR curves
4. Stress test results: EL by scenario grouped bar
5. Vintage curves

Export as /outputs/dashboard.html — must be fully self-contained 
(no external CDN dependencies) so it opens offline.
Style it professionally: dark theme, Anthropic/Bloomberg terminal aesthetic.