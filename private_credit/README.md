# Private credit fragility track

Thesis: *private credit can look stable while liquidity mismatch, macro drift, and shadow distress build beneath the surface.*

This folder is a **separate analytical line** from the root Kaggle-style `loan_portfolio.csv` pipeline (`src/`, `notebooks/`). It uses a **600-loan synthetic book** with its own schema and outputs.

## Layout

```
private_credit/
├── README.md
├── run_pipeline.py          # generator → credit models → ML → fragility HTML
├── sql/
│   ├── 01_schema.sql        # PostgreSQL DDL (optional warehouse)
│   └── 02_analytics_queries.sql
├── python/
│   ├── paths.py             # DATA_DIR; override with PRIVATE_CREDIT_DATA_DIR
│   ├── 01_data_generator.py
│   ├── 02_credit_models.py
│   └── 03_ml_classifier.py
├── data/                    # generated CSVs (see .gitignore)
├── excel/
│   └── private_credit_fragility.xlsx   # bundled workbook (static artifact)
```

## Dependencies

Use the repo [requirements.txt](../requirements.txt) (pandas, numpy, scipy, scikit-learn). Run from a venv that has those installed.

`run_pipeline.py` looks for a project [`.venv`](../.venv): if you launch it with plain `python` (system interpreter) but `.venv` exists, it **re-executes** itself under `.venv/bin/python` so dependencies match `make private-credit`. Override with `PRIVATE_CREDIT_PYTHON=/path/to/python` to skip auto-selection.

## Run from repository root

```bash
# Option A — full pipeline + fragility HTML report (uses .venv automatically when present)
python private_credit/run_pipeline.py

# Option B — step by step (use .venv/bin/python if system Python has no deps)
.venv/bin/python private_credit/python/01_data_generator.py
.venv/bin/python private_credit/python/02_credit_models.py
.venv/bin/python private_credit/python/03_ml_classifier.py
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from private_credit_report import build_fragility_report; build_fragility_report()"
```

Outputs:

- CSVs under `private_credit/data/` (`loans.csv`, `rate_scenarios.csv`, `funds.csv`, `fragility_scores.csv`, `ml_predictions.csv`)
- `outputs/private_credit/fragility_summary.html` (after `run_pipeline.py` or `build_fragility_report()`) — a multi-section **analytic atlas**: Plotly (tail risk, ICR ladders, heatmaps, maturity wall, macro tape, ML lift, fund stress, animated loss slider), Seaborn/Matplotlib static panels (KDE, joint plot, PIK path, correlations), and a D3.js animated NAV path chart for fund liquidity.

The main hub [outputs/dashboard.html](../outputs/dashboard.html) links to the fragility report when that file exists.

`02_credit_models.py` also writes `mc_summary.csv`, `mc_loss_samples.csv`, `icr_scenario_summary.csv`, `sector_icr.csv`, `maturity_wall.csv`, `pik_trajectory_example.csv`, `fund_liquidity.csv`, and `fund_nav_paths.csv` under `private_credit/data/` for the report (re-run credit models after changing the generator).

## Excel workbook

`excel/private_credit_fragility.xlsx` is included as a static model artifact. Automated `build_excel.py` / `recalc.py` scripts are **not** part of this repo; refresh the workbook outside the pipeline if needed.

## SQL analytics

`sql/01_schema.sql` and `sql/02_analytics_queries.sql` target **PostgreSQL**. To use them:

1. Create a database and run `psql -f private_credit/sql/01_schema.sql`.
2. Load tables from the CSVs (column names align with `loans.csv` / `funds.csv` where applicable) or use your ETL of choice.
3. Run individual queries from `02_analytics_queries.sql` in your client.

A DuckDB loader is intentionally omitted to keep dependencies minimal; add one locally if you want file-native SQL without Postgres.

## ML notes

The classifier uses **sklearn `GradientBoostingClassifier`** with calibrated probabilities. **SHAP** is not invoked in `03_ml_classifier.py`; for SHAP-style interpretability use the main pipeline’s boosting models under `outputs/models/`.

## Data directory override

```bash
export PRIVATE_CREDIT_DATA_DIR=/path/to/writable/dir
python private_credit/python/01_data_generator.py
```
