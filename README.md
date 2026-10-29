# Credit risk, without the shortcuts

A reproducible portfolio case study of 50,000 synthetic loans. It combines a chronological holdout, an explicit pre-outcome feature allowlist, probability diagnostics, concentration analysis, and expected-loss sensitivity.

**Start with [the evidence report](outputs/validated/REPORT.html)** or the [Markdown findings](outputs/validated/REPORT.md). The companion `privatecredit` project packages these results in the interactive **Credit Observatory**.

## Results

| Prespecified model | Holdout ROC AUC | Brier score |
|---|---:|---:|
| Logistic regression | 0.8768 | 0.0754 |
| Random forest | 0.8819 | 0.0649 |
| Constant training-prevalence baseline | — | 0.0773 |

39,897 training loans; 10,103 held-out loans from origination cohorts beginning March 2022. Confidence intervals, average precision, calibration points, and exact environment versions are in [summary.json](outputs/validated/summary.json).

These are synthetic classification results, not evidence of real-world default prediction. Loan lifetimes vary, so classifier outputs are not annual PD estimates. Origination-cohort separation does not establish a historical deployment backtest because outcome availability dates and censoring have not been validated.

## Reproduce

Python 3.11 is the reference runtime. The minimal portfolio environment is separate from the larger legacy notebook environment.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-portfolio.txt
.venv/bin/python run_analysis.py
.venv/bin/python -m pytest tests -q
```

No credentials or network data retrieval are needed after dependencies are installed. The pipeline reads `loan_portfolio.csv`, fits preprocessing only on training rows, reports both prespecified models, and writes:

- `outputs/validated/REPORT.html` — standalone, printable evidence report.
- `outputs/validated/REPORT.md` — compact findings and limitations.
- `outputs/validated/summary.json` — aggregate metrics, curves, assumptions, hashes, versions.
- `outputs/validated/holdout_predictions.csv` — local row-level evaluation evidence.
- `outputs/validated/sector_summary.csv` and `stress_summary.csv`.

The same input and reference environment produce deterministic results. No model is selected or tuned on the holdout. The 200-resample bootstrap describes holdout sampling uncertainty only.

## Why the redesign matters

The original exploratory path used supplied risk estimates and derived features, and encoded targets before downstream validation. Its near-perfect scores are unsuitable for claims about independent predictive performance. The canonical pipeline now uses an explicit feature allowlist and a single chronological holdout. See [AUDIT.md](AUDIT.md).

The annual stress calculation is deliberately separate: `sum(EAD × supplied annual PD × supplied LGD)`, with disclosed scenario multipliers and bounded inputs. It does not convert lifetime classifier scores into annual pricing signals.

## Publication scope

The original README identifies the input as a synthetic Kaggle dataset, but the original listing and license were not recorded. The public portfolio therefore contains aggregate results and methods, not raw loan rows, identifiers, or row-level predictions. The local CSV remains untouched. Confirm redistribution rights before publishing the raw dataset or this whole repository.

## Repository map

| Location | Status |
|---|---|
| `run_analysis.py`, `report.py`, `tests/` | Canonical portfolio implementation |
| `outputs/validated/` | Regenerated evidence |
| `src/`, `notebooks/`, old `outputs/` | Historical exploratory work; not the validated result |
| `private_credit/` | Separate legacy synthetic fragility experiment; excluded from portfolio evidence |
| `PROJECT_SUMMARY.md`, `PHASE_6_SUMMARY.md` | Historical project notes; superseded by this README |

The legacy 600-loan fragility track and static workbook are preserved for reference. Their numerical claims have not been revalidated and must not be presented as completed portfolio evidence.
