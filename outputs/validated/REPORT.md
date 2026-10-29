# Credit risk: validated research results

50,000 synthetic loans; 10,103 loans in the holdout beginning 2022-03-01.

| Model | Holdout ROC AUC | Average precision | Brier score |
|---|---:|---:|---:|
| Logistic regression | 0.8768 | 0.4252 | 0.0754 |
| Random forest | 0.8819 | 0.4389 | 0.0649 |

## Interpretation

The constant training-prevalence baseline has Brier score 0.0773. Lower is better. Compare precision-recall performance with holdout prevalence 7.73%.

## Limitations

- Synthetic outcomes do not establish real-world predictive ability.
- Chronological split is by origination cohort. Outcome timestamps and observation horizons are not validated, so this is not a historical deployment backtest.
- The target is the supplied default flag across varied loan lifetimes, not a calibrated one-year probability of default.
- Stress results use supplied annual PD and LGD with illustrative shocks; they are separate from classifier scores.
- No model is selected or tuned on the holdout. Both prespecified models are reported.
- Bootstrap AUC intervals reflect resampling of the holdout only.
