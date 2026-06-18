# Naive Baseline Summary

Deterministic baselines use only the target feature inside the 90-day input window. They are not trained and are reported as sanity checks, not as neural baselines.

| Model | Horizon | Test MSE | Test MAE |
| --- | ---: | ---: | ---: |
| input_mean | 90 | 0.736495 | 0.711796 |
| last_value | 90 | 0.923572 | 0.757890 |
| input_mean | 365 | 1.036502 | 0.817853 |
| last_value | 365 | 1.431294 | 0.960546 |
