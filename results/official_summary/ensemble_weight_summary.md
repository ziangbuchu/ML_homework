# Ensemble Weight Sweep Summary

Weights are selected on validation MSE for each horizon/seed. Test-oracle rows are diagnostic only and are not used for model selection.

| Horizon | Rule | Test MSE mean±std | Test MAE mean±std | Selected weights |
| --- | --- | ---: | ---: | --- |
| 90 | equal 0.5 | 0.374842±0.021491 | 0.476893±0.012027 | 0.5, 0.5, 0.5, 0.5, 0.5 |
| 90 | val-selected | 0.446209±0.043318 | 0.518641±0.027420 | 0.7, 1.0, 1.0, 1.0, 1.0 |
| 90 | test-oracle diagnostic | 0.374186±0.021537 | 0.475724±0.011638 | 0.5, 0.5, 0.6, 0.6, 0.5 |
| 365 | equal 0.5 | 0.444563±0.034875 | 0.516382±0.022740 | 0.5, 0.5, 0.5, 0.5, 0.5 |
| 365 | val-selected | 0.442416±0.017311 | 0.513660±0.010671 | 0.1, 0.2, 0.4, 0.3, 0.3 |
| 365 | test-oracle diagnostic | 0.429071±0.012882 | 0.504761±0.008479 | 0.4, 0.4, 0.7, 0.6, 0.0 |
