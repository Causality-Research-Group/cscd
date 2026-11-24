# nosub

Native flag rates (Wilson 95%) with each detector's own conformal threshold at alpha=0.1, and AUROC/power of its margin against the control scenario.

| scenario | method | detector | runs | flag_scenario | flag_control | auroc | power_at_10pct | skel_err_scen | ghat_reproduced |
|---|---|---|---|---|---|---|---|---|---|
| Hidden hubs | pc | CSCD (subsets + calibration) | 30 | 0.90 [0.74, 0.97] | 0.07 [0.02, 0.21] | 0.966 | 0.967 | 0.191 | 1.000 |
| Hidden hubs | pc | No subsets, full-data run | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.117 | 0.000 | 0.191 | 1.000 |
| Hidden hubs | pc | No subsets, split D_train | 30 | 0.00 [0.00, 0.11] | 0.07 [0.02, 0.21] | 0.138 | 0.000 | 0.191 | 1.000 |
| Hidden hubs | ges | CSCD (subsets + calibration) | 30 | 0.87 [0.70, 0.95] | 0.17 [0.07, 0.34] | 0.904 | 0.800 | 0.177 | 1.000 |
| Hidden hubs | ges | No subsets, full-data run | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.084 | 0.000 | 0.177 | 1.000 |
| Hidden hubs | ges | No subsets, split D_train | 30 | 0.03 [0.01, 0.17] | 0.00 [0.00, 0.11] | 0.184 | 0.067 | 0.177 | 1.000 |
| Hidden hubs | fci | CSCD (subsets + calibration) | 30 | 0.33 [0.19, 0.51] | 0.00 [0.00, 0.11] | 0.643 | 0.433 | 0.191 | 0.967 |
| Hidden hubs | fci | No subsets, full-data run | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.108 | 0.033 | 0.191 | 0.967 |
| Hidden hubs | fci | No subsets, split D_train | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.205 | 0.033 | 0.191 | 0.967 |
| p=50, n=1000 | pc | CSCD (subsets + calibration) | 20 | 0.15 [0.05, 0.36] | 0.00 [0.00, 0.16] | 0.917 | 0.750 | 0.006 | 1.000 |
| p=50, n=1000 | pc | No subsets, full-data run | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.995 | 0.950 | 0.006 | 1.000 |
| p=50, n=1000 | pc | No subsets, split D_train | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.895 | 0.450 | 0.006 | 1.000 |
| p=50, n=1000 | ges | CSCD (subsets + calibration) | 20 | 0.90 [0.70, 0.97] | 0.05 [0.01, 0.24] | 0.993 | 0.950 | 0.018 | 1.000 |
| p=50, n=1000 | ges | No subsets, full-data run | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.875 | 0.150 | 0.018 | 1.000 |
| p=50, n=1000 | ges | No subsets, split D_train | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.850 | 0.350 | 0.018 | 1.000 |
| p=50, n=1000 | fci | CSCD (subsets + calibration) | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.820 | 0.300 | 0.006 | 0.900 |
| p=50, n=1000 | fci | No subsets, full-data run | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.915 | 0.200 | 0.006 | 0.900 |
| p=50, n=1000 | fci | No subsets, split D_train | 20 | 0.00 [0.00, 0.16] | 0.00 [0.00, 0.16] | 0.975 | 0.950 | 0.006 | 0.900 |
| Quadratic (shared) | pc | CSCD (subsets + calibration) | 30 | 0.00 [0.00, 0.11] | 0.07 [0.02, 0.21] | 0.184 | 0.000 | 0.074 | 1.000 |
| Quadratic (shared) | pc | No subsets, full-data run | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.215 | 0.000 | 0.074 | 1.000 |
| Quadratic (shared) | pc | No subsets, split D_train | 30 | 0.07 [0.02, 0.21] | 0.07 [0.02, 0.21] | 0.297 | 0.067 | 0.074 | 1.000 |
| Quadratic (shared) | fci | CSCD (subsets + calibration) | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.024 | 0.000 | 0.076 | 1.000 |
| Quadratic (shared) | fci | No subsets, full-data run | 30 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 0.206 | 0.067 | 0.076 | 1.000 |
| Quadratic (shared) | fci | No subsets, split D_train | 30 | 0.10 [0.03, 0.26] | 0.00 [0.00, 0.11] | 0.443 | 0.133 | 0.076 | 1.000 |
