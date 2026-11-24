# sinks

hubs_v3: p=26, degree 3, 6 largest-out-degree variables hidden. sinks_v3: the same DAGs and data seeds, 6 sinks hidden (no hidden confounding, same density, 20 observed variables). Flags with each rule's own threshold (Wilson 95%).

| method | setting | runs | ghat_skel_err | cscd_flag | boot_flag | gstar | admitted | exceed |
|---|---|---|---|---|---|---|---|---|
| pc | hidden hubs (paper) | 30 | 36.000 | 0.90 [0.74, 0.97] | 0.00 [0.00, 0.11] | 0.333 | 0.043 | 0.090 |
| pc | hidden sinks (paired control) | 30 | 4.133 | 0.47 [0.30, 0.64] | 0.00 [0.00, 0.11] | 0.700 | 0.298 | 0.100 |
| pc | main setting (paper's control) | 30 | 2.767 | 0.07 [0.02, 0.21] | 0.00 [0.00, 0.11] | 1.000 | 0.572 | 0.070 |
| pc | AUROC hubs vs sinks: CSCD margin / bootstrap | 30 | nan | 0.87 | 0.37 | nan | nan | nan |
| ges | hidden hubs (paper) | 30 | 33.667 | 0.87 [0.70, 0.95] | 0.00 [0.00, 0.11] | 0.967 | 0.054 | 0.070 |
| ges | hidden sinks (paired control) | 30 | 5.867 | 0.30 [0.17, 0.48] | 0.00 [0.00, 0.11] | 1.000 | 0.325 | 0.093 |
| ges | main setting (paper's control) | 30 | 3.000 | 0.17 [0.07, 0.34] | 0.00 [0.00, 0.11] | 1.000 | 0.389 | 0.077 |
| ges | AUROC hubs vs sinks: CSCD margin / bootstrap | 30 | nan | 0.86 | 0.19 | nan | nan | nan |
| fci | hidden hubs (paper) | 30 | 36.633 | 0.33 [0.19, 0.51] | 0.00 [0.00, 0.11] | 0.433 | 0.587 | 0.100 |
| fci | hidden sinks (paired control) | 30 | 4.400 | 0.13 [0.05, 0.30] | 0.00 [0.00, 0.11] | 0.667 | 0.621 | 0.050 |
| fci | main setting (paper's control) | 30 | 2.300 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.11] | 1.000 | 0.828 | 0.110 |
| fci | AUROC hubs vs sinks: CSCD margin / bootstrap | 30 | nan | 0.54 | 0.37 | nan | nan | nan |
