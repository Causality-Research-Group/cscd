# kscope

Hard knowledge (30% of orderable pairs as forbidden edges; rev30: 30% of them reversed). all = every discovery uses K (paper); candidates = full-variable runs only, subset graphs without K; local = subset graphs only, candidates without K. ghat_flag = CSCD native flag on G-hat; auroc_vs_none = AUROC of the CSCD margin against the no-knowledge runs.

| arm | method | runs | gstar | admitted | ghat_flag | boot_flag | auroc_vs_none | ghat_skel_err | exceed | adj_prec | adj_rec | dir_prec | dir_rec | single_dir_prec | single_dir_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kscope_none | pc | 20 | 1.00 [0.84, 1.00] | 0.594 | 0.05 [0.01, 0.24] | 0.000 | nan | 2.950 | 0.145 | 0.981 | 0.877 | 0.932 | 0.469 | 0.832 | 0.581 |
| kscope_none | fci | 20 | 1.00 [0.84, 1.00] | 0.774 | 0.00 [0.00, 0.16] | 0.000 | nan | 2.450 | 0.130 | 0.995 | 0.909 | 1.000 | 0.048 | 0.982 | 0.118 |
| kscope_all_rev0 | pc | 20 | 1.00 [0.84, 1.00] | 0.636 | 0.05 [0.01, 0.24] | 0.000 | 0.461 | 2.950 | 0.140 | 0.976 | 0.922 | 0.958 | 0.588 | 0.878 | 0.661 |
| kscope_all_rev0 | fci | 20 | 1.00 [0.84, 1.00] | 0.754 | 0.00 [0.00, 0.16] | 0.000 | 0.536 | 2.450 | 0.125 | 0.995 | 0.912 | 1.000 | 0.050 | 0.982 | 0.118 |
| kscope_all_rev30 | pc | 20 | 1.00 [0.84, 1.00] | 0.608 | 0.05 [0.01, 0.24] | 0.000 | 0.521 | 2.950 | 0.140 | 0.977 | 0.919 | 0.921 | 0.521 | 0.839 | 0.598 |
| kscope_all_rev30 | fci | 20 | 1.00 [0.84, 1.00] | 0.712 | 0.00 [0.00, 0.16] | 0.000 | 0.578 | 2.450 | 0.130 | 0.995 | 0.909 | 1.000 | 0.043 | 0.981 | 0.104 |
| kscope_candidates_rev0 | pc | 20 | 1.00 [0.84, 1.00] | 0.634 | 0.05 [0.01, 0.24] | 0.000 | 0.454 | 2.950 | 0.145 | 0.974 | 0.922 | 0.958 | 0.588 | 0.878 | 0.661 |
| kscope_candidates_rev0 | fci | 20 | 1.00 [0.84, 1.00] | 0.747 | 0.00 [0.00, 0.16] | 0.000 | 0.544 | 2.450 | 0.130 | 0.995 | 0.912 | 1.000 | 0.050 | 0.982 | 0.118 |
| kscope_candidates_rev30 | pc | 20 | 1.00 [0.84, 1.00] | 0.596 | 0.05 [0.01, 0.24] | 0.000 | 0.511 | 2.950 | 0.145 | 0.974 | 0.915 | 0.897 | 0.527 | 0.826 | 0.617 |
| kscope_candidates_rev30 | fci | 20 | 1.00 [0.84, 1.00] | 0.674 | 0.05 [0.01, 0.24] | 0.000 | 0.651 | 2.450 | 0.130 | 0.990 | 0.909 | 1.000 | 0.036 | 0.987 | 0.098 |
