# nscand

Standard candidate family (101 graphs), N=9, alpha=0.1. fa = false adjacencies (pairs adjacent in the candidate, non-adjacent in the true marginal over V); miss = missing adjacencies.

| scenario | method | n | runs | tau | tau_zero | gstar | admitted | empty | fa_ghat | fa_pool_mean | fa_adm_mean | fa_adm_max | adm_zero_fa | miss_adm_mean | ghat_flag | adj_prec | adj_rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hubs | fci | 1600 | 15 | 0.126 | 0.000 | 0.333 | 0.799 | 0.000 | 0.333 | 1.265 | 1.141 | 4.200 | 0.371 | 35.493 | 0.067 | 1.000 | 0.410 |
| hubs | fci | 25600 | 15 | 0.080 | 0.000 | 0.267 | 0.110 | 0.667 | 0.600 | 1.363 | 1.026 | 2.800 | 0.469 | 12.178 | 0.733 | 0.985 | 0.228 |
| hubs | pc | 1600 | 15 | 0.128 | 0.000 | 0.333 | 0.172 | 0.267 | 0.467 | 1.416 | 0.704 | 2.182 | 0.568 | 35.153 | 0.733 | 0.992 | 0.302 |
| hubs | pc | 25600 | 15 | 0.083 | 0.000 | 0.333 | 0.003 | 0.933 | 1.133 | 2.009 | 0.000 | 0.000 | 1.000 | 4.200 | 1.000 | 1.000 | 0.054 |
