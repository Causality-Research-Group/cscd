# hubdose2_gating

Pooled over 0-6 hidden hubs (skeleton errors in pairs).

| method | rule | runs | flagged | skel_err_all | skel_err_kept | skel_err_flagged | reduction |
|---|---|---|---|---|---|---|---|
| pc | CSCD native flag | 80 | 0.200 | 7.862 | 5.594 | 16.938 | 0.289 |
| pc | Bootstrap own quantile | 80 | 0.000 | 7.862 | 7.862 | nan | 0.000 |
| pc | Uncal. threshold | 80 | 0.175 | 7.862 | 7.636 | 8.929 | 0.029 |
| ges | CSCD native flag | 0 | nan | nan | nan | nan | nan |
| ges | Bootstrap own quantile | 0 | nan | nan | nan | nan | nan |
| ges | Uncal. threshold | 0 | nan | nan | nan | nan | nan |
| fci | CSCD native flag | 80 | 0.025 | 7.800 | 7.500 | 19.500 | 0.038 |
| fci | Bootstrap own quantile | 80 | 0.000 | 7.800 | 7.800 | nan | 0.000 |
| fci | Uncal. threshold | 80 | 0.325 | 7.800 | 8.389 | 6.577 | -0.075 |
