# hubdose_gating

Pooled over 0-6 hidden hubs (skeleton errors in pairs).

| method | rule | runs | flagged | skel_err_all | skel_err_kept | skel_err_flagged | reduction |
|---|---|---|---|---|---|---|---|
| pc | CSCD native flag | 60 | 0.867 | 23.700 | 10.875 | 25.673 | 0.541 |
| pc | Bootstrap own quantile | 60 | 0.000 | 23.700 | 23.700 | nan | 0.000 |
| pc | Uncal. threshold | 60 | 0.233 | 23.700 | 23.609 | 24.000 | 0.004 |
| ges | CSCD native flag | 0 | nan | nan | nan | nan | nan |
| ges | Bootstrap own quantile | 0 | nan | nan | nan | nan | nan |
| ges | Uncal. threshold | 0 | nan | nan | nan | nan | nan |
| fci | CSCD native flag | 60 | 0.383 | 24.317 | 17.027 | 36.043 | 0.300 |
| fci | Bootstrap own quantile | 60 | 0.000 | 24.317 | 24.317 | nan | 0.000 |
| fci | Uncal. threshold | 60 | 0.183 | 24.317 | 25.347 | 19.727 | -0.042 |
