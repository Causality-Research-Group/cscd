# hubdose_gating

Pooled over 0-6 hidden hubs (skeleton errors in pairs).

| method | rule | runs | flagged | skel_err_all | skel_err_kept | skel_err_flagged | reduction |
|---|---|---|---|---|---|---|---|
| pc | CSCD native flag | 100 | 0.870 | 23.750 | 10.615 | 25.713 | 0.553 |
| pc | Bootstrap own quantile | 100 | 0.000 | 23.750 | 23.750 | nan | 0.000 |
| pc | Uncal. threshold | 100 | 0.200 | 23.750 | 23.875 | 23.250 | -0.005 |
| ges | CSCD native flag | 0 | nan | nan | nan | nan | nan |
| ges | Bootstrap own quantile | 0 | nan | nan | nan | nan | nan |
| ges | Uncal. threshold | 0 | nan | nan | nan | nan | nan |
| fci | CSCD native flag | 100 | 0.390 | 24.430 | 16.033 | 37.564 | 0.344 |
| fci | Bootstrap own quantile | 100 | 0.000 | 24.430 | 24.430 | nan | 0.000 |
| fci | Uncal. threshold | 100 | 0.170 | 24.430 | 25.795 | 17.765 | -0.056 |
