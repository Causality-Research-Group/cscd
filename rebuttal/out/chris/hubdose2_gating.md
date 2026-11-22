# hubdose2_gating

Pooled over 0-6 hidden hubs (skeleton errors in pairs).

| method | rule | runs | flagged | skel_err_all | skel_err_kept | skel_err_flagged | reduction |
|---|---|---|---|---|---|---|---|
| pc | CSCD native flag | 60 | 0.150 | 7.417 | 5.431 | 18.667 | 0.268 |
| pc | Bootstrap own quantile | 60 | 0.000 | 7.417 | 7.417 | nan | 0.000 |
| pc | Uncal. threshold | 60 | 0.183 | 7.417 | 7.041 | 9.091 | 0.051 |
| ges | CSCD native flag | 0 | nan | nan | nan | nan | nan |
| ges | Bootstrap own quantile | 0 | nan | nan | nan | nan | nan |
| ges | Uncal. threshold | 0 | nan | nan | nan | nan | nan |
| fci | CSCD native flag | 60 | 0.033 | 7.417 | 7.000 | 19.500 | 0.056 |
| fci | Bootstrap own quantile | 60 | 0.000 | 7.417 | 7.417 | nan | 0.000 |
| fci | Uncal. threshold | 60 | 0.317 | 7.417 | 8.146 | 5.842 | -0.098 |
