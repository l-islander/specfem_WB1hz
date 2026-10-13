### Forward validation: control: M0 vs M6 (Z4.16063_Z)

- ref : `work/forward/result_M6/Z4.16063_Z`
- test: `work/forward/result_M0/Z4.16063_Z/SYN`
- traces: ref 932, test 932, compared 932 (NSTEP 6900, DT 0.0145 s)

| metric | value |
|---|---|
| max rel. amplitude diff (worst trace) | 2.568e+00 |
| rel. amplitude diff p99 / median | 2.191e+00 / 8.574e-01 |
| rel. L2 diff max / median | 2.092e+00 / 8.797e-01 |
| zero-lag cc min / median | -0.6979524 / 0.6066044 |
| max abs best-lag (samples) / traces with lag != 0 | 47 / 898 |
| bitwise identical | no |
| **verdict** (pass<= 0.0001, warn<= 0.01, cc>= 0.99999) | **FAIL** |

Worst 3 traces by relative amplitude difference:

| trace | max abs ref | rel amp | rel L2 | cc0 | lag |
|---|---|---|---|---|---|
| ZO.1135.BXZ.semd | 6.584e+04 | 2.568e+00 | 2.045e+00 | -0.6822580 | 35 |
| ZO.1134.BXZ.semd | 7.881e+04 | 2.454e+00 | 2.020e+00 | -0.6645733 | 36 |
| YM.10524.BXZ.semd | 8.738e+04 | 2.284e+00 | 1.936e+00 | -0.5806401 | 33 |
