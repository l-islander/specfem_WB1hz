### Forward validation: binary output vs ascii reference (5090, 3A.20050_Z)

- ref : `work/forward/result_M6/3A.20050_Z`
- test: `work/gh200_prep/bin_M6_x86/3A.20050_Z`
- traces: ref 932, test 932, compared 932 (NSTEP 6900, DT 0.0145 s)

| metric | value |
|---|---|
| max rel. amplitude diff (worst trace) | 6.759e-06 |
| rel. amplitude diff p99 / median | 6.304e-06 / 3.583e-06 |
| rel. L2 diff max / median | 5.628e-06 / 3.215e-06 |
| zero-lag cc min / median | 1.0000000 / 1.0000000 |
| max abs best-lag (samples) / traces with lag != 0 | 0 / 0 |
| bitwise identical | no |
| **verdict** (pass<= 0.0001, warn<= 0.01, cc>= 0.99999) | **PASS** |

Worst 3 traces by relative amplitude difference:

| trace | max abs ref | rel amp | rel L2 | cc0 | lag |
|---|---|---|---|---|---|
| 3A.20127.BXZ.semd | 1.573e+05 | 6.759e-06 | 4.664e-06 | 1.0000000 | 0 |
| UW.WPO.BXZ.semd | 9.518e+04 | 6.705e-06 | 5.077e-06 | 1.0000000 | 0 |
| XD.1017.BXZ.semd | 1.247e+05 | 6.641e-06 | 4.883e-06 | 1.0000000 | 0 |
