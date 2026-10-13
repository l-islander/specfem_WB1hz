### Forward validation: x86 same-hardware rerun (5090, fresh genmesh + solve)

- ref : `work/forward/result_M6/3A.20050_Z`
- test: `work/gh200_prep/rerun_M6_x86/3A.20050_Z`
- traces: ref 932, test 932, compared 932 (NSTEP 6900, DT 0.0145 s)

| metric | value |
|---|---|
| max rel. amplitude diff (worst trace) | 8.291e-06 |
| rel. amplitude diff p99 / median | 6.891e-06 / 3.496e-06 |
| rel. L2 diff max / median | 5.405e-06 / 3.195e-06 |
| zero-lag cc min / median | 1.0000000 / 1.0000000 |
| max abs best-lag (samples) / traces with lag != 0 | 0 / 0 |
| bitwise identical | no |
| **verdict** (pass<= 0.0001, warn<= 0.01, cc>= 0.99999) | **PASS** |

Worst 10 traces by relative amplitude difference:

| trace | max abs ref | rel amp | rel L2 | cc0 | lag |
|---|---|---|---|---|---|
| Z4.16171.BXZ.semd | 8.104e+04 | 8.291e-06 | 5.405e-06 | 1.0000000 | 0 |
| 3A.20192.BXZ.semd | 7.726e+04 | 8.191e-06 | 5.075e-06 | 1.0000000 | 0 |
| 3A.20182.BXZ.semd | 8.302e+04 | 8.187e-06 | 5.398e-06 | 1.0000000 | 0 |
| Z4.16181.BXZ.semd | 7.985e+04 | 7.779e-06 | 5.023e-06 | 1.0000000 | 0 |
| XD.1059.BXZ.semd | 8.611e+04 | 7.417e-06 | 3.669e-06 | 1.0000000 | 0 |
| Z4.16163.BXZ.semd | 9.264e+04 | 7.273e-06 | 4.585e-06 | 1.0000000 | 0 |
| GS.LEVE.BXZ.semd | 6.139e+04 | 7.063e-06 | 4.667e-06 | 1.0000000 | 0 |
| UW.LEVE.BXZ.semd | 6.139e+04 | 7.063e-06 | 4.667e-06 | 1.0000000 | 0 |
| 3A.20175.BXZ.semd | 9.488e+04 | 7.061e-06 | 4.593e-06 | 1.0000000 | 0 |
| UW.QWAK.BXZ.semd | 6.547e+04 | 6.893e-06 | 4.358e-06 | 1.0000000 | 0 |
