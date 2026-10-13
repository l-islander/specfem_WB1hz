### Forward validation: ASDF output (synthetic.h5) vs ascii reference, 5090, 3A.20050_Z

- ref : `work/forward/result_M6/3A.20050_Z`
- test: `work/asdf_test/unpacked`
- traces: ref 932, test 932, compared 932 (NSTEP 6900, DT 0.0145 s)

| metric | value |
|---|---|
| max rel. amplitude diff (worst trace) | 7.705e-06 |
| rel. amplitude diff p99 / median | 6.645e-06 / 3.454e-06 |
| rel. L2 diff max / median | 6.069e-06 / 3.152e-06 |
| zero-lag cc min / median | 1.0000000 / 1.0000000 |
| max abs best-lag (samples) / traces with lag != 0 | 0 / 0 |
| bitwise identical | no |
| **verdict** (pass<= 0.0001, warn<= 0.01, cc>= 0.99999) | **PASS** |

Worst 3 traces by relative amplitude difference:

| trace | max abs ref | rel amp | rel L2 | cc0 | lag |
|---|---|---|---|---|---|
| 3A.20192.BXZ.semd | 7.726e+04 | 7.705e-06 | 5.933e-06 | 1.0000000 | 0 |
| Z4.16171.BXZ.semd | 8.104e+04 | 7.369e-06 | 5.249e-06 | 1.0000000 | 0 |
| Z4.16181.BXZ.semd | 7.985e+04 | 7.364e-06 | 5.877e-06 | 1.0000000 | 0 |
