# Validation reports

Output of `workflows/grace_hopper/validate_forward.py` (one virtual source, 932 traces each, final
2 s model; the control compares M0 with M6 on source Z4.16063). Each report lists max relative amplitude difference, relative L2 difference, zero-lag
correlation and best lag per trace. Paths inside the reports refer to the original run tree and are
not part of this repository.

| report | comparison | max relative amplitude difference | verdict |
|---|---|---|---|
| `rerun_floor_3A.20050_Z.md` | RTX 5090 rerun vs RTX 5090 reference (fresh database + solve) | 8.3e-6 | PASS (the run-to-run floor) |
| `bin_vs_ref_3A.20050_Z.md` | binary seismogram output vs ascii reference | 6.8e-6 | PASS |
| `asdf_vs_ref_3A.20050_Z.md` | ASDF seismogram output (`synthetic.h5`) vs ascii reference | 7.7e-6 | PASS |
| `control_M0_vs_M6.md` | negative control: two different models | 2.6 | FAIL (as intended) |

Grace Hopper vs x86 results (7.0–7.2e-6 on three sources) and all timings are in
`../docs/GH200_RESULTS.md`.
