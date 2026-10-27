# Software checks performed before release

- All six Python programs passed Python compilation checks.
- All six `--help` CLI options ran after fixing literal percent-sign formatting in help strings.
- Dataset manifest contains 17 calibration and 8 validation FE hotspot rows, accompanied by 24 calibration and 8 validation experimental input files.
- The dataset checker `python tools/check_dataset.py` passed.
- A **non-scientific smoke test** with temporary, synthetic eta tables and coarse critical-plane sampling successfully completed calibration and validation with all three criteria. Each validation pipeline processed 55 observations. The synthetic eta tables, smoke-test results and numerical values are not included and must not be interpreted as reproducing the published results.
- The original author-provided eta tables, corresponding calibration settings, and original full nodal FE data were unavailable for a scientific rerun.

The documented checks demonstrate that the script interfaces and reduced FE input formatting work; they do not validate the material model or published numerical results.
