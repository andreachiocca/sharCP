# Original source-file names

- `calibrate_findley.py` was supplied as `optimize_rc_ecp_welded_joint_FI.py`.
- `calibrate_fatemi_socie.py` was supplied as `optimize_rc_ecp_welded_joint_FS_v1.py`.
- `calibrate_swt.py` was supplied as `optimize_rc_ecp_welded_joint_SWT_with_data_v4.py`.
- `predict_findley.py` was supplied as `prevision_FI_v4.py`.
- `predict_fatemi_socie.py` was supplied as `prevision_FS_v3.py`.
- `predict_swt.py` was supplied as `prevision_SWT_v3.py`.

The numerical algorithms have not been refactored. Only literal percent signs in `argparse` help strings were escaped (`%` to `%%`) so all six scripts support `--help`.
