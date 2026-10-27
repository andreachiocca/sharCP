# sharCP | Critical-plane fatigue assessment of sharp notches

Research code and numerical inputs for applying the **Effective Critical Plane (ECP)** approach to sharp weld toes and roots through a criterion-dependent equivalent notch radius,

\[\varrho = \eta_{\mathcal Q}(\alpha,\mathrm{mode},\mathrm{constraint})\,r_c.\]

The implementation considers **Mode I and Mode III**, with **Fatemi–Socie (FS), Findley (FI), and Smith–Watson–Topper (SWT)** parameters. It includes calibration of the material control radius and fatigue master curves, followed by independent fatigue-life prediction for additional welded-joint geometries.

> **Reproducibility status:** The six supplied scripts, six criterion-dependent equivalent-radius coefficient tables, and reduced FE/experimental inputs are included. The complete FE meshes and the original published calibration settings are not included, so this release supports the calibration and prediction workflows but is not independently verified to reproduce every published numerical result. See [Reproducibility notes](docs/REPRODUCIBILITY.md).

## Contents

```text
scripts/
  calibrate_fatemi_socie.py  calibrate_findley.py  calibrate_swt.py
  predict_fatemi_socie.py    predict_findley.py    predict_swt.py
data/
  calibration/      # published-test series + condensed FE hotspot tensors
  validation/       # independent specimens + condensed FE hotspot tensors
  fe_node_selection.csv
  README.md
docs/REPRODUCIBILITY.md
tools/legacy_fe_postprocessing/
requirements.txt
LICENSE
```

## Setup

Requires Python 3.10 or newer (syntax and command-line checks performed with Python 3.13). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Step 1. Check the six ECP coefficient lookup tables

The six author-supplied `slopes_vs_alpha_*.dat` tables are included in both `data/calibration/` and `data/validation/`, where the original command-line programs expect to find them. Each table contains the dependence of the equivalent-radius factor on the notch half-opening angle. Its filename identifies the CP criterion and loading mode; the Mode I tables provide separate plane-stress and plane-strain columns. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Step 2. Run calibration

Run one criterion at a time from the repository root. Illustrative CLI commands (the radius ranges and material parameters should match the target study):

```bash
python scripts/calibrate_fatemi_socie.py --root data/calibration --outdir results/calibration/fs --planeI strain --rcmin 0.02 --rcmax 0.50 --rcsteps 40 --kFS 0.50
python scripts/calibrate_findley.py --root data/calibration --outdir results/calibration/fi --planeI strain --rcmin 0.02 --rcmax 0.50 --rcsteps 40 --kF 0.30
python scripts/calibrate_swt.py --root data/calibration --outdir results/calibration/swt --plane strain --rc_min 0.02 --rc_max 0.50 --rc_steps 40
```

> **Note:** the calibration scripts as supplied use a global Mode I PS/PE choice. The publication describes geometry-specific assignments. The commands above are **illustrative**, not claimed to exactly replicate the published calibration. See the reproducibility notes for this distinction.

Outputs include an optimum control radius, master-curve fit definitions, diagnostic plots, and per-test results. The script defaults for CP constants and confidence bands must be checked against the paper before drawing numerical conclusions.

## Step 3. Run independent prediction

After successful criterion-specific calibration, the predictor reads `powerlaw_fit_*_all.tex` from its calibration output and applies it to the validation samples. The calibration folder must also provide the required criterion-specific eta tables (or those files must be placed below the validation root). The coefficient tables are already included under `data/validation/`. The prediction commands below assume the calibration outputs are available as indicated. Example:

```bash
python scripts/predict_fatemi_socie.py --root data/validation --calibdir ../../results/calibration/fs --outdir ../../results/prediction/fs --kFS 0.50
python scripts/predict_findley.py --root data/validation --calibdir ../../results/calibration/fi --outdir ../../results/prediction/fi --kF 0.30
python scripts/predict_swt.py --root data/validation --calibdir ../../results/calibration/swt --outdir ../../results/prediction/swt
```

> Plane stress vs plane strain may be encoded by the `Mode_I_pstress`/`Mode_I_pstrain` validation input directory names. These folders should not be renamed casually.

## Reuse and citation

When using these scripts or data, cite the associated research article (complete bibliographic record/DOI to be added after publication) and acknowledge the original experimental datasets as described in that article.

Software is provided under the [MIT License](LICENSE). The software license **does not transfer ownership or reuse rights** for experimental measurements extracted from third-party studies. Verify the original sources and redistribution rights before further reuse.

## Related resources

- [Previous ECP research code](https://github.com/achiocca1/ECP), implementing related post-processing workflows.
- [Research data record for sharp-notch fatigue](https://doi.org/10.5281/zenodo.21875885), currently access-restricted. Confirm data sharing permissions and update this repository's data-availability statement before public release.

## Scope

This is research software, not a certified engineering design tool. The sharp–blunt mapping relies on linear-elastic notch fields and the stated proportional-loading assumptions. No guarantee of accuracy is provided outside the material, geometry and load conditions against which the method was calibrated and evaluated.
