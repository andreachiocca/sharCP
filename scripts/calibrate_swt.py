#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_rc_ecp_welded_joint_swt.py

Single pipeline to:
  1) scan all author datasets (each author folder can contain Modo_I and/or Modo_III)
  2) compute SWT (critical plane) at the selected FE hot-spot node
  3) apply ECP scaling for sharp notches using a control radius r_c (rc)
  4) sweep rc and pick the rc minimizing the global fatigue scatter T_ECP
  5) write the same article-oriented outputs used previously (tables, TikZ macros, plot)

SWT model (as requested):
  - For each candidate plane normal n, compute normal strain component eps_n = n^T E n
    using tensors at the two load steps (step0=max, step1=R*max).
  - Normal strain amplitude: eps_a = 0.5*|eps_n(step0) - eps_n(step1)|
  - Max tensile normal stress: sig_n,max = max(0, sig_n(step0), sig_n(step1))
  - SWT(n) = eps_a * sig_n,max
  - Plane selection criterion: maximize eps_a (equivalently |eps_n(unit)|),
    with tie-break choosing the largest SWT among planes with the same maximum eps_a.

Folder conventions (root directory):
  - slopes_vs_alpha_swt_modeI.dat     (3 cols: alpha_deg eta_planeStress eta_planeStrain)
  - slopes_vs_alpha_swt_modeIII.dat   (2 cols: alpha_deg eta)

  - Author folders named like:
      AUTHOR__EQ_RADIUS(mm)__SY(MPa)
    Example:  Bertini__0_3__355

  - Inside each author folder:
      Modo_I/   and/or   Modo_III/
    Each contains a Data/ folder with:
      - RESULTS_TOE_<alpha>.csv   and/or RESULTS_ROOT_<alpha>.csv
        where <alpha> is degrees with '_' as decimal separator (e.g. 67_5)
      - Experimental files like:  BendingAW_R_-1.txt  or  TorsionSRE_R_0.txt
        Each row contains at least: Nf  (optional nominal)  Mmax(Nmm)

Outputs (in --outdir):
  - powerlaw_fit_swt_all.tex          : TikZ macros (safe names, no '_' and no digits)
  - test_matrix_detailed.csv          : per-point metadata
  - test_matrix_summary.csv           : compact matrix (one row per Mode+Author)
  - test_matrix_summary.tex           : LaTeX table ready for \input{...}
  - fit_plot_loglog.png               : log-log plot of points + fitted curves (best rc)
  - tecp_vs_rc_swt.dat                : rc sweep results for plotting T_ECP(rc)

Usage (terminal):
  python optimize_rc_ecp_welded_joint_swt.py --root . --outdir export_txt --plane stress \
         --nsigma 1.2815 --Nref 2e6 --rc_min 0.05 --rc_max 0.50 --rc_steps 60

Usage (Jupyter):
  main(["--root",".","--outdir","export_txt"])
"""

from __future__ import print_function

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# PARAMETERS (defaults)
# ============================================================
DEFAULT_MODEI_SLOPES = "slopes_vs_alpha_swt_modeI.dat"
DEFAULT_MODEIII_SLOPES = "slopes_vs_alpha_swt_modeIII.dat"

# Golden-spiral sampling for critical plane search
DEFAULT_N_SPHERE = 6000

# Tolerance (relative) used to identify ties on the max strain-amplitude plane
# (set non-tiny on purpose to avoid missing tensile planes in pure torsion due to sphere discretization)
DEFAULT_TIE_TOL = 1e-3

# Unit load used in FE results (unit run was 1000 Nmm in your workflow)
UNIT_LOAD_NMM = 1000.0

# Tag used in per-series output filenames
PARAM_TAG = "SWT"


# ============================================================
# Small helpers
# ============================================================

def _try_float(tok):
    try:
        return float(tok)
    except Exception:
        return None


def _parse_float_token(s):
    """Parse tokens like '0_3' -> 0.3, '67_5'->67.5."""
    s = str(s).strip()
    s = s.replace(",", ".")
    s = s.replace("_", ".")
    return float(s)


def normal_cdf(z):
    # Phi(z) = 0.5*(1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ============================================================
# Load slopes tables (robust against header row)
# ============================================================

def _load_txt_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    # detect header
    first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip().split()
    skip = 0
    if first and _try_float(first[0]) is None:
        skip = 1

    arr = np.genfromtxt(str(path), dtype=float, comments="#", skip_header=skip)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # drop nan rows
    if arr.size == 0:
        raise ValueError("Empty table: {}".format(path))
    m = np.isfinite(arr[:, 0])
    arr = arr[m]
    if arr.size == 0:
        raise ValueError("No numeric rows in: {}".format(path))
    return arr


def build_eta_modeI(path_modeI):
    """Return dict {"stress": (alpha_deg, eta), "strain": (alpha_deg, eta)}."""
    arr = _load_txt_table(path_modeI)
    if arr.shape[1] < 3:
        raise ValueError("{} must have 3 columns: alpha_deg eta_stress eta_strain".format(path_modeI))
    alpha = arr[:, 0].astype(float)
    eta_stress = arr[:, 1].astype(float)
    eta_strain = arr[:, 2].astype(float)
    # ensure sorted by alpha
    idx = np.argsort(alpha)
    return {
        "stress": (alpha[idx], eta_stress[idx]),
        "strain": (alpha[idx], eta_strain[idx]),
    }


def build_eta_modeIII(path_modeIII):
    """Return tuple (alpha_deg, eta)."""
    arr = _load_txt_table(path_modeIII)
    if arr.shape[1] < 2:
        raise ValueError("{} must have 2 columns: alpha_deg eta".format(path_modeIII))
    alpha = arr[:, 0].astype(float)
    eta = arr[:, 1].astype(float)
    idx = np.argsort(alpha)
    return alpha[idx], eta[idx]


def interp_eta(alpha_deg, eta_table):
    """Linear interpolation with clamping at ends."""
    x, y = eta_table
    return float(np.interp(float(alpha_deg), x, y))


# ============================================================
# Lambda (eigen exponent) as function of alpha
# ============================================================

def det_lambda_modeI(lam, alpha_rad):
    """Monomaterial Mode I (Williams) characteristic equation."""
    return math.sin(2.0 * (math.pi - alpha_rad) * lam) + lam * math.sin(2.0 * (math.pi - alpha_rad))


def first_mode_lambda_modeI(alpha_rad):
    """Find the first positive root in (0, 1]."""
    # Robust bracketing search
    a = 1e-6
    b = 1.0

    fa = det_lambda_modeI(a, alpha_rad)
    fb = det_lambda_modeI(b, alpha_rad)

    if fa * fb > 0:
        # expand upper bound
        b = 5.0
        fb = det_lambda_modeI(b, alpha_rad)

    if fa * fb > 0:
        # last resort: scan
        grid = np.linspace(1e-6, 10.0, 5000)
        vals = np.array([det_lambda_modeI(t, alpha_rad) for t in grid], float)
        s = np.sign(vals)
        idx = np.where(np.diff(s) != 0)[0]
        if idx.size == 0:
            raise ValueError("Cannot bracket lambda for Mode I with alpha={} rad".format(alpha_rad))
        a = float(grid[int(idx[0])])
        b = float(grid[int(idx[0]) + 1])

    try:
        from scipy.optimize import brentq
    except Exception:
        brentq = None

    if brentq is None:
        # fallback: simple bisection
        for _ in range(200):
            m = 0.5 * (a + b)
            fm = det_lambda_modeI(m, alpha_rad)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    return float(brentq(lambda t: det_lambda_modeI(t, alpha_rad), a, b, maxiter=400))


def lambda_modeIII(alpha_rad):
    """Monomaterial Mode III first eigen exponent: lam = pi/(2*(pi-alpha))."""
    return float(math.pi / (2.0 * (math.pi - alpha_rad)))


# ============================================================
# Tensor helpers & FE results reader
# ============================================================

def build_eps_tensor_from_EPEL(exx, eyy, ezz, exy, eyz, exz):
    """EPEL provides engineering shear strains; convert to tensor components."""
    return np.array([
        [exx, 0.5 * exy, 0.5 * exz],
        [0.5 * exy, eyy, 0.5 * eyz],
        [0.5 * exz, 0.5 * eyz, ezz],
    ], float)


def build_sig_tensor(sxx, syy, szz, sxy, syz, sxz):
    return np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz],
    ], float)


def von_mises(sig):
    s = np.array(sig, float)
    m = np.trace(s) / 3.0
    dev = s - np.eye(3) * m
    return math.sqrt(1.5 * float(np.sum(dev * dev)))


def read_results_csv(path_csv):
    """Return list of (node, eps_tensor, sig_tensor) from RESULTS_*.csv."""
    rows = []
    with Path(path_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip headers/non-numeric
            try:
                _ = float(row[0])
                _ = float(row[1])
            except Exception:
                continue
            if len(row) < 20:
                continue

            node = int(float(row[0]))

            # strains
            exx, eyy, ezz, exy, eyz, exz = map(float, row[2:8])
            eps = build_eps_tensor_from_EPEL(exx, eyy, ezz, exy, eyz, exz)

            # stresses (skip plastic columns row[8:14])
            sxx, syy, szz, sxy, syz, sxz = map(float, row[14:20])
            sig = build_sig_tensor(sxx, syy, szz, sxy, syz, sxz)

            rows.append((node, eps, sig))

    if not rows:
        raise RuntimeError("No numeric rows parsed in {}".format(path_csv))
    return rows


def pick_max_vm_node(rows):
    vms = [von_mises(sig) for (_, _, sig) in rows]
    idx = int(np.argmax(vms))
    return rows[idx]  # (node, eps, sig)


# ============================================================
# Critical plane precompute for SWT
# ============================================================

def sphere_normals(n_sphere):
    phi = (1.0 + 5 ** 0.5) / 2.0
    out = np.zeros((int(n_sphere), 3), float)
    n_sphere = int(n_sphere)
    for k in range(n_sphere):
        z = 1.0 - 2.0 * (k + 0.5) / n_sphere
        r = math.sqrt(max(1.0 - z * z, 0.0))
        th = 2.0 * math.pi * k / phi
        out[k, 0] = r * math.cos(th)
        out[k, 1] = r * math.sin(th)
        out[k, 2] = z
    return out


def precompute_eps_sig_along_normals(E_unit, S_unit, normals):
    """Compute eps_n and sig_n along each normal n: n^T A n."""
    E_unit = np.asarray(E_unit, float)
    S_unit = np.asarray(S_unit, float)
    eps_n = np.einsum("ni,ij,nj->n", normals, E_unit, normals)
    sig_n = np.einsum("ni,ij,nj->n", normals, S_unit, normals)
    return eps_n.astype(float), sig_n.astype(float)


def best_swt_unit_for_R(eps_n, sig_n, R, tie_tol=DEFAULT_TIE_TOL):
    """SWT(unit) at plane selected by max normal strain amplitude (tie-break: max SWT)."""
    R = float(R)
    eps_abs = np.abs(eps_n)
    mmax = float(np.max(eps_abs))
    if not (mmax > 0.0):
        return 0.0

    thr = mmax * (1.0 - float(tie_tol))
    cand = eps_abs >= thr
    if not np.any(cand):
        cand = eps_abs == mmax

    eps_a_unit = 0.5 * abs(1.0 - R) * eps_abs[cand]

    sig0 = sig_n[cand]
    sig1 = R * sig0
    sigmax = np.maximum.reduce([sig0, sig1, np.zeros_like(sig0)])

    swt = eps_a_unit * sigmax
    return float(np.max(swt)) if swt.size else 0.0


# ============================================================
# Experimental file parsing
# ============================================================

def parse_author_folder(folder_name):
    """Parse AUTHOR__RADIUS(mm)__SY(MPa)."""
    base = folder_name
    rho_mm = None
    sy_mpa = None

    toks = folder_name.split("__")
    if len(toks) >= 3:
        base = toks[0]
        try:
            rho_mm = _parse_float_token(toks[1])
        except Exception:
            rho_mm = None
        try:
            sy_mpa = _parse_float_token(toks[2])
        except Exception:
            sy_mpa = None
    else:
        # fallback older style: Author_0_3
        m = re.match(r"^(.*)_(\d+_\d+)$", folder_name)
        if m:
            base = m.group(1)
            try:
                rho_mm = _parse_float_token(m.group(2))
            except Exception:
                rho_mm = None

    return base, rho_mm, sy_mpa


def parse_results_alpha(results_csv_name):
    """Parse alpha from RESULTS_TOE_67_5.csv -> 67.5."""
    stem = Path(results_csv_name).stem
    m = re.match(r"^RESULTS_(TOE|ROOT)_(.+)$", stem, re.IGNORECASE)
    if not m:
        m2 = re.match(r"^RESULTS_(TOE|ROOT)$", stem, re.IGNORECASE)
        if m2:
            return m2.group(1).upper(), 0.0
        raise ValueError("Cannot parse RESULTS alpha from: {}".format(results_csv_name))

    loc = m.group(1).upper()
    alpha_token = m.group(2)
    alpha_deg = _parse_float_token(alpha_token)
    return loc, float(alpha_deg)


def parse_specimen_filename(fname):
    """Parse loading, condition, R from names like BendingAW_R_-1.txt."""
    stem = Path(fname).stem
    if "_R_" not in stem:
        raise ValueError("Missing _R_ in specimen file: {}".format(fname))

    pre, rpart = stem.split("_R_", 1)
    R = _parse_float_token(rpart)

    m = re.match(r"^(.*?)(AW|SRE)$", pre, re.IGNORECASE)
    if m:
        loading = m.group(1)
        cond = m.group(2).upper()
    else:
        loading = pre
        cond = "NA"

    return loading, cond, float(R)


def read_specimen_rows(path_txt):
    """Return list of (Nf, Mmax_Nmm) for a specimen file."""
    out = []
    with Path(path_txt).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                Nf = float(parts[0])
            except Exception:
                continue
            try:
                Mmax = float(parts[2]) if len(parts) >= 3 else float(parts[-1])
            except Exception:
                continue
            if Nf > 0 and Mmax != 0:
                out.append((float(Nf), float(Mmax)))
    return out


# ============================================================
# Power-law fit and scatter
# SWT = A * N^b, lognormal vertical scatter
# ============================================================

def fit_powerlaw_with_scatter(N, Q, n_sigma):
    lnx = np.log(N)
    lny = np.log(Q)
    b, lnA = np.polyfit(lnx, lny, deg=1)
    lny_fit = lnA + b * lnx
    resid = lny - lny_fit
    std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    A = float(np.exp(lnA))
    A_plus = float(np.exp(lnA + n_sigma * std))
    A_minus = float(np.exp(lnA - n_sigma * std))
    T = float(np.exp(2.0 * n_sigma * std)) if std > 0 else 1.0

    return {
        "A": A,
        "b": float(b),
        "lnA": float(lnA),
        "std_ln": std,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "T_ecp": T,
        "n_sigma": float(n_sigma),
    }


def q_at_N(A, b, N):
    return float(A * (N ** b))


def cycles_from_ecp(Q, A, b):
    """Inverse power-law: Q = A*N^b -> N = (Q/A)^(1/b)."""
    if A <= 0 or b == 0 or Q <= 0:
        return float("nan")
    try:
        return float((Q / A) ** (1.0 / b))
    except Exception:
        return float("nan")


# ============================================================
# LaTeX table writer (centered, tabular*)
# ============================================================

def df_to_latex_table(rows, outpath, caption, label, columns):
    outpath = Path(outpath)

    def esc(s):
        return str(s).replace("_", r"\_")

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + "c" * len(columns) + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join([esc(c) for c in columns]) + r" \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(" & ".join([esc(r.get(c, "")) for c in columns]) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular*}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Plot
# ============================================================

def plot_loglog_fit(N_all, Q_all, fit, out_png, Nref, ylabel):
    if plt is None:
        return

    out_png = Path(out_png)

    A, b = fit["A"], fit["b"]
    Aplus, Aminus = fit["A_plus"], fit["A_minus"]

    xmin = max(float(np.min(N_all)) * 0.7, 1.0)
    xmax = float(np.max(N_all)) * 1.3
    N_line = np.logspace(np.log10(xmin), np.log10(xmax), 300)

    Q_med = A * (N_line ** b)
    Q_up = Aplus * (N_line ** b)
    Q_lo = Aminus * (N_line ** b)

    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(N_all, Q_all, "o", markersize=4, alpha=0.8, label="Experimental points")
    plt.loglog(N_line, Q_med, "-", linewidth=2, label="P50% fit")
    plt.loglog(N_line, Q_up, "--", linewidth=1.8, label="+{:.3g}$\\sigma$".format(fit["n_sigma"]))
    plt.loglog(N_line, Q_lo, "--", linewidth=1.8, label="-{:.3g}$\\sigma$".format(fit["n_sigma"]))

    Q50_ref = q_at_N(A, b, Nref)
    plt.loglog([Nref], [Q50_ref], "s", markersize=6, label="P50 at N={:.3g}".format(Nref))

    plt.xlabel("Cycles to failure $N_f$")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(str(out_png), dpi=300)
    plt.show()


# ============================================================
# Scan datasets and build per-point constants for fast rc sweep
# ============================================================

def scan_mode_folder(mode_dir, mode_tag, eta_modeI, eta_modeIII, plane_choice,
                     normals, n_sphere, tie_tol):
    """Scan a Modo_I or Modo_III folder and return list of point dicts."""
    data_dir = Path(mode_dir) / "Data"
    if not data_dir.exists():
        return []

    results_csvs = sorted(data_dir.glob("RESULTS_*.csv"))
    if not results_csvs:
        return []

    spec_files = sorted([p for p in data_dir.glob("*.txt") if "RESULTS" not in p.name.upper()])
    if not spec_files:
        return []

    spec_meta = []
    for sf in spec_files:
        try:
            loading, cond, R = parse_specimen_filename(sf.name)
        except Exception:
            continue
        rows = read_specimen_rows(sf)
        if rows:
            spec_meta.append((sf, loading, cond, R, rows))

    if not spec_meta:
        return []

    loc_unit = {}
    for rcsv in results_csvs:
        try:
            loc, alpha_deg = parse_results_alpha(rcsv.name)
        except Exception:
            continue

        rows = read_results_csv(rcsv)
        node, E_unit, S_unit = pick_max_vm_node(rows)

        loc_unit[loc] = {
            "alpha_deg": float(alpha_deg),
            "node": int(node),
            "E_unit": E_unit,
            "S_unit": S_unit,
        }

    if not loc_unit:
        return []

    loc_arrays = {}
    for loc, d in loc_unit.items():
        eps_n, sig_n = precompute_eps_sig_along_normals(d["E_unit"], d["S_unit"], normals)
        loc_arrays[loc] = (eps_n, sig_n)

    swt_unit_cache = {}
    unique_R = sorted(set([m[3] for m in spec_meta]))
    for loc, (eps_n, sig_n) in loc_arrays.items():
        for R in unique_R:
            swt_u = best_swt_unit_for_R(eps_n, sig_n, R, tie_tol=tie_tol)
            swt_unit_cache[(loc, float(R))] = float(swt_u)

    lam_cache = {}

    points = []
    for sf, loading, cond, R, rows in spec_meta:
        R = float(R)
        for Nf, Mmax in rows:
            load_scale = float(Mmax) / UNIT_LOAD_NMM
            if load_scale == 0:
                continue

            for loc, d in loc_unit.items():
                alpha_deg = d["alpha_deg"]
                alpha_rad = math.radians(alpha_deg)

                if mode_tag == "MI":
                    eta = interp_eta(alpha_deg, eta_modeI[plane_choice])
                    if alpha_deg not in lam_cache:
                        lam_cache[alpha_deg] = first_mode_lambda_modeI(alpha_rad)
                    lam = lam_cache[alpha_deg]
                else:
                    eta = interp_eta(alpha_deg, eta_modeIII)
                    if alpha_deg not in lam_cache:
                        lam_cache[alpha_deg] = lambda_modeIII(alpha_rad)
                    lam = lam_cache[alpha_deg]

                pexp = 2.0 * (lam - 1.0)

                swt_unit = swt_unit_cache.get((loc, float(R)), 0.0)
                if not (swt_unit > 0.0):
                    continue

                constC = (load_scale ** 2) * swt_unit

                points.append({
                    "Mode": mode_tag,
                    "Loading": loading,
                    "Condition": cond,
                    "R": R,
                    "Location": "Toe" if loc == "TOE" else "Root" if loc == "ROOT" else loc,
                    "alpha_deg": float(alpha_deg),
                    "eta": float(eta),
                    "pexp": float(pexp),
                    "constC": float(constC),
                    "Nf": float(Nf),
                    "spec_file": str(sf),
                })

    return points



# ============================================================
# Output point files in a PGF-friendly tree (like FI script)
# ============================================================
def make_out_point_file(out_data_root, mode, author, loading, cond, R, loc):
    mode = str(mode).upper()
    folder_mode = "MI" if mode in ("MI", "I") else "MIII"
    results_dir = Path(out_data_root) / folder_mode / str(author) / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    Rtxt = ("{:g}".format(float(R))).replace("+", "")
    locu = str(loc).upper()
    fname = "{}_{}_R_{}_{}_{}.txt".format(str(loading), str(cond), Rtxt, PARAM_TAG, locu)
    return results_dir / fname


# ============================================================
# Main
# ============================================================

def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", type=str, default=".", help="Root directory containing author folders")
    ap.add_argument("--outdir", type=str, default="export_txt", help="Output directory")

    ap.add_argument("--modeI_slopes", type=str, default=DEFAULT_MODEI_SLOPES, help="Mode I slopes file")
    ap.add_argument("--modeIII_slopes", type=str, default=DEFAULT_MODEIII_SLOPES, help="Mode III slopes file")
    ap.add_argument("--plane", type=str, default="stress", choices=["stress", "strain"],
                    help="Which Mode-I column to use (plane stress / plane strain)")

    ap.add_argument("--nsigma", type=float, default=1.2815, help="std dev multiplier for scatter bands (1.2815 for 90%% and 1.6449 for 95%%)")
    ap.add_argument("--Nref", type=float, default=2.0e6, help="Reference cycles for reporting")

    ap.add_argument("--rc_min", type=float, default=0.005, help="Min control radius rc [mm]")
    ap.add_argument("--rc_max", type=float, default=2, help="Max control radius rc [mm]")
    ap.add_argument("--rc_steps", type=int, default=100, help="Number of rc steps")

    ap.add_argument("--n_sphere", type=int, default=DEFAULT_N_SPHERE, help="Number of sphere samples")
    ap.add_argument("--tie_tol", type=float, default=DEFAULT_TIE_TOL, help="Tie tolerance for plane selection")

    args, _unknown = ap.parse_known_args(argv)

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    modeI_path = root / args.modeI_slopes
    modeIII_path = root / args.modeIII_slopes
    if not modeI_path.exists():
        raise SystemExit("Missing Mode I slopes file: {}".format(modeI_path))
    if not modeIII_path.exists():
        raise SystemExit("Missing Mode III slopes file: {}".format(modeIII_path))

    eta_modeI = build_eta_modeI(modeI_path)
    eta_modeIII = build_eta_modeIII(modeIII_path)

    normals = sphere_normals(args.n_sphere)

    all_points = []
    detailed_rows = []

    author_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in (outdir.name,)]
    for adir in sorted(author_dirs, key=lambda p: p.name.lower()):
        author_base, rho_fem_mm, sy_mpa = parse_author_folder(adir.name)
        if rho_fem_mm is None:
            continue
        rho_fem_m = float(rho_fem_mm) * 1e-3

        for mode_tag, subname in (("MI", "Modo_I"), ("MIII", "Modo_III")):
            mode_dir = adir / subname
            if not mode_dir.exists():
                continue

            pts = scan_mode_folder(
                mode_dir=mode_dir,
                mode_tag=mode_tag,
                eta_modeI=eta_modeI,
                eta_modeIII=eta_modeIII,
                plane_choice=args.plane,
                normals=normals,
                n_sphere=args.n_sphere,
                tie_tol=args.tie_tol,
            )

            for p in pts:
                p["Author"] = author_base
                p["rho_fem_m"] = rho_fem_m
                p["Sy_MPa"] = sy_mpa if sy_mpa is not None else "NA"
                all_points.append(p)

                detailed_rows.append({
                    "Mode": p["Mode"],
                    "Author": author_base,
                    "Loading": p["Loading"],
                    "Condition": p["Condition"],
                    "Load ratio": p["R"],
                    "Failure location": p["Location"],
                    "alpha_deg": p["alpha_deg"],
                    "rho_fem_mm": rho_fem_mm,
                    "Sy_MPa": sy_mpa if sy_mpa is not None else "NA",
                    "Specimen file": p["spec_file"],
                    "Nf": p["Nf"],
                })

    if not all_points:
        raise SystemExit("No valid points found. Check folder structure and input files.")

    Nf_arr = np.array([p["Nf"] for p in all_points], float)
    eta_arr = np.array([p["eta"] for p in all_points], float)
    pexp_arr = np.array([p["pexp"] for p in all_points], float)
    C_arr = np.array([p["constC"] for p in all_points], float)
    rho_fem_arr = np.array([p["rho_fem_m"] for p in all_points], float)

    rc_mm = np.linspace(float(args.rc_min), float(args.rc_max), int(args.rc_steps))
    rc_m = rc_mm * 1e-3

    sweep = []
    best_T = float("inf")
    best_fit = None
    best_rc = None
    best_i = None

    for i, rc in enumerate(rc_m):
        rho_eq = eta_arr * rc
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = C_arr * np.power(rho_eq / rho_fem_arr, pexp_arr)

        m = np.isfinite(Q) & (Q > 0) & np.isfinite(Nf_arr) & (Nf_arr > 0)
        if np.sum(m) < 3:
            continue

        fit = fit_powerlaw_with_scatter(Nf_arr[m], Q[m], args.nsigma)
        T = float(fit["T_ecp"])

        sweep.append({
            "rc_mm": float(rc_mm[i]),
            "T_ECP": T,
            "A": float(fit["A"]),
            "b": float(fit["b"]),
            "std_ln": float(fit["std_ln"]),
            "nData": int(np.sum(m)),
        })

        if T < best_T:
            best_T = T
            best_i = i
            best_fit = fit
            best_rc = float(rc)

    if best_fit is None:
        raise SystemExit("rc sweep failed: no rc produced enough valid points for a fit.")

    rho_eq_best = eta_arr * best_rc
    Q_best = C_arr * np.power(rho_eq_best / rho_fem_arr, pexp_arr)
    m_best = np.isfinite(Q_best) & (Q_best > 0) & (Nf_arr > 0)

    sweep_path = outdir / "tecp_vs_rc_swt.dat"
    with sweep_path.open("w", encoding="utf-8") as f:
        f.write("rc_mm\tT_ECP\tA\tb\tstd_ln\tnData\n")
        for r in sweep:
            f.write("{:.6g}\t{:.6g}\t{:.10e}\t{:.10e}\t{:.6g}\t{}\n".format(
                r["rc_mm"], r["T_ECP"], r["A"], r["b"], r["std_ln"], r["nData"]
            ))

    FS50_Nref = q_at_N(best_fit["A"], best_fit["b"], args.Nref)
    FSU_Nref = q_at_N(best_fit["A_plus"], best_fit["b"], args.Nref)
    FSL_Nref = q_at_N(best_fit["A_minus"], best_fit["b"], args.Nref)

    P_high = 100.0 * normal_cdf(args.nsigma)
    P_low = 100.0 * (1.0 - normal_cdf(args.nsigma))

    def fmt_e(x):
        s = "{:.10e}".format(float(x))
        return s.replace("+", "")

    def fmt_g(x):
        return "{:.6g}".format(float(x))

    def fmt_num(x):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return "{:.10g}".format(float(x))

    def fmt_pct(x):
        return "{:.1f}".format(float(x))

    tex = []
    tex.append("% Auto-generated power-law fit (GLOBAL) for SWT - safe macro names (no underscores, no digits)")
    tex.append("% Best rc = {:.6g} mm".format(best_rc * 1e3))
    tex.append("\\def\\FSFitA{{{}}}".format(fmt_e(best_fit["A"])))
    tex.append("\\def\\FSFitb{{{}}}".format(fmt_e(best_fit["b"])))
    tex.append("\\def\\FSFitStdLn{{{}}}".format(fmt_e(best_fit["std_ln"])))

    tex.append("\\def\\FSFitNsigma{{{}}}".format(fmt_num(args.nsigma)))
    tex.append("\\def\\FSFitAplus{{{}}}".format(fmt_e(best_fit["A_plus"])))
    tex.append("\\def\\FSFitAminus{{{}}}".format(fmt_e(best_fit["A_minus"])))
    tex.append("\\def\\FSFitTECP{{{}}}".format(fmt_g(best_fit["T_ecp"])))

    tex.append("\\def\\FSFitNref{{{}}}".format(fmt_num(args.Nref)))
    tex.append("\\def\\FSFitFSupNref{{{}}}".format(fmt_e(FSU_Nref)))
    tex.append("\\def\\FSFitFSmedNref{{{}}}".format(fmt_e(FS50_Nref)))
    tex.append("\\def\\FSFitFSloNref{{{}}}".format(fmt_e(FSL_Nref)))

    tex.append("\\def\\FSFitPsHigh{{{}}}".format(fmt_pct(P_high)))
    tex.append("\\def\\FSFitPsLow{{{}}}".format(fmt_pct(P_low)))

    tex_out = outdir / "powerlaw_fit_swt_all.tex"
    tex_out.write_text("\n".join(tex), encoding="utf-8")

    detailed_csv = outdir / "test_matrix_detailed.csv"
    if detailed_rows:
        cols = list(detailed_rows[0].keys())
        with detailed_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in detailed_rows:
                w.writerow(r)

    by = {}
    for r in detailed_rows:
        key = (r["Mode"], r["Author"])
        d = by.get(key)
        if d is None:
            d = {
                "Mode": r["Mode"],
                "Author": r["Author"],
                "Loading": set(),
                "Condition": set(),
                "Load ratio": set(),
                "Failure location": set(),
                "nData": 0,
            }
            by[key] = d
        if r["Loading"] and r["Loading"] != "NA":
            d["Loading"].add(str(r["Loading"]))
        if r["Condition"] and r["Condition"] != "NA":
            d["Condition"].add(str(r["Condition"]))
        d["Load ratio"].add(str(r["Load ratio"]))
        if r["Failure location"] and r["Failure location"] != "NA":
            d["Failure location"].add(str(r["Failure location"]))
        d["nData"] += 1

    def _format_R_set(Rset):
        vals = []
        for s in Rset:
            try:
                vals.append(float(str(s).replace(",", ".")))
            except Exception:
                pass
        if vals:
            vals = sorted(set(vals))
            return "-".join([("{:g}".format(v)).replace(".", ",") for v in vals])
        return "NA"

    summary_rows = []
    for (mode, author) in sorted(by.keys()):
        d = by[(mode, author)]
        summary_rows.append({
            "Mode": d["Mode"],
            "Author": d["Author"],
            "Loading": ", ".join(sorted(d["Loading"])) if d["Loading"] else "NA",
            "Condition": ", ".join(sorted(d["Condition"])) if d["Condition"] else "NA",
            "Load ratio": _format_R_set(d["Load ratio"]),
            "Failure location": ", ".join(sorted(d["Failure location"])) if d["Failure location"] else "NA",
            "nData": str(d["nData"]),
        })

    summary_csv = outdir / "test_matrix_summary.csv"
    if summary_rows:
        cols = ["Mode", "Author", "Loading", "Condition", "Load ratio", "Failure location", "nData"]
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

        caption = (
            "Test matrix of the experimental dataset. "
            "Condition codes: AW = as-welded; SRE = stress-relieved. "
            "Failure location is reported as Toe or Root. "
            "When multiple load ratios are available for the same (Mode, Author), they are reported as hyphen-separated values."
        )
        df_to_latex_table(
            rows=summary_rows,
            outpath=outdir / "test_matrix_summary.tex",
            caption=caption,
            label="tab:test_matrix",
            columns=cols,
        )


    # ---- Write per-series files to outdir/Data/... (like Findley script)
    out_data_root = outdir / "Data"
    out_data_root.mkdir(parents=True, exist_ok=True)

    buckets = {}
    for pr in all_points:
        rho_eq = float(pr["eta"]) * float(best_rc)   # [m]
        if rho_eq <= 0.0 or float(pr["rho_fem_m"]) <= 0.0:
            continue
        Q = float(pr["constC"]) * float((rho_eq / float(pr["rho_fem_m"])) ** float(pr["pexp"]))
        if not (Q > 0.0 and np.isfinite(Q)):
            continue

        key = (pr["Mode"], pr["Author"], pr["Loading"], pr["Condition"], pr["R"], pr["Location"])
        buckets.setdefault(key, [])
        buckets[key].append((float(pr["Nf"]), Q))

    point_file_rows = []
    for key, pts in buckets.items():
        mode, author, loading, cond, Rv, loc = key
        out_file = make_out_point_file(out_data_root, mode, author, loading, cond, Rv, loc)

        pts = sorted(pts, key=lambda t: t[0])
        arr = []
        for (Nf, Q) in pts:
            Npred = cycles_from_ecp(Q, best_fit["A"], best_fit["b"])
            arr.append((Nf, Q, Npred))

        np.savetxt(
            str(out_file),
            np.asarray(arr, float),
            fmt="%.10e",
            header="Nf\t{}\tNpred".format(PARAM_TAG),
            comments=""
        )

        point_file_rows.append({
            "Mode": str(mode),
            "Author": str(author),
            "Loading": str(loading),
            "Condition": str(cond),
            "Load ratio": float(Rv),
            "Failure location": str(loc),
            "nData": int(len(arr)),
            "File": str(out_file),
        })

    # Index CSV listing the generated point files
    if point_file_rows:
        cols_pf = ["Mode","Author","Loading","Condition","Load ratio","Failure location","nData","File"]
        with (outdir / "point_files_index.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols_pf)
            w.writeheader()
            for r in point_file_rows:
                w.writerow(r)

    plot_loglog_fit(
        N_all=Nf_arr[m_best],
        Q_all=Q_best[m_best],
        fit=best_fit,
        out_png=outdir / "fit_plot_loglog.png",
        Nref=args.Nref,
        ylabel="SWT (critical plane)",
    )

    print("\n=== SWT | GLOBAL OPTIMIZATION ===")
    print("Best rc [mm] = {:.6g}".format(best_rc * 1e3))
    print("nData used   = {}".format(int(np.sum(m_best))))
    print("A            = {:.6e}".format(best_fit["A"]))
    print("b            = {:.6e}".format(best_fit["b"]))
    print("std_ln       = {:.6e}".format(best_fit["std_ln"]))
    print("T_ECP        = {:.6g}".format(best_fit["T_ecp"]))
    print("Outputs written to: {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()