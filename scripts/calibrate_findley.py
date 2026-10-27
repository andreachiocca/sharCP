#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_rc_findley_welded_joint.py
----------------------------------
Findley version of optimize_rc_ecp_welded_joint.py.

Key differences vs FS version:
  - reads slopes_vs_alpha_findley_modeI.dat and slopes_vs_alpha_findley_modeIII.dat
  - computes Findley parameter on the critical plane:
        F = tau_a + kF * sigma_n,max
    and selects the plane by maximizing the FULL Findley factor (NOT only tau_a).

Run (terminal):
  python optimize_rc_findley_welded_joint.py --root . --outdir export_txt_findley --planeI strain --kF 0.3 \
      --rcmin 0.02 --rcmax 0.50 --rcsteps 40 --nsigma 1.2815 --Nref 2e6

Run (Jupyter):
  main(["--root",".","--outdir","export_txt_findley","--planeI","strain","--kF","0.3",
        "--rcmin","0.02","--rcmax","0.50","--rcsteps","40"])
"""

from __future__ import print_function
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_SY_MPA = 355.0          # parsed from Author__rho__Sy (kept for consistency; not used by Findley)
DEFAULT_RHOGEO_MM = 1.0         # FE equivalent radius (mm) used in RESULTS_*.csv if not in folder name
DEFAULT_KF = 0.1           # Findley kF (user input)
UNIT_LOAD_NMM = 1000.0          # FE unit load used to generate RESULTS_*.csv
DEFAULT_SPHERE = 6000           # sphere points for plane search
PARAM_TAG = "FINDLEY"           # appears in output filenames
# ============================================================


# ============================================================
# Robust numeric table reader (skips headers)
# ============================================================
def _load_txt_table(path, min_cols=2):
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        ln2 = ln.replace(";", " ")
        toks = ln2.split()
        try:
            vals = [float(t.replace(",", ".")) for t in toks]
        except Exception:
            continue  # header/non-numeric
        if len(vals) >= min_cols:
            rows.append(vals)
    if not rows:
        raise ValueError("No numeric data parsed from: {}".format(str(path)))
    ncol = max(len(r) for r in rows)
    arr = np.full((len(rows), ncol), np.nan, dtype=float)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    return arr


def build_eta_modeI(path_modeI):
    """
    slopes_vs_alpha_findley_modeI.dat:
      alpha_deg   eta_plane_stress   eta_plane_strain
    """
    arr = _load_txt_table(path_modeI, min_cols=3)
    alpha = arr[:, 0].astype(float)
    eta_stress = arr[:, 1].astype(float)
    eta_strain = arr[:, 2].astype(float)
    idx = np.argsort(alpha)
    return {"alpha": alpha[idx], "stress": eta_stress[idx], "strain": eta_strain[idx]}


def build_eta_modeIII(path_modeIII):
    """
    slopes_vs_alpha_findley_modeIII.dat:
      alpha_deg   eta
    """
    arr = _load_txt_table(path_modeIII, min_cols=2)
    alpha = arr[:, 0].astype(float)
    eta = arr[:, 1].astype(float)
    idx = np.argsort(alpha)
    return {"alpha": alpha[idx], "eta": eta[idx]}


def interp_clamped(x, xp, fp):
    x = float(x)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    return float(np.interp(x, xp, fp))


# ============================================================
# Author folder parsing: Author__rho(mm)__Sy(MPa)
# ============================================================
def parse_author_folder(author_folder, default_rho_mm=DEFAULT_RHOGEO_MM, default_Sy_MPa=DEFAULT_SY_MPA):
    """
    New: Author__0_3__355  (rho_geo=0.3 mm, Sy=355 MPa)
    Old: Author_0_3       (rho_geo=0.3 mm, Sy=default)
    Plain: Author         (defaults)
    """
    s = str(author_folder).strip()
    print(s)

    if "__" in s:
        parts = s.split("__")
        author = parts[0].strip()

        rho_mm = default_rho_mm
        Sy = default_Sy_MPa

        if len(parts) >= 2:
            rho_tok = parts[1].strip().replace(",", ".").replace("_", ".")
            try:
                rho_mm = float(rho_tok)
            except Exception:
                rho_mm = default_rho_mm
        print(rho_mm)

        if len(parts) >= 3:
            sy_tok = parts[2].strip().replace(",", ".").replace("_", ".")
            try:
                Sy = float(sy_tok)
            except Exception:
                Sy = default_Sy_MPa

        return author, float(rho_mm), float(Sy)

    toks = s.split("_")
    if len(toks) >= 3:
        maybe_rho = (toks[-2] + "." + toks[-1]).replace(",", ".")
        try:
            rho_mm = float(maybe_rho)
            author = "_".join(toks[:-2])
            return author, float(rho_mm), float(default_Sy_MPa)
        except Exception:
            pass

    return s, float(default_rho_mm), float(default_Sy_MPa)


# ============================================================
# Alpha from RESULTS csv name: RESULTS_TOE_67_5.csv -> 67.5
# ============================================================
def alpha_from_results_csv_name(csv_path):
    name = Path(csv_path).stem  # RESULTS_TOE_67_5
    m = re.search(r"RESULTS_(TOE|ROOT)_(.+)$", name, re.IGNORECASE)
    if not m:
        return 0.0  # backward compatibility: RESULTS_TOE.csv
    a_tok = m.group(2).strip().replace(",", ".").replace("_", ".")
    return float(a_tok)


def location_from_results_csv_name(csv_path):
    name = Path(csv_path).stem.upper()
    if "TOE" in name:
        return "TOE"
    if "ROOT" in name:
        return "ROOT"
    raise ValueError("Cannot infer location from: {}".format(csv_path))


# ============================================================
# Lambda functions (alpha-dependent)
# ============================================================
def det_lambda_modeI(lam, alpha_rad):
    phi = np.pi - alpha_rad
    a = (1.0 + lam) * np.cos((1.0 - lam) * phi)
    b = np.cos((1.0 + lam) * phi)
    c = (1.0 - lam) * np.sin((1.0 - lam) * phi)
    d = np.sin((1.0 + lam) * phi)
    return a * d - b * c


def first_mode_lambda_modeI(alpha_rad, grid=8000):
    xs = np.linspace(0.01, 0.99, grid)
    ys = det_lambda_modeI(xs, alpha_rad)
    s = np.sign(ys)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        raise ValueError("No Mode I eigenvalue in (0.01,0.99) for alpha_rad={}".format(alpha_rad))
    i0 = int(idx[0])
    a, b = xs[i0], xs[i0 + 1]
    return float(brentq(lambda t: det_lambda_modeI(t, alpha_rad), a, b, maxiter=300))


def first_mode_lambda_modeIII(alpha_rad):
    denom = 2.0 * (np.pi - alpha_rad)
    if denom <= 0:
        raise ValueError("Invalid alpha for Mode III: alpha_rad={}".format(alpha_rad))
    return float(np.pi / denom)


_LAM_CACHE = {}  # (mode, alpha_deg_rounded) -> lambda


def get_lambda(mode, alpha_deg):
    key = (mode.upper(), round(float(alpha_deg), 6))
    if key in _LAM_CACHE:
        return _LAM_CACHE[key]
    a_rad = np.deg2rad(float(alpha_deg))
    if mode.upper() in ("MI", "MODEI", "I"):
        lam = first_mode_lambda_modeI(a_rad)
    else:
        lam = first_mode_lambda_modeIII(a_rad)
    _LAM_CACHE[key] = lam
    return lam


# ============================================================
# FE csv reading (stress tensor only needed for Findley)
# ============================================================
def build_sig_tensor(sxx, syy, szz, sxy, syz, sxz):
    return np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ], float)


def von_mises(sig):
    s = np.array(sig, float)
    m = np.trace(s) / 3.0
    dev = s - np.eye(3) * m
    return math.sqrt(1.5 * np.sum(dev * dev))


def read_results_csv(path_csv):
    rows = []
    with Path(path_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                _ = float(row[0]); _ = float(row[1])
            except Exception:
                continue
            if len(row) < 20:
                continue
            node = int(float(row[0]))
            sxx, syy, szz, sxy, syz, sxz = map(float, row[14:20])
            sig = build_sig_tensor(sxx, syy, szz, sxy, syz, sxz)
            rows.append((node, sig))
    if not rows:
        raise RuntimeError("No numeric rows parsed in {}".format(path_csv))
    return rows


def pick_max_vm_node(rows):
    vms = [von_mises(sig) for (_, sig) in rows]
    idx = int(np.argmax(vms))
    return rows[idx]  # (node, sig)


# ============================================================
# Plane rotation and Findley unit arrays
# ============================================================
def _basis_from_n(n):
    n = np.asarray(n, float)
    n /= np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(n, up); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return n, t1, t2


def _rotate_tensor(A, Q):
    return Q.T @ A @ Q


def plane_tau_sig_arrays(S_unit, n_sphere):
    """
    For each plane normal:
      tau_unit  = sqrt(Sxz^2 + Syz^2)   (shear traction magnitude on plane)
      sig_unit  = Szz                   (normal stress on plane)
    in the rotated frame where z is the plane normal.
    """
    phi = (1.0 + 5 ** 0.5) / 2.0
    S_unit = np.asarray(S_unit, float)
    n_sphere = int(n_sphere)

    tau = np.zeros(n_sphere, float)
    sig = np.zeros(n_sphere, float)

    for k in range(n_sphere):
        z = 1.0 - 2.0 * (k + 0.5) / n_sphere
        r = math.sqrt(max(1.0 - z * z, 0.0))
        th = 2.0 * math.pi * k / phi
        n = np.array([r * math.cos(th), r * math.sin(th), z])
        n, t1, t2 = _basis_from_n(n)
        Q = np.stack([t1, t2, n], axis=1)

        Sr = _rotate_tensor(S_unit, Q)
        tau[k] = math.sqrt(float(Sr[0, 2]) ** 2 + float(Sr[1, 2]) ** 2)
        sig[k] = float(Sr[2, 2])

    return tau, sig


def best_findley_combo_for_R(tau_unit, sig_unit, R, kF):
    """
    Precompute the plane-maximised unit combination for a given load ratio R:
      tau_a = 0.5*|tau0 - tau1| = g0 * 0.5*|1-R| * tau_unit
      sigma_n,max = max(sigma0, sigma1, 0) = g0 * max(1,R) * max(sig_unit,0)

    So factor at unit g0 (excluding g0 and excluding geometric scaling) is:
      combo = max_planes( 0.5*|1-R| * tau_unit + kF * max(1,R) * max(sig_unit,0) )
    """
    c_shear = 0.5 * abs(1.0 - float(R))
    c_norm  = float(kF) * max(1.0, float(R))
    sig_pos = np.maximum(np.asarray(sig_unit, float), 0.0)
    val = c_shear * np.asarray(tau_unit, float) + c_norm * sig_pos
    return float(np.max(val))


# ============================================================
# Specimen file parsing
# ============================================================
KNOWN_CONDITIONS = ("AW", "SRE", "SR", "PWHT")


def parse_loading_condition(stem_before_R):
    s = stem_before_R.strip()
    cond = "NA"
    loading = s
    for c in KNOWN_CONDITIONS:
        if s.upper().endswith(c):
            cond = c
            loading = s[:-len(c)]
            loading = loading if loading else s
            return loading, cond
    return loading, cond


def parse_specimen_file(path_txt):
    path_txt = Path(path_txt)
    mR = re.search(r"_R_([-\d\.]+)$", path_txt.stem)
    if not mR:
        raise RuntimeError("Cannot parse load ratio from filename: {}".format(path_txt.name))
    R = float(mR.group(1))

    head = path_txt.stem[:mR.start()]
    loading, cond = parse_loading_condition(head)

    data = []
    with path_txt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                Nf = float(parts[0])
                Mmax = float(parts[2])  # Nmm
            except Exception:
                continue
            if Nf > 0 and Mmax != 0:
                data.append((Nf, Mmax))
    if not data:
        raise RuntimeError("No numeric rows found in {}".format(path_txt))
    return R, loading, cond, data


# ============================================================
# Fit / stats
# ============================================================
def fit_powerlaw_with_scatter(N, Y, n_sigma):
    lnx = np.log(N)
    lny = np.log(Y)
    b, lnA = np.polyfit(lnx, lny, deg=1)
    lny_fit = lnA + b * lnx
    resid = lny - lny_fit
    std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    A = float(np.exp(lnA))
    A_plus = float(np.exp(lnA + n_sigma * std))
    A_minus = float(np.exp(lnA - n_sigma * std))
    T = float(np.exp(2.0 * n_sigma * std)) if std > 0 else 1.0
    return {"A": A, "b": float(b), "std_ln": std, "A_plus": A_plus, "A_minus": A_minus,
            "T_ecp": T, "n_sigma": float(n_sigma)}


def y_at_N(A, b, N):
    return float(A * (N ** b))


def cycles_from_ecp(ecp, A, b):
    if A <= 0 or b == 0 or ecp <= 0:
        return float("nan")
    try:
        return float((ecp / A) ** (1.0 / b))
    except Exception:
        return float("nan")


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ============================================================
# Table helpers (no pandas)
# ============================================================
def _try_float(s):
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return None


def _format_R_values(R_list):
    vals = []
    for r in R_list:
        if r is None:
            continue
        vals.append(str(r).strip())
    vals = sorted(set(vals), key=lambda z: (1e99 if _try_float(z) is None else _try_float(z)))
    formatted = []
    for v in vals:
        fv = _try_float(v)
        if fv is None:
            formatted.append(v)
        else:
            formatted.append(("{:g}".format(fv)).replace(".", ","))
    return "-".join(formatted) if formatted else "NA"


def _map_location(loc):
    u = str(loc).strip().upper()
    if "TOE" in u:
        return "Toe"
    if "ROOT" in u:
        return "Root"
    return "NA"


def write_csv(path, rows, header):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, "") for h in header])


def df_to_latex_table(rows, cols, outpath, caption, label):
    outpath = Path(outpath)

    def esc(s):
        return str(s).replace("_", r"\_")

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + "c" * len(cols) + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join([esc(c) for c in cols]) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(" & ".join([esc(row.get(c, "")) for c in cols]) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular*}")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Plot
# ============================================================
def plot_loglog_fit(N_all, Y_all, fit, out_png, Nref):
    out_png = Path(out_png)
    A, b = fit["A"], fit["b"]
    Aplus, Aminus = fit["A_plus"], fit["A_minus"]

    xmin = max(np.min(N_all) * 0.7, 1.0)
    xmax = np.max(N_all) * 1.3
    N_line = np.logspace(np.log10(xmin), np.log10(xmax), 300)

    Y_med = A * (N_line ** b)
    Y_up = Aplus * (N_line ** b)
    Y_lo = Aminus * (N_line ** b)

    plt.figure(figsize=(7.5, 5.5))
    plt.loglog(N_all, Y_all, "o", markersize=4, alpha=0.8, label="Experimental points")
    plt.loglog(N_line, Y_med, "-", linewidth=2, label="P50% fit")
    plt.loglog(N_line, Y_up, "--", linewidth=1.8, label="+{:.3g}σ".format(fit["n_sigma"]))
    plt.loglog(N_line, Y_lo, "--", linewidth=1.8, label="-{:.3g}σ".format(fit["n_sigma"]))

    Y50_ref = y_at_N(A, b, Nref)
    plt.loglog([Nref], [Y50_ref], "s", markersize=6, label="P50 at N={:.3g}".format(Nref))

    plt.xlabel("Cycles to failure $N_f$")
    plt.ylabel("Findley parameter $\\widetilde{F}$")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=300)
    plt.show()


# ============================================================
# Core Findley evaluation for a single point
# ============================================================
def compute_findley_for_point(rc_m, mode, planeI, alpha_deg, etaI, etaIII,
                              rho_geo_m, combo_unit, R, Mmax_nmm):
    """
    Findley at control radius:
      rho_eq = eta(alpha)*rc
      s_geom = (rho_eq/rho_geo)^(lambda(alpha)-1)

      g0 = Mmax/UNIT_LOAD
      Findley = s_geom * g0 * combo_unit
    """
    modeU = mode.upper()

    if modeU == "MI":
        eta = interp_clamped(alpha_deg, etaI["alpha"], etaI[planeI])
    else:
        eta = interp_clamped(alpha_deg, etaIII["alpha"], etaIII["eta"])

    rho_eq = float(eta) * float(rc_m)
    if rho_eq <= 0 or rho_geo_m <= 0:
        return float("nan")

    lam = get_lambda(modeU, alpha_deg)
    s_geom = (rho_eq / rho_geo_m) ** (lam - 1.0)

    g0 = float(Mmax_nmm) / float(UNIT_LOAD_NMM)

    F = s_geom * g0 * float(combo_unit)
    return float(F)


# ============================================================
# Folder scanning
# ============================================================
def find_slopes_files(root):
    root = Path(root)
    modeI = None
    modeIII = None
    for p in root.rglob("slopes_vs_alpha_findley_modeI.dat"):
        modeI = p
        break
    for p in root.rglob("slopes_vs_alpha_findley_modeIII.dat"):
        modeIII = p
        break
    return modeI, modeIII


def find_author_dirs(root):
    root = Path(root)
    author_dirs = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name.lower() in ("export_txt", "export", "out"):
            continue
        if (d / "Modo_I").is_dir() or (d / "Modo_III").is_dir():
            author_dirs.append(d)
    return sorted(author_dirs)


def scan_mode_folder(mode_dir, n_sphere):
    """
    Returns:
      results_info: dict loc -> dict(alpha_deg, node, tau_unit[], sig_unit[])
      specimen_files: list of specimen txt paths
    """
    data_dir = Path(mode_dir) / "Data"
    if not data_dir.is_dir():
        return {}, []

    results_csvs = list(data_dir.glob("RESULTS_*.csv"))
    results_info = {}

    for csvp in results_csvs:
        loc = location_from_results_csv_name(csvp)
        a_deg = alpha_from_results_csv_name(csvp)

        rows = read_results_csv(csvp)
        node, S_unit = pick_max_vm_node(rows)

        tau_u, sig_u = plane_tau_sig_arrays(S_unit, n_sphere=n_sphere)

        results_info[loc] = {
            "alpha_deg": float(a_deg),
            "node": int(node),
            "tau_unit": tau_u,
            "sig_unit": sig_u,
            "_combo_cache": {},  # R -> combo_unit
        }

    specimen_files = sorted(data_dir.glob("*_R_*.txt"))
    return results_info, specimen_files


# ============================================================
# Output point files in a PGF-friendly tree
# ============================================================
def make_out_point_file(out_data_root, mode, author, loading, cond, R, loc):
    mode = mode.upper()
    folder_mode = "MI" if mode == "MI" else "MIII"
    results_dir = Path(out_data_root) / folder_mode / author / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    Rtxt = ("{:g}".format(float(R))).replace("+", "")
    fname = "{}_{}_R_{}_{}_{}.txt".format(loading, cond, Rtxt, PARAM_TAG, loc.upper())
    return results_dir / fname


# ============================================================
# Main
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", type=str, default=".", help="Root directory containing author folders and slopes files")
    ap.add_argument("--outdir", type=str, default="export_txt", help="Output directory")
    ap.add_argument("--planeI", type=str, default="strain", choices=["stress", "strain"],
                    help="Which eta(alpha) to use for Mode I (plane stress vs plane strain)")

    ap.add_argument("--rcmin", type=float, default=0.001, help="rc minimum [mm]")
    ap.add_argument("--rcmax", type=float, default=1, help="rc maximum [mm]")
    ap.add_argument("--rcsteps", type=int, default=500, help="number of rc values to test (grid search)")

    ap.add_argument("--kF", type=float, default=DEFAULT_KF, help="Findley kF parameter")
    ap.add_argument("--nsigma", type=float, default=2, help="std dev multiplier for scatter bands (1.2815 for 95%)")
    ap.add_argument("--Nref", type=float, default=2.0e6, help="Reference cycles for reporting Y@Nref and T_ECP")
    ap.add_argument("--defaultSy", type=float, default=DEFAULT_SY_MPA, help="Default Sy [MPa] if not in author folder name")
    ap.add_argument("--defaultRhoGeo", type=float, default=DEFAULT_RHOGEO_MM, help="Default FE rho_geo [mm] if not in author folder name")
    ap.add_argument("--sphere", type=int, default=DEFAULT_SPHERE, help="Sphere points for plane search (speed/accuracy)")

    # Jupyter-safe (ignores --f=kernel.json)
    args, _unknown = ap.parse_known_args(argv)

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # slopes files
    modeI_path, modeIII_path = find_slopes_files(root)
    if modeI_path is None:
        raise SystemExit("Missing Mode I slopes file: slopes_vs_alpha_findley_modeI.dat")
    if modeIII_path is None:
        raise SystemExit("Missing Mode III slopes file: slopes_vs_alpha_findley_modeIII.dat")

    eta_modeI = build_eta_modeI(modeI_path)
    eta_modeIII = build_eta_modeIII(modeIII_path)

    # author dirs
    author_dirs = find_author_dirs(root)
    if not author_dirs:
        raise SystemExit("No author folders found under: {}".format(root))

    # ---- build point records (precompute combo_unit for each point; no plane-search inside rc loop)
    point_recs = []

    for author_dir in author_dirs:
        author_full = author_dir.name
        author_base, rho_geo_mm, Sy_MPa = parse_author_folder(
            author_full,
            default_rho_mm=float(args.defaultRhoGeo),
            default_Sy_MPa=float(args.defaultSy)
        )
        rho_geo_m = rho_geo_mm * 1e-3

        # Mode I
        if (author_dir / "Modo_I").is_dir():
            results_info, specimen_files = scan_mode_folder(author_dir / "Modo_I", n_sphere=int(args.sphere))
            for spec in specimen_files:
                R, loading, cond, rows = parse_specimen_file(spec)
                for loc, locinfo in results_info.items():
                    # cache combo for this R
                    keyR = float(R)
                    if keyR not in locinfo["_combo_cache"]:
                        locinfo["_combo_cache"][keyR] = best_findley_combo_for_R(
                            locinfo["tau_unit"], locinfo["sig_unit"], R=keyR, kF=float(args.kF)
                        )
                    combo_unit = locinfo["_combo_cache"][keyR]

                    for (Nf, Mmax) in rows:
                        point_recs.append({
                            "Mode": "MI",
                            "Author": author_base,
                            "Loading": loading,
                            "Post": cond,
                            "R": float(R),
                            "Location": loc,
                            "alpha_deg": float(locinfo["alpha_deg"]),
                            "rho_geo_m": float(rho_geo_m),
                            "Sy_MPa": float(Sy_MPa),      # kept for metadata consistency
                            "combo_unit": float(combo_unit),
                            "Nf": float(Nf),
                            "Mmax": float(Mmax),
                        })

        # Mode III
        if (author_dir / "Modo_III").is_dir():
            results_info, specimen_files = scan_mode_folder(author_dir / "Modo_III", n_sphere=int(args.sphere))
            for spec in specimen_files:
                R, loading, cond, rows = parse_specimen_file(spec)
                for loc, locinfo in results_info.items():
                    keyR = float(R)
                    if keyR not in locinfo["_combo_cache"]:
                        locinfo["_combo_cache"][keyR] = best_findley_combo_for_R(
                            locinfo["tau_unit"], locinfo["sig_unit"], R=keyR, kF=float(args.kF)
                        )
                    combo_unit = locinfo["_combo_cache"][keyR]

                    for (Nf, Mmax) in rows:
                        point_recs.append({
                            "Mode": "MIII",
                            "Author": author_base,
                            "Loading": loading,
                            "Post": cond,
                            "R": float(R),
                            "Location": loc,
                            "alpha_deg": float(locinfo["alpha_deg"]),
                            "rho_geo_m": float(rho_geo_m),
                            "Sy_MPa": float(Sy_MPa),
                            "combo_unit": float(combo_unit),
                            "Nf": float(Nf),
                            "Mmax": float(Mmax),
                        })

    if not point_recs:
        raise SystemExit("No experimental points found (check Data folders and specimen files).")

    # ---- rc grid (mm -> m)
    rc_grid_mm = np.linspace(float(args.rcmin), float(args.rcmax), int(args.rcsteps))
    rc_grid_m = rc_grid_mm * 1e-3

    sweep_rows = []
    best = None

    for rc_mm, rc_m in zip(rc_grid_mm, rc_grid_m):
        N_list = []
        Y_list = []

        for pr in point_recs:
            Y = compute_findley_for_point(
                rc_m=rc_m,
                mode=pr["Mode"],
                planeI=args.planeI,
                alpha_deg=pr["alpha_deg"],
                etaI=eta_modeI,
                etaIII=eta_modeIII,
                rho_geo_m=pr["rho_geo_m"],
                combo_unit=pr["combo_unit"],
                R=pr["R"],
                Mmax_nmm=pr["Mmax"],
            )
            if Y > 0 and np.isfinite(Y):
                N_list.append(pr["Nf"])
                Y_list.append(Y)

        if len(N_list) < 5:
            continue

        N_all = np.array(N_list, float)
        Y_all = np.array(Y_list, float)
        fit = fit_powerlaw_with_scatter(N_all, Y_all, float(args.nsigma))

        sweep_rows.append({
            "rc_mm": float(rc_mm),
            "std_ln": float(fit["std_ln"]),
            "T_ECP": float(fit["T_ecp"]),
            "A": float(fit["A"]),
            "b": float(fit["b"]),
        })

        if best is None or fit["std_ln"] < best["fit"]["std_ln"]:
            best = {"rc_mm": float(rc_mm), "rc_m": float(rc_m), "fit": fit, "N_all": N_all, "Y_all": Y_all}

    if best is None:
        raise SystemExit("Optimization failed: no rc produced enough valid points.")

    best_rc_mm = best["rc_mm"]
    best_rc_m = best["rc_m"]
    fit_all = best["fit"]
    N_all = best["N_all"]
    Y_all = best["Y_all"]

    # ---- Save sweep (fatigue scatter vs rc)
    write_csv(outdir / "rc_sweep.csv", sweep_rows, header=["rc_mm", "std_ln", "T_ECP", "A", "b"])
    (outdir / "best_rc.txt").write_text("best_rc_mm = {:.10g}\n".format(best_rc_mm), encoding="utf-8")

    # ---- Values at Nref (+/-nsigma)
    Y50_Nref = y_at_N(fit_all["A"], fit_all["b"], float(args.Nref))
    YU_Nref  = y_at_N(fit_all["A_plus"], fit_all["b"], float(args.Nref))
    YL_Nref  = y_at_N(fit_all["A_minus"], fit_all["b"], float(args.Nref))

    # ---- TikZ macro file (safe names: no underscores, no digits)
    def fmt_e(x):
        s = "{:.10e}".format(float(x))
        return s.replace("+", "")

    def fmt_g(x):
        return "{:.6g}".format(float(x))

    def fmt_pct(x):
        return "{:.1f}".format(float(x))

    def fmt_num(x):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return "{:.10g}".format(float(x))

    P_high = 100.0 * normal_cdf(float(args.nsigma))
    P_low  = 100.0 * (1.0 - normal_cdf(float(args.nsigma)))

    tex = []
    tex.append("% Auto-generated power-law fit (GLOBAL) at BEST rc - Findley - safe macro names (no underscores, no digits)")
    tex.append("\\def\\FIFitRcmm{{{}}}".format(fmt_g(best_rc_mm)))
    tex.append("\\def\\FIFitA{{{}}}".format(fmt_e(fit_all["A"])))
    tex.append("\\def\\FIFitb{{{}}}".format(fmt_e(fit_all["b"])))
    tex.append("\\def\\FIFitStdLn{{{}}}".format(fmt_e(fit_all["std_ln"])))
    tex.append("\\def\\FIFitNsigma{{{}}}".format(fmt_num(float(args.nsigma))))
    tex.append("\\def\\FIFitAplus{{{}}}".format(fmt_e(fit_all["A_plus"])))
    tex.append("\\def\\FIFitAminus{{{}}}".format(fmt_e(fit_all["A_minus"])))
    tex.append("\\def\\FIFitTECP{{{}}}".format(fmt_g(fit_all["T_ecp"])))
    tex.append("\\def\\FIFitNref{{{}}}".format(fmt_num(float(args.Nref))))
    tex.append("\\def\\FIFitFIupNref{{{}}}".format(fmt_e(YU_Nref)))
    tex.append("\\def\\FIFitFImedNref{{{}}}".format(fmt_e(Y50_Nref)))
    tex.append("\\def\\FIFitFIloNref{{{}}}".format(fmt_e(YL_Nref)))
    tex.append("\\def\\FIFitPsHigh{{{}}}".format(fmt_pct(P_high)))
    tex.append("\\def\\FIFitPsLow{{{}}}".format(fmt_pct(P_low)))

    (outdir / "powerlaw_fit_fi_all.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---- Write per-series files to outdir/Data/...
    out_data_root = outdir / "Data"
    out_data_root.mkdir(parents=True, exist_ok=True)

    buckets = {}
    for pr in point_recs:
        Y = compute_findley_for_point(
            rc_m=best_rc_m,
            mode=pr["Mode"],
            planeI=args.planeI,
            alpha_deg=pr["alpha_deg"],
            etaI=eta_modeI,
            etaIII=eta_modeIII,
            rho_geo_m=pr["rho_geo_m"],
            combo_unit=pr["combo_unit"],
            R=pr["R"],
            Mmax_nmm=pr["Mmax"],
        )
        if not (Y > 0 and np.isfinite(Y)):
            continue

        key = (pr["Mode"], pr["Author"], pr["Loading"], pr["Post"], pr["R"], pr["Location"])
        buckets.setdefault(key, [])
        buckets[key].append((pr["Nf"], Y))

    detailed_rows = []
    for key, pts in buckets.items():
        mode, author, loading, post, R, loc = key
        out_file = make_out_point_file(out_data_root, mode, author, loading, post, R, loc)

        pts = sorted(pts, key=lambda t: t[0])
        arr = []
        for (Nf, Y) in pts:
            Npred = cycles_from_ecp(Y, fit_all["A"], fit_all["b"])
            arr.append((Nf, Y, Npred))

        np.savetxt(str(out_file), np.array(arr, float), fmt="%.10e",
                   header="Nf\t{}\tNpred".format(PARAM_TAG), comments="")

        detailed_rows.append({
            "Mode": mode,
            "Author": author,
            "Loading": loading,
            "Condition": post,
            "Load ratio": R,
            "Failure location": _map_location(loc),
            "nData": len(arr),
            "File": str(out_file),
        })

    write_csv(outdir / "test_matrix_detailed.csv", detailed_rows,
              header=["Mode", "Author", "Loading", "Condition", "Load ratio", "Failure location", "nData", "File"])

    # Compact test matrix (one row per Mode+Author)
    by_ma = {}
    for r in detailed_rows:
        k = (r["Mode"], r["Author"])
        by_ma.setdefault(k, {
            "Mode": r["Mode"],
            "Author": r["Author"],
            "Loading_set": set(),
            "Cond_set": set(),
            "R_list": [],
            "Loc_set": set(),
            "nData": 0
        })
        by_ma[k]["Loading_set"].add(r["Loading"])
        by_ma[k]["Cond_set"].add(r["Condition"])
        by_ma[k]["R_list"].append(r["Load ratio"])
        by_ma[k]["Loc_set"].add(r["Failure location"])
        by_ma[k]["nData"] += int(r["nData"])

    summary_rows = []
    for (mode, author), g in sorted(by_ma.items(), key=lambda x: (x[0][0], x[0][1])):
        summary_rows.append({
            "Mode": mode,
            "Author": author,
            "Loading": ", ".join(sorted([x for x in g["Loading_set"] if x])) if g["Loading_set"] else "NA",
            "Condition": ", ".join(sorted([x for x in g["Cond_set"] if x])) if g["Cond_set"] else "NA",
            "Load ratio": _format_R_values(g["R_list"]),
            "Failure location": ", ".join(sorted([x for x in g["Loc_set"] if x])) if g["Loc_set"] else "NA",
            "nData": int(g["nData"]),
        })

    write_csv(outdir / "test_matrix_summary.csv", summary_rows,
              header=["Mode", "Author", "Loading", "Condition", "Load ratio", "Failure location", "nData"])

    caption = (
        "Test matrix of the experimental dataset after ECP post-processing using the optimal control radius $r_c$. "
        "Condition codes: AW = as-welded; SRE = stress-relieved. "
        "Failure location is reported as Toe or Root. "
        "When multiple load ratios are available for the same (Mode, Author), they are reported as hyphen-separated values. "
        "Critical-plane parameter: Findley ({}).".format(PARAM_TAG)
    )
    df_to_latex_table(
        summary_rows,
        cols=["Mode", "Author", "Loading", "Condition", "Load ratio", "Failure location", "nData"],
        outpath=outdir / "test_matrix_summary.tex",
        caption=caption,
        label="tab:test_matrix"
    )

    plot_loglog_fit(N_all, Y_all, fit_all, outdir / "fit_plot_loglog.png", float(args.Nref))

    print("\n=== BEST rc (Findley) ===")
    print("rc = {:.6g} mm".format(best_rc_mm))
    print("A  = {:.6e}".format(fit_all["A"]))
    print("b  = {:.6e}".format(fit_all["b"]))
    print("std_ln = {:.6e}".format(fit_all["std_ln"]))
    print("T_ECP  = {:.6g}".format(fit_all["T_ecp"]))
    print("kF     = {:.6g}".format(float(args.kF)))

    print("\nWrote:")
    print(outdir / "powerlaw_fit_fi_all.tex")
    print(outdir / "test_matrix_detailed.csv")
    print(outdir / "test_matrix_summary.csv")
    print(outdir / "test_matrix_summary.tex")
    print(outdir / "fit_plot_loglog.png")
    print(outdir / "rc_sweep.csv")
    print(outdir / "best_rc.txt")
    print(outdir / "Data" / "MI")
    print(outdir / "Data" / "MIII")


if __name__ == "__main__":
    main()