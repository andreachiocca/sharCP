# MAIN_postproc_FS_SWT_multi_rho.py
# ---------------------------------
# Post-process ANSYS results for welded joint. For each specimen AAA_R_XXX.txt in Data/:
#   * read Nf and Mmax (Nmm)
#   * pick max-vM node from RESULTS_TOE.csv and RESULTS_ROOT.csv (unit 1000 Nmm)
#   * scale tensors by load and by geometry s = (rho_eq / rho_geo)^(lambda-1)
#     where rho_eq can differ by metric (FS vs SWT) and location (TOE vs ROOT)
#   * build tensors at two load steps: step0 = Mmax, step1 = R*Mmax; compute FS/SWT using both steps
#   * compute FS (original best γ_amp) and SWT (3D)
#   * write: out/AAA_R_XXX_FS_TOE.txt, FS_ROOT.txt, SWT_TOE.txt, SWT_ROOT.txt
#     (each with Nf, ECP value, and Npred computed from ECP = A*N^b)
#
# Optional input in Data/:
#   - rho_eq_map.json   OR   rho_eq_map.csv
#     with keys metric in {"FS","SWT"} and location in {"TOE","ROOT"},
#     values given in mm (will be converted to meters).

from pathlib import Path
import csv, json, math, re
import numpy as np

# ========= CONFIG (defaults if no mapping file found) =========
DATA_DIR   = Path("Data")
OUT_DIR    = Path("out"); OUT_DIR.mkdir(exist_ok=True)

# Material for FS
Sy  = 355     # MPa
kFS = 1.5     # FS parameter

# CAD notch radius used in FE (geometric radius)
rho_geo  = 0.30e-3  # m

# Eigen exponents (keep fixed as you asked)
lambda_toe  = 0.600190   # set your toe λ
lambda_root = 0.5        # set your root λ (can be same or different)

# --- Fatigue curve parameters (ECP = A * N^b) ---
# Set these at the beginning as requested. Provide separate pairs for FS and SWT.
A_FS  = 0.01392      # units consistent with FS damage parameter
b_FS  = -0.1938
A_SWT = 64.76      # units consistent with SWT damage parameter
b_SWT = -0.3684
# ------------------------------------------------

# Default equivalent radii in mm (used only if no map file is provided)
DEFAULT_RHO_EQ_MM = {
    "FS":  {"TOE": 10, "ROOT": 1.686},   # mm
    "SWT": {"TOE": 5.54, "ROOT": 0.536},   # mm
}

# Sphere sampling for plane search
N_SPHERE = 6000
# ===============================================================


# ---------- utilities ----------
def mm_to_m(x_mm: float) -> float:
    return float(x_mm) * 1e-3

def build_eps_tensor_from_EPEL(exx, eyy, ezz, exy, eyz, exz):
    """EPEL provides engineering shear strains; convert to tensor components ε_xy = γ_xy/2, etc."""
    return np.array([
        [exx, 0.5*exy, 0.5*exz],
        [0.5*exy, eyy, 0.5*eyz],
        [0.5*exz, 0.5*eyz, ezz]
    ], float)

def build_sig_tensor(sxx, syy, szz, sxy, syz, sxz):
    return np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz]
    ], float)

def von_mises(sig):
    s = np.array(sig, float)
    m = np.trace(s)/3.0
    dev = s - np.eye(3)*m
    return math.sqrt(1.5*np.sum(dev*dev))

def _basis_from_n(n):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    up = np.array([0.0,0.0,1.0]) if abs(n[2]) < 0.9 else np.array([1.0,0.0,0.0])
    t1 = np.cross(n, up); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return n, t1, t2

def cycles_from_ecp(ecp: float, A: float, b: float) -> float:
    """
    Invert ECP = A * N^b  =>  N = (ECP/A)^(1/b).
    Returns NaN if inputs are not physically meaningful (A<=0, b==0, ecp<=0).
    """
    if A <= 0 or b == 0 or ecp <= 0:
        return float("nan")
    try:
        return float((ecp / A) ** (1.0 / b))
    except Exception:
        return float("nan")

# ---------- FS (original best γ_amp plane) ----------
# ---------- Critical-plane search: FS & SWT (2 load steps) ----------
# These implementations mirror the reference formulations in fs.py and swt.py:
#   - Use stress/strain tensors at TWO load steps (step 0 and step 1).
#   - Rotate tensors to each candidate plane frame (local z = plane normal).
#   - FS:
#       Δγ = sqrt((Exz0-Exz1)^2 + (Eyz0-Eyz1)^2)
#       σn,max = max( max(Szz0,0), max(Szz1,0) )
#       FS = Δγ * (1 + k_FS * σn,max / Sy)
#       plane-selection metric = Δγ
#   - SWT:
#       Δε = |Ezz0 - Ezz1|
#       σn,max = max( max(Szz0,0), max(Szz1,0) )
#       SWT = σn,max * (Δε/2) = Δε * σn,max / 2
#       plane-selection metric = Δε
def _rotate_tensor(A, Q):
    """Rotate 2nd-order tensor A into local frame with columns of Q as local axes in global coords."""
    return Q.T @ A @ Q

def fs_3d_cp(S0, E0, S1, E1, Sy, kFS, n_sphere=N_SPHERE):
    """Fatemi–Socie critical-plane (3D) using two load steps (matches fs.py)."""
    Sy = float(Sy); kFS = float(kFS)
    if Sy <= 0.0:
        raise ValueError(f"Sy must be > 0. Got Sy={Sy}")
    if kFS < 0.0:
        raise ValueError(f"kFS must be >= 0. Got kFS={kFS}")

    phi = (1.0 + 5**0.5) / 2.0
    best_metric = -np.inf
    best_damage = float("-inf")

    S0 = np.asarray(S0, float); E0 = np.asarray(E0, float)
    S1 = np.asarray(S1, float); E1 = np.asarray(E1, float)

    for k in range(n_sphere):
        z = 1.0 - 2.0*(k+0.5)/n_sphere
        r = math.sqrt(max(1.0 - z*z, 0.0))
        th = 2.0*math.pi*k/phi
        n = np.array([r*math.cos(th), r*math.sin(th), z])
        n, t1, t2 = _basis_from_n(n)

        # Local axes: x=t1, y=t2, z=n (plane normal)
        Q = np.stack([t1, t2, n], axis=1)

        S0r = _rotate_tensor(S0, Q)
        E0r = _rotate_tensor(E0, Q)
        S1r = _rotate_tensor(S1, Q)
        E1r = _rotate_tensor(E1, Q)

        # σ_nn = Szz (clipped at 0) and σn,max across steps
        sig0 = max(float(S0r[2, 2]), 0.0)
        sig1 = max(float(S1r[2, 2]), 0.0)
        smax = max(sig0, sig1)

        # Δγ based on Exz and Eyz components (tensor shear strains)
        dgam_a = float(E0r[0, 2] - E1r[0, 2])
        dgam_b = float(E0r[1, 2] - E1r[1, 2])
        delta_gamma = float(math.sqrt(dgam_a*dgam_a + dgam_b*dgam_b))
        FS_m = float(delta_gamma * (1.0 + kFS * (smax / Sy)))

        metric = delta_gamma
        if metric > best_metric:
            best_metric = metric
            best_damage = float(delta_gamma * (1.0 + kFS * (smax / Sy)))

    return best_damage

def swt_3d_cp(S0, E0, S1, E1, n_sphere=N_SPHERE):
    """Smith–Watson–Topper critical-plane (3D) using two load steps (matches swt.py)."""
    phi = (1.0 + 5**0.5) / 2.0
    best_metric = -np.inf
    best_damage = float("-inf")

    S0 = np.asarray(S0, float); E0 = np.asarray(E0, float)
    S1 = np.asarray(S1, float); E1 = np.asarray(E1, float)

    for k in range(n_sphere):
        z = 1.0 - 2.0*(k+0.5)/n_sphere
        r = math.sqrt(max(1.0 - z*z, 0.0))
        th = 2.0*math.pi*k/phi
        n = np.array([r*math.cos(th), r*math.sin(th), z])
        n, t1, t2 = _basis_from_n(n)

        Q = np.stack([t1, t2, n], axis=1)

        S0r = _rotate_tensor(S0, Q)
        E0r = _rotate_tensor(E0, Q)
        S1r = _rotate_tensor(S1, Q)
        E1r = _rotate_tensor(E1, Q)

        sig0 = max(float(S0r[2, 2]), 0.0)
        sig1 = max(float(S1r[2, 2]), 0.0)
        smax = max(sig0, sig1)

        delta_eps = abs(float(E0r[2, 2] - E1r[2, 2]))
        metric = float(delta_eps)
        if metric > best_metric:
            best_metric = metric
            best_damage = float(delta_eps * smax / 2.0)

    return best_damage

# ---------- data readers ----------

def read_results_csv(path_csv: Path):
    """Return list of (node, eps_tensor, sig_tensor) from RESULTS_*.csv."""
    rows = []
    with path_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip headers/non-numeric
            try:
                _ = float(row[0]); _ = float(row[1])
            except Exception:
                continue
            if len(row) < 2+6+6+6:
                continue
            node = int(float(row[0]))
            exx, eyy, ezz, exy, eyz, exz = map(float, row[2:8])
            eps = build_eps_tensor_from_EPEL(exx, eyy, ezz, exy, eyz, exz)
            # skip plastic: row[8:14]
            sxx, syy, szz, sxy, syz, sxz = map(float, row[14:20])
            sig = build_sig_tensor(sxx, syy, szz, sxy, syz, sxz)
            rows.append((node, eps, sig))
    if not rows:
        raise RuntimeError(f"No numeric rows parsed in {path_csv}")
    return rows

def pick_max_vm_node(rows):
    vms = [von_mises(sig) for (_, _, sig) in rows]
    idx = int(np.argmax(vms))
    return rows[idx]  # (node, eps, sig)

def parse_specimen_file(path_txt: Path):
    """
    Return (R, [(Nf, Mmax_Nmm), ...]) from AAA_R_XXX.txt.
    Cols: Nf, nominal_stress(ignored), Mmax(Nmm).
    """
    m = re.search(r"_R_([-\d\.]+)\.txt$", path_txt.name)
    if not m:
        raise RuntimeError(f"Cannot parse load ratio from filename: {path_txt.name}")
    R = float(m.group(1))
    data = []
    with path_txt.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                Nf = float(parts[0])
                Mmax = float(parts[2])  # Nmm
                data.append((Nf, Mmax))
            except Exception:
                continue
    if not data:
        raise RuntimeError(f"No numeric rows found in {path_txt}")
    return R, data

# ---------- rho_eq mapping (FS/SWT × TOE/ROOT) ----------
def load_rho_eq_map(data_dir: Path):
    json_path = data_dir / "rho_eq_map.json"
    csv_path  = data_dir / "rho_eq_map.csv"
    mapping = {"FS": {}, "SWT": {}}
    if json_path.exists():
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        for metric in ["FS", "SWT"]:
            for loc in ["TOE", "ROOT"]:
                mapping[metric][loc] = mm_to_m(float(obj[metric][loc]))
        return mapping
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric  = row["metric"].strip().upper()
                loc     = row["location"].strip().upper()
                rho_mm  = float(row["rho_mm"])
                if metric in mapping and loc in mapping[metric]:
                    mapping[metric][loc] = mm_to_m(rho_mm)
                else:
                    # initialize missing keys if header order different
                    if metric not in mapping:
                        mapping[metric] = {}
                    mapping[metric][loc] = mm_to_m(rho_mm)
        # ensure defaults if some entries missing
        for metric in ["FS", "SWT"]:
            for loc in ["TOE", "ROOT"]:
                if loc not in mapping.get(metric, {}):
                    mapping.setdefault(metric, {})[loc] = mm_to_m(DEFAULT_RHO_EQ_MM[metric][loc])
        return mapping
    # fallback to defaults
    for metric in ["FS", "SWT"]:
        mapping[metric]["TOE"]  = mm_to_m(DEFAULT_RHO_EQ_MM[metric]["TOE"])
        mapping[metric]["ROOT"] = mm_to_m(DEFAULT_RHO_EQ_MM[metric]["ROOT"])
    return mapping

# ---------- main driver ----------
def main():
    results_toe  = DATA_DIR / "RESULTS_TOE.csv"
    if not results_toe.exists():
        raise FileNotFoundError("Missing RESULTS_TOE.csv and/or RESULTS_ROOT.csv in Data/")

    rho_map = load_rho_eq_map(DATA_DIR)
    # Pre-read & select critical node once per location (unit load 1000 Nmm)
    rows_toe  = read_results_csv(results_toe)
    node_toe,  E_unit_toe,  S_unit_toe  = pick_max_vm_node(rows_toe)

    print(f"[INFO] TOE  max-vM node: {node_toe}")
    print(f"[INFO] Using rho_eq map (meters): {rho_map}")
    print(f"[INFO] Fatigue curve (FS):  ECP = {A_FS} * N^{b_FS}")
    print(f"[INFO] Fatigue curve (SWT): ECP = {A_SWT} * N^{b_SWT}")

    spec_files = sorted(DATA_DIR.glob("*_R_*.txt"))
    if not spec_files:
        raise FileNotFoundError(f"No specimen files *_R_*.txt in {DATA_DIR}")

    for spec_path in spec_files:
        spec_name = spec_path.stem
        R, specimens = parse_specimen_file(spec_path)

        fs_toe_out   = OUT_DIR / f"{spec_name}_FS_TOE.txt"
        swt_toe_out  = OUT_DIR / f"{spec_name}_SWT_TOE.txt"

        # store as (Nf, ECP_value, Npred)
        fs_toe_vals, fs_root_vals = [], []
        swt_toe_vals, swt_root_vals = [], []

        for (Nf, Mmax) in specimens:
            load_scale_max = float(Mmax) / 1  # unit run was 1000 Nmm
            load_scale_min = float(R) * float(Mmax) / 1  # minimum step (Mmin = R*Mmax)

            # ----- TOE -----
            # FS scaling
            rho_eq_toe_FS  = rho_map["FS"]["TOE"]
            s_geom_toe_FS  = (rho_eq_toe_FS / rho_geo)**(lambda_toe - 1.0)
            S0_toe_FS  = load_scale_max * s_geom_toe_FS * S_unit_toe
            E0_toe_FS  = load_scale_max * s_geom_toe_FS * E_unit_toe
            S1_toe_FS  = load_scale_min * s_geom_toe_FS * S_unit_toe
            E1_toe_FS  = load_scale_min * s_geom_toe_FS * E_unit_toe
            FS_toe_val = fs_3d_cp(S0_toe_FS, E0_toe_FS, S1_toe_FS, E1_toe_FS, Sy, kFS, n_sphere=N_SPHERE)
            Npred_FS_toe   = cycles_from_ecp(FS_toe_val, A_FS, b_FS)
            fs_toe_vals.append((Nf, FS_toe_val, Npred_FS_toe))

            # SWT scaling
            rho_eq_toe_SWT  = rho_map["SWT"]["TOE"]
            s_geom_toe_SWT  = (rho_eq_toe_SWT / rho_geo)**(lambda_toe - 1.0)
            S0_toe_SWT  = load_scale_max * s_geom_toe_SWT * S_unit_toe
            E0_toe_SWT  = load_scale_max * s_geom_toe_SWT * E_unit_toe
            S1_toe_SWT  = load_scale_min * s_geom_toe_SWT * S_unit_toe
            E1_toe_SWT  = load_scale_min * s_geom_toe_SWT * E_unit_toe
            SWT_toe_val = swt_3d_cp(S0_toe_SWT, E0_toe_SWT, S1_toe_SWT, E1_toe_SWT, n_sphere=N_SPHERE)
            Npred_SWT_toe   = cycles_from_ecp(SWT_toe_val, A_SWT, b_SWT)
            swt_toe_vals.append((Nf, SWT_toe_val, Npred_SWT_toe))


        # Save three-column outputs: Nf, ECP_value, Npred
        np.savetxt(fs_toe_out,  np.array(fs_toe_vals,  float), fmt="%.10e",
                   header="Nf\tFS_toe\tNpred", comments="")
        np.savetxt(swt_toe_out,  np.array(swt_toe_vals,  float), fmt="%.10e",
                   header="Nf\tSWT_toe\tNpred", comments="")

        print(f"[{spec_name}] R={R:+.3f} saved:")
        print(f"  - {fs_toe_out.name}   (TOE FS, ρ_eq={rho_map['FS']['TOE']:.3e} m, columns: Nf, FS_toe, Npred)")
        print(f"  - {swt_toe_out.name}  (TOE SWT, ρ_eq={rho_map['SWT']['TOE']:.3e} m, columns: Nf, SWT_toe, Npred)")

if __name__ == "__main__":
    main()
