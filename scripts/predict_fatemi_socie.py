#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prevision_FS.py
----------------
Prediction script for Fatemi-Socie ECP master curve.

Expected folder layout when run from prevision_FS/:
  export_txt/                         # copied from FS calibration output
    powerlaw_fit_fs_all.tex
    best_rc.txt                       # optional, macro file is enough when it has FSFitRcmm
  slopes_vs_alpha_fs_modeI.dat        # required, can be anywhere under --root or --calibdir
  slopes_vs_alpha_fs_modeIII.dat      # required, can be anywhere under --root or --calibdir
  Author__rho(mm)__Sy(MPa)/Modo_I/Data/... and/or Modo_III/Data/...

Also supports Mode_I_pstress and Mode_I_pstrain folders. These override --planeI for Mode I.

Run:
  python prevision_FS.py --root . --calibdir export_txt --outdir prevision_txt --kFS 0.50
"""
from __future__ import print_function
import argparse, csv, math, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

DEFAULT_SY_MPA=355.0
DEFAULT_RHOGEO_MM=1.0
DEFAULT_KFS=0.5
UNIT_LOAD_NMM=1000.0
DEFAULT_SPHERE=6000
PARAM_TAG="FS"


def _load_txt_table(path, min_cols=2):
    rows=[]
    for ln in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        ln=ln.strip()
        if not ln or ln.startswith('#'): continue
        try: vals=[float(t.replace(',','.')) for t in ln.replace(';',' ').split()]
        except Exception: continue
        if len(vals)>=min_cols: rows.append(vals)
    if not rows: raise ValueError('No numeric data parsed from {}'.format(path))
    ncol=max(len(r) for r in rows); arr=np.full((len(rows),ncol),np.nan,float)
    for i,r in enumerate(rows): arr[i,:len(r)]=r
    return arr

def build_eta_modeI(path):
    a=_load_txt_table(path,3); idx=np.argsort(a[:,0])
    return {'alpha':a[idx,0].astype(float),'stress':a[idx,1].astype(float),'strain':a[idx,2].astype(float)}
def build_eta_modeIII(path):
    a=_load_txt_table(path,2); idx=np.argsort(a[:,0])
    return {'alpha':a[idx,0].astype(float),'eta':a[idx,1].astype(float)}
def interp_clamped(x,xp,fp):
    x=float(x); xp=np.asarray(xp,float); fp=np.asarray(fp,float)
    if x<=xp[0]: return float(fp[0])
    if x>=xp[-1]: return float(fp[-1])
    return float(np.interp(x,xp,fp))

def parse_author_folder(name, default_rho_mm=DEFAULT_RHOGEO_MM, default_Sy_MPa=DEFAULT_SY_MPA):
    s=str(name).strip()
    if '__' in s:
        p=s.split('__'); author=p[0].strip(); rho=default_rho_mm; sy=default_Sy_MPa
        if len(p)>=2:
            try: rho=float(p[1].replace(',','.').replace('_','.'))
            except Exception: pass
        if len(p)>=3:
            try: sy=float(p[2].replace(',','.').replace('_','.'))
            except Exception: pass
        return author,float(rho),float(sy)
    toks=s.split('_')
    if len(toks)>=3:
        try: return '_'.join(toks[:-2]),float((toks[-2]+'.'+toks[-1]).replace(',','.')),float(default_Sy_MPa)
        except Exception: pass
    return s,float(default_rho_mm),float(default_Sy_MPa)

def alpha_from_results_csv_name(p):
    m=re.search(r'RESULTS_(TOE|ROOT)_(.+)$',Path(p).stem,re.I)
    return 0.0 if not m else float(m.group(2).replace(',','.').replace('_','.'))
def location_from_results_csv_name(p):
    s=Path(p).stem.upper()
    if 'TOE' in s: return 'TOE'
    if 'ROOT' in s: return 'ROOT'
    raise ValueError('Cannot infer location from {}'.format(p))

def det_lambda_modeI(lam,a):
    phi=np.pi-a
    return (1+lam)*np.cos((1-lam)*phi)*np.sin((1+lam)*phi)-np.cos((1+lam)*phi)*(1-lam)*np.sin((1-lam)*phi)
def first_mode_lambda_modeI(a,grid=20000):
    xs=np.linspace(1e-6,1.999,int(grid)); ys=det_lambda_modeI(xs,a); idx=np.where(np.diff(np.sign(ys))!=0)[0]
    if idx.size==0: raise ValueError('No Mode I eigenvalue')
    i=int(idx[0]); return float(brentq(lambda t: det_lambda_modeI(t,a),xs[i],xs[i+1],maxiter=400))
def first_mode_lambda_modeIII(a): return float(np.pi/(2*(np.pi-a)))
_LAM={}
def get_lambda(mode,alpha):
    k=(mode.upper(),round(float(alpha),6))
    if k in _LAM: return _LAM[k]
    ar=np.deg2rad(float(alpha)); lam=first_mode_lambda_modeI(ar) if mode.upper() in ('MI','MODEI','I') else first_mode_lambda_modeIII(ar)
    _LAM[k]=lam; return lam

def build_eps_tensor_from_EPEL(exx,eyy,ezz,exy,eyz,exz):
    return np.array([[exx,0.5*exy,0.5*exz],[0.5*exy,eyy,0.5*eyz],[0.5*exz,0.5*eyz,ezz]],float)
def build_sig_tensor(sxx,syy,szz,sxy,syz,sxz):
    return np.array([[sxx,sxy,sxz],[sxy,syy,syz],[sxz,syz,szz]],float)
def von_mises(sig):
    s=np.asarray(sig,float); dev=s-np.eye(3)*np.trace(s)/3.0; return math.sqrt(1.5*np.sum(dev*dev))
def read_results_csv(path):
    rows=[]
    with Path(path).open('r',encoding='utf-8',newline='') as f:
        for row in csv.reader(f):
            try: float(row[0]); float(row[1])
            except Exception: continue
            if len(row)<20: continue
            node=int(float(row[0])); exx,eyy,ezz,exy,eyz,exz=map(float,row[2:8]); sxx,syy,szz,sxy,syz,sxz=map(float,row[14:20])
            rows.append((node,build_eps_tensor_from_EPEL(exx,eyy,ezz,exy,eyz,exz),build_sig_tensor(sxx,syy,szz,sxy,syz,sxz)))
    if not rows: raise RuntimeError('No numeric rows parsed in {}'.format(path))
    return rows
def pick_max_vm_node(rows): return rows[int(np.argmax([von_mises(sig) for _,_,sig in rows]))]

def _basis_from_n(n):
    n=np.asarray(n,float); n/=np.linalg.norm(n); up=np.array([0.,0.,1.]) if abs(n[2])<0.9 else np.array([1.,0.,0.])
    t1=np.cross(n,up); t1/=np.linalg.norm(t1); t2=np.cross(n,t1); return n,t1,t2

def best_plane_gam_and_szz(E_unit,S_unit,n_sphere):
    phi=(1+5**0.5)/2; best=-1e99; bg=0.; bs=0.
    for k in range(int(n_sphere)):
        z=1-2*(k+0.5)/int(n_sphere); rr=math.sqrt(max(1-z*z,0)); th=2*math.pi*k/phi
        n=np.array([rr*math.cos(th),rr*math.sin(th),z]); n,t1,t2=_basis_from_n(n); Q=np.stack([t1,t2,n],axis=1)
        Er=Q.T@np.asarray(E_unit,float)@Q; Sr=Q.T@np.asarray(S_unit,float)@Q
        gam=math.sqrt(float(Er[0,2])**2+float(Er[1,2])**2)
        if gam>best: best=gam; bg=gam; bs=float(Sr[2,2])
    return float(bg),float(bs)

KNOWN_CONDITIONS=('AW','SRE','SR','PWHT')
def parse_loading_condition(s):
    for c in KNOWN_CONDITIONS:
        if s.upper().endswith(c): return (s[:-len(c)] if s[:-len(c)] else s), c
    return s,'NA'
def parse_specimen_file(path):
    p=Path(path); m=re.search(r'_R_([-\d\.]+)$',p.stem)
    if not m: raise RuntimeError('Cannot parse load ratio from {}'.format(p.name))
    R=float(m.group(1)); loading,cond=parse_loading_condition(p.stem[:m.start()]); data=[]
    for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
        parts=line.split()
        if len(parts)<2: continue
        try: Nf=float(parts[0]); Mmax=float(parts[2]) if len(parts)>=3 else float(parts[-1])
        except Exception: continue
        if Nf>0 and Mmax!=0: data.append((Nf,Mmax))
    if not data: raise RuntimeError('No numeric rows found in {}'.format(path))
    return R,loading,cond,data


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def shifted_master_A(fit, pred_sigma):
    """Return A shifted by pred_sigma standard deviations in log-space."""
    return float(fit["A"]) * math.exp(float(pred_sigma) * float(fit.get("std_ln", 0.0)))


def probability_from_sigma(pred_sigma):
    return 100.0 * normal_cdf(float(pred_sigma))

def cycles_from_ecp(q,A,b):
    if A<=0 or b==0 or q<=0: return float('nan')
    try: return float((q/A)**(1/b))
    except Exception: return float('nan')

def parse_fit_macros(tex_path,prefix):
    text=Path(tex_path).read_text(encoding='utf-8',errors='ignore'); vals={}
    for name,val in re.findall(r'\\def\\([A-Za-z]+)\{([^{}]+)\}',text):
        try: vals[name]=float(val)
        except Exception: pass
    def pick(suf):
        key=prefix+suf
        if key not in vals: raise ValueError('Missing macro \\{} in {}'.format(key,tex_path))
        return vals[key]
    return {'rc_mm':pick('Rcmm'), 'A':pick('A'), 'b':pick('b'), 'std_ln':pick('StdLn'), 'nsigma':vals.get(prefix+'Nsigma',1.2815)}

def find_file(root,calibdir,name):
    for base in [Path(calibdir),Path(root)]:
        for p in base.rglob(name): return p
    return None

def find_author_dirs(root,outdir_name,calibdir_name):
    out=[]
    for d in Path(root).iterdir():
        if not d.is_dir(): continue
        if d.name.startswith('.') or d.name in (outdir_name,calibdir_name): continue
        if any((d/x).is_dir() for x in ('Modo_I','Modo_III','Mode_I_pstress','Mode_I_pstrain')): out.append(d)
    return sorted(out)
def iter_mode_dirs(adir,default_plane):
    for folder,plane in (('Mode_I_pstress','stress'),('Mode_I_pstrain','strain')):
        d=Path(adir)/folder
        if d.is_dir(): yield 'MI',d,plane,folder
    d=Path(adir)/'Modo_I'
    if d.is_dir(): yield 'MI',d,default_plane,'Modo_I'
    d=Path(adir)/'Modo_III'
    if d.is_dir(): yield 'MIII',d,None,'Modo_III'

def scan_mode_folder(mode_dir,n_sphere):
    data_dir=Path(mode_dir)/'Data'
    if not data_dir.is_dir(): return {},[]
    res={}
    for csvp in sorted(data_dir.glob('RESULTS_*.csv')):
        loc=location_from_results_csv_name(csvp); a=alpha_from_results_csv_name(csvp); rows=read_results_csv(csvp); node,E,S=pick_max_vm_node(rows)
        gam,szz=best_plane_gam_and_szz(E,S,n_sphere)
        res[loc]={'alpha_deg':a,'node':node,'gam_unit_best':gam,'szz_unit_best':szz}
    return res, sorted(data_dir.glob('*_R_*.txt'))

def compute_FS_for_point(rc_m, mode, planeI, alpha_deg, etaI, etaIII, rho_geo_m, Sy_MPa, kFS, gam_unit_best, szz_unit_best, R, Mmax_nmm):
    eta=interp_clamped(alpha_deg,etaI['alpha'],etaI[planeI]) if mode.upper()=='MI' else interp_clamped(alpha_deg,etaIII['alpha'],etaIII['eta'])
    rho_eq=eta*float(rc_m)
    if rho_eq<=0 or rho_geo_m<=0: return float('nan')
    lam=get_lambda(mode,alpha_deg); s_geom=(rho_eq/rho_geo_m)**(lam-1.0)
    g0=float(Mmax_nmm)/UNIT_LOAD_NMM; g1=float(R)*g0
    dgam=abs(g0-g1)*float(gam_unit_best)
    sigmax=max(g0*float(szz_unit_best),g1*float(szz_unit_best),0.0)
    return float(s_geom*dgam*(1.0+float(kFS)*(s_geom*sigmax/float(Sy_MPa))))

def write_csv(path,rows,header):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([r.get(h,'') for h in header])


def prediction_error_summary(rows, n_col="Nf", npred_col="Npred"):
    """Return average and sample standard deviation of ln(Npred/Nexp)."""
    vals = []
    for r in rows:
        try:
            n_exp = float(r.get(n_col, float("nan")))
            n_pred = float(r.get(npred_col, float("nan")))
        except Exception:
            continue
        if n_exp > 0.0 and n_pred > 0.0 and np.isfinite(n_exp) and np.isfinite(n_pred):
            vals.append(math.log(n_pred / n_exp))
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return {
            "n": 0,
            "average_ln_Npred_over_Nexp": float("nan"),
            "std_ln_Npred_over_Nexp": float("nan"),
        }
    return {
        "n": int(vals.size),
        "average_ln_Npred_over_Nexp": float(np.mean(vals)),
        "std_ln_Npred_over_Nexp": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
    }


def write_error_summary(outdir, tag, summary):
    outdir = Path(outdir)
    txt_path = outdir / ("prediction_error_summary_{}.txt".format(tag.lower()))
    csv_path = outdir / ("prediction_error_summary_{}.csv".format(tag.lower()))

    lines = [
        "error_metric = ln(Npred/Nexp)",
        "n = {}".format(summary["n"]),
        "average_ln_Npred_over_Nexp = {:.12e}".format(summary["average_ln_Npred_over_Nexp"]),
        "std_ln_Npred_over_Nexp = {:.12e}".format(summary["std_ln_Npred_over_Nexp"]),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error_metric", "n", "average_ln_Npred_over_Nexp", "std_ln_Npred_over_Nexp"])
        w.writerow([
            "ln(Npred/Nexp)",
            summary["n"],
            summary["average_ln_Npred_over_Nexp"],
            summary["std_ln_Npred_over_Nexp"],
        ])

    return txt_path, csv_path

def plot_prediction(N,Q,fit,out_png):
    N=np.asarray(N,float); Q=np.asarray(Q,float); m=np.isfinite(N)&np.isfinite(Q)&(N>0)&(Q>0)
    if np.sum(m)==0: return
    A,b,std,ns=fit['A'],fit['b'],fit['std_ln'],fit['nsigma']; x=np.logspace(np.log10(max(np.min(N[m])*0.7,1)),np.log10(np.max(N[m])*1.3),300)
    y=A*x**b; yp=np.exp(np.log(A)+ns*std)*x**b; ym=np.exp(np.log(A)-ns*std)*x**b
    plt.figure(figsize=(7.5,5.5)); plt.loglog(N[m],Q[m],'o',label='Prediction data'); plt.loglog(x,y,'-',lw=2,label='Calibration master curve'); plt.loglog(x,yp,'--',label='+{:.3g}σ'.format(ns)); plt.loglog(x,ym,'--',label='-{:.3g}σ'.format(ns)); plt.xlabel('Cycles to failure $N_f$'); plt.ylabel('Fatemi--Socie parameter $FS$'); plt.grid(True,which='both',ls=':',alpha=0.5); plt.legend(); plt.tight_layout(); plt.savefig(str(out_png),dpi=300); plt.show()

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--calibdir',default='export_txt'); ap.add_argument('--outdir',default='prevision_txt'); ap.add_argument('--planeI',default='strain',choices=['stress','strain']); ap.add_argument('--kFS',type=float,default=DEFAULT_KFS); ap.add_argument('--defaultSy',type=float,default=DEFAULT_SY_MPA); ap.add_argument('--defaultRhoGeo',type=float,default=DEFAULT_RHOGEO_MM); ap.add_argument('--sphere',type=int,default=DEFAULT_SPHERE); ap.add_argument('--pred_sigma',type=float,default=-2.0,help='Standard-deviation offset for N prediction: -2=about 2.3%, 0=P50, 2=about 97.7%.')
    args,_=ap.parse_known_args(argv); root=Path(args.root); calib=root/args.calibdir; outdir=root/args.outdir; outdir.mkdir(parents=True,exist_ok=True)
    fit=parse_fit_macros(calib/'powerlaw_fit_fs_all.tex','FSFit'); A_pred=shifted_master_A(fit,args.pred_sigma); P_pred=probability_from_sigma(args.pred_sigma); rc_m=fit['rc_mm']*1e-3
    mi=find_file(root,calib,'slopes_vs_alpha_fs_modeI.dat'); miii=find_file(root,calib,'slopes_vs_alpha_fs_modeIII.dat')
    if mi is None or miii is None: raise SystemExit('Missing FS slopes files. Copy them under the prevision folder or export_txt.')
    etaI=build_eta_modeI(mi); etaIII=build_eta_modeIII(miii)
    rows=[]; series={}
    for adir in find_author_dirs(root,outdir.name,calib.name):
        author,rho_geo_mm,Sy=parse_author_folder(adir.name,args.defaultRhoGeo,args.defaultSy); rho_geo_m=rho_geo_mm*1e-3
        for mode,mdir,plane,source in iter_mode_dirs(adir,args.planeI):
            res,specs=scan_mode_folder(mdir,args.sphere)
            for spec in specs:
                R,loading,cond,data=parse_specimen_file(spec)
                for loc,li in res.items():
                    for Nf,Mmax in data:
                        Q=compute_FS_for_point(rc_m,mode,plane or args.planeI,li['alpha_deg'],etaI,etaIII,rho_geo_m,Sy,args.kFS,li['gam_unit_best'],li['szz_unit_best'],R,Mmax)
                        Npred=cycles_from_ecp(Q,A_pred,fit['b'])
                        row={'Mode':mode,'Author':author,'Folder':source,'PlaneI':plane or 'NA','Loading':loading,'Condition':cond,'R':R,'Location':loc,'alpha_deg':li['alpha_deg'],'Node':li['node'],'rho_geo_mm':rho_geo_mm,'Sy_MPa':Sy,'Nf':Nf,'Mmax':Mmax,'FS':Q,'Npred':Npred,'Npred_over_Nf':Npred/Nf if Nf>0 else np.nan,'error_ln_Npred_over_Nexp':math.log(Npred/Nf) if Npred>0 and Nf>0 else np.nan,'pred_sigma':args.pred_sigma,'pred_probability_percent':P_pred,'log_residual':math.log(Q)-math.log(A_pred*Nf**fit['b']) if Q>0 and Nf>0 else np.nan}
                        rows.append(row); series.setdefault((mode,author,loading,cond,R,loc),[]).append((Nf,Q,Npred))
    if not rows: raise SystemExit('No prediction points found.')
    write_csv(outdir/'prevision_points_fs.csv',rows,list(rows[0].keys()))
    err_summary=prediction_error_summary(rows)
    err_txt,err_csv=write_error_summary(outdir,'fs',err_summary)
    data_root=outdir/'Data'; data_root.mkdir(exist_ok=True)
    for (mode,author,loading,cond,R,loc),pts in series.items():
        d=data_root/('MI' if mode=='MI' else 'MIII')/author/'Results'; d.mkdir(parents=True,exist_ok=True); Rtxt=('{:g}'.format(float(R))).replace('+',''); p=d/'{}_{}_R_{}_FS_{}.txt'.format(loading,cond,Rtxt,str(loc).upper()); np.savetxt(p,np.asarray(sorted(pts),float),fmt='%.10e',header='Nf\tFS\tNpred',comments='')
    plot_prediction([r['Nf'] for r in rows],[r['FS'] for r in rows],fit,outdir/'prevision_plot_loglog_fs.png')
    print('Loaded calibration: rc = {:.6g} mm, A = {:.6e}, b = {:.6e}'.format(fit['rc_mm'],fit['A'],fit['b'])); print('Prediction curve sigma = {:.6g} (P = {:.3g}%), A_pred = {:.6e}'.format(args.pred_sigma,P_pred,A_pred)); print('AVG(eN) = {:.6e}'.format(err_summary['average_ln_Npred_over_Nexp'])); print('STD(eN) = {:.6e}'.format(err_summary['std_ln_Npred_over_Nexp'])); print('n(eN) = {}'.format(err_summary['n'])); print('Wrote',outdir)
if __name__=='__main__': main()