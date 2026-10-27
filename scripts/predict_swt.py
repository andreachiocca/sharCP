#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prevision_SWT.py
-----------------
Prediction script for SWT ECP master curve.

Expected folder layout when run from prevision_SWT/:
  export_txt/                       # copied from SWT calibration output
    powerlaw_fit_swt_all.tex
    tecp_vs_rc_swt.dat              # used to recover best rc if no best_rc.txt exists
  slopes_vs_alpha_swt_modeI.dat     # required, can be anywhere under --root or --calibdir
  slopes_vs_alpha_swt_modeIII.dat   # required, can be anywhere under --root or --calibdir
  Author__rho(mm)__Sy(MPa)/Modo_I/Data/... and/or Modo_III/Data/...

Also supports Mode_I_pstress and Mode_I_pstrain folders. These override --plane for Mode I.

Run:
  python prevision_SWT.py --root . --calibdir export_txt --outdir prevision_txt
"""
from __future__ import print_function
import argparse, csv, math, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

DEFAULT_MODEI_SLOPES='slopes_vs_alpha_swt_modeI.dat'
DEFAULT_MODEIII_SLOPES='slopes_vs_alpha_swt_modeIII.dat'
DEFAULT_N_SPHERE=6000
DEFAULT_TIE_TOL=1e-3
UNIT_LOAD_NMM=1000.0
PARAM_TAG='SWT'


def _try_float(tok):
    try: return float(str(tok).replace(',','.'))
    except Exception: return None

def _parse_float_token(s): return float(str(s).strip().replace(',','.').replace('_','.'))
def normal_cdf(z): return 0.5*(1.0+math.erf(z/math.sqrt(2.0)))

def shifted_master_A(fit, pred_sigma):
    """Return A shifted by pred_sigma standard deviations in log-space."""
    return float(fit["A"]) * math.exp(float(pred_sigma) * float(fit.get("std_ln", 0.0)))


def probability_from_sigma(pred_sigma):
    return 100.0 * normal_cdf(float(pred_sigma))

def _load_txt_table(path):
    rows=[]
    for ln in Path(path).read_text(encoding='utf-8',errors='ignore').splitlines():
        ln=ln.strip()
        if not ln or ln.startswith('#'): continue
        try: vals=[float(t.replace(',','.')) for t in ln.replace(';',' ').split()]
        except Exception: continue
        if vals: rows.append(vals)
    if not rows: raise ValueError('No numeric data parsed from {}'.format(path))
    ncol=max(len(r) for r in rows); arr=np.full((len(rows),ncol),np.nan,float)
    for i,r in enumerate(rows): arr[i,:len(r)]=r
    return arr

def build_eta_modeI(path):
    arr=_load_txt_table(path)
    if arr.shape[1]<3: raise ValueError('{} must have 3 columns'.format(path))
    idx=np.argsort(arr[:,0]); return {'stress':(arr[idx,0],arr[idx,1]),'strain':(arr[idx,0],arr[idx,2])}
def build_eta_modeIII(path):
    arr=_load_txt_table(path)
    if arr.shape[1]<2: raise ValueError('{} must have 2 columns'.format(path))
    idx=np.argsort(arr[:,0]); return arr[idx,0],arr[idx,1]
def interp_eta(alpha, table):
    x,y=table; return float(np.interp(float(alpha), x, y))

def det_lambda_modeI(lam,alpha_rad):
    # same characteristic equation used in the SWT calibration script
    return math.sin(2.0*(math.pi-alpha_rad)*lam)+lam*math.sin(2.0*(math.pi-alpha_rad))
def first_mode_lambda_modeI(alpha_rad):
    a=1e-6; b=1.0; fa=det_lambda_modeI(a,alpha_rad); fb=det_lambda_modeI(b,alpha_rad)
    if fa*fb>0:
        grid=np.linspace(1e-6,10.0,5000); vals=np.array([det_lambda_modeI(t,alpha_rad) for t in grid]); idx=np.where(np.diff(np.sign(vals))!=0)[0]
        if idx.size==0: raise ValueError('Cannot bracket Mode I lambda')
        a=float(grid[int(idx[0])]); b=float(grid[int(idx[0])+1])
    return float(brentq(lambda t: det_lambda_modeI(t,alpha_rad),a,b,maxiter=400))
def lambda_modeIII(alpha_rad): return float(math.pi/(2.0*(math.pi-alpha_rad)))

def build_eps_tensor_from_EPEL(exx,eyy,ezz,exy,eyz,exz):
    return np.array([[exx,0.5*exy,0.5*exz],[0.5*exy,eyy,0.5*eyz],[0.5*exz,0.5*eyz,ezz]],float)
def build_sig_tensor(sxx,syy,szz,sxy,syz,sxz):
    return np.array([[sxx,sxy,sxz],[sxy,syy,syz],[sxz,syz,szz]],float)
def von_mises(sig):
    s=np.asarray(sig,float); dev=s-np.eye(3)*np.trace(s)/3.0; return math.sqrt(1.5*float(np.sum(dev*dev)))
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

def sphere_normals(n_sphere):
    phi=(1+5**0.5)/2; out=np.zeros((int(n_sphere),3),float)
    for k in range(int(n_sphere)):
        z=1-2*(k+0.5)/int(n_sphere); r=math.sqrt(max(1-z*z,0)); th=2*math.pi*k/phi
        out[k]=[r*math.cos(th),r*math.sin(th),z]
    return out
def precompute_eps_sig_along_normals(E,S,normals):
    return (np.einsum('ni,ij,nj->n',normals,E,normals).astype(float), np.einsum('ni,ij,nj->n',normals,S,normals).astype(float))
def best_swt_unit_for_R(eps_n,sig_n,R,tie_tol=DEFAULT_TIE_TOL):
    R=float(R); eps_abs=np.abs(eps_n); mmax=float(np.max(eps_abs))
    if not (mmax>0): return 0.0
    cand=eps_abs>=mmax*(1.0-float(tie_tol))
    eps_a=0.5*abs(1.0-R)*eps_abs[cand]
    sig0=sig_n[cand]; sig1=R*sig0; sigmax=np.maximum.reduce([sig0,sig1,np.zeros_like(sig0)])
    swt=eps_a*sigmax
    return float(np.max(swt)) if swt.size else 0.0

def parse_author_folder(folder_name):
    base=folder_name; rho=None; sy=None; toks=str(folder_name).split('__')
    if len(toks)>=3:
        base=toks[0]
        try: rho=_parse_float_token(toks[1])
        except Exception: pass
        try: sy=_parse_float_token(toks[2])
        except Exception: pass
    else:
        m=re.match(r'^(.*)_(\d+_\d+)$',str(folder_name))
        if m:
            base=m.group(1)
            try: rho=_parse_float_token(m.group(2))
            except Exception: pass
    return base,rho,sy

def parse_results_alpha(name):
    stem=Path(name).stem; m=re.match(r'^RESULTS_(TOE|ROOT)_(.+)$',stem,re.I)
    if not m:
        m2=re.match(r'^RESULTS_(TOE|ROOT)$',stem,re.I)
        if m2: return m2.group(1).upper(),0.0
        raise ValueError('Cannot parse RESULTS alpha from {}'.format(name))
    return m.group(1).upper(), _parse_float_token(m.group(2))
def parse_specimen_filename(fname):
    stem=Path(fname).stem
    if '_R_' not in stem: raise ValueError('Missing _R_ in {}'.format(fname))
    pre,rpart=stem.split('_R_',1); R=_parse_float_token(rpart); m=re.match(r'^(.*?)(AW|SRE|SR|PWHT)$',pre,re.I)
    return (m.group(1),m.group(2).upper(),float(R)) if m else (pre,'NA',float(R))
def read_specimen_rows(path):
    out=[]
    for line in Path(path).read_text(encoding='utf-8',errors='ignore').splitlines():
        p=line.split()
        if len(p)<2: continue
        try: Nf=float(p[0]); Mmax=float(p[2]) if len(p)>=3 else float(p[-1])
        except Exception: continue
        if Nf>0 and Mmax!=0: out.append((Nf,Mmax))
    return out

def parse_fit_macros(tex_path):
    text=Path(tex_path).read_text(encoding='utf-8',errors='ignore'); vals={}
    for name,val in re.findall(r'\\def\\([A-Za-z]+)\{([^{}]+)\}',text):
        try: vals[name]=float(val)
        except Exception: pass
    # SWT calibration currently uses FSFit* macro names for historical compatibility.
    prefix='SWTFit' if 'SWTFitA' in vals else 'FSFit'
    def pick(suf):
        key=prefix+suf
        if key not in vals: raise ValueError('Missing macro \\{} in {}'.format(key,tex_path))
        return vals[key]
    fit={'A':pick('A'),'b':pick('b'),'std_ln':pick('StdLn'),'nsigma':vals.get(prefix+'Nsigma',1.2815)}
    m=re.search(r'Best rc\s*=\s*([0-9eE+\-.]+)\s*mm',text)
    if m: fit['rc_mm']=float(m.group(1))
    return fit

def recover_best_rc_mm(calibdir,fit):
    if 'rc_mm' in fit: return float(fit['rc_mm'])
    br=Path(calibdir)/'best_rc.txt'
    if br.exists():
        m=re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',br.read_text())
        if m: return float(m.group(1))
    sweep=Path(calibdir)/'tecp_vs_rc_swt.dat'
    if sweep.exists():
        arr=_load_txt_table(sweep)
        # columns: rc_mm T_ECP A b std_ln nData
        j=int(np.nanargmin(arr[:,1])); return float(arr[j,0])
    raise ValueError('Cannot recover best rc. Need Best rc comment, best_rc.txt, or tecp_vs_rc_swt.dat.')

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

def scan_mode_folder(mode_dir,mode_tag,etaI,etaIII,plane_choice,normals,tie_tol):
    data_dir=Path(mode_dir)/'Data'
    if not data_dir.exists(): return []
    results=sorted(data_dir.glob('RESULTS_*.csv')); specs=sorted([p for p in data_dir.glob('*.txt') if 'RESULTS' not in p.name.upper()])
    if not results or not specs: return []
    spec_meta=[]
    for sf in specs:
        try: loading,cond,R=parse_specimen_filename(sf.name)
        except Exception: continue
        rows=read_specimen_rows(sf)
        if rows: spec_meta.append((sf,loading,cond,R,rows))
    if not spec_meta: return []
    loc_unit={}
    for rcsv in results:
        try: loc,alpha=parse_results_alpha(rcsv.name)
        except Exception: continue
        rows=read_results_csv(rcsv); node,E,S=pick_max_vm_node(rows); eps_n,sig_n=precompute_eps_sig_along_normals(E,S,normals)
        loc_unit[loc]={'alpha_deg':alpha,'node':node,'eps_n':eps_n,'sig_n':sig_n}
    points=[]; lam_cache={}; swt_cache={}
    for sf,loading,cond,R,rows in spec_meta:
        for loc,d in loc_unit.items():
            key=(loc,float(R))
            if key not in swt_cache: swt_cache[key]=best_swt_unit_for_R(d['eps_n'],d['sig_n'],R,tie_tol)
            swt_unit=swt_cache[key]
            if not (swt_unit>0): continue
            alpha=d['alpha_deg']; ar=math.radians(alpha)
            if mode_tag=='MI': eta=interp_eta(alpha,etaI[plane_choice]); lam=lam_cache.setdefault(alpha, first_mode_lambda_modeI(ar))
            else: eta=interp_eta(alpha,etaIII); lam=lam_cache.setdefault(alpha, lambda_modeIII(ar))
            pexp=2.0*(lam-1.0)
            for Nf,Mmax in rows:
                load_scale=float(Mmax)/UNIT_LOAD_NMM; constC=(load_scale**2)*swt_unit
                points.append({'Mode':mode_tag,'Loading':loading,'Condition':cond,'R':R,'Location':'Toe' if loc=='TOE' else 'Root' if loc=='ROOT' else loc,'alpha_deg':alpha,'PlaneI':plane_choice if mode_tag=='MI' else 'NA','Node':d['node'],'eta':eta,'pexp':pexp,'constC':constC,'Nf':Nf,'Mmax':Mmax,'spec_file':str(sf)})
    return points

def cycles_from_ecp(q,A,b):
    if A<=0 or b==0 or q<=0: return float('nan')
    try: return float((q/A)**(1.0/b))
    except Exception: return float('nan')
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
    plt.figure(figsize=(7.5,5.5)); plt.loglog(N[m],Q[m],'o',label='Prediction data'); plt.loglog(x,y,'-',lw=2,label='Calibration master curve'); plt.loglog(x,yp,'--',label='+{:.3g}σ'.format(ns)); plt.loglog(x,ym,'--',label='-{:.3g}σ'.format(ns)); plt.xlabel('Cycles to failure $N_f$'); plt.ylabel('SWT parameter'); plt.grid(True,which='both',ls=':',alpha=0.5); plt.legend(); plt.tight_layout(); plt.savefig(str(out_png),dpi=300); plt.show()

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--calibdir',default='export_txt'); ap.add_argument('--outdir',default='prevision_txt'); ap.add_argument('--plane',default='stress',choices=['stress','strain']); ap.add_argument('--modeI_slopes',default=DEFAULT_MODEI_SLOPES); ap.add_argument('--modeIII_slopes',default=DEFAULT_MODEIII_SLOPES); ap.add_argument('--n_sphere',type=int,default=DEFAULT_N_SPHERE); ap.add_argument('--tie_tol',type=float,default=DEFAULT_TIE_TOL); ap.add_argument('--pred_sigma',type=float,default=-2.0,help='Standard-deviation offset for N prediction: -2=about 2.3%, 0=P50, 2=about 97.7%.')
    args,_=ap.parse_known_args(argv); root=Path(args.root); calib=root/args.calibdir; outdir=root/args.outdir; outdir.mkdir(parents=True,exist_ok=True)
    fit=parse_fit_macros(calib/'powerlaw_fit_swt_all.tex'); fit['rc_mm']=recover_best_rc_mm(calib,fit); A_pred=shifted_master_A(fit,args.pred_sigma); P_pred=probability_from_sigma(args.pred_sigma); best_rc=fit['rc_mm']*1e-3
    mi=find_file(root,calib,args.modeI_slopes); miii=find_file(root,calib,args.modeIII_slopes)
    if mi is None or miii is None: raise SystemExit('Missing SWT slopes files. Copy them under the prevision folder or export_txt.')
    etaI=build_eta_modeI(mi); etaIII=build_eta_modeIII(miii); normals=sphere_normals(args.n_sphere)
    all_points=[]
    for adir in find_author_dirs(root,outdir.name,calib.name):
        author,rho_mm,sy=parse_author_folder(adir.name)
        if rho_mm is None: continue
        rho_m=float(rho_mm)*1e-3
        for mode,mdir,plane,source in iter_mode_dirs(adir,args.plane):
            pts=scan_mode_folder(mdir,mode,etaI,etaIII,plane or args.plane,normals,args.tie_tol)
            for p in pts:
                p['Author']=author; p['Folder']=source; p['rho_fem_mm']=rho_mm; p['rho_fem_m']=rho_m; p['Sy_MPa']=sy if sy is not None else 'NA'; all_points.append(p)
    if not all_points: raise SystemExit('No prediction points found.')
    rows=[]; series={}
    for p in all_points:
        rho_eq=float(p['eta'])*best_rc
        if rho_eq<=0 or p['rho_fem_m']<=0: continue
        Q=float(p['constC'])*((rho_eq/float(p['rho_fem_m']))**float(p['pexp']))
        if not (Q>0 and np.isfinite(Q)): continue
        Npred=cycles_from_ecp(Q,A_pred,fit['b'])
        row={'Mode':p['Mode'],'Author':p['Author'],'Folder':p['Folder'],'PlaneI':p['PlaneI'],'Loading':p['Loading'],'Condition':p['Condition'],'R':p['R'],'Location':p['Location'],'alpha_deg':p['alpha_deg'],'Node':p['Node'],'rho_fem_mm':p['rho_fem_mm'],'Nf':p['Nf'],'Mmax':p['Mmax'],'SWT':Q,'Npred':Npred,'Npred_over_Nf':Npred/p['Nf'] if p['Nf']>0 else np.nan,'error_ln_Npred_over_Nexp':math.log(Npred/p['Nf']) if Npred>0 and p['Nf']>0 else np.nan,'pred_sigma':args.pred_sigma,'pred_probability_percent':P_pred,'log_residual':math.log(Q)-math.log(A_pred*p['Nf']**fit['b']) if Q>0 and p['Nf']>0 else np.nan}
        rows.append(row); series.setdefault((p['Mode'],p['Author'],p['Loading'],p['Condition'],p['R'],p['Location']),[]).append((p['Nf'],Q,Npred))
    if not rows: raise SystemExit('All computed SWT values were invalid.')
    write_csv(outdir/'prevision_points_swt.csv',rows,list(rows[0].keys()))
    err_summary=prediction_error_summary(rows)
    err_txt,err_csv=write_error_summary(outdir,'swt',err_summary)
    data_root=outdir/'Data'; data_root.mkdir(exist_ok=True)
    for (mode,author,loading,cond,R,loc),pts in series.items():
        d=data_root/('MI' if mode=='MI' else 'MIII')/author/'Results'; d.mkdir(parents=True,exist_ok=True); Rtxt=('{:g}'.format(float(R))).replace('+',''); p=d/'{}_{}_R_{}_SWT_{}.txt'.format(loading,cond,Rtxt,str(loc).upper()); np.savetxt(p,np.asarray(sorted(pts),float),fmt='%.10e',header='Nf\tSWT\tNpred',comments='')
    plot_prediction([r['Nf'] for r in rows],[r['SWT'] for r in rows],fit,outdir/'prevision_plot_loglog_swt.png')
    print('Loaded calibration: rc = {:.6g} mm, A = {:.6e}, b = {:.6e}'.format(fit['rc_mm'],fit['A'],fit['b'])); print('Prediction curve sigma = {:.6g} (P = {:.3g}%), A_pred = {:.6e}'.format(args.pred_sigma,P_pred,A_pred)); print('AVG(eN) = {:.6e}'.format(err_summary['average_ln_Npred_over_Nexp'])); print('STD(eN) = {:.6e}'.format(err_summary['std_ln_Npred_over_Nexp'])); print('n(eN) = {}'.format(err_summary['n'])); print('Wrote',outdir)
if __name__=='__main__': main()