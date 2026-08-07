import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from fgm.latent_kalman import kalman_filter, rts_smoother, independent_placement, build_extended, MEAS
from fgm.wp2_canonical import assemble_canonical, FactorAnalysisMissing
NPC=5; PCS=[f"PC{i+1}" for i in range(NPC)]
z=np.load("/Users/tiago/dev/fgr-geometry/results/img_align/emb_usfm_multilayer.npz",allow_pickle=True)
imp=(pd.Series(z["dataset_type"]).astype(str).values=="impact")
E=z["emb_l5"].astype(np.float32)[imp]
fid=pd.to_numeric(pd.Series(z["fetus_id"]).astype(str),errors="coerce").values[imp]
ga=pd.to_numeric(pd.Series(z["ga_weeks_recovered"]).astype(str),errors="coerce").values[imp]
sdte=pd.Series(z["study_date"]).astype(str).values[imp]
ok=np.isfinite(fid)&np.isfinite(ga)&(ga>10)&(ga<45)
P=PCA(NPC,random_state=0).fit(E[ok]); S=P.transform(E[ok])
V=pd.DataFrame(S,columns=PCS); V["fid"]=fid[ok].astype(int); V["ga"]=ga[ok]; V["sdate"]=sdte[ok]
G=V.groupby(["fid","sdate"]).agg({**{c:"mean" for c in PCS},"ga":"mean"}); G["k"]=V.groupby(["fid","sdate"]).size(); G=G.reset_index()
for c in PCS: G[c]=(G[c]-G[c].mean())/G[c].std()
vc=G.groupby("fid").size()
print("=== LONGITUDINAL CHECK ===")
print(f"  image-visits={len(G)}  fetuses={G.fid.nunique()}  visits/fetus counts={dict(vc.value_counts().head(4))}")
print(f"  {100*(vc==1).mean():.1f}% of fetuses have exactly ONE image-visit -> no image trajectory\n")
X,cols,blocks,fids,*_=assemble_canonical(); cols=list(cols)
BMI=dict(zip([int(f) for f in fids],X[:,cols.index("maternal_bmi")]))
G["bmi"]=[BMI.get(int(f),np.nan) for f in G.fid]
print(f"BMI coverage on image-visits: {np.isfinite(G.bmi).mean():.2f}")
Gr=G.copy(); m=np.isfinite(Gr.bmi).values
for c in PCS:
    sl=np.polyfit(Gr.bmi.values[m],Gr[c].values[m],1)
    Gr[c]=Gr[c].astype(float)
    Gr.loc[m,c]=Gr[c].values[m]-np.polyval(sl,Gr.bmi.values[m]); Gr[c]=(Gr[c]-Gr[c].mean())/Gr[c].std()
lo,_=build_extended("data/visits_long_z.csv","data/IMPACT_merged_by_Cod.xlsx",0)
fa=FactorAnalysisMissing(n_factors=1,seed=0).fit(lo[MEAS].to_numpy(float))
b=fa.W_[:,0].copy(); Psi=fa.psi_.copy(); mu_b=fa.mu_.copy()
if b[MEAS.index("ac_z_ig21")]<0: b=-b
base=[]
for f,g in lo.groupby("_fid"):
    g=g.sort_values("ga_weeks"); gas=g["ga_weeks"].to_numpy(float)
    Xm=g[MEAS].to_numpy(float)-mu_b; o=~np.isnan(gas)
    base.append((int(f),gas[o],[Xm[i] for i in range(len(gas)) if o[i]],[~np.isnan(Xm[i]) for i in range(len(gas)) if o[i]]))
def nll(lq):
    q=np.exp(lq); t=0.0
    for f,gas,xs,ms in base:
        if len(gas): t+=kalman_filter(gas,xs,ms,b,Psi,q)[4]
    return -t
q0=float(np.exp(minimize_scalar(nll,bounds=(np.log(1e-4),np.log(5.0)),method="bounded").x))
szmap={}
for f,gas,xs,ms in base:
    mi,_=independent_placement(xs,ms,b,Psi)
    if len(mi): szmap[f]=float(np.mean(mi))
print("\n=== PC LOADINGS ON THE SIZE STATE ===")
load={}
for tag,DF in (("raw",G),("BMI-residualised",Gr)):
    fm=DF.groupby("fid")[PCS].mean(); com=[f for f in fm.index if f in szmap]
    sv=np.array([szmap[f] for f in com])
    load[tag]={c:float(pearsonr(fm.loc[com,c].values,sv)[0]) for c in PCS}
    print(f"  {tag:18s} "+"  ".join(f"{c}={load[tag][c]:+.3f}" for c in PCS)+f"  (n={len(com)})")
print("  reference: the maturation lag loaded +0.117 on this same state")
print("\n=== KALMAN: PCs as measurement channels ===")
res={}
for tag,DF,key in (("biometry only",None,None),("+ raw PCs",G,"raw"),("+ BMI-resid PCs",Gr,"BMI-residualised")):
    if DF is not None:
        fm=DF.groupby("fid")[PCS].mean()
        bl=np.array([load[key][c] for c in PCS]); Pl=np.array([max(1-l**2,1e-3) for l in bl])
    sds=[]
    for f,gas,xs,ms in base:
        if len(gas)==0: continue
        if DF is None: bb,PP,X2,M2=b,Psi,xs,ms
        else:
            bb=np.append(b,bl); PP=np.append(Psi,Pl)
            v=fm.loc[f,PCS].values.astype(float) if f in fm.index else np.full(NPC,np.nan)
            X2=[np.append(x,v) for x in xs]; M2=[np.append(mm,np.isfinite(v)) for mm in ms]
        mf,Pf,_,_,_=kalman_filter(gas,X2,M2,bb,PP,q0)
        _,Ps=rts_smoother(gas,mf,Pf,q0); sds.append(np.sqrt(Ps).mean())
    res[tag]=float(np.mean(sds)); print(f"  {tag:20s} mean posterior SD = {np.mean(sds):.4f} (n={len(sds)})")
print(f"\n  narrowing raw PCs       : {100*(1-res['+ raw PCs']/res['biometry only']):.2f}%")
print(f"  narrowing BMI-resid PCs : {100*(1-res['+ BMI-resid PCs']/res['biometry only']):.2f}%")
print(f"  reference: lag channel gave 0.03%")
json.dump(dict(load=load,sd=res,n_visits=int(len(G)),n_fet=int(G.fid.nunique()),
  pct_single=float((vc==1).mean()),evr=[float(x) for x in P.explained_variance_ratio_],
  q=q0),open("/tmp/kal_pc.json","w"),indent=1)
