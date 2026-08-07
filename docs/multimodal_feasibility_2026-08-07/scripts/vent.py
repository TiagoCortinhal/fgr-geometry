"""Split-sample ventricular test per prespec_ventricular.md.
Select on half A, test on half B, both representations."""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/tools/impact_fetal_panel")
from fgm_tools import *
fgm_setup(); P=fgm_panel(); Z,cols,bl,fids=P["Z"],P["cols"],np.array(P["blocks"]),P["fids"]
GA=fgm_ga_at_echo(fids); BMI=Z[:,cols.index("maternal_bmi")]
RAD,_=fgm_radiomics(fids,n_pc=12,artifact_path_fn=lambda v:"/tmp/_rad.parquet")
USFM,_=fgm_image_pcs(fids,n_pc=12)
CARD=[cols[i] for i in range(len(cols)) if bl[i]=="cardiac"]
Y=Z[:,[cols.index(c) for c in CARD]]
COV=fgm_nuisance_design(GA,BMI,np.zeros((len(Z),0)))
out={}
for nm,IM in (("radiomics",RAD),("USFM",USFM)):
    r=fgm_split_sample_screen(Y,CARD,IM,COV,n_top=2,seed=0,nperm=1000)
    out[nm]=r
    print(f"\n=== {nm} ===")
    print("  HALF-A ranking (top 5):")
    for x in r["ranking_A"][:5]: print(f"     {x['var']:28s} r_A={x['r']:+.3f} n={x['n']}")
    print("  HALF-B test of the A-selected:")
    for x in r["tested_B"]:
        print(f"     {x['var']:28s} r_A={x['r_A']:+.3f} -> r_B={x['r_B']:+.3f} (n={x['n_B']}) "
              f"null p95={x['null_p95']:+.3f} p={x['p']:.3f}")
    print("  full cohort (confirmatory only, NOT independent):")
    for x in r["full_cohort"]: print(f"     {x['var']:28s} r={x['r']:+.3f} n={x['n']}")
# H2 specificity: basal vs longitudinal on the full cohort
print("\n=== H2 SPECIFICITY: basal vs longitudinal (full cohort, GA+BMI adj) ===")
pairs=["Percentil_LV_basal","Percentil_RV_basal","Percentil_LV_longitudinal","Percentil_RV_longitudinal"]
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
def oof(y,X,C,seed=0):
    m=np.isfinite(y)&np.isfinite(X).all(1)&np.isfinite(C).all(1)
    A=C[m]; yy=y[m]-A@np.linalg.lstsq(A,y[m],rcond=None)[0]
    Xs=X[m]-A@np.linalg.lstsq(A,X[m],rcond=None)[0]
    p=np.zeros_like(yy)
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(Xs):
        pc=PCA(min(8,Xs.shape[1],len(tr)-1),random_state=0).fit(Xs[tr])
        p[te]=RidgeCV(alphas=np.logspace(-2,3,20)).fit(pc.transform(Xs[tr]),yy[tr]).predict(pc.transform(Xs[te]))
    return float(np.corrcoef(p,yy)[0,1]),int(m.sum())
spec={}
print(f"  {'variable':30s} {'radiomics':>10s} {'USFM':>8s}")
for v in pairs:
    y=Z[:,cols.index(v)]
    a,_=oof(y,RAD,COV); b,n_=oof(y,USFM,COV)
    print(f"  {v:30s} {a:+10.3f} {b:+8.3f}  n={n_}")
    spec[v]=dict(radiomics=float(a),usfm=float(b),n=int(n_))
out["H2_specificity"]=spec
json.dump(out,open("/tmp/vent.json","w"),indent=1)
