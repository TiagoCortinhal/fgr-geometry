"""The maternal block aligns with images at 0.384 fully adjusted. Is it physiology
or acquisition? Maternal BMI degrades ultrasound quality -- that is the obvious confound."""
import numpy as np, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
d=np.load("/tmp/card_sub.npz",allow_pickle=True)
Zp,cols,bl=d["Zpanel"],[str(x) for x in d["cols"]],[str(x) for x in d["bl"]]
IMG,GA,EFW=d["IMG"],d["GA"],d["EFW"]
MAT=[i for i in range(len(cols)) if bl[i]=="maternal"]; MN=[cols[i] for i in MAT]
BIOi=[i for i in range(len(cols)) if bl[i]=="growth"]
def cc(X,Y,npc=10,seed=0):
    o=[]
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(X):
        p=PCA(min(npc,X.shape[1],len(tr)-1),random_state=0).fit(X[tr])
        c=CCA(n_components=1,max_iter=2000).fit(p.transform(X[tr]),Y[tr])
        a,b=c.transform(p.transform(X[te]),Y[te]); o.append(np.corrcoef(a[:,0],b[:,0])[0,1])
    return float(np.mean(o))
def rs(X,cv):
    if not cv: return X
    A=np.column_stack([np.ones(len(X))]+[np.asarray(c).reshape(len(X),-1) for c in cv])
    return X-A@np.linalg.lstsq(A,X,rcond=None)[0]
sel=np.isfinite(IMG).all(1)&np.isfinite(Zp[:,MAT]).all(1)&np.isfinite(GA)
Is=IMG[sel]; Y=Zp[sel][:,MAT]; g=GA[sel][:,None]
e=np.where(np.isfinite(EFW[sel]),EFW[sel],0.0)[:,None]
B=np.where(np.isfinite(Zp[sel][:,BIOi]),Zp[sel][:,BIOi],0.0)
print(f"n={int(sel.sum())}\n")
print("WHICH MATERNAL VARIABLE DRIVES IT? (each alone vs images, fully adjusted)")
for j,nm in enumerate(MN):
    v=cc(rs(Is,[g,e,B]),rs(Y[:,[j]],[g,e,B]))
    print(f"   {nm:20s} {v:+.4f}")
print("\nWHICH IMAGE PC CARRIES IT? (per-PC correlation with each maternal var)")
for j,nm in enumerate(MN):
    rr=[pearsonr(Is[:,k],Y[:,j])[0] for k in range(6)]
    print(f"   {nm:20s} "+"  ".join(f"PC{k+1}={rr[k]:+.2f}" for k in range(6)))
print("\nIS IT ACQUISITION? drop BMI+weight (the habitus axis), keep age+height:")
keep=[j for j,nm in enumerate(MN) if nm in ("maternal_age","maternal_height_cm")]
print(f"   age+height only        {cc(rs(Is,[g,e,B]),rs(Y[:,keep],[g,e,B])):+.4f}")
hab=[j for j,nm in enumerate(MN) if nm in ("maternal_bmi","maternal_weight_kg")]
print(f"   BMI+weight only        {cc(rs(Is,[g,e,B]),rs(Y[:,hab],[g,e,B])):+.4f}")
print("\nRESIDUALISE THE IMAGES ON BMI, then retest the full maternal block:")
bmi=Y[:,MN.index("maternal_bmi")][:,None]
print(f"   images residualised on BMI  {cc(rs(Is,[g,e,B,bmi]),rs(Y,[g,e,B,bmi])):+.4f}")
print("\nCONTROL -- does the image side predict a PURELY administrative variable?")
rng=np.random.default_rng(0)
print(f"   images vs random gaussian   {cc(rs(Is,[g,e,B]),rng.standard_normal((len(Is),4))):+.4f}")
json.dump({"n":int(sel.sum())},open("/tmp/mm_mat.json","w"))
