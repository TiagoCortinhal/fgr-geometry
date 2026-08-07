"""THE DECIDING TEST: repeat the split-sample over 20 random seeds.
A real effect transfers in most splits; a lucky maximum in few."""
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
for nm,IM in (("USFM",USFM),("radiomics",RAD)):
    sel_counts={}; rBs=[]; lv=[]
    for s in range(20):
        r=fgm_split_sample_screen(Y,CARD,IM,COV,n_top=2,seed=s,nperm=200)
        for x in r["tested_B"]:
            sel_counts[x["var"]]=sel_counts.get(x["var"],0)+1
            rBs.append((x["var"],x["r_B"],x["p"]))
            if x["var"]=="Percentil_LV_basal": lv.append((x["r_B"],x["p"]))
    top=sorted(sel_counts.items(),key=lambda t:-t[1])
    print(f"\n=== {nm} — 20 random splits ===")
    print(f"  how often each variable is A-SELECTED (top-2 of 11): {dict(top[:6])}")
    sig=[(v,p) for v,r_,p in rBs if p<0.05]
    print(f"  B-tests with p<0.05: {len(sig)} of {len(rBs)}")
    if lv:
        a=np.array([x[0] for x in lv]); pp=np.array([x[1] for x in lv])
        print(f"  LV_basal: selected in {len(lv)}/20 splits | r_B mean {a.mean():+.3f} "
              f"[{a.min():+.3f},{a.max():+.3f}] | p<0.05 in {(pp<0.05).sum()}/{len(lv)}")
    out[nm]=dict(selection_counts=sel_counts,n_sig=len(sig),n_tests=len(rBs),
                 lv_basal=dict(n_selected=len(lv),r_B=[float(x[0]) for x in lv],p=[float(x[1]) for x in lv]) if lv else None)
json.dump(out,open("/tmp/vent20.json","w"),indent=1)
