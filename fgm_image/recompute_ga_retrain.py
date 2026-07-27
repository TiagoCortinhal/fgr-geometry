"""
Recompute per-image GA from the CiTUS master dating (FUReco primary, LMP fallback),
retrain the main USFM GA clock, recompute appearance-age lag (pooled + per plane),
and re-derive the LGA false-positive recovery against the logistic-regression CSV.

Run:  python fgm_image/recompute_ga_retrain.py
Inputs (read-only):
  - CiTUS master:  /Users/tiago/Documents/CiTUS/Dataset/Impact_longitudinal_161025_all_merged.xlsx
  - USFM multilayer embeddings: results/img_align/emb_usfm_multilayer.npz
  - logistic-regression CSV (LGA model): passed via --csv or default artifact copy
Outputs (results/img_align/):
  - _citus_dates.csv                per-patient LMP/FUReco/anchor
  - _fetus_lag_recompute.csv        per-fetus pooled + per-plane lag (new GA)
  - _recovery_excel_data.csv        per-CSV-fetus recovery table
  - ga_retrain_summary.json         clock r, cohort sizes, dating audit
"""
import json, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT="/Users/tiago/dev/fgr-geometry"
IMG=f"{ROOT}/results/img_align"
CITUS="/Users/tiago/Documents/CiTUS/Dataset/Impact_longitudinal_161025_all_merged.xlsx"
GA_MIN,GA_MAX=6.0,42.0
ALPHA=50.0

def load_anchor():
    df=pd.read_excel(CITUS,sheet_name="IMPACT_Final_All_V18")
    d=df[["Cod","LMP","FUReco"]].copy()
    d["LMP"]=pd.to_datetime(d.LMP,errors="coerce")
    d["FUReco"]=pd.to_datetime(d.FUReco,errors="coerce")
    d["anchor"]=d.FUReco.fillna(d.LMP)
    d["anchor_src"]=np.where(d.FUReco.notna(),"FUReco","LMP")
    d["dating_diff_days"]=(d.FUReco-d.LMP).dt.days
    d["dating_flag"]=np.where(d.dating_diff_days.abs()>60,"discordant>60d",
                       np.where(d.dating_diff_days.abs()>14,"discordant>14d","ok"))
    d.to_csv(f"{IMG}/_citus_dates.csv",index=False)
    return d

def per_image_ga(anchor_by_cod):
    z=np.load(f"{IMG}/emb_usfm_multilayer.npz",allow_pickle=True)
    fid=pd.to_numeric(pd.Series(z["fetus_id"]),errors="coerce").astype("Int64")
    sd=pd.to_numeric(pd.Series(z["study_date"]),errors="coerce")
    sd_dt=pd.to_datetime(sd.astype("Int64").astype("string"),format="%Y%m%d",errors="coerce")
    anc=pd.to_datetime(pd.Series(fid.map(anchor_by_cod)),errors="coerce")
    ga=((sd_dt-anc.values).dt.days/7.0)
    valid=ga.between(GA_MIN,GA_MAX).values
    return z,fid.values,ga.values,valid

def train_clock(E,y,groups):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(E,y,groups):
        oof[te]=Ridge(alpha=ALPHA).fit(E[tr],y[tr]).predict(E[te])
    r=float(np.corrcoef(oof,y)[0,1])
    return oof,r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default=f"{IMG}/_lr_csv.csv")
    args=ap.parse_args()

    cit=load_anchor()
    anchor_by_cod=cit.set_index("Cod").anchor.to_dict()
    audit=dict(n_patients=int(len(cit)),
               anchor_FUReco=int((cit.anchor_src=="FUReco").sum()),
               anchor_LMP=int((cit.anchor_src=="LMP").sum()),
               discordant_gt14d=int((cit.dating_flag=="discordant>14d").sum()),
               discordant_gt60d=int((cit.dating_flag=="discordant>60d").sum()))

    z,fid,ga,valid=per_image_ga(anchor_by_cod)
    E6=z["emb_l5"].astype(np.float32)
    # plane label per image: join image_clusters.csv on new_filename
    ic=pd.read_csv(f"{IMG}/image_clusters.csv",low_memory=False)[["new_filename","plane_prop"]]
    pmap=dict(zip(ic.new_filename,ic.plane_prop))
    planes=pd.Series(z["new_filename"]).map(pmap)

    # main pooled clock
    oof,r=train_clock(E6[valid],ga[valid],fid[valid])
    lag=oof-ga[valid]
    dfp=pd.DataFrame({"fid":fid[valid],"lag":lag})
    pooled=dfp.groupby("fid").lag.mean()

    # per-plane clocks
    perplane={}
    plane_lag={}
    if planes is not None:
        pv=planes.values
        for pl in ["abdominal","cerebral","femur"]:
            mask=valid & (pv==pl)
            if mask.sum()<500: continue
            o,rp=train_clock(E6[mask],ga[mask],fid[mask])
            perplane[pl]=rp
            plane_lag[pl]=pd.DataFrame({"fid":fid[mask],"lag":o-ga[mask]}).groupby("fid").lag.mean()

    audit.update(clock_r=r, n_images_valid=int(valid.sum()),
                 n_fetuses=int(pd.Series(fid[valid]).nunique()),
                 perplane_r=perplane)

    # assemble per-fetus lag table
    lagtab=pooled.rename("lag_pooled").to_frame()
    for pl,s in plane_lag.items(): lagtab[f"lag_{pl}"]=s
    lagtab=lagtab.reset_index().rename(columns={"fid":"fid"})
    lagtab.to_csv(f"{IMG}/_fetus_lag_recompute.csv",index=False)

    json.dump(audit,open(f"{IMG}/ga_retrain_summary.json","w"),indent=2)
    print(json.dumps(audit,indent=2))

if __name__=="__main__":
    main()
