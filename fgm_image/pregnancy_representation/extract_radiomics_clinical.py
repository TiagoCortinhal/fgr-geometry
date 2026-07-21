#!/usr/bin/env python3
"""Clinical radiomics extraction. Flood-fill cone + existing caliper mask or v2
detector. Compact wavelet stack (~222 feat). Shards to disk. Run in radiomics env."""
import os, sys, time, json, numpy as np, pandas as pd
from PIL import Image
import scipy.ndimage as ndi
import SimpleITK as sitk
from radiomics import featureextractor
import logging; logging.getLogger("radiomics").setLevel(logging.CRITICAL)

IDX="handoff/clinical_frame_index.parquet"
OUT="handoff/radiomics_clinical"; os.makedirs(OUT,exist_ok=True)
SHARD=1000

def load_gray(p): return np.asarray(Image.open(p).convert("L"))
def align_to(m,shape):
    if m.shape==shape: return m
    return np.asarray(Image.fromarray(m.astype(np.uint8)).resize((shape[1],shape[0]),Image.NEAREST))
def floodfill_cone(a,thr=10):
    lbl,n=ndi.label(a<thr)
    border=set(np.unique(np.concatenate([lbl[0],lbl[-1],lbl[:,0],lbl[:,-1]])))-{0}
    return ndi.binary_fill_holes(~np.isin(lbl,list(border)))
def detect_annotations(a,cone,bright_pct=99.3):
    thr=np.percentile(a[cone],bright_pct); bright=(a>=thr)&cone
    lbl,n=ndi.label(bright)
    if n==0: return np.zeros_like(cone)
    H,W=a.shape; by,bx=int(0.06*H),int(0.06*W)
    bb=np.zeros_like(cone); bb[:by]=1;bb[-by:]=1;bb[:,:bx]=1;bb[:,-bx:]=1
    ann=np.zeros_like(cone); objs=ndi.find_objects(lbl)
    for i in range(1,n+1):
        sl=objs[i-1]; comp=(lbl[sl]==i); sz=comp.sum()
        h=sl[0].stop-sl[0].start; w=sl[1].stop-sl[1].start
        fill=sz/(h*w+1e-9); elong=max(h,w)/(min(h,w)+1e-9)
        if (bb[sl]&comp).any() or (elong>=4 and fill<0.5) or (sz<40 and fill<0.6):
            ann[sl]|=comp
    return ndi.binary_dilation(ann,iterations=2)
def build_roi(img_p,cal_p,erode_px=4,dilate_cal=3):
    a=load_gray(img_p); H,W=a.shape; cone=ndi.binary_erosion(floodfill_cone(a),iterations=erode_px)
    roi=cone.copy()
    if cal_p and os.path.exists(cal_p):
        cal=ndi.binary_dilation(align_to(load_gray(cal_p),(H,W))>127,iterations=dilate_cal); roi=roi&~cal
    else:
        roi=roi&~detect_annotations(a,cone)
    return a,roi

def write_shard(df,base):
    try: df.to_parquet(base+".parquet")
    except Exception: df.to_pickle(base+".pkl")

_EXT=None
def get_ext():
    global _EXT
    if _EXT is None:
        s={"binWidth":16,"force2D":True,"force2Ddimension":0,"label":1,
           "resampledPixelSpacing":None,"normalize":True,"normalizeScale":100}
        e=featureextractor.RadiomicsFeatureExtractor(**s); e.disableAllFeatures()
        for c in ["firstorder","glcm","glrlm","glszm"]: e.enableFeatureClassByName(c)
        e.enableImageTypeByName("Wavelet"); _EXT=e
    return _EXT
def work(name,img,cal):
    try:
        a,roi=build_roi(img,cal or None)
        if roi.sum()<500: return (name,None,"roi_small")
        res=get_ext().execute(sitk.GetImageFromArray(a.astype(np.float32)),sitk.GetImageFromArray(roi.astype(np.uint8)))
        return (name,{k:float(v) for k,v in res.items() if not k.startswith("diagnostics")},"ok")
    except Exception as ex: return (name,None,str(ex)[:80])

def main():
    idx=pd.read_parquet(IDX)
    done=set()
    for f in os.listdir(OUT):
        p=os.path.join(OUT,f)
        if f.endswith(".parquet"): done|=set(pd.read_parquet(p)["name"])
        elif f.endswith(".pkl"): done|=set(pd.read_pickle(p)["name"])
    idx=idx[~idx.new_filename.isin(done)]
    print(f"todo {len(idx)}",flush=True)
    t0=time.time(); buf=[]; errs=[]; shard=len([f for f in os.listdir(OUT) if f.endswith(('.parquet','.pkl'))])
    for i,r in enumerate(idx.itertuples(index=False)):
        name,feats,st=work(r.new_filename,r.png,r.caliper)
        if feats is not None: feats["name"]=name; buf.append(feats)
        else: errs.append((name,st))
        if len(buf)>=SHARD: write_shard(pd.DataFrame(buf),f"{OUT}/shard_{shard:03d}");shard+=1;buf=[]
        if (i+1)%100==0:
            el=time.time()-t0; rate=(i+1)/el
            print(f"  {i+1}/{len(idx)} {rate:.1f}/s ETA {(len(idx)-i-1)/rate/60:.0f}min errs {len(errs)}",flush=True)
    if buf: write_shard(pd.DataFrame(buf),f"{OUT}/shard_{shard:03d}")
    json.dump(errs,open(f"{OUT}/_errors.json","w"))
    print(f"DONE {time.time()-t0:.0f}s errs {len(errs)}",flush=True)

if __name__=="__main__": main()
