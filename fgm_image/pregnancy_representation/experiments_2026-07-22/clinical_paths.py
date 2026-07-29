#!/usr/bin/env python3
"""Resolve clinical frame paths: RECURSIVE scan + PREFIX-TOLERANT name join.

The clinical store is nested (processed/grouped/IMPACT_CLINICAL/<machine>/<subdir>/...) and the
on-disk filenames carry a leading numeric prefix that clinical_index.csv does not have:
    disk : 60813_IMP0469_20171213_20171213.131709.293.3816910.OBMBFET.png
    index:       IMP0469_20171213_20171213.131709.293.3816910.OBMBFET
so a flat basename join resolves 0 frames. Rule: strip a leading '<digits>_' (and an optional
'NHC_<digits>_') to get the index key. Debug/mask artefacts are excluded -- they are not frames.

USAGE (probe the match rate before a long run):
    python clinical_paths.py --root /mnt/beegfs/groups/collage/data/IMPACT_CLINICAL
"""
import os, re, argparse, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
INDEX=os.path.join(HERE,"clinical_index.csv")
# suffixes written by the preprocessing debug passes -- not real frames
BAD_SUFFIX=("_inpainting_mask_debug","_text_detection_debug","_cone_debug","_mask_debug","_debug")
BAD_DIR=("debug","overlay","mask")
PREFIX=re.compile(r"^\d+_(?:NHC_\d+_)?")

def norm_key(fn:str)->str:
    k=fn[:-4] if fn.lower().endswith(".png") else fn
    for s in BAD_SUFFIX:
        if k.endswith(s): k=k[:-len(s)]
    return PREFIX.sub("",k)

def build_lookup(root:str):
    """-> {normalised_key: fullpath}. Skips debug/overlay/mask dirs and debug-suffixed files."""
    look={}
    for dp,dns,fns in os.walk(root):
        low=dp.lower()
        if any(b in low for b in BAD_DIR):
            dns[:]=[]; continue
        for fn in fns:
            if not fn.lower().endswith(".png"): continue
            if any(s in fn for s in BAD_SUFFIX): continue
            look.setdefault(norm_key(fn),os.path.join(dp,fn))
    return look

def resolve(root:str,index:str=INDEX):
    df=pd.read_csv(index).copy()
    look=build_lookup(root)
    df["img"]=df["new_filename"].astype(str).map(lambda n: look.get(norm_key(n),""))
    return df,look

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=os.environ.get("CLINICAL_ROOT",
        "/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL"))
    a=ap.parse_args()
    df,look=resolve(a.root)
    ok=df["img"]!=""
    print(f"root            {a.root}")
    print(f"pngs indexed    {len(look)} (debug/overlay/mask excluded)")
    print(f"index rows      {len(df)}")
    print(f"RESOLVED        {ok.sum()} / {len(df)}  ({100*ok.mean():.1f}%)")
    if ok.sum():
        sub=df[ok]
        print(f"GA              {sub.ga_weeks_recovered.min():.1f}-{sub.ga_weeks_recovered.max():.1f} "
              f"SD {sub.ga_weeks_recovered.std():.2f} | fetuses {sub.nid.nunique()}")
        print("example         ",sub['img'].iloc[0])
        dirs=sub['img'].map(lambda p: os.path.basename(os.path.dirname(p))).value_counts().head(5)
        print("top dirs        ",dict(dirs))
    if ok.sum()<len(df):
        miss=df[~ok]["new_filename"].head(3).tolist()
        print("unresolved e.g. ",miss)

if __name__=="__main__": main()
