#!/usr/bin/env python3
"""Extract pixel spacing from IMPACT source DICOMs -> mapping CSV.
Key = SOPInstanceUID (== our frame new_filename trailing UID). Also record IMP code + study date
from the path. Prefers PhysicalDeltaX (ultrasound region, cm) -> mm; falls back to PixelSpacing.
"""
import os, glob, re, csv, pydicom, time
ROOT="/Users/tiago/usb/Tiago/imagenes_impact_sin_label"
OUT="handoff/dicom_spacing.csv"
os.makedirs("handoff",exist_ok=True)

def spacing_mm(ds):
    # ultrasound region PhysicalDeltaX is in cm/pixel when PhysicalUnitsXDirection==3
    try:
        r=ds.SequenceOfUltrasoundRegions[0]
        dx=float(getattr(r,"PhysicalDeltaX")); ux=int(getattr(r,"PhysicalUnitsXDirection",3))
        if dx>0: return dx*10.0 if ux==3 else dx   # cm->mm
    except Exception: pass
    try:
        ps=ds.PixelSpacing; return float(ps[0])
    except Exception: return None

def main():
    files=glob.glob(f"{ROOT}/*/*/*")
    print(f"dicoms {len(files)}",flush=True)
    t0=time.time(); n=0
    with open(OUT,"w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["uid","imp_dir","study_date","spacing_mm"])
        for i,f in enumerate(files):
            uid=os.path.basename(f); parts=f.split("/")
            imp_dir=parts[-3]; study=parts[-2]
            try:
                ds=pydicom.dcmread(f,stop_before_pixels=True,specific_tags=None)
                sp=spacing_mm(ds)
            except Exception:
                sp=None
            if sp is not None: n+=1
            w.writerow([uid,imp_dir,study,sp])
            if (i+1)%2000==0:
                fh.flush(); print(f"  {i+1}/{len(files)} got_spacing={n} {(i+1)/(time.time()-t0):.0f}/s",flush=True)
    print(f"DONE {n}/{len(files)} with spacing, {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
