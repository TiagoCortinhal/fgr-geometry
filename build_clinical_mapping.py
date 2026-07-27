#!/usr/bin/env python3
"""Build CLINICAL image mapping CSV (no copy), porting flatten_dicom.py clinical logic.
Structure: {nhc}/{study_id}/{timestamp.machine}  and year-range/{nhc_suffix}/{study_suffix}/...
Emits same schema as IMPACT mapping so the two concatenate."""
import os, re, json, csv
INPUT_DIR="/Users/tiago/usb/IMPACT_clinical_anon/export_PACS"
JSON_PATH="/Users/tiago/dev/fetal_mc2vae/data/csv/biometric_results_with_brain_labels.json.bak"
OUT="/Users/tiago/dev/fgr-geometry/results/img_align/clinical_image_mapping.csv"
import pydicom
SKIP_FOLDERS={'.claude','duplicate_examples','flatten_images'}
SKIP_FILES={'.DS_Store','flatten_mapping.csv','flatten_duplicates.csv','flatten_errors.csv','Seleccion_estudios.xlsx'}
YEAR_RANGE={'2015-2017','2018-2021'}

def extract_imp_number(t):
    if not t: return None
    m=re.search(r'IMP[ACT]*\s*(\d+)',str(t),re.IGNORECASE); return int(m.group(1)) if m else None

def resolve_clinical(folder,pid,pname):
    n=extract_imp_number(pid)
    if n is None: n=extract_imp_number(pname)
    if n is not None: return f"IMP{n:04d}"
    fc=folder
    for suf in ['_ok','_SI','_NO']:
        if fc.endswith(suf): fc=fc[:-len(suf)]; break
    if fc.startswith('IMP'): return fc[:7] if len(fc)>=7 else fc
    if fc.isdigit(): return f"NHC_{fc}"
    return fc

def norm(pid):
    if pid and pid.upper().startswith('IMP'):
        num=''.join(c for c in pid[3:] if c.isdigit())
        if num: return f"IMP_{int(num):04d}"
    if pid and pid.startswith('NHC_'): return pid
    return pid

# ---- discover clinical files ----
files=[]
def proc_nhc(nhc_path,nhc_name,yr=None):
    nhc_clean=nhc_name
    for suf in ['_ok','_SI','_NO']:
        if nhc_clean.endswith(suf): nhc_clean=nhc_clean[:-len(suf)]; break
    for sid in os.listdir(nhc_path):
        sp=os.path.join(nhc_path,sid)
        if not os.path.isdir(sp): continue
        sid_clean=sid
        for suf in ['_SI','_NO','_ok']:
            if sid_clean.endswith(suf): sid_clean=sid_clean[:-len(suf)]; break
        for fn in os.listdir(sp):
            full=os.path.join(sp,fn)
            if not os.path.isfile(full): continue
            if fn.endswith('.png') or fn.startswith('.') or fn in SKIP_FILES: continue
            files.append(dict(folder=nhc_clean,subfolder=sid_clean,filename=fn,full_path=full,
                              relative_path=f"{nhc_clean}/{sid_clean}/{fn}"))
for folder in os.listdir(INPUT_DIR):
    fp=os.path.join(INPUT_DIR,folder)
    if not os.path.isdir(fp) or folder in SKIP_FOLDERS: continue
    if folder in YEAR_RANGE:
        for nhc in os.listdir(fp):
            np_=os.path.join(fp,nhc)
            if os.path.isdir(np_) and not nhc.startswith('.'): proc_nhc(np_,nhc,folder)
    elif folder.isdigit() or folder.startswith('IMP'):
        proc_nhc(fp,folder,None)
print(f"discovered {len(files)} clinical DICOM files",flush=True)

# ---- JSON lookup (clinical rel paths) ----
J=json.load(open(JSON_PATH)); lut={}
for pid,pat in J['patients'].items():
    for sid,sc in pat.get('scans',{}).items():
        for it in ['fetal_brain','fetal_abdomen','fetal_femur']:
            for idx,img in enumerate(sc.get(it,[])):
                rp=img.get('relative_path','')
                if rp: lut[rp]=dict(json_patient_id=pid,scan_id=sid,img_type=it,
                                    GA_weeks=sc.get('GA_weeks',''),scan_date=sc.get('scan_date',''),
                                    data_source=sc.get('data_source',''),
                                    brain_subplane=img.get('brain_subplane',''),
                                    pixel_spacing_mm=img.get('pixel_spacing_mm',''),
                                    label_confidence=img.get('label_confidence',''))
print(f"JSON labeled paths (all): {len(lut)}",flush=True)

PLANE={'fetal_brain':'cerebral','fetal_abdomen':'abdominal','fetal_femur':'femur'}
rows=[]; copied=set(); n_json=0; n_dup=0
for i,d in enumerate(files):
    if (i+1)%5000==0: print(f"  read {i+1}/{len(files)}",flush=True)
    try:
        ds=pydicom.dcmread(d['full_path'],stop_before_pixels=True)
        pid=str(ds.get('PatientID','')).strip(); pname=str(ds.get('PatientName','')).strip()
        sdate=str(ds.get('StudyDate','')).strip(); sop=str(ds.get('SOPInstanceUID','')).strip()
    except Exception:
        pid=pname=sdate=sop=''
    if not sdate:
        fn=d['filename']
        if '.' in fn and fn.split('.')[0].isdigit() and len(fn.split('.')[0])==8: sdate=fn.split('.')[0]
        else: sdate=d['subfolder']
    rpid=resolve_clinical(d['folder'],pid,pname)
    j=lut.get(d['relative_path'])
    if j: n_json+=1
    dk=(rpid,sdate,d['filename']); is_dup=dk in copied
    if is_dup: n_dup+=1
    else: copied.add(dk)
    rows.append(dict(
        resolved_patient_id=rpid, normalized_patient_id=norm(rpid), study_date=sdate,
        sop_uid=sop or d['filename'], plane=(PLANE.get(j['img_type'],'') if j else ''),
        ga_weeks=(j['GA_weeks'] if j else ''), scan_id=(j['scan_id'] if j else ''),
        brain_subplane=(j['brain_subplane'] if j else ''),
        pixel_spacing_mm=(j['pixel_spacing_mm'] if j else ''),
        label_confidence=(j['label_confidence'] if j else ''),
        in_json=bool(j), is_duplicate=is_dup, dicom_patient_id=pid,
        original_folder=d['folder'], original_subfolder=d['subfolder'],
        original_filename=d['filename'], original_path=d['relative_path'],
        new_filename=f"{rpid}_{sdate}_{d['filename']}", full_path=d['full_path'],
        dataset_type='clinical'))
os.makedirs(os.path.dirname(OUT),exist_ok=True)
cols=['resolved_patient_id','normalized_patient_id','study_date','sop_uid','plane','ga_weeks','scan_id',
      'brain_subplane','pixel_spacing_mm','label_confidence','in_json','is_duplicate','dicom_patient_id',
      'original_folder','original_subfolder','original_filename','original_path','new_filename','full_path','dataset_type']
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"WROTE {len(rows)} rows -> {OUT}",flush=True)
print(f"in_json={n_json} not_in_json={len(rows)-n_json} dup={n_dup} unique_fetuses={len(set(r['resolved_patient_id'] for r in rows))}",flush=True)
