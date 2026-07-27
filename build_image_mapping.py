#!/usr/bin/env python3
"""Build image mapping CSV for flattened IMPACT DICOMs WITHOUT copying files.
Reuses the identifier-resolution logic from flatten_dicom.py (discovery + resolve
+ dedup), joins to biometric JSON for plane/GA/biometry, emits one row per DICOM.
"""
import os, re, json, csv, sys
INPUT_DIR="/Users/tiago/usb/Tiago/imagenes_impact_sin_label"
JSON_PATH="/Users/tiago/dev/fetal_mc2vae/data/csv/biometric_results_with_brain_labels.json.bak"
OUT="/Users/tiago/dev/fgr-geometry/results/img_align/impact_image_mapping.csv"
import pydicom
SKIP_FOLDERS={'.claude','duplicate_examples','flatten_images'}

def extract_imp_number(text):
    if not text: return None
    m=re.search(r'IMP[ACT]*\s*(\d+)', str(text), re.IGNORECASE)
    return int(m.group(1)) if m else None

def resolve_patient_id_impact(folder_name, pid, pname):
    n=extract_imp_number(pid)
    if n is None: n=extract_imp_number(pname)
    if n is not None: return f"IMP{n:04d}"
    return folder_name[:7]

def normalize_pid(pid):
    if pid and pid.upper().startswith('IMP'):
        num=''.join(c for c in pid[3:] if c.isdigit())
        if num: return f"IMP_{int(num):04d}"
    return pid

# ---- discover (IMPACT structure: IMP{code}{date}/{date}/{uid}) ----
files=[]
for folder in os.listdir(INPUT_DIR):
    fp=os.path.join(INPUT_DIR,folder)
    if not os.path.isdir(fp) or not folder.startswith('IMP') or folder in SKIP_FOLDERS: continue
    for datef in os.listdir(fp):
        dp=os.path.join(fp,datef)
        if not os.path.isdir(dp): continue
        for fn in os.listdir(dp):
            if fn.startswith('1.2.') and not fn.endswith('.png'):
                full=os.path.join(dp,fn)
                if os.path.isfile(full):
                    files.append(dict(folder=folder,subfolder=datef,filename=fn,full_path=full,
                                      relative_path=f"{folder}/{datef}/{fn}"))
print(f"discovered {len(files)} DICOM files", flush=True)

# ---- JSON lookup: relative_path -> plane/GA/biometry ----
J=json.load(open(JSON_PATH)); lut={}
for pid,pat in J['patients'].items():
    for sid,sc in pat.get('scans',{}).items():
        for it in ['fetal_brain','fetal_abdomen','fetal_femur']:
            for idx,img in enumerate(sc.get(it,[])):
                rp=img.get('relative_path','')
                if rp:
                    lut[rp]=dict(json_patient_id=pid,scan_id=sid,img_type=it,image_index=idx,
                                 GA_weeks=sc.get('GA_weeks',''),scan_date=sc.get('scan_date',''),
                                 data_source=sc.get('data_source',''),
                                 brain_subplane=img.get('brain_subplane',''),
                                 pixel_spacing_mm=img.get('pixel_spacing_mm',''),
                                 label_confidence=img.get('label_confidence',''),
                                 HC_cm=img.get('HC_cm',''),BPD_cm=img.get('BPD_cm',''))
print(f"JSON has {len(lut)} labeled image paths", flush=True)

PLANE={'fetal_brain':'cerebral','fetal_abdomen':'abdominal','fetal_femur':'femur'}
rows=[]; copied=set(); n_json=0; n_dup=0
for i,d in enumerate(files):
    if (i+1)%2000==0: print(f"  read {i+1}/{len(files)}", flush=True)
    try:
        ds=pydicom.dcmread(d['full_path'],stop_before_pixels=True)
        pid=str(ds.get('PatientID','')).strip(); pname=str(ds.get('PatientName','')).strip()
        sdate=str(ds.get('StudyDate','')).strip()
        sop=str(ds.get('SOPInstanceUID','')).strip()
    except Exception:
        pid=pname=sdate=sop=''
    if not sdate:
        sdate=d['subfolder'] if (d['subfolder'].isdigit() and len(d['subfolder'])==8) else d['subfolder']
    rpid=resolve_patient_id_impact(d['folder'],pid,pname)
    j=lut.get(d['relative_path'])
    if j: n_json+=1
    dedup=(rpid,sdate,d['filename'])
    is_dup=dedup in copied
    if is_dup: n_dup+=1
    else: copied.add(dedup)
    fp=d['folder'][:7] if d['folder'].startswith('IMP') else d['folder']
    rows.append(dict(
        resolved_patient_id=rpid, normalized_patient_id=normalize_pid(rpid),
        study_date=sdate, sop_uid=sop or d['filename'],
        plane=(PLANE.get(j['img_type'],'') if j else ''),
        ga_weeks=(j['GA_weeks'] if j else ''), scan_id=(j['scan_id'] if j else ''),
        brain_subplane=(j['brain_subplane'] if j else ''),
        pixel_spacing_mm=(j['pixel_spacing_mm'] if j else ''),
        label_confidence=(j['label_confidence'] if j else ''),
        in_json=bool(j), is_duplicate=is_dup,
        dicom_patient_id=pid, original_folder=d['folder'], original_subfolder=d['subfolder'],
        original_filename=d['filename'], original_path=d['relative_path'],
        new_filename=f"{rpid}_{sdate}_{d['filename']}",
        full_path=d['full_path']))
os.makedirs(os.path.dirname(OUT),exist_ok=True)
cols=['resolved_patient_id','normalized_patient_id','study_date','sop_uid','plane','ga_weeks',
      'scan_id','brain_subplane','pixel_spacing_mm','label_confidence','in_json','is_duplicate',
      'dicom_patient_id','original_folder','original_subfolder','original_filename','original_path',
      'new_filename','full_path']
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"WROTE {len(rows)} rows -> {OUT}", flush=True)
print(f"in_json={n_json} not_in_json={len(rows)-n_json} duplicates={n_dup} unique_fetuses={len(set(r['resolved_patient_id'] for r in rows))}", flush=True)
