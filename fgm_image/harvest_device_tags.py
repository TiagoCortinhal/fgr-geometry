"""Harvest DICOM device tags (manufacturer/model/serial/station/software/date)
for the GA-lag images, keyed by new_filename. Resumable-ish: writes one CSV.

    python -m fgm_image.harvest_device_tags
"""
import os, json
import numpy as np
import pandas as pd
import pydicom

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = f"{WS}/results/img_align"
TAGS = ["Manufacturer", "ManufacturerModelName", "DeviceSerialNumber",
        "StationName", "SoftwareVersions", "InstitutionName", "StudyDate"]


def main():
    wl = pd.read_csv(f"{IMG}/_lag_device_worklist.csv")
    rows = []
    n = 0
    for r in wl.itertuples(index=False):
        rec = {"new_filename": r.new_filename, "dataset_type": r.dataset_type}
        try:
            ds = pydicom.dcmread(r.full_path, stop_before_pixels=True, force=True)
            for t in TAGS:
                rec[t] = getattr(ds, t, None)
        except Exception as e:
            for t in TAGS:
                rec[t] = None
            rec["err"] = str(e)[:40]
        rows.append(rec)
        n += 1
        if n % 2000 == 0:
            print(f"{n}/{len(wl)} headers read", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{IMG}/_lag_device_tags.csv", index=False)
    have = out.ManufacturerModelName.notna() | out.DeviceSerialNumber.notna()
    print(f"DONE {n} headers | device-tagged {have.sum()} ({100*have.mean():.0f}%)", flush=True)
    print("models:", out.ManufacturerModelName.value_counts().to_dict(), flush=True)
    print("serials:", out.DeviceSerialNumber.nunique(), "distinct", flush=True)


if __name__ == "__main__":
    main()
