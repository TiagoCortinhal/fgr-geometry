#!/usr/bin/env python
"""Verify the frame paths and inputs BEFORE burning GPU hours.

    python check_paths.py --image-root $IMPACT --image-root-clinical $CLINICAL

Runs from any working directory (it puts its own directory on sys.path), so it
works inside a container where `.` is not the package root.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default=os.environ.get("IMPACT"),
                   help="IMPACT frames (evaluation cohort); defaults to $IMPACT")
    p.add_argument("--image-root-clinical", default=os.environ.get("CLINICAL"),
                   help="clinical frames (pretraining only); defaults to $CLINICAL")
    p.add_argument("--manifest", default=os.path.join(HERE, "data/image_clusters.csv"))
    p.add_argument("--panel", default=os.path.join(HERE, "data/panel.npz"))
    p.add_argument("--frozen", default=os.path.join(HERE, "data/frozen_usfm.npz"))
    p.add_argument("--keep-csv", default=None)
    a = p.parse_args()

    from fgm_ssl.data import FrameManifest

    problems = []
    print(f"[cwd] {os.getcwd()}\n[package] {HERE}\n")

    for label, path in (("manifest", a.manifest), ("panel", a.panel),
                        ("frozen_usfm", a.frozen)):
        ok = path and os.path.exists(path)
        size = f"{os.path.getsize(path) / 1e6:.1f} MB" if ok else "MISSING"
        print(f"[input] {label:12s} {size:>10s}  {path}")
        if not ok:
            problems.append(f"{label} not found at {path}")

    if not a.image_root:
        problems.append("no --image-root (or $IMPACT) given")
    print()
    if problems:
        for x in problems:
            print(f"  PROBLEM: {x}")
        sys.exit(1)

    roots = {"impact": a.image_root}
    if a.image_root_clinical:
        roots["clinical"] = a.image_root_clinical
    for k, v in roots.items():
        print(f"[root] {k:9s} {'exists' if os.path.isdir(v) else 'NOT A DIRECTORY'}  {v}")
    print()

    ev = FrameManifest(a.manifest, roots, ["impact"], a.keep_csv)
    n_listed = len(ev.df)
    diag = ev.diagnose()          # capture BEFORE existing() filters rows away
    ev.existing()
    print(f"[eval] IMPACT: {n_listed} rows in manifest -> {len(ev.df)} files found on disk")
    print(f"       {ev.counts()}  | extension resolved: {ev.resolved_ext}")
    if len(ev.df) == 0:
        problems.append("no IMPACT files resolved")
        print("\n  --- why nothing matched ---")
        for ds, d in diag.items():
            print(f"  [{ds}] root: {d['root']}")
            for x in d["manifest_expects"]:
                print(f"      manifest expects: {x}")
            for x in d["found_on_disk"]:
                print(f"      actually on disk: {x}")
        print("  Compare the two: if the stems match but the suffix differs, the "
              "extension probe missed it -- tell me the real suffix. If the stems "
              "differ, --image-root is the wrong directory (try inpainted/ or "
              "cropped/, or a level deeper).")
    elif len(ev.df) < 0.5 * n_listed:
        problems.append(f"only {len(ev.df)}/{n_listed} IMPACT files resolved -- "
                        f"partial path mismatch")

    if a.image_root_clinical:
        pm = FrameManifest(a.manifest, roots, ["impact", "clinical"], a.keep_csv)
        n2 = len(pm.df)
        pm.existing()
        print(f"\n[pretrain] {n2} rows -> {len(pm.df)} files found")
        print(f"           {pm.counts()}")
        print(f"           ids shared across cohorts: {pm.id_collision()} "
              f"(namespaced as '<cohort>:<id>', NOT merged)")
        if len(pm.df) <= len(ev.df):
            problems.append("pretrain pool did not grow beyond IMPACT -- "
                            "--image-root-clinical is wrong (try inpainted/ "
                            "instead of preprocessed/)")
    else:
        print("\n[pretrain] no clinical root given; mae/contrast will pretrain on "
              "IMPACT only (~21k frames instead of ~68k)")

    print()
    if problems:
        for x in problems:
            print(f"  PROBLEM: {x}")
        print("\nFAILED -- fix the above before training.")
        sys.exit(1)
    print("OK -- ready to train.")


if __name__ == "__main__":
    main()
