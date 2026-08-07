#!/usr/bin/env python
"""Score every arm against the prespecified endpoints and apply the stop rule.

    python score_arms.py --panel data/panel.npz --results results \
                         --frozen data/frozen_usfm.npz

Runs AFTER training, on CPU. Deliberately separate from run_ssl.py so identical
scoring code touches every arm including the frozen incumbent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fgm_ssl.evaluate import (benjamini_hochberg, evaluate_arm, heldout_cc,
                              residualise, split_spread_delta)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel.npz")
    p.add_argument("--results", default="results")
    p.add_argument("--frozen", default="data/frozen_usfm.npz",
                   help="npz with E (n,d) + fids: the incumbent representation")
    p.add_argument("--n-pc", type=int, default=12)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-splits", type=int, default=200,
                   help="independent CV seeds for the stop-rule interval (NOT a bootstrap)")
    p.add_argument("--min-n", type=int, default=120,
                   help="minimum fetuses per endpoint; lower only for smoke tests")
    p.add_argument("--out", default="results/scores.json")
    return p.parse_args()


def to_pcs(E, k):
    m = np.isfinite(E).all(1)
    S = np.full((len(E), min(k, E.shape[1])), np.nan)
    P = PCA(min(k, E.shape[1], int(m.sum()) - 1), random_state=0).fit(E[m])
    S[m, :P.n_components_] = P.transform(E[m])
    return (S - np.nanmean(S, 0)) / np.nanstd(S, 0)


def main():
    a = get_args()
    z = np.load(a.panel, allow_pickle=True)
    Z, cols, blocks = z["Z"], [str(c) for c in z["cols"]], [str(b) for b in z["blocks"]]
    fids, ga, bmi = np.array([int(f) for f in z["fids"]]), z["ga"], z["bmi"]

    idx = {b: np.array([i for i, x in enumerate(blocks) if x == b]) for b in set(blocks)}
    endpoints = {"E1_growth": idx["growth"], "E2_Doppler": idx["Doppler"],
                 "E3_cardiac": idx["cardiac"]}
    lv = cols.index("Percentil_LV_basal") if "Percentil_LV_basal" in cols else None
    if lv is not None:
        endpoints["E4_LV_basal"] = np.array([lv])
    # positive controls -- if these are null the encoder did not train
    endpoints["C1_maternal_BMI"] = np.asarray(bmi).reshape(-1, 1)
    endpoints["C2_GA"] = np.asarray(ga).reshape(-1, 1)

    # ---- SCORING MASK: a trained encoder must NEVER be scored on the fetuses it
    # trained on. The supervised arm sees the target block during training, so
    # scoring it on its own training fetuses measures memorisation, not signal
    # (a smoke test scored +0.851 on the block it was trained on and ~0 on every
    # other block -- the signature of exactly this leak). run_ssl.py records
    # heldout_fids in each npz; we honour it here.
    arms, masks = {}, {}
    if os.path.exists(a.frozen):
        fz = np.load(a.frozen, allow_pickle=True)
        arms["frozen"] = to_pcs(fz["E"], a.n_pc)
        masks["frozen"] = np.ones(len(fids), bool)  # never trained: all rows usable
    for f in sorted(glob.glob(os.path.join(a.results, "*_embeddings.npz"))):
        nm = os.path.basename(f).replace("_embeddings.npz", "")
        z2 = np.load(f, allow_pickle=True)
        arms[nm] = to_pcs(z2["E"], a.n_pc)
        if "heldout_fids" in z2.files:
            ho = {int(x) for x in z2["heldout_fids"]}
            masks[nm] = np.array([int(x) in ho for x in fids])
        else:
            raise SystemExit(f"{f} has no heldout_fids -- retrain with the current "
                             f"run_ssl.py; scoring a trained arm on its training "
                             f"fetuses is not permitted")
    assert arms, "no embeddings found -- run run_ssl.py first"
    for nm in arms:
        print(f"[score] {nm:12s} scoring on {int(masks[nm].sum()):4d} fetuses "
              f"({'held-out only' if nm != 'frozen' else 'all -- never trained'})",
              flush=True)

    out = {}
    for nm, IMG in arms.items():
        mk = masks[nm]
        IMGm = np.where(mk[:, None], IMG, np.nan)   # NaN rows are dropped downstream
        out[nm] = evaluate_arm(IMGm, Z, ga, bmi, endpoints, n_perm=a.n_perm,
                               min_n=a.min_n)
        out[nm]["_scored_on"] = dict(n=int(mk.sum()),
                                     basis="held-out fetuses only" if nm != "frozen"
                                     else "all fetuses (never trained)")
        for e, r in out[nm].items():
            if e.startswith("_") or not isinstance(r, dict) or r.get("skipped"):
                continue
            lad = r["ladder"]
            fin = lad.get("GA+BMI", lad.get("adjusted", float("nan")))
            print(f"  {nm:12s} {e:18s} n={r['n']:4d} "
                  f"raw {lad['raw']:+.3f} adj {fin:+.3f} p={r['p']:.4f}", flush=True)

    # ---- positive-control gate: a null result is only interpretable if the
    # encoder demonstrably learned something. Report, do not silently pass.
    gate = {}
    for nm in arms:
        r1, r2 = out[nm].get("C1_maternal_BMI", {}), out[nm].get("C2_GA", {})
        c1 = r1.get("ladder", {}).get("raw", np.nan)
        c2 = r2.get("ladder", {}).get("raw", np.nan)
        gate[nm] = dict(bmi_raw=float(c1), ga_raw=float(c2),
                        skipped=bool(r1.get("skipped")),
                        encoder_trained=bool(np.isfinite(c1) and c1 > 0.15))
    out["_positive_control_gate"] = gate
    print("\n[gate] encoder-works check (raw image->BMI should be strong):")
    for nm, g in gate.items():
        if g["skipped"]:
            status = "NOT RUN (n below min) -- gate says nothing"
        elif g["encoder_trained"]:
            status = "OK"
        else:
            status = "FAILED -- nulls uninterpretable"
        print(f"  {nm:12s} BMI {g['bmi_raw']:+.3f}  GA {g['ga_raw']:+.3f}  {status}")

    # ---- the prespecified stop rule: beat the incumbent on E2 or E4
    if "frozen" in arms:
        ref = arms["frozen"]
        cov = [np.asarray(ga).reshape(-1, 1), np.asarray(bmi).reshape(-1, 1)]
        deltas = {}
        for nm, IMG in arms.items():
            if nm == "frozen":
                continue
            for e in ("E2_Doppler", "E4_LV_basal"):
                if e not in endpoints:
                    continue
                tgt = endpoints[e]
                Y = Z[:, tgt] if np.asarray(tgt).dtype.kind == "i" else np.asarray(tgt)
                # compare the arms on the SAME fetuses, and only ones the new
                # encoder never trained on
                keep = (masks[nm] & np.isfinite(IMG).all(1) & np.isfinite(ref).all(1) &
                        (np.isfinite(Y).sum(1) >= Y.shape[1] - 1))
                if keep.sum() < a.min_n:
                    continue
                cv = [c[keep] for c in cov]
                Ya = residualise(np.where(np.isfinite(Y[keep]), Y[keep], 0.0), cv)
                d = split_spread_delta(residualise(IMG[keep], cv),
                                       residualise(ref[keep], cv), Ya, heldout_cc,
                                       n_splits=a.n_splits)
                d["beats_incumbent"] = bool(d["interval"][0] > 0)
                deltas[f"{nm}|{e}"] = d
                print(f"[stop-rule] {nm:10s} {e:14s} delta {d['delta']:+.3f} "
                      f"interval [{d['interval'][0]:+.3f},{d['interval'][1]:+.3f}] "
                      f"{'BEATS INCUMBENT' if d['beats_incumbent'] else 'no'}")
        out["_stop_rule"] = deltas
        # A verdict is only meaningful if trained arms exist AND their positive
        # controls passed. Without both, the correct output is "inconclusive" --
        # printing "the null survives" from a broken or empty run would be the
        # worst failure mode this script has.
        trained = [nm for nm in arms if nm != "frozen"]
        gated_ok = [nm for nm in trained if gate.get(nm, {}).get("encoder_trained")]
        if not trained:
            out["_verdict"] = ("INCONCLUSIVE -- no trained arm was scored (only the "
                               "frozen incumbent was present). Run run_ssl.py first.")
        elif not gated_ok:
            out["_verdict"] = (f"INCONCLUSIVE -- {len(trained)} trained arm(s) scored but "
                               f"none passed the positive-control gate (image->BMI), so "
                               f"their nulls carry no information about the data.")
        elif any(deltas[k]["beats_incumbent"] for k in deltas
                 if k.split("|")[0] in gated_ok):
            out["_verdict"] = "SSL/fine-tuning BEATS the frozen incumbent -- null overturned"
        else:
            out["_verdict"] = (f"no arm beats the frozen incumbent on E2 or E4 "
                               f"(gate-passing arms: {gated_ok}) -- the multimodal null "
                               f"survives a cohort-trained encoder")
        out["_arms_passing_gate"] = gated_ok
        print(f"\n[verdict] {out['_verdict']}")

    ps, lbl = [], []
    for nm in arms:
        for e, r in out[nm].items():
            if not e.startswith("_") and isinstance(r, dict) and "p" in r:
                ps.append(r["p"])
                lbl.append(f"{nm}|{e}")
    if ps:
        rej, q = benjamini_hochberg(ps, q=0.10)
        out["_bh"] = {l: dict(p=float(p), q=float(qq), reject=bool(r))
                      for l, p, qq, r in zip(lbl, ps, q, rej)}

    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
