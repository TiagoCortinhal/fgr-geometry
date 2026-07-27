"""
LGA false-positive recovery via the appearance-age lag.

Question: a logistic-regression model (external CSV) predicts LGA from biometry-derived
features (B, C). Some of its positives are false positives (baby is not LGA at birth).
Do those fetuses *look* their age on ultrasound (recoverable) or look genuinely old (missed)?

Pipeline (all GA-blind to the LGA label — projection, never a retrain on CSV labels):
  1. GA dating from CiTUS master (FUReco primary, LMP fallback) -> per-image GA.
  2. Main GA clock: USFM block-6 mean embedding -> GA, Ridge, out-of-fold GroupKFold.
     Per-plane clocks (abdominal / cerebral / femur) trained the same way.
  3. Appearance-age lag = predicted GA - true GA, averaged per fetus (pooled + per plane).
  4. Project the CSV fetuses onto the lag. Recovery threshold = lowest lag among true LGA
     (so no true LGA is ever misclassified). A predicted-LGA fetus with lag below the
     threshold "looks too young to be LGA" -> recovered false positive.

Key findings (retrained CiTUS-dated clock, 948 fetuses / 55,051 images, clock r=0.847):
  - 18/42 false positives recovered at the zero-TP-loss threshold (+0.11 wk); relaxing the
    threshold recovers more but starts losing true LGA (dial, not a fixed point).
  - The recoverable signal is ABDOMINAL: true LGA vs missed FP gap +0.60 wk on the abdominal
    clock, ~0 on cerebral/femur. LGA is an abdominal/soft-tissue phenotype.
  - The ~24 "missed" FP are mostly AGA babies that genuinely look advanced (cerebral lag as
    high as true LGA); they are false positives only against birth weight, not appearance.
  - Combining plane clocks as model features OVERFITS the ~8 true LGA (CV AUC < abdominal
    alone); use the single abdominal projection, not a multi-plane classifier.

Depends on recompute_ga_retrain.py having produced:
  results/img_align/_fetus_lag_recompute.csv   (fid, lag_pooled, lag_abdominal/cerebral/femur)
Inputs:
  --csv  logistic-regression CSV with columns Cod, B, C, A_real, A_predicho, prob_A1, acierto
         (A_real = is-LGA ground truth, A_predicho = model prediction)
Outputs (results/img_align/):
  _recovery_excel_data.csv     per-CSV-fetus recovery table (also feeds lag_recovery_CSV.xlsx)
  lga_recovery_summary.json    counts, threshold, per-plane gaps, threshold sweep
"""
import json, argparse
import numpy as np, pandas as pd

ROOT = "/Users/tiago/dev/fgr-geometry"
IMG = f"{ROOT}/results/img_align"
GA_MIN, GA_MAX = 6.0, 42.0

# Raw IMPACT folders present on USB but never preprocessed -> embedded.
# NOTE: several of these codes (749, 310, 583, 611) ALSO have clinical images that WERE
# embedded, so they recover a valid lag from those; their raw IMPACT folder is a separate,
# additional un-processed set. The n_raw_not_preprocessed column below is therefore only
# populated when raw-not-preprocessed is the *operative* reason a fetus has no lag
# (i.e. it has no valid lag AND no usable embedded images) — otherwise it is left 0 so the
# diagnostic column never contradicts the final status.
RAW_FOLDER_INVENTORY = {749: 49, 629: 44, 758: 36, 536: 29, 611: 71, 310: 33, 583: 35}


def build_recovery_table(csv_path, lag_path=f"{IMG}/_fetus_lag_recompute.csv",
                         embed_path=f"{IMG}/emb_usfm_multilayer.npz",
                         grp_path=f"{IMG}/_fetus_lag_crosssec.csv",
                         citus_path=f"{IMG}/_citus_dates.csv"):
    lag = pd.read_csv(lag_path)
    lr = pd.read_csv(csv_path)
    grp = pd.read_csv(grp_path)[["fid", "grp"]].rename(columns={"grp": "outcome_group"})
    cit = pd.read_csv(citus_path)
    fur_cod = set(cit.dropna(subset=["FUReco"]).Cod.astype(int))
    cit_cod = set(cit.Cod.astype(int))

    z = np.load(embed_path, allow_pickle=True)
    n_embed = pd.to_numeric(pd.Series(z["fetus_id"]), errors="coerce").astype("Int64").value_counts()

    m = (lr.merge(lag, left_on="Cod", right_on="fid", how="left")
           .merge(grp, left_on="Cod", right_on="fid", how="left"))
    m["n_embedded_images"] = m.Cod.map(n_embed).fillna(0).astype(int)
    # raw folder count is only the operative explanation when there is no valid lag AND
    # no usable embedded images; otherwise leave 0 (the fetus recovered via embedded images).
    raw_inv = m.Cod.map(RAW_FOLDER_INVENTORY).fillna(0).astype(int)
    m["n_raw_not_preprocessed"] = np.where(
        m.lag_pooled.isna() & (m.n_embedded_images == 0), raw_inv, 0).astype(int)
    m["in_citus"] = m.Cod.isin(cit_cod)
    m["has_FUReco"] = m.Cod.isin(fur_cod)

    TP = m[(m.A_real == 1) & (m.A_predicho == 1)]
    thr = TP.lag_pooled.min()

    def verdict(r):
        if np.isfinite(r.lag_pooled):
            if r.A_real == 1 and r.A_predicho == 1: return "true LGA (correct)"
            if r.A_real == 1 and r.A_predicho == 0: return "true LGA (model missed)"
            if r.A_real == 0 and r.A_predicho == 1:
                return ("false positive - RECOVERED by lag" if r.lag_pooled < thr
                        else "false positive - missed (looks old)")
            return "true negative (correct)"
        if r.n_embedded_images > 0:
            return "embedded but GA out-of-range (wrong-pregnancy scans)"
        if r.n_raw_not_preprocessed > 0:
            return "raw images exist - NOT preprocessed/embedded"
        return "no images found anywhere"

    m["status"] = m.apply(verdict, axis=1)
    cols = ["Cod", "B", "C", "A_real", "A_predicho", "prob_A1", "acierto", "outcome_group",
            "n_embedded_images", "n_raw_not_preprocessed", "in_citus", "has_FUReco",
            "lag_pooled", "lag_abdominal", "lag_cerebral", "lag_femur", "status"]
    out = m[cols].rename(columns={
        "A_real": "is_LGA_true", "A_predicho": "model_pred_LGA", "prob_A1": "model_prob_LGA",
        "acierto": "model_correct", "lag_pooled": "appearance_lag_pooled_wk",
        "lag_abdominal": "lag_abdominal_wk", "lag_cerebral": "lag_cerebral_wk",
        "lag_femur": "lag_femur_wk"})
    return out.sort_values(["status", "appearance_lag_pooled_wk"]).round(3), float(thr)


def threshold_sweep(out, thresholds=None):
    """Recovery count vs true-LGA loss across pooled and abdominal clocks."""
    d = out[out.appearance_lag_pooled_wk.notna()]
    TP = d[(d.is_LGA_true == 1) & (d.model_pred_LGA == 1)]
    FP = d[(d.is_LGA_true == 0) & (d.model_pred_LGA == 1)]
    if thresholds is None:
        thresholds = [TP.appearance_lag_pooled_wk.min(), 0.0, 0.3, 0.5, 0.74, 0.99, 1.2]
    sweep = {}
    for clock, col in [("pooled", "appearance_lag_pooled_wk"), ("abdominal", "lag_abdominal_wk")]:
        tp, fp = TP[col].dropna(), FP[col].dropna()
        sweep[clock] = [{"thr": round(float(t), 3),
                         "recovered": int((fp < t).sum()), "n_fp": int(len(fp)),
                         "true_lga_lost": int((tp < t).sum()), "n_tp": int(len(tp))}
                        for t in thresholds]
    return sweep


def analyze(csv_path):
    out, thr = build_recovery_table(csv_path)
    out.to_csv(f"{IMG}/_recovery_excel_data.csv", index=False)
    d = out[out.appearance_lag_pooled_wk.notna()]
    TP = d[(d.is_LGA_true == 1) & (d.model_pred_LGA == 1)]
    FP = d[(d.is_LGA_true == 0) & (d.model_pred_LGA == 1)]
    missed = FP[FP.appearance_lag_pooled_wk >= thr]
    summary = {
        "recovery_threshold_wk": round(thr, 3),
        "n_true_lga": int(len(TP)), "n_false_pos": int(len(FP)),
        "recovered": int((FP.appearance_lag_pooled_wk < thr).sum()),
        "missed": int(len(missed)),
        "status_counts": out.status.value_counts().to_dict(),
        "perplane_gap_trueLGA_minus_missed": {
            p: round(float(TP[c].mean() - missed[c].mean()), 3)
            for p, c in [("pooled", "appearance_lag_pooled_wk"), ("abdominal", "lag_abdominal_wk"),
                         ("cerebral", "lag_cerebral_wk"), ("femur", "lag_femur_wk")]},
        "missed_fp_outcome_groups": missed.outcome_group.value_counts().to_dict(),
        "threshold_sweep": threshold_sweep(out),
    }
    json.dump(summary, open(f"{IMG}/lga_recovery_summary.json", "w"), indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{IMG}/_lr_csv.csv",
                    help="logistic-regression LGA CSV (Cod,B,C,A_real,A_predicho,prob_A1,acierto)")
    args = ap.parse_args()
    print(json.dumps(analyze(args.csv), indent=2))
