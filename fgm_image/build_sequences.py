"""
Build per-fetus sequences for the temporal GRU-VAEs (biometry, image, merged).

Outputs (written to results/img_align/):
  _biom_seq.npz    biometry visit sequences   X (N,4,11), L, S(doppler), fids
  _img_seq.npz     image GA-bin trajectories  X (N,8,34), L, fids
  _merged_seq.npz  fused GA-ordered timeline  X (N,12,44), L, fids, F, K

Feature layouts:
  biom  : [5 biom-z | 5 mask | GA_norm=(ga-26)/14]
  image : [32 PCA image-emb (per GA-week mean) | GA_norm=(ga-6)/36 | mask]
  merged: [5 biom-z | 5 mask | 32 img-emb | img_mask | GA_norm=(ga-6)/36]

The image embeddings are 32-d PCA of the 768-d USFM vectors (cohort-fit).
Requires emb_usfm_all.npz + image_clusters.csv + the tabular CSVs.
"""
import numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA = "/Users/tiago/PythonProject/fetal_growth_mechanism/data"
IMG = "/Users/tiago/PythonProject/fgr-geometry/results/img_align"
BIOM = ["ac_z_ig21", "hc_z_ig21", "bpd_z_ig21", "fl_z_ig21", "efw_z_ig21"]
VO = {"20s": 0, "28s": 1, "32s": 2, "eco": 3}
F, K = 5, 32


def _image_pca32():
    z = np.load(f"{IMG}/emb_usfm_all.npz", allow_pickle=True)
    E = z["emb"].astype(np.float32)
    meta = pd.read_csv(f"{IMG}/image_clusters.csv", low_memory=False)
    coh = meta.in_cohort.values
    sc = StandardScaler().fit(E[coh])
    pca = PCA(32, random_state=0).fit(sc.transform(E[coh]))
    Ep = pca.transform(sc.transform(E)).astype(np.float32)
    return Ep, meta, coh


def build_image_seq(MAXT=8):
    Ep, meta, coh = _image_pca32()
    sel = coh & meta.ga_weeks_recovered.between(6, 42).values
    img = meta[sel].copy()
    img["fid"] = img.fetus_id.astype(int)
    img["idx"] = np.where(sel)[0]
    img["gab"] = img.ga_weeks_recovered.round().astype(int)
    fids = sorted(img.fid.unique())
    N = len(fids); fi = {f: i for i, f in enumerate(fids)}
    X = np.zeros((N, MAXT, K + 2), np.float32); L = np.zeros(N, int)
    GA0, GAsc = 6.0, 36.0
    for f, sub in img.groupby("fid"):
        binmean = sub.groupby("gab").apply(lambda d: Ep[d.idx.values].mean(0), include_groups=False)
        bins = sorted(binmean.index)
        if len(bins) > MAXT:
            top = set(sub.gab.value_counts().index[:MAXT]); bins = sorted(b for b in bins if b in top)
        i = fi[f]; L[i] = len(bins)
        for t, b in enumerate(bins):
            X[i, t, :K] = binmean[b]; X[i, t, K] = (b - GA0) / GAsc; X[i, t, K + 1] = 1.0
    np.savez(f"{IMG}/_img_seq.npz", X=X, L=L, fids=np.array(fids), K=K, MAXT=MAXT)
    return X, L, np.array(fids)


def build_biom_seq():
    vl = pd.read_csv(f"{DATA}/visits_long.csv")
    vz = pd.read_csv(f"{DATA}/visits_long_z.csv").set_index(["fetus_id", "visit"])
    gaw = vl.set_index(["fetus_id", "visit"]).ga_weeks
    present = set(zip(vl.fetus_id, vl.visit))
    fids = sorted(vl.fetus_id.unique()); fi = {f: i for i, f in enumerate(fids)}
    N = len(fids); X = np.zeros((N, 4, 2 * F + 1), np.float32); L = np.zeros(N, int)
    for f in fids:
        i = fi[f]; maxt = 0
        for v, t in VO.items():
            if (f, v) not in present:
                continue
            maxt = max(maxt, t + 1)
            try: vals = vz.loc[(f, v), BIOM].values.astype(float)
            except Exception: vals = np.full(F, np.nan)
            m = np.isfinite(vals)
            X[i, t, :F] = np.nan_to_num(vals); X[i, t, F:2 * F] = m
            g = gaw.get((f, v), np.nan); X[i, t, 2 * F] = (g - 26.0) / 14.0 if np.isfinite(g) else 0.0
        L[i] = max(1, maxt)
    feat = pd.read_csv(f"{DATA}/impact_features.csv").set_index("Cod")
    dop = [c for c in feat.columns if c.startswith("Percentil_")][:6]
    S = np.nan_to_num(feat.reindex(fids)[dop].values.astype(np.float32), nan=0.0)
    S = (S - S.mean(0)) / (S.std(0) + 1e-6)
    np.savez(f"{IMG}/_biom_seq.npz", X=X, L=L, S=S, fids=np.array(fids))
    return X, L, S, np.array(fids)


def build_merged_seq(MAXT=12):
    Ep, meta, coh = _image_pca32()
    sel = coh & meta.ga_weeks_recovered.between(6, 42).values
    img = meta[sel].copy(); img["fid"] = img.fetus_id.astype(int)
    img["idx"] = np.where(sel)[0]; img["gab"] = img.ga_weeks_recovered.round().astype(int)
    img_bin = {(f, gb): Ep[sub.idx.values].mean(0) for (f, gb), sub in img.groupby(["fid", "gab"])}
    vl = pd.read_csv(f"{DATA}/visits_long.csv")
    vz = pd.read_csv(f"{DATA}/visits_long_z.csv").set_index(["fetus_id", "visit"])
    gaw = vl.set_index(["fetus_id", "visit"]).ga_weeks
    biom_pts = {}
    for (f, v) in set(zip(vl.fetus_id, vl.visit)):
        try: vals = vz.loc[(f, v), BIOM].values.astype(float)
        except Exception: vals = np.full(F, np.nan)
        g = gaw.get((f, v), np.nan)
        if np.isfinite(g):
            biom_pts[(f, int(round(g)))] = (np.nan_to_num(vals), np.isfinite(vals).astype(float))
    common = sorted(set(vl.fetus_id.unique()) & set(img.fid.unique()))
    Din = 2 * F + K + 1 + 1
    N = len(common); X = np.zeros((N, MAXT, Din), np.float32); L = np.zeros(N, int)
    fmap = {f: i for i, f in enumerate(common)}
    for f in common:
        i = fmap[f]
        gas = sorted({g for (ff, g) in biom_pts if ff == f} | {g for (ff, g) in img_bin if ff == f})
        if len(gas) > MAXT:
            gas = [gas[j] for j in np.linspace(0, len(gas) - 1, MAXT).astype(int)]
        for t, g in enumerate(gas):
            if (f, g) in biom_pts:
                zz, mm = biom_pts[(f, g)]; X[i, t, :F] = zz; X[i, t, F:2 * F] = mm
            if (f, g) in img_bin:
                X[i, t, 2 * F:2 * F + K] = img_bin[(f, g)]; X[i, t, 2 * F + K] = 1.0
            X[i, t, -1] = (g - 6.0) / 36.0
        L[i] = max(1, len(gas))
    np.savez(f"{IMG}/_merged_seq.npz", X=X, L=L, fids=np.array(common), F=F, K=K)
    return X, L, np.array(common)


if __name__ == "__main__":
    build_biom_seq(); build_image_seq(); build_merged_seq()
    print("built _biom_seq / _img_seq / _merged_seq .npz in", IMG)
