"""Frame dataset + fetus-level splitting for the fetal-ultrasound SSL runs.

THE CRITICAL INVARIANT: every split in this package is BY FETUS, never by frame.
A fetus contributes ~22 frames; splitting at frame level puts near-duplicate
images of the same fetus in train and test and manufactures a positive result.
`fetus_level_folds` is the only splitter used anywhere in this codebase.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FrameManifest:
    """Resolves fetus_id -> list of image paths, with an optional cohort filter.

    manifest_csv needs columns: new_filename, fetus_id, dataset_type.

    THE TWO COHORTS LIVE IN DIFFERENT DIRECTORIES, so `roots` is a dict
    {dataset_type: path}, e.g. {"impact": /.../impact, "clinical": /.../clinical}.
    Passing a bare string is accepted and treated as the root for every selected
    dataset_type.

    The clinical set carries non-fetal frames (transvaginal, gynaecological,
    Doppler traces) at roughly 15%; pass `keep_csv` (the fetal-gate output) to
    drop them. Pretraining on those frames would spend capacity on anatomy that
    is not the fetus.
    """

    def __init__(self, manifest_csv, roots, dataset_type="impact",
                 keep_csv=None, keep_col="keep_fetal"):
        m = pd.read_csv(manifest_csv)
        if isinstance(dataset_type, str):
            dataset_type = [dataset_type]
        if dataset_type is not None:
            m = m[m.dataset_type.astype(str).isin([str(d) for d in dataset_type])]
        if keep_csv is not None and os.path.exists(keep_csv):
            k = pd.read_csv(keep_csv)
            if keep_col in k.columns and "new_filename" in k.columns:
                rejected = set(k.loc[~k[keep_col].astype(bool),
                                     "new_filename"].astype(str))
                m = m[~m.new_filename.astype(str).isin(rejected)]
        m = m.copy()
        m["fid"] = pd.to_numeric(m.fetus_id, errors="coerce")
        m = m[np.isfinite(m.fid)]
        if isinstance(roots, str):
            roots = {str(d): roots for d in (dataset_type or ["impact"])}
        missing = set(m.dataset_type.astype(str)) - set(roots)
        assert not missing, f"no --image-root given for dataset_type(s): {missing}"
        m["path"] = [os.path.join(roots[str(d)], str(f))
                     for d, f in zip(m.dataset_type, m.new_filename)]
        self.df = m.reset_index(drop=True)

    def counts(self):
        return {str(d): dict(frames=int(len(g)), fetuses=int(g.fid.nunique()))
                for d, g in self.df.groupby("dataset_type")}

    EXTS = ("", ".png", ".jpg", ".jpeg", ".PNG", ".tif", ".tiff")

    def existing(self, exts=None):
        """Drop rows whose file is not on disk, resolving the extension.

        The manifest stores `new_filename` WITHOUT an extension
        (`IMP0324_20171030_1.2.276...`) while the frames on disk are `.png`.
        Joining the bare name resolves nothing, so we probe a small set of
        extensions, decide the winner from a sample, then apply it to all rows
        (one stat per row rather than len(EXTS) stats per row).
        """
        exts = list(exts or self.EXTS)
        if len(self.df) == 0:
            return self
        # decide the extension per dataset_type from a sample
        chosen = {}
        for ds, g in self.df.groupby("dataset_type"):
            sample = g.path.head(40).tolist()
            best, hits = "", 0
            for e in exts:
                h = sum(os.path.exists(p + e) for p in sample)
                if h > hits:
                    best, hits = e, h
                if hits == len(sample):
                    break
            chosen[str(ds)] = best
        self.resolved_ext = chosen
        self.df = self.df.copy()
        self.df["path"] = [p + chosen.get(str(d), "")
                           for p, d in zip(self.df.path, self.df.dataset_type)]
        keep = [os.path.exists(p) for p in self.df.path]
        self.df = self.df[np.array(keep)].reset_index(drop=True)
        return self

    def diagnose(self, n=3):
        """When nothing resolves: what does the manifest expect vs what is there?"""
        out = {}
        for ds, g in self.df.groupby("dataset_type"):
            root = os.path.dirname(g.path.iloc[0])
            try:
                on_disk = sorted(os.listdir(root))[:n]
            except OSError as e:
                on_disk = [f"<cannot list: {e}>"]
            out[str(ds)] = dict(root=root,
                                manifest_expects=[os.path.basename(p)
                                                  for p in g.path.head(n)],
                                found_on_disk=on_disk)
        return out

    def by_fetus(self, namespaced=False):
        """fetus key -> paths.

        namespaced=True keys as "<dataset_type>:<fid>". USE IT whenever the two
        cohorts are combined: their fetus_id numbering is independent, so a
        clinical fetus 293 and an IMPACT fetus 293 would otherwise be merged into
        one "fetus" -- which would silently corrupt contrastive training (frames
        from two different pregnancies treated as positives) and pool unrelated
        images together.
        """
        if namespaced:
            k = (self.df.dataset_type.astype(str) + ":" +
                 self.df.fid.astype(int).astype(str))
            return {str(key): list(g.path) for key, g in self.df.groupby(k)}
        return {int(f): list(g.path) for f, g in self.df.groupby("fid")}

    def id_collision(self):
        """How many fetus_ids appear in more than one cohort? Diagnostic."""
        s = self.df.groupby("fid").dataset_type.nunique()
        return int((s > 1).sum())


def fetus_level_folds(fids, n_folds=5, seed=0):
    """Yield (train_fids, test_fids). THE ONLY SPLITTER IN THIS PACKAGE.

    Splitting by frame instead of fetus is the one bug that would invalidate
    every number this code produces -- near-duplicate frames of the same fetus
    would appear on both sides.
    """
    u = np.array(sorted({int(f) for f in fids}))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    parts = np.array_split(perm, n_folds)
    for i in range(n_folds):
        te = set(u[parts[i]].tolist())
        tr = [int(x) for x in u if int(x) not in te]
        yield tr, sorted(te)


def load_gray(path, size=224):
    im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


class FrameDataset(Dataset):
    """Single frames. mode='mae' returns one view; mode='pair' returns two frames
    of the SAME fetus (contrastive positives)."""

    def __init__(self, paths_by_fetus, fids, size=224, mode="mae", augment=True):
        """fids may be ints (IMPACT-only) or namespaced strings like
        "clinical:293" when the two cohorts are pooled for pretraining.

        Keys are kept as given -- never coerced to int -- because a clinical and
        an IMPACT fetus can share a numeric id, and merging them would treat
        frames from two different pregnancies as the same fetus.
        """
        self.size, self.mode, self.augment = size, mode, augment
        self.items = []
        self.byf = {}
        for f in fids:
            key = f if f in paths_by_fetus else (
                int(f) if not isinstance(f, str) and int(f) in paths_by_fetus else None)
            if key is None:
                continue
            ps = paths_by_fetus[key]
            if not ps:
                continue
            self.byf[key] = ps
            for p in ps:
                self.items.append((key, p))
        self.fids = sorted(self.byf, key=str)
        # integer group codes for the supervised arm's per-fetus pooling
        self._code = {k: i for i, k in enumerate(self.fids)}

    def __len__(self):
        return len(self.items) if self.mode == "mae" else len(self.fids)

    def _aug(self, a):
        if not self.augment:
            return a
        if np.random.rand() < 0.5:
            a = a[:, ::-1].copy()
        g = 0.8 + 0.4 * np.random.rand()
        a = np.clip(a ** g, 0, 1)
        s = int(0.9 * self.size)
        if s < self.size:
            i = np.random.randint(0, self.size - s)
            j = np.random.randint(0, self.size - s)
            a = np.asarray(Image.fromarray((a[i:i + s, j:j + s] * 255).astype(np.uint8))
                           .resize((self.size, self.size), Image.BILINEAR),
                           dtype=np.float32) / 255.0
        return a

    def __getitem__(self, i):
        if self.mode == "mae":
            f, p = self.items[i]
            a = self._aug(load_gray(p, self.size))
            return torch.from_numpy(a)[None], f
        f = self.fids[i]
        ps = self.byf[f]
        j, k = np.random.choice(len(ps), 2, replace=len(ps) < 2)
        a = self._aug(load_gray(ps[j], self.size))
        b = self._aug(load_gray(ps[k], self.size))
        return torch.from_numpy(a)[None], torch.from_numpy(b)[None], f


def build_frame_cache(paths, cache_path, size=224, log_every=2000):
    """Decode every PNG ONCE into a uint8 memmap; epochs then read from RAM.

    The binding constraint on a CPU-starved node is PNG decode, not GPU compute:
    with one dataloader worker, 17k frames/epoch takes ~158 s while the network
    itself needs ~5 s. Decoding once and reusing the array removes that entirely
    -- 21k frames at 224x224 uint8 is ~1.1 GB, which fits comfortably in RAM.

    Returns (memmap, index) where index maps path -> row.
    """
    import numpy as np
    n = len(paths)
    arr = np.lib.format.open_memmap(cache_path, mode="w+", dtype="uint8",
                                    shape=(n, size, size))
    for i, p in enumerate(paths):
        arr[i] = (load_gray(p, size) * 255).astype("uint8")
        if log_every and i % log_every == 0:
            print(f"  [cache] {i}/{n}", flush=True)
    arr.flush()
    return arr, {p: i for i, p in enumerate(paths)}


class CachedFrameDataset(FrameDataset):
    """FrameDataset backed by a decoded uint8 cache instead of PNG files.

    Identical semantics and identical augmentation -- only the source of pixels
    changes, so a run with and without the cache is the same experiment.
    """

    def __init__(self, paths_by_fetus, fids, cache, index, size=224,
                 mode="mae", augment=True):
        super().__init__(paths_by_fetus, fids, size=size, mode=mode,
                         augment=augment)
        self.cache, self.index = cache, index

    def _load(self, path):
        import numpy as np
        i = self.index.get(path)
        if i is None:
            return load_gray(path, self.size)
        return self.cache[i].astype("float32") / 255.0

    def __getitem__(self, i):
        import numpy as np
        import torch
        if self.mode == "mae":
            f, p = self.items[i]
            return torch.from_numpy(self._aug(self._load(p)))[None], f
        f = self.fids[i]
        ps = self.byf[f]
        j, k = np.random.choice(len(ps), 2, replace=len(ps) < 2)
        return (torch.from_numpy(self._aug(self._load(ps[j])))[None],
                torch.from_numpy(self._aug(self._load(ps[k])))[None], f)
