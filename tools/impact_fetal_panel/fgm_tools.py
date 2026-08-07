"""Helpers for the IMPACT fetal-growth multiblock panel (fgr-geometry / fetal_growth_mechanism)."""
import os
import sys

FGM_REPO = "/Users/tiago/dev/fetal_growth_mechanism"
FGR_REPO = "/Users/tiago/dev/fgr-geometry"
FGM_PY = "/Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python"
LAG_NPZ = "/Users/tiago/dev/fgr-geometry/results/img_align/_lag_seq_debiased.npz"
EMB_NPZ = "/Users/tiago/dev/fgr-geometry/results/img_align/emb_usfm_multilayer.npz"


def fgm_setup(repo=None):
    """Put the fgm package on sys.path. Call once per kernel before any other helper."""
    if repo is None:
        repo = FGM_REPO
    if repo not in sys.path:
        sys.path.insert(0, repo)
    cwd = os.getcwd()
    if cwd != repo:
        os.chdir(repo)
    return repo


def fgm_panel(standardise=True, repo=None):
    """Canonical 25-var panel with SENTINEL_Z outliers masked to NaN.

    Returns dict: X (raw), Z (standardised or None), cols, blocks, fids, mu, sd.
    This is the boilerplate that opens nearly every analysis cell.
    """
    import numpy as np
    fgm_setup(repo)
    from fgm.wp2_canonical import assemble_canonical
    from fgm.loadings_heatmap_wide import SENTINEL_Z
    X, cols, blocks, fids = assemble_canonical()[:4]
    cols = list(cols)
    Xc = X.copy()
    for i, c in enumerate(cols):
        if c.endswith("_z_ig21"):
            Xc[np.abs(Xc[:, i]) > SENTINEL_Z, i] = np.nan
    mu = np.nanmean(Xc, 0)
    sd = np.nanstd(Xc, 0)
    sd[sd == 0] = 1.0
    Z = (Xc - mu) / sd if standardise else None
    return dict(X=Xc, Z=Z, cols=cols, blocks=list(blocks),
                fids=[int(f) for f in fids], mu=mu, sd=sd)


def fgm_image_lag(fids, path=None):
    """Per-fetus mean maturation lag + image count k, aligned to `fids`.

    Reliability caveat: pooled one-way ICC is invalid (mixes single-image
    fetuses, lag var 0.98, with repeated ones, 6.54). Use ICC ~0.06 from the
    >=4-image subset, or split-half reliability 0.27-0.53. k drives precision.
    """
    import numpy as np
    import warnings
    if path is None:
        path = LAG_NPZ
    z = np.load(path, allow_pickle=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = np.nanmean(np.where(z["lag_mask"], z["lag_seq"], np.nan), axis=1)
    kc = z["lag_mask"].sum(1)
    zf = [int(x) for x in z["fids"]]
    lut = dict(zip(zf, pooled))
    kut = dict(zip(zf, kc))
    lag = np.array([lut.get(int(f), np.nan) for f in fids])
    k = np.array([kut.get(int(f), 0) for f in fids])
    return lag, k


def fgm_image_pcs(fids, n_pc=5, layer="emb_l5", per_visit=False, path=None):
    """Per-fetus (or per-visit) pooled USFM embedding reduced to n_pc PCs.

    per_visit=True groups by (fetus, study_date) and carries GA -- but note
    98.5% of IMPACT fetuses have exactly ONE image study-date, so per-visit
    PCs cannot form a trajectory. PC3 correlates ~0.34 with maternal BMI
    (acquisition condition, not physiology): residualise before interpreting.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    if path is None:
        path = EMB_NPZ
    z = np.load(path, allow_pickle=True)
    imp = pd.Series(z["dataset_type"]).astype(str).values == "impact"
    E = z[layer].astype("float32")[imp]
    fi = pd.to_numeric(pd.Series(z["fetus_id"]).astype(str), errors="coerce").values[imp]
    ok = np.isfinite(fi)
    if per_visit:
        ga = pd.to_numeric(pd.Series(z["ga_weeks_recovered"]).astype(str), errors="coerce").values[imp]
        sd_ = pd.Series(z["study_date"]).astype(str).values[imp]
        ok = ok & np.isfinite(ga) & (ga > 10) & (ga < 45)
        P = PCA(n_pc, random_state=0).fit(E[ok])
        V = pd.DataFrame(P.transform(E[ok]), columns=["PC%d" % (i + 1) for i in range(n_pc)])
        V["fid"] = fi[ok].astype(int)
        V["ga"] = ga[ok]
        V["sdate"] = sd_[ok]
        agg = {c: "mean" for c in V.columns if c.startswith("PC")}
        agg["ga"] = "mean"
        G = V.groupby(["fid", "sdate"]).agg(agg)
        G["k"] = V.groupby(["fid", "sdate"]).size()
        return G.reset_index(), P
    df = pd.DataFrame(E[ok])
    df["fid"] = fi[ok].astype(int)
    pf = df.groupby("fid").mean()
    P = PCA(n_pc, random_state=0).fit(pf.values)
    S = P.transform(pf.values)
    lut = dict(zip([int(x) for x in pf.index], S))
    return np.array([lut.get(int(f), [np.nan] * n_pc) for f in fids]), P


def fgm_cv_r2(y, A, seed=0, folds=5):
    """Out-of-fold ridge R2 of y from A. NaNs in A are zero-filled, y rows dropped.

    Never bootstrap-resample inside CV to get an interval -- duplicated rows
    leak train into test. Use the spread over independent CV splits instead.
    """
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    m = np.isfinite(y)
    yy = np.asarray(y)[m]
    AA = np.where(np.isfinite(A[m]), A[m], 0.0)
    p = np.zeros_like(yy, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(AA):
        p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(AA[tr], yy[tr]).predict(AA[te])
    return 1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def fgm_auc(score, label):
    """AUC = P(score | label=1 > score | label=0). Mann-Whitney U / (n1*n0).

    Do NOT write `1 - U/(n1*n0)` -- U is already P(a>b) and the complement
    silently inverts every value. This bit me once; the tell was every AUC
    landing below 0.5 including EFW-z for smallness.
    """
    import numpy as np
    from scipy.stats import mannwhitneyu
    s = np.asarray(score, dtype=float)
    y = np.asarray(label)
    m = np.isfinite(s) & np.isfinite(y)
    a = s[m][y[m] == 1]
    b = s[m][y[m] == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(mannwhitneyu(a, b).statistic / (len(a) * len(b)))


def fgm_block_shuffle_null(Z, blocks, stat_fn, n=200, seed=0):
    """Permute ROWS WITHIN each block independently, preserving within-block
    structure and missingness while destroying cross-block covariance.

    Permuting columns independently is the WRONG null -- it destroys
    within-block structure too and makes the test trivially easy to pass.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Zs = Z.copy()
        for g in blocks:
            Zs[:, g] = Zs[rng.permutation(len(Zs))][:, g]
        out.append(stat_fn(Zs))
    return np.array(out)


def fgm_o_information(Z, groups):
    """Target-free redundancy/synergy split (Rosas 2019), Gaussian closed form.

    Omega > 0 redundancy-dominated, Omega < 0 synergy-dominated, ~0 neither.
    Returns (TC, DTC, Omega). Validate with planted controls: independent ~0,
    four noisy copies of one factor > 0, sum-of-three < 0.
    """
    import numpy as np
    C = np.cov(Z, rowvar=False)

    def ent(idx):
        s, ld = np.linalg.slogdet(C[np.ix_(idx, idx)])
        return 0.5 * ld if s > 0 else float("nan")

    allidx = [i for g in groups for i in g]
    hall = ent(allidx)
    hi = sum(ent(g) for g in groups)
    hmi = sum(ent([i for j, g in enumerate(groups) if j != k for i in g])
              for k in range(len(groups)))
    tc = hi - hall
    dtc = hmi - (len(groups) - 1) * hall
    return tc, dtc, tc - dtc


def fgm_positive_control(fit_predict, levels=None, n=977, seed=0):
    """Plant a known-strength nonlinear signal and check the method recovers it.

    fit_predict(X, y, train_idx, test_idx) -> predictions on test_idx.
    Gate every null on this. Measured floor for this cohort: nonlinear
    detection dies below R2 ~0.10 at n~883, so a threshold under that is
    inside the noise and a failed gate means "underpowered", not "absent".
    """
    import numpy as np
    from sklearn.model_selection import KFold
    if levels is None:
        levels = [0.30, 0.10]
    rng = np.random.default_rng(seed)
    out = {}
    for t in levels:
        X = rng.standard_normal((n, 6))
        sig = X[:, 0] * X[:, 1] + X[:, 2] ** 2
        sig = (sig - sig.mean()) / sig.std()
        y = np.sqrt(t) * sig + np.sqrt(1 - t) * rng.standard_normal(n)
        tr, te = next(KFold(5, shuffle=True, random_state=0).split(X))
        p = fit_predict(X, y, tr, te)
        out[t] = float(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return out


def fgm_run(script_path, background=False):
    """Return the bash command to run a script against the repo + fgrgeom env."""
    tail = " > /tmp/%s.log 2>&1 &" % os.path.basename(script_path).replace(".py", "")
    return "cd %s && PYTHONPATH=. %s %s%s" % (
        FGM_REPO, FGM_PY, script_path, tail if background else "")


def fgm_omega_null(Z, groups, nperm=200, seed=0):
    """O-information plus its block-shuffle null in one call.

    Returns dict with TC, DTC, Omega, null_lo, null_hi, verdict. The verdict
    compares Omega against the 95% null band, not against zero -- with a
    multiplet of singletons the null band is very tight (~1e-4) and almost any
    real data reads as significant, so report TC alongside Omega for scale.
    """
    tc, dtc, om = fgm_o_information(Z, groups)
    nl = fgm_block_shuffle_null(Z, groups,
                                lambda A: fgm_o_information(A, groups)[2],
                                n=nperm, seed=seed)
    import numpy as np
    lo = float(np.percentile(nl, 2.5))
    hi = float(np.percentile(nl, 97.5))
    verdict = "redundancy" if om > hi else "synergy" if om < lo else "inside null"
    return dict(TC=float(tc), DTC=float(dtc), Omega=float(om),
                null_lo=lo, null_hi=hi, verdict=verdict, n=int(len(Z)))


def fgm_all_tabular(repo=None, min_coverage=0.5, max_abs_corr=0.999):
    """EVERY numeric tabular variable in the IMPACT registry, not just the
    canonical 25-variable panel.

    Pulls data/IMPACT_merged_by_Cod.xlsx (the ~400-column master) plus the
    outcomes file, keeps numeric columns above min_coverage, drops constants
    and exact duplicates. Returns dict: Z (standardised), cols, fids, coverage.

    Beware: this registry stores many numerics as EUROPEAN-DECIMAL STRINGS
    ('25,97') -- to_numeric returns NaN for them. fgm_eurofloat handles it.
    """
    import numpy as np
    import pandas as pd
    fgm_setup(repo)
    df = pd.read_excel("data/IMPACT_merged_by_Cod.xlsx")
    key = pd.to_numeric(df["Cod"], errors="coerce")
    num = {}
    for c in df.columns:
        if c == "Cod":
            continue
        v = fgm_eurofloat(df[c])
        if v is None:
            continue
        cov = np.isfinite(v).mean()
        if cov < min_coverage:
            continue
        if np.nanstd(v) == 0:
            continue
        num[str(c)] = v
    cols = list(num)
    X = np.column_stack([num[c] for c in cols])
    keep = []
    for j in range(X.shape[1]):
        dup = False
        for i in keep:
            m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if m.sum() > 10 and abs(np.corrcoef(X[m, i], X[m, j])[0, 1]) > max_abs_corr:
                dup = True
                break
        if not dup:
            keep.append(j)
    X = X[:, keep]
    cols = [cols[j] for j in keep]
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    sd[sd == 0] = 1.0
    return dict(Z=(X - mu) / sd, X=X, cols=cols,
                fids=[int(f) if np.isfinite(f) else -1 for f in key],
                coverage=np.isfinite(X).mean(0))


def fgm_eurofloat(series):
    """Parse a column that may store numbers as European-decimal strings.

    Returns a float array, or None if the column is not numeric-like. Only
    swaps ',' for '.' when the comma is acting as a DECIMAL separator -- blanket
    stripping corrupts genuine thousands separators. This defect silently
    discarded most values of a key column in this project once.
    """
    import numpy as np
    import pandas as pd
    v = pd.to_numeric(series, errors="coerce")
    if np.isfinite(v).mean() > 0.5:
        return v.to_numpy(dtype=float)
    s = series.astype(str).str.strip()
    looks_eu = s.str.match(r"^-?\d{1,3}(?:\.\d{3})*,\d+$|^-?\d+,\d+$").mean()
    if looks_eu < 0.3:
        return None
    s2 = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    v2 = pd.to_numeric(s2, errors="coerce")
    return v2.to_numpy(dtype=float) if np.isfinite(v2).mean() > 0.3 else None


def fgm_visit_matrix(measure="efw_z_ig21", repo=None):
    """Wide per-visit matrix for one longitudinal biometry measure.

    Rows are fetuses, columns are the visit nodes (20s / 28s / 32s / eco).
    Use for the longitudinal arm of an O-information test: visits of the SAME
    measure should read strongly REDUNDANT, which is the real-data positive
    control the simulated ones cannot provide.
    """
    import pandas as pd
    fgm_setup(repo)
    v = pd.read_csv("data/visits_long_z.csv")
    v["fid"] = pd.to_numeric(v.fetus_id, errors="coerce")
    return v.pivot_table(index="fid", columns="visit", values=measure, aggfunc="first")


def fgm_decorrelate(M, tol=1e-6):
    """Greedy rank filter: drop columns that are near-linear combinations of
    earlier ones, keeping the covariance positive-definite.

    MANDATORY before fgm_o_information on more than ~15 registry variables.
    The Gaussian entropy is a log-determinant, so a near-zero eigenvalue sends
    it to -inf and Omega to a large arbitrary NEGATIVE number that reads as
    spectacular synergy. On this registry the raw covariance hit condition
    number 3.8e12 with a negative eigenvalue at d=40, producing a spurious
    Omega of -31. Independent noise at the same n and d gives Omega ~ -0.001,
    so the estimator is sound and the data were rank-deficient.
    """
    import numpy as np
    keep = []
    for j in range(M.shape[1]):
        if not keep:
            keep.append(j)
            continue
        T = M[:, keep + [j]]
        if np.linalg.eigvalsh(np.cov(T, rowvar=False)).min() > tol:
            keep.append(j)
    return keep


def fgm_omega_report(Z, groups, nperm=100, check_rank=True):
    """fgm_omega_null plus the conditioning diagnostics you must report with it.

    Adds min_eig, cond and n_over_d. Treat any Omega whose min_eig is below
    ~1e-2, or whose n/d is under ~20, as not interpretable without shrinkage.
    """
    import numpy as np
    M = np.where(np.isfinite(Z), Z, 0.0)
    if check_rank:
        kp = fgm_decorrelate(M)
        dropped = M.shape[1] - len(kp)
        M = M[:, kp]
        groups = [[j] for j in range(M.shape[1])] if all(len(g) == 1 for g in groups) else groups
    else:
        dropped = 0
    ev = np.linalg.eigvalsh(np.cov(M, rowvar=False))
    r = fgm_omega_null(M, groups, nperm=nperm)
    r.update(min_eig=float(ev.min()), cond=float(ev.max() / max(ev.min(), 1e-12)),
             n_over_d=float(len(M) / M.shape[1]), dropped_rank_deficient=int(dropped),
             trustworthy=bool(ev.min() > 1e-2 and len(M) / M.shape[1] > 20))
    return r


def fgm_classify_registry_columns(cols):
    """Split registry column names into administrative vs clinical.

    Coverage-ranking the raw registry selects mostly ADMINISTRATIVE fields --
    on this cohort 50% of the top 40 were dates, identifiers or one-hot
    ethnicity dummies. One-hot sets sum to 1 by construction and visit dates
    are collinear with gestational age, so any redundancy verdict computed over
    them is clerical, not biological. Filter before interpreting.

    Returns dict with keys: dates, ids, onehot, clinical.
    """
    import re
    datepat = re.compile(r"fecha|date|_days$|^ga\d+_days|FUR", re.I)
    idpat = re.compile(r"^Cod|NHC|^Group$|Protocol", re.I)
    onehotpat = re.compile(r"^etnia", re.I)
    out = dict(dates=[], ids=[], onehot=[], clinical=[])
    for c in cols:
        if onehotpat.search(str(c)):
            out["onehot"].append(c)
        elif datepat.search(str(c)):
            out["dates"].append(c)
        elif idpat.search(str(c)):
            out["ids"].append(c)
        else:
            out["clinical"].append(c)
    return out


def fgm_canonical_block_vars():
    """The paper's 25-variable panel by block, with recorded coverage.

    growth 5, maternal 4, Doppler 5, cardiac 11. Percentil_Sapse at 0.478 is
    the coverage bottleneck -- one variable halves the complete-case count.
    """
    return {
        "growth": ["ac_z_ig21", "hc_z_ig21", "bpd_z_ig21", "fl_z_ig21", "efw_z_ig21"],
        "maternal": ["maternal_age", "maternal_height_cm", "maternal_weight_kg", "maternal_bmi"],
        "Doppler": ["Percentil_AU", "Percentil_UTA", "Percentil_ACM", "Percentil_CPR", "Percentil_DV"],
        "cardiac": ["Percentil_MPI", "Percentil_Tapse", "Percentil_Mapse", "Percentil_Sapse",
                    "Percentil_RV_longitudinal", "Percentil_LV_longitudinal",
                    "Percentil_RV_basal", "Percentil_LV_basal",
                    "Percentil_ICTms", "Percentil_ETms", "Percentil_IRTms"],
    }


def fgm_derived_variables():
    """Variables that are algebraic functions of other panel members.

    Each is exactly synergistic with its own inputs, so it dominates any
    within-block O-information. Refit without it before interpreting:
    maternal Omega goes -1.596 -> -0.0001 without BMI.
    """
    return {
        "efw_z_ig21": ["ac_z_ig21", "hc_z_ig21", "fl_z_ig21", "bpd_z_ig21"],
        "maternal_bmi": ["maternal_weight_kg", "maternal_height_cm"],
        "Percentil_MPI": ["Percentil_ICTms", "Percentil_IRTms", "Percentil_ETms"],
    }


ECHO_XLSX = "/Users/tiago/dev/fgr-geometry/data_local/IMPACT_ecocardio_zscores_corrected.xlsx"


def fgm_echo_raw(fids=None, path=None):
    """RAW cardiac-morphology parameters from the echo workbook.

    NOT the same variables as the canonical panel's 11 Percentil_* cardiac
    columns. The archived image<->cardiac finding (held-out cc 0.248) used the
    RAW morphology params (circumference, basal, longitudinal, LV_SI, RV_SI),
    and the percentile-scored panel columns do NOT reproduce it (0.093,
    p=0.066 at n=660). Always state which cardiac representation you used.

    Returns (array aligned to fids or in file order, list of column names).
    """
    import re
    import numpy as np
    import pandas as pd
    if path is None:
        path = ECHO_XLSX
    d = pd.read_excel(path)
    pat = re.compile(r"circunf|basal|long|LV_SI|RV_SI|septum|area", re.I)
    keep = []
    for c in d.columns:
        if not pat.search(str(c)):
            continue
        v = fgm_eurofloat(d[c])
        if v is None or np.isfinite(v).mean() < 0.3 or np.nanstd(v) == 0:
            continue
        keep.append((str(c), v))
    names = [k[0] for k in keep]
    X = np.column_stack([k[1] for k in keep]) if keep else np.zeros((len(d), 0))
    if fids is None:
        return X, names
    key = pd.to_numeric(d["Cod"], errors="coerce")
    lut = {}
    for i, f in enumerate(key):
        if np.isfinite(f):
            lut[int(f)] = X[i]
    out = np.full((len(fids), X.shape[1]), np.nan)
    for r, f in enumerate(fids):
        if int(f) in lut:
            out[r] = lut[int(f)]
    return out, names


def fgm_heldout_cca(X, Y, npc=10, folds=5, seed=0, ncomp=1):
    """Held-out canonical correlation: PCA + CCA fit on TRAIN, correlated on TEST.

    The image-side PCA MUST be fitted inside the fold -- fitting it on all rows
    leaks test structure and inflates the correlation. Returns the mean
    across folds. NaNs must be filled by the caller; a NaN anywhere sends
    lstsq into 'SVD did not converge'.
    """
    import numpy as np
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    out = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        p = PCA(min(npc, X.shape[1], len(tr) - 1), random_state=0).fit(X[tr])
        c = CCA(n_components=ncomp, max_iter=2000).fit(p.transform(X[tr]), Y[tr])
        a, b = c.transform(p.transform(X[te]), Y[te])
        out.append(np.corrcoef(a[:, 0], b[:, 0])[0, 1])
    return float(np.mean(out))


def fgm_residualise(X, covars):
    """Regress X on covars (list of 2-D arrays) and return residuals.

    Fill NaNs in BOTH X and covars first -- one NaN propagates through lstsq
    and raises 'SVD did not converge in Linear Least Squares'.
    """
    import numpy as np
    if not covars:
        return X
    A = np.column_stack([np.ones(len(X))] + [np.asarray(c).reshape(len(X), -1) for c in covars])
    return X - A @ np.linalg.lstsq(A, X, rcond=None)[0]


def fgm_ga_at_echo(fids, repo=None):
    """Gestational age at the echo visit, aligned to fids. Returns a float array.

    MANDATORY covariate for any image<->tabular alignment test. Image appearance
    encodes GA (pooled-USFM PC1 correlates -0.276 with it), so an unadjusted
    cross-modal correlation is a shared gestational-age channel, not a finding.
    """
    import numpy as np
    import pandas as pd
    fgm_setup(repo)
    v = pd.read_csv("data/visits_long_z.csv")
    v["fid"] = pd.to_numeric(v.fetus_id, errors="coerce")
    g = v[v.visit == "eco"].groupby("fid").ga_weeks.mean()
    return np.array([g.get(int(f), np.nan) for f in fids])


def fgm_crossmodal_ladder(IMG, Y, GA, EFW=None, BIO=None, npc=10, nperm=500, seed=0):
    """Image<->tabular alignment across an adjustment ladder, with a permutation null.

    Returns dict with the ladder (raw / GA / GA+size / GA+size+biometry), the
    permutation p on the fully-adjusted arm, and a bootstrap CI.

    Read the LADDER, never the raw value. A cross-modal correlation that falls
    on GA adjustment is a maturation channel; only one that survives (or
    strengthens) is a finding about the tabular block itself. All inputs are
    NaN-filled internally because one NaN makes lstsq fail to converge.
    """
    import numpy as np
    Xf = np.where(np.isfinite(IMG), IMG, 0.0)
    Yf = np.where(np.isfinite(Y), Y, 0.0)
    g = np.where(np.isfinite(GA), GA, np.nanmean(GA)).reshape(-1, 1)
    cov = [g]
    lad = {}
    lad["raw"] = fgm_heldout_cca(Xf, Yf, npc=npc, seed=seed)
    lad["GA"] = fgm_heldout_cca(fgm_residualise(Xf, cov), fgm_residualise(Yf, cov), npc=npc, seed=seed)
    if EFW is not None:
        cov = cov + [np.where(np.isfinite(EFW), EFW, 0.0).reshape(-1, 1)]
        lad["GA+size"] = fgm_heldout_cca(fgm_residualise(Xf, cov), fgm_residualise(Yf, cov),
                                         npc=npc, seed=seed)
    if BIO is not None:
        cov = cov + [np.where(np.isfinite(BIO), BIO, 0.0)]
        lad["GA+size+biometry"] = fgm_heldout_cca(fgm_residualise(Xf, cov), fgm_residualise(Yf, cov),
                                                  npc=npc, seed=seed)
    Xa = fgm_residualise(Xf, cov)
    Ya = fgm_residualise(Yf, cov)
    obs = fgm_heldout_cca(Xa, Ya, npc=npc, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(Xa)
    nl = np.array([fgm_heldout_cca(Xa, Ya[rng.permutation(n)], npc=npc, seed=seed)
                   for _ in range(nperm)])
    bs = np.array([fgm_heldout_cca(Xa[i], Ya[i], npc=npc, seed=seed)
                   for i in (rng.integers(0, n, n) for _ in range(150))])
    return dict(ladder=lad, adjusted=float(obs), n=int(n),
                p=float((1 + (nl >= obs).sum()) / (1 + len(nl))),
                null_p95=float(np.percentile(nl, 95)),
                ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                null=nl.tolist())


def fgm_ga_leakage(X, GA, names=None):
    """Per-column |correlation| with gestational age -- the confound audit.

    Run this on any tabular block BEFORE a cross-modal test. Unscaled
    measurements (cardiac circumference, chamber lengths in cm) carry GA at
    mean |r| ~0.25; percentile-scored columns carry ~0.04. A block in the first
    category will produce a large unadjusted cross-modal correlation that is
    pure maturation.
    """
    import numpy as np
    from scipy.stats import pearsonr
    out = []
    for j in range(X.shape[1]):
        m = np.isfinite(X[:, j]) & np.isfinite(GA)
        r = pearsonr(X[m, j], GA[m])[0] if m.sum() > 10 else np.nan
        out.append((names[j] if names else j, float(r)))
    return dict(per_column=out,
                mean_abs=float(np.nanmean([abs(v) for _, v in out])))


def fgm_registry_variables(fids, min_coverage=0.60, min_levels=3, drop_admin=True):
    """Every numeric REGISTRY variable aligned to fids, individually usable.

    Differs from fgm_all_tabular: aligns to a caller-supplied fetus order,
    drops administrative fields (dates, ids, postal codes, one-hot ethnicity)
    and near-binary flags, and returns a name->standardised-vector dict so each
    variable can be tested on its own rather than as part of a defined block.

    Registry scale: 1431 columns, 1013 numeric; ~200-600 survive depending on
    the coverage floor. Routes everything through fgm_eurofloat.
    """
    import re
    import numpy as np
    import pandas as pd
    fgm_setup()
    R = pd.read_excel("data/IMPACT_merged_by_Cod.xlsx")
    key = pd.to_numeric(R["Cod"], errors="coerce")
    pos = {}
    for i, f in enumerate(key):
        if np.isfinite(f):
            pos[int(f)] = i
    idx = np.array([pos.get(int(f), -1) for f in fids])
    adm = re.compile(r"fecha|date|_days$|FUR|^Cod|NHC|^Group|Protocol|^etnia|postal|LMP", re.I)
    out = {}
    for c in R.columns:
        if c == "Cod":
            continue
        if drop_admin and adm.search(str(c)):
            continue
        v = fgm_eurofloat(R[c])
        if v is None:
            continue
        a = np.array([v[i] if i >= 0 else np.nan for i in idx])
        ok = np.isfinite(a)
        if ok.mean() < min_coverage or np.nanstd(a) == 0:
            continue
        if len(np.unique(a[ok])) < min_levels:
            continue
        out[str(c)] = (a - np.nanmean(a)) / np.nanstd(a)
    return out


def fgm_image_screen(IMG, variables, COV, min_n=150, folds=5, seed=0, npc=10):
    """Screen images against MANY tabular variables one at a time.

    variables: dict name -> standardised vector (see fgm_registry_variables).
    COV: design matrix of nuisance covariates (intercept, GA, size, biometry) --
    BOTH sides are residualised on it inside the observed rows of each variable.

    Returns rows sorted by adjusted held-out correlation, each carrying its own
    GA leakage so a hit driven by maturation is visible immediately. Apply BH
    across the returned rows; screening hundreds of variables without it is
    the definition of p-hacking.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    rows = []
    for nm, y in variables.items():
        m = np.isfinite(y) & np.isfinite(IMG).all(1)
        if m.sum() < min_n:
            continue
        A = COV[m]
        yy = y[m]
        yy = yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]
        Xs = IMG[m]
        Xs = Xs - A @ np.linalg.lstsq(A, Xs, rcond=None)[0]
        p = np.zeros_like(yy)
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xs):
            pc = PCA(min(npc, Xs.shape[1], len(tr) - 1), random_state=0).fit(Xs[tr])
            p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
                pc.transform(Xs[tr]), yy[tr]).predict(pc.transform(Xs[te]))
        rows.append(dict(var=nm, r=float(np.corrcoef(p, yy)[0, 1]), n=int(m.sum())))
    rows.sort(key=lambda d: -d["r"])
    return rows


def fgm_bh(pvals, q=0.10):
    """Benjamini-Hochberg. Returns (rejected bool list, qvalues) in input order."""
    import numpy as np
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out_q = np.empty(n)
    out_q[order] = np.minimum(qv, 1.0)
    return (out_q <= q).tolist(), out_q.tolist()


def fgm_confound_ladder(y, IMG, BASE, extras, folds=5, seed=0, npc=10):
    """Add confounds one at a time and watch a cross-modal signal live or die.

    y: tabular target. IMG: image representation. BASE: nuisance design matrix
    (intercept, GA, size, biometry). extras: list of (label, array) appended
    cumulatively.

    This is the test that decides every image<->tabular hit in this cohort.
    Maternal blood pressure went 0.215 -> 0.046 on adding BMI alone; the whole
    maternal block went +0.384 -> -0.014. Report the LADDER, never one value.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    def _oof(yv, X, COV):
        m = np.isfinite(yv) & np.isfinite(X).all(1) & np.isfinite(COV).all(1)
        if m.sum() < 150:
            return float("nan"), int(m.sum())
        A = COV[m]
        yy = yv[m] - A @ np.linalg.lstsq(A, yv[m], rcond=None)[0]
        Xs = X[m] - A @ np.linalg.lstsq(A, X[m], rcond=None)[0]
        p = np.zeros_like(yy)
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Xs):
            pc = PCA(min(npc, Xs.shape[1], len(tr) - 1), random_state=0).fit(Xs[tr])
            p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
                pc.transform(Xs[tr]), yy[tr]).predict(pc.transform(Xs[te]))
        return float(np.corrcoef(p, yy)[0, 1]), int(m.sum())

    out = []
    COV = BASE
    r0, n0 = _oof(y, IMG, COV)
    out.append(dict(step="base", r=r0, n=n0))
    for lab, arr in extras:
        a = np.asarray(arr).reshape(len(y), -1)
        COV = np.column_stack([COV, np.where(np.isfinite(a), a, 0.0)])
        r_, n_ = _oof(y, IMG, COV)
        out.append(dict(step="+" + lab, r=r_, n=n_))
    return out


def fgm_nuisance_design(GA, EFW, BIO):
    """Standard nuisance design matrix: intercept, GA, size, biometry block.

    Every cross-modal test in this project residualises BOTH sides on this
    before reporting anything. NaNs are zero-filled here so lstsq converges.
    """
    import numpy as np
    n = len(GA)
    g = np.where(np.isfinite(GA), GA, np.nanmean(GA)).reshape(n, 1)
    e = np.where(np.isfinite(EFW), EFW, 0.0).reshape(n, 1)
    b = np.where(np.isfinite(BIO), BIO, 0.0).reshape(n, -1)
    return np.column_stack([np.ones(n), g, e, b])


def fgm_visit_deviations(measures=None, min_visits=3, repo=None, interpolate_only=True):
    """Per-visit deviation from each fetus's OWN growth line, leave-one-visit-out.

    A fetus cannot leave its own trajectory and rejoin it, so the deviation of a
    visit from a line fitted through that fetus's OTHER visits approximates
    MEASUREMENT ERROR rather than biology.

    TWO GUARDS, both learned the hard way:
    (1) DEGREE 1 ONLY. A quadratic through the 3 retained points of a 4-visit
        fetus fits them exactly and the held-out prediction becomes a pure
        extrapolation -- this produced deviations of 584 z-units on a z-score,
        driven by fetuses with two visits ~0.1 weeks apart.
    (2) INTERPOLATION ONLY (interpolate_only=True). The held-out GA must lie
        inside the retained GA range; extrapolating past the ends is where the
        remaining blow-ups live.

    Returns a DataFrame: fid, visit, ga_weeks, measure, value, fitted, deviation.
    """
    import numpy as np
    import pandas as pd
    if measures is None:
        measures = ["ac_z_ig21", "hc_z_ig21", "bpd_z_ig21", "fl_z_ig21"]
    fgm_setup(repo)
    v = pd.read_csv("data/visits_long_z.csv")
    v["fid"] = pd.to_numeric(v.fetus_id, errors="coerce")
    rows = []
    for fid, g in v.groupby("fid"):
        g = g.sort_values("ga_weeks")
        for meas in measures:
            sub = g[np.isfinite(g[meas]) & np.isfinite(g.ga_weeks)]
            if len(sub) < min_visits:
                continue
            ga = sub.ga_weeks.to_numpy(float)
            y = sub[meas].to_numpy(float)
            for i in range(len(sub)):
                keep = np.arange(len(sub)) != i
                gk = ga[keep]
                if interpolate_only and not (gk.min() <= ga[i] <= gk.max()):
                    continue
                if np.ptp(gk) < 1e-6:
                    continue
                c = np.polyfit(gk, y[keep], 1)
                fit = float(np.polyval(c, ga[i]))
                rows.append(dict(fid=int(fid), visit=str(sub.visit.iloc[i]),
                                 ga_weeks=float(ga[i]), measure=meas,
                                 value=float(y[i]), fitted=fit,
                                 deviation=float(y[i] - fit)))
    return pd.DataFrame(rows)


def fgm_precision_test(dev_df, IMG, fids, GA, BMI=None, nperm=400, seed=0):
    """Do images predict per-fetus MEASUREMENT NOISE (not measurement value)?

    dev_df: output of fgm_visit_deviations. Target is the per-fetus RMS deviation
    across visits and measures -- a scalar noise level per fetus.

    Reports the image->noise correlation adjusted for GA, then additionally for
    BMI. The BMI rung is the one that matters: maternal habitus sets ultrasound
    penetration, so images beating BMI is the only way this adds anything.
    Also returns the DIRECTION check -- predicted noise must rise with BMI.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr
    rms = dev_df.groupby("fid").deviation.apply(lambda s: float(np.sqrt((s ** 2).mean())))
    y = np.array([rms.get(int(f), np.nan) for f in fids])
    y = np.log1p(y)

    def _oof(target, COV, seed=seed):
        m = np.isfinite(target) & np.isfinite(IMG).all(1) & np.isfinite(COV).all(1)
        if m.sum() < 100:
            return float("nan"), int(m.sum()), None
        A = COV[m]
        yy = target[m] - A @ np.linalg.lstsq(A, target[m], rcond=None)[0]
        Xs = IMG[m] - A @ np.linalg.lstsq(A, IMG[m], rcond=None)[0]
        p = np.zeros_like(yy)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xs):
            pc = PCA(min(10, Xs.shape[1], len(tr) - 1), random_state=0).fit(Xs[tr])
            p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
                pc.transform(Xs[tr]), yy[tr]).predict(pc.transform(Xs[te]))
        return float(np.corrcoef(p, yy)[0, 1]), int(m.sum()), (m, p)

    n = len(y)
    g = np.where(np.isfinite(GA), GA, np.nanmean(GA)).reshape(n, 1)
    COV1 = np.column_stack([np.ones(n), g])
    r1, n1, aux = _oof(y, COV1)
    out = dict(n_fetuses=int(np.isfinite(y).sum()), target="log1p(RMS visit deviation)",
               r_adj_GA=r1, n_GA=n1)
    if BMI is not None:
        b = np.where(np.isfinite(BMI), BMI, 0.0).reshape(n, 1)
        COV2 = np.column_stack([COV1, b])
        r2, n2, aux2 = _oof(y, COV2)
        out.update(r_adj_GA_BMI=r2, n_GA_BMI=n2)
        mb = np.isfinite(y) & np.isfinite(BMI)
        out["bmi_predicts_noise_r"] = float(pearsonr(BMI[mb], y[mb])[0])
        if aux is not None:
            m, p = aux
            mm = np.isfinite(BMI[m])
            out["direction_pred_noise_vs_BMI"] = float(pearsonr(p[mm], BMI[m][mm])[0])
    if aux is not None:
        m, p = aux
        rng = np.random.default_rng(seed)
        yy = y[m]
        nl = []
        for _ in range(nperm):
            nl.append(float(np.corrcoef(rng.permutation(p), yy)[0, 1]))
        out["null_p95"] = float(np.percentile(nl, 95))
        out["p"] = float((1 + sum(x >= r1 for x in nl)) / (1 + len(nl)))
    return out


MANIFEST_CSV = "/Users/tiago/dev/fgr-geometry/results/img_align/image_clusters.csv"


def fgm_image_pcs_by_plane(fids, n_pc=4, layer="emb_l5", planes=None,
                           use_labelled=True, path=None, manifest=None):
    """Per-fetus image PCs computed SEPARATELY PER ANATOMICAL PLANE.

    The default pooled representation (fgm_image_pcs) averages ~22 frames of
    DIFFERENT planes into one vector, on top of USFM's own patch pooling — two
    averaging steps that destroy any plane-specific signal before the statistics
    start. This builds one block per plane instead.

    use_labelled=True uses IMPACT's ground-truth `plane` column (12,929 frames);
    False uses the kNN-propagated `plane_prop` (all frames, but a visual audit
    found it unreliable on the clinical set — 3 of 6 spot checks wrong).

    Returns (dict plane -> (n_fetuses, n_pc) array aligned to fids, dict plane -> n).
    Unlike a PCA of the pooled vector, these blocks have genuine WITHIN-block
    structure: cerebral/abdominal/femur embeddings are correlated with each other.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    if planes is None:
        planes = ["cerebral", "abdominal", "femur"]
    if path is None:
        path = EMB_NPZ
    if manifest is None:
        manifest = MANIFEST_CSV
    z = np.load(path, allow_pickle=True)
    man = pd.read_csv(manifest)
    col = "plane" if use_labelled else "plane_prop"
    lut = dict(zip(man.new_filename.astype(str), man[col].astype(str)))
    nf = pd.Series(z["new_filename"]).astype(str).values
    pl = np.array([lut.get(x, "nan") for x in nf])
    imp = pd.Series(z["dataset_type"]).astype(str).values == "impact"
    E = z[layer].astype("float32")
    fi = pd.to_numeric(pd.Series(z["fetus_id"]).astype(str), errors="coerce").values
    out, counts = {}, {}
    for p in planes:
        m = imp & (pl == p) & np.isfinite(fi)
        if m.sum() < 50:
            continue
        df = pd.DataFrame(E[m])
        df["fid"] = fi[m].astype(int)
        pf = df.groupby("fid").mean()
        k = min(n_pc, pf.shape[0] - 1, pf.shape[1])
        S = PCA(k, random_state=0).fit_transform(pf.values)
        d = dict(zip([int(x) for x in pf.index], S))
        arr = np.array([d.get(int(f), [np.nan] * k) for f in fids])
        out[p] = (arr - np.nanmean(arr, 0)) / np.nanstd(arr, 0)
        counts[p] = dict(frames=int(m.sum()), fetuses=int(pf.shape[0]))
    return out, counts


RADIOMICS_VID = "f132472f-c3bd-4131-bb56-058a6a57999f"


def fgm_radiomics(fids, n_pc=12, per_plane=False, use_labelled=True,
                  planes=None, artifact_path_fn=None):
    """PyRadiomics texture features -- an ENCODER-INDEPENDENT image representation.

    226 columns in the stored parquet, but only the genuine radiomics families
    count: firstorder / glcm / glrlm / glszm (plus their wavelet-L variants).
    The file also carries metadata columns (ga_weeks*, plane_conf, *_id, *_date)
    that share the underscore naming and MUST be excluded -- ga_weeks in
    particular would inject the gestational-age channel straight into the
    "image" block and manufacture a cross-modal correlation.

    Computed from raw pixels with no neural network, so agreement with USFM
    means an image null is encoder-general rather than a USFM artefact.

    Returns (array aligned to fids, list of feature names) or, with
    per_plane=True, (dict plane -> array, dict plane -> counts).
    """
    import re
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    if planes is None:
        planes = ["cerebral", "abdominal", "femur"]
    getp = artifact_path_fn
    if getp is None:
        import host as _h
        getp = _h.artifact_path
    rad = pd.read_parquet(getp(RADIOMICS_VID))
    fam = re.compile(r"(firstorder|glcm|glrlm|glszm)", re.I)
    meta = re.compile(r"ga_weeks|plane_conf|_id$|^id|date|in_cohort|dataset", re.I)
    feat = [c for c in rad.columns
            if fam.search(str(c)) and not meta.search(str(c))
            and rad[c].dtype.kind in "ifc"]
    rad["fid"] = rad["name"].astype(str).str.extract(r"IMP0*(\d+)_")[0].astype(float)
    rad["key"] = rad["name"].astype(str).str.replace(r"\.png$", "", regex=True)
    if per_plane:
        man = pd.read_csv(MANIFEST_CSV)
        man["key"] = man.new_filename.astype(str).str.replace(r"\.png$", "", regex=True)
        col = "plane" if use_labelled else "plane_prop"
        rad = rad.merge(man[["key", col]], on="key", how="left", suffixes=("", "_m"))
        pcol = col if col in rad.columns else col + "_m"
        out, counts = {}, {}
        for p in planes:
            sub = rad[(rad[pcol] == p) & np.isfinite(rad.fid)]
            X = sub[feat].to_numpy(dtype=float)
            ok = np.isfinite(X).all(1)
            if ok.sum() < 50:
                continue
            df = pd.DataFrame(X[ok])
            df["fid"] = sub.fid.to_numpy()[ok].astype(int)
            pf = df.groupby("fid").mean()
            V = (pf.values - pf.values.mean(0)) / (pf.values.std(0) + 1e-9)
            k = min(n_pc, V.shape[0] - 1, V.shape[1])
            S = PCA(k, random_state=0).fit_transform(V)
            d = dict(zip([int(x) for x in pf.index], S))
            arr = np.array([d.get(int(f), [np.nan] * k) for f in fids])
            out[p] = (arr - np.nanmean(arr, 0)) / np.nanstd(arr, 0)
            counts[p] = dict(frames=int(ok.sum()), fetuses=int(pf.shape[0]))
        return out, counts
    sub = rad[np.isfinite(rad.fid)]
    X = sub[feat].to_numpy(dtype=float)
    ok = np.isfinite(X).all(1)
    df = pd.DataFrame(X[ok])
    df["fid"] = sub.fid.to_numpy()[ok].astype(int)
    pf = df.groupby("fid").mean()
    V = (pf.values - pf.values.mean(0)) / (pf.values.std(0) + 1e-9)
    S = PCA(min(n_pc, V.shape[0] - 1, V.shape[1]), random_state=0).fit_transform(V)
    d = dict(zip([int(x) for x in pf.index], S))
    arr = np.array([d.get(int(f), [np.nan] * S.shape[1]) for f in fids])
    return (arr - np.nanmean(arr, 0)) / np.nanstd(arr, 0), feat


def fgm_split_sample_screen(Y, names, IMG, COV, n_top=2, seed=0, nperm=1000, folds=5, npc=8):
    """Select variables on HALF A, test them on HALF B — breaks selection circularity.

    When a variable is chosen because it topped a screen, re-testing it on the
    same rows is circular and inflates the result. This ranks all candidates on
    a random half and evaluates only the top n_top on the held-out half, with a
    permutation null computed in B.

    Y: (n, k) candidate targets. IMG: image representation. COV: nuisance design.
    Returns dict with the A-ranking, the B-test of the A-selected variables, and
    the same variables tested on the full cohort (flagged as confirmatory-only).
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    def _oof(y, X, C, rows, sd=0):
        m = rows & np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(C).all(1)
        if m.sum() < 80:
            return float("nan"), int(m.sum())
        A = C[m]
        yy = y[m] - A @ np.linalg.lstsq(A, y[m], rcond=None)[0]
        Xs = X[m] - A @ np.linalg.lstsq(A, X[m], rcond=None)[0]
        p = np.zeros_like(yy)
        for tr, te in KFold(folds, shuffle=True, random_state=sd).split(Xs):
            pc = PCA(min(npc, Xs.shape[1], len(tr) - 1), random_state=0).fit(Xs[tr])
            p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(
                pc.transform(Xs[tr]), yy[tr]).predict(pc.transform(Xs[te]))
        return float(np.corrcoef(p, yy)[0, 1]), int(m.sum())

    rng = np.random.default_rng(seed)
    n = len(Y)
    half = rng.random(n) < 0.5
    A_rows, B_rows = half, ~half
    rank = []
    for j, nm in enumerate(names):
        r, nn = _oof(Y[:, j], IMG, COV, A_rows)
        rank.append((nm, j, r, nn))
    rank.sort(key=lambda t: -(t[2] if np.isfinite(t[2]) else -9))
    sel = rank[:n_top]
    out = dict(ranking_A=[dict(var=a, r=float(c), n=int(d)) for a, b, c, d in rank],
               selected=[a for a, b, c, d in sel], tested_B=[], full_cohort=[])
    for nm, j, rA, _ in sel:
        rB, nB = _oof(Y[:, j], IMG, COV, B_rows)
        m = B_rows & np.isfinite(Y[:, j]) & np.isfinite(IMG).all(1) & np.isfinite(COV).all(1)
        nl = []
        for _ in range(nperm):
            ys = Y[:, j].copy()
            ys[m] = rng.permutation(Y[m, j])
            nl.append(_oof(ys, IMG, COV, B_rows)[0])
        nl = np.array([x for x in nl if np.isfinite(x)])
        out["tested_B"].append(dict(var=nm, r_A=float(rA), r_B=float(rB), n_B=int(nB),
                                    null_p95=float(np.percentile(nl, 95)),
                                    p=float((1 + (nl >= rB).sum()) / (1 + len(nl)))))
        rF, nF = _oof(Y[:, j], IMG, COV, np.ones(n, bool))
        out["full_cohort"].append(dict(var=nm, r=float(rF), n=int(nF),
                                       flag="NOT independent of selection"))
    return out
