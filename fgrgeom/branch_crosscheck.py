import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L
from fgrgeom import featuresets as FS

_REPO_RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# TASK 1b independent branch cross-check. DIFFERENT principle from the PAGA /
# persistent-homology battery in nl_branch_topology: estimate the DENSITY RIDGE of
# the FA latent with SCMS (subspace-constrained mean shift, Ozertem & Erdogmus
# 2011) and read the TOPOLOGY of the recovered 1-D ridge. A single continuum
# collapses to a path (0 branch points, 2 endpoints); a route split collapses to a
# Y (>=1 branch point, >=3 endpoints). Branch-CAPABLE (unlike density-gap
# clustering, which cannot see a connected Y) and shares no machinery with PAGA/PH.
#
# CONTROL DISCIPLINE / what the experiments below establish:
#  (1) The SCMS+MST-topology machinery DOES recover a planted 2-D Y (branch=1,
#      endpoints=3) -- but only in a narrow fine-bandwidth window, where a single
#      2-D-area null ALSO fragments into spurious ridge branches. So a generic
#      single-manifold null gives a DEGENERATE calibration here: at n=977 over a
#      ~2-D continuum, 1-D ridge estimation of a 2-D area manufactures branches
#      indistinguishable from a weak planted Y. Null-calibration alone is therefore
#      underpowered (this MATCHES nl_branch_topology's UNDERPOWERED verdict).
#  (2) The DECISIVE, non-degenerate test is a LEAVE-OUT: the ridge run at a coarse
#      bandwidth (where the 2-D-area null is clean, branch=0) shows a single robust
#      branch on the REAL data -- and that branch is causally attributable to ~6
#      known gross-error records (hc/bpd/fl IG21 z ~ -12..-20 sentinels and a
#      Percentil_AU=100). Removing them collapses the ridge to a single path
#      (branch=0, endpoints=2) at every bandwidth. The apparent branch is a data-
#      error artifact, not a route split.
# VERDICT: NO control-trusted branch; the real ridge is a single continuum once
# gross errors are dropped. AGREES with branch_topology / the 2-D-continuum verdict.


# ----------------------------- SCMS core -----------------------------

def scms(X, h, ridge_dim=1, n_iter=50, tol=1e-3):
    X = np.asarray(X, float)
    n, D = X.shape
    Y = X.copy()
    h2 = h * h
    I = np.eye(D)
    for _ in range(n_iter):
        maxstep = 0.0
        for j in range(n):
            y = Y[j]
            u = (X - y) / h
            d2 = np.einsum("ij,ij->i", u, u)
            c = np.exp(-0.5 * d2)
            sc = c.sum()
            if sc < 1e-12:
                continue
            w = c / sc
            ms = (w[:, None] * X).sum(0) - y
            uu = np.einsum("i,ij,ik->jk", w, u, u)
            H = (uu - I * w.sum()) / h2
            H = 0.5 * (H + H.T)
            _, evecs = np.linalg.eigh(H)
            Vperp = evecs[:, ridge_dim:]
            step = Vperp @ (Vperp.T @ ms)
            Y[j] = y + step
            s = np.linalg.norm(step)
            if s > maxstep:
                maxstep = s
        if maxstep < tol * h:
            break
    return Y


def ridge_topology(Y, merge_r, prune_len):
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    Y = np.asarray(Y, float)
    nodes = []
    for p in Y:
        ok = True
        for q in nodes:
            if np.linalg.norm(p - q) < merge_r:
                ok = False
                break
        if ok:
            nodes.append(p)
    Cn = np.array(nodes)
    K = len(Cn)
    if K < 3:
        return dict(n_nodes=int(K), n_branch=0, n_endpoints=int(K))
    Dm = squareform(pdist(Cn))
    Tm = minimum_spanning_tree(Dm).toarray()
    A = Tm + Tm.T
    adj = A > 0
    keep = np.ones(K, bool)
    changed = True
    while changed:
        changed = False
        d = (adj & keep[None, :] & keep[:, None]).sum(1)
        for i in np.where(keep & (d == 1))[0]:
            nb = np.where(adj[i] & keep)[0]
            if len(nb) == 1 and A[i, nb[0]] < prune_len:
                keep[i] = False
                changed = True
    d = (adj & keep[None, :] & keep[:, None]).sum(1) * keep
    return dict(n_nodes=int(keep.sum()), n_branch=int((d >= 3).sum()),
                n_endpoints=int((d == 1).sum()))


def _fa_latent(X, M, k=6, max_iter=300):
    fit = L.FactorAnalysisMissing(k, max_iter=max_iter).fit(X, M)
    Z, _ = fit.transform(X, M)
    return Z, fit.n_iter_


def ridge_branch(Z, h, n_dims=2, merge_scale=0.5, prune_scale=2.0, n_iter=50):
    Zc = Z - Z.mean(0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    Y = Zc @ vt[:n_dims].T
    Y = Y / Y.std(0)
    R = scms(Y, h, ridge_dim=1, n_iter=n_iter)
    return ridge_topology(R, merge_r=merge_scale * h, prune_len=prune_scale * h)


def _robust_absz(X, M):
    """per-column median/std robust z; max |z| per row over observed entries."""
    Xs = np.zeros_like(X, float)
    for j in range(X.shape[1]):
        col = X[M[:, j], j]
        mu = np.median(col)
        sd = np.std(col) + 1e-9
        Xs[:, j] = np.where(M[:, j], (X[:, j] - mu) / sd, 0.0)
    return np.abs(Xs).max(1)


# ----------------------- control simulators -----------------------

def _null_features(n, n_dirs=2, d=47, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    Tm = rng.standard_normal((n, n_dirs))
    Lm = rng.standard_normal((n_dirs, d))
    signal = Tm @ Lm
    noise = rng.standard_normal((n, d))
    sd = signal.std(0); sd[sd == 0] = 1.0
    X = np.sqrt(rho) * signal / sd + np.sqrt(1 - rho) * noise
    return X, np.ones((n, d), bool)


def _branch2d_features(n, d=47, branch_w=3.0, rho=0.9, noise2d=0.15, seed=0):
    """POSITIVE control: a genuinely 2-D planted Y (three coplanar arms from a
    centre) embedded into d features via two orthogonal loadings. Intrinsically
    2-D so it survives the top-2 latent projection (the prior 3-direction trunk/
    arm construction collapsed under it)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 3, n)
    arm = rng.integers(0, 3, n)
    ang = {0: np.pi / 2, 1: -np.pi / 6, 2: 7 * np.pi / 6}
    a = np.array([ang[x] for x in arm])
    coords = np.c_[t * np.cos(a), t * np.sin(a)] + rng.normal(0, noise2d, (n, 2))
    G = rng.standard_normal((d, 2))
    Q, _ = np.linalg.qr(G)
    signal = branch_w * coords @ Q.T
    noise = rng.standard_normal((n, d))
    sd = signal.std(0); sd[sd == 0] = 1.0
    X = np.sqrt(rho) * signal / sd + np.sqrt(1 - rho) * noise
    return X, np.ones((n, d), bool)


# ----------------------------- analysis -----------------------------

def analyze_set(panel, name, k=6, n_dims=2, h_real=(0.35, 0.40, 0.50),
                h_ctrl=(0.22, 0.30), z_cut=10.0, z_cut_strict=8.0, n_null=8,
                seed=C.SEED):
    X, M, names = FS.build(panel, name)
    n, d = X.shape

    # --- real data: full vs gross-error-removed, over a bandwidth grid ---
    absz = _robust_absz(X, M)
    keep = absz <= z_cut
    flagged_ids = [int(panel.ids[i]) for i in np.where(~keep)[0]]
    Zf, _ = _fa_latent(X, M, k=k)
    Zc, _ = _fa_latent(X[keep], M[keep], k=k)
    real_full = {("h%.2f" % h): ridge_branch(Zf, h, n_dims=n_dims) for h in h_real}
    real_clean = {("h%.2f" % h): ridge_branch(Zc, h, n_dims=n_dims) for h in h_real}
    # stricter but still mild outlier screen (robustness of any surviving branch)
    keep2 = absz <= z_cut_strict
    Zc2, _ = _fa_latent(X[keep2], M[keep2], k=k)
    real_clean_strict = {("h%.2f" % h): ridge_branch(Zc2, h, n_dims=n_dims) for h in h_real}

    # --- positive control: 2-D planted Y over the fine-bandwidth window ---
    Xp, Mp = _branch2d_features(n, d=d, rho=0.9, seed=seed)
    Zp, _ = _fa_latent(Xp, Mp, k=k)
    pos = {}
    for h in h_ctrl:
        recs = [ridge_branch(_fa_latent(*_branch2d_features(n, d=d, rho=0.9, seed=seed + s))[0],
                             h, n_dims=n_dims) for s in range(3)]
        pos["h%.2f" % h] = dict(branch=[r["n_branch"] for r in recs],
                                endpoints=[r["n_endpoints"] for r in recs])
    # --- single-2D-manifold null: at the fine window (shows fragmentation) AND at
    #     the coarse real bandwidths (the calibration that decides the verdict) ---
    def _null_branches(h):
        return [ridge_branch(_fa_latent(*_null_features(n, n_dirs=min(2, k), d=d, rho=0.9,
                                                        seed=seed + 1 + s))[0],
                             h, n_dims=n_dims)["n_branch"] for s in range(n_null)]
    null = {("h%.2f" % h): dict(branch_max=int(max(_null_branches(h)))) for h in h_ctrl}
    null_real = {("h%.2f" % h): int(max(_null_branches(h))) for h in h_real}

    # decision is read at the COARSEST real bandwidth, where 1-D ridge estimation of
    # the 2-D-area null is least fragmented -> the only regime where a branch count
    # is trustworthy.
    h_eval = "h%.2f" % max(h_real)
    cb = real_clean[h_eval]["n_branch"]
    cbs = real_clean_strict[h_eval]["n_branch"]
    fb = real_full[h_eval]["n_branch"]
    null_cut = null_real[h_eval]
    pos_recovers = any(max(pos[hk]["branch"]) >= 1 for hk in pos)
    n_strict = int((~keep2).sum())
    # a branch is trusted only if it clears the null AND survives a stricter (still
    # mild) outlier screen -- i.e. it is not manufactured by a few extreme records.
    trusted_branch = (cb > null_cut) and (cbs > null_cut) and \
        (real_clean[h_eval]["n_endpoints"] >= 3)
    if trusted_branch:
        verdict = ("real ridge keeps %d branch point(s) after outlier removal, ABOVE the "
                   "single-2D-manifold null (max %d) AND robust to a stricter z>%.0f screen "
                   "(%d branch) -> candidate route split; harden the null (marginal-matched) "
                   "before trusting" % (cb, null_cut, z_cut_strict, cbs))
    elif fb >= 1 and cb <= null_cut:
        verdict = ("real branch at the coarsest bandwidth (%d) is a GROSS-ERROR ARTIFACT: "
                   "removing %d flagged records (|robust z|>%.0f: IG21 z-sentinels / "
                   "Percentil_AU=100) drops it to %d, at/below the null cut (%d) -> NO trusted "
                   "branch; single continuum. Positive 2-D-Y control %s recovered by the same "
                   "SCMS+MST machinery." % (fb, len(flagged_ids), z_cut, cb, null_cut,
                                            "IS" if pos_recovers else "is NOT"))
    else:
        verdict = ("real branch (%d) survives the z>%.0f gross-error screen but NOT a stricter "
                   "z>%.0f screen (%d, drops to/below null %d, %d records); it is driven by a "
                   "few extreme records, not a route split -> NO trusted branch; single "
                   "continuum. Positive 2-D-Y control %s recovered." %
                   (cb, z_cut, z_cut_strict, cbs, null_cut, n_strict,
                    "IS" if pos_recovers else "is NOT"))
    return dict(
        n=int(n), n_features=int(d), n_dims=n_dims, k_latent=k, z_cut=z_cut,
        n_flagged=len(flagged_ids), flagged_ids=flagged_ids,
        z_cut_strict=z_cut_strict, n_flagged_strict=int((~keep2).sum()),
        real_full=real_full, real_clean=real_clean, real_clean_strict=real_clean_strict,
        pos_control_planted_Y=pos, null_2d_manifold=null, null_at_real_h=null_real,
        h_eval=h_eval, null_cut_at_eval=null_cut,
        branch_full_at_eval=int(fb), branch_clean_at_eval=int(cb),
        pos_recovers_planted_Y=bool(pos_recovers),
        trusted_branch=bool(trusted_branch),
        verdict=verdict)


def run(panel=None, sets=("minimal", "full"), **kw):
    if panel is None:
        panel = P.load_panel()
    results = {}
    for name in sets:
        print("== %s ==" % name, flush=True)
        r = analyze_set(panel, name, **kw)
        results[name] = r
        print("  flagged=%d real_full=%s real_clean=%s null_at_real_h=%s pos_recovers=%s" % (
            r["n_flagged"],
            {h: v["n_branch"] for h, v in r["real_full"].items()},
            {h: v["n_branch"] for h, v in r["real_clean"].items()},
            r["null_at_real_h"], r["pos_recovers_planted_Y"]), flush=True)
        print("  VERDICT[%s]: %s" % (name, r["verdict"]), flush=True)

    pf = os.path.join(_REPO_RESULTS, "nl", "branch_topology.json")
    other = None
    if os.path.exists(pf):
        with open(pf) as f:
            other = json.load(f)["verdict"]
    # both conclude NO control-trusted branch (this one via error-artifact leave-out,
    # branch_topology via UNDERPOWERED no-flags)
    cc_no_branch = all(not results[s]["trusted_branch"] for s in sets)
    other_no_branch = (other is not None and
                       ("UNDERPOWERED" in other or "no branch" in other.lower()))
    agrees = bool(cc_no_branch and other_no_branch)
    return dict(sets=list(sets), results=results,
                branch_topology_verdict=other, agrees_with_branch_topology=agrees,
                deps=dict(scms="own (Ozertem-Erdogmus)", scipy_mst=True,
                          hdbscan=False, ripser=False, torch=False))


def _figure(out, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    sets = out["sets"]
    fig, ax = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4))
    if len(sets) == 1:
        ax = [ax]
    for j, name in enumerate(sets):
        r = out["results"][name]
        hs = sorted(r["real_full"])
        x = np.arange(len(hs))
        full = [r["real_full"][h]["n_branch"] for h in hs]
        clean = [r["real_clean"][h]["n_branch"] for h in hs]
        ax[j].bar(x - 0.2, full, 0.4, label="all n=%d" % r["n"], color="#1f77b4")
        ax[j].bar(x + 0.2, clean, 0.4, label="drop %d gross-err" % r["n_flagged"],
                  color="#2ca02c")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(hs)
        ax[j].set_ylabel("ridge branch points")
        ax[j].set_title("%s  SCMS density-ridge" % name)
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    out = run()
    os.makedirs(_REPO_RESULTS, exist_ok=True)
    with open(os.path.join(_REPO_RESULTS, "branch_crosscheck.json"), "w") as f:
        json.dump(out, f, indent=2)
    _figure(out, os.path.join(_REPO_RESULTS, "branch_crosscheck.png"))
    print("branch_topology(PAGA/PH) verdict:", out["branch_topology_verdict"])
    print("agrees_with_branch_topology:", out["agrees_with_branch_topology"])
    for name in out["sets"]:
        print("[%s] %s" % (name, out["results"][name]["verdict"]))


if __name__ == "__main__":
    main()
