import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path, minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors

from fgrgeom import config as C
from fgrgeom import panel as P
from fgrgeom import latent as L


# Hartigan & Hartigan (1985) dip statistic, faithful port of the GCM/LCM walk.
def dip(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n < 4:
        return 0.0
    # work on the ecdf; lo/hi are the touching indices of greatest convex
    # minorant from below and least concave majorant from above.
    low, high = 0, n - 1
    dipval = 0.0
    # cumulative count fractions
    while True:
        # GCM from low over [low,high]
        gcm = _convex_minorant(x, low, high)
        lcm = _concave_majorant(x, low, high)
        # distances ecdf vs the two envelopes
        d_gcm, ig = _maxdist(x, n, gcm, low, high, upper=False)
        d_lcm, il = _maxdist(x, n, lcm, low, high, upper=True)
        if d_gcm >= d_lcm:
            d = d_gcm
            new_low, new_high = _bracket(gcm, ig)
        else:
            d = d_lcm
            new_low, new_high = _bracket(lcm, il)
        if d <= dipval or new_low >= new_high or (new_low == low and new_high == high):
            dipval = max(dipval, d)
            break
        dipval = max(dipval, d)
        low, high = new_low, new_high
    return dipval / (2.0 * n)


def _convex_minorant(x, lo, hi):
    # indices on the lower convex hull of points (i, i) wrt value x as abscissa
    pts = []
    for i in range(lo, hi + 1):
        while len(pts) >= 2 and _cross(x, pts[-2], pts[-1], i) <= 0:
            pts.pop()
        pts.append(i)
    return pts


def _concave_majorant(x, lo, hi):
    pts = []
    for i in range(lo, hi + 1):
        while len(pts) >= 2 and _cross(x, pts[-2], pts[-1], i) >= 0:
            pts.pop()
        pts.append(i)
    return pts


def _cross(x, a, b, c):
    # sign of cross product of (b-a) x (c-a) in (value, index) plane
    return (x[b] - x[a]) * (c - a) - (x[c] - x[a]) * (b - a)


def _maxdist(x, n, hull, lo, hi, upper):
    # max vertical gap between ecdf and the piecewise-linear hull
    best, bi = 0.0, hull[0]
    j = 0
    for i in range(lo, hi + 1):
        while j < len(hull) - 1 and hull[j + 1] < i:
            j += 1
        a, b = hull[j], hull[min(j + 1, len(hull) - 1)]
        if b == a:
            hv = a
        else:
            t = (x[i] - x[a]) / (x[b] - x[a]) if x[b] != x[a] else 0.0
            hv = a + t * (b - a)
        ecdf = i
        gap = (ecdf - hv) if upper else (hv - ecdf)
        if gap > best:
            best, bi = gap, i
    return best / n, bi


def _bracket(hull, idx):
    lo = hull[0]
    hi = hull[-1]
    for j in range(len(hull) - 1):
        if hull[j] <= idx <= hull[j + 1]:
            return hull[j], hull[j + 1]
    return lo, hi


def dip_test(x, n_boot=2000, seed=C.SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    try:
        import diptest as _dt
        d, p = _dt.diptest(x)
        return float(d), float(p), n
    except Exception:
        pass
    # fallback: hand-rolled dip + uniform bootstrap null (less validated)
    d = dip(x)
    rng = np.random.default_rng(seed)
    null = np.array([dip(rng.random(n)) for _ in range(n_boot)])
    p = float((null >= d).mean())
    return d, p, n


# --- embedding / graph ---
def _principal_axis(panel, k=4):
    fit = L.fit_latent(panel, k=k, include=("biom", "doppler"))
    Z = fit["Z"]
    # variance-rank the latent columns so axis 1 is the dominant spread direction
    order = np.argsort(-Z.var(axis=0))
    Zo = Z[:, order]
    u, s, vt = np.linalg.svd(Zo - Zo.mean(0), full_matrices=False)
    pc1 = (Zo - Zo.mean(0)) @ vt[0]
    return pc1, Zo, s


def _graph(panel):
    # prefer the sibling embedding module if it exists; else kNN on latent
    try:
        from fgrgeom import embedding as E
        emb = E.embed(panel) if hasattr(E, "embed") else None
        if emb is not None:
            Y = emb["Y"] if isinstance(emb, dict) else emb
            return np.asarray(Y), "embedding.embed"
    except Exception:
        pass
    _, Zo, _ = _principal_axis(panel)
    return Zo, "latent(biom,doppler)"


def knn_branch(Y, n_neighbors=15):
    n = Y.shape[0]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n - 1)).fit(Y)
    A = nn.kneighbors_graph(mode="distance")
    A = A.maximum(A.T)
    ncomp, _ = connected_components(A, directed=False)
    mst = minimum_spanning_tree(A)
    mst = mst + mst.T
    deg = np.asarray((mst > 0).sum(axis=1)).ravel()
    # branch points = MST nodes of degree >=3; leaves degree 1
    n_branch = int((deg >= 3).sum())
    n_leaf = int((deg == 1).sum())
    # graph diameter proxy on shortest paths over kNN graph (geodesic spread)
    D = shortest_path(A, method="D", directed=False)
    finite = np.isfinite(D)
    diam = float(D[finite].max()) if finite.any() else np.inf
    return dict(n_components=int(ncomp), n_branch_nodes=n_branch,
                n_leaves=n_leaf, mst_max_degree=int(deg.max()),
                geodesic_diameter=diam, n_neighbors=int(min(n_neighbors, n - 1)))


def persistent_homology(Y, maxdim=1):
    try:
        from ripser import ripser
        # greedy-permutation subsample keeps H1 tractable on ~1k points
        n_perm = min(Y.shape[0], 300)
        res = ripser(Y, maxdim=maxdim, n_perm=n_perm)
        out = {}
        for d, dg in enumerate(res["dgms"]):
            dg = dg[np.isfinite(dg).all(axis=1)] if dg.size else dg
            life = (dg[:, 1] - dg[:, 0]) if dg.size else np.array([])
            out[f"H{d}"] = dict(n=int(dg.shape[0]),
                                max_life=float(life.max()) if life.size else 0.0,
                                top3=sorted(map(float, life))[-3:] if life.size else [])
        return out, "ripser"
    except Exception:
        pass
    try:
        import gudhi
        rc = gudhi.RipsComplex(points=Y.tolist())
        st = rc.create_simplex_tree(max_dimension=maxdim + 1)
        st.compute_persistence()
        out = {}
        for d in range(maxdim + 1):
            ints = st.persistence_intervals_in_dimension(d)
            ints = np.asarray(ints) if len(ints) else np.empty((0, 2))
            ints = ints[np.isfinite(ints).all(axis=1)] if ints.size else ints
            life = (ints[:, 1] - ints[:, 0]) if ints.size else np.array([])
            out[f"H{d}"] = dict(n=int(ints.shape[0]),
                                max_life=float(life.max()) if life.size else 0.0,
                                top3=sorted(map(float, life))[-3:] if life.size else [])
        return out, "gudhi"
    except Exception:
        return None, "none (ripser/gudhi unavailable)"


def ph_h1_ratio(ph):
    # significance of the strongest 1-cycle: its lifetime vs the H0 merge scale.
    # a real loop/branch-closure should be a sizeable fraction of the H0 spread;
    # values << 1 are sampling noise loops in a dense cloud.
    if not ph or "H1" not in ph or ph["H1"]["n"] == 0:
        return 0.0
    h0 = ph.get("H0", {}).get("max_life", 0.0) or 1e-9
    return ph["H1"]["max_life"] / h0


def verdict(dip_p, branch, ph):
    # MST degree/branch_nodes are NOT used here: a dense cloud trivially yields
    # hundreds of degree>=3 MST nodes, so they carry no branch signal. The honest
    # signals are (1) multimodality of the principal axis (dip) and (2) a
    # persistent 1-cycle (H1 lifetime relative to the H0 merge scale).
    multimodal = dip_p < 0.05
    loop = ph_h1_ratio(ph) > 0.25
    if branch["n_components"] > 1 and multimodal:
        return "blob / disconnected (multiple components + multimodal axis)"
    if multimodal and loop:
        return "Y / branch (multimodal axis + persistent 1-cycle)"
    if multimodal:
        return "multimodal arc (dip significant, no persistent loop)"
    if loop:
        return "looped/branched (persistent 1-cycle, unimodal axis)"
    return "single arc / unimodal continuum"


def run(panel=None, n_neighbors=15, n_boot=2000):
    if panel is None:
        panel = P.load_panel()
    pc1, Zo, s = _principal_axis(panel)
    d, p, n = dip_test(pc1, n_boot=n_boot)
    Y, src = _graph(panel)
    branch = knn_branch(Y, n_neighbors=n_neighbors)
    ph, ph_src = persistent_homology(Y)
    v = verdict(p, branch, ph)
    return dict(n=n, dip=d, dip_p=p, sv=s.tolist(), graph_source=src,
                branch=branch, ph=ph, ph_source=ph_src, verdict=v)


def main():
    r = run()
    print(f"n={r['n']}  graph={r['graph_source']}  ph_lib={r['ph_source']}")
    print(f"dip={r['dip']:.5f}  dip_p={r['dip_p']:.4f}  (p<.05 => multimodal)")
    print(f"latent singular values: {[round(x,3) for x in r['sv']]}")
    b = r["branch"]
    print(f"kNN/MST: components={b['n_components']} branch_nodes={b['n_branch_nodes']} "
          f"leaves={b['n_leaves']} max_deg={b['mst_max_degree']} "
          f"geo_diam={b['geodesic_diameter']:.2f} k={b['n_neighbors']}")
    print(f"persistent homology: {r['ph']}")
    print(f"H1/H0 lifetime ratio: {ph_h1_ratio(r['ph']):.3f} (>0.25 => real loop)")
    print(f"NOTE: MST branch_nodes/leaves are descriptive only, not a branch signal.")
    print(f"VERDICT: {r['verdict']}")


if __name__ == "__main__":
    main()
