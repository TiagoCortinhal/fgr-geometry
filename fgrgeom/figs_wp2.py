"""WP2 publication figures. Diagnostics of the latent geometry, not performance.

Reads the canonical embedding (k=6 FA on biom+doppler) and the consolidated
results JSONs; refits nothing whose numbers are already published except the
loading matrix W (needed to draw the mechanism directions), which is checked
against the saved latent before use.

main() writes results/figs/{latent_centile,flow_field,linear_vs_nonlinear,
branch_control}.png and results/figs/figs_wp2.json with the actual numbers.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fgrgeom.panel as P
from fgrgeom import config as C

_RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
_FIGS = os.path.join(_RES, "figs")
_INCLUDE = ("biom", "doppler")
_K = 6


def _load_json(rel):
    with open(os.path.join(_RES, rel)) as f:
        return json.load(f)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _latent_with_loadings():
    """Return pca scores of the canonical k=6 latent + W projected into the
    top-2 PCA plane. W is refit; Z is verified against the saved embedding."""
    from fgrgeom import latent as L
    from fgrgeom import embedding as E
    panel = P.load_panel()
    d = L.fit_latent(panel, k=_K, include=_INCLUDE)
    Z, W, names = d["Z"], d["W"], d["colnames"]
    saved = np.load(os.path.join(_RES, "embedding_arrays.npz"))
    drift = float(np.abs(Z - saved["Z"]).max())
    scores, comps, evr, _ = E.pca(Z)             # comps rows = PCA dirs in Z space
    V2 = comps[:2]                               # (2, k)
    # feature j increases along W[j,:] in Z space -> V2 @ W[j,:] in the 2D plane
    Wp = (V2 @ W.T).T                            # (d, 2)
    bc = panel.outcomes["percentile_birth_pop"].reindex(panel.ids).to_numpy(float)
    return scores[:, :2], Wp, names, bc, evr[:2], drift


def fig_latent_centile(path):
    XY, Wp, names, bc, evr, drift = _latent_with_loadings()
    biom_idx = [i for i, n in enumerate(names) if n.startswith(("20s:", "28s:", "32s:", "eco:"))]
    size_dir = _unit(np.mean(Wp[biom_idx], axis=0))
    # redistribution toward placental insufficiency: low CPR, low ACM, high AU/UtA
    sign = {"dop:Percentil_CPR": -1, "dop:Percentil_ACM": -1,
            "dop:Percentil_AU": +1, "dop:Percentil_UTA": +1}
    redist = np.zeros(2)
    for nm, s in sign.items():
        if nm in names:
            redist += s * _unit(Wp[names.index(nm)])
    redist = _unit(redist)
    angle = float(np.degrees(np.arccos(np.clip(abs(size_dir @ redist), -1, 1))))

    span = np.percentile(np.abs(XY), 99)
    scale = 0.85 * span
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fin = np.isfinite(bc)
    sc = ax.scatter(XY[fin, 0], XY[fin, 1], c=bc[fin], s=10, cmap="viridis",
                    alpha=0.8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("birth-weight centile (population)")
    for d, lab, col in [(size_dir, "size", "C3"), (redist, "redistribution", "C0")]:
        ax.annotate("", xy=tuple(d * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
        ax.text(d[0] * scale * 1.08, d[1] * scale * 1.08, lab, color=col,
                fontsize=11, ha="center", va="center", weight="bold")
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("latent PC1 (evr %.2f)" % evr[0])
    ax.set_ylabel("latent PC2 (evr %.2f)" % evr[1])
    ax.set_title("2-D latent coloured by birth centile\n"
                 "size vs redistribution loading directions  (angle %.0f deg)" % angle)
    ax.set_aspect("equal", "box")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"angle_size_redist_deg": angle, "z_drift_vs_saved": drift,
            "evr_pc12": [float(evr[0]), float(evr[1])],
            "n_finite_centile": int(fin.sum())}


def fig_flow_field(path):
    from fgrgeom import flow as F
    panel = P.load_panel()
    stats = _load_json("flow/flow_stats.json")
    samples = F.velocity_samples(panel)
    allp = np.vstack([s["p"] for s in samples.values() if s["n"] > 0])
    mu = allp.mean(0)
    _, _, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    pcs = Vt[:2]
    labs = [l for l in samples if samples[l]["n"] >= 20]
    fig, axes = plt.subplots(1, len(labs) + 1, figsize=(4.3 * (len(labs) + 1), 4.2))
    for ax, lab in zip(axes[:-1], labs):
        s = samples[lab]
        pp = (s["p"] - mu) @ pcs.T
        vv = s["v"] @ pcs.T
        gx = np.linspace(pp[:, 0].min(), pp[:, 0].max(), 9)
        gy = np.linspace(pp[:, 1].min(), pp[:, 1].max(), 9)
        ix = np.clip(np.digitize(pp[:, 0], gx), 1, len(gx) - 1) - 1
        iy = np.clip(np.digitize(pp[:, 1], gy), 1, len(gy) - 1) - 1
        cx, cy, ux, uy = [], [], [], []
        for a in range(len(gx) - 1):
            for c in range(len(gy) - 1):
                m = (ix == a) & (iy == c)
                if m.sum() >= 3:
                    cx.append((gx[a] + gx[a + 1]) / 2); cy.append((gy[c] + gy[c + 1]) / 2)
                    ux.append(vv[m, 0].mean()); uy.append(vv[m, 1].mean())
        ax.scatter(pp[:, 0], pp[:, 1], s=4, c="0.85", zorder=1)
        ax.quiver(cx, cy, ux, uy, angles="xy", color="C3", zorder=2, width=0.006)
        ax.plot(0, 0, "k+", ms=12, mew=2, zorder=3)
        w = stats["windows"][lab]
        ax.set_title("%s\nabscissa=%.3f (CI %.3f..%.3f)" % (
            lab, w["spectral_abscissa"], w["abscissa_ci"][0], w["abscissa_ci"][1]))
        ax.set_xlabel("velocity-PC1"); ax.set_ylabel("velocity-PC2")
        ax.set_aspect("equal", "box")
    ax = axes[-1]
    absc = [(l, stats["windows"][l]["spectral_abscissa"],
             stats["windows"][l]["abscissa_ci"]) for l in labs]
    y = np.arange(len(absc))
    vals = [a[1] for a in absc]
    err = [[a[1] - a[2][0] for a in absc], [a[2][1] - a[1] for a in absc]]
    ax.barh(y, vals, xerr=err, color="C0", height=0.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([a[0] for a in absc])
    ax.set_xlabel("max Re(eigenvalue) of flow Jacobian")
    ax.set_title("all directions contract (abscissa<0)\nmean reversion, no divergent axis")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {l: stats["windows"][l]["spectral_abscissa"] for l in labs}


def fig_linear_vs_nonlinear(path):
    comp = _load_json("nl/comparison.json")
    idd = comp["intrinsic_dimension"]
    sets = [s for s in ["minimal", "plus_ratios", "plus_cardiac", "plus_maternal", "full"]
            if s in idd]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lin = [idd[s]["linear_pr"] for s in sets]
    twonn = [idd[s]["twonn"] for s in sets]
    mle = [idd[s]["mle_k20"] for s in sets]
    cdim = [idd[s]["corr_dim"] for s in sets]
    x = np.arange(len(sets))
    axA.plot(x, lin, "o-", color="C0", lw=2, label="linear participation ratio")
    axA.plot(x, twonn, "s--", color="C3", label="TwoNN (nonlinear)")
    axA.plot(x, mle, "^--", color="C1", label="MLE k20 (nonlinear)")
    axA.plot(x, cdim, "v--", color="C2", label="correlation dim (nonlinear)")
    axA.set_xticks(x); axA.set_xticklabels(sets, rotation=20, ha="right")
    axA.set_ylabel("intrinsic dimension")
    axA.set_title("nonlinear ID tracks linear rank, no excess\n(curvature flag false in every set; TwoNN over-estimates at small n)")
    axA.legend(fontsize=8)

    # held-out reconstruction, normalized MSE (lower = better); one metric only
    ae = comp["autoencoder_k6"]
    msets = [s for s in ["minimal", "full"] if s in ae]
    models = [("recon_FA", "FA (linear)", "C0"),
              ("recon_AE", "AE", "C3"),
              ("recon_VAE", "VAE", "C1")]
    w = 0.25
    xb = np.arange(len(msets))
    for j, (key, lab, col) in enumerate(models):
        axB.bar(xb + (j - 1) * w, [ae[s][key] for s in msets], w, label=lab, color=col)
    axB.axhline(1.0, color="0.5", ls=":", lw=1)
    axB.text(len(msets) - 0.5, 1.0, "mean baseline", color="0.5", fontsize=7,
             ha="right", va="bottom")
    axB.set_xticks(xb); axB.set_xticklabels(msets)
    axB.set_ylabel("held-out reconstruction MSE  (lower = better)")
    axB.set_title("linear FA reconstructs as well or better\nthan AE/VAE at matched k=6")
    axB.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"intrinsic_dim_sets": sets,
            "recon_minimal": {m[0]: ae["minimal"][m[0]] for m in models},
            "recon_full": {m[0]: ae["full"][m[0]] for m in models} if "full" in ae else {}}


def fig_branch_control(path):
    comp = _load_json("nl/comparison.json")
    cc = comp["calibrated_controls_minimal"]
    bt = comp["branch_topology"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
    pos = cc["positive_control_route_auc"]
    null = cc["null_randlabel_auc"]
    axA.bar([0, 1], [pos, null], color=["C2", "0.6"], width=0.55)
    axA.axhline(0.5, color="k", ls=":", lw=1)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["planted branch\n(positive control)", "random label\n(null)"])
    axA.set_ylim(0, 1.0)
    for xx, vv in [(0, pos), (1, null)]:
        axA.text(xx, vv + 0.02, "%.2f" % vv, ha="center", fontsize=10)
    axA.set_ylabel("route-separation AUC (CV)")
    axA.set_title("minimal-set branch detector WORKS\nplanted route recovered, random flat")

    rf = bt["real_full"]
    nc = bt["neg_control"]
    pc = bt["pos_control_planted_branch"]
    labels = ["real data", "neg control\n(no branch)", "planted branch\n(pos control)"]
    h1h0 = [rf["h1_h0"], nc["h1_h0"], pc["h1_h0"]]
    dipp = [rf.get("dip_p_pc1", rf.get("dip_p_dc1")), None, pc["dip_p_dc1"]]
    axB.bar(range(3), h1h0, color=["C0", "0.6", "C3"], width=0.55)
    for i, (h, p) in enumerate(zip(h1h0, dipp)):
        txt = "H1/H0=%.3f" % h + (("\ndip p=%.2f" % p) if p is not None else "")
        axB.text(i, h + 0.01, txt, ha="center", fontsize=8)
    axB.set_xticks(range(3)); axB.set_xticklabels(labels)
    axB.set_ylabel("persistent H1/H0 loop ratio")
    axB.set_title("real data is flat (no loop/branch)\nbut full-set planted branch NOT recovered -> underpowered")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return {"pos_route_auc": pos, "null_route_auc": null,
            "real_h1_h0": rf["h1_h0"], "neg_h1_h0": nc["h1_h0"],
            "planted_h1_h0": pc["h1_h0"], "real_dip_p": dipp[0],
            "planted_dip_p": pc["dip_p_dc1"]}


def main():
    os.makedirs(_FIGS, exist_ok=True)
    out = {}
    out["latent_centile"] = fig_latent_centile(os.path.join(_FIGS, "latent_centile.png"))
    out["flow_field"] = fig_flow_field(os.path.join(_FIGS, "flow_field.png"))
    out["linear_vs_nonlinear"] = fig_linear_vs_nonlinear(
        os.path.join(_FIGS, "linear_vs_nonlinear.png"))
    out["branch_control"] = fig_branch_control(os.path.join(_FIGS, "branch_control.png"))
    with open(os.path.join(_FIGS, "figs_wp2.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
