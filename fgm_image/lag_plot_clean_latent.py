"""
Trajectory plots for the CLEAN config-B latent (_lagB_traj_clean.npy).

Latent = frozen config-B GRU-VAE (biometry+lag+GA, h=32, Z=8, beta=0.1), original lag,
retrained with corrupt biometry (any terminal z < -10) masked. See lag_in_gruvae_clean.py.

Dimension roles in the clean latent (reassigned by retraining):
  size = z4 (bpd/hc), lag = z0 (lag -0.72, cleanest), growth/EFW = z6 (efw/ac).
Chosen for readable + separated decile paths (jaggedness 65deg ~= old contaminated plot's 63deg,
decile spread 1.34 vs 0.88). Axis numbers do NOT match the old _lagB_traj.npy latent — a VAE
assigns concepts to slots arbitrarily, so every retrain renumbers them.

Builds:
  lagB_configB_meanpaths_decile_z4.html  — 10 birth-pct decile mean paths (week-labelled markers)
  lagB_configB_trajectory_3d_z4.html     — all 906 individual fetus trajectories, birth-pct coloured
"""
import numpy as np, pandas as pd, plotly.graph_objects as go
import matplotlib.cm as cm, matplotlib.colors as mcolors

IMG = "/Users/tiago/dev/fgr-geometry/results/img_align"
DIMS = [4, 0, 6]
AXNAMES = ["size (z4, bpd/hc)", "maturation-lag (z0)", "growth/EFW (z6, efw/ac)"]


def _load():
    traj = np.load(f"{IMG}/_lagB_traj_clean.npy")
    z = np.load(f"{IMG}/_merged_seq.npz", allow_pickle=True)
    lab = np.load(f"{IMG}/_merged_labels.npz", allow_pickle=True)
    L, fids = z["L"], z["fids"]; N = len(fids)
    birth = pd.Series(lab["birth"], index=lab["fids"]).reindex(fids).values
    ga_wk = z["X"][:, :, -1] * 36.0 + 6.0
    E = np.array([traj[i, L[i] - 1, :] for i in range(N)])
    A = [(traj[:, :, d] - E[:, d].mean()) / E[:, d].std() for d in DIMS]  # z-scored display axes
    return traj, L, fids, N, birth, ga_wk, A


def decile_meanpaths():
    traj, L, fids, N, birth, ga_wk, A = _load()
    ok = np.isfinite(birth); dec = np.full(N, -1); dec[ok] = np.clip((birth[ok] / 10).astype(int), 0, 9)
    cmap = cm.RdYlBu_r; norm = mcolors.Normalize(0, 9); fig = go.Figure()
    for d in range(10):
        idx = np.where(dec == d)[0]
        if len(idx) == 0: continue
        maxt = int(L[idx].max())
        mp = [[np.mean([A[a][i, t] for i in idx if L[i] > t]) for t in range(maxt)] for a in range(3)]
        gaavg = [np.mean([ga_wk[i, t] for i in idx if L[i] > t]) for t in range(maxt)]
        ct = [sum(1 for i in idx if L[i] > t) for t in range(maxt)]
        c = mcolors.to_hex(cmap(norm(d)))
        fig.add_trace(go.Scatter3d(x=mp[0], y=mp[1], z=mp[2], mode="lines+markers",
            line=dict(color=c, width=6), marker=dict(size=4, color=c), name=f"decile {d*10}-{d*10+10}% (n={len(idx)})",
            hovertext=[f"decile {d*10}-{d*10+10}%<br>visit {t+1} · GA {gaavg[t]:.1f}w · n={ct[t]}" for t in range(maxt)], hoverinfo="text"))
        fig.add_trace(go.Scatter3d(x=[mp[0][0]], y=[mp[1][0]], z=[mp[2][0]], mode="markers+text",
            marker=dict(size=7, color=c, symbol="circle", line=dict(color="black", width=1)),
            text=[f"start {gaavg[0]:.0f}w"], textposition="top center", textfont=dict(size=9, color="black"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[mp[0][-1]], y=[mp[1][-1]], z=[mp[2][-1]], mode="markers+text",
            marker=dict(size=7, color=c, symbol="diamond", line=dict(color="black", width=1)),
            text=[f"end {gaavg[-1]:.0f}w"], textposition="bottom center", textfont=dict(size=9, color="black"), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title="Config-B (original lag, corrupt biometry masked) — birth-pct DECILE mean paths<br>size z4 / lag z0 / growth-EFW z6",
        scene=dict(xaxis_title=AXNAMES[0], yaxis_title=AXNAMES[1], zaxis_title=AXNAMES[2]), width=1050, height=800)
    fig.write_html(f"{IMG}/lagB_configB_meanpaths_decile_z4.html")


def all_trajectories():
    traj, L, fids, N, birth, ga_wk, A = _load()
    ok = np.isfinite(birth); cmap = cm.RdYlBu_r; norm = mcolors.Normalize(0, 100); fig = go.Figure()
    for i in range(N):
        if not ok[i]: continue
        t = int(L[i]); c = mcolors.to_hex(cmap(norm(birth[i])))
        fig.add_trace(go.Scatter3d(x=[A[0][i, j] for j in range(t)], y=[A[1][i, j] for j in range(t)], z=[A[2][i, j] for j in range(t)],
            mode="lines+markers", line=dict(color=c, width=2), marker=dict(size=2, color=c), opacity=0.35, showlegend=False,
            hovertext=[f"fetus {int(fids[i])} · birth pct {birth[i]:.0f}<br>visit {j+1} · GA {ga_wk[i,j]:.1f}w" for j in range(t)], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers",
        marker=dict(size=0.1, color=[0, 100], colorscale=[[0, mcolors.to_hex(cmap(0.0))], [1, mcolors.to_hex(cmap(1.0))]],
        colorbar=dict(title="birth pct", thickness=15, len=0.6), cmin=0, cmax=100), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"Config-B (original lag, corrupt biometry masked) — ALL {int(ok.sum())} fetus trajectories<br>size z4 / lag z0 / growth-EFW z6 · birth-pct coloured",
        scene=dict(xaxis_title=AXNAMES[0], yaxis_title=AXNAMES[1], zaxis_title=AXNAMES[2]), width=1050, height=800)
    fig.write_html(f"{IMG}/lagB_configB_trajectory_3d_z4.html")


if __name__ == "__main__":
    decile_meanpaths(); all_trajectories(); print("built both clean-latent plots")
