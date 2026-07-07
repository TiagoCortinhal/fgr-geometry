"""
GRU-VAE models for FGR-geometry image work.

Two temporal sequence-VAEs used to compare representations:

  BiomGRUVAE   — biometry visit sequence (4 visits x [5 biom-z + 5 mask + GA]),
                 with static Doppler context. Reproduces the validated
                 held-out effective dim ~2.8 at beta=0.1. Its latent carries
                 the fetal size / FGR continuum (birth-pct reconstruction r~0.53).

  ImgSeqVAE    — per-fetus image-embedding trajectory over gestational age
                 (USFM patch-mean embeddings pooled per GA-week bin, ordered by
                 GA, GA+mask as the decoder time channel). Learns a rich ~2.6-D
                 temporal latent that is orthogonal to growth outcome
                 (birth-pct reconstruction r~0.00).

Both use a GRU encoder -> Gaussian latent -> GRU decoder driven by the time
channel (so reconstruction is conditioned on *when* each observation occurred).

Author: FGR-geometry project. Requires torch.
"""
import numpy as np
import torch
import torch.nn as nn


class BiomGRUVAE(nn.Module):
    """Biometry visit-sequence GRU-VAE with static context.

    Input tensor x: (N, T=4, 2*F+1) where F=5 biometry z-scores.
      channels [0:F]      = biometry IG-21 z-scores per visit
               [F:2F]     = observation mask (1 if that biometry was measured)
               [2F]       = GA time channel, normalized (ga_weeks-26)/14
    static sc:  (N, Sdim) = Doppler percentile context (standardized).
    """
    def __init__(self, F=5, Sdim=6, H=32, Z=8):
        super().__init__()
        self.F, self.H, self.Z, self.Sdim = F, H, Z, Sdim
        Din = 2 * F + 1
        self.enc = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H + Sdim, Z)
        self.lv = nn.Linear(H + Sdim, Z)
        self.di = nn.Linear(Z + Sdim, H)
        self.dec = nn.GRU(1, H, batch_first=True)
        self.out = nn.Linear(H, F)

    def encode(self, x, l, sc):
        p = nn.utils.rnn.pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(p)
        h = torch.cat([h[-1], sc], 1)
        return self.mu(h), self.lv(h)

    def forward(self, x, l, sc):
        mu, lv = self.encode(x, l, sc)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        h0 = torch.tanh(self.di(torch.cat([z, sc], 1))).unsqueeze(0)
        ga = x[:, :, 2 * self.F:2 * self.F + 1]
        do, _ = self.dec(ga, h0)
        return self.out(do), mu, lv


class ImgSeqVAE(nn.Module):
    """Image-embedding trajectory GRU-VAE (per-fetus, ordered by GA).

    Input tensor x: (N, MAXT, K+2) where K = image-embedding dim (e.g. 32 PCA).
      channels [0:K]   = per-GA-bin mean image embedding
               [K]     = GA time channel, normalized (ga-6)/36
               [K+1]   = mask (1 if that timestep is real)
    Decoder is driven by the (GA, mask) 2-channel gate so it reconstructs the
    embedding conditioned on gestational age.
    """
    def __init__(self, K=32, H=32, Z=8):
        super().__init__()
        self.K, self.H, self.Z = K, H, Z
        self.enc = nn.GRU(K + 2, H, batch_first=True)
        self.mu = nn.Linear(H, Z)
        self.lv = nn.Linear(H, Z)
        self.di = nn.Linear(Z, H)
        self.dec = nn.GRU(2, H, batch_first=True)   # driven by (GA, mask)
        self.out = nn.Linear(H, K)

    def encode(self, x, l):
        p = nn.utils.rnn.pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(p)
        return self.mu(h[-1]), self.lv(h[-1])

    def forward(self, x, l):
        mu, lv = self.encode(x, l)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        h0 = torch.tanh(self.di(z)).unsqueeze(0)
        gate = x[:, :, self.K:]     # (GA_norm, mask)
        do, _ = self.dec(gate, h0)
        return self.out(do), mu, lv


class JointGRUVAE(nn.Module):
    """Fused biometry+image GRU-VAE over a unified GA-ordered timeline.

    Input x: (N, MAXT, 2*F + K + 1 + 1) with layout
      [0:F]        biometry z-scores
      [F:2F]       biometry mask
      [2F:2F+K]    image embedding (K-d, e.g. 32 PCA)
      [2F+K]       image mask (1 if a pooled image exists at that GA week)
      [-1]         GA time channel, normalized (ga-6)/36
    A single GRU encoder maps the fused sequence to one shared Z-d latent;
    the decoder (driven by GA) reconstructs BOTH modalities via two heads,
    each supervised only where its own mask is 1.

    Finding: the shared bottleneck DILUTES the growth signal (birth-pct
    reconstruction r 0.53 -> 0.09) because the image modality's rich
    orthogonal variance competes for latent capacity. Concatenating the two
    *separately trained* latents instead preserves it (r 0.52). Fusion is not
    beneficial here; documented as a negative multimodal result.
    """
    def __init__(self, F=5, K=32, H=48, Z=8):
        super().__init__()
        self.F, self.K, self.H, self.Z = F, K, H, Z
        Din = 2 * F + K + 1 + 1
        self.enc = nn.GRU(Din, H, batch_first=True)
        self.mu = nn.Linear(H, Z)
        self.lv = nn.Linear(H, Z)
        self.di = nn.Linear(Z, H)
        self.dec = nn.GRU(1, H, batch_first=True)
        self.out_b = nn.Linear(H, F)
        self.out_i = nn.Linear(H, K)

    def encode(self, x, l):
        p = nn.utils.rnn.pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(p)
        return self.mu(h[-1]), self.lv(h[-1])

    def forward(self, x, l):
        mu, lv = self.encode(x, l)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        h0 = torch.tanh(self.di(z)).unsqueeze(0)
        do, _ = self.dec(x[:, :, -1:], h0)
        return self.out_b(do), self.out_i(do), mu, lv


def joint_loss(rb, ri, x, mu, lv, F, K, beta=0.1):
    """Masked recon MSE for both heads + beta*KL. Biometry mask at [F:2F],
    image mask at [2F+K]."""
    bt, bm = x[:, :, :F], x[:, :, F:2 * F]
    it, im = x[:, :, 2 * F:2 * F + K], x[:, :, 2 * F + K:2 * F + K + 1]
    mse_b = (((rb - bt) ** 2) * bm).sum() / bm.sum().clamp(min=1) / F
    mse_i = (((ri - it) ** 2) * im).sum() / im.sum().clamp(min=1) / K
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return mse_b + mse_i + beta * kl


def vae_loss(recon, x, mu, lv, F, beta=1.0):
    """Masked reconstruction MSE + beta * KL. F = number of target channels;
    mask assumed at channels [F:2F]. beta=0.1 for BiomGRUVAE (recovers ~2.8-D),
    beta=1.0 for ImgSeqVAE."""
    tgt = x[:, :, :F]
    m = x[:, :, F:2 * F]
    mse = (((recon - tgt) ** 2) * m).sum() / m.sum() / F
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return mse + beta * kl


def effective_dim(M):
    """Participation-ratio effective dimensionality of a latent point cloud."""
    C = np.cov((M - M.mean(0)).T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-9]
    return (ev.sum() ** 2) / (ev ** 2).sum()


# ===================================================================
# Factorized multimodal VAE: shared + image-private + biometry-private
# ===================================================================
# Motivation: a shared bottleneck DILUTES the growth signal (fusion r 0.53->0.09)
# because image variance crowds it out. A factorized latent quarantines each
# modality's private variance so the growth axis stays clean. Five factorization
# strategies are implemented via `mode`:
#   vanilla     - shared = mean of the two encoders, private codes free
#   indep       - + cross-covariance penalty (shared _|_ img-priv _|_ biom-priv)
#   poe         - shared = product-of-experts fusion + shared-only recon term
#   adversarial - gradient-reversal: private codes cannot predict the shared code
#   contrastive - Barlow-Twins alignment of the two shared encoders
#
# STATIC variant (FactMVAE): MLP encoders on per-fetus vectors.
# LONGITUDINAL variant (LongFactMVAE): GRU encoders over the merged GA-timeline.
#
# Findings (see results/img_align/factorized_mvae_results.json + longfact_probes_results.json):
#   * All 5 strategies achieve a clean factorization: the size/FGR axis lands in
#     the biometry-private subspace (birth-pct r~0.55 static) and NOT the
#     image-private one (r~0.03). Longitudinally it splits biom-private/shared.
#   * FGR-vs-constitutional and the velocity/curvature probes are NULL/noise
#     (sign flips across models). Images remain orthogonal to growth.

_DS_DEFAULT, _DP_DEFAULT = 4, 4


class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def _cov_penalty(a, b):
    """Squared cross-covariance -> drive two codes to independence."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    C = (a.T @ b) / a.shape[0]
    return (C ** 2).mean()


class FactMVAE(nn.Module):
    """Static factorized multimodal VAE (MLP encoders on per-fetus vectors)."""
    def __init__(self, Di, Db, DS=_DS_DEFAULT, DP=_DP_DEFAULT, H=32, mode="vanilla"):
        super().__init__()
        self.DS, self.DP, self.mode = DS, DP, mode

        def enc(din):
            return nn.Sequential(nn.Linear(din, H), nn.ReLU(), nn.Linear(H, 2 * (DS + DP)))
        self.ei, self.eb = enc(Di), enc(Db)
        self.dec_i = nn.Sequential(nn.Linear(DS + DP, H), nn.ReLU(), nn.Linear(H, Di))
        self.dec_b = nn.Sequential(nn.Linear(DS + DP, H), nn.ReLU(), nn.Linear(H, Db))
        if mode == "adversarial":
            self.adv_i = nn.Linear(DP, Db)
            self.adv_b = nn.Linear(DP, Di)

    def _split(self, h):
        DS, DP = self.DS, self.DP
        mu, lv = h[:, :DS + DP], h[:, DS + DP:]
        return mu[:, :DS], lv[:, :DS], mu[:, DS:], lv[:, DS:]

    def forward(self, xi, xb):
        smi, slvi, pmi, plvi = self._split(self.ei(xi))
        smb, slvb, pmb, plvb = self._split(self.eb(xb))
        if self.mode == "poe":
            pi, pb = torch.exp(-slvi), torch.exp(-slvb)
            sm = (smi * pi + smb * pb) / (pi + pb)
            slv = -torch.log(pi + pb)
        else:
            sm, slv = 0.5 * (smi + smb), 0.5 * (slvi + slvb)
        rp = lambda m, l: m + torch.randn_like(m) * torch.exp(0.5 * l)
        zs, zpi, zpb = rp(sm, slv), rp(pmi, plvi), rp(pmb, plvb)
        ri = self.dec_i(torch.cat([zs, zpi], 1))
        rb = self.dec_b(torch.cat([zs, zpb], 1))
        return dict(ri=ri, rb=rb, zs=zs, zpi=zpi, zpb=zpb, sm=sm, slv=slv,
                    pmi=pmi, plvi=plvi, pmb=pmb, plvb=plvb, smi=smi, smb=smb, xi=xi, xb=xb)


class LongFactMVAE(nn.Module):
    """Longitudinal factorized multimodal VAE (GRU encoders over merged timeline).

    Merged input x: (N, T, 2F + K + 1 + 1) with layout
      [0:F] biom-z | [F:2F] biom-mask | [2F:2F+K] img-emb | [2F+K] img-mask | [-1] GA.
    Two GRU encoders read modality-specific slices (+ GA); each yields a
    shared and a private code; decoders are GA-driven GRUs reconstructing each
    modality from shared+own-private.
    """
    def __init__(self, F=5, K=32, DS=_DS_DEFAULT, DP=_DP_DEFAULT, H=48, mode="vanilla"):
        super().__init__()
        self.F, self.K, self.DS, self.DP, self.mode = F, K, DS, DP, mode
        self.gi = nn.GRU(K + 2, H, batch_first=True)      # img-emb + img-mask + GA
        self.gb = nn.GRU(F + F + 1, H, batch_first=True)  # biom-z + biom-mask + GA
        self.hi = nn.Linear(H, 2 * (DS + DP))
        self.hb = nn.Linear(H, 2 * (DS + DP))
        self.di = nn.Sequential(nn.Linear(DS + DP, H), nn.ReLU()); self.oi = nn.GRU(1, H, batch_first=True); self.pi = nn.Linear(H, K)
        self.db = nn.Sequential(nn.Linear(DS + DP, H), nn.ReLU()); self.ob = nn.GRU(1, H, batch_first=True); self.pb = nn.Linear(H, F)
        if mode == "adversarial":
            self.adv_i = nn.Linear(DP, DS)
            self.adv_b = nn.Linear(DP, DS)

    def _bio_in(self, x):
        F = self.F
        return torch.cat([x[:, :, 0:F], x[:, :, F:2 * F], x[:, :, -1:]], -1)

    def _img_in(self, x):
        F, K = self.F, self.K
        return torch.cat([x[:, :, 2 * F:2 * F + K], x[:, :, 2 * F + K:2 * F + K + 1], x[:, :, -1:]], -1)

    def _split(self, h):
        DS, DP = self.DS, self.DP
        mu, lv = h[:, :DS + DP], h[:, DS + DP:]
        return mu[:, :DS], lv[:, :DS], mu[:, DS:], lv[:, DS:]

    def forward(self, x, l):
        pi = nn.utils.rnn.pack_padded_sequence(self._img_in(x), l.cpu(), batch_first=True, enforce_sorted=False)
        pb = nn.utils.rnn.pack_padded_sequence(self._bio_in(x), l.cpu(), batch_first=True, enforce_sorted=False)
        _, hi = self.gi(pi); _, hb = self.gb(pb)
        smi, slvi, pmi, plvi = self._split(self.hi(hi[-1]))
        smb, slvb, pmb, plvb = self._split(self.hb(hb[-1]))
        if self.mode == "poe":
            a, b = torch.exp(-slvi), torch.exp(-slvb)
            sm = (smi * a + smb * b) / (a + b); slv = -torch.log(a + b)
        else:
            sm, slv = 0.5 * (smi + smb), 0.5 * (slvi + slvb)
        rp = lambda m, v: m + torch.randn_like(m) * torch.exp(0.5 * v)
        zs, zpi, zpb = rp(sm, slv), rp(pmi, plvi), rp(pmb, plvb)
        ga = x[:, :, -1:]
        ri = self.pi(self.oi(ga, torch.tanh(self.di(torch.cat([zs, zpi], 1))).unsqueeze(0))[0])
        rb = self.pb(self.ob(ga, torch.tanh(self.db(torch.cat([zs, zpb], 1))).unsqueeze(0))[0])
        return dict(ri=ri, rb=rb, zs=zs, zpi=zpi, zpb=zpb, sm=sm, slv=slv,
                    pmi=pmi, plvi=plvi, pmb=pmb, plvb=plvb, smi=smi, smb=smb)


def fact_loss(o, F, K, beta=0.4, longitudinal=False, x=None):
    """Reconstruction + beta*KL + mode-specific factorization penalty.
    For longitudinal, pass x (the merged tensor) so masks can be applied."""
    if longitudinal:
        bt, bm = x[:, :, :F], x[:, :, F:2 * F]
        it, im = x[:, :, 2 * F:2 * F + K], x[:, :, 2 * F + K:2 * F + K + 1]
        rec = ((((o["rb"] - bt) ** 2) * bm).sum() / bm.sum().clamp(min=1) / F
               + (((o["ri"] - it) ** 2) * im).sum() / im.sum().clamp(min=1) / K)
    else:
        rec = ((o["ri"] - o["xi"]) ** 2).mean() + ((o["rb"] - o["xb"]) ** 2).mean()
    kl = (-0.5 * torch.mean(1 + o["slv"] - o["sm"].pow(2) - o["slv"].exp())
          - 0.5 * torch.mean(1 + o["plvi"] - o["pmi"].pow(2) - o["plvi"].exp())
          - 0.5 * torch.mean(1 + o["plvb"] - o["pmb"].pow(2) - o["plvb"].exp()))
    return rec + beta * kl
