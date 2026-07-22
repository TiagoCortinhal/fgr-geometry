"""GRU trajectory VAE with 3 variants: (a) unsup baseline, (b) +SupCon, (c) +regression.
Fetus-grouped 5-fold. All structure metrics on HELD-OUT fetuses.
"""
import numpy as np, torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from collections import defaultdict

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def build_sequences(mae_pca, ga, nid, ga_center, ga_scale):
    """Return per-fetus sequences (visits sorted by GA), plus fetus id order."""
    idx_by = defaultdict(list)
    for i, f in enumerate(nid):
        idx_by[f].append(i)
    uniq, fetuses = [], []
    for f, idxs in idx_by.items():
        idxs = sorted(idxs, key=lambda i: ga[i])
        uniq.append(f)
        feat = mae_pca[idxs]                      # (T,P)
        gac = ((ga[idxs] - ga_center) / ga_scale).reshape(-1, 1)
        x = np.concatenate([feat, gac], axis=1)   # (T,P+1)
        fetuses.append(x.astype(np.float32))
    return uniq, fetuses

def pad_batch(seqs, idxs, device):
    sub = [seqs[i] for i in idxs]
    lens = np.array([len(s) for s in sub])
    T = int(lens.max()); P = sub[0].shape[1]; B = len(sub)
    X = np.zeros((B, T, P), np.float32); M = np.zeros((B, T), np.float32)
    for b, s in enumerate(sub):
        X[b, :len(s)] = s; M[b, :len(s)] = 1.0
    return (torch.tensor(X, device=device), torch.tensor(lens),
            torch.tensor(M, device=device))

class TrajVAE(nn.Module):
    def __init__(self, in_dim, hid=64, z=16):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
        self.fc_mu = nn.Linear(hid, z); self.fc_lv = nn.Linear(hid, z); self.z = z
        self.dec = nn.Sequential(nn.Linear(z + 1, hid), nn.ReLU(),
                                 nn.Linear(hid, in_dim - 1))
        self.reg = nn.Linear(z, 1)
        self.proj = nn.Sequential(nn.Linear(z, z), nn.ReLU(), nn.Linear(z, z))

    def encode(self, X, lens):
        packed = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, zsamp, X):
        B, T, _ = X.shape
        ga_t = X[:, :, -1:]
        zrep = zsamp.unsqueeze(1).expand(-1, T, -1)
        return self.dec(torch.cat([zrep, ga_t], dim=-1))

def supcon_loss(feat, labels, temp=0.1):
    device = feat.device
    feat = nn.functional.normalize(feat, dim=1)
    sim = feat @ feat.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    labels = labels.view(-1, 1)
    mask = (labels == labels.t()).float()
    logits_mask = 1 - torch.eye(len(feat), device=device)
    mask = mask * logits_mask
    exp = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    denom = mask.sum(1); valid = denom > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    mean_log = (mask * log_prob).sum(1)[valid] / denom[valid]
    return -mean_log.mean()

def train_fold(seqs_tr, dec_tr, bp_tr, in_dim, variant, device,
               seed=0, epochs=120, hid=64, z=16, beta=0.5,
               w_sup=0.5, w_reg=0.5, temp=0.1, lr=1e-3, batch=128):
    set_seed(seed)
    model = TrajVAE(in_dim, hid, z).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(seqs_tr)
    dec_t = torch.tensor(dec_tr, device=device)
    bp_t = torch.tensor(bp_tr, dtype=torch.float32, device=device)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        kl_w = beta * min(1.0, (ep + 1) / 40)
        for st in range(0, n, batch):
            bidx = perm[st:st + batch]
            X, lens, M = pad_batch(seqs_tr, bidx, device)
            mu, lv = model.encode(X, lens)
            zs = model.reparam(mu, lv)
            recon = model.decode(zs, X)
            tgt = X[:, :, :-1]
            mse = (((recon - tgt) ** 2).mean(-1) * M).sum() / M.sum()
            kl = -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))
            loss = mse + kl_w * kl
            if variant == 'b':
                loss = loss + w_sup * supcon_loss(model.proj(mu), dec_t[bidx], temp)
            elif variant == 'c':
                loss = loss + w_reg * ((model.reg(mu).squeeze(-1) - bp_t[bidx]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model

@torch.no_grad()
def encode_all(model, seqs, device):
    model.eval()
    Z = []
    for st in range(0, len(seqs), 256):
        X, lens, M = pad_batch(seqs, list(range(st, min(st + 256, len(seqs)))), device)
        mu, lv = model.encode(X, lens)
        Z.append(mu.cpu().numpy())
    return np.concatenate(Z, 0)
