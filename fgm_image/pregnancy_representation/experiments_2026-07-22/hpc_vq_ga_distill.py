#!/usr/bin/env python3
"""VQ-VAE codebook DISTILLATION of the GA clock.

Idea: quantize the frozen-encoder embedding into a discrete CODEBOOK (K prototypes), then
predict GA from the code. Trained to distill the continuous GA-clock read (teacher) into a
discrete "maturation-state" vocabulary. NOT expected to beat the continuous clock on
accuracy (quantization loses info) — the payoff is INTERPRETABILITY: do the K codes lay out
as an ordered maturation ladder, and how much accuracy does discretisation cost?

Reads a per-layer summary from the extraction (LS (N, n_layers, 2*dim)); uses one layer's
pooled [CLS, mean-patch] vector as the encoder embedding. Fetus-grouped CV.

USAGE (on HPC, in repo dir):
  python hpc_vq_ga_distill.py                       # default: USF-MAE summary, last layer, K=32
  python hpc_vq_ga_distill.py --summary out_usfmae/summaries.npz --layer -1 --K 32
  python hpc_vq_ga_distill.py --summary out_usfmae/summaries_FetalCLIP.npz --K 64
Outputs: out_probe/vq_ga_<tag>.json  +  out_probe/vq_ga_<tag>.png  (GA-per-code ladder)
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"out_probe"); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"

# ---------------- VQ layer (straight-through + commitment loss) ----------------
class VectorQuantizer(nn.Module):
    """EMA-updated codebook (van den Oord 2017) with dead-code reinitialisation — the two
    standard cures for codebook collapse. Codebook is a BUFFER updated by EMA, NOT by the
    optimiser; only a commitment loss flows to the encoder."""
    def __init__(s, K, dim, beta=0.25, decay=0.99, eps=1e-5, reinit_thresh=1.0):
        super().__init__(); s.K=K; s.dim=dim; s.beta=beta; s.decay=decay; s.eps=eps; s.reinit_thresh=reinit_thresh
        emb=torch.randn(K,dim)
        s.register_buffer("codebook",emb)
        s.register_buffer("cluster_size",torch.zeros(K))
        s.register_buffer("ema_w",emb.clone())
    def forward(s, z):                      # z: (B,dim)
        d=(z.pow(2).sum(1,keepdim=True) - 2*z@s.codebook.t() + s.codebook.pow(2).sum(1))  # (B,K)
        idx=d.argmin(1); zq=s.codebook[idx]
        if s.training:
            with torch.no_grad():
                oh=torch.zeros(z.shape[0],s.K,device=z.device); oh.scatter_(1,idx[:,None],1)  # (B,K)
                n=oh.sum(0)                                   # code usage this batch
                s.cluster_size.mul_(s.decay).add_(n,alpha=1-s.decay)
                dw=oh.t()@z                                   # (K,dim) sum of assigned z
                s.ema_w.mul_(s.decay).add_(dw,alpha=1-s.decay)
                N=s.cluster_size.sum()
                cs=(s.cluster_size+s.eps)/(N+s.K*s.eps)*N
                s.codebook.copy_(s.ema_w/cs[:,None])
                # dead-code reinit: any code used < thresh (EMA) is reseeded to a random batch vector
                dead=s.cluster_size < s.reinit_thresh
                if dead.any():
                    pick=z[torch.randint(0,z.shape[0],(int(dead.sum()),),device=z.device)]
                    s.codebook[dead]=pick; s.ema_w[dead]=pick; s.cluster_size[dead]=1.0
        loss=s.beta*((zq.detach()-z)**2).mean()               # commitment only (codebook is EMA)
        zq=z + (zq-z).detach()                                # straight-through
        return zq, idx, loss

class VQGA(nn.Module):
    """encoder MLP -> VQ codebook -> GA head. Distills the teacher GA read."""
    def __init__(s, d_in, code_dim=64, K=32, hidden=256):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(d_in,hidden),nn.GELU(),nn.Linear(hidden,code_dim))
        s.vq=VectorQuantizer(K,code_dim)
        s.head=nn.Sequential(nn.Linear(code_dim,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x):
        z=s.enc(x); zq,idx,vql=s.vq(z); ga=s.head(zq).squeeze(1)
        return ga, idx, vql

def teacher_clock(X,ga,nid):
    """continuous OOF GA clock — distillation target. PCA-64 + Ridge(a=100) to match the
    stable clinical clock (raw 2048-d Ridge is ill-conditioned on FetalCLIP features)."""
    from sklearn.decomposition import PCA
    pred=np.zeros(len(ga))
    for tr,te in GroupKFold(5).split(X,groups=nid):
        sc=StandardScaler().fit(X[tr]); pca=PCA(64,random_state=0).fit(sc.transform(X[tr]))
        Xtr=pca.transform(sc.transform(X[tr])); Xte=pca.transform(sc.transform(X[te]))
        pred[te]=Ridge(alpha=100).fit(Xtr,ga[tr]).predict(Xte)
    return pred

def train_vq(Xtr,ytr_teacher,ytr_true,Xte, d_in, K, code_dim, epochs=60, lr=1e-3, distill_w=0.7, bs=256):
    # codebook is an EMA buffer -> optimise only encoder+head; minibatches so codebook sees
    # diverse assignments each step (full-batch was a collapse cause).
    m=VQGA(d_in,code_dim,K).to(DEV)
    params=[p for n,p in m.named_parameters() if not n.startswith("vq.")]
    opt=torch.optim.Adam(params,lr)
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=DEV)
    yT=torch.tensor(ytr_teacher,dtype=torch.float32,device=DEV)
    yG=torch.tensor(ytr_true,dtype=torch.float32,device=DEV)
    n=len(Xtr_t)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(n,device=DEV)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad()
            ga,idx,vql=m(Xtr_t[b])
            loss=distill_w*((ga-yT[b])**2).mean() + (1-distill_w)*((ga-yG[b])**2).mean() + vql
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        gate,idxe,_=m(torch.tensor(Xte,dtype=torch.float32,device=DEV))
    return gate.cpu().numpy(), idxe.cpu().numpy(), m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",default=os.path.join(HERE,"out_usfmae","summaries.npz"))
    ap.add_argument("--layer",type=int,default=-1,help="which transformer block (default last)")
    ap.add_argument("--K",type=int,default=32,help="codebook size")
    ap.add_argument("--code_dim",type=int,default=64)
    a=ap.parse_args()
    tag=os.path.basename(a.summary).replace("summaries","").replace(".npz","").strip("_") or "usfmae"
    tag=f"{tag}_L{a.layer}_K{a.K}"
    z=np.load(a.summary,allow_pickle=True,mmap_mode="r")
    LS=np.asarray(z["LS"]); ga=np.asarray(z["ga"]).astype(np.float32); nid=np.asarray(z["nid"]).astype(str)
    plane=np.asarray(z["plane"]).astype(str)
    m=np.isfinite(ga)&(ga>=6)&(ga<=42)
    X=LS[m][:,a.layer,:].astype(np.float32); ga=ga[m]; nid=nid[m]
    print(f"[{tag}] X {X.shape} | {len(set(nid))} fetuses | GA {ga.min():.0f}-{ga.max():.0f}",flush=True)
    # teacher (continuous clock)
    teach=teacher_clock(X,ga,nid)
    r_teacher=pearsonr(teach,ga)[0]; mae_teacher=np.abs(teach-ga).mean()
    print(f"  TEACHER continuous clock: r={r_teacher:.3f} MAE={mae_teacher:.2f}wk",flush=True)
    # VQ student, fetus-grouped OOF
    pred=np.zeros(len(ga)); codes=np.zeros(len(ga),int)
    for tr,te in GroupKFold(5).split(X,groups=nid):
        sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[te])
        p,c,_=train_vq(Xtr,teach[tr],ga[tr],Xte,X.shape[1],a.K,a.code_dim)
        pred[te]=p; codes[te]=c
    r_vq=pearsonr(pred,ga)[0]; mae_vq=np.abs(pred-ga).mean()
    used=len(set(codes)); 
    # GA per code (ladder?)
    code_ga={int(c):float(ga[codes==c].mean()) for c in sorted(set(codes))}
    order=sorted(code_ga,key=code_ga.get)
    # monotonicity of the ladder: spearman of code-rank(by mean GA) vs frames' GA
    from scipy.stats import spearmanr
    rank={c:i for i,c in enumerate(order)}; ladder=np.array([rank[c] for c in codes])
    lad_rho=spearmanr(ladder,ga)[0]
    print(f"  VQ student (K={a.K}): r={r_vq:.3f} MAE={mae_vq:.2f}wk | codes used {used}/{a.K} | ladder spearman {lad_rho:.3f}",flush=True)
    print(f"  cost of discretisation: Δr={r_vq-r_teacher:+.3f} ΔMAE={mae_vq-mae_teacher:+.2f}wk",flush=True)
    res={"tag":tag,"n":int(len(ga)),"teacher":{"r":float(r_teacher),"mae":float(mae_teacher)},
         "vq":{"r":float(r_vq),"mae":float(mae_vq),"K":a.K,"codes_used":used,"ladder_spearman":float(lad_rho)},
         "code_mean_ga":code_ga}
    json.dump(res,open(os.path.join(OUT,f"vq_ga_{tag}.json"),"w"),indent=2)
    # figure: GA-per-code ladder
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        gas=[code_ga[c] for c in order]; ns=[int((codes==c).sum()) for c in order]
        ax.bar(range(len(order)),gas,width=0.8)
        ax.set_xlabel(f"codebook entry (sorted by mean GA) — {used}/{a.K} used")
        ax.set_ylabel("mean GA of frames in code (wk)")
        ax.set_title(f"VQ maturation codebook [{tag}] — ladder ρ={lad_rho:.2f}, VQ r={r_vq:.2f} vs teacher {r_teacher:.2f}")
        fig.tight_layout(); fig.savefig(os.path.join(OUT,f"vq_ga_{tag}.png"),dpi=140,bbox_inches="tight")
        print(f"  figure -> out_probe/vq_ga_{tag}.png",flush=True)
    except Exception as ex: print("  figure skipped:",ex,flush=True)
    print("DONE",flush=True)

if __name__=="__main__": main()
