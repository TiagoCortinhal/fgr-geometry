#!/usr/bin/env python3
"""DISCRETE VAE (Gumbel-Softmax) over ALL 24 FetalCLIP layers, with a NONLINEAR encoder.

WHAT IS DIFFERENT FROM THE VQ WE ALREADY HAVE.
  VQ-VAE : hard nearest-neighbour assignment; gradients reach the encoder through the
           straight-through estimator, which is BIASED (you pretend quantisation was the identity).
  dVAE   : assignment is a CATEGORICAL DISTRIBUTION, relaxed to be differentiable via Gumbel-Softmax
           with an annealed temperature. Soft during training, argmax at inference. Every code
           receives gradient every step, weighted by its probability -- no straight-through bias.
  This sits BETWEEN the VQ (hard 1-of-K) and the SAE (sparse mixture of 4096), which is the
  principled middle ground for the objection that 1-of-16 cannot express a patch containing several
  things at once -- without the SAE's identifiability failure (median across-seed matched cosine
  0.098 at M=4096, while all three seeds hit the same loss to four decimals).

WHAT I EXPECT, stated before running. dVAE's documented advantage is CODEBOOK UTILISATION -- and that
is NOT our problem: our VQ has 16/16 codes alive on both cohorts and 2477/4096 effective in the SAE.
Dead codes were the failure we AVOIDED (EMA + dead-code reinit), not the one we hit. Our failure was
IDENTIFIABILITY, and a softer gradient estimator changes the optimisation PATH, not the UNIQUENESS of
the optimum: if the objective has a plateau of equivalent solutions, dVAE picks arbitrarily among them
too. So the seed-stability gate below is the whole point of this run.

THE TWO REQUESTED FEATURES AND THEIR COSTS, both handled rather than waved away:

 1. NONLINEAR ENCODER (requested). Encoder is an MLP with GELU + LayerNorm. This is safe and is what
    DALL-E's dVAE does (conv encoder). BUT THE DECODER STAYS LINEAR BY DEFAULT, and that is
    load-bearing: a code's MEANING is its decoder column, i.e. a direction in FetalCLIP's activation
    space. With a nonlinear decoder a code is no longer a point in feature space and cannot be
    described at all -- you would be interpreting your own MLP, the same referent-shift that
    disqualified the distilled-student idea. --nonlinear-decoder is offered for reconstruction/MAE
    work, where interpretability is not claimed.

 2. ALL 24 LAYERS (requested). 24 x 1024 = 24,576-d per patch. Memory: 1M rows would be 49 GB
    (the SAE used 1M rows at 1024-d = 2 GB), so fitting uses --fit-rows 150k = 7.4 GB fp16, still
    9,400 rows per code at K=16. Streamed, never held whole.
    THE REAL RISK IS LAYER IDENTITY, not memory: the design agent's simulation found
    concatenated-layer atoms spend 0.20-0.23 of their norm in ONE dominant layer against 0.083 at
    chance for 12 layers (1/24 = 0.042 here), i.e. atoms index WHICH LAYER rather than what the
    tissue is -- an artefact that reads as a finding. Mitigations, both active:
      per-layer z-scoring (each layer contributes on its own scale), and
      a REPORTED per-code layer-concentration statistic (max share of decoder-column norm falling in
      any single layer, against the 1/24 chance line). If the codes are layer detectors we see it in
      the output instead of discovering it after naming them.

GATE (same as the VQ and SAE): across-seed agreement on the discrete assignment (AMI, chance-corrected,
whole-image permutation reference). VQ reached 0.710-0.803 on both cohorts; the SAE's typical feature
reached 0.098. A dVAE that lands near the VQ earns a place; near the SAE it does not.

USAGE:
  python hpc_dvae.py --check
  python hpc_dvae.py --K 16 --seeds 3                    # all 24 layers, nonlinear encoder
  python hpc_dvae.py --K 32 --n-layers 12                # fewer layers / bigger vocabulary
  python hpc_dvae.py --nonlinear-decoder                 # reconstruction only, NO naming claim
Outputs: out_probe/dvae_K<K>.json, out_usfmae/dvae_codes_K<K>.npz
"""
import os, sys, json, time, argparse, itertools, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import adjusted_mutual_info_score as ami
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
BATCH=32; EPOCHS=6; FIT_ROWS=150_000
TEMP0,TEMP1=1.0,0.1          # Gumbel-Softmax anneal; too cold degenerates to VQ, too warm = blurry
KILL_AMI=0.50
from hpc_extract_4encoders import BUILDERS, frame_table
from hpc_crossenc_factvq import patch_tokens

class DVAE(nn.Module):
    """Nonlinear encoder -> categorical logits -> Gumbel-Softmax -> LINEAR decoder.

    The decoder is a single matrix (K x D): column k IS code k's meaning, a direction in FetalCLIP's
    activation space. That is what makes a code describable. --nonlinear-decoder replaces it with an
    MLP for reconstruction work, at the cost of the codes no longer being interpretable directions."""
    def __init__(s,d,K=16,hidden=(2048,512),seed=0,nonlinear_decoder=False):
        super().__init__(); torch.manual_seed(seed)
        h1,h2=hidden
        s.enc=nn.Sequential(nn.Linear(d,h1),nn.LayerNorm(h1),nn.GELU(),
                            nn.Linear(h1,h2),nn.LayerNorm(h2),nn.GELU(),
                            nn.Linear(h2,K))
        s.K=K
        s.dec=(nn.Sequential(nn.Linear(K,h2),nn.GELU(),nn.Linear(h2,d)) if nonlinear_decoder
               else nn.Linear(K,d,bias=True))
        s.linear_decoder=not nonlinear_decoder
    def forward(s,x,tau,hard=False):
        logits=s.enc(x)
        z=F.gumbel_softmax(logits,tau=tau,hard=hard,dim=-1)
        return s.dec(z),logits,z

@torch.no_grad()
def stream_tokens(df,layers,enc="FetalCLIP"):
    """yield fp16 CPU batches of (rows, n_layers*D), per-frame token mean removed per layer."""
    m,tf,_=BUILDERS[enc]()
    t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        t=patch_tokens(enc,m,x)[:,layers]                     # (B, L, Np, D)
        t=t-t.mean(2,keepdim=True)                            # per-frame mean, per layer
        B,L,Np,D=t.shape
        yield b0,B,Np,t.permute(0,2,1,3).reshape(B*Np,L*D).half().cpu()
        if (b0//BATCH)%50==0: print(f"    tokens {b0}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    del m; torch.cuda.empty_cache() if DEV=="cuda" else None

def fit_subsample(df,layers,fit_rows,L,D,seed=0):
    """random patch subsample for fitting + PER-LAYER streaming mean/std (float64 running sums)."""
    rng=np.random.default_rng(seed); keep=[]; ssum=None; ssq=None; seen=0; npatch=None
    for b0,B,Np,X in stream_tokens(df,layers):
        npatch=Np; f=X.float()
        ssum=f.sum(0).double() if ssum is None else ssum+f.sum(0).double()
        ssq=(f*f).sum(0).double() if ssq is None else ssq+(f*f).sum(0).double()
        seen+=X.shape[0]
        take=max(1,int(round(X.shape[0]*fit_rows/(len(df)*Np))))
        sel=rng.choice(X.shape[0],min(take,X.shape[0]),replace=False)
        keep.append(X[sel]); del X,f
    mu=(ssum/seen).float(); sd=(ssq/seen-(ssum/seen)**2).clamp_min(1e-12).sqrt().float()
    # PER-LAYER z-scoring: rescale each layer block to unit RMS so no layer dominates the distance
    mu_l=mu.view(L,D); sd_l=sd.view(L,D)
    Xf=torch.cat(keep); keep.clear()
    Xf.sub_(mu.half()).div_((sd+1e-6).half())
    lay_rms=Xf.float().view(-1,L,D).pow(2).mean(dim=(0,2)).sqrt()
    print(f"  fit rows {Xf.shape} = {Xf.numel()*2/1e9:.1f}GB | per-layer RMS after z-score: "
          f"min {lay_rms.min():.3f} max {lay_rms.max():.3f} (want ~1 for all -> no layer dominates)",flush=True)
    return Xf,mu,sd,npatch,lay_rms.numpy()

def train(Xf,K,seed,epochs,nonlinear_decoder,bs=2048,lr=1e-3):
    d=Xf.shape[1]; model=DVAE(d,K=K,seed=seed,nonlinear_decoder=nonlinear_decoder).to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    n=Xf.shape[0]; g=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        tau=TEMP0*(TEMP1/TEMP0)**(ep/max(epochs-1,1))          # geometric anneal
        perm=torch.randperm(n,generator=g); tot=0.0; used=torch.zeros(K,device=DEV)
        for a in range(0,n,bs):
            xb=Xf[perm[a:a+bs]].to(DEV).float()
            xh,logits,z=model(xb,tau)
            recon=(xh-xb).pow(2).mean()
            # KL to the uniform categorical: keeps the posterior from collapsing to one code
            q=F.softmax(logits,-1); kl=(q*(q.clamp_min(1e-9).log()-np.log(1.0/K))).sum(-1).mean()
            loss=recon+0.01*kl
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): used+=F.one_hot(logits.argmax(-1),K).float().sum(0)
            tot+=loss.item()*len(xb)
        print(f"      seed{seed} ep{ep} tau {tau:.3f} loss {tot/n:.4f} codes_used {int((used>0).sum())}/{K}",flush=True)
    return model

@torch.no_grad()
def assign(model,df,layers,mu,sd,K):
    """hard argmax assignment over ALL frames (streamed)."""
    labs=[]; nfr=0
    for b0,B,Np,X in stream_tokens(df,layers):
        xb=((X.float()-mu)/(sd+1e-6)).to(DEV)
        labs.append(model.enc(xb).argmax(-1).cpu().numpy().reshape(B,Np).astype(np.int16)); nfr+=B
    return np.concatenate(labs)

def layer_concentration(model,L,D):
    """per code: the max share of its decoder-column L2 norm falling in any ONE layer.
    Chance is 1/L. High values mean the code is a LAYER DETECTOR, not a tissue descriptor --
    the artefact the design agent measured for concatenated layers (0.20-0.23 vs 0.083 chance)."""
    if not model.linear_decoder: return None
    W=model.dec.weight.detach().cpu()                 # (d, K)
    per=W.pow(2).T.view(-1,L,D).sum(-1)               # (K, L) energy per layer
    share=(per/per.sum(1,keepdim=True)).numpy()
    return share.max(1), share

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--K",type=int,default=16); ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--n-layers",type=int,default=24); ap.add_argument("--epochs",type=int,default=EPOCHS)
    ap.add_argument("--fit-rows",type=int,default=FIT_ROWS); ap.add_argument("--check",action="store_true")
    ap.add_argument("--nonlinear-decoder",action="store_true")
    ap.add_argument("--cohort",default="impact",choices=["impact","clinical"])
    a=ap.parse_args()
    df=frame_table() if a.cohort=="impact" else None
    if df is None:
        import clinical_paths
        df,_=clinical_paths.resolve(os.environ.get("CLINICAL_ROOT",
            "/mnt/beegfs/groups/collage/data/IMPACT_CLINICAL/processed"))
        df=df[df["img"].astype(str).str.len()>0].reset_index(drop=True)
    m,tf,_=BUILDERS["FetalCLIP"]()
    x=torch.stack([tf(Image.open(p).convert("RGB")) for p in df["img"].head(2)]).to(DEV)
    with torch.no_grad(): probe=patch_tokens("FetalCLIP",m,x)
    Ltot,Np,D=probe.shape[1],probe.shape[2],probe.shape[3]
    layers=np.linspace(0,Ltot-1,min(a.n_layers,Ltot)).round().astype(int)
    L=len(layers); del m,probe; torch.cuda.empty_cache() if DEV=="cuda" else None
    res={"K":a.K,"seeds":a.seeds,"n_layers":L,"layers":layers.tolist(),"dim_per_patch":int(L*D),
         "encoder":"nonlinear MLP (GELU+LayerNorm) 24576->2048->512->K",
         "decoder":"NONLINEAR (reconstruction only, codes NOT interpretable directions)" if a.nonlinear_decoder
                   else "LINEAR -- code k IS decoder column k, a direction in FetalCLIP activation space",
         "temperature_anneal":[TEMP0,TEMP1],"cohort":a.cohort,
         "gate":f"across-seed AMI on hard assignments; VQ reached 0.710-0.803, SAE 0.098; kill <{KILL_AMI}",
         "layer_identity_chance_line":1.0/L}
    print(f"  {L} layers of {Ltot} -> {L*D:,}-d per patch | K={a.K} | decoder "
          f"{'NONLINEAR' if a.nonlinear_decoder else 'linear'} | {len(df)} frames",flush=True)
    if a.check:
        Xf=torch.zeros(8,L*D).half()
        mdl=DVAE(L*D,K=a.K,nonlinear_decoder=a.nonlinear_decoder).to(DEV)
        xh,lg,z=mdl(Xf.to(DEV).float(),tau=1.0)
        print(f"  CHECK fwd: in {tuple(Xf.shape)} recon {tuple(xh.shape)} logits {tuple(lg.shape)} "
              f"| z row-sum {float(z.sum(-1).mean()):.3f} (Gumbel-Softmax simplex -> 1.0)",flush=True)
        print(f"  CHECK params: {sum(p.numel() for p in mdl.parameters())/1e6:.1f}M",flush=True)
        json.dump(res,open(os.path.join(OUTP,"dvae_check.json"),"w"),indent=2); print("CHECK OK",flush=True); return

    Xf,mu,sd,npatch,lay_rms=fit_subsample(df,layers,a.fit_rows,L,D)
    res["per_layer_rms_after_zscore"]={"min":float(lay_rms.min()),"max":float(lay_rms.max())}
    labs=[]; concs=[]
    for sN in range(a.seeds):
        mdl=train(Xf,a.K,sN,a.epochs,a.nonlinear_decoder)
        labs.append(assign(mdl,df,layers,mu,sd,a.K).reshape(-1))
        lc=layer_concentration(mdl,L,D)
        if lc is not None:
            concs.append(lc[0])
            if sN==0: res["layer_concentration_per_code"]=lc[0].tolist()
        del mdl; torch.cuda.empty_cache() if DEV=="cuda" else None
    pair=[float(ami(labs[i],labs[j])) for i,j in itertools.combinations(range(a.seeds),2)]
    rng=np.random.default_rng(0)
    nfr=len(labs[0])//npatch
    shuf=labs[1].reshape(nfr,npatch)[rng.permutation(nfr)].reshape(-1)
    res["across_seed_AMI"]={"mean":float(np.mean(pair)),"min":float(np.min(pair)),"pairs":pair,
                            "whole_image_perm_ref":float(ami(labs[0],shuf))}
    if concs:
        cc=np.mean(concs,axis=0)
        res["layer_identity"]={"mean_max_layer_share":float(cc.mean()),"chance":1.0/L,
          "ratio_to_chance":float(cc.mean()*L),
          "verdict":("CODES ARE LAYER DETECTORS -- max layer share %.3f is >2x the %.3f chance line; "
                     "the concatenation artefact the design agent measured. Interpret nothing until "
                     "this is resolved."%(cc.mean(),1.0/L)) if cc.mean()>2.0/L else
                    ("acceptable -- max layer share %.3f vs %.3f chance, so codes are not simply "
                     "indexing depth"%(cc.mean(),1.0/L))}
        print(f"  LAYER IDENTITY: mean max-layer share {cc.mean():.3f} vs chance {1.0/L:.3f} "
              f"({cc.mean()*L:.1f}x) -> {res['layer_identity']['verdict'][:60]}",flush=True)
    m_=np.mean(pair)
    res["VERDICT"]=("DVAE NOT IDENTIFIABLE (AMI %.3f < %.2f) -- same failure as the SAE; the softer "
        "gradient estimator did not make the optimum unique."%(m_,KILL_AMI) if m_<KILL_AMI else
        "DVAE STABLE (AMI %.3f) -- comparable to the VQ's 0.710-0.803, so it earns a place alongside it."%m_
        if m_>=0.70 else
        "DVAE MARGINAL (AMI %.3f) -- above the kill line but below the VQ's 0.710-0.803."%m_)
    print(f"\n  across-seed AMI {np.mean(pair):.3f} (min {np.min(pair):.3f}) | "
          f"whole-image-perm ref {res['across_seed_AMI']['whole_image_perm_ref']:.3f}",flush=True)
    print(f"  {res['VERDICT']}",flush=True)
    np.savez(os.path.join(OUT,f"dvae_codes_K{a.K}{'' if a.cohort=='impact' else '_clin'}.npz"),
             codes=labs[0].reshape(nfr,npatch).astype(np.int16),
             nid=df["nid"].astype(str).values,plane=df["plane_prop"].values,
             ga=df["ga_weeks_recovered"].values,layers=layers)
    json.dump(res,open(os.path.join(OUTP,f"dvae_K{a.K}.json"),"w"),indent=2,default=str)
    print(f"saved out_probe/dvae_K{a.K}.json\nDONE",flush=True)

if __name__=="__main__": main()
