#!/usr/bin/env python3
"""TopK SPARSE AUTOENCODER over frozen FetalCLIP patch tokens -- with the control arms that decide it.

WHY AN SAE AT ALL. A VQ code assigns each patch to exactly ONE of K classes, so it provably cannot
name a concept present in more than 1/K of patches, and cannot express a patch containing SEVERAL
things at once (tissue type + depth artefact + boundary + gain) -- which an ultrasound patch plausibly
does. A sparse mixture can. This is a DIFFERENT question from the WP2 annotation (which wants
proportions of coarse appearance classes, where 1-of-16 is the right instrument and stays in use).

THE ARM THAT DECIDES EVERYTHING (8a). Sparse dictionaries yield interpretable-LOOKING features even
from RANDOMLY-INITIALISED networks: best-single-unit AUC 0.541 random vs 0.567 trained in the design
agent's simulation. So top-activating-patch grids are NOT evidence. Prior that features will look
nameable: ~0.95, carrying almost no information. Nothing is claimed until the trained-minus-random
gap exists. The random arm uses Heap's RE-RANDOMISED scheme -- per-matrix Gaussians matched to each
trained matrix's mean and variance -- NOT naive default init, which is a weaker null because
parameter norms drift from the trained model.

FIXED BEFORE ANY NUMBER IS COMPUTED (deviating from any of these is a new construction and pays the
multiplicity cost against the project's 218-construction / 584-test ledger):
  LAYER    L18 of 24, FetalCLIP's best for the GA clock (r=0.469). ONE layer, frozen.
           DO NOT CONCATENATE: simulated best-single-unit concept AUC 0.551 (last layer) / 0.556
           (mid layer) / 0.549 (12-layer concat) -- concatenation buys NOTHING, and its atoms spend
           0.20-0.23 of their norm in one dominant layer vs 0.083 at chance, i.e. the atoms index
           LAYER IDENTITY, an artefact that would look like a finding.
  GRID     M=4096 (4x expansion on 1024-d), k=16. ONE grid point.
  ARCH     TopK (Gao et al. 2406.04093) -- sets sparsity directly, no L1-coefficient sweep, and its
           auxiliary loss keeps dead latents low. NOT L1/ReLU (shrinkage-biased).
  PREP     per-dim standardise, then SUBTRACT THE PER-FRAME TOKEN MEAN. That drift term is what
           produced layer-identity atoms in simulation.
  SEEDS    3. A feature is REPORTABLE only at min-pairwise cosine >= 0.7 across all three on its
           per-frame activation profile (Paulo & Belrose 2501.16615: only ~30% feature overlap
           between seeds at 131k latents, and TopK is MORE seed-dependent than L1).
  SPLITS   GroupKFold by fetus, 5 folds; SAE fitted inside training folds only.

CONTROL ARMS -- the denominator of every reported number, not an appendix:
  8a RANDOM-WEIGHT  re-randomised FetalCLIP, matched per-matrix moments. THE dispositive arm.
  8b SHUFFLED       trained encoder, patches spatially permuted within frame: kills anatomy,
                    preserves marginal token statistics.
  8c CROSS-ENCODER  a feature needs a counterpart on USF-MAE or USFM at cosine >= 0.5. NECESSARY BUT
                    NOT SUFFICIENT -- this project's mimic effect reached Mantel r=0.70 between
                    streams independent BY CONSTRUCTION. Dispositive only JOINTLY with 8a.
  8d DOMAIN CONTRAST DINOv2 is natural-image-trained, the other three ultrasound. Directional
                    prediction stated in advance, ALLOWED TO FAIL.
  BASELINES         PCA-64 on the same tokens, run alongside. If PCA names our concepts as well as
                    the SAE does, the SAE has not earned the switch. (ICA deliberately NOT built on
                    -- ruled out earlier for the tabular latent.)

KILL CRITERION -- ANY ONE ends the track, and each is itself publishable:
  1. fewer than 10 features replicate across all 3 seeds at cosine >= 0.7
  2. trained-minus-random best-single-unit AUC gap > 0.05 for fewer than 10 replicated features
     (bar deliberately set ABOVE the simulation's concept-aligned gap of +0.026, i.e. a threshold a
     weakly-aligned encoder would fail)
  3. trained vs random arms indistinguishable on activation-entropy over spatial position and plane
     (KS, alpha=0.01) -- Heap's ONLY working separator failing
  4. alive fraction < 0.25 of M on held-out patches
  NOT a kill: failing to beat VQ on GA or any outcome. That is the EXPECTED result and does not bear
  on the interpretability claim. Conversely a prediction win does NOT save the track if 1-4 fail.

MULTIPLICITY -- the trap specific to M=4096. The best of 4096 features correlates with something by
chance: simulated label-permutation max-statistic at n=908 gives |r| bars 0.145 (95th) / 0.155 (99th)
at M=4096, vs 0.129 at M=320. NEVER report a per-feature p-value; only the permutation max-statistic
(>=2000 perms, permuted at FETUS level). And 'SAE beats VQ' needs per-fetus r >= 0.68, not 0.63,
because paired-bootstrap resolution near r=0.615 at n=908 excludes zero only at |delta| >~ 0.065.

USAGE:
  python hpc_sae.py --check                  # shapes, one batch, no fitting
  python hpc_sae.py --arm trained            # main arm
  python hpc_sae.py --arm random             # 8a, the dispositive control
  python hpc_sae.py --arm shuffled           # 8b
  python hpc_sae.py --compare                # seed replication, trained-vs-random gap, kill verdict
Outputs: out_probe/sae_<arm>_M4096.json, out_usfmae/sae_<arm>_M4096.npz
"""
import os, sys, json, time, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
OUT=os.environ.get("GA_OUT_DIR",os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
LAYER=18-1          # L18 of 24, 0-indexed; FetalCLIP's best layer for the GA clock (r=0.469)
M=4096; TOPK=16; SEEDS=3; BATCH=32; EPOCHS=4
FIT_ROWS=1_000_000   # rows used to FIT (244 per dictionary element); assignment stays full-cohort
NAMING_BAR={"95":0.145,"99":0.155}      # label-permutation max-statistic at n=908, M=4096
KILL={"min_replicated":10,"min_gap":0.05,"min_alive_frac":0.25}
from hpc_extract_4encoders import BUILDERS, frame_table
from hpc_crossenc_factvq import patch_tokens

# ---------------- TopK SAE (Gao et al. 2406.04093) ----------------
class TopKSAE(nn.Module):
    """encoder -> TopK sparsity -> decoder. TopK sets L0 exactly, so there is no L1 coefficient to
    sweep (and no shrinkage bias). Auxiliary loss reconstructs the residual from the top dead
    latents, which is what keeps the dictionary alive at scale."""
    def __init__(s,d,m=M,k=TOPK,seed=0):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        s.k=k; s.m=m
        W=torch.randn(m,d,generator=g)/np.sqrt(d)
        s.W_enc=nn.Parameter(W.clone()); s.W_dec=nn.Parameter(W.clone().T)
        s.b_enc=nn.Parameter(torch.zeros(m)); s.b_pre=nn.Parameter(torch.zeros(d))
        s.register_buffer("fires",torch.zeros(m))
    def encode(s,x):
        z=(x-s.b_pre)@s.W_enc.T+s.b_enc
        v,i=torch.topk(z,s.k,dim=-1)
        v=torch.relu(v)
        out=torch.zeros_like(z).scatter_(-1,i,v)
        return out,i
    def forward(s,x):
        a,i=s.encode(x)
        xh=a@s.W_dec.T+s.b_pre
        return xh,a,i
    @torch.no_grad()
    def aux(s,x,xh,a,k_aux=256):
        """Gao auxiliary loss: reconstruct the residual using only DEAD latents."""
        dead=(s.fires==0)
        if dead.sum()<k_aux: return torch.tensor(0.0,device=x.device)
        r=x-xh
        z=(x-s.b_pre)@s.W_enc.T+s.b_enc
        z=z.masked_fill(~dead,-1e9)
        v,i=torch.topk(z,min(k_aux,int(dead.sum())),dim=-1)
        aa=torch.zeros_like(z).scatter_(-1,i,torch.relu(v))
        return ((aa@s.W_dec.T)-r).pow(2).mean()

def rerandomise(model,seed=0):
    """Heap's re-randomised null: per-matrix Gaussians MATCHED to each trained matrix's mean and
    variance. Not default init -- default init is a weaker null because parameter norms drift from
    the trained model, which their L1-norm result shows."""
    g=torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim()>=2:
                mu,sd=p.mean().item(),p.std().item()
                p.copy_((torch.randn(p.shape,generator=g)*sd+mu).to(p.device))
    return model

@torch.no_grad()
def stream_tokens(df,arm,enc="FetalCLIP",seed=0):
    """GENERATOR over batches of (n_rows, D) fp16 CPU tokens at L18, per-frame-mean removed.

    MEMORY HISTORY -- two OOMs before this design, both mine:
      attempt 1: held all 5.2M rows x 1024 in float32 = 21.4 GB, and torch.cat plus a copy-based
                 standardisation took the peak to 43-64 GB.
      attempt 2: fp16 halved the STORED size to 10.7 GB but left both doublings intact --
                 torch.cat(feats) allocates a second full copy while the list is still alive, and
                 X.float().mean(0) materialises a 21 GB float32 temporary just to compute a mean.
                 Died at ~43 GB peak.
    THIS design never holds the full set: tokens are yielded batch by batch and the caller keeps only
    what it needs. Fitting keeps a 1M-row random subsample (2.0 GB fp16); profiles are computed in a
    SECOND streaming pass after fitting, pooling each batch to per-frame immediately and discarding
    the tokens. Peak is the fit subsample plus one batch. Cost: two passes per arm (~18 min not 9)."""
    m,tf,_=BUILDERS[enc]()
    if arm=="random": m=rerandomise(m,seed=seed)
    t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p_).convert("RGB")) for p_ in bs["img"]]).to(DEV)
        t=patch_tokens(enc,m,x)[:,LAYER]                       # (B,Np,D)
        if arm=="shuffled":                                    # 8b: permute patches WITHIN frame
            idx=torch.argsort(torch.rand(t.shape[0],t.shape[1],device=t.device),dim=1)
            t=torch.gather(t,1,idx.unsqueeze(-1).expand_as(t))
        t=t-t.mean(1,keepdim=True)                             # per-frame token mean (drift term)
        yield b0,len(bs),t.shape[1],t.reshape(-1,t.shape[-1]).half().cpu()
        if (b0//BATCH)%50==0: print(f"    {arm} {b0}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    del m; torch.cuda.empty_cache() if DEV=="cuda" else None

def fit_subsample(df,arm,fit_rows=None,seed=0):
    """collect a RANDOM subsample of patch rows for fitting, plus streaming mean/std.
    Statistics are accumulated as running sums in float64 -- never a full-size float32 copy."""
    fit_rows=fit_rows or FIT_ROWS
    rng=np.random.default_rng(seed); keep=[]; kept=0; seen=0
    ssum=None; ssq=None; npatch=None
    for b0,nb,npa,X in stream_tokens(df,arm,seed=seed):
        npatch=npa
        f=X.float()
        ssum=f.sum(0).double() if ssum is None else ssum+f.sum(0).double()
        ssq=(f*f).sum(0).double() if ssq is None else ssq+(f*f).sum(0).double()
        seen+=X.shape[0]
        # reservoir-style: take a fixed fraction of every batch so the subsample spans the cohort
        take=max(1,int(round(X.shape[0]*fit_rows/ (len(df)*npa))))
        sel=rng.choice(X.shape[0],min(take,X.shape[0]),replace=False)
        keep.append(X[sel]); kept+=len(sel)
        del X,f
    mu=(ssum/seen).float(); sd=(ssq/seen-(ssum/seen)**2).clamp_min(1e-12).sqrt().float()
    Xf=torch.cat(keep); keep.clear()
    Xf.sub_(mu.half()).div_((sd+1e-6).half())
    print(f"  {arm}: fit subsample {tuple(Xf.shape)} = {Xf.numel()*2/1e9:.1f}GB "
          f"({Xf.shape[0]/M:.0f} rows per dictionary element); stats from all {seen:,} rows",flush=True)
    return Xf,mu,sd,npatch

@torch.no_grad()
def stream_profiles(df,arm,sae,mu,sd,npatch,seed=0):
    """SECOND pass: per-frame profiles + SPATIAL SUMMARIES. Tokens pooled and discarded per batch.

    WHY SPATIAL SUMMARIES. max/mean pooling over the 256 patches answers "how much did this feature
    fire in this frame" but throws away WHERE -- whether it fired in a tight cluster or scattered,
    and in which region. Keeping all 256 values per frame is 86 GB (that is what OOM'd twice), so
    instead we keep three cheap statistics that preserve the spatial information that matters:
      centroid (row,col)  weighted mean position of the feature's activation on the 16x16 grid
      dispersion          activation-weighted RMS distance from that centroid: SMALL = the feature
                          fires on a localised structure, LARGE = it fires diffusely (which is the
                          signature of an acquisition/texture feature rather than an anatomical one)
      position histogram  cohort-level, per feature: how often each grid cell is in the feature's
                          top quartile. One 256 x M matrix, not per frame -- 4 MB.
    Cost ~1.0 GB on disk, ~0 extra peak RAM (all accumulated per batch).
    MULTIPLICITY: these are DECLARED as spatial descriptors, NOT as new endpoints -- they characterise
    features, they are not additional tests against the WP2 axis or any outcome."""
    gg=int(round(npatch**0.5))
    rr,cc=torch.meshgrid(torch.arange(gg,device=DEV).float(),
                         torch.arange(gg,device=DEV).float(),indexing="ij")
    rr=rr.reshape(-1); cc=cc.reshape(-1)                       # (npatch,)
    mx=np.zeros((len(df),M),np.float32); mn=np.zeros((len(df),M),np.float32)
    cen=np.zeros((len(df),2,M),np.float32); disp=np.zeros((len(df),M),np.float32)
    pos=torch.zeros(npatch,M,device=DEV); topq=torch.zeros(npatch,M,device=DEV); nfr=0
    for b0,nb,npa,X in stream_tokens(df,arm,seed=seed):
        xb=((X.float()-mu)/(sd+1e-6)).to(DEV)
        a,_=sae.encode(xb)
        A=a.reshape(nb,npa,M)                                  # (B, npatch, M)
        mx[b0:b0+nb]=A.max(1).values.cpu().numpy(); mn[b0:b0+nb]=A.mean(1).cpu().numpy()
        w=A/(A.sum(1,keepdim=True)+1e-9)                       # activation weights over position
        cr=(w*rr[None,:,None]).sum(1); ccl=(w*cc[None,:,None]).sum(1)
        cen[b0:b0+nb,0]=cr.cpu().numpy(); cen[b0:b0+nb,1]=ccl.cpu().numpy()
        d2=(rr[None,:,None]-cr[:,None,:])**2+(cc[None,:,None]-ccl[:,None,:])**2
        disp[b0:b0+nb]=(w*d2).sum(1).clamp_min(0).sqrt().cpu().numpy()
        pos+=A.sum(0)
        thr=A.quantile(0.75,dim=1,keepdim=True)                # per frame per feature
        topq+=(A>=thr).float().sum(0)
        nfr+=nb; del X,xb,a,A,w,d2
    pos=(pos/max(nfr,1)).cpu().numpy(); pos=pos/(pos.sum(0,keepdims=True)+1e-9)
    return (mx,mn,-(pos*np.log(pos+1e-12)).sum(0),cen,disp,
            (topq/max(nfr,1)).cpu().numpy().astype(np.float32))


def fit_sae(X,seed,epochs=EPOCHS,bs=8192,lr=3e-4,m=None):
    """Fit the TopK SAE. X is the already-subsampled fp16 CPU tensor from fit_subsample(); batches
    are cast to float32 on device so training stays full precision.

    m OVERRIDES the dictionary size EXPLICITLY. TopKSAE's default argument m=M was captured at
    DEFINITION time, so mutating the global M did nothing and the first M-sweep draft would have
    silently fit M=4096 three times while reporting three different sizes. An assertion on
    W_dec's shape now catches that class of error."""
    sae=TopKSAE(X.shape[1],m=m or M,seed=seed).to(DEV)
    opt=torch.optim.Adam(sae.parameters(),lr=lr)
    n=X.shape[0]; g=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm=torch.randperm(n,generator=g); tot=0.0; sae.fires.zero_()
        for a in range(0,n,bs):
            xb=X[perm[a:a+bs]].to(DEV).float()
            xh,act,idx=sae(xb)
            loss=(xh-xb).pow(2).mean()+(1/32)*sae.aux(xb,xh,act)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): sae.fires.scatter_add_(0,idx.reshape(-1),torch.ones(idx.numel(),device=DEV))
            tot+=loss.item()*len(xb)
        alive=float((sae.fires>0).float().mean())
        print(f"      seed{seed} ep{ep} loss {tot/n:.4f} alive {alive:.3f}",flush=True)
    return sae

def m_sweep(df,arm,Ms=(256,1024,4096),seeds=3,seed_fit=0):
    """SEED-STABILITY SWEEP over dictionary size M. Fitting only -- no profiles, no GA, no outcome.

    WHY. At M=4096 the dictionary is NOT identifiable on this cohort: median matched cosine across
    seeds 0.098, only 8.4% of features at cosine>=0.7 and 0.32% at >=0.9, while all three seeds landed
    on the same loss to four decimals (0.2187/0.2186/0.2190). Many different decompositions reconstruct
    equally well and initialisation picks arbitrarily. For contrast the K=16 VQ codebook on the SAME
    encoders reached across-seed AMI 0.710-0.797. So the question is not 'is the SAE interpretable'
    but 'at what dictionary size does it become IDENTIFIABLE at all'.

    Note the earlier kill bar ('fewer than 10 features replicate') was miscalibrated -- 10 of 4096 is
    0.24% and cannot realistically fail. The bar used here is the MEDIAN matched cosine plus the
    fraction at >=0.7, which describe the typical feature rather than the best few.

    Stability is measured on DECODER DIRECTIONS (what an SAE feature IS: a direction in the encoder's
    activation space), greedily matched across seeds by cosine, min over seed pairs. That needs no
    per-frame profiles, so this is fitting cost only."""
    import itertools
    Xf,mu,sd,npatch=fit_subsample(df,arm,seed=seed_fit)
    out={}
    for m in Ms:
        Ws=[]
        for sN in range(seeds):
            sae=fit_sae(Xf,seed=sN,m=m)                     # m passed EXPLICITLY, no global mutation
            assert sae.W_dec.shape[1]==m, f"dictionary size not honoured: {sae.W_dec.shape}"
            W=sae.W_dec.detach().T                          # (m, d) one row per feature direction
            Ws.append((W/ (W.norm(dim=1,keepdim=True)+1e-9)).cpu())
            del sae
        best=[]
        for i,j in itertools.combinations(range(seeds),2):
            C=(Ws[i]@Ws[j].T).abs()                          # abs: sign of a direction is arbitrary
            best.append(C.max(1).values.numpy())
        rep=np.min(best,axis=0)
        out[m]={"median_matched_cosine":float(np.median(rep)),
                "frac_ge_0.7":float((rep>=0.7).mean()),"frac_ge_0.9":float((rep>=0.9).mean()),
                "n_ge_0.7":int((rep>=0.7).sum()),"M":m}
        print(f"  M={m:5d}: median matched cosine {np.median(rep):.3f} | "
              f">=0.7 {(rep>=0.7).mean():.3f} ({int((rep>=0.7).sum())}/{m}) | >=0.9 {(rep>=0.9).mean():.3f}",flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",default="trained",choices=["trained","random","shuffled"])
    ap.add_argument("--check",action="store_true"); ap.add_argument("--compare",action="store_true")
    ap.add_argument("--clock",action="store_true",help="GA clock / MAE at frame vs study vs fetus aggregation")
    ap.add_argument("--m-sweep",action="store_true",help="seed-stability vs dictionary size M (fitting only)")
    ap.add_argument("--frames",type=int,default=0,help="0 = all")
    a=ap.parse_args()
    df=frame_table()
    if a.frames: df=df.head(a.frames)
    res={"arm":a.arm,"layer":"L18 of 24 (FetalCLIP best for GA clock r=0.469)","M":M,"k":TOPK,
         "seeds":SEEDS,"naming_bar_permutation_maxstat":NAMING_BAR,"kill_criterion":KILL,
         "n_frames":int(len(df)),
         "note":"top-activating-patch grids are NOT evidence until the trained-minus-random gap exists"}

    if a.check:
        m,tf,_=BUILDERS["FetalCLIP"]()
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in df["img"].head(2)]).to(DEV)
        t=patch_tokens("FetalCLIP",m,x)
        print(f"  CHECK tokens {tuple(t.shape)} -> using layer index {LAYER} (L{LAYER+1} of {t.shape[1]})",flush=True)
        mr=rerandomise(BUILDERS["FetalCLIP"]()[0],seed=0)
        tr=patch_tokens("FetalCLIP",mr,x)
        print(f"  CHECK re-randomised arm produces finite tokens: {bool(torch.isfinite(tr).all())} "
              f"| trained std {t[:,LAYER].std():.3f} vs random std {tr[:,LAYER].std():.3f}",flush=True)
        sae=TopKSAE(t.shape[-1]).to(DEV)
        xb=t[:,LAYER].reshape(-1,t.shape[-1]).float().to(DEV)
        xh,act,idx=sae(xb)
        print(f"  CHECK SAE fwd: in {tuple(xb.shape)} recon {tuple(xh.shape)} "
              f"L0 per patch {int((act>0).sum(1).float().mean())} (target {TOPK})",flush=True)
        json.dump(res,open(os.path.join(OUTP,f"sae_check.json"),"w"),indent=2); print("CHECK OK",flush=True); return

    if a.compare:
        import itertools
        from scipy.stats import ks_2samp
        got={}
        for arm in ("trained","random","shuffled"):
            p=os.path.join(OUT,f"sae_{arm}_M{M}.npz")
            if os.path.exists(p): got[arm]=np.load(p,allow_pickle=True)
        res["arms_present"]=list(got)
        if "trained" not in got:
            print("  need the trained arm first",flush=True)
            json.dump(res,open(os.path.join(OUTP,"sae_compare.json"),"w"),indent=2); return
        T=got["trained"]
        # 1. seed replication on per-frame profiles
        P=[T[f"prof_max_s{s}"] for s in range(SEEDS)]
        def cos_match(A,B):
            An=A/ (np.linalg.norm(A,axis=0,keepdims=True)+1e-9); Bn=B/(np.linalg.norm(B,axis=0,keepdims=True)+1e-9)
            C=An.T@Bn; return C.max(1)
        rep=np.min([cos_match(P[i],P[j]) for i,j in itertools.combinations(range(SEEDS),2)],axis=0)
        n_rep=int((rep>=0.7).sum())
        res["seed_replication"]={"n_features_cos_ge_0.7":n_rep,"median_cos":float(np.median(rep))}
        # 4. alive fraction
        res["alive_fraction"]={arm:float(got[arm]["alive"]) for arm in got}
        # 3. Heap separator: activation entropy over spatial position
        ent={arm:got[arm]["spatial_entropy"] for arm in got if "spatial_entropy" in got[arm]}
        if "trained" in ent and "random" in ent:
            ks=ks_2samp(ent["trained"],ent["random"])
            res["heap_separator_spatial_entropy"]={"KS_stat":float(ks.statistic),"p":float(ks.pvalue),
              "distinguishable_at_0.01":bool(ks.pvalue<0.01),
              "trained_median":float(np.median(ent["trained"])),"random_median":float(np.median(ent["random"]))}
        print(f"\n  SEED REPLICATION: {n_rep} of {M} features at cosine >= 0.7 "
              f"(median matched cosine {np.median(rep):.3f}) | kill threshold <{KILL['min_replicated']}",flush=True)
        print(f"  replication curve: "
              +" ".join(f">={t}:{int((rep>=t).sum())}" for t in (0.5,0.6,0.7,0.8,0.9)),flush=True)
        for arm in got: print(f"  alive fraction [{arm}]: {float(got[arm]['alive']):.3f}",flush=True)
        if "heap_separator_spatial_entropy" in res:
            h=res["heap_separator_spatial_entropy"]
            print(f"  Heap separator (spatial entropy) KS={h['KS_stat']:.3f} p={h['p']:.2e} "
                  f"trained median {h['trained_median']:.3f} vs random {h['random_median']:.3f}",flush=True)
        # ULTRA-SPARSE fraction -- 'alive' (fires>0 ever) is near-vacuous at k=16 over 1M rows, where
        # each feature fires ~3900 times by chance. The plan asked for this and it was missing.
        for arm in got:
            if "prof_max_s0" in got[arm]:
                P=got[arm]["prof_max_s0"]; rate=(P>0).mean(0)
                res.setdefault("activation_rates",{})[arm]={
                    "frac_features_firing_lt_0.1pct_of_frames":float((rate<0.001).mean()),
                    "frac_lt_1pct":float((rate<0.01).mean()),"median_rate":float(np.median(rate)),
                    "effective_dictionary_at_1pct":int((rate>=0.01).sum())}
                a_=res["activation_rates"][arm]
                print(f"  activation rates [{arm}]: median {a_['median_rate']:.4f} | "
                      f"ultra-sparse(<0.1% frames) {a_['frac_features_firing_lt_0.1pct_of_frames']:.3f} | "
                      f"EFFECTIVE dict at >=1% = {a_['effective_dictionary_at_1pct']} of {M}",flush=True)
        verdicts=[]
        if n_rep<KILL["min_replicated"]: verdicts.append(f"KILL 1: only {n_rep} features replicate across 3 seeds (<{KILL['min_replicated']})")
        if res["alive_fraction"].get("trained",1)<KILL["min_alive_frac"]: verdicts.append("KILL 4: alive fraction below 0.25")
        if res.get("heap_separator_spatial_entropy",{}).get("distinguishable_at_0.01") is False:
            verdicts.append("KILL 3: trained and random arms indistinguishable on Heap's separator")
        res["VERDICT"]=verdicts or ["no kill triggered on the arms present; the trained-minus-random AUC gap (KILL 2) needs the naming step"]
        for v in res["VERDICT"]: print("  "+v,flush=True)
        json.dump(res,open(os.path.join(OUTP,"sae_compare.json"),"w"),indent=2,default=str)
        print("saved out_probe/sae_compare.json\nDONE",flush=True); return

    if a.m_sweep:
        sw=m_sweep(df,a.arm)
        res["m_sweep"]=sw
        vals=[(m,v["median_matched_cosine"]) for m,v in sw.items()]
        best=max(vals,key=lambda t:t[1])
        res["verdict"]=(f"most identifiable at M={best[0]} (median matched cosine {best[1]:.3f}). "
            +("stability RISES as M falls, so the M=4096 instability is a capacity/identifiability "
              "problem and the smaller M is the defensible grid point."
              if best[0]==min(sw) else
              "stability does NOT improve at smaller M, so the dictionary is not identifiable at any "
              "size tried on this cohort -- the SAE track dies on stability alone, which is itself a "
              "reportable negative given how rarely this control is run."))
        print("\n  "+res["verdict"],flush=True)
        json.dump(res,open(os.path.join(OUTP,f"sae_msweep_{a.arm}.json"),"w"),indent=2,default=str)
        print(f"saved out_probe/sae_msweep_{a.arm}.json\nDONE",flush=True); return

    if a.clock:
        # ---------------- GA CLOCK / MAE EVALUATION ----------------
        # HONEST FRAMING, stated before the numbers: the SAE is NOT expected to be the MAE lever.
        # Failing to beat VQ on GA is the EXPECTED result and does not bear on the interpretability
        # claim (it is explicitly NOT a kill criterion). The real MAE lever is AGGREGATION: the
        # 14.8-day clinical figure is PER FRAME, but a clinician dates a STUDY. Averaging ~22 frames
        # cancels the independent error component. So this arm reports the same features at THREE
        # aggregation levels and attributes any gain to the level, not to the representation.
        #
        # PRE-REGISTERED BAR (from the design plan, simulated): paired-bootstrap resolution near
        # r=0.615 at n=908 excludes zero only at |delta r| >~ 0.065, so "SAE beats VQ" requires
        # per-fetus r >= 0.68, NOT 0.63. Anything smaller is within noise and must be said so.
        # MAE IS REPORTED IN DAYS, per GA window, because a single pooled MAE hides that late
        # gestation is intrinsically harder (obstetric dating error is +-14d at 20-30wk, +-21-24d
        # beyond 30wk; fetal anatomy simply does not specify GA more precisely there).
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from scipy.stats import pearsonr
        p=os.path.join(OUT,f"sae_{a.arm}_M{M}.npz")
        assert os.path.exists(p), f"missing {p} -- fit the arm first"
        z=np.load(p,allow_pickle=True)
        F=z["prof_max_s0"]; nid=z["nid"].astype(str); ga=z["ga"].astype(float)
        m=np.isfinite(ga)&(ga>=6)&(ga<=42); F,nid,ga=F[m],nid[m],ga[m]
        res["clock"]={"note":"MAE in DAYS; aggregation levels compared on IDENTICAL features",
                      "prereg_bar_to_beat_VQ":"per-fetus r >= 0.68 (not 0.63); |delta r| < 0.065 is within noise"}
        levels={}
        # frame level (what the 14.8-day figure was)
        levels["frame"]=(F,ga,nid)
        # study level: frames sharing a fetus AND a GA value are one study
        key=np.array([f"{a_}_{b_:.2f}" for a_,b_ in zip(nid,ga)])
        for lbl,grp in (("study",key),("fetus",nid)):
            u=np.unique(grp)
            Fa=np.vstack([F[grp==g].mean(0) for g in u])
            gaa=np.array([ga[grp==g][0] if lbl=="study" else ga[grp==g].mean() for g in u])
            nida=np.array([nid[grp==g][0] for g in u])
            levels[lbl]=(Fa,gaa,nida)
        for lbl,(Fa,gaa,nida) in levels.items():
            pred=cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,25)),Fa,gaa,
                                   cv=GroupKFold(5),groups=nida)
            r=float(pearsonr(pred,gaa)[0]); mae=float(np.abs(pred-gaa).mean())
            win={}
            for lo,hi in ((6,14),(14,18),(18,24),(24,32),(32,42)):
                w=(gaa>=lo)&(gaa<hi)
                if w.sum()>20: win[f"{lo}-{hi}wk"]={"mae_days":float(np.abs(pred[w]-gaa[w]).mean()*7),"n":int(w.sum())}
            res["clock"][lbl]={"n":int(len(gaa)),"r":r,"mae_wk":mae,"mae_days":mae*7,
                               "ga_sd":float(gaa.std()),"window_mae_days":win}
            print(f"  [{lbl:6s}] n={len(gaa):6d} r={r:.3f} MAE={mae*7:5.1f} days (GA SD {gaa.std():.2f}wk)",flush=True)
            for k,v in win.items(): print(f"            {k}: {v['mae_days']:.1f}d (n={v['n']})",flush=True)
        # baselines on the SAME frames: PCA-64 of the raw tokens is the "did the SAE earn it" test
        res["clock"]["interpretation"]=(
          "Compare r across aggregation levels FIRST. If frame->study is the large jump, the MAE gain "
          "is aggregation, not the SAE, and must be reported that way. Then compare the SAE's "
          "per-fetus r against the VQ K=320 histogram (0.615) using the 0.68 bar.")
        json.dump(res,open(os.path.join(OUTP,f"sae_clock_{a.arm}.json"),"w"),indent=2,default=str)
        print(f"saved out_probe/sae_clock_{a.arm}.json\nDONE",flush=True); return

    # ---- fit one arm, 3 seeds ----
    Xf,mu,sd,npatch=fit_subsample(df,a.arm)
    saves={}; alive=[]; sp_ent=None
    for sN in range(SEEDS):
        sae=fit_sae(Xf,seed=sN)
        alive.append(float((sae.fires>0).float().mean()))
        mx,mn,ent,cen,disp,topq=stream_profiles(df,a.arm,sae,mu,sd,npatch)   # second pass, nothing full-size held
        saves[f"prof_max_s{sN}"]=mx; saves[f"prof_mean_s{sN}"]=mn
        if sN==0:
            saves["spatial_centroid"]=cen        # (n_frames, 2, M) row/col weighted mean position
            saves["spatial_dispersion"]=disp     # (n_frames, M) RMS radius: small=localised, large=diffuse
            saves["topquartile_position"]=topq   # (npatch, M) cohort-level position histogram
        if sN==0:
            sp_ent=ent; saves["W_dec"]=sae.W_dec.detach().cpu().numpy().astype(np.float32)
        del sae
    np.savez(os.path.join(OUT,f"sae_{a.arm}_M{M}.npz"),alive=np.mean(alive),
             spatial_entropy=np.asarray(sp_ent,dtype=np.float32),
             nid=df["nid"].astype(str).values,plane=df["plane_prop"].values,
             ga=df["ga_weeks_recovered"].values,**saves)
    res["alive_fraction_mean"]=float(np.mean(alive))
    res["fit_rows"]=int(Xf.shape[0])
    res["protocol_deviation"]=("fitted on a %d-row random subsample of the 5.2M patch rows, forced by "
        "two OOMs; the pre-registered protocol said all patches. Statistics come from ALL rows and "
        "ASSIGNMENT covers every frame, so per-frame profiles are full-cohort. %d rows is %d per "
        "dictionary element for a 4x-overcomplete M=4096."%(Xf.shape[0],Xf.shape[0],Xf.shape[0]//M))
    print(f"  {a.arm}: alive {np.mean(alive):.3f} -> sae_{a.arm}_M{M}.npz",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"sae_{a.arm}_M{M}.json"),"w"),indent=2,default=str)
    print("DONE",flush=True)

if __name__=="__main__": main()
