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
def tokens_for(df,arm,seed=0,enc="FetalCLIP"):
    """patch tokens at L18, per-dim standardised then per-FRAME mean subtracted."""
    m,tf,_=BUILDERS[enc]()
    if arm=="random": m=rerandomise(m,seed=seed)
    feats=[]; t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        t=patch_tokens(enc,m,x)[:,LAYER]                      # (B,Np,D)
        if arm=="shuffled":                                   # 8b: permute patches WITHIN frame
            idx=torch.argsort(torch.rand(t.shape[0],t.shape[1],device=t.device),dim=1)
            t=torch.gather(t,1,idx.unsqueeze(-1).expand_as(t))
        t=t-t.mean(1,keepdim=True)                            # per-frame token mean (drift term)
        feats.append(t.reshape(-1,t.shape[-1]).float().cpu())
        if (b0//BATCH)%50==0: print(f"    {arm} tokens {b0}/{len(df)} {time.time()-t0:.0f}s",flush=True)
    del m; torch.cuda.empty_cache() if DEV=="cuda" else None
    X=torch.cat(feats)
    return ((X-X.mean(0))/(X.std(0)+1e-6))

def fit_sae(X,seed,epochs=EPOCHS,bs=8192,lr=3e-4):
    sae=TopKSAE(X.shape[1],seed=seed).to(DEV)
    opt=torch.optim.Adam(sae.parameters(),lr=lr)
    n=X.shape[0]; g=torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm=torch.randperm(n,generator=g); tot=0.0; sae.fires.zero_()
        for a in range(0,n,bs):
            xb=X[perm[a:a+bs]].to(DEV)
            xh,act,idx=sae(xb)
            loss=(xh-xb).pow(2).mean()+ (1/32)*sae.aux(xb,xh,act)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): sae.fires.scatter_add_(0,idx.reshape(-1),torch.ones(idx.numel(),device=DEV))
            tot+=loss.item()*len(xb)
        alive=float((sae.fires>0).float().mean())
        print(f"      seed{seed} ep{ep} loss {tot/n:.4f} alive {alive:.3f}",flush=True)
    return sae

@torch.no_grad()
def frame_profiles(sae,X,n_frames,npatch):
    """per-frame feature profile, max AND mean pooled over patches (both pre-specified)."""
    mx=torch.zeros(n_frames,M); mn=torch.zeros(n_frames,M)
    for f in range(0,n_frames,256):
        sl=slice(f*npatch,min((f+256),n_frames)*npatch)
        a,_=sae.encode(X[sl].to(DEV))
        a=a.reshape(-1,npatch,M)
        mx[f:f+a.shape[0]]=a.max(1).values.cpu(); mn[f:f+a.shape[0]]=a.mean(1).cpu()
    return mx.numpy(),mn.numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",default="trained",choices=["trained","random","shuffled"])
    ap.add_argument("--check",action="store_true"); ap.add_argument("--compare",action="store_true")
    ap.add_argument("--clock",action="store_true",help="GA clock / MAE at frame vs study vs fetus aggregation")
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
        verdicts=[]
        if n_rep<KILL["min_replicated"]: verdicts.append(f"KILL 1: only {n_rep} features replicate across 3 seeds (<{KILL['min_replicated']})")
        if res["alive_fraction"].get("trained",1)<KILL["min_alive_frac"]: verdicts.append("KILL 4: alive fraction below 0.25")
        if res.get("heap_separator_spatial_entropy",{}).get("distinguishable_at_0.01") is False:
            verdicts.append("KILL 3: trained and random arms indistinguishable on Heap's separator")
        res["VERDICT"]=verdicts or ["no kill triggered on the arms present; the trained-minus-random AUC gap (KILL 2) needs the naming step"]
        for v in res["VERDICT"]: print("  "+v,flush=True)
        json.dump(res,open(os.path.join(OUTP,"sae_compare.json"),"w"),indent=2,default=str)
        print("saved out_probe/sae_compare.json\nDONE",flush=True); return

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
    X=tokens_for(df,a.arm)
    npatch=X.shape[0]//len(df)
    print(f"  {a.arm}: tokens {tuple(X.shape)} ({npatch} patches/frame)",flush=True)
    saves={}; alive=[]; sp_ent=[]
    for s in range(SEEDS):
        sae=fit_sae(X,seed=s)
        mx,mn=frame_profiles(sae,X,len(df),npatch)
        saves[f"prof_max_s{s}"]=mx.astype(np.float32); saves[f"prof_mean_s{s}"]=mn.astype(np.float32)
        alive.append(float((sae.fires>0).float().mean()))
        if s==0:
            with torch.no_grad():
                act,_=sae.encode(X[:min(len(X),400_000)].to(DEV))
                A=act.reshape(-1,npatch,M).cpu().numpy()
            pos=A.mean(0)                                  # (npatch, M) mean activation per position
            pos=pos/ (pos.sum(0,keepdims=True)+1e-9)
            sp_ent=-(pos*np.log(pos+1e-12)).sum(0)         # entropy over spatial position per feature
            saves["W_dec"]=sae.W_dec.detach().cpu().numpy().astype(np.float32)
    np.savez(os.path.join(OUT,f"sae_{a.arm}_M{M}.npz"),alive=np.mean(alive),
             spatial_entropy=np.asarray(sp_ent,dtype=np.float32),
             nid=df["nid"].astype(str).values,plane=df["plane_prop"].values,
             ga=df["ga_weeks_recovered"].values,**saves)
    res["alive_fraction_mean"]=float(np.mean(alive))
    print(f"  {a.arm}: alive {np.mean(alive):.3f} -> sae_{a.arm}_M{M}.npz",flush=True)
    json.dump(res,open(os.path.join(OUTP,f"sae_{a.arm}_M{M}.json"),"w"),indent=2,default=str)
    print("DONE",flush=True)

if __name__=="__main__": main()
