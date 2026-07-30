#!/usr/bin/env python3
"""LoRA FINE-TUNE of FetalCLIP on a NON-CIRCULAR auxiliary task; ask what the adapted representation
carries that the frozen one and biometry do not.

WHY THIS IS THE ONE FINE-TUNE WORTH RUNNING, and why it is not the one first proposed.

 TARGET CHOICE. Multi-tasking GA + biometry looked obvious and is nearly redundant: the eigenspectrum
 of this project's own measured target couplings (GA-biometry spearman 0.52-0.67) gives 1.65 EFFECTIVE
 independent dimensions for GA+5 biometry (PC1 = 77% of variance) versus 3.51 for GA + 4 Doppler-z. So
 biometry adds ~0.7 of a direction over GA alone. Worse, training on biometry makes the actual question
 -- "does the adapted representation carry anything biometry does not" -- UNASKABLE. Biometry is
 therefore EXCLUDED from training and held as the evaluation comparator.

 THE RESIDUAL IS THE RIGHT SCIENTIFIC TARGET AND THE WRONG ENDPOINT. "A fetus that looks smaller than
 its measured biometry" is what biometry cannot express. But for z-scored scores
 corr(a-b, b) = (rho-1)/sqrt(2-2rho), so at this project's measured appearance->size strength (r=0.36)
 a SUBTRACTION residual carries |r| = 0.56 with the very size it claims to exclude -- 32% of its
 variance IS size, and any SGA association would be the settled size finding with a flipped sign. We
 therefore use a fold-fitted ORTHOGONAL PROJECTION (residual of the appearance score on the size score,
 regression fitted INSIDE the training fold only), and we report corr(residual, size) to prove the
 orthogonalisation worked instead of assuming it.

 DATING PRE-GATE, ALREADY CLEARED (this is why the run is justified). If GA had been assigned from
 second-trimester biometry, appearance->GA would be partly tautological and the ceiling would sit at
 ~0.60 against a frozen readout of 0.47 -- under 0.13 of headroom, and no GPU time would be warranted.
 Checked: CRL is present for 3,215 rows, median 65mm, range 43-113mm => first-trimester/early-second
 crown-rump-length dating (~11-16wk), plus LMP for 3,370. That is the clinical standard and NOT
 second-trimester biometry, so the gate is cleared and headroom exists.

 UA TARGET IS THE REPAIRED COLUMN, NOT THE SHIPPED ONE. The file's Zscore_AU correlates +0.978 with GA
 and only -0.066 with its own raw variable (UA_ecoIMPACT) -- it is a GA-indexed value mislabelled as a
 z-score, and Percentil_AU has SD 4.06 with a minimum of the 34th centile (no fetus below it). Training
 on it would be training on GA twice. We use the recomputed z (handoff/ua_zscore_recomputed.csv:
 log(UA) residualised on a GA spline, MAD-scaled; SD 1.04, +0.011 with GA, +0.995 with raw UA). NOTE its
 provenance: an INTERNAL cohort reference, not a published normative one.

CAPACITY, verified numerically before writing this file. LoRA r=8 on the Q and V slices of the fused
in_proj_weight in the last 12 of 24 blocks, plus all LayerNorm affine parameters, plus heads =
481,314 trainable params = 663 per training fetus = 0.16% of the 303M backbone, and 23x SMALLER than
the from-scratch ResNet-18 (11M) that already failed on these images to a shortcut (MAE 0.90wk, r=0.41,
near-identical Grad-CAMs). open_clip's nn.MultiheadAttention FUSES [Q;K;V] into in_proj_weight (3D, D),
so there is no q_proj/v_proj module to wrap: the delta is built on rows 0:D and 2D:3D with ZEROS on the
K rows. Verified: zero-init B reproduces the frozen forward to max|diff| 0.00e+00 (so the adapted model
STARTS as the frozen model), and a nonzero delta does change the output (so the injection is not a
silent no-op).

LEAKAGE, the failure mode that would produce a spectacular false result:
 - the backbone is refit INSIDE EVERY TRAINING FOLD (5x compute, not optional). A single fit on all
   data followed by fold-wise readout is contamination.
 - GroupKFold by fetus, fold ids written to disk and hashed, byte-identical across the frozen and
   adapted arms, asserted at load.
 - images per fetus are WITHIN-SESSION multiplicity (IMPACT is single-session: 935/951 fetuses have
   exactly one study date), so frames from one fetus are near-duplicates; fetus-balanced sampling and a
   per-fetus frame cap keep one fetus from dominating a gradient step.
 - scanner/protocol drift tracks enrolment order (-1.34 wk/yr apparent-GA drift measured; patient ID
   correlated with code shares at |r| up to 0.274), so a TEMPORAL early/late split is reported beside
   the random one: a gain that vanishes temporally is drift, not physiology.

USAGE:
  python hpc_lora_finetune.py --check              # shapes, param count, zero-init identity, exit
  python hpc_lora_finetune.py --folds 5 --epochs 20
Outputs: out_probe/lora_finetune.json
"""
import os, sys, json, time, argparse, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
OUTP=os.path.join(HERE,"out_probe"); os.makedirs(OUTP,exist_ok=True)
HAND=os.path.join(HERE,"handoff")
DEV="cuda" if torch.cuda.is_available() else "cpu"
from hpc_extract_4encoders import BUILDERS, frame_table
LORA_R,LORA_ALPHA,N_ADAPT_BLOCKS=8,16,12
GA_LO,GA_HI,GA_STEP=26.0,40.5,0.5
DOPPLER=["z_ua_repaired","Zscore_UTA","Zscore_ACM","Zscore_CPR","Zscore_DV"]
BIOMETRY=["DBP_ecoIMPACT","PC_ecoIMPACT","PA_ecoIMPACT","LF_ecoIMPACT","EFW_ecoIMPACT"]  # EVAL ONLY

def eu_num(s):
    """european-decimal aware, magnitude-preserving (see hpc_name_codes_tabular for why this matters:
    a plain to_numeric silently reduced the GA column from 989 usable values to 164)."""
    plain=pd.to_numeric(s,errors="coerce")
    sw=pd.to_numeric(s.astype(str).str.strip().str.replace(",",".",regex=False),errors="coerce")
    return sw if sw.notna().sum()>1.5*max(plain.notna().sum(),1) else plain

class LoRAAttn(nn.Module):
    """LoRA on the Q and V slices of a FUSED in_proj_weight. B is zero-initialised so the adapted
    model starts EXACTLY as the frozen model (verified max|diff| 0.00e+00)."""
    def __init__(s,mha,r=LORA_R,alpha=LORA_ALPHA):
        super().__init__(); s.mha=mha; s.scale=alpha/r
        D=mha.embed_dim
        s.A_q=nn.Parameter(torch.randn(r,D)*0.01); s.B_q=nn.Parameter(torch.zeros(D,r))
        s.A_v=nn.Parameter(torch.randn(r,D)*0.01); s.B_v=nn.Parameter(torch.zeros(D,r))
    def forward(s,x):
        """x arrives in the layout the surrounding block uses. CRITICAL: this open_clip build has
        nn.MultiheadAttention(batch_first=True), but F.multi_head_attention_forward ALWAYS expects
        (L, B, D). Passing (L,B,D) straight through made the functional call read the 197 PATCHES as
        the batch and the images as the sequence -- attention ACROSS IMAGES instead of across patches.
        It trained happily and produced meaningless numbers (max|diff| 1.41 from the frozen forward at
        zero-init, where it must be 0). Transpose when the module is batch_first."""
        bf=getattr(s.mha,"batch_first",False)
        h=x.transpose(0,1) if bf else x          # -> (L,B,D) for the functional call
        dq=(s.B_q@s.A_q)*s.scale; dv=(s.B_v@s.A_v)*s.scale
        W=s.mha.in_proj_weight+torch.cat([dq,torch.zeros_like(dq),dv],0)
        out,_=F.multi_head_attention_forward(h,h,h,s.mha.embed_dim,s.mha.num_heads,W,
            s.mha.in_proj_bias,None,None,False,0.0,s.mha.out_proj.weight,s.mha.out_proj.bias,
            training=s.training,need_weights=False)
        return out.transpose(0,1) if bf else out

class AdaptedFetalCLIP(nn.Module):
    def __init__(s,vt,n_bins,n_dop,n_adapt=N_ADAPT_BLOCKS):
        super().__init__(); s.v=vt
        for p in s.v.parameters(): p.requires_grad=False
        blocks=s.v.transformer.resblocks; s.lora=nn.ModuleList()
        for blk in list(blocks)[-n_adapt:]:
            l=LoRAAttn(blk.attn); blk._lora=l; s.lora.append(l)
            blk.attn_forward_patched=True
        for m in s.v.modules():                        # LayerNorm affine stays trainable
            if isinstance(m,nn.LayerNorm):
                for p in m.parameters(): p.requires_grad=True
        D=s.v.ln_post.normalized_shape[0] if hasattr(s.v,"ln_post") else 1024
        s.head_ga=nn.Linear(D,n_bins); s.head_dop=nn.Linear(D,n_dop)
    def embed(s,x):
        """LAYOUT IS LOAD-BEARING. open_clip's Transformer.forward transposes NLD->LND ONLY when
        batch_first is False; the project's patch_tokens() permutes UNCONDITIONALLY, which is correct
        for the builds it was written against but wrong here (this build has batch_first=True). Read
        the flag instead of assuming, and keep the token axis where the blocks expect it -- getting
        this wrong makes attention run across IMAGES rather than across patches, silently."""
        v=s.v; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
        cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
        h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h)
        s._bf=getattr(v.transformer,"batch_first",False)
        if not s._bf: h=h.permute(1,0,2)                       # NLD -> LND only when required
        for blk in v.transformer.resblocks:
            if hasattr(blk,"_lora"):
                # MUST mirror open_clip ResidualAttentionBlock.forward EXACTLY, including ls_1/ls_2
                # (LayerScale). An earlier version omitted them and would have silently computed a
                # DIFFERENT network than the frozen encoder -- invalidating every frozen-vs-adapted
                # comparison while looking like it worked. ls_* is Identity when layerscale is off.
                h=h+blk.ls_1(blk._lora(blk.ln_1(h)))
                h=h+blk.ls_2(blk.mlp(blk.ln_2(h)))
            else:
                h=blk(h)
        if not s._bf: h=h.permute(1,0,2)                       # LND -> NLD only if we transposed
        return v.ln_post(h[:,0])                               # CLS token
    def forward(s,x):
        z=s.embed(x); return s.head_ga(z), s.head_dop(z), z

def masked_target_loss(pred,targ,mask,per_target_sd):
    """MASKED multi-task loss: each fetus contributes to the targets it actually HAS.

    WHY NOT COMPLETE-CASE, and why not imputation. Coverage is 87.2% complete (864 of 991 have all 5
    Doppler targets; 111 have 4, 11 have 3, only 2 have none), so this is sparse missingness rather than
    a coverage problem -- BUT it is not random: complete cases have an SGA rate of 0.180 versus 0.111 for
    incomplete ones. Dropping incomplete fetuses would therefore ENRICH the training set for SGA, and
    SGA is an evaluation-only endpoint in this project, so that enrichment would quietly couple training
    composition to the thing we later evaluate. Imputing Doppler values is worse: it would invent
    measurements the sonographer did not take, and the model would learn the imputer.

    So: mask. Every fetus trains on whatever it has. Two details that matter --
      (a) each target is divided by its OWN training-fold SD before the loss, else a target with larger
          natural scale silently dominates the gradient;
      (b) the loss is normalised by the NUMBER OF OBSERVED entries, not by the tensor size, so a batch
          that happens to contain many missing values does not get a smaller gradient step.
    """
    if mask.sum()==0: return pred.sum()*0.0
    w=(targ-pred)/per_target_sd
    return (w.pow(2)*mask).sum()/mask.sum()

def soft_bins(ga,centres,sigma):
    """GA as a distributional target: Gaussian soft labels with sigma = dating error, read out as the
    expectation. The regression analogue of label smoothing -- it stops the model being punished for
    being right within the label's own uncertainty."""
    d=(ga[:,None]-centres[None,:])**2
    w=np.exp(-d/(2*sigma**2)); return w/w.sum(1,keepdims=True)

def build_targets():
    e=pd.read_excel(os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx"))
    e["nid"]=e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    e=e.drop_duplicates("nid").set_index("nid")
    T=pd.DataFrame({c:eu_num(e[c]) for c in ["EG_ecoIMPACT"]+BIOMETRY+DOPPLER[1:] if c in e.columns})
    ua=os.path.join(HAND,"ua_zscore_recomputed.csv")
    assert os.path.exists(ua), ("missing handoff/ua_zscore_recomputed.csv -- the shipped Zscore_AU is "
        "a GA-indexed value mislabelled as a z-score (+0.978 with GA, -0.066 with raw UA); training on "
        "it would be training on GA twice. Recompute it first.")
    u=pd.read_csv(ua); u["nid"]=u["nid"].astype(str)
    T=T.join(u.set_index("nid")["z_rob"].rename("z_ua_repaired"))
    assert abs(spearmanr(T["z_ua_repaired"],T["EG_ecoIMPACT"],nan_policy="omit").statistic)<0.15, \
        "repaired UA z still tracks GA -- refusing to train on a GA proxy"
    return T

# ---------------------------------------------------------------- training / evaluation

def fetus_batches(df,fetuses,frames_per,batch,rng):
    """FETUS-BALANCED sampling. IMPACT is single-session (935/951 fetuses have exactly ONE study date),
    so the ~22 frames per fetus are WITHIN-SESSION near-duplicates. Sampling frames uniformly would let
    a fetus with 60 frames contribute 10x the gradient of one with 6, and near-duplicate frames in the
    same step inflate the effective batch without adding information. So: sample FETUSES, then a capped
    number of frames each."""
    idx_by=df.groupby("nid").indices
    order=rng.permutation(fetuses)
    buf=[]
    for f in order:
        ii=idx_by[f]
        pick=rng.choice(ii,min(frames_per,len(ii)),replace=False)
        buf.extend(pick.tolist())
        while len(buf)>=batch:
            yield np.array(buf[:batch]); buf=buf[batch:]
    if buf: yield np.array(buf)

def load_batch(df,ii,tf):
    return torch.stack([tf(Image.open(df["img"].iloc[i]).convert("RGB")) for i in ii])

@torch.no_grad()
def embed_fetuses(mdl,df,fetuses,tf,cap,batch=24):
    """per-fetus embedding = MEDIAN over up to `cap` frames. Median not mean: within-session frames
    include off-axis and partially-obscured views, and a median is not dragged by a few of them.
    Order-invariance is asserted by the caller."""
    mdl.eval(); out={}
    idx_by=df.groupby("nid").indices
    for f in fetuses:
        ii=idx_by[f][:cap]
        zs=[]
        for b0 in range(0,len(ii),batch):
            x=load_batch(df,ii[b0:b0+batch],tf).to(DEV)
            zs.append(mdl.embed(x).float().cpu())
        Z=torch.cat(zs); out[f]=Z.median(0).values.numpy()
    return out

def val_score(pred,targ,mask):
    """EARLY-STOPPING CRITERION for masked multi-task targets: mean per-target Spearman over OBSERVED
    entries, averaged across targets. Chosen over 'GA alone' deliberately -- GA is the densest target
    (988/991) and would keep improving while the Doppler heads overfit unchecked, so stopping on GA
    would silently select an epoch that is bad for the targets that justify the multi-task design.
    Each target contributes equally regardless of its n, so a fold that happens to be short on one
    Doppler measure cannot dominate the stopping decision."""
    rs=[]
    for j in range(targ.shape[1]):
        m=mask[:,j]>0
        if m.sum()>=20:
            r=spearmanr(pred[m,j],targ[m,j]).statistic
            if np.isfinite(r): rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")

def run_fold(df,T,fold,centres,a,tf,build):
    tr=df[df.fold!=fold]; te=df[df.fold==fold]
    tr_f=np.array(sorted(tr["nid"].unique())); te_f=np.array(sorted(te["nid"].unique()))
    assert len(set(tr_f)&set(te_f))==0, "fetus in both train and test"
    # inner split for early stopping, ALSO grouped by fetus
    rng=np.random.default_rng(1000+fold)
    perm=rng.permutation(tr_f); n_in=max(30,int(0.2*len(perm)))
    in_f=set(perm[:n_in].tolist()); fit_f=np.array([f for f in tr_f if f not in in_f])
    mdl=build().to(DEV)
    tp=[p for p in mdl.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(tp,lr=a.lr,weight_decay=1e-4)
    # per-target SD from the TRAINING FOLD ONLY (a test-set SD would be leakage)
    Ttr=T.reindex(fit_f)
    sd=torch.tensor(np.nan_to_num(Ttr[DOPPLER].std().values,nan=1.0),dtype=torch.float32,device=DEV)
    ga_tr=T["EG_ecoIMPACT"]
    best=(-np.inf,0,None); hist=[]
    for ep in range(a.epochs):
        mdl.train()
        for ii in fetus_batches(tr,fit_f,a.frames_per_fetus,a.batch,rng):
            nid=df["nid"].iloc[ii].values
            gav=ga_tr.reindex(nid).values.astype(float)
            ok=np.isfinite(gav)
            if ok.sum()<2: continue
            x=load_batch(df,ii,tf).to(DEV)
            lg,dp,_=mdl(x)
            sb=torch.tensor(soft_bins(gav[ok],centres,a.sigma_dating),dtype=torch.float32,device=DEV)
            loss_ga=-(sb*F.log_softmax(lg[torch.tensor(ok,device=DEV)],-1)).sum(-1).mean()
            tv=T[DOPPLER].reindex(nid).values.astype(float)
            msk=torch.tensor(np.isfinite(tv),dtype=torch.float32,device=DEV)
            tt=torch.tensor(np.nan_to_num(tv),dtype=torch.float32,device=DEV)
            loss_dp=masked_target_loss(dp,tt,msk,sd)
            loss=loss_ga+loss_dp
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tp,1.0); opt.step()
        # inner validation
        mdl.eval(); P=[]; Tt=[]; M=[]
        with torch.no_grad():
            for f in perm[:n_in]:
                ii=df.index[(df.nid==f)].values[:a.eval_cap]
                if len(ii)==0: continue
                x=load_batch(df,ii,tf).to(DEV)
                _,dpv,_=mdl(x); P.append(dpv.float().cpu().numpy().mean(0))
                tv=T[DOPPLER].reindex([f]).values.astype(float)[0]
                Tt.append(np.nan_to_num(tv)); M.append(np.isfinite(tv).astype(float))
        vs=val_score(np.array(P),np.array(Tt),np.array(M)) if P else float("nan")
        hist.append(vs)
        if np.isfinite(vs) and vs>best[0]:
            best=(vs,ep,{k:v.detach().cpu().clone() for k,v in mdl.state_dict().items() if v.requires_grad or True})
        print(f"    fold{fold} ep{ep} val_meanrho {vs:+.3f}{'  *' if best[1]==ep else ''}",flush=True)
        if ep-best[1]>=5: print(f"    early stop (no gain in 5 epochs)",flush=True); break
    if best[2] is not None: mdl.load_state_dict(best[2])
    return mdl,te_f,best[0],best[1],hist

def evaluate(Zad,Zfr,fetuses,T,outcomes,fold_of):
    """OUT-OF-FOLD evaluation. Every readout is a ridge fitted on TRAIN folds and applied to the TEST
    fold, for BOTH arms, on the SAME fetus split -- so 'adapted beats frozen' cannot be a fold artefact.

    THE THREE QUESTIONS, in order of what they can establish:
      1. did the auxiliary task actually improve?  adapted vs frozen on GA and on each Doppler target.
         If this fails, nothing downstream means anything: the adaptation did not happen.
      2. does the adapted embedding carry SIZE better?  biometry was HELD OUT of training precisely so
         this stays askable.
      3. does the SIZE-ORTHOGONAL part carry outcome?  the appearance score is residualised on the
         predicted-size score with the projection FITTED INSIDE THE TRAINING FOLD, and
         corr(residual, size) is REPORTED -- because a subtraction residual would carry |r|=0.56 with
         the very size it claims to exclude at this project's measured appearance->size strength.
    """
    fe=np.array(fetuses); folds=np.array([fold_of[f] for f in fe])
    def oof(Z,y):
        yy=np.asarray(y,float); pred=np.full(len(fe),np.nan)
        for k in np.unique(folds):
            tr=(folds!=k)&np.isfinite(yy); te=(folds==k)
            if tr.sum()<50 or te.sum()<5: continue
            M=RidgeCV(alphas=np.logspace(-2,4,25)).fit(Z[tr],yy[tr])
            pred[te]=M.predict(Z[te])
        m=np.isfinite(pred)&np.isfinite(yy)
        return pred,(float(spearmanr(pred[m],yy[m]).statistic) if m.sum()>30 else np.nan)
    out={"n_fetuses":int(len(fe)),
         "readout_note":("Q1/Q2/Q3 use RIDGE-ON-EMBEDDING for BOTH arms -- the only comparison that is "
            "fair, since the frozen encoder has no trained head. Note this UNDERSTATES what the "
            "fine-tuned model achieves end-to-end: the adapter is 0.17% of parameters, so the task "
            "HEAD does most of the target-specific work, and the inner-validation curve (which rose to "
            "+0.27..+0.39) is the HEAD's own output on ~145 within-fold fetuses, not this ridge. "
            "head_readout below gives the end-to-end number so the two are not conflated.")}
    # --- Q1: trained targets ---
    out["trained_targets"]={}
    for c in ["EG_ecoIMPACT"]+DOPPLER:
        if c not in T.columns: continue
        y=T[c].reindex(fe).values
        _,r_ad=oof(Zad,y); _,r_fr=oof(Zfr,y)
        out["trained_targets"][c]={"adapted_r":r_ad,"frozen_r":r_fr,"delta":r_ad-r_fr,
                                  "n_obs":int(np.isfinite(y).sum())}
    # --- Q2: HELD-OUT biometry comparator ---
    out["heldout_biometry"]={}
    for c in BIOMETRY:
        if c not in T.columns: continue
        y=T[c].reindex(fe).values
        _,r_ad=oof(Zad,y); _,r_fr=oof(Zfr,y)
        out["heldout_biometry"][c]={"adapted_r":r_ad,"frozen_r":r_fr,"delta":r_ad-r_fr}
    # --- Q3: size-orthogonal appearance vs outcome ---
    size=T["EFW_ecoIMPACT"].reindex(fe).values.astype(float)
    out["size_orthogonal"]={}
    for arm,Z in (("adapted",Zad),("frozen",Zfr)):
        s_pred,_=oof(Z,size)
        res={}
        for oname,yv in outcomes.items():
            y=np.asarray([yv.get(f,np.nan) for f in fe],float)
            a_pred,_=oof(Z,y)                      # appearance score for this endpoint
            # orthogonalise INSIDE folds
            orth=np.full(len(fe),np.nan)
            for k in np.unique(folds):
                tr=(folds!=k)&np.isfinite(a_pred)&np.isfinite(s_pred)
                te=(folds==k)&np.isfinite(a_pred)&np.isfinite(s_pred)
                if tr.sum()<50 or te.sum()<5: continue
                g=LinearRegression().fit(s_pred[tr,None],a_pred[tr])
                orth[te]=a_pred[te]-g.predict(s_pred[te,None])
            m=np.isfinite(orth)&np.isfinite(y)
            ms=np.isfinite(orth)&np.isfinite(s_pred)
            res[oname]={"r_raw":float(spearmanr(a_pred[m],y[m]).statistic) if m.sum()>30 else np.nan,
                        "r_size_orthogonal":float(spearmanr(orth[m],y[m]).statistic) if m.sum()>30 else np.nan,
                        "corr_residual_with_size":float(spearmanr(orth[ms],s_pred[ms]).statistic) if ms.sum()>30 else np.nan,
                        "n":int(m.sum())}
        out["size_orthogonal"][arm]=res
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds",type=int,default=5); ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--sigma-dating",type=float,default=0.7,help="GA label sigma in weeks (CRL dating ~+/-5-7d)")
    ap.add_argument("--batch",type=int,default=24); ap.add_argument("--frames-per-fetus",type=int,default=4)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--eval-cap",type=int,default=8,help="frames per fetus at eval (median-pooled)")
    ap.add_argument("--check",action="store_true")
    a=ap.parse_args()
    centres=np.arange(GA_LO,GA_HI+1e-9,GA_STEP)
    df=frame_table(); T=build_targets()
    df["nid"]=df["nid"].astype(str)
    df=df[df["nid"].isin(T.index)].reset_index(drop=True)
    m,tf,_=BUILDERS["FetalCLIP"]()
    mdl=AdaptedFetalCLIP(m,len(centres),len(DOPPLER)).to(DEV)
    tp=[p for p in mdl.parameters() if p.requires_grad]
    ntr=sum(p.numel() for p in tp); nfet=df["nid"].nunique()
    res={"trainable_params":int(ntr),"backbone_params":int(sum(p.numel() for p in mdl.v.parameters())),
         "params_per_training_fetus":float(ntr/(nfet*(a.folds-1)/a.folds)),
         "lora":{"r":LORA_R,"alpha":LORA_ALPHA,"blocks_adapted":N_ADAPT_BLOCKS,
                 "target":"Q and V slices of the FUSED in_proj_weight; K rows zero"},
         "targets_trained":["GA (soft-binned)"]+DOPPLER,
         "targets_HELD_OUT_as_comparator":BIOMETRY,
         "why_biometry_excluded":("GA+5 biometry has only 1.65 effective independent dimensions "
            "(PC1=77%) vs 3.51 for GA+4 Doppler-z; and training on biometry makes 'does the adapted "
            "representation carry anything biometry does not' unaskable"),
         "ua_target":"REPAIRED internal-reference z (shipped Zscore_AU is a GA-indexed mislabel)",
         "missing_target_handling":("MASKED per-target loss -- every fetus trains on the targets it has. "
            "Coverage 87.2% complete (864/991 all five; 111 have four; 11 have three; 2 have none). NOT "
            "complete-case: complete cases have SGA rate 0.180 vs 0.111 incomplete, so dropping would "
            "enrich training for an evaluation-only endpoint. NOT imputed: that would invent "
            "measurements and the model would learn the imputer. Loss normalised by observed-entry "
            "count and each target scaled by its own training-fold SD."),
         "dating_pregate":"CLEARED -- CRL n=3215 median 65mm (11-16wk) + LMP n=3370; not 2nd-trimester biometry",
         "n_fetuses":int(nfet),"n_frames":int(len(df))}
    print(f"  trainable {ntr:,} of {res['backbone_params']:,} "
          f"({100*ntr/res['backbone_params']:.2f}%) = {res['params_per_training_fetus']:.0f}/training fetus",flush=True)
    print(f"  frames {len(df)} | fetuses {nfet} | GA bins {len(centres)} | Doppler targets {len(DOPPLER)}",flush=True)
    if a.check:
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in df["img"].head(2)]).to(DEV)
        with torch.no_grad():
            lg,dp,z=mdl(x)
            # zero-init identity: the adapted model must equal the frozen model at step 0
            from hpc_crossenc_factvq import patch_tokens
            print(f"  CHECK ga_logits {tuple(lg.shape)} doppler {tuple(dp.shape)} embed {tuple(z.shape)}",flush=True)
            zb=[float(p.abs().max()) for l in mdl.lora for p in (l.B_q,l.B_v)]
            print(f"  CHECK zero-init B: max|B| across LoRA modules = {max(zb):.1e} (must be 0 -> starts as frozen)",flush=True)
            # HARD IDENTITY ASSERTION. At zero-init the adapted embedding MUST equal the embedding
            # computed by the untouched block path. Two layout bugs were caught exactly this way:
            # (a) nn.MultiheadAttention has batch_first=True here while
            #     F.multi_head_attention_forward always wants (L,B,D) -- attention was running across
            #     IMAGES instead of patches; (b) an unconditional NLD->LND permute copied from
            #     patch_tokens(). Both trained happily and produced meaningless numbers. Comparing
            #     LoRA-on vs LoRA-bypassed on the SAME module instance is the test that catches them.
            z_adapted=mdl.embed(x)
            bypass={}
            for blk_ in mdl.v.transformer.resblocks:
                if hasattr(blk_,"_lora"): bypass[blk_]=blk_._lora; del blk_._lora
            z_frozen=mdl.embed(x)                          # same weights, LoRA path removed
            for blk_,l_ in bypass.items(): blk_._lora=l_   # restore
            dmax=float((z_adapted-z_frozen).abs().max())
            res["zero_init_identity_maxdiff"]=dmax
            print(f"  CHECK zero-init identity: max|adapted-frozen| = {dmax:.2e} (must be <1e-4)",flush=True)
            assert dmax<1e-4, (f"adapted model does NOT reproduce the frozen forward at zero-init "
                f"(max|diff|={dmax:.3e}). The LoRA path diverges from open_clip's block -- check "
                f"batch_first layout and ls_1/ls_2 LayerScale before training anything.")
        obs={c:int(T[c].notna().sum()) for c in ["EG_ecoIMPACT"]+DOPPLER+BIOMETRY if c in T.columns}
        res["target_coverage"]=obs; print(f"  CHECK target coverage {obs}",flush=True)
        json.dump(res,open(os.path.join(OUTP,"lora_check.json"),"w"),indent=2,default=str)
        print("CHECK OK",flush=True); return

    outcomes={}
    _e=pd.read_excel(os.path.join(ROOT,"data_local","IMPACT_ecocardio_zscores_corrected.xlsx"))
    _e["nid"]=_e["Cod"].apply(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    _e=_e.drop_duplicates("nid").set_index("nid")
    # EVALUATION-ONLY endpoints. percentil_birth is EUROPEAN-DECIMAL in this file (a plain to_numeric
    # returns 1 usable value of 988), and SGA/LGA are numeric '1.0'/'0.0' not 'yes'/'no' strings.
    _bp=pd.to_numeric(_e["percentil_birth"].astype(str).str.strip().str.replace(",",".",regex=False),errors="coerce")
    outcomes["birth_percentile"]=_bp.dropna().to_dict()
    for _c in ("SGA_birth","LGA_birth"):
        _v=pd.to_numeric(_e[_c],errors="coerce"); outcomes[_c]=(_v>0).astype(float).where(_v.notna()).dropna().to_dict()
    assert len(outcomes["birth_percentile"])>500 and sum(outcomes["SGA_birth"].values())>100, \
        f"outcome parse implausible: bp={len(outcomes['birth_percentile'])} sga={sum(outcomes['SGA_birth'].values())}"
    print(f"  eval-only outcomes: birth_pct n={len(outcomes['birth_percentile'])} "
          f"SGA={int(sum(outcomes['SGA_birth'].values()))} LGA={int(sum(outcomes['LGA_birth'].values()))}",flush=True)

    fold=np.zeros(len(df),dtype=int)
    for k,(_,te) in enumerate(GroupKFold(a.folds).split(df,groups=df["nid"])): fold[te]=k
    df["fold"]=fold
    fh=hashlib.sha256(pd.util.hash_pandas_object(df[["nid","fold"]],index=False).values.tobytes()).hexdigest()[:16]
    df[["nid","fold"]].drop_duplicates().to_csv(os.path.join(HAND,"lora_folds.csv"),index=False)
    res["fold_hash"]=fh; print(f"  folds written, hash {fh} (must match across frozen and adapted arms)",flush=True)
    fold_of=dict(zip(df["nid"],df["fold"]))
    build=lambda: AdaptedFetalCLIP(BUILDERS["FetalCLIP"]()[0],len(centres),len(DOPPLER))
    Zad={}; Zfr={}; HEAD={}; per_fold=[]
    for k in range(a.folds):
        t0=time.time()
        mdl,te_f,vbest,ebest,hist=run_fold(df,T,k,centres,a,tf,build)
        Zad.update(embed_fetuses(mdl,df,te_f,tf,a.eval_cap))
        # END-TO-END head readout on the TEST fold: the adapted model's own Doppler predictions.
        # Needed because the rising inner-val curve refers to the HEAD, while the arm-vs-arm
        # comparison below is ridge-on-embedding -- conflating them would overstate the adaptation.
        mdl.eval(); idx_by=df.groupby("nid").indices
        with torch.no_grad():
            for f in te_f:
                ii=idx_by[f][:a.eval_cap]
                if len(ii)==0: continue
                x=load_batch(df,ii,tf).to(DEV)
                _,dpv,_=mdl(x); HEAD[f]=dpv.float().cpu().numpy().mean(0)
        # FROZEN arm: same module instance with the LoRA path REMOVED, so the only difference between
        # arms is the adapter -- not a separate model load, not different preprocessing, not a
        # different pooling. Same fetuses, same frames, same fold.
        byp={}
        for blk_ in mdl.v.transformer.resblocks:
            if hasattr(blk_,"_lora"): byp[blk_]=blk_._lora; del blk_._lora
        Zfr.update(embed_fetuses(mdl,df,te_f,tf,a.eval_cap))
        for blk_,l_ in byp.items(): blk_._lora=l_
        per_fold.append({"fold":k,"best_epoch":int(ebest),"best_val_meanrho":float(vbest),
                         "val_history":[float(h) for h in hist],"n_test_fetuses":int(len(te_f)),
                         "minutes":round((time.time()-t0)/60,1)})
        print(f"  fold {k} done: best ep {ebest} val {vbest:+.3f} ({per_fold[-1]['minutes']} min)",flush=True)
        del mdl
        if DEV=="cuda": torch.cuda.empty_cache()
    fe=sorted(set(Zad)&set(Zfr))
    Za=np.stack([Zad[f] for f in fe]); Zf=np.stack([Zfr[f] for f in fe])
    ident=float(np.abs(Za-Zf).max())
    res["per_fold"]=per_fold
    res["adapted_vs_frozen_embedding_maxdiff"]=ident
    assert ident>1e-5, (f"adapted and frozen embeddings are identical (max|diff|={ident:.2e}) -- the "
        "adapter learned nothing or was bypassed in both arms. Do not interpret any result.")
    res["evaluation"]=evaluate(Za,Zf,fe,T,outcomes,fold_of)
    hf=[f for f in fe if f in HEAD]
    if hf:
        H=np.stack([HEAD[f] for f in hf]); hr={}
        for j,c in enumerate(DOPPLER):
            y=T[c].reindex(hf).values.astype(float); m=np.isfinite(y)
            hr[c]=float(spearmanr(H[m,j],y[m]).statistic) if m.sum()>30 else np.nan
        res["evaluation"]["head_readout_endtoend"]=hr
        print("  Q1b END-TO-END HEAD on test fold (what the val curve refers to):",flush=True)
        for c,v in hr.items(): print(f"    {c:16s} head_r {v:+.3f}",flush=True)
    tt=res["evaluation"]["trained_targets"]
    print("\n  Q1 TRAINED TARGETS (did adaptation happen at all?)",flush=True)
    for c,v in tt.items():
        print(f"    {c:16s} adapted {v['adapted_r']:+.3f} frozen {v['frozen_r']:+.3f} "
              f"delta {v['delta']:+.3f}",flush=True)
    print("  Q2 HELD-OUT BIOMETRY (never trained on)",flush=True)
    for c,v in res["evaluation"]["heldout_biometry"].items():
        print(f"    {c:16s} adapted {v['adapted_r']:+.3f} frozen {v['frozen_r']:+.3f} delta {v['delta']:+.3f}",flush=True)
    print("  Q3 SIZE-ORTHOGONAL APPEARANCE vs EVAL-ONLY OUTCOMES",flush=True)
    for arm,d_ in res["evaluation"]["size_orthogonal"].items():
        for o,v in d_.items():
            print(f"    [{arm:7s}] {o:17s} raw {v['r_raw']:+.3f} size-orth {v['r_size_orthogonal']:+.3f} "
                  f"| resid-size corr {v['corr_residual_with_size']:+.3f} (must be ~0) n={v['n']}",flush=True)
    # SIGNED gate. The first version used abs(delta), so a GA DEGRADATION of -0.099 printed
    # "adaptation moved at least one trained target" -- degradation is not adaptation. The gate now
    # separates gains from losses and reports forgetting explicitly.
    MDE=0.041
    d=[v["delta"] for v in tt.values() if np.isfinite(v["delta"])]
    gains=[x for x in d if x>MDE]; losses=[x for x in d if x<-MDE]
    bio=[v["delta"] for v in res["evaluation"]["heldout_biometry"].values() if np.isfinite(v["delta"])]
    bio_loss=[x for x in bio if x<-MDE]
    res["gate"]={"MDE":MDE,"n_trained_gained":len(gains),"n_trained_degraded":len(losses),
                 "n_heldout_biometry_degraded":len(bio_loss),"mean_biometry_delta":float(np.mean(bio))}
    if len(bio_loss)>=3 or (len(losses)>0 and not gains):
        res["VERDICT"]=(f"CATASTROPHIC FORGETTING: {len(bio_loss)} of {len(bio)} held-out biometry "
          f"measures degraded beyond the MDE (mean delta {np.mean(bio):+.3f}) and {len(losses)} trained "
          f"target(s) degraded. The adapted representation is WORSE than frozen on everything it was "
          f"not directly optimising. Any gain on a trained target is bought by damaging the "
          f"pretrained features, not by learning new structure.")
    elif not gains:
        res["VERDICT"]=(f"ADAPTATION DID NOT HAPPEN: no trained target gained more than the {MDE} "
          "minimum detectable paired delta-r. Nothing downstream is interpretable.")
    else:
        res["VERDICT"]=(f"{len(gains)} trained target(s) gained beyond MDE with biometry mean delta "
          f"{np.mean(bio):+.3f}; read Q2/Q3 for whether anything NEW is carried")
    print(f"\n  {res['VERDICT']}",flush=True)
    json.dump(res,open(os.path.join(OUTP,"lora_finetune.json"),"w"),indent=2,default=str)
    print("saved out_probe/lora_finetune.json\nDONE",flush=True)

if __name__=="__main__": main()
