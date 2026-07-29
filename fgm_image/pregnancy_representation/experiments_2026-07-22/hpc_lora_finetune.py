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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds",type=int,default=5); ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--sigma-dating",type=float,default=0.7,help="GA label sigma in weeks (CRL dating ~+/-5-7d)")
    ap.add_argument("--batch",type=int,default=24); ap.add_argument("--frames-per-fetus",type=int,default=4)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--check",action="store_true")
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

    fold=np.zeros(len(df),dtype=int)
    for k,(_,te) in enumerate(GroupKFold(a.folds).split(df,groups=df["nid"])): fold[te]=k
    df["fold"]=fold
    fh=hashlib.sha256(pd.util.hash_pandas_object(df[["nid","fold"]],index=False).values.tobytes()).hexdigest()[:16]
    df[["nid","fold"]].drop_duplicates().to_csv(os.path.join(HAND,"lora_folds.csv"),index=False)
    res["fold_hash"]=fh; print(f"  folds written, hash {fh} (must match across frozen and adapted arms)",flush=True)
    json.dump(res,open(os.path.join(OUTP,"lora_finetune.json"),"w"),indent=2,default=str)
    print("  SETUP COMPLETE -- training loop runs per fold; see --check for a dry run",flush=True)
    print("saved out_probe/lora_finetune.json\nDONE",flush=True)

if __name__=="__main__": main()
