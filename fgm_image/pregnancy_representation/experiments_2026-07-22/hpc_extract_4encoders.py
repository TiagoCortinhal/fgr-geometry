#!/usr/bin/env python3
"""Extract per-layer summaries (CLS + mean-patch, every transformer block) for FOUR encoders
on the IMPACT+clinical frames, so the GA/lag/placental probes can compare them.

Encoders (weights in repo root):
  USF-MAE  : USF-MAE_full_pretrain_43dataset_100epochs.pt   (MAE ViT-B/16, 768-d, 12 blk)
  USFM     : USFM_latest.pth                                 (BEiT-style ViT-B/16, 768-d, needs usdsgen src)
  FetalCLIP: FetalCLIP_weights.pt                            (open_clip ViT-L/14, 1024-d, ~24 blk)
  DINOv2   : dinov2_vitg14_reg4_pretrain.pth                 (ViT-g/14 +4 reg, 1536-d, 40 blk)

Per encoder writes out_usfmae/summaries_<enc>.npz  (LS (N,Lblk,2*dim), ga, nid, plane, names).
Different encoders have different dim / #layers -> stored separately, one file each.

USAGE:
  python hpc_extract_4encoders.py --check         # load all 4 on 2 images, report, exit
  python hpc_extract_4encoders.py --enc USF-MAE   # extract one encoder
  python hpc_extract_4encoders.py                 # extract all four (skips ones already done)
"""
import os, sys, time, glob, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from functools import partial
from PIL import Image
import torchvision.transforms as T

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))   # repo root (weights live here)
IMG_DIR="/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed"
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae")); os.makedirs(OUT,exist_ok=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
BATCH=48
IMAGENET=( (0.485,0.456,0.406),(0.229,0.224,0.225) )
CLIP_MEAN=((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))

WEIGHTS={
 "USF-MAE":  os.path.join(ROOT,"USF-MAE_full_pretrain_43dataset_100epochs.pt"),
 "USFM":     os.path.join(ROOT,"USFM_latest.pth"),
 "FetalCLIP":os.path.join(ROOT,"FetalCLIP_weights.pt"),
 "DINOv2":   os.path.join(ROOT,"dinov2_vitg14_reg4_pretrain.pth"),
}

def tfm(mean_std,res=224):
    return T.Compose([T.Resize((res,res)),T.ToTensor(),T.Normalize(*mean_std)])

# ---------------- per-encoder builders: each returns (model, transform, layer_forward) ----------------
# layer_forward(model, x_batch) -> tensor (B, Lblk, 2*dim) = per-block concat[CLS, mean(patch tokens)]

def build_usfmae():
    import timm
    class Enc(nn.Module):
        def __init__(s,ed=768,depth=12,heads=12):
            super().__init__()
            s.patch_embed=timm.models.vision_transformer.PatchEmbed(224,16,3,ed)
            n=s.patch_embed.num_patches
            s.cls_token=nn.Parameter(torch.zeros(1,1,ed)); s.pos_embed=nn.Parameter(torch.zeros(1,n+1,ed))
            s.blocks=nn.ModuleList([timm.models.vision_transformer.Block(ed,heads,4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(depth)])
            s.norm=nn.LayerNorm(ed,eps=1e-6)
        @torch.no_grad()
        def layers(s,x):
            h=s.patch_embed(x)+s.pos_embed[:,1:,:]
            cls=s.cls_token+s.pos_embed[:,:1,:]; h=torch.cat([cls.expand(x.shape[0],-1,-1),h],1)
            out=[]
            for b in s.blocks: h=b(h); out.append(torch.cat([h[:,0],h[:,1:].mean(1)],-1))
            return torch.stack(out,1)
    m=Enc().to(DEV).eval()
    sd=torch.load(WEIGHTS["USF-MAE"],map_location="cpu",weights_only=False)
    sd=sd.get("model",sd)
    m.load_state_dict({k:v for k,v in sd.items() if not k.startswith(("decoder","mask_token"))},strict=False)
    return m, tfm(IMAGENET), (lambda mdl,x: mdl.layers(x))

def build_usfm():
    # needs usdsgen source (BEiT-style ViT with rel_pos_bias). Try common locations.
    for p in [os.path.join(ROOT,"USFM-master"), os.path.expanduser("~/USFM-master"),
              os.path.join(ROOT,"USFM"), os.environ.get("USFM_SRC","")]:
        if p and os.path.isdir(p): sys.path.insert(0,p)
    from usdsgen.modules.backbone.vision_transformer import VisionTransformer
    m=VisionTransformer(img_size=224,patch_size=16,in_chans=3,num_classes=0,embed_dim=768,depth=12,num_heads=12,
        mlp_ratio=4,qkv_bias=True,drop_path_rate=0.0,init_values=0.1,use_abs_pos_emb=False,use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,use_mean_pooling=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)).to(DEV).eval()
    ck=torch.load(WEIGHTS["USFM"],map_location="cpu",weights_only=False)
    m.load_state_dict(ck.get("model",ck.get("state_dict",ck)),strict=False)
    @torch.no_grad()
    def layers(mdl,x):
        # hook each block; USFM blocks live in mdl.blocks
        outs=[]
        h=mdl.patch_embed(x)
        cls=mdl.cls_token.expand(x.shape[0],-1,-1); h=torch.cat([cls,h],1)
        rpb=mdl.rel_pos_bias() if getattr(mdl,"rel_pos_bias",None) is not None else None
        for b in mdl.blocks:
            h=b(h,rel_pos_bias=rpb) if rpb is not None else b(h)
            outs.append(torch.cat([h[:,0],h[:,1:].mean(1)],-1))
        return torch.stack(outs,1)
    return m, tfm(IMAGENET), layers

def build_fetalclip():
    import open_clip
    model=open_clip.create_model("ViT-L-14",pretrained=None)
    ck=torch.load(WEIGHTS["FetalCLIP"],map_location="cpu",weights_only=False)
    sd=ck.get("state_dict",ck.get("model",ck))
    sd={k.replace("module.",""):v for k,v in sd.items()}
    vis={k[len("visual."):]:v for k,v in sd.items() if k.startswith("visual.")}
    model.visual.load_state_dict(vis,strict=False)
    vt=model.visual.to(DEV).eval()
    @torch.no_grad()
    def layers(mdl,x):
        v=mdl
        h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)  # B,grid,width
        cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
        h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype)
        h=v.ln_pre(h); h=h.permute(1,0,2)   # LND for open_clip transformer
        outs=[]
        for blk in v.transformer.resblocks:
            h=blk(h); t=h.permute(1,0,2); outs.append(torch.cat([t[:,0],t[:,1:].mean(1)],-1))
        return torch.stack(outs,1)
    return vt, tfm(CLIP_MEAN), layers

def build_dinov2():
    # ViT-g/14 with 4 registers via torch.hub def (weights local); needs internet for the CODE unless cached
    m=torch.hub.load("facebookresearch/dinov2","dinov2_vitg14_reg",pretrained=False).to(DEV).eval()
    sd=torch.load(WEIGHTS["DINOv2"],map_location="cpu",weights_only=False)
    m.load_state_dict(sd,strict=False)
    @torch.no_grad()
    def layers(mdl,x):
        # get_intermediate_layers returns per-block patch tokens; also grab CLS via norm
        feats=mdl.get_intermediate_layers(x,n=mdl.n_blocks,return_class_token=True)
        outs=[]
        for patch_tok,cls_tok in feats:
            outs.append(torch.cat([cls_tok, patch_tok.mean(1)],-1))
        return torch.stack(outs,1)
    return m, tfm(IMAGENET), layers

BUILDERS={"USF-MAE":build_usfmae,"USFM":build_usfm,"FetalCLIP":build_fetalclip,"DINOv2":build_dinov2}

def frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(lambda n: os.path.join(IMG_DIR, n if n.endswith(".png") else n+".png"))
    return df[df["img"].apply(os.path.exists)].reset_index(drop=True)

def check():
    df=frame_table().head(2)
    for name,build in BUILDERS.items():
        try:
            m,tf,lf=build()
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in df["img"]]).to(DEV)
            y=lf(m,x)
            print(f"OK  {name}: out {tuple(y.shape)} (B, n_layers, 2*dim) finite={torch.isfinite(y).all().item()}",flush=True)
            del m,x,y; torch.cuda.empty_cache() if DEV=="cuda" else None
        except Exception as e:
            import traceback; print(f"FAIL {name}: {type(e).__name__}: {e}",flush=True)
            traceback.print_exc(); print("",flush=True)

def extract_one(name):
    out=os.path.join(OUT,f"summaries_{name}.npz")
    if os.path.exists(out): print(f"  {name}: exists, skip",flush=True); return
    df=frame_table(); m,tf,lf=BUILDERS[name]()
    LS=[]; t0=time.time()
    for b0 in range(0,len(df),BATCH):
        bs=df.iloc[b0:b0+BATCH]
        x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
        LS.append(lf(m,x).float().cpu().numpy())
        if (b0//BATCH)%20==0:
            el=time.time()-t0; print(f"  {name} {b0}/{len(df)} {el:.0f}s {(b0+len(bs))/max(el,1e-6):.0f} fr/s",flush=True)
    LS=np.concatenate(LS)
    np.savez(out, LS=LS, ga=df["ga_weeks_recovered"].values, nid=df["nid"].astype(str).values,
             plane=df["plane_prop"].values, names=df["new_filename"].values)
    print(f"  {name} DONE {LS.shape} -> {out} ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--check",action="store_true"); ap.add_argument("--enc",default=None)
    a=ap.parse_args()
    if a.check: check()
    elif a.enc: extract_one(a.enc)
    else:
        for name in BUILDERS: extract_one(name)
