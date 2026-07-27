#!/usr/bin/env python3
"""FULL-TOKEN extraction (CLS + all patch tokens, NO mean pooling) at a few KEY LAYERS,
fp16, for four encoders — so per-patch spatial work (saliency, attention-pool, patch
clocks) is possible later without the ~2 TB all-layer/all-token blowup.

KEY_LAYERS = 4 evenly-spaced blocks including the final one, per encoder:
  USF-MAE/USFM (12 blk) -> [3,6,9,12]   FetalCLIP (24) -> [6,12,18,24]   DINOv2 (40) -> [10,20,30,40]
Per-encoder size (fp16, 20,413 frames): USF-MAE/USFM ~25GB, FetalCLIP ~43GB, DINOv2 ~65GB.

Writes SHARDED to out_usfmae/fulltok_<enc>/shard_XXXX.npz (resumable). Each shard:
  tokens (n, n_key_layers, n_tokens, dim) fp16, key_layers (int list), names/ga/nid/plane.

Reuses the encoder builders from hpc_extract_4encoders.py but with a full-token forward.
USAGE:  python hpc_extract_4enc_fulltokens.py --check
        python hpc_extract_4enc_fulltokens.py --enc FetalCLIP
        python hpc_extract_4enc_fulltokens.py            # all four
"""
import os, sys, time, glob, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from functools import partial
from PIL import Image
import torchvision.transforms as T

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,"..","..",".."))
IMG_DIR="/mnt/beegfs/groups/collage/data/IMPACT_FULL/processed/IMPACT_FULL/preprocessed"
INDEX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
OUT=os.environ.get("GA_OUT_DIR", os.path.join(HERE,"out_usfmae"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
BATCH=32; SHARD=512
IMAGENET=((0.485,0.456,0.406),(0.229,0.224,0.225))
CLIP_MEAN=((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
WEIGHTS={"USF-MAE":os.path.join(ROOT,"USF-MAE_full_pretrain_43dataset_100epochs.pt"),
         "USFM":os.path.join(ROOT,"USFM_latest.pth"),
         "FetalCLIP":os.path.join(ROOT,"FetalCLIP_weights.pt"),
         "DINOv2":os.path.join(ROOT,"dinov2_vitg14_reg4_pretrain.pth")}
def tfm(ms,res=224): return T.Compose([T.Resize((res,res)),T.ToTensor(),T.Normalize(*ms)])
# GA_ALL_LAYERS=1 (default) -> keep every block; =0 -> 4 evenly-spaced key layers (incl last)
ALL_LAYERS = os.environ.get("GA_ALL_LAYERS","1")=="1"
def key_layers(nblk):
    if ALL_LAYERS: return list(range(1,nblk+1))          # every block
    step=nblk//4; return [step,2*step,3*step,nblk]        # 4 key layers

# ---- builders return (model, transform, fulltok_fn) where fulltok_fn(m,x,keyset)->(B,len(key),Ntok,dim)
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
        def ft(s,x,keyset):
            h=s.patch_embed(x)+s.pos_embed[:,1:,:]
            cls=s.cls_token+s.pos_embed[:,:1,:]; h=torch.cat([cls.expand(x.shape[0],-1,-1),h],1)
            out=[]
            for i,b in enumerate(s.blocks,1):
                h=b(h)
                if i in keyset: out.append(h)
            return torch.stack(out,1)     # (B,len(key),Ntok,dim)
    m=Enc().to(DEV).eval()
    sd=torch.load(WEIGHTS["USF-MAE"],map_location="cpu",weights_only=False); sd=sd.get("model",sd)
    m.load_state_dict({k:v for k,v in sd.items() if not k.startswith(("decoder","mask_token"))},strict=False)
    return m,tfm(IMAGENET),(lambda mdl,x,ks: mdl.ft(x,ks)),12

def build_usfm():
    for p in [os.path.join(ROOT,"USFM-master"),os.path.expanduser("~/USFM-master"),os.path.join(ROOT,"USFM"),os.environ.get("USFM_SRC","")]:
        if p and os.path.isdir(p): sys.path.insert(0,p)
    from usdsgen.modules.backbone.vision_transformer import VisionTransformer
    m=VisionTransformer(img_size=224,patch_size=16,in_chans=3,num_classes=0,embed_dim=768,depth=12,num_heads=12,
        mlp_ratio=4,qkv_bias=True,drop_path_rate=0.0,init_values=0.1,use_abs_pos_emb=False,use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,use_mean_pooling=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)).to(DEV).eval()
    ck=torch.load(WEIGHTS["USFM"],map_location="cpu",weights_only=False); m.load_state_dict(ck.get("model",ck.get("state_dict",ck)),strict=False)
    @torch.no_grad()
    def ft(mdl,x,keyset):
        h=mdl.patch_embed(x); cls=mdl.cls_token.expand(x.shape[0],-1,-1); h=torch.cat([cls,h],1)
        rpb=mdl.rel_pos_bias() if getattr(mdl,"rel_pos_bias",None) is not None else None
        out=[]
        for i,b in enumerate(mdl.blocks,1):
            h=b(h,rel_pos_bias=rpb) if rpb is not None else b(h)
            if i in keyset: out.append(h)
        return torch.stack(out,1)
    return m,tfm(IMAGENET),ft,12

def build_fetalclip():
    import open_clip
    model=open_clip.create_model("ViT-L-14",pretrained=None)
    ck=torch.load(WEIGHTS["FetalCLIP"],map_location="cpu",weights_only=False)
    sd=ck.get("state_dict",ck.get("model",ck)); sd={k.replace("module.",""):v for k,v in sd.items()}
    vis={k[len("visual."):]:v for k,v in sd.items() if k.startswith("visual.")}
    model.visual.load_state_dict(vis,strict=False); vt=model.visual.to(DEV).eval()
    @torch.no_grad()
    def ft(mdl,x,keyset):
        v=mdl; h=v.conv1(x); h=h.reshape(h.shape[0],h.shape[1],-1).permute(0,2,1)
        cls=v.class_embedding.to(h.dtype)+torch.zeros(h.shape[0],1,h.shape[-1],device=h.device,dtype=h.dtype)
        h=torch.cat([cls,h],1)+v.positional_embedding.to(h.dtype); h=v.ln_pre(h); h=h.permute(1,0,2)
        out=[]
        for i,blk in enumerate(v.transformer.resblocks,1):
            h=blk(h)
            if i in keyset: out.append(h.permute(1,0,2))
        return torch.stack(out,1)
    return vt,tfm(CLIP_MEAN),ft,24

def build_dinov2():
    m=torch.hub.load("facebookresearch/dinov2","dinov2_vitg14_reg",pretrained=False).to(DEV).eval()
    sd=torch.load(WEIGHTS["DINOv2"],map_location="cpu",weights_only=False); m.load_state_dict(sd,strict=False)
    @torch.no_grad()
    def ft(mdl,x,keyset):
        # n=all blocks, keep class token; select key layers; concat cls+patches per layer
        feats=mdl.get_intermediate_layers(x,n=mdl.n_blocks,return_class_token=True)
        out=[]
        for i,(patch_tok,cls_tok) in enumerate(feats,1):
            if i in keyset: out.append(torch.cat([cls_tok[:,None,:],patch_tok],1))
        return torch.stack(out,1)
    return m,tfm(IMAGENET),ft,40

BUILDERS={"USF-MAE":build_usfmae,"USFM":build_usfm,"FetalCLIP":build_fetalclip,"DINOv2":build_dinov2}

def frame_table():
    df=pd.read_csv(INDEX).copy()
    df["img"]=df["new_filename"].astype(str).apply(lambda n: os.path.join(IMG_DIR,n if n.endswith(".png") else n+".png"))
    return df[df["img"].apply(os.path.exists)].reset_index(drop=True)

def check():
    df=frame_table().head(2)
    for name,build in BUILDERS.items():
        try:
            m,tf,ft,nblk=build(); ks=set(key_layers(nblk))
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in df["img"]]).to(DEV)
            y=ft(m,x,ks)
            mb=y.element_size()*y[0].numel()/1e6
            print(f"OK  {name}: {tuple(y.shape)} keylayers={sorted(ks)} {mb*0.5:.1f}MB/frame(fp16) finite={torch.isfinite(y).all().item()}",flush=True)
            del m,x,y; torch.cuda.empty_cache() if DEV=="cuda" else None
        except Exception as e:
            import traceback; print(f"FAIL {name}: {type(e).__name__}: {e}",flush=True); traceback.print_exc()

def extract_one(name):
    d=os.path.join(OUT,f"fulltok_{name}"); os.makedirs(d,exist_ok=True)
    df=frame_table(); m,tf,ft,nblk=BUILDERS[name](); ks=set(key_layers(nblk)); ksl=sorted(ks)
    # probe 1 frame to size a shard to ~4GB (all-layer DINOv2 is 64MB/frame -> ~64 frames/shard)
    x0=torch.stack([tf(Image.open(df["img"].iloc[0]).convert("RGB"))]).to(DEV)
    mb=ft(m,x0,ks).half().cpu().numpy().nbytes/1e6; del x0
    shard=max(BATCH, min(SHARD, int(4000/max(mb,1e-3))//BATCH*BATCH or BATCH))
    nsh=(len(df)+shard-1)//shard; t0=time.time()
    print(f"{name}: {len(df)} frames, {len(ksl)} layers, {mb:.1f}MB/frame -> {nsh} shards x {shard}",flush=True)
    for si in range(nsh):
        outp=os.path.join(d,f"shard_{si:04d}.npz")
        if os.path.exists(outp): continue
        sl=df.iloc[si*shard:(si+1)*shard]; buf=[]
        for b0 in range(0,len(sl),BATCH):
            bs=sl.iloc[b0:b0+BATCH]
            x=torch.stack([tf(Image.open(p).convert("RGB")) for p in bs["img"]]).to(DEV)
            buf.append(ft(m,x,ks).half().cpu().numpy())
        tok=np.concatenate(buf)
        np.savez(outp, tokens=tok, key_layers=np.array(ksl), names=sl["new_filename"].values,
                 ga=sl["ga_weeks_recovered"].values, nid=sl["nid"].astype(str).values, plane=sl["plane_prop"].values)
        print(f"  {name} shard {si+1}/{nsh} {time.time()-t0:.0f}s {(si+1)*shard/max(time.time()-t0,1):.0f} fr/s {tok.nbytes/1e9:.2f}GB",flush=True)
    print(f"{name} DONE {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--check",action="store_true"); ap.add_argument("--enc",default=None)
    a=ap.parse_args()
    if a.check: check()
    elif a.enc: extract_one(a.enc)
    else:
        for name in BUILDERS: extract_one(name)
