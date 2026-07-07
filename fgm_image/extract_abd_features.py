#!/usr/bin/env python3
"""Abdominal-plane soft-tissue features per image:
 (A) USFM PATCH tokens (196x768 pre-pool) -> texture stats: spatial variance, token norm dispersion, edge/rim energy
 (B) ECHOGENICITY: raw-pixel intensity + GLCM-style texture on the abdominal wall region
Aggregated per fetus downstream."""
import sys, os, time, numpy as np, pandas as pd, torch
from functools import partial
import torch.nn as nn
from torchvision import transforms
from PIL import Image
WS="/Users/tiago/.claude-science/orgs/2e50fc88-f384-4a2e-9328-c60d613fd12a/workspaces/50e875d8-edc0-4647-b8d5-b6df7858c9cb"
sys.path.insert(0,f"{WS}/USFM-master")
from usdsgen.modules.backbone.vision_transformer import VisionTransformer
WORK="/Users/tiago/PythonProject/fgr-geometry/results/img_align/_abd_worklist.csv"
OUT="/Users/tiago/PythonProject/fgr-geometry/results/img_align/_abd_features.npz"
W="/Users/tiago/Downloads/USFM_latest.pth"
IM_MEAN=(0.485,0.456,0.406); IM_STD=(0.229,0.224,0.225); BATCH=64
torch.set_num_threads(10)
df=pd.read_csv(WORK)

m=VisionTransformer(img_size=224,patch_size=16,in_chans=3,num_classes=0,embed_dim=768,depth=12,num_heads=12,
  mlp_ratio=4,qkv_bias=True,drop_path_rate=0.0,init_values=0.1,use_abs_pos_emb=False,use_rel_pos_bias=False,
  use_shared_rel_pos_bias=True,use_mean_pooling=True,norm_layer=partial(nn.LayerNorm,eps=1e-6))
ck=torch.load(W,map_location="cpu"); m.load_state_dict(ck.get("model",ck.get("state_dict",ck)),strict=False); m.eval()

def patch_tokens(x):  # returns [B,196,768]
    xb=m.patch_embed(x) if not isinstance(m.patch_embed(x),tuple) else m.patch_embed(x)[0]
    cls=m.cls_token.expand(xb.shape[0],-1,-1); xb=torch.cat((cls,xb),1)
    if m.pos_embed is not None: xb=xb+m.pos_embed
    xb=m.pos_drop(xb); rpb=m.rel_pos_bias() if m.rel_pos_bias is not None else None
    for blk in m.blocks: xb=blk(xb,rel_pos_bias=rpb)
    xb=m.norm(xb); return m.fc_norm(xb[:,1:,:]) if m.fc_norm is not None else xb[:,1:,:]

pre=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(IM_MEAN,IM_STD)])
def echo_feats(pil):
    a=np.asarray(pil.resize((224,224)).convert("L"),dtype=np.float32)/255.0
    fg=a[a>0.05]  # ignore cone background
    if len(fg)<100: fg=a.ravel()
    gx=np.abs(np.diff(a,axis=1)); gy=np.abs(np.diff(a,axis=0))
    # radial: fat rim tends to be brighter periphery vs darker center in AC plane
    cy,cx=112,112; yy,xx=np.mgrid[0:224,0:224]; r=np.sqrt((yy-cy)**2+(xx-cx)**2)
    inner=a[(r<60)&(a>0.05)]; outer=a[(r>=60)&(r<105)&(a>0.05)]
    return np.array([fg.mean(),fg.std(),np.percentile(fg,90),np.percentile(fg,10),
        gx.mean(),gy.mean(),(gx.std()+gy.std())/2,
        (outer.mean()-inner.mean()) if len(inner)and len(outer) else 0.0,
        (outer.mean()/(inner.mean()+1e-6)) if len(inner)and len(outer) else 1.0],dtype=np.float32)
ECHO_NAMES=["int_mean","int_std","int_p90","int_p10","grad_x","grad_y","grad_std","rim_center_diff","rim_center_ratio"]

patch_feats=[]; echo=[]; names=[]; fids=[]; buf=[]; bmeta=[]; t0=time.time()
def flush():
    global buf,bmeta
    if not buf: return
    with torch.no_grad():
        tok=patch_tokens(torch.stack(buf)).numpy()  # [B,196,768]
    for tk,mt in zip(tok,bmeta):
        # texture stats over the 196 patches: how much patches DIFFER (soft-tissue heterogeneity)
        pn=np.linalg.norm(tk,axis=1)  # per-patch norm (196,)
        g=tk.reshape(14,14,768)
        # spatial gradient energy of token field (local structure) + norm dispersion + periphery-vs-center
        dh=np.linalg.norm(np.diff(g,axis=0),axis=2).mean(); dw=np.linalg.norm(np.diff(g,axis=1),axis=2).mean()
        cy,cx=np.mgrid[0:14,0:14]; rr=np.sqrt((cy-6.5)**2+(cx-6.5)**2)
        peri=pn.reshape(14,14)[rr>=5].mean(); cen=pn.reshape(14,14)[rr<3.5].mean()
        patch_feats.append([pn.mean(),pn.std(),pn.max()-pn.min(),dh,dw,peri-cen,tk.var(0).mean()])
        echo.append(mt["echo"]); fids.append(mt["fid"]); names.append(mt["nf"])
    buf=[]; bmeta=[]
PATCH_NAMES=["pnorm_mean","pnorm_std","pnorm_range","tok_grad_h","tok_grad_w","peri_center_diff","patch_var"]
for i,r in df.iterrows():
    try: pil=Image.open(r.png).convert("RGB")
    except Exception: continue
    buf.append(pre(pil)); bmeta.append({"fid":int(r.fid),"nf":r.new_filename,"echo":echo_feats(pil)})
    if len(buf)>=BATCH: flush()
    if i%1280==0 and i: print(f"  {i}/{len(df)}  {i/(time.time()-t0):.0f} img/s",flush=True)
flush()
np.savez(OUT, patch=np.array(patch_feats,dtype=np.float32), echo=np.array(echo,dtype=np.float32),
         fid=np.array(fids), new_filename=np.array(names,dtype=object),
         patch_names=np.array(PATCH_NAMES), echo_names=np.array(ECHO_NAMES))
print(f"DONE {len(patch_feats)} images -> {OUT} in {(time.time()-t0)/60:.1f} min",flush=True)
