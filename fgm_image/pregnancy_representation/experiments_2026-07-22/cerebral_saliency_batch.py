#!/usr/bin/env python3
"""Per-frame GA-saliency overlays for all IMPACT cerebral frames.
For each frame: USF-MAE patch tokens (196) -> project GA-clock patch-weights -> 14x14
GA-contribution map -> upsample, mask to tissue (cone mask) -> overlay on scan -> save PNG.
Organized in GA-week folders. Resumable (skips existing PNGs).

Inputs: handoff/cerebral_saliency_index.csv (path, cone, ga), handoff/ga_clock_weights.npz
Run in env fgrgeom.
"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from functools import partial
from PIL import Image
import torchvision.transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

OUT="handoff/cerebral_saliency"; os.makedirs(OUT,exist_ok=True)
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class MAEEncoder(nn.Module):
    def __init__(s,img=224,patch=16,ed=768,depth=12,heads=12):
        super().__init__()
        s.patch_embed=timm.models.vision_transformer.PatchEmbed(img,patch,3,ed)
        n=s.patch_embed.num_patches
        s.cls_token=nn.Parameter(torch.zeros(1,1,ed)); s.pos_embed=nn.Parameter(torch.zeros(1,n+1,ed))
        s.blocks=nn.ModuleList([timm.models.vision_transformer.Block(ed,heads,4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(depth)])
        s.norm=nn.LayerNorm(ed,eps=1e-6)
    def forward(s,x):
        x=s.patch_embed(x)+s.pos_embed[:,1:,:]
        cls=s.cls_token+s.pos_embed[:,:1,:]; x=torch.cat([cls.expand(x.shape[0],-1,-1),x],1)
        for b in s.blocks: x=b(x)
        return s.norm(x)

def load_encoder():
    ck=torch.load("/Users/tiago/Downloads/USF-MAE_full_pretrain_43dataset_100epochs.pt",map_location="cpu",weights_only=False)
    sd=ck.get("model",ck) if isinstance(ck,dict) else ck
    enc=MAEEncoder()
    enc.load_state_dict({k:v for k,v in sd.items() if not k.startswith(("decoder","mask_token"))},strict=False)
    enc.eval(); return enc

def main():
    idx=pd.read_csv("handoff/cerebral_saliency_index.csv")
    w=np.load("handoff/ga_clock_weights.npz")
    coef_p=w["coef"][768:]; mean_p=w["mean"][768:]; scale_p=w["scale"][768:]  # patch-half of clock
    vmax=0.5  # fixed color scale across frames for comparability
    enc=load_encoder()
    print(f"frames {len(idx)}",flush=True); t0=time.time(); done=0
    for i,r in idx.iterrows():
        wk=int(round(r.ga_weeks_recovered)); d=f"{OUT}/GA{wk:02d}"; os.makedirs(d,exist_ok=True)
        base=os.path.basename(str(r.path)).replace(".png","")
        outp=f"{d}/{base}.png"
        if os.path.exists(outp): done+=1; continue
        try:
            im=Image.open(r.path).convert("RGB")
            with torch.no_grad(): tok=enc(tf(im).unsqueeze(0))[0,1:].numpy()  # (196,768)
            contrib=(((tok-mean_p)/scale_p)*coef_p).sum(-1)/196               # (196,) per-patch GA contribution
            heat=zoom(contrib.reshape(14,14),224/14,order=1)
            # tissue mask from cone
            if isinstance(r.cone,str) and os.path.exists(r.cone):
                cone=np.asarray(Image.open(r.cone).convert("L").resize((224,224),Image.NEAREST))>127
                heat=np.where(cone,heat,np.nan)
            img=np.array(im.convert("L").resize((224,224)))
            fig,ax=plt.subplots(figsize=(4,4))
            ax.imshow(img,cmap="gray"); ax.imshow(heat,cmap="RdBu_r",alpha=0.5,vmin=-vmax,vmax=vmax)
            ax.set_title(f"{base[:24]}  GA{r.ga_weeks_recovered:.1f}w",fontsize=7); ax.axis("off")
            fig.savefig(outp,dpi=90,bbox_inches="tight"); plt.close(fig)
            done+=1
        except Exception as e:
            print(f"  ERR {base}: {str(e)[:60]}",flush=True)
        if (i+1)%200==0: print(f"  {i+1}/{len(idx)} done={done} {(i+1)/(time.time()-t0):.1f}/s",flush=True)
    print(f"DONE {done}/{len(idx)} in {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
