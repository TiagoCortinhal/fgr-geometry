#!/usr/bin/env python3
"""Extract FULL USF-MAE token stack for every IMPACT frame: all 12 transformer blocks,
all 197 tokens (CLS + 196 patches), 768-d, float32 — NO mean pooling, NO layer selection.

Per frame: (12, 197, 768) ~= 7.26 MB. All 20,413 IMPACT frames ~= 148 GB.
Written SHARDED to the USB (one .npz per shard) so it's resumable and never one giant file.

Index: handoff/ga_cnn_index.csv (all-plane IMPACT, path/ga/plane/split).
Output dir: /Users/tiago/usb/usfmae_all_layers/  (shard_XXXX.npz + names_XXXX.npy)
Run: /Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python extract_all_layers.py
"""
import os, sys, time, glob, numpy as np, pandas as pd, torch, torch.nn as nn, timm
from functools import partial
from PIL import Image
import torchvision.transforms as T

HERE=os.path.dirname(os.path.abspath(__file__))
IDX=os.path.join(HERE,"ga_cnn","ga_cnn_index.csv")
OUT="/Users/tiago/usb/usfmae_all_layers"; os.makedirs(OUT,exist_ok=True)
CKPT="/Users/tiago/Downloads/USF-MAE_full_pretrain_43dataset_100epochs.pt"
SHARD=512                      # frames per shard (~3.7 GB/shard)
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

class MAEEncoder(nn.Module):
    def __init__(s,img=224,patch=16,ed=768,depth=12,heads=12):
        super().__init__()
        s.patch_embed=timm.models.vision_transformer.PatchEmbed(img,patch,3,ed)
        n=s.patch_embed.num_patches
        s.cls_token=nn.Parameter(torch.zeros(1,1,ed)); s.pos_embed=nn.Parameter(torch.zeros(1,n+1,ed))
        s.blocks=nn.ModuleList([timm.models.vision_transformer.Block(ed,heads,4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(depth)])
        s.norm=nn.LayerNorm(ed,eps=1e-6)
    @torch.no_grad()
    def all_layers(s,x):
        h=s.patch_embed(x)+s.pos_embed[:,1:,:]
        cls=s.cls_token+s.pos_embed[:,:1,:]; h=torch.cat([cls.expand(x.shape[0],-1,-1),h],1)
        outs=[]
        for b in s.blocks: h=b(h); outs.append(h)         # each (B,197,768), CLS included
        return torch.stack(outs,1)                        # (B,12,197,768)

def load_encoder():
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    sd=ck.get("model",ck) if isinstance(ck,dict) else ck
    enc=MAEEncoder(); enc.load_state_dict({k:v for k,v in sd.items() if not k.startswith(("decoder","mask_token"))},strict=False)
    enc.eval(); return enc

def main():
    df=pd.read_csv(IDX).reset_index(drop=True)
    enc=load_encoder(); torch.set_num_threads(8)
    n=len(df); nsh=(n+SHARD-1)//SHARD
    print(f"frames {n} | {nsh} shards x {SHARD} | out {OUT}",flush=True); t0=time.time()
    for si in range(nsh):
        outp=os.path.join(OUT,f"shard_{si:04d}.npz")
        if os.path.exists(outp): print(f"  shard {si} exists, skip",flush=True); continue
        sl=df.iloc[si*SHARD:(si+1)*SHARD]
        arr=np.zeros((len(sl),12,197,768),np.float32); names=[]
        for j,(_,r) in enumerate(sl.iterrows()):
            x=tf(Image.open(r.path).convert("RGB")).unsqueeze(0)
            arr[j]=enc.all_layers(x)[0].numpy()
            names.append(os.path.basename(str(r.path)).replace(".png",""))
        np.savez(outp, tokens=arr,
                 names=np.array(names), ga=sl.ga_weeks_recovered.values,
                 plane=sl.plane_prop.values, nid=sl.nid.astype(str).values, split=sl.split.values)
        el=time.time()-t0
        print(f"  shard {si+1}/{nsh} done ({len(sl)} frames) | {el:.0f}s | {(si+1)*SHARD/el:.1f} fr/s | {os.path.getsize(outp)/1e9:.2f}GB",flush=True)
    print(f"DONE {nsh} shards in {(time.time()-t0)/60:.0f}min",flush=True)

if __name__=="__main__": main()
