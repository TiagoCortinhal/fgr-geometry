#!/usr/bin/env python3
"""Full image-derivation extraction: DINOv2 + USF-MAE embeddings + USF-MAE recon-error ladder.
Per frame saves: dino(384), mae(1536), err_raw(scalar), err_roi(scalar), err_patch(196), tissue(196).
Sharded to .npz, resumable. Run in fgrgeom env. ~3.6h for 50k frames on CPU."""
import os,sys,time,json,numpy as np,torch,torch.nn as nn
from functools import partial
from PIL import Image
import torchvision.transforms as T
from timm.models.vision_transformer import PatchEmbed, Block

IDX="handoff/full_frame_index.csv"
OUT="handoff/imgderiv"; os.makedirs(OUT,exist_ok=True)
CKPT="/Users/tiago/Downloads/USF-MAE_full_pretrain_43dataset_100epochs.pt"
SHARD=2000; SEEDS=4
torch.set_num_threads(10)

class MAE(nn.Module):
    def __init__(self,img=224,p=16,ec=768,ed=12,eh=12,dc=512,dd=8,dh=16):
        super().__init__()
        self.patch_embed=PatchEmbed(img,p,3,ec); n=self.patch_embed.num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,ec))
        self.pos_embed=nn.Parameter(torch.zeros(1,n+1,ec),requires_grad=False)
        self.blocks=nn.ModuleList([Block(ec,eh,4,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(ed)])
        self.norm=nn.LayerNorm(ec,eps=1e-6); self.decoder_embed=nn.Linear(ec,dc,bias=True)
        self.mask_token=nn.Parameter(torch.zeros(1,1,dc))
        self.decoder_pos_embed=nn.Parameter(torch.zeros(1,n+1,dc),requires_grad=False)
        self.decoder_blocks=nn.ModuleList([Block(dc,dh,4,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-6)) for _ in range(dd)])
        self.decoder_norm=nn.LayerNorm(dc,eps=1e-6); self.decoder_pred=nn.Linear(dc,p*p*3,bias=True); self.p=p
    def patchify(self,imgs):
        p=self.p; h=w=imgs.shape[2]//p
        return imgs.reshape(imgs.shape[0],3,h,p,w,p).permute(0,2,4,3,5,1).reshape(imgs.shape[0],h*w,p*p*3)
    def forward_encoder(self,x,mr):
        x=self.patch_embed(x)+self.pos_embed[:,1:,:]; N,L,D=x.shape; lk=int(L*(1-mr))
        noise=torch.rand(N,L); ids=torch.argsort(noise,1); idr=torch.argsort(ids,1)
        xk=torch.gather(x,1,ids[:,:lk].unsqueeze(-1).repeat(1,1,D))
        mask=torch.ones(N,L); mask[:,:lk]=0; mask=torch.gather(mask,1,idr)
        cls=self.cls_token+self.pos_embed[:,:1,:]; xk=torch.cat([cls.expand(N,-1,-1),xk],1)
        for b in self.blocks: xk=b(xk)
        return self.norm(xk),mask,idr
    def forward_decoder(self,x,idr):
        x=self.decoder_embed(x); N=x.shape[0]
        mt=self.mask_token.repeat(N,idr.shape[1]+1-x.shape[1],1)
        x_=torch.cat([x[:,1:,:],mt],1); x_=torch.gather(x_,1,idr.unsqueeze(-1).repeat(1,1,x.shape[2]))
        x=torch.cat([x[:,:1,:],x_],1)+self.decoder_pos_embed
        for b in self.decoder_blocks: x=b(x)
        return self.decoder_pred(self.decoder_norm(x))[:,1:,:]
    def recon_error_map(self,imgs,mr=0.75,seeds=SEEDS):
        tgt=self.patchify(imgs); tgt=(tgt-tgt.mean(-1,keepdim=True))/(tgt.var(-1,keepdim=True)+1e-6)**.5
        num=0;den=0
        for s in range(seeds):
            torch.manual_seed(s); lat,mask,idr=self.forward_encoder(imgs,mr); pred=self.forward_decoder(lat,idr)
            pe=((pred-tgt)**2).mean(-1); num=num+pe*mask; den=den+mask
        return (num/(den+1e-9))
    def embed(self,imgs):
        x=self.patch_embed(imgs)+self.pos_embed[:,1:,:]; N=x.shape[0]
        cls=self.cls_token+self.pos_embed[:,:1,:]; x=torch.cat([cls.expand(N,-1,-1),x],1)
        for b in self.blocks: x=b(x)
        x=self.norm(x); return torch.cat([x[:,0],x[:,1:].mean(1)],-1)

tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
def patch_tissue(cone_p,cal_p):
    if not cone_p or not os.path.exists(cone_p): return np.ones(196,dtype=np.float32)
    cone=np.asarray(Image.open(cone_p).convert("L").resize((224,224),Image.NEAREST))>127
    if cal_p and os.path.exists(cal_p):
        cal=np.asarray(Image.open(cal_p).convert("L").resize((224,224),Image.NEAREST))>127
        roi=cone&~cal
    else: roi=cone
    return roi.reshape(14,16,14,16).mean((1,3)).ravel().astype(np.float32)

def main():
    import pandas as pd
    idx=pd.read_csv(IDX)
    done=set()
    for f in os.listdir(OUT):
        if f.endswith(".npz"): done|=set(np.load(os.path.join(OUT,f),allow_pickle=True)["names"])
    idx=idx[~idx.new_filename.isin(done)].reset_index(drop=True)
    print(f"todo {len(idx)}",flush=True)
    mae=MAE(); mae.load_state_dict(torch.load(CKPT,map_location="cpu"),strict=True); mae.eval()
    dino=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',verbose=False); dino.eval()
    shard=len([f for f in os.listdir(OUT) if f.endswith(".npz")])
    buf={k:[] for k in ["names","dino","mae","err_raw","err_roi","err_patch","tissue"]}
    t0=time.time(); errs=[]
    def flush(sh):
        np.savez(f"{OUT}/shard_{sh:03d}.npz",**{k:np.array(v) for k,v in buf.items()})
        for k in buf: buf[k]=[]
    with torch.no_grad():
        for i,r in enumerate(idx.itertuples(index=False)):
            try:
                img=tf(Image.open(r.png).convert("RGB")).unsqueeze(0)
                d=dino(img).cpu().numpy()[0]; m=mae.embed(img).cpu().numpy()[0]
                em=mae.recon_error_map(img).cpu().numpy()[0]  # (196,)
                pt=patch_tissue(r.cone if isinstance(r.cone,str) else "", r.calip if isinstance(r.calip,str) else "")
                tiss=pt>0.5
                er_roi=float((em*tiss).sum()/(tiss.sum()+1e-9)) if tiss.sum()>0 else float(em.mean())
                buf["names"].append(r.new_filename); buf["dino"].append(d); buf["mae"].append(m)
                buf["err_raw"].append(float(em.mean())); buf["err_roi"].append(er_roi)
                buf["err_patch"].append(em.astype(np.float32)); buf["tissue"].append(pt)
            except Exception as ex: errs.append((r.new_filename,str(ex)[:80]))
            if len(buf["names"])>=SHARD: flush(shard); shard+=1
            if (i+1)%200==0:
                el=time.time()-t0; rate=(i+1)/el
                print(f"  {i+1}/{len(idx)} {rate:.1f}/s ETA {(len(idx)-i-1)/rate/60:.0f}min errs {len(errs)}",flush=True)
    if buf["names"]: flush(shard)
    json.dump(errs,open(f"{OUT}/_errors.json","w"))
    print(f"DONE {time.time()-t0:.0f}s errs {len(errs)}",flush=True)

if __name__=="__main__": main()
