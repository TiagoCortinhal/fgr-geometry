#!/usr/bin/env python3
"""From-scratch GA clock: ResNet-18 (no pretrained weights) regressing gestational age
from IMPACT ultrasound frames (all planes). CPU. Fetus-grouped splits.

Confound control baked into training: random resized crop + flip + intensity jitter so the
net can't shortcut on absolute scale / zoom / pixel-spacing. Grayscale->3ch.

Inputs: handoff/ga_cnn_index.csv (path, ga, plane, split). Run in env fgrgeom.
Outputs: handoff/ga_cnn/{best.pt, log.json}.
"""
import os, sys, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as M

HERE=os.path.dirname(os.path.abspath(__file__))
IDX=os.path.join(HERE,"ga_cnn_index.csv")          # index sits next to this script
OUT=os.path.join(HERE,"out"); os.makedirs(OUT,exist_ok=True)
# device: prefer Apple-Silicon GPU (MPS) if reachable, else CPU
DEV = "mps" if (torch.backends.mps.is_available()) else "cpu"
if DEV=="cpu": torch.set_num_threads(8)
NWORK = int(os.environ.get("GA_CNN_WORKERS", "4"))  # set GA_CNN_WORKERS=0 if workers get killed
RES=160  # 160px (not 224) — cuts compute ~2x with minimal accuracy loss on a GA clock

class Frames(Dataset):
    def __init__(s, df, train):
        s.df=df.reset_index(drop=True)
        if train:
            s.tf=T.Compose([T.Grayscale(3),T.RandomResizedCrop(RES,scale=(0.7,1.0)),
                            T.RandomHorizontalFlip(),T.ColorJitter(0.2,0.2),
                            T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
        else:
            s.tf=T.Compose([T.Grayscale(3),T.Resize((RES,RES)),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        return s.tf(Image.open(r.path).convert("L")), np.float32(r.ga_weeks_recovered)

def make_model():
    m=M.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,1); return m

def run(epochs=30, bs=32, lr=3e-4, benchmark=False):
    df=pd.read_csv(IDX)
    print(f"device={DEV} workers={NWORK} res={RES} | train={sum(df.split=='train')} val={sum(df.split=='val')} test={sum(df.split=='test')}",flush=True)
    pw = NWORK>0
    dl_tr=DataLoader(Frames(df[df.split=="train"],True),bs,shuffle=True,num_workers=NWORK,persistent_workers=pw)
    dl_va=DataLoader(Frames(df[df.split=="val"],False),bs,num_workers=max(NWORK//2,0),persistent_workers=pw)
    m=make_model().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    lossf=nn.HuberLoss(delta=2.0)
    best=1e9; log=[]; nb=len(dl_tr)
    print(f"starting: {epochs} epochs x {nb} batches/epoch (bs={bs})",flush=True)
    for ep in range(epochs):
        m.train(); t0=time.time(); tl=0
        for bi,(x,y) in enumerate(dl_tr):
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            p=m(x).squeeze(1); l=lossf(p,y); l.backward(); opt.step(); tl+=l.item()*len(x)
            if benchmark: print(f"  [bench] 1 batch {time.time()-t0:.1f}s",flush=True); return time.time()-t0
            if bi % 20 == 0:
                el=time.time()-t0; rate=(bi+1)/max(el,1e-6)
                print(f"  ep{ep} batch {bi}/{nb}  loss={l.item():.2f}  {rate:.1f} batch/s  "
                      f"eta_epoch={(nb-bi)/rate/60:.1f}min",flush=True)
        sched.step()
        # val
        m.eval(); ps=[]; ys=[]
        with torch.no_grad():
            for x,y in dl_va: ps.append(m(x.to(DEV)).squeeze(1).cpu().numpy()); ys.append(y.numpy())
        ps=np.concatenate(ps); ys=np.concatenate(ys)
        from scipy.stats import pearsonr
        mae=np.abs(ps-ys).mean(); r=pearsonr(ps,ys)[0]
        log.append({"ep":ep,"train_loss":tl/len(dl_tr.dataset),"val_mae":float(mae),"val_r":float(r),"sec":time.time()-t0})
        print(f"ep{ep}: train_loss={tl/len(dl_tr.dataset):.3f} val_MAE={mae:.2f}wk val_r={r:.3f} ({time.time()-t0:.0f}s)",flush=True)
        json.dump(log,open(os.path.join(OUT,"log.json"),"w"))
        if mae<best: best=mae; torch.save(m.state_dict(),os.path.join(OUT,"best.pt"))
    print(f"DONE best val MAE={best:.2f}wk",flush=True)

if __name__=="__main__":
    run(benchmark=("--bench" in sys.argv))
