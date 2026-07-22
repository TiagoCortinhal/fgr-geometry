#!/usr/bin/env python3
"""Evaluate the trained GA CNN on the held-out TEST fetuses + Grad-CAM saliency.
Reports test MAE / r (vs frozen-USF-MAE clock r=0.94 benchmark), per-plane accuracy,
and saves Grad-CAM overlays (true gradient-based, causal — unlike the linear projection).

Inputs: handoff/ga_cnn/best.pt, handoff/ga_cnn_index.csv. Run in env fgrgeom AFTER training.
Outputs: handoff/ga_cnn/test_results.json, handoff/ga_cnn/gradcam/*.png
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision.models as M, torchvision.transforms as T
from PIL import Image
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import zoom

DEV="mps" if torch.backends.mps.is_available() else "cpu"
RES=160
tf=T.Compose([T.Grayscale(3),T.Resize((RES,RES)),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])

def load_model():
    m=M.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,1)
    m.load_state_dict(torch.load("handoff/ga_cnn/best.pt",map_location=DEV)); m.to(DEV).eval(); return m

def evaluate(m, df):
    te=df[df.split=="test"]; ps=[]; ys=[]; pl=[]
    with torch.no_grad():
        for _,r in te.iterrows():
            x=tf(Image.open(r.path).convert("L")).unsqueeze(0).to(DEV)
            ps.append(float(m(x).squeeze().cpu())); ys.append(r.ga_weeks_recovered); pl.append(r.plane_prop)
    ps,ys,pl=np.array(ps),np.array(ys),np.array(pl)
    res={"test_MAE":float(np.abs(ps-ys).mean()),"test_r":float(pearsonr(ps,ys)[0]),"n":len(ys),
         "per_plane":{p:{"MAE":float(np.abs(ps[pl==p]-ys[pl==p]).mean()),
                          "r":float(pearsonr(ps[pl==p],ys[pl==p])[0]),"n":int((pl==p).sum())}
                      for p in np.unique(pl)}}
    json.dump(res,open("handoff/ga_cnn/test_results.json","w"),indent=2); print(json.dumps(res,indent=2))
    return res

def gradcam(m, path, out):
    """Grad-CAM on the last conv block (layer4) — true gradient saliency for the GA prediction."""
    feats={}; grads={}
    h1=m.layer4.register_forward_hook(lambda mod,i,o: feats.__setitem__("a",o.detach()))
    h2=m.layer4.register_full_backward_hook(lambda mod,gi,go: grads.__setitem__("g",go[0].detach()))
    x=tf(Image.open(path).convert("L")).unsqueeze(0).to(DEV)
    m.zero_grad(); y=m(x).squeeze(); y.backward()
    a=feats["a"][0]; g=grads["g"][0]                     # (C,h,w)
    w=g.mean((1,2),keepdim=True); cam=(w*a).sum(0).relu().cpu().numpy()
    cam=cam/(cam.max()+1e-8); cam=zoom(cam,RES/cam.shape[0],order=1)
    h1.remove(); h2.remove()
    img=np.array(Image.open(path).convert("L").resize((RES,RES)))
    fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(img,cmap="gray"); ax.imshow(cam,cmap="jet",alpha=0.45)
    ax.set_title(f"GA pred {float(y):.1f}w",fontsize=8); ax.axis("off")
    fig.savefig(out,dpi=90,bbox_inches="tight"); plt.close(fig)

def main():
    df=pd.read_csv("handoff/ga_cnn_index.csv"); m=load_model()
    evaluate(m, df)
    os.makedirs("handoff/ga_cnn/gradcam",exist_ok=True)
    # Grad-CAM on a GA-balanced sample of test frames
    te=df[df.split=="test"]; samp=te.groupby(te.ga_weeks_recovered.round(),group_keys=False).apply(lambda g:g.sample(min(len(g),5),random_state=0),include_groups=False)
    samp=te.loc[samp.index] if hasattr(samp,"index") else te.sample(60,random_state=0)
    for _,r in samp.iterrows():
        gradcam(m, r.path, f"handoff/ga_cnn/gradcam/{os.path.basename(r.path)}")
    print(f"saved {len(samp)} Grad-CAM overlays")

if __name__=="__main__": main()
