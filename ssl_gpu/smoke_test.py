"""Smoke test on SYNTHETIC data: does the pipeline run end-to-end, and do the
sanity gates actually fire? Runs on CPU in ~2 min. Not a scientific test."""
import numpy as np, os, sys, tempfile, json, shutil
from PIL import Image
sys.path.insert(0,"/Users/tiago/dev/fgr-geometry/ssl_gpu")
D=tempfile.mkdtemp(); F=os.path.join(D,"frames"); os.makedirs(F)
rng=np.random.default_rng(0)
NF, NFR = 60, 6
rows=[]
lat=rng.normal(size=NF)                      # a planted per-fetus factor
for i in range(NF):
    for j in range(NFR):
        # brightness encodes the latent -> a detectable signal must be found
        img=np.clip(rng.normal(0.5+0.15*lat[i],0.12,(64,64)),0,1)
        fn=f"F{i:03d}_{j}.png"
        Image.fromarray((img*255).astype(np.uint8)).save(os.path.join(F,fn))
        rows.append(dict(new_filename=fn,fetus_id=i,dataset_type="impact"))
import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(D,"man.csv"),index=False)
Z=np.column_stack([lat+rng.normal(0,0.4,NF) for _ in range(5)]+
                  [rng.normal(size=NF) for _ in range(5)]+
                  [rng.normal(size=NF) for _ in range(11)])
cols=[f"g{i}" for i in range(5)]+[f"d{i}" for i in range(5)]+[f"c{i}" for i in range(11)]
blocks=["growth"]*5+["Doppler"]*5+["cardiac"]*11
np.savez(os.path.join(D,"panel.npz"),Z=Z,cols=np.array(cols),blocks=np.array(blocks),
         fids=np.arange(NF),ga=rng.normal(33,1,NF),bmi=rng.normal(25,4,NF))
# --- test 1: fetus-level splitter never leaks
from fgm_ssl.data import fetus_level_folds, FrameManifest
allf=list(range(NF)); leaks=0
for tr,te in fetus_level_folds(allf,5,0):
    if set(tr)&set(te): leaks+=1
print("TEST fetus-split leak-free:", leaks==0)
# --- test 2: manifest resolves
man=FrameManifest(os.path.join(D,"man.csv"),F).existing()
print("TEST manifest frames:", len(man.df), "expected", NF*NFR, "->", len(man.df)==NF*NFR)
# --- test 3: MAE trains and the loss-moved gate would pass
os.system(f"cd /Users/tiago/dev/fgr-geometry/ssl_gpu && "
  f"/Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python run_ssl.py --arm mae "
  f"--manifest {D}/man.csv --image-root {F} --panel {D}/panel.npz --out {D}/res "
  f"--epochs 6 --batch 16 --size 64 --width 8 --dim 32 --workers 0 --min-fetuses 10 2>&1 | tail -6")
ok=os.path.exists(os.path.join(D,"res","mae_embeddings.npz"))
print("TEST mae wrote embeddings:", ok)
if ok:
    lg=json.load(open(os.path.join(D,"res","mae_log.json")))
    f0=lg["folds"][0]
    print("   loss",round(f0["loss_first"],4),"->",round(f0["loss_last"],4),
          "| dim",lg["embed_dim"],"| scorable",lg["n_scorable"],"| oof",lg["out_of_fold"])
# --- test 4: supervised arm
os.system(f"cd /Users/tiago/dev/fgr-geometry/ssl_gpu && "
  f"/Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python run_ssl.py --arm supervised "
  f"--manifest {D}/man.csv --image-root {F} --panel {D}/panel.npz --out {D}/res "
  f"--epochs 6 --batch 16 --size 64 --width 8 --dim 32 --workers 0 --min-fetuses 10 --cv-folds 3 --target growth 2>&1 | tail -4")
print("TEST supervised wrote:", os.path.exists(os.path.join(D,"res","supervised_embeddings.npz")))
# --- test 5: scoring runs and recovers the planted growth signal
np.savez(os.path.join(D,"frozen.npz"),E=rng.normal(size=(NF,32)),fids=np.arange(NF))
os.system(f"cd /Users/tiago/dev/fgr-geometry/ssl_gpu && "
  f"/Users/tiago/.claude-science/conda/envs/fgrgeom/bin/python score_arms.py "
  f"--panel {D}/panel.npz --results {D}/res --frozen {D}/frozen.npz "
  f"--n-perm 60 --min-n 40 --n-splits 40 --out {D}/scores.json 2>&1 | tail -14")
print("\nSMOKE TEST DIR:",D)
