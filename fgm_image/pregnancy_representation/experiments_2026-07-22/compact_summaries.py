#!/usr/bin/env python3
"""One-time pass: read each heavy all-layer shard ONCE, extract compact summaries, save small.
LS = concat[CLS, mean-patch] per layer (N,12,1536); PT = last-layer patches (N,196,768).
Output: /Users/tiago/usb/usfmae_summaries.npz  (loads in seconds vs 33s/shard decompress).
"""
import numpy as np, glob, zipfile, time, os
fs=sorted(glob.glob('/Users/tiago/usb/usfmae_all_layers/shard_*.npz'))
LS=[];PT=[];GA=[];NID=[];PL=[];t0=time.time()
for i,f in enumerate(fs):
    try: z=np.load(f,allow_pickle=True); tok=z['tokens']
    except zipfile.BadZipFile: print(f"skip {os.path.basename(f)}",flush=True); continue
    LS.append(np.concatenate([tok[:,:,0,:],tok[:,:,1:,:].mean(2)],-1).astype(np.float32))
    PT.append(tok[:,-1,1:,:].astype(np.float32))
    GA.append(z['ga']);NID.append(z['nid'].astype(str));PL.append(z['plane'])
    del tok,z
    print(f"  {i+1}/{len(fs)} {time.time()-t0:.0f}s",flush=True)
np.savez('/Users/tiago/usb/usfmae_summaries.npz',
    LS=np.concatenate(LS),PT=np.concatenate(PT),ga=np.concatenate(GA),
    nid=np.concatenate(NID),plane=np.concatenate(PL))
print(f"DONE {sum(len(x) for x in GA)} frames in {(time.time()-t0)/60:.1f}min",flush=True)
