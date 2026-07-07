import numpy as np, umap, time
Ep=np.load("/Users/tiago/PythonProject/fgr-geometry/results/img_align/_pca50.npy")
t0=time.time()
reducer=umap.UMAP(n_neighbors=30,min_dist=0.1,n_components=2,metric="euclidean",random_state=42,verbose=True)
emb2d=reducer.fit_transform(Ep[:,:50])
np.save("/Users/tiago/PythonProject/fgr-geometry/results/img_align/_umap2d.npy",emb2d)
print("UMAP done %.0fs shape %s"%(time.time()-t0,emb2d.shape))
