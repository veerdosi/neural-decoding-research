from retrieval import *
X=np.concatenate([raw[c['id']] for c in meta['calibration']]).astype(float)
mu=X.mean(0)
w,v=np.linalg.eigh(np.cov(X.T));w=w[::-1];v=v[:,::-1]
np.savez('unsupervised_pca.npz',mean=mu,eigenvalues=w,eigenvectors=v)
