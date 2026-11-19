exec(open('explore.py').read().split("print('words'")[0])
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
mu=Y.mean(0); R=np.stack([z[x['id']]-Y[words.index(x['word'])] for x in inp['calibration']])
C=np.einsum('ntc,ntd->cd',R,R)/(len(R)*20)
e,U=eigh(C); W=(U*(1/np.sqrt(e+.2)))@U.T
V=gaussian_filter1d((Y-mu)@W,1,axis=1)
print('noise covariance eigen quantiles',np.round(np.quantile(e,[0,.25,.5,.75,1]),3))
print('signal energy time',np.round(np.mean(V**2,axis=(0,2)),3))
for w in ['no','not','need','nurse','i','my','good','goodbye','feel','tell','here','where','you','do','that','they']:
 i=words.index(w); ds=((V-V[i])**2).sum((1,2)); inds=np.argsort(ds)[1:5]; print(w,[(words[j],round(ds[j],1)) for j in inds])
np.savez('whiten.npz',W=W,mu=mu,R=R)
