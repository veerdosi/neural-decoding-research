from models import *
from scipy.linalg import solve
B=np.load('whiten.npz'); R=B['R'];C=np.einsum('ntc,ntd->cd',R,R)/(len(R)*20);e,U=eigh(C);W=(U*(1/np.sqrt(e+.2)))@U.T
y=gaussian_filter1d(Y@W,1,axis=1)
# Word-held-out regression R2 over time, comparing phoneme location hypotheses.
res=[]
for kind,n in [('bag',1),('rel',3),('rel',5),('abs',3),('abs',5)]:
 X=featurize(P,kind,n,.5); yp=np.empty_like(y)
 for i in range(42):
  mask=np.arange(42)!=i;yp[i]=fitpred(X[mask],y[mask].reshape(41,-1),X[i:i+1],1).reshape(20,256)
 err=((yp-y)**2).sum((0,2));base=((y-y.mean(0))**2).sum((0,2));r2=1-err/base
 res.append(dict(kind=kind,n=n,r2=r2.tolist()));print(kind,n,np.round(r2,2))
json.dump(res,open('temporal_validation.json','w'),indent=2)
