from models import *
from scipy.stats import rankdata
# Exact leave-one-word-out dictionary retrieval for compositional encoding ridge; rank includes homophones.
cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10]
prons=[p for w,p in cands];cw=np.array([w for w,p in cands]);y,q,_=response()
results=[]
settings=[('rel',5,.9,1),('rel',3,1,1),('abs',5,.5,.3),('abs',3,.5,1),('mix',5,.5,.3),('rel',5,.5,1),('rel',5,.9,3),('rel',5,.9,.3),('rel',5,.9,10),('abs',5,.5,3),('abs',5,.5,10),('bag',1,.9,3)]
for kind,n,art,alpha in settings:
 X=featurize(P,kind,n,art);C=featurize(prons,kind,n,art);xm=X.mean(0);X-=xm;C-=xm
 inv=np.linalg.inv(X@X.T+alpha*np.eye(42));H=C@X.T@inv+1/42;HT=X@X.T@inv+1/42
 U=H@y;E=(y-HT@y)/(1-np.diag(HT))[:,None]
 scores=(U*U).sum(1)[:,None]-2*U@y.T-2*H*(U@E.T)+H*H*(E*E).sum(1)[None,:]+2*H*(y*E).sum(1)[None,:]
 out=[]
 for i,w in enumerate(words):
  best=np.min(scores[cw==w,i]);rank=int(1+(scores[:,i]<best-1e-8).sum());inds=np.argsort(scores[:,i])[:5]
  out.append(dict(word=w,rank=rank,best=[cands[k][0] for k in inds]))
 r=dict(kind=kind,n=n,art=art,alpha=alpha,median_rank=float(np.median([x['rank'] for x in out])),mean_log_rank=float(np.mean(np.log([x['rank'] for x in out]))),top10=sum(x['rank']<=10 for x in out),details=out);results.append(r)
 print({k:v for k,v in r.items() if k!='details'},flush=True)
json.dump(results,open('open_vocabulary_validation.json','w'),indent=2)
