# INVALID DIAGNOSTIC: validation-label leakage in learned metric. Do not use or run for inference.
from models import *
y,q,_=response();cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10];cw=np.array([w for w,p in cands]);results=[];outs={}
# Infer articulatory sequence descriptors from neural activity, then compare dictionary descriptors.
for kind,n,art in [('rel',3,1),('rel',5,.9),('rel',5,1),('abs',3,.5)]:
 X=featurize(P,kind,n,art);C=featurize([p for w,p in cands],kind,n,art);xm=X.mean(0);xx=X-xm;cc=C-xm
 K=y@y.T; Kq=q@y.T
 for alpha in [50,200,500,1000,3000]:
  inv=np.linalg.inv(K+np.eye(42)*alpha);H=K@inv;pred=H@xx;cv=xx-(xx-pred)/(1-np.diag(H))[:,None];qp=Kq@inv@xx
  for cov in [0,.5,1]:
   if cov:
    err=xx-cv; V=err.T@err/42; ev,U=np.linalg.eigh(V); trans=(U*(1/np.sqrt(ev+.02)))@U.T; trans=(1-cov)*np.eye(len(V))+cov*trans
   else:trans=np.eye(xx.shape[1])
   ct=cc@trans;pt=cv@trans;qt=qp@trans;scores=(ct*ct).sum(1)[:,None]-2*ct@pt.T;ranks=[];details=[]
   for i,w in enumerate(words):
    best=np.min(scores[cw==w,i]);r=int(1+(scores[:,i]<best-1e-8).sum());ranks.append(r);inds=np.argsort(scores[:,i])[:3];details.append(dict(word=w,rank=r,best=[cands[k][0] for k in inds]))
   qs=(ct*ct).sum(1)[:,None]-2*ct@qt.T;out={}
   for j,qid in enumerate(qids):
    inds=np.argsort(qs[:,j]+np.isin(cw,words)*1e9)[:100];out[qid]=[dict(word=cands[k][0],phonemes=cands[k][1],score=float(qs[k,j])) for k in inds]
   key=f'{kind}_{n}_{art}_{alpha}_{cov}';outs[key]=out;r=dict(kind=kind,n=n,art=art,alpha=alpha,cov=cov,median_rank=float(np.median(ranks)),mean_log_rank=float(np.mean(np.log(ranks))),top10=sum(v<=10 for v in ranks),details=details,key=key);results.append(r);print({k:v for k,v in r.items() if k not in ['details','key']},flush=True)
results.sort(key=lambda r:r['mean_log_rank']);json.dump(results,open('inverse_validation.json','w'),indent=2);json.dump(outs,open('inverse_candidates.json','w'),indent=2)
print('best predictions',[(qid,[v['word'] for v in outs[results[0]['key']][qid]][:10]) for qid in qids],flush=True)
