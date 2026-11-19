from models import *
from variants import variant
y,q,_=response();cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10];cw=np.array([w for w,p in cands]);results=[];outs={};alls=[]
for n,art,vw,alpha in [(3,1,1,1),(5,.9,1,.3),(5,.9,2,.3),(5,.5,2,1)]:
 X=variant(P,n,art,vw);C=variant([p for w,p in cands],n,art,vw);xm=X.mean(0);X-=xm;C-=xm;cross=C@X.T
 inv=np.linalg.inv(X@X.T+alpha*np.eye(42));H=cross@inv+1/42;HT=X@X.T@inv+1/42;U=H@y;E=(y-HT@y)/(1-np.diag(HT))[:,None]
 base=(U*U).sum(1)[:,None]-2*U@y.T-2*H*(U@E.T)+H*H*(E*E).sum(1)[None,:]+2*H*(y*E).sum(1)[None,:]
 novelty=(C*C).sum(1)-((H-1/42)*cross).sum(1)+alpha/42
 novcv=novelty[:,None]+alpha*H*H/(1-np.diag(HT))[None,:]
 qbase=(U*U).sum(1)[:,None]-2*U@q.T
 for gamma in [0,5,15,40,100,250,600]:
  scores=base+gamma*novcv;ranks=[];details=[]
  for i,w in enumerate(words):
   best=np.min(scores[cw==w,i]);r=int(1+(scores[:,i]<best-1e-8).sum());ranks.append(r);inds=np.argsort(scores[:,i])[:3];details.append(dict(word=w,rank=r,best=[cands[k][0] for k in inds]))
  qs=qbase+gamma*novelty[:,None];out={}
  for j,qid in enumerate(qids):
   inds=np.argsort(qs[:,j]+np.isin(cw,words)*1e9)[:150];out[qid]=[dict(word=cands[k][0],phonemes=cands[k][1],score=float(qs[k,j])) for k in inds]
  key=f'{n}_{art}_{vw}_{alpha}_{gamma}';outs[key]=out;r=dict(n=n,art=art,vowelweight=vw,alpha=alpha,gamma=gamma,median_rank=float(np.median(ranks)),mean_log_rank=float(np.mean(np.log(ranks))),top10=sum(v<=10 for v in ranks),details=details,key=key);results.append(r);print({k:v for k,v in r.items() if k not in ['details','key']},flush=True)
  alls.append((key,qs))
results.sort(key=lambda r:r['mean_log_rank']);json.dump(results,open('uncertainty_validation.json','w'),indent=2);json.dump(outs,open('uncertainty_candidates.json','w'),indent=2)
best=results[0]['key'];scores=next(s for k,s in alls if k==best);np.savez_compressed('uncertainty_scores.npz',scores=scores,words=cw,phonemes=[' '.join(p) for w,p in cands],key=best)
print('best predictions',best,[(qid,[v['word'] for v in outs[best][qid]][:10]) for qid in qids],flush=True)
