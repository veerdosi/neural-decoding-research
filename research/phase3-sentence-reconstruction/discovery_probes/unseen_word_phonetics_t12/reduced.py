from models import *
y,q,_=response();cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10];cw=np.array([w for w,p in cands]);results=[];outputs={}
for n,art,alpha in [(3,1,.3),(3,1,1),(5,.9,.3),(5,.9,1)]:
 X=featurize(P,'rel',n,art);C=featurize([p for w,p in cands],'rel',n,art);xm=X.mean(0);X-=xm;C-=xm
 inv=np.linalg.inv(X@X.T+alpha*np.eye(42));H=C@X.T@inv+1/42;HT=X@X.T@inv+1/42;U=H@y;E=(y-HT@y)/(1-np.diag(HT))[:,None]
 for rank in [2,4,6,8,12,20]:
  ranks=[];details=[]
  for i,w in enumerate(words):
   keep=np.arange(42)!=i
   trainpred=HT@y-HT[:,i,None]*E[i,None,:]
   u,s,v=np.linalg.svd(trainpred[keep],full_matrices=False);v=v[:rank].T
   pred=U@v-H[:,i,None]*(E[i]@v)[None,:];target=y[i]@v;scores=(pred*pred).sum(1)-2*pred@target
   best=np.min(scores[cw==w]);r=int(1+(scores<best-1e-8).sum());ranks.append(r);inds=np.argsort(scores)[:3];details.append(dict(word=w,rank=r,best=[cands[k][0] for k in inds]))
  r=dict(n=n,art=art,alpha=alpha,response_rank=rank,median_rank=float(np.median(ranks)),mean_log_rank=float(np.mean(np.log(ranks))),top10=sum(k<=10 for k in ranks),details=details);results.append(r);print({k:v for k,v in r.items() if k!='details'},flush=True)
  u,s,v=np.linalg.svd(HT@y,full_matrices=False);v=v[:rank].T;pred=U@v;qq=q@v;scores=(pred*pred).sum(1)[:,None]-2*pred@qq.T;out={}
  for j,qid in enumerate(qids):
   inds=np.argsort(scores[:,j]+np.isin(cw,words)*1e9)[:100];out[qid]=[dict(word=cands[k][0],phonemes=cands[k][1],score=float(scores[k,j])) for k in inds]
  key=f'{n}_{art}_{alpha}_{rank}';r['key']=key;outputs[key]=out
results.sort(key=lambda r:r['mean_log_rank']);json.dump(results,open('reduced_validation.json','w'),indent=2);json.dump(outputs,open('reduced_candidates.json','w'),indent=2)
print('best',results[0]['key'],[(qid,[v['word'] for v in outputs[results[0]['key']][qid]][:10]) for qid in qids],flush=True)
