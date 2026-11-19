from models import *
from variants import variant
from pathlib import Path
# Final search includes every entry/pronunciation in the supplied dictionary; no candidate phone-count bound.
y,q,trans=response();cands=[(w,p) for w,ps in D.items() for p in ps];cw=np.array([w for w,p in cands]);cp=np.array([' '.join(p) for w,p in cands]);N=len(cands)
settings=[dict(n=3,art=1,vowelweight=1,normalize=0,alpha=1,gamma=0),dict(n=5,art=.9,vowelweight=2,normalize=0,alpha=.3,gamma=15),dict(n=5,art=1,vowelweight=2,normalize=.5,alpha=.1,gamma=0)]
allcv=[];allq=[];val=[];fits={};primary_top={}
for k,s in enumerate(settings):
 args={n:s[n] for n in ['n','art','vowelweight','normalize']};X=variant(P,**args);C=variant([p for w,p in cands],**args);xm=X.mean(0);X-=xm;C-=xm;alpha=s['alpha'];inv=np.linalg.inv(X@X.T+alpha*np.eye(42));cross=C@X.T;H=cross@inv+1/42;HT=X@X.T@inv+1/42;U=H@y;E=(y-HT@y)/(1-np.diag(HT))[:,None]
 base=(U*U).sum(1)[:,None]-2*U@y.T-2*H*(U@E.T)+H*H*(E*E).sum(1)[None,:]+2*H*(y*E).sum(1)[None,:]
 novelty=np.maximum((C*C).sum(1)-((H-1/42)*cross).sum(1)+alpha/42,0);novcv=novelty[:,None]+alpha*H*H/(1-np.diag(HT))[None,:]
 cvscore=base+s['gamma']*novcv;qs=(U*U).sum(1)[:,None]-2*U@q.T+s['gamma']*novelty[:,None];allcv.append(cvscore);allq.append(qs)
 details=[]
 for i,w in enumerate(words):
  best=cvscore[cw==w,i].min();rank=int(1+(cvscore[:,i]<best-1e-8).sum());inds=np.argsort(cvscore[:,i])[:5];details.append(dict(word=w,rank=rank,top5=[dict(word=cands[j][0],phonemes=cands[j][1]) for j in inds]))
 resid=E;null=y*42/41
 val.append(dict(model=s,mean_neural_squared_error=float(np.mean((resid**2).sum(1))),null_mean_neural_squared_error=float(np.mean((null**2).sum(1))),relative_error_reduction=float(1-np.sum(resid**2)/np.sum(null**2)),median_dictionary_rank=float(np.median([v['rank'] for v in details])),top1=sum(v['rank']==1 for v in details),top10=sum(v['rank']<=10 for v in details),top100=sum(v['rank']<=100 for v in details),details=details))
 fits[f'coef_{k}']=X.T@inv@y;fits[f'feature_mean_{k}']=xm;fits[f'inverse_{k}']=inv;fits[f'train_features_{k}']=X
 if k==1:
  for j,qid in enumerate(qids):
   order=np.argsort(qs[:,j]+np.isin(cw,words)*1e12)[:100];primary_top[qid]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(qs[i,j])) for i in order]
 print('model',k,{a:b for a,b in val[-1].items() if a not in ['model','details']},flush=True)
cvscore=np.mean(allcv,0);qs=np.mean(allq,0);details=[]
for i,w in enumerate(words):
 best=cvscore[cw==w,i].min();r=int(1+(cvscore[:,i]<best-1e-8).sum());details.append(dict(word=w,rank=r))
val.append(dict(model='unweighted ensemble of preceding three',median_dictionary_rank=float(np.median([v['rank'] for v in details])),top1=sum(v['rank']==1 for v in details),top10=sum(v['rank']<=10 for v in details),top100=sum(v['rank']<=100 for v in details),details=details))
outs={};tops=[]
for j,qid in enumerate(qids):
 order=np.argsort(qs[:,j]+np.isin(cw,words)*1e12)[:200];outs[qid]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(qs[i,j])) for i in order];t=outs[qid][0];tops.append(dict(id=qid,word=t['word'],phonemes=t['phonemes']))
np.savez_compressed('final_fitted_model.npz',**fits,W=trans[0],response_mean=trans[1],response_basis=trans[2],phone_articulatory_features=F,phones=phones,feature_names=fn,settings=json.dumps(settings),training_words=words,training_pronunciations=[' '.join(p) for p in P])
np.savez_compressed('final_full_dictionary_scores.npz',scores=qs,words=cw,phonemes=cp,query_ids=qids)
json.dump(dict(dictionary_entries=len(D),dictionary_pronunciations=N,validation=val),open('final_validation.json','w'),indent=2)
json.dump(dict(primary_model_candidates=primary_top,ensemble_candidates=outs,ensemble_top1=tops),open('algorithm_only_outputs.json','w'),indent=2)
print('ensemble',val[-1],flush=True);print('algorithm top1',tops,flush=True)
