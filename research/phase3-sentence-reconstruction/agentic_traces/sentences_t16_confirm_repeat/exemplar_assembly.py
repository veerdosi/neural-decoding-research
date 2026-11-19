"""Auditable nearest interval exemplar assembly; no fitted parameters.
Uses only manual calibration interval ledger + unsupervised features.
The outputs are ALGORITHM ONLY, and are not final reconstructions.
"""
from local_retrieval import *
A=make_arrays('drop2')
ledger=json.load(open('calibration_interval_ledger.json'))
words=sorted({x['word'] for x in ledger});wi={w:i for i,w in enumerate(words)}
ex=[];V=[]
for t in ledger:
 for ds,de in [(0,0),(2,-2),(-2,0),(0,2)]:
  s=max(2,t['start']+ds);e=min(len(A[t['id']])-2,t['end']+de)
  if e-s<4:continue
  ex.append(dict(t,start=s,end=e));V.append(interval_vector(A[t['id']],s,e,8))
V=np.array(V,dtype=np.float32);groups=[np.array([i for i,t in enumerate(ex) if t['word']==w]) for w in words]
preds=[];evidence={}
for q in meta['queries']:
 qid=q['id'];a=A[qid];n=len(a);spans=[(s,e) for s in range(1,n-5) for e in range(s+5,min(n-1,s+27)+1)]
 Q=np.array([interval_vector(a,s,e,8) for s,e in spans],np.float32);S=Q@V.T
 # Modest preference for durations within a factor of 2 of the stored exemplar.
 qdur=np.array([e-s for s,e in spans]);cdur=np.array([t['end']-t['start'] for t in ex]);S-=.035*np.maximum(0,np.abs(np.log(qdur[:,None]/cdur[None,:]))-.35)
 best=np.empty((len(spans),len(words)));which=np.empty(best.shape,int)
 for j,g in enumerate(groups):
  ix=np.argmax(S[:,g],axis=1);which[:,j]=g[ix];best[:,j]=S[np.arange(len(spans)),g[ix]]
 # Semi-Markov path: complete coverage apart from <=5 leading/trailing bins.
 dp=np.full(n+1,-1e9);back={}
 dp[1:6]=0
 for k,(s,e) in enumerate(spans):
  j=int(np.argmax(best[k]));value=dp[s]+(best[k,j]-.21)*(e-s)-.52
  if value>dp[e]:dp[e]=value;back[e]=(s,k,j)
 end=max(range(n-5,n),key=lambda e:dp[e]);end0=end;path=[]
 while end in back:
  s,k,j=back[end];t=ex[which[k,j]];path.append({'word':words[j],'start':s,'end':end,'similarity':float(best[k,j]),'exemplar':t});end=s
 path.reverse();preds.append({'id':qid,'tokens':[r['word'] for r in path]});evidence[qid]={'path':path,'score':float(dp[end0]),'leading_bins':end,'trailing_bins':n-end0}
 # Save compact rank evidence for windows plus best-per-word lattice arrays.
 np.savez_compressed('lattice_'+qid+'.npz',spans=np.array(spans),scores=best,which=which,words=np.array(words))
Path('algorithm_only_predictions.json').write_text(json.dumps(preds,indent=2))
Path('algorithm_only_alignment.json').write_text(json.dumps(evidence,indent=2))
Path('exemplars.json').write_text(json.dumps(ex))
for p in preds:print(p['id'],' '.join(p['tokens']))
