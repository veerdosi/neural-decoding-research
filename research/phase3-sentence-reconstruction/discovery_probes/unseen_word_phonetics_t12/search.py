from models import *
y,q,trans=response()
cands=[]
for w,ps in D.items():
 if not w.isalpha() or w in words:continue
 for p in ps:
  if 1<=len(p)<=10:cands.append((w,p))
print('dictionary candidates',len(cands))
models=[('rel',5,.9,1),('abs',5,.5,.3),('abs',5,0,.1),('mix',5,.5,.3),('rel',3,1,1),('abs',3,.5,1)]
results={}; arr=[]
for kind,n,art,alpha in models:
 X=featurize(P,kind,n,art);C=featurize([p for w,p in cands],kind,n,art);yp=fitpred(X,y,C,alpha)
 scores=(yp**2).sum(1)[:,None]-2*yp@q.T
 ranks=np.argsort(scores,axis=0)[:100]
 key=f'{kind}_{n}_{art}_{alpha}'; results[key]={}
 for j,qid in enumerate(qids):
  results[key][qid]=[dict(word=cands[i][0],phonemes=cands[i][1],score=round(float(scores[i,j]),4)) for i in ranks[:,j]]
 print(key,[(qid,[(cands[i][0],round(float(scores[i,j]-scores[ranks[0,j],j]),1)) for i in ranks[:7,j]]) for j,qid in enumerate(qids)])
 arr.append(scores)
json.dump(results,open('algorithm_candidates.json','w'),indent=2)
np.savez_compressed('candidate_scores.npz',scores=np.stack(arr),words=np.array([w for w,p in cands]),phonemes=np.array([' '.join(p) for w,p in cands]),models=np.array([str(m) for m in models]))
np.savez_compressed('response_transform.npz',W=trans[0],mu=trans[1],V=trans[2])
