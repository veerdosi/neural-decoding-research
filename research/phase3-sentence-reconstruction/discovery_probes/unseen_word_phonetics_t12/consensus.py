from models import *
from variants import variant
y,q,tr=response();cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10];cw=np.array([w for w,p in cands]);allscore=[]
for n,art,vw,alpha,gamma in [(3,1,1,1,0),(5,.9,2,.3,15),(5,1,2,.1,0)]:
 X=variant(P,n,art,vw,normalize=.5 if n==5 and art==1 else 0);C=variant([p for w,p in cands],n,art,vw,normalize=.5 if n==5 and art==1 else 0);xm=X.mean(0);X-=xm;C-=xm;inv=np.linalg.inv(X@X.T+alpha*np.eye(42));H=C@X.T@inv+1/42;pred=H@y
 novelty=(C*C).sum(1)-((H-1/42)*(C@X.T)).sum(1)+alpha/42
 score=(pred*pred).sum(1)[:,None]-2*pred@q.T+gamma*novelty[:,None];score[np.isin(cw,words)]=1e9;allscore.append(score)
scores=np.mean(allscore,axis=0);pairs=[[0,5],[1,2],[3,12],[4,8],[6,14],[7,13],[9,10],[11,15]];out={};single={}
for j,qid in enumerate(qids):
 inds=np.argsort(scores[:,j])[:500];single[qid]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(scores[i,j])) for i in inds]
for pair in pairs:
 s=scores[:,pair].mean(1);inds=np.argsort(s)[:500];key=','.join(qids[i] for i in pair);out[key]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(s[i])) for i in inds];print(key,[(cands[i][0],round(float(s[i]-s[inds[0]]),1)) for i in inds[:80]],flush=True)
json.dump(out,open('possible_repeat_consensus.json','w'),indent=2);json.dump(single,open('consensus_individual.json','w'),indent=2)
np.savez_compressed('consensus_scores.npz',scores=scores,words=cw,phonemes=[' '.join(p) for w,p in cands])
