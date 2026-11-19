from models import *

def variant(prons,n=5,art=.9,vowelweight=1,normalize=0,role='relative'):
 E=np.concatenate([np.eye(len(phones))*np.sqrt(1-art),F*np.sqrt(art)],axis=1);d=E.shape[1];out=[]
 for p in prons:
  L=len(p);e=E[[pi[x] for x in p]]
  if role=='relative':
   dur=np.array([vowelweight if x in vowels else 1 for x in p]);pos=(np.cumsum(dur)-dur/2)/dur.sum(); anchors=np.linspace(0,1,n)
   wt=np.exp(-.5*((pos[:,None]-anchors)/(0.65/(n-1)))**2);wt/=wt.sum(1,keepdims=True); x=(wt.T@e).ravel()/L**normalize
  else:
   x=np.zeros((7,d));vi=[i for i,x in enumerate(p) if x in vowels]
   v0=vi[0] if vi else 0;v1=vi[-1] if vi else L-1
   for j in range(L):
    if j<v0:r=0
    elif j==v0:r=1
    elif j<v1:r=3
    elif j==v1:r=4
    else:r=5
    x[r]+=e[j];x[6]+=e[j]*.2
   if len(vi)==1:
    x[2]=x[5];x[5]=0
   x=x.ravel()
  out.append(x)
 return np.array(out)
if __name__=='__main__':
 y,q,_=response();cands=[(w,p) for w,ps in D.items() if w.isalpha() for p in ps if len(p)<=10];prons=[p for w,p in cands];cw=np.array([w for w,p in cands]);results=[];outs={};alls=[]
 settings=[dict(n=n,art=art,vowelweight=vw,normalize=nm,role=role) for role,n,art,vw,nm in [('relative',5,.9,1,0),('relative',3,1,1,0),('relative',3,.9,2,0),('relative',5,.9,2,0),('relative',5,.9,.5,0),('relative',5,.9,1,.5),('relative',3,1,1,1),('relative',5,1,2,.5),('syllable',5,.9,1,0),('syllable',5,.5,1,0)]]
 for args in settings:
  X=variant(P,**args);C=variant(prons,**args);xm=X.mean(0);X-=xm;C-=xm
  for alpha in [.03,.1,.3,1]:
   inv=np.linalg.inv(X@X.T+alpha*np.eye(42));H=C@X.T@inv+1/42;HT=X@X.T@inv+1/42;U=H@y;E=(y-HT@y)/(1-np.diag(HT))[:,None]
   scores=(U*U).sum(1)[:,None]-2*U@y.T-2*H*(U@E.T)+H*H*(E*E).sum(1)[None,:]+2*H*(y*E).sum(1)[None,:]
   ranks=[];det=[]
   for i,w in enumerate(words):
    best=np.min(scores[cw==w,i]);rank=int(1+(scores[:,i]<best-1e-8).sum());ranks.append(rank);inds=np.argsort(scores[:,i])[:3];det.append(dict(word=w,rank=rank,best=[cands[k][0] for k in inds]))
   qs=(U*U).sum(1)[:,None]-2*U@q.T;out={}
   for j,qid in enumerate(qids):
    inds=np.argsort(qs[:,j]+np.isin(cw,words)*1e9)[:80];out[qid]=[dict(word=cands[k][0],phonemes=cands[k][1],score=float(qs[k,j])) for k in inds]
   key=str(args)+str(alpha);outs[key]=out
   r=dict(**args,alpha=alpha,median_rank=float(np.median(ranks)),mean_log_rank=float(np.mean(np.log(ranks))),top10=sum(v<=10 for v in ranks),details=det,key=key);results.append(r)
   print({k:v for k,v in r.items() if k not in ['details','key']},flush=True)
 results.sort(key=lambda r:r['mean_log_rank']);json.dump(results,open('variant_validation.json','w'),indent=2);json.dump(outs,open('variant_candidates.json','w'),indent=2)
 print('best predictions',[(qid,[v['word'] for v in outs[results[0]['key']][qid]][:10]) for qid in qids])
