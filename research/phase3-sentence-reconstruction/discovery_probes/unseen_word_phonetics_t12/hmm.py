from models import *
from scipy.linalg import cho_factor,cho_solve
from scipy.ndimage import gaussian_filter1d
B=np.load('whiten.npz');W=B['W'];YY=gaussian_filter1d(Y@W,1,axis=1);QQ=gaussian_filter1d(Q@W,1,axis=1)
E=np.concatenate([np.eye(len(phones))*.7,F*.7],axis=1);E=np.vstack([E,np.zeros((1,E.shape[1]))]);sil=len(phones)
def initial(prons):
 out=[]
 for p in prons:
  L=len(p);lab=np.full(20,sil);lab[2:15]=np.array([pi[x] for x in p])[np.minimum((np.arange(13)/13*L).astype(int),L-1)];out.append(lab)
 return np.array(out)
def fit(lab,y,alpha=20):
 n,t,c=y.shape;X=E[lab];xm=X.mean(0);ym=y.mean(0);XX=(X-xm).reshape(-1,E.shape[1]);Y0=(y-ym).reshape(-1,c);coef=np.linalg.solve(XX.T@XX+np.eye(XX.shape[1])*alpha,XX.T@Y0);return coef,xm,ym

def align(cost,seq,durpen=1,startpen=.5):
 # Segmental DP with positive phone durations, leading/trailing silence 0..8.
 L=len(seq);T=len(cost);cs=np.vstack([np.zeros(cost.shape[1]),np.cumsum(cost,axis=0)])
 dp=np.full((L+1,T+1),1e12);back=np.zeros((L+1,T+1),int)
 for t in range(9):dp[0,t]=cs[t,sil]+startpen*(t-2)**2
 for j,p in enumerate(seq,1):
  for t in range(j,T+1):
   ss=np.arange(max(j-1,t-9),t);ds=t-ss;vals=dp[j-1,ss]+cs[t,p]-cs[ss,p]+durpen*(ds-2.6)**2
   k=np.argmin(vals);dp[j,t]=vals[k];back[j,t]=ss[k]
 ends=np.arange(max(L,T-9),T+1);vals=dp[L,ends]+cs[T,sil]-cs[ends,sil]+startpen*(T-ends-5)**2
 t=ends[np.argmin(vals)];score=float(np.min(vals));labels=np.full(T,sil)
 for j in range(L,0,-1):
  s=back[j,t];labels[s:t]=seq[j-1];t=s
 return labels,score

def train(prons,y,alpha=20,iters=5,durpen=3):
 lab=initial(prons)
 for it in range(iters):
  coef,xm,ym=fit(lab,y,alpha);pred=(E[None,:,:]-xm[:,None,:])@coef+ym[:,None,:]
  # whitened residual neural distances scaled: smooth sample variance about .2, mean noise .05; use 1/4 so penalty moderate
  for i,p in enumerate(prons):
   cost=((y[i,:,None,:]-pred)**2).sum(2)/4
   lab[i],_=align(cost,[pi[x] for x in p],durpen)
 return fit(lab,y,alpha),lab
if __name__=='__main__':
 params,lab=train(P,YY,20,6,3);coef,xm,ym=params;pred=(E[None,:,:]-xm[:,None,:])@coef+ym[:,None,:]
 np.savez_compressed('hmm_fit.npz',coef=coef,xm=xm,ym=ym,lab=lab,emissions=pred,E=E)
 print('alignments',[(w,[phones[p] if p!=sil else '_' for p in l]) for w,l in zip(words,lab)],flush=True)
 # Save neural emission scores for arbitrary phoneme sequence hypotheses.
 costs=np.array([((q[:,None,:]-pred)**2).sum(2)/4 for q in QQ]);np.save('hmm_query_costs.npy',costs)
 # Decode dictionary with dynamic programming over all pronunciation prefixes in a trie.
 cands=[(w,p) for w,ps in D.items() if w.isalpha() and w not in words for p in ps if len(p)<=9]
 trie={};ends=[]
 for idx,(w,p) in enumerate(cands):
  node=trie
  for ph in p:node=node.setdefault(pi[ph],{})
  node.setdefault(-1,[]).append(idx)
 out={};allscore=[]
 for qi,cost in enumerate(costs):
  T=20;cs=np.vstack([np.zeros(cost.shape[1]),np.cumsum(cost,axis=0)]);dp=np.full(21,1e12)
  for t in range(9):dp[t]=cs[t,sil]+.5*(t-2)**2
  score=np.full(len(cands),1e12)
  def walk(node,prev,depth):
   for p,child in node.items():
    if p==-1:
     ee=np.arange(max(depth,T-9),T+1);val=np.min(prev[ee]+cs[T,sil]-cs[ee,sil]+.5*(T-ee-5)**2)
     score[child]=val;continue
    new=np.full(21,1e12)
    for t in range(depth+1,T+1):
     ss=np.arange(max(depth,t-9),t);new[t]=np.min(prev[ss]+cs[t,p]-cs[ss,p]+3*(t-ss-2.6)**2)
    walk(child,new,depth+1)
  walk(trie,dp,0);inds=np.argsort(score)[:100];out[qids[qi]]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(score[i])) for i in inds];allscore.append(score)
  print(qids[qi],[(cands[i][0],round(float(score[i]-score[inds[0]]),1)) for i in inds[:10]],flush=True)
 json.dump(out,open('hmm_candidates.json','w'),indent=2);np.savez_compressed('hmm_scores.npz',scores=np.array(allscore).T,words=[w for w,p in cands],phonemes=[' '.join(p) for w,p in cands])
