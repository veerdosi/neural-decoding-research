from models import *
from scipy.linalg import cho_factor,cho_solve
B=np.load('whiten.npz'); W=B['W']; YY=gaussian_filter1d(Y@W,1,axis=1); QQ=gaussian_filter1d(Q@W,1,axis=1)
# Temporal phoneme code: each phoneme drives a Gaussian neural gesture, with channel code shared across every word and position.
def gesture(prons,start=2,span=9,width=1.5,art=.5,mode='relative',deriv=False):
 E=np.concatenate([np.eye(len(phones))*np.sqrt(1-art),F*np.sqrt(art)],axis=1);t=np.arange(20);out=[]
 for p in prons:
  L=len(p);e=E[[pi[x] for x in p]]
  if mode=='relative':c=start+span*(np.arange(L)+.5)/L
  else:c=start+span*np.arange(L)
  v=(t[:,None]-c[None,:])/width;G=np.exp(-.5*v*v)
  if deriv:out.append(np.concatenate([G@e,((v)*G)@e],axis=1))
  else:out.append(G@e)
 return np.array(out)
def fitgen(X,y,alpha):
 n,t,d=X.shape; xm=X.mean(0);ym=y.mean(0); xx=(X-xm).reshape(-1,d); yy=(y-ym).reshape(n*t,-1)
 inv=np.linalg.inv(xx.T@xx+alpha*np.eye(d));coef=inv@xx.T@yy
 return coef,xm,ym,inv

def cv(X,y,alpha,rank=False):
 n,t,d=X.shape
 # Explicit holds to keep centering exact.
 errs=[];ranks=[]
 for i in range(n):
  keep=np.arange(n)!=i;coef,xm,ym,inv=fitgen(X[keep],y[keep],alpha);pred=(X-xm)@coef+ym
  errs.append(((pred[i]-y[i])**2).sum())
  if rank:
   ds=((pred-y[i])**2).sum((1,2));ranks.append(1+(ds<ds[i]).sum())
 return float(np.mean(errs)),ranks
if __name__=='__main__':
 results=[]
 for mode in ['relative','absolute']:
  for start in [0,2,4]:
   for span in ([6,10,14] if mode=='relative' else [1.5,3,4.5]):
    for width in [1,2,3]:
     for art in [0,.7,1]:
      X=gesture(P,start,span,width,art,mode)
      for alpha in [1,10,100]:
       err,ranks=cv(X,YY,alpha)
       results.append(dict(mode=mode,start=start,span=span,width=width,art=art,alpha=alpha,mse=err))
 results.sort(key=lambda r:r['mse']);json.dump(results,open('generative_validation.json','w'),indent=2)
 print(results[:10],flush=True)
 r=results[0];X=gesture(P,**{k:r[k] for k in ['start','span','width','art','mode']});err,ranks=cv(X,YY,r['alpha'],True);print('best retrieval',sum(v==1 for v in ranks),np.mean(1/np.array(ranks)),flush=True)
 coef,xm,ym,inv=fitgen(X,YY,r['alpha']);np.savez('generative_fit.npz',coef=coef,xm=xm,ym=ym,W=W)
 # All dictionary pronunciations, no word-type menu.
 cands=[(w,p) for w,ps in D.items() if w.isalpha() and w not in words for p in ps if len(p)<=10]
 scores=[]
 for k in range(0,len(cands),5000):
  C=gesture([p for w,p in cands[k:k+5000]],**{k:r[k] for k in ['start','span','width','art','mode']});pred=(C-xm)@coef+ym
  pred=pred.reshape(len(C),-1);q=QQ.reshape(16,-1);scores.append((pred**2).sum(1)[:,None]-2*pred@q.T)
 scores=np.concatenate(scores);out={}
 for j,qid in enumerate(qids):
  inds=np.argsort(scores[:,j])[:100];out[qid]=[dict(word=cands[i][0],phonemes=cands[i][1],score=float(scores[i,j])) for i in inds];print(qid,[(cands[i][0],round(float(scores[i,j]-scores[inds[0],j]),1)) for i in inds[:10]],flush=True)
 json.dump(out,open('generative_candidates.json','w'),indent=2)
 np.savez_compressed('generative_scores.npz',scores=scores,words=[w for w,p in cands],phonemes=[' '.join(p) for w,p in cands])
