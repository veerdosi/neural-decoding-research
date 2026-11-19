import json,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
A=np.load('analysis_data.npz'); words=list(A['words']); P=[s.split() for s in A['P']]; Y=A['Y']; Q=A['Q']; qids=list(A['qids'])
D=json.load(open('pronunciations.json')); phones=sorted(set(x for v in D.values() for p in v for x in p)); pi={p:i for i,p in enumerate(phones)}
# Shared articulatory descriptors: consonant manner, place, voicing; vowel height, backness, rounding, diphthong endpoints.
cons={
'P':('stop','labial',0),'B':('stop','labial',1),'M':('nasal','labial',1),'F':('fric','labdent',0),'V':('fric','labdent',1),
'TH':('fric','dental',0),'DH':('fric','dental',1),'T':('stop','alveolar',0),'D':('stop','alveolar',1),'N':('nasal','alveolar',1),
'S':('fric','alveolar',0),'Z':('fric','alveolar',1),'SH':('fric','postalv',0),'ZH':('fric','postalv',1),'CH':('affric','postalv',0),'JH':('affric','postalv',1),
'K':('stop','velar',0),'G':('stop','velar',1),'NG':('nasal','velar',1),'HH':('fric','glottal',0),'L':('lateral','alveolar',1),'R':('rhotic','postalv',1),'W':('glide','velar',1),'Y':('glide','palatal',1)}
vowels={'IY':(0,0,0,0,0),'IH':(.2,.1,0,.2,.1),'EH':(.5,0,0,.5,0),'AE':(.9,0,0,.9,0),'AA':(1,1,0,1,1),'AO':(.6,1,1,.6,1),'AH':(.6,.5,0,.6,.5),'ER':(.5,.5,0,.5,.5),'UH':(.25,.8,1,.25,.8),'UW':(0,1,1,0,1),'EY':(.45,0,0,0,0),'AY':(1,.5,0,0,0),'OW':(.5,1,1,0,1),'AW':(1,.5,0,0,1),'OY':(.6,1,1,0,0)}
fn=['consonant','vowel','voiced','stop','fric','nasal','affric','lateral','rhotic','glide','labial','labdent','dental','alveolar','postalv','velar','glottal','palatal','high','mid','low','front','central','back','round','diph','endhigh','endfront','endback']
F=np.zeros((len(phones),len(fn)))
for p,i in pi.items():
 f={}
 if p in cons:
  manner,place,voiced=cons[p];f={'consonant':1,'voiced':voiced,manner:1,place:1}
  if p=='W':f.update(labial=1,round=1)
 else:
  h,b,r,h2,b2=vowels.get(p,(.5,.5,0,.5,.5));f={'vowel':1,'voiced':1,'high':1-h,'mid':1-abs(h-.5)*2,'low':h,'front':1-b,'central':1-abs(b-.5)*2,'back':b,'round':r,'diph':float(h!=h2 or b!=b2),'endhigh':1-h2,'endfront':1-b2,'endback':b2}
  if p=='ER':f['rhotic']=1
 for k,v in f.items(): F[i,fn.index(k)]=v
# Scale features to similarity rather than raw dimensionality.
F=F/np.sqrt((F**2).sum(1,keepdims=True))
def featurize(prons,kind='rel',n=4,art=.5):
 E=np.concatenate([np.eye(len(phones))*np.sqrt(1-art),F*np.sqrt(art)],axis=1); d=E.shape[1]
 out=[]
 for p in prons:
  L=len(p); e=E[[pi[x] for x in p]]
  if kind=='bag': x=e.sum(0)
  elif kind=='rel':
   pos=(np.arange(L)+.5)/L; anchors=np.linspace(0,1,n)
   wt=np.exp(-.5*((pos[:,None]-anchors[None,:])/(.65/(n-1)))**2);wt/=wt.sum(1,keepdims=True)
   x=(wt.T@e).ravel()
  elif kind=='abs':
   x=np.zeros((n*2+1,d))
   for j,v in enumerate(e):
    if j<n:x[j]+=v
    if L-1-j<n:x[n+L-1-j]+=v
    x[-1]+=v*.4
   x=x.ravel()
  elif kind=='mix':
   x=np.concatenate([featurize([p],'rel',n,art)[0],featurize([p],'abs',3,art)[0]*.5])
  out.append(x)
 return np.array(out)
def response(smooth=1,whiten=.2):
 B=np.load('whiten.npz'); R=B['R']; C=np.einsum('ntc,ntd->cd',R,R)/(len(R)*20);e,U=eigh(C);W=(U*(1/np.sqrt(e+whiten)))@U.T
 y=gaussian_filter1d(Y@W,smooth,axis=1).reshape(42,-1);q=gaussian_filter1d(Q@W,smooth,axis=1).reshape(16,-1)
 # compress response span losslessly for squared-distance ranking
 mu=y.mean(0);u,s,v=np.linalg.svd(y-mu,full_matrices=False);y=(y-mu)@v.T;q=(q-mu)@v.T
 return y,q,(W,mu,v)
def fitpred(X,y,C,alpha):
 xm=X.mean(0); ym=y.mean(0);X=X-xm; C=C-xm
 dual=np.linalg.solve(X@X.T+np.eye(len(X))*alpha,y-ym)
 return C@X.T@dual+ym
if __name__=='__main__':
 y,q,_=response();results=[]
 for kind in ['bag','rel','abs','mix']:
  for n in ([1] if kind=='bag' else [3,5]):
   for art in [0,.5,.9,1]:
    X=featurize(P,kind,n,art)
    for alpha in [.03,.1,.3,1,3,10,30]:
     preds=[];ranks=[];errs=[]
     for i in range(42):
      mask=np.arange(42)!=i; yp=fitpred(X[mask],y[mask],X,alpha)
      dist=((yp-y[i])**2).sum(1); ranks.append(1+(dist<dist[i]).sum());preds.append(yp[i]);errs.append(dist[i])
     results.append(dict(kind=kind,n=n,art=art,alpha=alpha,mse=float(np.mean(errs)),top1=float(np.mean(np.array(ranks)==1)),mrr=float(np.mean(1/np.array(ranks))),median_rank=float(np.median(ranks))))
 results.sort(key=lambda r:r['mse']);json.dump(results,open('model_validation.json','w'),indent=2)
 print('best mse',results[:6]);print('best retrieval',sorted(results,key=lambda r:-r['mrr'])[:6])
