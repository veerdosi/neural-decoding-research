from retrieval import *
X=np.concatenate([raw[c['id']] for c in meta['calibration']]).astype(float)
mu=X.mean(0)
w,v=np.linalg.eigh(np.cov(X.T));w=w[::-1];v=v[:,::-1]
np.savez('unsupervised_pca.npz',mean=mu,eigenvalues=w,eigenvectors=v)
outs={}
for tag,power,drop,sig in [('drop2',0,2,1.2),('white',.5,0,1.2),('regwhite',.5,0,2),('pca32',0,0,.8)]:
 weights=1/np.maximum(w,1)**power
 weights[:drop]=0
 if tag=='pca32':weights[32:]=0
 A={}
 for k in raw.files:
  a=gaussian_filter1d((raw[k]-mu)@v*weights,sig,axis=0)
  A[k]=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
 out={}
 print('\nVARIANT',tag)
 for q in meta['queries']:
  rows=[]
  for c in meta['calibration']:
   s,path,C=dtw(A[q['id']],A[c['id']]);rows.append({'id':c['id'],'score':s,'path':path})
  rows.sort(key=lambda r:r['score']);out[q['id']]=rows
  print(q['id'],[(r['id'],round(r['score'],3),' '.join(meta['calibration'][int(r['id'][1:])]['tokens'])) for r in rows[:6]])
 outs[tag]=out
Path('variant_retrieval.json').write_text(json.dumps(outs))
