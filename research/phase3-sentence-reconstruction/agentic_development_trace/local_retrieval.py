from retrieval import *
from scipy.interpolate import interp1d

def make_arrays(tag):
 p=np.load('unsupervised_pca.npz');w=p['eigenvalues'];v=p['eigenvectors'];mu=p['mean']
 weights=np.ones(256)
 if tag=='drop2':weights[:2]=0
 elif tag=='white':weights=1/np.maximum(w,1)**.5
 elif tag=='pca64drop2':weights[:2]=0;weights[64:]=0
 elif tag=='pca32drop2':weights[:2]=0;weights[32:]=0
 elif tag=='raw':pass
 A={}
 for k in raw.files:
  x=gaussian_filter1d((raw[k]-mu)@v*weights,1.2,axis=0)
  A[k]=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
 return A

def interval_vector(a,s,e,n=8):
 # interpolate each normalized trajectory and retain temporal shape
 t=np.linspace(s,e-1,n);lo=np.floor(t).astype(int);hi=np.minimum(lo+1,len(a)-1);alpha=(t-lo)[:,None]
 x=a[lo]*(1-alpha)+a[hi]*alpha
 return (x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)).ravel()/np.sqrt(n)

def retrieve(tag):
 A=make_arrays(tag);cand=[];V=[]
 for c in meta['calibration']:
  a=A[c['id']]
  for length in range(4,19):
   for start in range(2,len(a)-length-1):
    cand.append((c['id'],start,start+length));V.append(interval_vector(a,start,start+length))
 V=np.asarray(V,dtype=np.float32)
 out={}
 for q in meta['queries']:
  a=A[q['id']];rows=[];vec=[]
  for length in [5,8,11,15]:
   for start in range(2,len(a)-length-1,3):
    rows.append((start,start+length));vec.append(interval_vector(a,start,start+length))
  sims=np.asarray(vec,dtype=np.float32)@V.T
  qr=[]
  for r,sim in zip(rows,sims):
   ix=np.argsort(sim)[-150:][::-1];seen=set();matches=[]
   for j in ix:
    cid,s,e=cand[j]
    if cid in seen:continue
    seen.add(cid);matches.append({'id':cid,'start':s,'end':e,'similarity':float(sim[j])})
    if len(matches)==10:break
   qr.append({'start':r[0],'end':r[1],'matches':matches})
  out[q['id']]=qr
 Path('local_'+tag+'.json').write_text(json.dumps(out))
 print('wrote',tag,flush=True)

if __name__=='__main__':
 import sys
 for tag in sys.argv[1:] or ['drop2','white','raw','pca32drop2']:
  retrieve(tag)
