from exemplar_decode import *
import ctypes
lib=ctypes.CDLL('./interval_dtw.so');f=lib.all_dtw
f.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
A=make_arrays('drop2');T=json.load(open('refined_calibration_intervals.json'));words=sorted(set(t['word'] for t in T));out={};npout={}
for q in meta['queries']:
 qid=q['id'];a=A[qid];n=len(a);spans=[(s,e) for s in range(1,n-4) for e in range(s+4,min(n-1,s+40)+1)]
 TS=[]
 for t in T:
  b=A[t['id']][t['start']:t['end']];S=np.ascontiguousarray(a@b.T);res=np.full((n+1,n+1),-1.,dtype=np.float64);f(S,n,len(b),res)
  TS.append([res[s,e] for s,e in spans])
 TS=np.asarray(TS,dtype=np.float32).T
 W=np.full((len(spans),len(words)),-1.,np.float32);J=np.zeros_like(W,dtype=np.int32)
 for wi,w in enumerate(words):
  jj=[j for j,t in enumerate(T) if t['word']==w];vals=TS[:,jj];arg=vals.argmax(1);best=vals[np.arange(len(spans)),arg]
  second=np.sort(vals,axis=1)[:,-2] if len(jj)>1 else best
  W[:,wi]=.8*best+.2*second;J[:,wi]=np.asarray(jj)[arg]
 npout[qid+'_scores']=W;npout[qid+'_spans']=np.asarray(spans);npout[qid+'_template_index']=J
 D=np.full(n+1,-1e9);prev={}
 for s in range(1,5):D[s]=-.1*abs(s-2)
 for si,(s,e) in enumerate(spans):
  wi=int(W[si].argmax());score=float(W[si,wi]);val=D[s]+(score-.25)*(e-s)-.40
  if val>D[e]:D[e]=val;prev[e]=(s,si,wi)
 end=max(range(n-5,n),key=lambda e:D[e]-.12*abs(e-(n-3)))
 path=[];ee=end
 while ee in prev:
  s,si,wi=prev[ee];ti=int(J[si,wi]);alts=np.argsort(W[si])[-6:][::-1]
  path.append({'word':words[wi],'start':s,'end':ee,'similarity':float(W[si,wi]),'exemplar':T[ti],'alternatives':[{'word':words[j],'score':float(W[si,j]),'exemplar':T[J[si,j]]} for j in alts]});ee=s
 path=path[::-1];out[qid]={'tokens':[r['word'] for r in path],'segments':path,'objective':float(D[end])}
 print(qid,' '.join(out[qid]['tokens']),flush=True)
Path('refined_algorithm_only_predictions.json').write_text(json.dumps(out,indent=2));np.savez_compressed('refined_exemplar_scores.npz',**npout)
