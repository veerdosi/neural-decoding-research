from fast_helpers import *
import ctypes
lib.interval_scores.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_void_p]
A=make_arrays('drop2');ex=json.load(open('exemplars.json'));words=sorted({t['word'] for t in ex});wi={w:i for i,w in enumerate(words)};preds=[];evidence={}
for q in meta['queries']:
 qid=q['id'];a=A[qid];n=len(a);best=np.full((n+1,n+1,len(words)),-1e9);which=np.zeros(best.shape,int)
 for ei,t in enumerate(ex):
  b=A[t['id']][t['start']:t['end']];S=np.ascontiguousarray(a@b.T);out=np.full((n+1,n+1),-1e9)
  lib.interval_scores(S.ctypes.data,n,len(b),5,26,.035,out.ctypes.data)
  j=wi[t['word']];mask=out>best[:,:,j];best[:,:,j][mask]=out[mask];which[:,:,j][mask]=ei
 dp=np.full(n+1,-1e9);dp[1:6]=0;back={}
 for e in range(6,n):
  for s in range(max(1,e-26),e-4):
   j=int(np.argmax(best[s,e]));value=dp[s]+(best[s,e,j]-.21)*(e-s)-.52
   if value>dp[e]:dp[e]=value;back[e]=(s,j)
 end=max(range(n-6,n),key=lambda e:dp[e]);end0=end;path=[]
 while end in back:
  s,j=back[end];path.append(dict(word=words[j],start=s,end=end,similarity=float(best[s,end,j]),exemplar=ex[which[s,end,j]]));end=s
 path.reverse();preds.append(dict(id=qid,tokens=[t['word'] for t in path]));evidence[qid]=path
 np.savez_compressed('dtw_lattice_'+qid+'.npz',scores=best,which=which,words=np.array(words))
Path('algorithm_only_dtw_predictions.json').write_text(json.dumps(preds,indent=2));Path('algorithm_only_dtw_alignment.json').write_text(json.dumps(evidence,indent=2))
for p in preds:print(p['id'],' '.join(p['tokens']))
