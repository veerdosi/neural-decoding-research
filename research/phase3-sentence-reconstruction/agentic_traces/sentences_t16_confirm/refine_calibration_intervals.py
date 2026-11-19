"""One leave-one-recording-out forced-alignment pass on calibration only.
Known calibration transcript order is fixed. Each boundary may move at most
four rows from the documented overlap-based estimates. Original neural
exemplars are retained; no neural feature weights or encoder are trained.
"""
from exemplar_decode import *
import ctypes
lib=ctypes.CDLL('./interval_dtw.so');f=lib.all_dtw
f.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
A=make_arrays('drop2');T=templates(A);newT=[];evidence={}
for ci,bounds in B.items():
 c=meta['calibration'][ci];a=A[c['id']];n=len(a);tokens=c['tokens'];scoremap={}
 for w in set(tokens):
  ts=[t for t in T if t['word']==w and t['id']!=c['id']];ss=[]
  for t in ts:
   b=A[t['id']][t['start']:t['end']];S=np.ascontiguousarray(a@b.T);res=np.full((n+1,n+1),-1.,dtype=np.float64);f(S,n,len(b),res);ss.append(res)
  if ss:
   stack=np.sort(np.asarray(ss),axis=0);scoremap[w]=.8*stack[-1]+.2*stack[-2] if len(ss)>1 else stack[-1]
 D={bounds[0]:0.};prev={}
 for k,w in enumerate(tokens):
  nextD={};ends=range(max(bounds[k+1]-4,bounds[k]+1),min(n-1,bounds[k+1]+4)+1) if k<len(tokens)-1 else [bounds[-1]]
  for s,prevscore in D.items():
   for e in ends:
    if e-s<3:continue
    if w not in scoremap:
     if s!=bounds[k] or e!=bounds[k+1]:continue
     sc=.4
    else:sc=scoremap[w][s,e]
    val=prevscore+float(sc)*(e-s)-.015*((e-bounds[k+1])**2)
    if val>nextD.get(e,-1e9):nextD[e]=val;prev[k,e]=(s,float(sc))
  D=nextD
 if bounds[-1] not in D:
  for k,w in enumerate(tokens):newT.append({'word':w,'id':c['id'],'start':bounds[k],'end':bounds[k+1]})
  continue
 e=bounds[-1];path=[]
 for k in range(len(tokens)-1,-1,-1):
  s,sc=prev[k,e];path.append({'word':tokens[k],'id':c['id'],'start':s,'end':e,'similarity_loo':sc});e=s
 path.reverse();newT.extend([{k:v for k,v in t.items() if k!='similarity_loo'} for t in path]);evidence[c['id']]={'old_boundaries':bounds,'new_segments':path}
Path('refined_calibration_intervals.json').write_text(json.dumps(newT,indent=2));Path('calibration_refinement_evidence.json').write_text(json.dumps(evidence,indent=2))
print('Saved one calibration-only alignment refinement pass.')
