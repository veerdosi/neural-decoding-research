"""Transcript-constrained cross-recording alignment using temporal exemplars.
All interval vectors remain observed calibration samples; no fitted encoder,
classifier, word vectors, or language model. Three deterministic passes refine
unknown calibration cuts, maximizing top-two independent exemplar cosine.
"""
from local_retrieval import *
A=make_arrays('drop2');C=[c for c in meta['calibration'] if len(A[c['id']])/len(c['tokens'])<12];bounds={c['id']:np.rint(np.linspace(2,len(A[c['id']])-2,len(c['tokens'])+1)).astype(int).tolist() for c in C}
history=[]
for iteration in range(3):
 T=[{'id':c['id'],'token':w,'position':j,'start':bounds[c['id']][j],'end':bounds[c['id']][j+1]} for c in C for j,w in enumerate(c['tokens'])];V=np.array([interval_vector(A[t['id']],t['start'],t['end'],6) for t in T]);new={}
 for c in C:
  cid=c['id'];a=A[cid];N=len(a)-2;L=len(c['tokens']);spans=[(s,e) for s in range(2,N) for e in range(s+2,min(s+20,N)+1)];Q=np.array([interval_vector(a,s,e,6) for s,e in spans]);sim=Q@V.T;E=np.full((len(spans),L),-.2)
  for j,w in enumerate(c['tokens']):
   ix=[k for k,t in enumerate(T) if t['token']==w and t['id']!=cid]
   if not ix:continue
   z=np.sort(sim[:,ix],axis=1);E[:,j]=z[:,-min(2,len(ix)):].mean(1)
  D=np.full((L+1,N+1),-1e6);D[0,2]=0;P={}
  for j in range(L):
   for k,(s,e) in enumerate(spans):
    d=e-s;val=D[j,s]+d*E[k,j]-.006*(d-6)**2
    if val>D[j+1,e]:D[j+1,e]=val;P[j+1,e]=s
  b=[N];e=N
  for j in range(L,0,-1):e=P[j,e];b.append(e)
  new[cid]=list(reversed(b))
 bounds=new;history.append(new)
Path('temporal_calibration_alignment.json').write_text(json.dumps({'method':__doc__,'iterations':history},indent=2));Path('temporal_word_exemplars.json').write_text(json.dumps([{'id':c['id'],'token':w,'position':j,'start':bounds[c['id']][j],'end':bounds[c['id']][j+1]} for c in C for j,w in enumerate(c['tokens'])],indent=2))
for c in C:print(c['id'],' '.join(f'{w}:{bounds[c["id"]][j]}-{bounds[c["id"]][j+1]}' for j,w in enumerate(c['tokens'])))
