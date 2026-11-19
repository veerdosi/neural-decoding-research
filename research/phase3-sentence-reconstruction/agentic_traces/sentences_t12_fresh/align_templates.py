"""Deterministic exemplar alignment, not a fitted neural-to-word model.
Initial calibration cuts use equal token intervals (a weak timing assumption).
Cross-recording cosine similarities refine calibration cuts by monotone DP.
Every word score remains traceable to observed, labeled calibration intervals.
"""
from local_retrieval import *
from scipy.special import logsumexp
A=make_arrays('drop2'); C=meta['calibration']; vocabulary=sorted({w for c in C for w in c['tokens']})
# Exclude recordings with unusually long pauses from boundary estimation.
usable=[c for c in C if len(A[c['id']])/len(c['tokens'])<12]
bounds={c['id']:np.rint(np.linspace(2,len(A[c['id']])-2,len(c['tokens'])+1)).astype(int).tolist() for c in usable}

def templates(bounds):
 return [(c['id'],w,j,bounds[c['id']][j],bounds[c['id']][j+1]) for c in usable for j,w in enumerate(c['tokens'])]

def align(E,words):
 n=len(E); L=len(words);D=np.full((L+1,n+1),-1e6);D[0,0]=0;P={}
 for j,w in enumerate(words):
  for end in range(1,n+1):
   for start in range(max(j,end-22),end-1):
    dur=end-start
    # Broad weak duration prior, especially prevents one frame function words.
    val=D[j,start]+E[start:end,j].sum()-.006*(dur-6)**2
    if val>D[j+1,end]:D[j+1,end]=val;P[j+1,end]=start
 out=[n];end=n
 for j in range(L,0,-1):end=P[j,end];out.append(end)
 return list(reversed(out))

history=[]
for iteration in range(3):
 T=templates(bounds); new={}
 for c in usable:
  cid=c['id'];x=A[cid][2:-2];E=[]
  for w in c['tokens']:
   scores=[]
   for other,ww,j,s,e in T:
    if ww!=w or other==cid:continue
    # Smooth frame affinity to each actual word exemplar, then top-3 consensus.
    z=x@A[other][s:e].T
    scores.append(z.max(1))
   if scores:
    z=np.sort(np.array(scores),axis=0);E.append(z[-min(3,len(z)):].mean(0))
   else:E.append(np.full(len(x),.05))
  cuts=align(np.array(E).T,c['tokens']);new[cid]=[v+2 for v in cuts]
 bounds=new;history.append(new)
Path('calibration_alignment.json').write_text(json.dumps({'description':__doc__,'iterations':history},indent=2))
# Save exemplars as observed intervals only. No neural transform is learned here.
Path('word_exemplars.json').write_text(json.dumps([{'id':cid,'token':w,'position':j,'start':s,'end':e} for cid,w,j,s,e in templates(bounds)],indent=2))
for c in usable:
 print(c['id'],' '.join(f'{w}:{bounds[c["id"]][j]}-{bounds[c["id"]][j+1]}' for j,w in enumerate(c['tokens'])))
