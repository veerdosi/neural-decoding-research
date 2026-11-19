"""Auditable exemplar search, not a fitted neural encoder.
Calibration word boundaries are latent; initialize from robust transcript-duration
regression, then refine via leave-recording-out nearest observed intervals.
Each comparison is cosine of eight interpolated smoothed PCA trajectory samples.
No query transcript, external text model, or query-derived word associations.
"""
from local_retrieval import *
from scipy.optimize import least_squares
import sys
A=make_arrays('drop2');cal=meta['calibration'];words=sorted({w for c in cal for w in c['tokens']});wi={w:i for i,w in enumerate(words)}
M=np.zeros((len(cal),len(words)))
for i,c in enumerate(cal):
 for w in c['tokens']:M[i,wi[w]]+=1
lens=np.array([len(A[c['id']])-4 for c in cal])
# Robust durations, weak regularization toward six bins, fixed four-bin margins.
fit=least_squares(lambda d:np.r_[(M@d-lens),.9*(d-6)],np.full(len(words),6.),bounds=(2,18),loss='soft_l1',f_scale=2.)
dur=dict(zip(words,fit.x));Path('duration_priors.json').write_text(json.dumps(dur,indent=2))
# candidate segment vectors indexed by recording and start/end
candidates={};vectors={};span_index={}
for c in cal+meta['queries']:
 cid=c['id'];a=A[cid];spans=[(s,e) for s in range(1,len(a)-3) for e in range(s+3,min(s+23,len(a)-1)+1)]
 candidates[cid]=np.array(spans);vectors[cid]=np.array([interval_vector(a,s,e) for s,e in spans],np.float32);span_index[cid]={tuple(p):i for i,p in enumerate(spans)}
boundaries={}
for c in cal:
 ds=np.array([dur[w] for w in c['tokens']]);b=np.r_[2,2+np.cumsum(ds)/sum(ds)*(len(A[c['id']])-4)]
 boundaries[c['id']]=np.round(b).astype(int)

def get_templates(bounds):
 ts=[];labels=[];sources=[]
 for c in cal:
  cid=c['id'];bs=bounds[cid]
  for j,w in enumerate(c['tokens']):
   s,e=bs[j:j+2]
   if e-s<3:continue
   ts.append(interval_vector(A[cid],s,e));labels.append(wi[w]);sources.append((cid,j,int(s),int(e)))
 return np.asarray(ts,np.float32),np.array(labels),sources

def score_intervals(cid,T,L,S):
 # Top three independent calibration exemplars per word; self recording excluded.
 sim=vectors[cid]@T.T
 scores=np.full((len(sim),len(words)),-1.,np.float32)
 for k,w in enumerate(words):
  ix=[j for j in range(len(S)) if L[j]==k and S[j][0]!=cid]
  if not ix:continue
  sub=sim[:,ix];n=min(3,len(ix));scores[:,k]=np.sort(sub,axis=1)[:,-n:].mean(1)
 return scores

def force_align(cid,tokens,scores):
 spans=candidates[cid];idx=span_index[cid];n=len(A[cid]);K=len(tokens)
 D=np.full((K+1,n+1),-1e6);D[0,2]=0;P={}
 # Durations already scaled for unusual speech rates within each sentence.
 rate=(n-4)/sum(dur[w] for w in tokens)
 for k,w in enumerate(tokens):
  ds=dur[w]*rate
  for s,e in spans:
   if D[k,s]<-1e5:continue
   score=(scores[idx[s,e],wi[w]]-.01*(np.log((e-s)/ds))**2)*(e-s)
   v=D[k,s]+score
   if v>D[k+1,e]:D[k+1,e]=v;P[k+1,e]=s
 end=n-2
 if (K,end) not in P:return boundaries.get(cid)
 b=[end]
 for k in range(K,0,-1):b.append(P[k,b[-1]])
 return np.array(b[::-1])

def decode(cid,scores,nbest=8):
 spans=candidates[cid];idx=span_index[cid];n=len(A[cid]);D=[[] for _ in range(n+1)];D[2]=[(0.,[])]
 for s in range(2,n-3):
  if not D[s]:continue
  for e in range(s+3,min(s+23,n-2)+1):
   if (s,e) not in idx:continue
   sc=scores[idx[s,e]].astype(float)
   # Duration likelihood with broad log-normal support; no language model.
   ds=np.array([dur[w] for w in words]);sc-=.025*np.log((e-s)/ds)**2
   # Per-word cost discourages spurious 3-bin fragments.
   vals=sc*(e-s)-.22
   for k in np.argsort(vals)[-6:]:
    for total,path in D[s]:D[e].append((total+vals[k],path+[(words[k],s,e,float(scores[idx[s,e],k]))]))
   D[e]=sorted(D[e],key=lambda v:-v[0])[:nbest]
 return sorted(D[n-2],key=lambda v:-v[0])[:nbest]

if __name__=='__main__':
 for it in range(4):
  T,L,S=get_templates(boundaries);new={}
  for c in cal:
   cid=c['id'];sc=score_intervals(cid,T,L,S);new[cid]=force_align(cid,c['tokens'],sc)
  boundaries=new
  Path(f'calibration_boundaries_{it}.json').write_text(json.dumps({cid:b.tolist() for cid,b in boundaries.items()},indent=2))
  print('boundary refinement',it,flush=True)
 T,L,S=get_templates(boundaries);out={}
 for q in meta['queries']:
  cid=q['id'];sc=score_intervals(cid,T,L,S);np.savez_compressed('scores_'+cid+'.npz',spans=candidates[cid],scores=sc,words=words)
  paths=decode(cid,sc);out[cid]=[dict(score=v,tokens=[p[0] for p in ps],intervals=ps) for v,ps in paths]
  print(cid,' | '.join(' '.join(p[0] for p in ps) for v,ps in paths[:3]),flush=True)
 Path('algorithm_only_sequences.json').write_text(json.dumps(out,indent=2))
 Path('exemplar_sources.json').write_text(json.dumps(S,indent=2))
