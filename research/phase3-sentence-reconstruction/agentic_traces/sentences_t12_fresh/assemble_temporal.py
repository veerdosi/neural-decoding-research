"""Auditable, grammar-free interval-exemplar retrieval and sequence assembly.
Uses observed calibration word intervals from calibration_alignment.json.
Scores resampled cosine trajectory matches. Top two distinct recordings vote
per word; a weak log duration ratio penalty limits extreme time stretching.
Semi-Markov DP maximizes duration-weighted similarity minus fixed costs.
This automatic output is distinct from the final interpreted reconstruction.
"""
from local_retrieval import *
A=make_arrays('drop2');T=json.load(open('temporal_word_exemplars.json'));W=sorted({t['token'] for t in T});V=np.array([interval_vector(A[t['id']],t['start'],t['end'],6) for t in T]);lens=np.array([t['end']-t['start'] for t in T]);wi={w:[i for i,t in enumerate(T) if t['token']==w] for w in W};out={};details={}
for q in meta['queries']:
 qid=q['id'];a=A[qid];N=len(a)-2;spans=[(s,e) for s in range(2,N) for e in range(s+3,min(s+22,N)+1)];Q=np.array([interval_vector(a,s,e,6) for s,e in spans]);dur=np.array([e-s for s,e in spans]);sims=Q@V.T-.025*np.abs(np.log(dur[:,None]/lens[None,:]));score=np.zeros((len(spans),len(W)));best={}
 for j,w in enumerate(W):
  ix=wi[w];z=sims[:,ix];order=np.argsort(z,axis=1)[:,-2:];score[:,j]=np.take_along_axis(z,order,axis=1).mean(1);best[w]=np.array(ix)[order[:,-1]]
 # retain five best independently assembled complete paths, with no language prior.
 dp={2:[(0,[])]}
 for e in range(5,N+1):
  candidates=[]
  for k,(s,ee) in enumerate(spans):
   if ee!=e or s not in dp:continue
   for j in np.argsort(score[k])[-6:]:
    val=(e-s)*(score[k,j]-.10)-.40
    t=T[int(best[W[j]][k])]
    seg={'token':W[j],'start':s,'end':e,'similarity':float(score[k,j]),'exemplar':t}
    for prev,path in dp[s]:candidates.append((prev+val,path+[seg]))
  candidates.sort(key=lambda p:-p[0]);seen=set();dp[e]=[]
  for val,path in candidates:
   key=tuple(t['token'] for t in path)
   if key in seen:continue
   seen.add(key);dp[e].append((val,path))
   if len(dp[e])>=5:break
 out[qid]=[{'score':val,'tokens':[t['token'] for t in path],'segments':path} for val,path in dp[N]]
 np.savez_compressed('temporal_interval_scores_'+qid+'.npz',spans=np.array(spans),words=np.array(W),scores=score)
 print(qid,' | '.join(' '.join(p['tokens']) for p in out[qid]))
Path('temporal_algorithm_only_sequences.json').write_text(json.dumps({'method':__doc__,'queries':out},indent=2))
