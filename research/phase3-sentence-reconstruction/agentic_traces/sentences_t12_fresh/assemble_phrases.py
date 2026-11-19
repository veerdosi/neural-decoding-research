"""Deterministic observed-phrase retrieval and semi-Markov assembly.
Extract every contiguous 1..5-token calibration interval under the saved
transcript-constrained alignment. Compare cosine trajectories resampled to
10 temporal points. Duration ratio penalizes extreme stretches. A beam DP
covers all query frames without a language model or given word counts.
"""
from local_retrieval import *
A=make_arrays('drop2');B=json.load(open('temporal_calibration_alignment.json'))['iterations'][-1];T=[]
for c in meta['calibration']:
 cid=c['id']
 if cid not in B:continue
 for i in range(len(c['tokens'])):
  for j in range(i+1,min(i+5,len(c['tokens']))+1):T.append({'id':cid,'tokens':c['tokens'][i:j],'start':B[cid][i],'end':B[cid][j]})
V=np.array([interval_vector(A[t['id']],t['start'],t['end'],10) for t in T]);lens=np.array([t['end']-t['start'] for t in T]);tlens=np.array([len(t['tokens']) for t in T]);out={}
for q in meta['queries']:
 qid=q['id'];a=A[qid];N=len(a)-2;spans=[(s,e) for s in range(2,N) for e in range(s+3,min(s+48,N)+1)];Q=np.array([interval_vector(a,s,e,10) for s,e in spans]);dur=np.array([e-s for s,e in spans]);sim=Q@V.T-.045*np.abs(np.log(dur[:,None]/lens[None,:]));sim[np.abs(np.log(dur[:,None]/lens[None,:]))>np.log(2.3)]=-1
 rows=[]
 for k,(s,e) in enumerate(spans):
  top=np.argsort(sim[k])[-12:][::-1];rows.append([{'exemplar':T[int(j)],'similarity':float(sim[k,j])} for j in top])
 dp={2:[(0,[],[])]}
 for e in range(5,N+1):
  candidates=[]
  for k,(s,ee) in enumerate(spans):
   if ee!=e or s not in dp:continue
   for r in rows[k][:8]:
    t=r['exemplar'];val=(e-s)*(r['similarity']-.10)-.22-.12*len(t['tokens'])
    for prev,words,path in dp[s]:candidates.append((prev+val,words+t['tokens'],path+[{'start':s,'end':e,**r}]))
  candidates.sort(key=lambda p:-p[0]);seen=set();dp[e]=[]
  for val,words,path in candidates:
   key=tuple(words)
   if key in seen:continue
   seen.add(key);dp[e].append((val,words,path))
   if len(dp[e])>=8:break
 out[qid]=[{'score':val,'tokens':words,'segments':path} for val,words,path in dp[N]]
 print(qid,' | '.join(' '.join(p['tokens']) for p in out[qid][:5]))
Path('phrase_algorithm_only_sequences.json').write_text(json.dumps({'method':__doc__,'queries':out},indent=2))
Path('phrase_exemplars.json').write_text(json.dumps(T,indent=2))
