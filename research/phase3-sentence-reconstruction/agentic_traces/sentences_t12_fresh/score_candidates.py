"""Forced candidate comparisons over observed calibration phrase exemplars.
Candidates in interpreted_candidates.json are human evidence-based alternatives.
This scores them with cosine interpolation only; it does not establish identity.
No grammatical score, given word count, or query labels enter the calculation.
"""
from local_retrieval import *
A=make_arrays('drop2');T=json.load(open('phrase_exemplars.json'));V=np.array([interval_vector(A[t['id']],t['start'],t['end'],10) for t in T]);lens=np.array([t['end']-t['start'] for t in T]);phrases={}
for j,t in enumerate(T):phrases.setdefault(tuple(t['tokens']),[]).append(j)
Q=json.load(open('interpreted_candidates.json'));out={}
for qid,candidates in Q.items():
 a=A[qid];N=len(a)-2;spans=[(s,e) for s in range(2,N) for e in range(s+2,min(s+48,N)+1)];Z=np.array([interval_vector(a,s,e,10) for s,e in spans]);dur=np.array([e-s for s,e in spans]);sim=Z@V.T-.035*np.abs(np.log(dur[:,None]/lens[None,:]));sim[np.abs(np.log(dur[:,None]/lens[None,:]))>np.log(2.5)]=-1;results=[]
 for candidate in candidates:
  words=candidate.split();L=len(words);D=np.full((L+1,N+1),-1e6);D[0,2]=0;P={}
  for j in range(L):
   for h in range(j+1,min(j+4,L)+1):
    ix=phrases.get(tuple(words[j:h]),[])
    if not ix:continue
    best=np.argmax(sim[:,ix],axis=1);sc=sim[np.arange(len(spans)),np.array(ix)[best]]
    for k,(s,e) in enumerate(spans):
     val=D[j,s]+(e-s)*(sc[k]-.10)-.10*(h-j)-.15
     if val>D[h,e]:D[h,e]=val;P[h,e]=(j,s,k,int(ix[best[k]]))
  if D[L,N]<-1e5:continue
  path=[];j=L;e=N
  while j:
   p,s,k,t=P[j,e];path.append({'start':s,'end':e,'tokens':words[p:j],'similarity':float(sim[k,t]),'exemplar':T[t]});j=p;e=s
  results.append({'candidate':candidate,'score':float(D[L,N]),'path':path[::-1]})
 results.sort(key=lambda r:-r['score']);out[qid]=results
 print(qid,' | '.join(f"{r['score']:.2f} {r['candidate']}" for r in results[:6]))
Path('interpreted_candidate_scores.json').write_text(json.dumps({'method':__doc__,'queries':out},indent=2))
