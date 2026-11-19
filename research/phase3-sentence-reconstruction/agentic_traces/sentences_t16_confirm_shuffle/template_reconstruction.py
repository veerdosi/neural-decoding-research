"""Auditable interval-template retrieval; no fitted neural-language weights.
Boundaries below are analyst estimates from overlapping calibration retrieval.
Each hypothesis is scored by temporal cosine to individual calibration intervals.
Unrestricted vocabulary segmentation uses no sentence list or language model.
"""
from local_retrieval import *
# One boundary per word plus final endpoint; bins are 100 ms nominal.
B={
'c000':[2,17,24,36,45,54,69],
'c001':[2,8,17,29],
'c002':[2,17,27,35,47,64],
'c003':[2,25,36,45,54],
'c004':[2,13,23,35],
'c005':[2,18,29,38,48,57,68],
'c006':[2,19,28,36,47,58],
'c007':[2,16,28,37,44],
'c008':[2,12,21,27,36,43,55],
'c009':[2,17,30,39,49,59],
'c010':[2,17,26,43],
'c011':[2,24,33,44,55],
'c012':[2,14,28,40,45,52,65],
'c013':[2,17,25,36,53],
'c014':[2,17,24,32,39],
'c015':[2,14,25,33,45,59],
'c016':[2,20,31,46],
'c017':[2,14,25,33],
'c018':[2,14,20,32],
'c019':[2,13,26,31,37,47],
'c021':[2,19,34,49],
'c022':[2,24,33,43,56],
'c023':[2,18,26,33,44],
'c024':[2,16,25,33,41,49,57],
'c025':[2,24,40],
'c026':[2,17,31],
'c027':[2,15,28,38,45,54],
'c028':[2,13,22,29,38,45,56],
'c029':[2,15,22,32,40,48,57,66,81],
'c030':[2,21,29,36,48],
'c031':[2,13,21,27,36],
'c032':[2,26,36,45],
'c033':[2,22,29,35,48],
'c034':[2,15,23,30,41],
'c035':[2,15,24,31,39,47,56,64,77],
'c036':[2,18,30,39,50],
'c038':[2,23,30,35,47],
'c039':[2,19,27,35,43,50,64],
'c040':[2,20,27,35,47],
'c041':[2,19,25,33,41],
'c042':[2,13,21,33],
'c043':[2,13,20,32,38,46,55,63],
'c044':[2,29,43,49,57,64,72],
'c045':[2,13,21,26,36],
'c046':[2,9,18,25,34,44,54],
'c047':[2,26,33,40,54],
'c048':[2,24,29,33,41,50,59],
'c049':[2,25,38],
'c050':[2,16,28,38,46,53,61,67,83],
'c051':[2,14,22,32],
'c052':[2,23,33,40,51],
'c053':[2,22,34,44],
'c054':[2,23,32,39,48,54,66],
'c055':[2,15,25,34,41,55],
'c056':[2,24,31,39,47,60],
'c057':[2,13,20,27,36],
'c058':[2,14,23,32,40,48,62],
'c059':[2,24,30,41],
'c060':[2,14,19,23,30,43],
'c061':[2,15,24,36,45,62],
'c062':[2,10,19,26,33,47],
'c063':[2,26,35,47,59,70],
'c064':[2,15,25,39],
'c065':[2,23,29,46],
'c066':[2,14,21,29,43],
'c067':[2,17,27,34,42,57],
'c068':[2,17,30,39,47,61],
'c069':[2,28,39,54],
'c070':[2,16,27],
'c071':[2,14,23,31,41,51,57,76],
'c072':[2,24,34,48],
'c073':[2,15,30],
'c074':[2,15,23,30,42],
'c075':[2,17,26,39],
'c076':[2,11,23,37,47,58,73],
'c077':[2,26,33,41,53,63],
'c078':[2,18,30,39,55,63,71,81],
'c079':[2,18,24,31,46],
}

def run():
 A=make_arrays('drop2');T=[];V=[]
 for c in meta['calibration']:
  cid=c['id']
  if cid not in B:continue
  bounds=B[cid];assert len(bounds)==len(c['tokens'])+1,cid
  for i,w in enumerate(c['tokens']):
   s,e=bounds[i:i+2]
   T.append(dict(word=w,id=cid,start=s,end=e,position=i))
   V.append(interval_vector(A[cid],s,e,n=12))
 V=np.array(V,dtype=np.float32);out={};pred=[]
 for q in meta['queries']:
  qid=q['id'];a=A[qid];n=len(a);spans=[];Q=[]
  for s in range(1,n-4):
   for e in range(s+4,min(s+30,n-1)+1):
    spans.append((s,e));Q.append(interval_vector(a,s,e,n=12))
  S=np.array(Q,dtype=np.float32)@V.T
  # Moderate time stretch; no word/transition priors. Penalize each new word
  # to prevent assembling tiny high-similarity fragments.
  scores={};matches={}
  for k,(s,e) in enumerate(spans):
   valid=np.array([.45 <= (e-s)/(t['end']-t['start']) <= 2.1 for t in T])
   sim=S[k].copy();sim[~valid]=-100
   # Trial onset templates contain generic onset activity, so require onset
   # position consistency; later words use only internal-word templates.
   if s<=3: sim[np.array([t['position']!=0 for t in T])]= -100
   else: sim[np.array([t['position']==0 for t in T])]= -100
   ix=np.argsort(sim)[-8:][::-1]
   matches[s,e]=[dict(T[j],similarity=float(sim[j])) for j in ix]
   scores[s,e]=float(sim[ix[0]])*(e-s)-.8
  dp=np.full(n+1,-1e9);prev={};dp[2]=0
  for e in range(6,n):
   for s in range(max(2,e-30),e-3):
    if (s,e) not in scores:continue
    score=dp[s]+scores[s,e]
    if score>dp[e]:dp[e]=score;prev[e]=s
  end=max(range(max(6,n-5),n),key=lambda e:dp[e]/(e-2))
  segments=[];e=end
  while e in prev:
   s=prev[e];segments.append(dict(start=s,end=e,matches=matches[s,e]));e=s
  segments=segments[::-1]
  out[qid]=dict(score=float(dp[end]),segments=segments,
    interval_candidates=[dict(start=s,end=e,matches=matches[s,e][:4]) for s,e in spans])
  tok=[seg['matches'][0]['word'] for seg in segments]
  pred.append(dict(id=qid,tokens=tok))
  print(qid,' '.join(tok))
 Path('algorithm_only_predictions.json').write_text(json.dumps(pred,indent=2))
 Path('template_evidence.json').write_text(json.dumps(out))
 Path('calibration_intervals.json').write_text(json.dumps({cid:[dict(token=w,start=s,end=e) for w,s,e in zip(meta['calibration'][int(cid[1:])]['tokens'],b,b[1:])] for cid,b in B.items()},indent=2))

if __name__=='__main__':run()
