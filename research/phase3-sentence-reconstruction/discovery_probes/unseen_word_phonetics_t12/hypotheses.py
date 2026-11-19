import json,numpy as np
A=np.load('candidate_scores.npz');G=np.load('generative_scores.npz');S=A['scores'];words=A['words'];Q=np.load('analysis_data.npz');qids=list(Q['qids'])
hyp='number hungry mother father brother sister family water wanted outside inside upset thirsty hunger bathroom computer television music drink drinking food pain happy sad sorry thank thanks love stop help open close again today tomorrow yesterday blanket doctor nurse friend paper window door garden dinner lunch breakfast hot cold warmer colder comfortable uncomfortable bed tired sleepy sleep awake blanket pillow children child phone telephone please cough mouth goodbye question answer hospital medicine emergency goodnight'.split()
# Per-query ranks in broad dictionary, not a restricted candidate contest.
out={}
for j,qid in enumerate(qids):
 r=[]
 for w in hyp:
  ii=np.flatnonzero(words==w)
  if not len(ii):continue
  rs=[int(1+(s[:,j]<s[ii,j].min()-1e-8).sum()) for s in S]
  gi=np.flatnonzero(G['words']==w);gr=int(1+(G['scores'][:,j]<G['scores'][gi,j].min()-1e-8).sum()) if len(gi) else 999999
  r.append(dict(word=w,ranks=rs,gesture_rank=gr,aggregate=float(np.mean(np.log(np.array([rs[0],rs[4],gr])+1)))))
 r.sort(key=lambda r:r['aggregate']);out[qid]=r
 print(qid,[(x['word'],x['ranks'][0],x['ranks'][4],x['gesture_rank']) for x in r[:12]])
json.dump(out,open('human_hypothesis_ranks.json','w'),indent=2)
