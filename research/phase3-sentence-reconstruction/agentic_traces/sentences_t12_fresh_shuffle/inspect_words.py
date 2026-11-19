from template_inference import *
import sys
A=make_arrays('drop2');p=np.load('calibration_interval_vectors.npz');rv=p['vectors'];rw=p['word_indices'];ri=p['recording_indices'];ex=json.load(open('calibration_interval_exemplars.json'))['intervals'];sc=np.load('query_interval_scores.npz')
qs=sys.argv[1:] or [q['id'] for q in meta['queries']]
for qid in qs:
 print('\n###',qid)
 spans=sc[qid+'_spans'];scores=sc[qid+'_scores']
 for s in range(2,len(A[qid])-5,3):
  length=min(7,len(A[qid])-3-s);e=s+length
  v=interval_vector(A[qid],s,e,5);sim=rv@v
  ix=np.argsort(sim)[::-1];best=[];seen=set()
  for k in ix:
   if WORDS[rw[k]] in seen:continue
   seen.add(WORDS[rw[k]]);r=ex[k];best.append(f"{r['word']} {sim[k]:.3f} {r['id']}:{r['start']}-{r['end']}")
   if len(best)==6:break
  si=np.where((spans[:,0]==s)&(spans[:,1]==e))[0][0];ws=np.argsort(scores[si])[-6:][::-1]
  print(f'{s:2}-{e:2}', ' | '.join(best))
  print(' vote',' '.join(f'{WORDS[w]}:{scores[si,w]:.3f}' for w in ws))
