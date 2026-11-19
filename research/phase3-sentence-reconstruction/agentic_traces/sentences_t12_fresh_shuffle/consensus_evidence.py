"""Pool observed calibration interval similarities to reduce exemplar noise.
This computes cosine to the normalized arithmetic mean of time-resampled
calibration exemplars; it fits no encoder or feature weights. All outputs here
are numerical-only, separate from the interpreted final result.
"""
from template_inference import *
if __name__=='__main__':
 A=make_arrays('drop2');x=json.load(open('calibration_interval_exemplars.json'));p=np.load('calibration_interval_vectors.npz');rv=p['vectors'];rw=p['word_indices'];ri=p['recording_indices'];dur=np.array([x['duration_frames'][w] for w in WORDS])
 mean=np.array([rv[rw==wi].mean(0) for wi in range(len(WORDS))]);mean/=np.linalg.norm(mean,axis=1,keepdims=True)
 out={};saved={}
 for q in meta['queries']:
  qid=q['id'];spans,vec=spans_vectors(A[qid]);scores=vec@mean.T
  seq,total=decode(spans,scores,len(A[qid]),dur,penalty=.38)
  out[qid]={'tokens':[x['token'] for x in seq],'intervals':seq,'total_score':total};saved[qid+'_spans']=spans;saved[qid+'_scores']=scores
 Path('consensus_algorithm_only.json').write_text(json.dumps(out,indent=2));np.savez_compressed('consensus_scores.npz',**saved)
 with open('consensus_evidence.txt','w') as f:
  for qid,r in out.items():
   print(qid,' '.join(r['tokens']),file=f)
   print(' '.join(f"{v['token']}:{v['start']}-{v['end']}({v['similarity']:.3f})" for v in r['intervals']),file=f)
