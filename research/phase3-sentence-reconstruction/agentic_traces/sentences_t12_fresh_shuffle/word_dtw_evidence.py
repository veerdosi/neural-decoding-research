"""Slope-constrained DTW retrieval against observed calibration word intervals.
Unlike resampled-vector retrieval, this aligns every time bin while preserving
order. Only observed calibration intervals from the saved segmentation are used.
"""
from compare_candidates import *

def dtw_library(a,bs,refwords):
 K=len(bs);M=max(len(b) for b in bs);lens=np.array([len(b) for b in bs]);B=np.zeros((K,M,a.shape[1]))
 for k,b in enumerate(bs):B[k,:len(b)]=b
 S=np.einsum('nd,kmd->knm',a,B);N=len(a);D=np.full((K,N+1,M+1),-1e9);D[:,0,0]=0
 for i in range(1,N+1):
  v=D[:,i-1,:M]+S[:,i-1,:]
  if i>=2:v=np.maximum(v,D[:,i-2,:M]+S[:,i-2,:]+S[:,i-1,:]-.035)
  v[:,1:]=np.maximum(v[:,1:],D[:,i-1,:M-1]+.5*(S[:,i-1,:M-1]+S[:,i-1,1:])-.035)
  D[:,i,1:]=v
 result=D[np.arange(K)[:,None],np.arange(1,N+1)[None,:],lens[:,None]].T/np.arange(1,N+1)[:,None]
 result[result< -1]=-1
 scores=np.empty((N,len(WORDS)))
 for wi in range(len(WORDS)):
  z=result[:,refwords==wi];k=min(3,z.shape[1]);top=np.partition(z,-k,axis=1)[:,-k:]
  scores[:,wi]=.5*top.max(1)+.5*top.mean(1)
 return scores

if __name__=='__main__':
 A=make_arrays('drop2');x=json.load(open('calibration_interval_exemplars.json'));detail=x['intervals'];bs=[A[r['id']][r['start']:r['end']] for r in detail];rw=np.array([WI[r['word']] for r in detail]);dur=np.array([x['duration_frames'][w] for w in WORDS]);out={};cand={};save={}
 for q in meta['queries']:
  qid=q['id'];T=len(A[qid]);spans=[];scores=[]
  for s in range(1,T-4):
   end=min(s+30,T-2);sc=dtw_library(A[qid][s:end],bs,rw)
   for length in range(2,end-s+1):spans.append((s,s+length));scores.append(sc[length-1])
  spans=np.array(spans);scores=np.array(scores);save[qid+'_spans']=spans;save[qid+'_scores']=scores
  seq,total=decode(spans,scores,T,dur,penalty=.15);out[qid]={'tokens':[r['token'] for r in seq],'intervals':seq,'score':total}
  rows=[]
  for txt in CANDIDATES[qid]:
   score,ali=force(spans,scores,txt.split(),T,dur);rows.append({'tokens':txt.split(),'score':score,'alignment':ali})
  rows.sort(key=lambda r:-r['score']);cand[qid]=rows
 Path('word_dtw_algorithm_only.json').write_text(json.dumps(out,indent=2));Path('word_dtw_candidate_comparison.json').write_text(json.dumps(cand,indent=2));np.savez_compressed('word_dtw_scores.npz',**save)
 with open('word_dtw_evidence.txt','w') as f:
  for qid,r in out.items():
   print('\n###',qid,' '.join(r['tokens']),file=f)
   for row in cand[qid][:6]:print(round(row['score'],3),' '.join(row['tokens']),row['alignment'],file=f)
 print('Saved word DTW evidence')
