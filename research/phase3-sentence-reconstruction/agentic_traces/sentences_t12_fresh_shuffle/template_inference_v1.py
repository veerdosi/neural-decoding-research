"""Deterministic exemplar retrieval and transcript-constrained calibration segmentation.
No feature weights, encoder, classifier, or language model is fitted.
Duration statistics use only the labeled calibration. All neural scoring is cosine
between resampled observed intervals. Iterations refine calibration boundaries;
query decoding uses the fixed calibration interval library, with no grammar.
"""
from local_retrieval import *
from scipy.optimize import least_squares

WORDS=sorted({w for c in meta['calibration'] for w in c['tokens']})
WI={w:i for i,w in enumerate(WORDS)}

def spans_vectors(a,maxlen=30,n=5):
 spans=[(s,e) for s in range(1,len(a)-3) for e in range(s+2,min(s+maxlen,len(a)-2)+1)]
 vec=np.array([interval_vector(a,s,e,n) for s,e in spans],dtype=np.float32)
 return np.array(spans),vec

def score_refs(vec,refvec,refwords,refids,exclude=None):
 sims=vec@refvec.T
 if exclude is not None:sims[:,refids==exclude]=-1
 scores=np.empty((len(vec),len(WORDS)),np.float32)
 for wi,w in enumerate(WORDS):
  ix=np.where(refwords==wi)[0]
  if len(ix)==0:scores[:,wi]=-1;continue
  z=sims[:,ix];k=min(3,len(ix))
  top=np.partition(z,-k,axis=1)[:,-k:]
  scores[:,wi]=.5*top.max(1)+.5*top.mean(1)
 return scores

def fixed_align(spans,scores,tokens,T,durations,scale):
 D=np.full((len(tokens)+1,T+1),-1e9);D[0,2]=0
 P=np.full((len(tokens)+1,T+1),-1,int)
 s=spans[:,0];e=spans[:,1];length=e-s
 for i,w in enumerate(tokens):
  expected=durations[WI[w]]*scale
  v=length*scores[:,WI[w]]-.13*expected*(np.log(length/expected)**2)
  vals=D[i,s]+v
  for ix in np.argsort(vals):
   if vals[ix]>D[i+1,e[ix]]:D[i+1,e[ix]]=vals[ix];P[i+1,e[ix]]=s[ix]
 end=T-3;bounds=[end]
 for i in range(len(tokens),0,-1):
  end=int(P[i,end]);bounds.append(end)
 return list(reversed(bounds))

def decode(spans,scores,T,durations,penalty=.38):
 s=spans[:,0];e=spans[:,1];length=e-s
 # No transcript, bigram statistics, sentence candidates, or query counts.
 reward=length[:,None]*scores-.13*durations[None,:]*(np.log(length[:,None]/durations[None,:])**2)-penalty
 wi=np.argmax(reward,axis=1);best=reward[np.arange(len(spans)),wi]
 D=np.full(T+1,-1e9);D[2]=0;P=[None]*(T+1)
 for end in range(4,T-1):
  ix=np.where(e==end)[0]
  if not len(ix):continue
  val=D[s[ix]]+best[ix];k=ix[np.argmax(val)]
  D[end]=D[s[k]]+best[k];P[end]=(int(s[k]),int(wi[k]),float(scores[k,wi[k]]))
 end=T-3;out=[]
 while end>2:
  p=P[end]
  if p is None:break
  start,w,sc=p;out.append({'token':WORDS[w],'start':start,'end':end,'similarity':sc});end=start
 return out[::-1],float(D[T-3])

if __name__=='__main__':
 A=make_arrays('drop2');cs=meta['calibration'];X=np.zeros((len(cs),len(WORDS)+1));y=[]
 for k,c in enumerate(cs):
  X[k,0]=1
  for w in c['tokens']:X[k,WI[w]+1]+=1
  y.append(len(A[c['id']]))
 fit=least_squares(lambda d:X@d-y,np.r_[5,np.full(len(WORDS),5.)],bounds=(np.r_[0,np.full(len(WORDS),1.5)],np.r_[12,np.full(len(WORDS),20.)]),loss='soft_l1',f_scale=2)
 durations=fit.x[1:];bounds={};scales={};cache={}
 for c in cs:
  cid=c['id'];T=len(A[cid]);ds=np.array([durations[WI[w]] for w in c['tokens']]);scale=(T-5)/ds.sum();scales[cid]=scale
  bounds[cid]=[2]+np.rint(2+np.cumsum(ds)*scale).astype(int).tolist()
  cache[cid]=spans_vectors(A[cid],maxlen=min(40,max(24,int(20*scale))))
 def references(bounds):
  rv=[];rw=[];ri=[];detail=[]
  for ci,c in enumerate(cs):
   for j,w in enumerate(c['tokens']):
    s,e=bounds[c['id']][j:j+2]
    if e<=s:continue
    rv.append(interval_vector(A[c['id']],s,e,5));rw.append(WI[w]);ri.append(ci)
    detail.append({'id':c['id'],'word':w,'start':int(s),'end':int(e)})
  return np.asarray(rv,dtype=np.float32),np.array(rw),np.array(ri),detail
 history=[bounds]
 for iteration in range(3):
  rv,rw,ri,detail=references(bounds);new={}
  for ci,c in enumerate(cs):
   spans,vec=cache[c['id']];scores=score_refs(vec,rv,rw,ri,ci)
   new[c['id']]=fixed_align(spans,scores,c['tokens'],len(A[c['id']]),durations,scales[c['id']])
  bounds=new;history.append(bounds);print('calibration segmentation',iteration+1,flush=True)
 rv,rw,ri,detail=references(bounds)
 Path('calibration_interval_exemplars.json').write_text(json.dumps({'duration_frames':dict(zip(WORDS,durations.tolist())),'boundaries_history':history,'intervals':detail},indent=2))
 np.savez('calibration_interval_vectors.npz',vectors=rv,word_indices=rw,recording_indices=ri)
 out={};allscore={}
 for q in meta['queries']:
  qid=q['id'];spans,vec=spans_vectors(A[qid]);scores=score_refs(vec,rv,rw,ri)
  seq,total=decode(spans,scores,len(A[qid]),durations)
  out[qid]={'tokens':[x['token'] for x in seq],'intervals':seq,'total_score':total}
  allscore[qid+'_spans']=spans;allscore[qid+'_scores']=scores
 Path('algorithm_only_predictions.json').write_text(json.dumps(out,indent=2))
 np.savez_compressed('query_interval_scores.npz',**allscore)
 with open('template_evidence.txt','w') as f:
  for qid,v in out.items():
   print(qid,' '.join(v['tokens']),file=f)
   for x in v['intervals']:print(' ',x,file=f)
 print('Saved algorithm-only reconstructions and evidence')
