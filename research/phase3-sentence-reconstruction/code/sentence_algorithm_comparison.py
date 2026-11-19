"""Matched neural retrieval without an LLM. All tuning uses calibration blocks.

Actions calibrate -> predict -> evaluate are separate: only evaluate loads query
sentenceText. All generated files live outside Sol's directories. No model calls.
"""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,ctypes,hashlib,itertools,json,pathlib,re,subprocess,time
from collections import defaultdict
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits

ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
SANDBOX=ROOT/'agentic_development_trace'
OUT=ROOT/'evaluation/algorithm_sentence_baselines'
OUT.mkdir(exist_ok=True)
SOURCE=PROJECT/'local/sources/t12.2023.11.28_attempted_50words.mat'

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def distance(a,b):
 d=np.arange(len(b)+1)
 for i,x in enumerate(a):
  row=np.empty(len(b)+1,int);row[0]=i+1
  for j,y in enumerate(b):row[j+1]=min(row[j]+1,d[j+1]+1,d[j]+(x!=y))
  d=row
 return int(d[-1])

def inputs():
 meta=read(SANDBOX/'inputs.json');z=np.load(SANDBOX/'neural_inputs.npz')
 return meta,{k:z[k] for k in z.files}

class DTW:
 def __init__(self):
  lib=OUT/'sentence_dtw.dylib';src=ROOT/'code/sentence_dtw.c'
  if not lib.exists() or lib.stat().st_mtime<src.stat().st_mtime:
   subprocess.run(['clang','-O3','-std=c99','-dynamiclib',str(src),'-o',str(lib)],check=True)
  self.fn=ctypes.CDLL(str(lib)).subsequence_dtw
  self.fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
  self.fn.restype=ctypes.c_double
 def __call__(self,s):
  s=np.ascontiguousarray(s,np.float64);n,m=s.shape;p=np.empty((2*(n+m),2),np.int32);length=ctypes.c_int();start=ctypes.c_int();end=ctypes.c_int()
  score=self.fn(s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),n,m,p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ctypes.byref(length),ctypes.byref(start),ctypes.byref(end))
  return float(score),p[:length.value].copy(),start.value,end.value

def arrays(raw,calids):
 # Exact frozen development drop2 representation, fitted on fold calibration.
 x=np.concatenate([raw[k] for k in calids]);mu=x.mean(0);w,v=np.linalg.eigh(np.cov(x.T));v=v[:,::-1]
 weights=np.ones(x.shape[1]);weights[:2]=0
 ans={}
 for k,a in raw.items():
  a=gaussian_filter1d((a-mu)@v*weights,1.2,axis=0)
  ans[k]=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-9)
 return ans

def retrieval(a,calids,queryids,dtw):
 out={}
 for qi,qid in enumerate(queryids):
  rows=[];sims={c:a[qid]@a[c][2:-2].T for c in calids}
  for length in [6,10,16]:
   for start in range(2,len(a[qid])-length-1,3):
    matches=[]
    for cid in calids:
     score,path,s,e=dtw(sims[cid][start:start+length]);path[:,0]+=start;path[:,1]+=2
     matches.append({'id':cid,'similarity':score,'start':s+2,'end':e+2,'path':path.tolist()})
    matches.sort(key=lambda r:-r['similarity']);rows.append({'start':start,'end':start+length,'matches':matches[:12]})
  out[qid]=rows
  if qi%10==0:print('retrieved',qi+1,'/',len(queryids),flush=True)
 return out

def weak_labels(recording,length,lookup,power):
 words=recording['tokens'];weights=np.array([len(w)**power for w in words]);cuts=np.r_[0,np.cumsum(weights)/sum(weights)]
 # The exact retrieval already excludes the first/last two frames.
 positions=(np.arange(length)-2+.5)/max(length-4,1)
 y=np.minimum(len(words)-1,np.maximum(0,np.searchsorted(cuts,positions,side='right')-1))
 return np.array([lookup[words[i]] for i in y])

def votes(rows,length,cal,raw,vocab,k,power):
 lookup={w:i for i,w in enumerate(vocab)};labels={c['id']:weak_labels(c,len(raw[c['id']]),lookup,power) for c in cal}
 score=np.zeros((length,len(vocab)))
 for row in rows:
  matches=row['matches'][:k];best=matches[0]['similarity']
  for match in matches:
   weight=np.exp(16*(match['similarity']-best));path=np.asarray(match['path']);counts=np.bincount(path[:,0],minlength=length)
   np.add.at(score,(path[:,0],labels[match['id']][path[:,1]]),weight/np.maximum(counts[path[:,0]],1))
 score=score[2:-2];score+=.005
 return np.log(score/score.sum(1,keepdims=True))

def language(cal,vocab):
 lookup={w:i for i,w in enumerate(vocab)};n=len(vocab);big=np.zeros((n,n));first=np.ones(n);last=np.ones(n);uni=np.ones(n)
 for c in cal:
  y=[lookup[w] for w in c['tokens']];first[y[0]]+=1;last[y[-1]]+=1
  for a in y:uni[a]+=1
  for a,b in zip(y,y[1:]):big[a,b]+=1
 uni/=uni.sum();big=(big+5*uni[None,:])/(big.sum(1,keepdims=True)+5)
 return np.log(big),np.log(first/first.sum()),np.log(last/last.sum())

def decode(emission,lm,vocab,lm_scale,penalty,min_frames):
 n=len(vocab);big,first,last=lm;trans=lm_scale*big+penalty;np.fill_diagonal(trans,-np.inf)
 if min_frames==1:
  d=emission[0]+lm_scale*first;back=[]
  for e in emission[1:]:
   s=d[:,None]+trans;np.fill_diagonal(s,d);p=s.argmax(0);d=s[p,np.arange(n)]+e;back.append(p)
  k=int(np.argmax(d+lm_scale*last));ys=[k]
  for p in back[::-1]:k=int(p[k]);ys.append(k)
 else:
  # Two explicit phases give each emitted word >=2 frames without a word count.
  d=np.full((2,n),-np.inf);d[0]=emission[0]+lm_scale*first;back=[]
  for e in emission[1:]:
   s=d[1,:,None]+trans;p=s.argmax(0);n0=s[p,np.arange(n)]+e
   take0=d[0]>d[1];n1=np.maximum(d[0],d[1])+e
   back.append((p,take0));d=np.array([n0,n1])
  phase=1;k=int(np.argmax(d[1]+lm_scale*last));ys=[k]
  for p,take0 in back[::-1]:
   if phase==0:k=int(p[k]);phase=1
   else:phase=0 if take0[k] else 1
   ys.append(k)
 ys=ys[::-1];return [vocab[y] for i,y in enumerate(ys) if i==0 or y!=ys[i-1]]

def calibrate():
 before=time.perf_counter();meta,raw=inputs();cal=meta['calibration'];vocab=sorted({w for c in cal for w in c['tokens']})
 # Read block metadata only. Query sentenceText is deliberately not loaded.
 source=loadmat(SOURCE,variable_names=['goTrialEpochs','blockNum'],simplify_cells=True);go=source['goTrialEpochs'].astype(int)-1;blocks=source['blockNum'][go[:,0]]
 case=np.load(ROOT/'cases/sentences_agentic_dev.npz');groups=blocks[case['train']];dtw=DTW();folds=[]
 for block in sorted(set(map(int,groups))):
  tr=[c for c,g in zip(cal,groups) if g!=block];va=[c for c,g in zip(cal,groups) if g==block];cache=OUT/f'calibration_block_{block}_retrieval.json'
  if cache.exists():r=read(cache)
  else:
   a=arrays({c['id']:raw[c['id']] for c in cal},[c['id'] for c in tr]);r=retrieval(a,[c['id'] for c in tr],[c['id'] for c in va],dtw);save(cache,r)
  folds.append((tr,va,r,language(tr,vocab)));print('fold',block,'ready',flush=True)
 candidates=[]
 # Shared deterministic settings; every selection score is calibration-only WER.
 grid=list(itertools.product([1,4,12],[0.,1.],[0.,.5,1.],[-1.,-2.5,-4.],[1,2]))
 cache={}
 for fi,(tr,va,r,lm) in enumerate(folds):
  for k,power in itertools.product([1,4,12],[0.,1.]):
   cache[fi,k,power]={c['id']:votes(r[c['id']],len(raw[c['id']]),tr,raw,vocab,k,power) for c in va}
 for k,power,lms,penalty,minimum in grid:
  edits=0;total=0;exact=0;fold_scores=[]
  for fi,(tr,va,r,lm) in enumerate(folds):
   fe=0;ft=0
   for c in va:
    pred=decode(cache[fi,k,power][c['id']],lm,vocab,lms,penalty,minimum);error=distance(c['tokens'],pred);fe+=error;ft+=len(c['tokens']);exact+=pred==c['tokens']
   edits+=fe;total+=ft;fold_scores.append(fe/ft)
  candidates.append({'top_k':k,'duration_letter_power':power,'lm_scale':lms,'transition_penalty':penalty,'min_word_frames':minimum,'calibration_cv_wer':edits/total,'calibration_cv_exact':exact,'fold_wer':fold_scores})
 candidates.sort(key=lambda c:(c['calibration_cv_wer'],-c['calibration_cv_exact'],c['top_k'],c['duration_letter_power'],c['lm_scale'],c['transition_penalty'],c['min_word_frames']))
 config={'method':'Exact Sol drop2/subsequence-DTW evidence, weak calibration word labels, weighted interval votes, free word-sequence Viterbi','selected':candidates[0],'all_candidates':candidates,'vocabulary':vocab,'folds':'Leave one calibration recording block out (1,4,6); PCA fitted on the other two blocks; no query text loaded','word_boundaries':'Weak initialization only: uniform duration or letter-length-proportional duration inside frames 2:T-2, chosen by calibration CV; no observed word boundaries','same_retrieval':'Development prediction will consume the actual local_dtw_drop2.json from the completed Sol call, unchanged. Top-k truncation and non-LLM assembly are calibration-selected. PCA/drop2/Gaussian sigma1.2/cosine/DTW recurrence are frozen.','query_labels_used':False,'input_sha256':{f:sha(SANDBOX/f) for f in ['inputs.json','neural_inputs.npz','local_dtw_drop2.json']},'code_sha256':sha(pathlib.Path(__file__)),'wall_seconds_calibration':time.perf_counter()-before,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 save(OUT/'configuration_lock.json',config);print('LOCKED',json.dumps(candidates[0]),'seconds',config['wall_seconds_calibration'],flush=True)

def splice_bank(cal):
 # Same calibration-only prefix/suffix shared-span construction used by the
 # completed no-signal prior composer. No query word counts/text are consulted.
 originals={tuple(c['tokens']) for c in cal};minimum=min(map(len,originals));maximum=max(map(len,originals));support=defaultdict(set)
 for left in cal:
  a=tuple(left['tokens'])
  for right in cal:
   if left['id']==right['id']:continue
   b=tuple(right['tokens'])
   for i in range(1,len(a)):
    for j in range(len(b)-1):
     for k in range(1,min(len(a)-i,len(b)-j-1)+1):
      if a[i:i+k]!=b[j:j+k]:break
      candidate=a[:i+k]+b[j+k:]
      if candidate not in originals and minimum<=len(candidate)<=maximum:support[candidate].add((left['id'],right['id']))
 candidates=sorted(support)
 return candidates,np.array([len(support[c]) for c in candidates],float)

def forced_candidate_scores(e,candidates,vocab):
 lookup={w:i for i,w in enumerate(vocab)};lengths=np.array([len(c) for c in candidates]);maximum=max(lengths);indices=np.zeros((len(candidates),maximum),int);valid=np.arange(maximum)[None,:]<lengths[:,None]
 for row,c in enumerate(candidates):indices[row,:len(c)]=[lookup[w] for w in c]
 d=np.full(indices.shape,-np.inf);d[:,0]=e[0,indices[:,0]]
 for frame in e[1:]:
  shifted=np.column_stack([np.full(len(candidates),-np.inf),d[:,:-1]])
  d=np.maximum(d,shifted)+frame[indices];d[~valid]=-np.inf
 return d[np.arange(len(candidates)),lengths-1]

def duration_prior(cal,raw,candidates,length):
 groups=defaultdict(list)
 for c in cal:groups[len(c['tokens'])].append(len(raw[c['id']]))
 masses={n:float(np.exp(-.5*((np.array(v)-length)/4.)**2).sum()) for n,v in groups.items()}
 return np.array([np.log(max(masses.get(len(c),0.),1e-12)) for c in candidates])

def calibrate_splice():
 before=time.perf_counter();cfg=read(OUT/'configuration_lock.json');meta,raw=inputs();cal=meta['calibration'];vocab=cfg['vocabulary'];s=cfg['selected']
 source=loadmat(SOURCE,variable_names=['goTrialEpochs','blockNum'],simplify_cells=True);go=source['goTrialEpochs'].astype(int)-1;groups=source['blockNum'][go[:,0]][np.load(ROOT/'cases/sentences_agentic_dev.npz')['train']]
 table=[];fold_counts=[]
 for block in sorted(set(map(int,groups))):
  tr=[c for c,g in zip(cal,groups) if g!=block];va=[c for c,g in zip(cal,groups) if g==block];r=read(OUT/f'calibration_block_{block}_retrieval.json');bank,support=splice_bank(tr);lengths=np.array([len(c) for c in bank]);fold_counts.append({'held_block':block,'candidate_count':len(bank)})
  for c in va:
   e=votes(r[c['id']],len(raw[c['id']]),tr,raw,vocab,s['top_k'],s['duration_letter_power']);scores=forced_candidate_scores(e,bank,vocab);duration=duration_prior(tr,raw,bank,len(raw[c['id']]))
   table.append({'truth':c['tokens'],'scores':scores,'support':np.log(support),'lengths':lengths,'duration':duration,'candidates':bank,'block':block})
 choices=[]
 for penalty,prior,ds in itertools.product([-4.,-2.5,-1.,0.,1.],[0.,.25,.75],[0.,.25,.75]):
  edits=0;words=0;exact=0;coverage=0;byblock=defaultdict(lambda:[0,0])
  for row in table:
   scores=row['scores']+penalty*(row['lengths']-1)+prior*row['support']+ds*row['duration'];best=int(np.argmax(scores));pred=list(row['candidates'][best]);error=distance(row['truth'],pred);edits+=error;words+=len(row['truth']);exact+=pred==row['truth'];coverage+=tuple(row['truth']) in row['candidates'];byblock[row['block']][0]+=error;byblock[row['block']][1]+=len(row['truth'])
  choices.append({'transition_penalty':penalty,'support_prior_weight':prior,'duration_prior_weight':ds,'calibration_cv_wer':edits/words,'calibration_cv_exact':exact,'calibration_cv_candidate_coverage':coverage,'fold_wer':{b:e/n for b,(e,n) in byblock.items()}})
 choices.sort(key=lambda c:(c['calibration_cv_wer'],-c['calibration_cv_exact'],c['support_prior_weight'],c['duration_prior_weight'],c['transition_penalty']))
 bank,support=splice_bank(cal);save(OUT/'full_calibration_splice_bank.json',{'candidates':[list(c) for c in bank],'support':support.astype(int).tolist(),'uses_query_text_or_counts':False})
 lock={'selected':choices[0],'all_candidates':choices,'fold_banks':fold_counts,'full_calibration_candidate_count':len(bank),'weak_word_vote_configuration_sha256':sha(OUT/'configuration_lock.json'),'calibration_only':True,'query_labels_used':False,'method':'Restrict the exact-DTW interval word-vote evidence to novel calibration-prefix/suffix splices; forced monotone word alignment, empirical text support and recording-duration priors. No LLM.','candidate_generator_provenance':'Same shared-contiguous-span construction as sentences_t12_confirm_nosignal/reconstruct_priors.py, reimplemented here without reading its predictions or query text. Rebuild independently on each training fold.','wall_seconds_calibration':time.perf_counter()-before,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'code_sha256':sha(pathlib.Path(__file__))}
 save(OUT/'splice_configuration_lock.json',lock);print('SPLICE LOCKED',json.dumps(choices[0]),'candidate_count',len(bank),'seconds',lock['wall_seconds_calibration'],flush=True)

def predict():
 before=time.perf_counter();cfg=read(OUT/'configuration_lock.json');meta,raw=inputs();cal=meta['calibration'];vocab=cfg['vocabulary'];r=read(SANDBOX/'local_dtw_drop2.json');lm=language(cal,vocab);s=cfg['selected'];pred=[]
 for q in meta['queries']:
  e=votes(r[q['id']],len(raw[q['id']]),cal,raw,vocab,s['top_k'],s['duration_letter_power']);tokens=decode(e,lm,vocab,s['lm_scale'],s['transition_penalty'],s['min_word_frames']);pred.append({'id':q['id'],'tokens':tokens})
 whole=read(SANDBOX/'variant_retrieval.json')['drop2'];lookup={c['id']:c['tokens'] for c in cal}
 references=[{'id':q['id'],'tokens':lookup[whole[q['id']][0]['id']],'retrieved_calibration_id':whole[q['id']][0]['id']} for q in meta['queries']]
 out={'name':'sentences_agentic_dev','matched_dtw_compositor':pred,'nearest_whole_recording_dtw':references,'configuration_sha256':sha(OUT/'configuration_lock.json'),'query_labels_used':False,'wall_seconds_decode':time.perf_counter()-before}
 if (OUT/'splice_configuration_lock.json').exists():
  slock=read(OUT/'splice_configuration_lock.json');selected=slock['selected'];bank,support=splice_bank(cal);lengths=np.array([len(c) for c in bank]);splice=[]
  for q in meta['queries']:
   e=votes(r[q['id']],len(raw[q['id']]),cal,raw,vocab,s['top_k'],s['duration_letter_power']);scores=forced_candidate_scores(e,bank,vocab)+selected['transition_penalty']*(lengths-1)+selected['support_prior_weight']*np.log(support)+selected['duration_prior_weight']*duration_prior(cal,raw,bank,len(raw[q['id']]));best=int(np.argmax(scores));splice.append({'id':q['id'],'tokens':list(bank[best])})
  out['matched_dtw_text_splice']=splice;out['splice_configuration_sha256']=sha(OUT/'splice_configuration_lock.json');out['wall_seconds_decode']=time.perf_counter()-before
 save(OUT/'development_predictions_locked.json',out);print('PREDICTIONS LOCKED',json.dumps(out),flush=True)

def evaluate():
 # This is the first baseline action to access query target text.
 before=time.perf_counter();p=read(OUT/'development_predictions_locked.json');meta=read(ROOT/'cases/sentences_agentic_dev.json');case=np.load(ROOT/'cases/sentences_agentic_dev.npz');source=loadmat(SOURCE,variable_names=['sentenceText'],simplify_cells=True);truth={q:re.findall('[a-z]+',str(source['sentenceText'][i]).lower()) for q,i in zip(meta['query_ids'],case['test'])};results={}
 for key in ['matched_dtw_compositor','nearest_whole_recording_dtw']+(['matched_dtw_text_splice'] if 'matched_dtw_text_splice' in p else []):
  rows=[{'id':r['id'],'prediction':r['tokens'],'truth':truth[r['id']],'edit_distance':distance(truth[r['id']],r['tokens'])} for r in p[key]]
  results[key]={'n':len(rows),'exact_sentences':sum(r['prediction']==r['truth'] for r in rows),'edits':sum(r['edit_distance'] for r in rows),'reference_words':sum(len(r['truth']) for r in rows),'wer':sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),'rows':rows}
 results['protocol']={'scope':'Development 8 only; baseline tuning and prediction lock preceded this evaluator reading query labels','configuration_sha256':sha(OUT/'configuration_lock.json'),'prediction_sha256':sha(OUT/'development_predictions_locked.json'),'same_candidate_intervals_as_Sol':True,'not_an_oracle_baseline':'Calibration word boundaries are estimated, not known. A poor result would not establish unique Sol capability; it could reflect weak interval-to-word labeling or decoding.','runtime_decode_seconds':p['wall_seconds_decode'],'runtime_calibration_seconds':read(OUT/'configuration_lock.json')['wall_seconds_calibration'],'runtime_evaluation_seconds':time.perf_counter()-before}
 save(OUT/'development_results.json',results);print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in results.items()},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['calibrate','calibrate_splice','predict','evaluate']);args=parser.parse_args()
 with threadpool_limits(limits=1):{'calibrate':calibrate,'calibrate_splice':calibrate_splice,'predict':predict,'evaluate':evaluate}[args.action]()
