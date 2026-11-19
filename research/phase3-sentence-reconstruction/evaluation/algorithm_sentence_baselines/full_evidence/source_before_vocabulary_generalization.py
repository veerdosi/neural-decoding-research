"""Non-LLM assembly using the complete structured numerical evidence inventory.

Frozen replay inherits Sol's adaptive choice of algorithms/calibration contexts.
The separate generic workflow recomputes a predetermined evidence inventory,
without Sol choosing intervals/candidates on a new query. No model calls.
Only evaluate() opens query sentenceText. All outputs are outside Sol inputs.
"""
import argparse,ctypes,itertools,json,pathlib,time
from collections import defaultdict
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
from sentence_algorithm_comparison import ROOT,PROJECT,SANDBOX,OUT,SOURCE,DTW,inputs,decode,language,weak_labels,read,save,sha,distance
from sentence_weak_word_baseline import forced_align

DEST=OUT/'full_evidence'
DEST.mkdir(exist_ok=True)
INSPECT='inspect_drop2_c080_c083_c010_c072_c100.json'
INSPECT_ORIGINS=['c080','c083','c010','c072','c100']
PREFIX_ORIGINS=['c007','c082','c060','c021','c045','c023','c111','c080','c030']
PREFIX_CANDIDATES=['c007','c082','c060','c021','c045','c023','c111','c080','c030','c017','c075','c079']
FAMILIES=['subsequence_drop2','local_drop2','local_white','local_raw','local_pca64drop2','local_pca32drop2','edge_drop2','global_raw','global_drop2','global_white','global_regwhite','global_pca32']

class Align:
 def __init__(self,kind):
  obj=DTW();self.fn=getattr(ctypes.CDLL(str(OUT/'sentence_dtw.dylib')),kind+'_dtw');self.fn.argtypes=obj.fn.argtypes;self.fn.restype=ctypes.c_double
 def __call__(self,s):
  s=np.ascontiguousarray(s,np.float64);n,m=s.shape;p=np.empty((2*(n+m),2),np.int32);length=ctypes.c_int();start=ctypes.c_int();end=ctypes.c_int();score=self.fn(s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),n,m,p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ctypes.byref(length),ctypes.byref(start),ctypes.byref(end))
  return float(score),p[:length.value].copy(),start.value,end.value

def representations(raw,calids):
 x=np.concatenate([raw[k] for k in calids]);mu=x.mean(0);w,v=np.linalg.eigh(np.cov(x.T));w=w[::-1];v=v[:,::-1];result={}
 for tag in ['drop2','white','raw','pca64drop2','pca32drop2','regwhite','pca32','globalraw']:
  weights=np.ones(x.shape[1]);sigma=1.2
  if tag in ['drop2','pca64drop2','pca32drop2']:weights[:2]=0
  if tag=='pca64drop2':weights[64:]=0
  if tag in ['pca32drop2','pca32']:weights[32:]=0
  if tag in ['white','regwhite']:weights=1/np.maximum(w,1)**.5
  if tag=='regwhite':sigma=2.
  if tag in ['pca32','globalraw']:sigma=.8
  a={}
  for key,z in raw.items():
   z=gaussian_filter1d(z.astype(float) if tag=='globalraw' else (z-mu)@v*weights,sigma,axis=0);a[key]=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-9)
  result[tag]=a
 return result

def interval_vector(a,s,e,n=8):
 t=np.linspace(s,e-1,n);lo=np.floor(t).astype(int);hi=np.minimum(lo+1,len(a)-1);alpha=(t-lo)[:,None];x=a[lo]*(1-alpha)+a[hi]*alpha
 return (x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)).ravel()/np.sqrt(n)

def cosine_retrieval(a,calids,queryids,inspection=False):
 cand=[];vectors=[]
 for cid in calids:
  for length in range(4,17 if inspection else 19):
   for start in range(2,len(a[cid])-length-1):cand.append((cid,start,start+length));vectors.append(interval_vector(a[cid],start,start+length))
 v=np.asarray(vectors,dtype=np.float32);out={}
 for qid in queryids:
  spans=[(s,s+l) for l in ([6] if inspection else [5,8,11,15]) for s in range(2,len(a[qid])-l-1,3)]
  if not spans:out[qid]=[];continue
  q=np.asarray([interval_vector(a[qid],s,e) for s,e in spans],np.float32);similarities=q@v.T;rows=[]
  for (s,e),sim in zip(spans,similarities):
   # Exact original rank rule: top150 intervals, then distinct calibration IDs.
   ix=np.argsort(sim)[::-1] if inspection else np.argsort(sim)[-150:][::-1];seen={qid} if inspection else set();matches=[]
   for j in ix:
    cid,cs,ce=cand[j]
    if cid in seen:continue
    seen.add(cid);matches.append({'id':cid,'start':cs,'end':ce,'similarity':float(sim[j])})
    if len(matches)==(6 if inspection else 10):break
   rows.append({'start':s,'end':e,'matches':matches})
  out[qid]=rows
 return out

def global_retrieval(a,calids,queryids):
 fn=Align('global');out={}
 for qid in queryids:
  rows=[]
  for cid in calids:
   score,path,s,e=fn(a[qid]@a[cid].T);rows.append({'id':cid,'similarity':-score,'path':path.tolist()})
  rows.sort(key=lambda r:-r['similarity']);out[qid]=[{'start':0,'end':len(a[qid]),'matches':rows}]
 return out

def edge_retrieval(a,calids,queryids):
 fn=Align('prefix');out={}
 for qid in queryids:
  rows=[]
  for length in [5,8,12,16]:
   if len(a[qid])<length+4:continue
   matches=[]
   for cid in calids:
    score,path,s,e=fn(a[qid][2:2+length]@a[cid][2:-2].T);path[:,0]+=2;path[:,1]+=2;matches.append({'id':cid,'similarity':score,'path':path.tolist(),'start':s+2,'end':e+2})
   matches.sort(key=lambda r:-r['similarity']);rows.append({'start':2,'end':2+length,'matches':matches})
  out[qid]=rows
 return out

def build(cal,raw,queryids,cache,generic=False):
 if cache.exists():return read(cache)
 ids=[c['id'] for c in cal];a=representations(raw,ids);result={}
 from sentence_algorithm_comparison import retrieval
 result['subsequence_drop2']=retrieval(a['drop2'],ids,queryids,DTW())
 for tag in ['drop2','white','raw','pca64drop2','pca32drop2']:
  print('local cosine',tag,len(ids),len(queryids),flush=True);result['local_'+tag]=cosine_retrieval(a[tag],ids,queryids)
 for tag in ['raw','drop2','white','regwhite','pca32']:result['global_'+tag]=global_retrieval(a['globalraw' if tag=='raw' else tag],ids,queryids)
 result['edge_drop2']=edge_retrieval(a['drop2'],ids,queryids)
 # Sol selected these calibration contexts adaptively on development. Generic
 # mode replaces that choice by a fixed label-free, evenly spaced ID subset.
 origins=(ids[::max(1,len(ids)//8)] if generic else [k for k in INSPECT_ORIGINS if k in ids])
 result['calibration_inspection']=cosine_retrieval(a['drop2'],ids,origins,True)
 origins=([k for k in ids[::max(1,len(ids)//8)]] if generic else [k for k in PREFIX_ORIGINS if k in ids]);candidates=ids if generic else [k for k in PREFIX_CANDIDATES if k in ids]
 result['calibration_prefix']=edge_retrieval(a['drop2'],candidates,origins)
 save(cache,result);return result

def frozen():
 # Replay actual saved scores/intervals. Path recovery adds no learned labels.
 meta,raw=inputs();allids=[c['id'] for c in meta['calibration']];a=representations(raw,allids);out={}
 out['subsequence_drop2']=read(SANDBOX/'local_dtw_drop2.json')
 for tag in ['drop2','white','raw','pca64drop2','pca32drop2']:out['local_'+tag]=read(SANDBOX/f'local_{tag}.json')
 whole=read(SANDBOX/'global_retrieval.json');variants=read(SANDBOX/'variant_retrieval.json')
 for tag in ['raw','drop2','white','regwhite','pca32']:
  source=whole if tag=='raw' else variants[tag];out['global_'+tag]={q:[{'start':0,'end':len(raw[q]),'matches':[{'id':r['id'],'similarity':-r['score'],'path':r['path']} for r in rows]}] for q,rows in source.items()}
 edge=read(SANDBOX/'edge_drop2.json');fn=Align('prefix');out['edge_drop2']={};errors=[]
 for qid,rows in edge.items():
  qr=[]
  for row in rows:
   n=row['length'];matches=[]
   for r in row['matches']:
    score,path,s,e=fn(a['drop2'][qid][2:2+n]@a['drop2'][r['id']][2:-2].T);errors.append(abs(score-r['similarity']));path[:,0]+=2;path[:,1]+=2;matches.append(dict(r,path=path.tolist()))
   qr.append({'start':2,'end':2+n,'matches':matches})
  out['edge_drop2'][qid]=qr
 out['calibration_inspection']=read(SANDBOX/INSPECT)
 # Prefix-comparison query rows duplicate exact n=12 edge scores. Calibration
 # rows are recovered with the same prefix recurrence over the specified IDs.
 prefix=read(SANDBOX/'prefix_comparisons.json');out['calibration_prefix']={}
 for origin,values in prefix.items():
  if not origin.startswith('c'):continue
  matches=[]
  for cid,rounded in values.items():
   score,path,s,e=fn(a['drop2'][origin][2:14]@a['drop2'][cid][2:-2].T);path[:,0]+=2;path[:,1]+=2;matches.append({'id':cid,'similarity':score,'path':path.tolist()})
  matches.sort(key=lambda r:-r['similarity']);out['calibration_prefix'][origin]=[{'start':2,'end':14,'matches':matches}]
 save(DEST/'frozen_prefix_path_recovery.json',{'maximum_edge_score_difference':max(errors),'edge_alignments':len(errors),'prefix_comparison_scores':'Original prefix_comparisons.json rounds to 3 decimals; paths recovered from the identical underlying neural arrays/recurrence.'})
 return out

def counts(rows,length,labels,k):
 v=np.zeros((length,50));mass=np.zeros(length)
 for row in rows:
  matches=row['matches'][:k]
  if not matches:continue
  best=matches[0]['similarity']
  for r in matches:
   weight=np.exp(16*(r['similarity']-best))
   if 'path' in r:p=np.asarray(r['path'],int)
   else:
    q=np.arange(row['start'],row['end']);c=np.rint(np.linspace(r['start'],r['end']-1,len(q))).astype(int);p=np.column_stack([q,c])
   n=np.bincount(p[:,0],minlength=length);w=weight/np.maximum(n[p[:,0]],1);np.add.at(v,(p[:,0],labels[r['id']][p[:,1]]),w);np.add.at(mass,p[:,0],w)
 return v,mass

def label_set(cal,raw,vocab,power,evidence,k,refine):
 lookup={w:i for i,w in enumerate(vocab)};labels={c['id']:weak_labels(c,len(raw[c['id']]),lookup,power) for c in cal}
 if not refine:return labels
 copied={k:v.copy() for k,v in labels.items()};calbyid={c['id']:c for c in cal}
 origins=set(evidence['calibration_inspection'])|set(evidence['calibration_prefix'])
 for origin in origins:
  n=len(raw[origin]);v=np.zeros((n,50));mass=np.zeros(n)
  for source in ['calibration_inspection','calibration_prefix']:
   c,m=counts(evidence[source].get(origin,[]),n,labels,k);v+=c;mass+=m
  if not mass.any():continue
  probability=v/(v.sum(1,keepdims=True)+1e-12);prior=np.full_like(probability,.005);prior[np.arange(n),labels[origin]]+=1
  score=np.log(probability*.65+prior*.35+.0001)
  aligned=forced_align(score[2:-2],[lookup[w] for w in calbyid[origin]['tokens']]);copied[origin][2:-2]=aligned
 return copied

def probabilities(evidence,cal,queries,raw,vocab,k,power,refine):
 labels=label_set(cal,raw,vocab,power,evidence,k,refine);out={}
 for family in FAMILIES:
  out[family]={}
  for qid in queries:
   v,m=counts(evidence[family][qid],len(raw[qid]),labels,k);v=v[2:-2];m=m[2:-2];out[family][qid]=(v/(v.sum(1,keepdims=True)+1e-12),m>0)
 return out

GROUPS=[[f] for f in FAMILIES]+[
 FAMILIES,
 [f for f in FAMILIES if not f.startswith('global')],
 [f for f in FAMILIES if f.startswith('global')],
 [f for f in FAMILIES if f.startswith('local')],
 ['subsequence_drop2','local_white','local_pca64drop2','edge_drop2'],
 ['subsequence_drop2','edge_drop2','global_drop2']]

def ensemble(prob,qid,group):
 first=prob[group[0]][qid][0];v=np.zeros_like(first);mass=np.zeros(len(first))
 for family in group:
  p,m=prob[family][qid];v+=p*m[:,None];mass+=m
 v/=np.maximum(mass[:,None],1);v+=.0001;return np.log(v/v.sum(1,keepdims=True))

def calibrate(generic=False):
 before=time.perf_counter();meta,raw=inputs();cal=meta['calibration'];vocab=sorted({w for c in cal for w in c['tokens']});case=np.load(ROOT/'cases/sentences_agentic_dev.npz');source=loadmat(SOURCE,variable_names=['goTrialEpochs','blockNum'],simplify_cells=True);go=source['goTrialEpochs'].astype(int)-1;groups=source['blockNum'][go[:,0]][case['train']];folds=[]
 for block in sorted(set(map(int,groups))):
  tr=[c for c,g in zip(cal,groups) if g!=block];va=[c for c,g in zip(cal,groups) if g==block];e=build(tr,{c['id']:raw[c['id']] for c in cal},[c['id'] for c in va],DEST/f'calibration_block_{block}_all_evidence.json')
  if generic:
   cache=DEST/f'calibration_block_{block}_generic_contexts.json'
   if cache.exists():contexts=read(cache)
   else:
    ids=[c['id'] for c in tr];a=representations({c['id']:raw[c['id']] for c in cal},ids)['drop2'];origins=ids[::max(1,len(ids)//8)];contexts={'calibration_inspection':cosine_retrieval(a,ids,origins,True),'calibration_prefix':edge_retrieval(a,ids,origins)};save(cache,contexts)
   e={**e,**contexts}
  else:
   # The logged calibration-prefix comparison used twelve frames only; its
   # query-prefix inventory separately includes 5/8/12/16-frame edge scores.
   e={**e,'calibration_prefix':{origin:[row for row in rows if row['end']-row['start']==12] for origin,rows in e['calibration_prefix'].items()}}
  folds.append((tr,va,e,language(tr,vocab)))
 records=[]
 # Select evidence families/ensembles and assembly settings on calibration only.
 for k,power,refine in itertools.product([4,12],[0.,1.],[False,True]):
  cached=[probabilities(e,tr,[c['id'] for c in va],raw,vocab,k,power,refine) for tr,va,e,lm in folds]
  for gi,group in enumerate(GROUPS):
   emissions=[{c['id']:ensemble(p,c['id'],group) for c in va} for p,(tr,va,e,lm) in zip(cached,folds)]
   for lms,penalty in itertools.product([0.,.5],[-2.5,-4.]):
    errors=0;total=0;exact=0
    for em,(tr,va,e,lm) in zip(emissions,folds):
     for c in va:
      pred=decode(em[c['id']],lm,vocab,lms,penalty,1);errors+=distance(c['tokens'],pred);total+=len(c['tokens']);exact+=pred==c['tokens']
    records.append({'families':group,'group_index':gi,'top_k':k,'duration_letter_power':power,'calibration_context_refinement':refine,'lm_scale':lms,'transition_penalty':penalty,'min_word_frames':1,'calibration_cv_wer':errors/total,'calibration_cv_exact':exact})
 records.sort(key=lambda c:(c['calibration_cv_wer'],-c['calibration_cv_exact'],len(c['families']),c['group_index'],c['calibration_context_refinement']))
 lock={'selected':records[0],'all_candidates':records,'vocabulary':vocab,'folds':'Leave one calibration block out, including PCA/retrieval/calibration-label refinement/text priors; no query text loaded','query_labels_used':False,'evidence_inventory':FAMILIES,'scope':('Fixed generic workflow: predetermined full numerical inventory and evenly spaced calibration-context IDs; no Sol query-dependent orchestration at application.' if generic else 'Assembly given frozen Sol-selected evidence inventory and selected calibration-context IDs. This inherits Sol orchestration; not a causal elimination of every Sol contribution.'),'weak_word_boundaries':'Uniform or letter-length-proportional initial segmentation; optional cross-calibration neural-context votes constrained to each known calibration transcript','wall_seconds':time.perf_counter()-before,'code_sha256':sha(pathlib.Path(__file__)),'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};save(DEST/('generic_configuration_lock.json' if generic else 'configuration_lock.json'),lock);print('GENERIC' if generic else 'FROZEN','FULL EVIDENCE LOCKED',json.dumps(records[0]),'seconds',lock['wall_seconds'],flush=True)

def predict():
 before=time.perf_counter();lock=read(DEST/'configuration_lock.json');s=lock['selected'];meta,raw=inputs();cal=meta['calibration'];vocab=lock['vocabulary'];e=frozen();p=probabilities(e,cal,[q['id'] for q in meta['queries']],raw,vocab,s['top_k'],s['duration_letter_power'],s['calibration_context_refinement']);lm=language(cal,vocab);predictions=[]
 for q in meta['queries']:
  em=ensemble(p,q['id'],s['families']);pred=decode(em,lm,vocab,s['lm_scale'],s['transition_penalty'],1);predictions.append({'id':q['id'],'tokens':pred})
 save(DEST/'frozen_replay_predictions_locked.json',{'predictions':predictions,'query_labels_used':False,'configuration_sha256':sha(DEST/'configuration_lock.json'),'wall_seconds':time.perf_counter()-before,'uses_model_authored_hypothesis_text_or_predictions':False});print('FULL REPLAY PREDICTIONS LOCKED',json.dumps(predictions),flush=True)

def evaluate():
 p=read(DEST/'frozen_replay_predictions_locked.json');meta=read(ROOT/'cases/sentences_agentic_dev.json');case=np.load(ROOT/'cases/sentences_agentic_dev.npz');source=loadmat(SOURCE,variable_names=['sentenceText'],simplify_cells=True)
 import re
 truth={q:re.findall('[a-z]+',str(source['sentenceText'][i]).lower()) for q,i in zip(meta['query_ids'],case['test'])};rows=[{'id':r['id'],'prediction':r['tokens'],'truth':truth[r['id']],'edit_distance':distance(truth[r['id']],r['tokens'])} for r in p['predictions']];out={'n':len(rows),'exact_sentences':sum(r['prediction']==r['truth'] for r in rows),'edits':sum(r['edit_distance'] for r in rows),'reference_words':sum(len(r['truth']) for r in rows),'wer':sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),'rows':rows,'scope':'Development evidence replay; all structured neural-query score families available, with family/ensemble selection calibration-only. This inherits Sol orchestration and does not remove it causally.','runtime_assembly_seconds':p['wall_seconds']};save(DEST/'development_results.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

def predict_generic(name):
 before=time.perf_counter();lock=read(DEST/'generic_configuration_lock.json');s=lock['selected'];case=read(ROOT/'cases'/f'{name}.json');folder=pathlib.Path(case['sandbox']);meta=read(folder/'inputs.json');z=np.load(folder/'neural_inputs.npz');raw={k:z[k] for k in z.files};cal=meta['calibration'];vocab=lock['vocabulary'];assert sorted({w for c in cal for w in c['tokens']})==vocab
 # No Sol-produced helper, score, hypothesis or output is read in this workflow.
 evidence=build(cal,raw,[q['id'] for q in meta['queries']],DEST/f'generic_{name}_evidence.json',True);p=probabilities(evidence,cal,[q['id'] for q in meta['queries']],raw,vocab,s['top_k'],s['duration_letter_power'],s['calibration_context_refinement']);lm=language(cal,vocab);predictions=[]
 for q in meta['queries']:
  e=ensemble(p,q['id'],s['families']);predictions.append({'id':q['id'],'tokens':decode(e,lm,vocab,s['lm_scale'],s['transition_penalty'],1)})
 result={'case':name,'predictions':predictions,'configuration_sha256':sha(DEST/'generic_configuration_lock.json'),'query_labels_used':False,'input_sha256':{f:sha(folder/f) for f in ['inputs.json','neural_inputs.npz']},'workflow':'Fixed generic neural evidence generation and deterministic word assembly; no Sol at application or Sol-produced artifacts consumed. Numerical algorithm inventory was derived during development.','wall_seconds_end_to_end':time.perf_counter()-before,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};save(DEST/f'generic_{name}_predictions_locked.json',result);print('GENERIC PREDICTIONS LOCKED',json.dumps(predictions),'seconds',result['wall_seconds_end_to_end'],flush=True)

def evaluate_generic(name):
 p=read(DEST/f'generic_{name}_predictions_locked.json');meta=read(ROOT/'cases'/f'{name}.json');case=np.load(ROOT/'cases'/f'{name}.npz');source=loadmat(PROJECT/'work/sources'/meta['source'],variable_names=['sentenceText'],simplify_cells=True)
 import re
 truth={q:re.findall('[a-z]+',str(source['sentenceText'][i]).lower()) for q,i in zip(meta['query_ids'],case['test'])};rows=[{'id':r['id'],'prediction':r['tokens'],'truth':truth[r['id']],'edit_distance':distance(truth[r['id']],r['tokens'])} for r in p['predictions']];out={'n':len(rows),'exact_sentences':sum(r['prediction']==r['truth'] for r in rows),'edits':sum(r['edit_distance'] for r in rows),'reference_words':sum(len(r['truth']) for r in rows),'wer':sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),'rows':rows,'workflow':p['workflow'],'runtime_end_to_end_seconds':p['wall_seconds_end_to_end'],'configuration_sha256':p['configuration_sha256'],'prediction_sha256':sha(DEST/f'generic_{name}_predictions_locked.json')};save(DEST/f'generic_{name}_results.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['calibrate','predict','evaluate','calibrate_generic','predict_generic','evaluate_generic']);parser.add_argument('--case',default='sentences_t12_fresh');args=parser.parse_args()
 with threadpool_limits(limits=1):{'calibrate':calibrate,'predict':predict,'evaluate':evaluate,'calibrate_generic':lambda:calibrate(True),'predict_generic':lambda:predict_generic(args.case),'evaluate_generic':lambda:evaluate_generic(args.case)}[args.action]()
