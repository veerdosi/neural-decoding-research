from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, subprocess, time, concurrent.futures
import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import LedoitWolf
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

PACKAGE=pathlib.Path(__file__).resolve().parents[1]
ROOT=pathlib.Path(os.environ.get('NEURAL_PHASE2_OUTPUT_DIR',str(PACKAGE))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
PROJECT=PACKAGE.parents[1]
WORK=PROJECT/'local'
PHASE1=PACKAGE/'phase1_reference' if (PACKAGE/'phase1_reference').exists() else PACKAGE.parent/'phase1-word-decoding'
WORDS=['ban','choice','day','feel','kite','though','were']
SEED=2026111802
for name in ['cases','data','prompts','responses','evaluation']:(ROOT/name).mkdir(exist_ok=True)

class ShrinkageLDA:
 """Shared within-class covariance shrunk to a scaled identity.
 For one calibration point per class, covariance is unidentifiable and the
 prespecified regularized limit is identity covariance (nearest centroid).
 """
 def __init__(self,shrinkage=.1):self.shrinkage=shrinkage
 def fit(self,x,y):
  self.classes_=np.unique(y);means=np.array([x[y==c].mean(axis=0) for c in self.classes_])
  residual=x-np.array([means[np.flatnonzero(self.classes_==c)[0]] for c in y])
  covariance=residual.T@residual/len(x);scale=np.trace(covariance)/x.shape[1]
  if scale<1e-10:covariance=np.eye(x.shape[1]);scale=1.
  elif self.shrinkage=='auto':covariance=LedoitWolf().fit(residual).covariance_
  if self.shrinkage!='auto':covariance=(1-self.shrinkage)*covariance+self.shrinkage*scale*np.eye(x.shape[1])
  covariance+=1e-8*np.eye(x.shape[1]);self.coef_=np.linalg.solve(covariance,means.T)
  self.intercept_=-.5*np.sum(means*self.coef_.T,axis=1)
  return self
 def predict(self,x):return self.classes_[np.argmax(x@self.coef_+self.intercept_,axis=1)]

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def extract(name,file):
 d=loadmat(file,simplify_cells=True)
 go=np.asarray(d['goTrialEpochs'],int)-1
 labels=np.asarray(d['trialCues'],int)-2
 blocks=np.asarray(d['blockNum'])[go[:,0]].astype(int)
 arrays=list(d['chanSetNames'])
 bins=int(d['binSize'])
 def feature(region,start,stop):
  ch=np.asarray(d['chanSets'][arrays.index(region)],int)-1
  cont=np.column_stack([d['binnedTX'][:,ch],np.log1p(np.asarray(d['spikePow'][:,ch],float))])
  rows=[]; centered=[]
  means={b:cont[np.asarray(d['blockNum'])==b].mean(axis=0) for b in np.unique(blocks)}
  for (a,e),b in zip(go,blocks):
   lo=a+round(start/bins);hi=a+round(stop/bins)
   assert lo>=0 and hi<=len(cont) and hi>lo
   assert np.all(np.asarray(d['blockNum'])[lo:hi]==b)
   if start>=0:assert hi<=e+1
   row=cont[lo:hi].mean(axis=0);rows.append(row);centered.append(row-means[b])
  return np.array(rows),np.array(centered)
 region='i6v' if 'i6v' in arrays else 'ventral6v'
 post,centered=feature(region,200,1200)
 delay,_=feature(region,-1200,-200)
 late,_=feature(region,1200,2200)
 other,_=feature('s6v',200,1200) if 's6v' in arrays else (np.empty((len(go),0)),None)
 eligible=np.unique(blocks[labels>=0]);cut=2 if len(eligible)>2 else 1
 train=np.flatnonzero((labels>=0)&np.isin(blocks,eligible[:cut]));test=np.flatnonzero((labels>=0)&np.isin(blocks,eligible[cut:]))
 assert len(test)>0 and not set(train)&set(test)
 np.savez_compressed(ROOT/'data'/f'{name}.npz',raw=post,delay=delay,delta=post-delay,late=late,
                     blockcenter=centered,tx=post[:,:64],sp=post[:,64:],other=other,
                     labels=labels,blocks=blocks,train=train,test=test,go=go)
 # Structure and calibration counts only: no confirmation evaluation labels/accuracy printed.
 intake={'dataset':name,'source_name':pathlib.Path(file).name,'source_sha256':hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest(),
         'array_names':arrays,'bin_ms':bins,'trial_n':len(go),'calibration_blocks':eligible[:cut].tolist(),
         'evaluation_blocks':eligible[cut:].tolist(),'calibration_n':len(train),'evaluation_n':len(test),
         'calibration_counts':dict(collections.Counter(labels[train].tolist()))}
 save(ROOT/'data'/f'{name}.intake.json',intake);print(json.dumps(intake))

def subset(d,n,draw):
 rng=np.random.default_rng(SEED+draw)
 candidates=d['train'];bl=np.unique(d['blocks'][candidates]);chosen=[]
 for word in range(7):
  for i,b in enumerate(bl):
   ids=candidates[(d['labels'][candidates]==word)&(d['blocks'][candidates]==b)]
   ids=rng.permutation(ids)
   take=n if len(bl)==1 else n//2+(i==(draw%2) and n%2)
   assert take<=len(ids),(n,len(ids))
   chosen.extend(ids[:take])
 return rng.permutation(chosen).astype(int)

def transform(cal,query,representation):
 mu=cal.mean(axis=0);sc=cal.std(axis=0,ddof=1);sc[sc<1e-8]=1
 a=(cal-mu)/sc;b=(query-mu)/sc
 if representation in ['pca','pca_tail']:
  _,_,vt=np.linalg.svd(a,full_matrices=False)
  pc=vt[:min(12,len(a)-1)] if representation=='pca' else vt[12:len(a)-1]
  a=a@pc.T;b=b@pc.T
 return np.round(a,2),np.round(b,2)

def make_case(name,dataset='t12',n=14,draw=0,representation='raw',queries=None,control=None,exact=False,prompt_extra='',alias=False):
 if (ROOT/'cases'/f'{name}.json').exists():raise ValueError('Case already exists; never overwrite '+name)
 d=np.load(ROOT/'data'/f'{dataset}.npz');rng=np.random.default_rng(SEED+10000+draw)
 train=subset(d,n,draw);test=rng.permutation(d['test']) if queries is None else np.asarray(queries,int)
 matrix=d['raw' if representation in ['pca','pca_tail'] else representation]
 xcal,xquery=transform(matrix[train],matrix[test],representation)
 ids=[f'q{v:08x}' for v in rng.choice(2**32,len(test),replace=False)]
 if exact:
  p=np.load(PHASE1/'evaluation/prepared.npz');train=p['train'];test=p['test'];xcal=np.round(p['z'][train],2);xquery=np.round(p['z'][test],2)
  ids=[r['id'] for r in json.loads((PHASE1/'evaluation/ground_truth.json').read_text())]
 donor=test.copy()
 if control=='shuffled':
  for b in np.unique(d['blocks'][test]):
   idx=np.flatnonzero(d['blocks'][test]==b);donor[idx]=rng.permutation(test[idx])
  _,xquery=transform(matrix[train],matrix[donor],representation)
 tokens=WORDS.copy()
 if alias:tokens=['vax','nup','zef','kog','bim','rud','pel'];rng.shuffle(tokens)
 text=(
 'You are the inference subject in a controlled neuroscience experiment. Use only information in this prompt. '
 'Do not use tools, read files, browse, or delegate. Identify the cued word for each independent neural recording. '
 'The participant was instructed to imagine speaking a single word. Candidate labels: '+', '.join(tokens)+'. '
 'Query IDs are random and contain no information. Query order is random. '
 'No metadata, timestamps, text stimuli, or evaluation labels are supplied. '
 'Do not infer any label-count constraint or force a balanced output. '
 'Infer participant-specific patterns from labeled calibration examples. You may reason about distances, '
 'recurring dimensions, clusters, and noisy examples internally. Do not use an external trained decoder. '
 'Commit to one candidate label per query. Return ONLY a JSON object with predictions (list of objects with id '
 'and word, each query exactly once) and method (short description of your decision method and its main uncertainty).\n'
 )
 text+='\nRepresentation: '+str(xcal.shape[1])+' numerical neural features; all dimensions match between examples. '
 text+='Each coordinate is standardized using only the supplied calibration recordings. '
 if representation=='raw':text+='First 64 are mean threshold-crossing counts; last 64 are mean log1p spike power, 200-1200 ms after go.'
 elif representation=='delta':text+='Mean post-go (200-1200 ms) minus same-trial pre-go (-1200 to -200 ms), with 64 crossing and 64 log1p power coordinates.'
 elif representation=='blockcenter':text+='Crossing and log1p power means, 200-1200 ms post-go, after subtracting the unlabeled continuous recording mean within each block. This is a batch transformation.'
 elif representation=='delay':text+='Mean threshold crossings and log1p spike power, -1200 to -200 ms before go, during the text-cue delay.'
 else:text+=representation+' deterministic neural representation.'
 if alias:text+=' Candidate labels are arbitrary tokens, with no semantic relationship to physiology. Learn only the supplied mapping.'
 text+='\n'+prompt_extra+'\nLabeled calibration examples, in random order:\n'
 for i,t in enumerate(train):text+=json.dumps({'word':tokens[d['labels'][t]],'x':xcal[i].tolist()},separators=(',',':'))+'\n'
 text+='\nUnlabeled queries:\n'
 for i,q in enumerate(ids):
  row={'id':q}
  if control!='no_signal':row['x']=xquery[i].tolist()
  text+=json.dumps(row,separators=(',',':'))+'\n'
 if control=='no_signal':text+='\nQuery neural recordings are withheld. Still commit to one candidate per query using available priors.\n'
 if exact:text=(PHASE1/'prompts/raw128_icl.txt').read_text()
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',cal_raw=matrix[train],query_raw=matrix[test],cal_x=xcal,query_x=xquery,
                     cal_y=d['labels'][train],query_y=d['labels'][test],donor_y=d['labels'][donor],
                     cal_blocks=d['blocks'][train],query_blocks=d['blocks'][test],train=train,test=test)
 meta={'name':name,'dataset':dataset,'n_per_word':n,'draw':draw,'representation':representation,'control':control,'alias_tokens':tokens,
       'query_ids':ids,'exact_phase1_prompt':exact,'query_n':len(test),'evaluation_status':'previously_evaluated_development' if dataset=='t12' else 'untouched_participant_confirmation',
       'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 (ROOT/'prompts'/f'{name}.txt').write_text(text);save(ROOT/'cases'/f'{name}.json',meta)
 return name

def phase1_chunk(name,rows):
 parent='dev_raw_n14_repeat1';p=np.load(ROOT/'cases'/f'{parent}.npz');meta=json.loads((ROOT/'cases'/f'{parent}.json').read_text())
 rows=np.asarray(rows,int);arrays={}
 for key in p.files:arrays[key]=p[key][rows] if key in ['query_raw','query_x','query_y','donor_y','query_blocks','test'] else p[key]
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',**arrays)
 sep='\nUnlabeled queries:\n';header,queries=(PHASE1/'prompts/raw128_icl.txt').read_text().split(sep)
 lines=queries.strip().splitlines();text=header+sep+'\n'.join(lines[i] for i in rows)+'\n'
 meta.update(name=name,query_ids=[meta['query_ids'][i] for i in rows],query_n=len(rows),exact_phase1_prompt=False,
             phase1_header_exact=True,parent_query_rows=rows.tolist(),prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
 save(ROOT/'cases'/f'{name}.json',meta);(ROOT/'prompts'/f'{name}.txt').write_text(text)
 return name

def rotation_case(name='dev_rotation_n14'):
 parent='dev_raw_n14_repeat1';d=np.load(ROOT/'cases'/f'{parent}.npz');meta=json.loads((ROOT/'cases'/f'{parent}.json').read_text())
 rng=np.random.default_rng(SEED+991);rotation,_=np.linalg.qr(rng.normal(size=(128,128)))
 a=np.round(d['cal_x']@rotation,2);b=np.round(d['query_x']@rotation,2)
 arrays={k:d[k] for k in d.files};arrays.update(cal_raw=a,query_raw=b,cal_x=a,query_x=b)
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',**arrays);np.save(ROOT/'data/orthogonal_rotation.npy',rotation)
 original=(PHASE1/'prompts/raw128_icl.txt').read_text();sep='\nLabeled calibration examples, in random order:\n';header=original.split(sep)[0]
 header=header.split('\nRepresentation:')[0]+'\nRepresentation: 128 rotated coordinates of calibration-standardized neural features. A fixed label-independent orthogonal matrix mixes all 128 original dimensions. No dimensions were discarded; Euclidean information is preserved up to rounding. Each individual coordinate has no anatomical or semantic label. Coordinates match across examples.\n'
 text=header+sep
 for y,x in zip(d['cal_y'],a):text+=json.dumps({'word':WORDS[y],'x':x.tolist()},separators=(',',':'))+'\n'
 text+='\nUnlabeled queries:\n'
 for q,x in zip(meta['query_ids'],b):text+=json.dumps({'id':q,'x':x.tolist()},separators=(',',':'))+'\n'
 meta.update(name=name,representation='rotation',exact_phase1_prompt=False,prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),normalization_rule='Original calibration standardization, then full orthogonal rotation. No renormalization.')
 save(ROOT/'cases'/f'{name}.json',meta);(ROOT/'prompts'/f'{name}.txt').write_text(text)
 base=json.loads((ROOT/'evaluation'/f'{parent}.baseline.json').read_text());result={}
 for method,cls in [('knn',KNeighborsClassifier),('lda',ShrinkageLDA)]:
  before=time.perf_counter();pred=cls(**base[method]['parameters']).fit(a,d['cal_y']).predict(b)
  result[method]={**base[method],'prediction':pred.tolist(),'accuracy':float(np.mean(pred==d['query_y'])),'correct':int(np.sum(pred==d['query_y'])),
                  'parameters_origin':'Frozen original-representation calibration CV; orthogonal-equivariant control, no renormalization',
                  'elapsed_seconds_tuning_fit_predict':time.perf_counter()-before,
                  'prediction_agreement_with_original':float(np.mean(pred==np.asarray(base[method]['prediction'])))}
 save(ROOT/'evaluation'/f'{name}.baseline.json',result)
 return name

def baseline_case(name,mean_align=False):
 meta=json.loads((ROOT/'cases'/f'{name}.json').read_text());d=np.load(ROOT/'cases'/f'{name}.npz')
 cal=d['cal_raw'];y=d['cal_y'];blocks=d['cal_blocks'];q=d['query_raw'];repr=meta['representation']
 result={}
 grids={'knn':[{'n_neighbors':k,'weights':w} for k in [1,3,5,9] for w in ['uniform','distance']],
        'lda':[{'shrinkage':s} for s in [.1,.5,.9,1.,'auto']],
        'gnb':[{'var_smoothing':s} for s in [1e-9,1e-4,.01,.1,1.]]}
 folds=[(np.flatnonzero(blocks!=b),np.flatnonzero(blocks==b)) for b in np.unique(blocks)]
 if len(folds)==1:
  folds=list(StratifiedKFold(n_splits=min(3,int(np.bincount(y).min())),shuffle=True,random_state=SEED).split(cal,y))
 with threadpool_limits(limits=1):
  for method,grid in grids.items():
   before=time.perf_counter();scores=[]
   cls={'knn':KNeighborsClassifier,'lda':ShrinkageLDA,'gnb':GaussianNB}[method]
   for params in grid:
    vals=[]
    for tr,va in folds:
     if method=='knn' and params['n_neighbors']>len(tr):continue
     x,z=transform(cal[tr],cal[va],repr)
     if mean_align:z=z-z.mean(axis=0)+x.mean(axis=0)
     model=cls(**params).fit(x,y[tr]);vals.append(float(np.mean(model.predict(z)==y[va])))
    scores.append(float(np.mean(vals)) if vals else -1.0)
   best=int(np.argmax(scores));model=cls(**grid[best]).fit(d['cal_x'],y)
   # Same rounded query features as Sol for non-controls. No control-derived baseline training.
   _,qx=transform(cal,q,repr)
   if mean_align:qx=qx-qx.mean(axis=0)+d['cal_x'].mean(axis=0)
   pred=model.predict(qx)
   result[method]={'parameters':grid[best],'cv_grid_scores':scores,'prediction':pred.tolist(),'accuracy':float(np.mean(pred==d['query_y'])),
                   'correct':int(np.sum(pred==d['query_y'])),'elapsed_seconds_tuning_fit_predict':time.perf_counter()-before,
                   'block_accuracy':{str(b):float(np.mean(pred[d['query_blocks']==b]==d['query_y'][d['query_blocks']==b])) for b in np.unique(d['query_blocks'])}}
 if mean_align:
  for v in result.values():v['query_mean_alignment']='Uses only the unlabeled query feature mean; applied inside calibration CV too. No query labels or extra neural data.'
 suffix='baseline_mean_aligned' if mean_align else 'baseline'
 save(ROOT/'evaluation'/f'{name}.{suffix}.json',result)
 return result

def conservative_tie_sensitivity(name,mean_align=False):
 """Alternative calibration-only tie rule; retain original results unchanged."""
 suffix='baseline_mean_aligned' if mean_align else 'baseline'
 path=ROOT/'evaluation'/f'{name}.{suffix}.json';result=json.loads(path.read_text());d=np.load(ROOT/'cases'/f'{name}.npz')
 for method,grid,cls in [('lda',[.1,.5,.9,1.,'auto'],ShrinkageLDA),('gnb',[1e-9,1e-4,.01,.1,1.],GaussianNB)]:
  if method not in result:continue
  scores=result[method]['cv_grid_scores'];ties=[i for i,s in enumerate(scores) if abs(s-max(scores))<1e-12]
  selected=max(ties,key=lambda i:float(grid[i]) if isinstance(grid[i],(int,float)) else -1.)
  params={'shrinkage' if method=='lda' else 'var_smoothing':grid[selected]}
  q=d['query_x']
  if mean_align:q=q-q.mean(axis=0)+d['cal_x'].mean(axis=0)
  with threadpool_limits(limits=1):pred=cls(**params).fit(d['cal_x'],d['cal_y']).predict(q)
  result[method+'_conservative_ties']={'parameters':params,'tie_indices':ties,'selection_policy':'Most regularization among calibration-CV ties; sensitivity added after initial results, never use evaluation scores for selection',
    'prediction':pred.tolist(),'accuracy':float(np.mean(pred==d['query_y'])),'correct':int(np.sum(pred==d['query_y']))}
 save(path,result);return result

def run(name):
 cli=os.environ.get('CODEX_APP_RUNTIME_CLI_PATH') or 'codex';sandbox=WORK/'phase2_model_sandbox';sandbox.mkdir(exist_ok=True)
 prompt=ROOT/'prompts'/f'{name}.txt';response=ROOT/'responses'/f'{name}.json';inv=ROOT/'responses'/f'{name}.invocation.json'
 if inv.exists() and json.loads(inv.read_text()).get('returncode')==0:print(name,'already complete',flush=True);return {'returncode':0}
 command=[cli,'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-C',str(sandbox),'-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(response),'-']
 data={'name':name,'model':'gpt-6.1-sol','reasoning_effort':'high','command':command,'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};save(inv,data)
 with prompt.open() as stdin,(ROOT/'responses'/f'{name}.events.jsonl').open('w') as stdout,(ROOT/'responses'/f'{name}.stderr.txt').open('w') as stderr:
  try:data['returncode']=subprocess.run(command,stdin=stdin,stdout=stdout,stderr=stderr,timeout=1800).returncode
  except subprocess.TimeoutExpired:data['returncode']=124
 data['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 if data['returncode']==0:
  save(inv,data);print(name,'exit',0,flush=True)
 else:
  event_file=ROOT/'responses'/f'{name}.events.jsonl'
  data['usage_limited']='usage limit' in event_file.read_text().lower() if event_file.exists() else False
  for path in (ROOT/'responses').glob(name+'.*'):path.unlink()
 return data

def score(name):
 meta=json.loads((ROOT/'cases'/f'{name}.json').read_text());d=np.load(ROOT/'cases'/f'{name}.npz');path=ROOT/'responses'/f'{name}.json'
 inv=json.loads(path.with_suffix('.invocation.json').read_text())
 if inv.get('returncode')!=0:raise ValueError('Incomplete/failed call '+name)
 text=path.read_text().strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 response=json.loads(text);items=response['predictions'];lookup={x['id']:x['word'] for x in items};assert len(items)==len(lookup)==len(meta['query_ids']);assert set(lookup)==set(meta['query_ids'])
 pred=np.array([meta['alias_tokens'].index(lookup[q]) for q in meta['query_ids']]);correct=pred==d['query_y'];k=int(correct.sum());n=len(pred);ci=binomtest(k,n).proportion_ci(method='exact')
 events=[json.loads(l) for l in path.with_suffix('.events.jsonl').read_text().splitlines() if l.strip()];tools=[e['item']['type'] for e in events if e.get('item') and e['item']['type'] not in ['agent_message','reasoning']];assert not tools,tools
 result={'name':name,'correct':k,'n':n,'accuracy':k/n,'ci95_trial_descriptive':[ci.low,ci.high],'prediction':pred.tolist(),'method_self_report':response.get('method'),
         'block_accuracy':{str(b):float(np.mean(correct[d['query_blocks']==b])) for b in np.unique(d['query_blocks'])},
         'per_word_accuracy':{w:float(np.mean(correct[d['query_y']==i])) for i,w in enumerate(WORDS) if np.any(d['query_y']==i)},
         'donor_word_accuracy':float(np.mean(pred==d['donor_y'])),'tool_calls':len(tools),'usage':[e['usage'] for e in events if e['type']=='turn.completed']}
 save(ROOT/'evaluation'/f'{name}.sol.json',result);print(name,k,n,flush=True);return result

def bounded_queue(names):
 """At most three active calls across this experiment; stop submitting on quota exhaustion."""
 todo=collections.deque(names);active={};stop=False
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while todo or active:
   external=0
   for path in (ROOT/'responses').glob('*.invocation.json'):
    try:inv=json.loads(path.read_text())
    except ValueError:continue
    if 'finished_utc' not in inv and inv['name'] not in active.values():external+=1
   while todo and len(active)+external<3 and not stop:
    name=todo.popleft();active[pool.submit(run,name)]=name
   if not active:
    if stop:break
    time.sleep(2);continue
   done,_=concurrent.futures.wait(active,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    name=active.pop(future);status=future.result()
    if status and status.get('usage_limited'):stop=True
   if stop and not active:break

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','score','baseline','queue']);parser.add_argument('names',nargs='+');args=parser.parse_args()
 if args.action=='queue':
  bounded_queue(args.names)
 else:
  for name in args.names:globals()[{'baseline':'baseline_case'}.get(args.action,args.action)](name)
