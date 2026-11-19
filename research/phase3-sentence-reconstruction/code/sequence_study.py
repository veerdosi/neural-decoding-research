from __future__ import annotations
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,hashlib,json,pathlib,os,subprocess,time
import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT=pathlib.Path(__file__).resolve().parents[1]
WORK=PROJECT/'local'
SOURCE=WORK/'sources/t12.2023.09.28_binnedTX_spikePow_20ms_speak.mat'
SEED=2026111803
for folder in ['data','cases','prompts','responses','evaluation']:(ROOT/folder).mkdir(exist_ok=True)
def save(path,value):path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def prepare():
 d=loadmat(SOURCE,simplify_cells=True);go=d['goTrialEpochs'].astype(int)-1
 cues=[str(x).split('_') for x in d['cueList']]
 bits=np.array([[int(w=='up') for w in cues[int(y)-1]] for y in d['trialCues']])
 code=bits@np.array([4,2,1]);ch=np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')],int)-1
 cont=np.column_stack([np.asarray(d['binnedTX'][:,ch],float),np.asarray(d['spikePow'][:,ch],float)])
 # Source data already have block mean correction. Power values can be negative;
 # retain them directly rather than taking log1p.
 frames=[]
 for a,e in go:
  start=a+5;stop=a+205;assert stop<=e+1
  frames.append(cont[start:stop].reshape(20,10,128).mean(axis=1))
 frames=np.asarray(frames);cut=len(go)//2
 schemes={'homogeneous':[0,7],'even_parity':[0,3,5,6]}
 manifest={}
 for scheme,classes in schemes.items():
  n=6 if scheme=='homogeneous' else 3
  train=np.concatenate([np.flatnonzero((code==c)&(np.arange(len(go))<cut))[:n] for c in classes])
  assert len(train)==n*len(classes)
  dev_classes=[c for c in range(8) if c not in classes]
  dev=np.concatenate([np.flatnonzero((code==c)&(np.arange(len(go))<cut))[:2] for c in dev_classes])
  confirmation=np.flatnonzero((np.arange(len(go))>=cut)&~np.isin(code,classes))
  assert not set(train)&set(dev) and not set(train)&set(confirmation)
  assert not set(code[train])&set(code[dev]) and not set(code[train])&set(code[confirmation])
  manifest[scheme]={'calibration_trial_indices':train.tolist(),'development_trial_indices':dev.tolist(),
   'reserved_trial_indices':confirmation.tolist(),'calibration_sequences':[' '.join('up' if b else 'right' for b in [(c>>2)&1,(c>>1)&1,c&1]) for c in classes],
   'n_calibration':len(train),'n_development':len(dev),'n_reserved':len(confirmation)}
 np.savez_compressed(ROOT/'data/sequence_features.npz',frames=frames,bits=bits,code=code,go=go)
 save(ROOT/'protocol.json',{'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  'array':'i6v','channel_indices_zero_based':ch.tolist(),'source_correction':'Authors supplied block-mean-corrected features; no block metadata in this file',
  'time_ms_after_go':[100,4100],'frame_ms':200,'features':'64 crossing means + 64 mean-centered spike-power means, no supervised encoder',
  'chronological_reserve_start_trial':cut,'schemes':manifest,'model':'gpt-6.1-sol','reasoning_effort':'high'})
 print(json.dumps({s:{k:v for k,v in m.items() if not k.endswith('indices')} for s,m in manifest.items()},indent=2))

def representation(cal,query,components=12):
 mu=cal.reshape(-1,128).mean(axis=0);sc=cal.reshape(-1,128).std(axis=0,ddof=1);sc[sc<1e-8]=1
 a=(cal-mu)/sc;b=(query-mu)/sc
 _,singular,vt=np.linalg.svd(a.reshape(-1,128),full_matrices=False)
 if components:
  pcs=vt[:components].copy()
  for p in pcs:
   if p[np.argmax(np.abs(p))]<0:p*=-1
  a=a@pcs.T;b=b@pcs.T
 return np.round(a,2),np.round(b,2),float(np.sum(singular[:components]**2)/np.sum(singular**2)) if components else 1.

def make_case(name,scheme='homogeneous',split='development',components=12,control=None):
 p=json.loads((ROOT/'protocol.json').read_text());d=np.load(ROOT/'data/sequence_features.npz');m=p['schemes'][scheme]
 train=np.asarray(m['calibration_trial_indices']);test=np.asarray(m['development_trial_indices' if split=='development' else 'reserved_trial_indices'])
 rng=np.random.default_rng(SEED);train=rng.permutation(train);test=rng.permutation(test)
 a,b,variance=representation(d['frames'][train],d['frames'][test],components)
 donor=test.copy()
 if control=='shuffled_within_word_bag':
  counts=d['bits'][test].sum(axis=1)
  for count in np.unique(counts):
   idx=np.flatnonzero(counts==count);donor[idx]=rng.permutation(test[idx])
  _,b,_=representation(d['frames'][train],d['frames'][donor],components)
 if control=='time_reversed':b=b[:,::-1].copy()
 ids=[f'q{v:08x}' for v in rng.choice(2**32,len(test),replace=False)]
 text=(
 'You are the inference subject in a controlled neural-to-language reconstruction experiment. '
 'Use only the examples and recordings below. Do not browse, read files, use tools, or delegate. '
 'A participant attempted to speak THREE consecutive words. Each word is either up or right. '
 'Reconstruct the actual ordered three-word utterance for every query. Output token sequences, not class IDs. '
 'Any of the eight possible word orders is allowed. Do not force output label counts. '
 'Labels and query order contain no information. Query IDs are random. '
 'All query ground truth is hidden. Some word combinations may not occur among the calibration utterances; '
 'infer reusable word-related temporal patterns rather than copying whole known phrases. '
 'Calibration and query rows are consecutive 200-ms windows from 100 to 4100 ms after the go cue. '
 'Speech onset and word timing may vary: use temporal patterns and calibration evidence to infer order. '
 'Each row has '+str(a.shape[-1])+' numerical coordinates. '
 +('These are unwhitened principal components of 128 calibration-standardized neural features. PCA uses only calibration inputs, without labels; coordinates match across recordings. ' if components else 'The first 64 are threshold-crossing means and the last 64 spike-band-power means. Every dimension uses calibration-only standardization. ')
 +'No neural-to-language model has been fitted externally. Learn the association only from the labeled examples. '
 'Return ONLY JSON with predictions (list of objects with id and tokens, exactly three strings up/right per query) '
 'and method (a short account of the evidence and main uncertainty).\n\nCalibration utterances and trajectories:\n')
 for t,x in zip(train,a):
  text+=json.dumps({'tokens':['up' if b else 'right' for b in d['bits'][t]],'trajectory':x.tolist()},separators=(',',':'))+'\n'
 text+='\nUnlabeled query trajectories:\n'
 for q,x in zip(ids,b):
  row={'id':q}
  if control!='no_signal':row['trajectory']=x.tolist()
  text+=json.dumps(row,separators=(',',':'))+'\n'
 if control=='no_signal':text+='\nQuery neural trajectories are withheld. Still predict each sequence from the available priors.\n'
 (ROOT/'prompts'/f'{name}.txt').write_text(text)
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',cal=a,query=b,train=train,test=test,ycal=d['bits'][train],y=d['bits'][test],donor_y=d['bits'][donor])
 meta={'name':name,'scheme':scheme,'split':split,'components':components,'control':control,'query_ids':ids,
  'calibration_n':len(train),'query_n':len(test),'pca_variance_retained':variance,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
 save(ROOT/'cases'/f'{name}.json',meta);print(name,'cal',len(train),'query',len(test),'prompt bytes',len(text),'variance',round(variance,3))

def run(name):
 prompt=ROOT/'prompts'/f'{name}.txt';out=ROOT/'responses'/f'{name}.json';sandbox=WORK/'sequence_model_sandbox';sandbox.mkdir(exist_ok=True)
 cmd=[os.environ.get('CODEX_APP_RUNTIME_CLI_PATH','codex'),'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-C',str(sandbox),'-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(out),'-']
 before=time.time();meta={'name':name,'model':'gpt-6.1-sol','reasoning_effort':'high','command':cmd,'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 with prompt.open() as stdin,(ROOT/'responses'/f'{name}.events.jsonl').open('w') as stdout,(ROOT/'responses'/f'{name}.stderr.txt').open('w') as stderr:
  try:ret=subprocess.run(cmd,stdin=stdin,stdout=stdout,stderr=stderr,timeout=1800).returncode
  except subprocess.TimeoutExpired:ret=124
 if ret==0:
  meta.update(returncode=ret,wall_seconds=time.time()-before);save(ROOT/'responses'/f'{name}.invocation.json',meta);print(name,'completed',flush=True)
 else:
  for path in (ROOT/'responses').glob(name+'.*'):path.unlink()
  print(name,'no completed prediction',flush=True)

def evaluate(name):
 m=json.loads((ROOT/'cases'/f'{name}.json').read_text());d=np.load(ROOT/'cases'/f'{name}.npz');path=ROOT/'responses'/f'{name}.json';text=path.read_text().strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 out=json.loads(text);rows=out['predictions'];lookup={r['id']:r['tokens'] for r in rows};assert len(rows)==len(lookup)==len(m['query_ids']);assert set(lookup)==set(m['query_ids'])
 assert all(len(v)==3 and all(t in ['up','right'] for t in v) for v in lookup.values())
 pred=np.array([[int(t=='up') for t in lookup[q]] for q in m['query_ids']]);correct=pred==d['y']
 events=[json.loads(l) for l in path.with_suffix('.events.jsonl').read_text().splitlines() if l.strip()]
 assert not [e['item']['type'] for e in events if e.get('item') and e['item']['type'] not in ['agent_message','reasoning']]
 result={'name':name,'n':len(pred),'exact_sequence_correct':int(correct.all(axis=1).sum()),'exact_sequence_accuracy':float(correct.all(axis=1).mean()),
  'token_accuracy':float(correct.mean()),'position_accuracy':correct.mean(axis=0).tolist(),'prediction_bits':pred.tolist(),'method_self_report':out.get('method'),
  'donor_exact_accuracy':float(np.mean((pred==d['donor_y']).all(axis=1))),'usage':[e['usage'] for e in events if e['type']=='turn.completed']}
 save(ROOT/'evaluation'/f'{name}.sol.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','make','run','evaluate']);p.add_argument('--name');p.add_argument('--scheme',default='homogeneous');p.add_argument('--split',default='development');p.add_argument('--components',type=int,default=12);p.add_argument('--control');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='make':make_case(a.name,a.scheme,a.split,a.components,a.control)
 elif a.action=='run':run(a.name)
 else:evaluate(a.name)
