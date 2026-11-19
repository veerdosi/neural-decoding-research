"""Frozen sentence-confirmation inputs, controls, and authentic Codex execution."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,hashlib,json,os,pathlib,shutil,subprocess,time
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,SEED,save
from sentence_study import words,distance

HELPERS=['retrieval.py','local_retrieval.py','local_dtw.py','show_local.py','edge_retrieval.py','inspect_intervals.py']

def prepare(name,source,cal_blocks,query_block,start,n,control=None,roi=False):
 d=loadmat(WORK/'sources'/source,variable_names=['binnedTX','spikePow','goTrialEpochs','blockNum','sentenceText','chanSets','chanSetNames','binSize'],simplify_cells=True);go=np.asarray(d['goTrialEpochs'],int)-1;block=d['blockNum'][go[:,0]];ch=np.arange(d['binnedTX'].shape[1])
 if roi:ch=np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')],int)-1
 cont=np.column_stack([np.asarray(d['binnedTX'][:,ch],float),np.log1p(d['spikePow'][:,ch])]);del d['binnedTX'],d['spikePow']
 for b in np.unique(d['blockNum']):
  idx=d['blockNum']==b;cont[idx]-=cont[idx].mean(0)
 frames=[];bin_ms=float(d['binSize'])
 for a,b in go:
  signal=cont[a:b+1];frames.append(np.array([v.mean(0) for v in np.array_split(signal,max(1,int(round(len(signal)*bin_ms/100))))]))
 train=np.flatnonzero(np.isin(block,cal_blocks));test=np.flatnonzero(block==query_block)[start:start+n];assert len(test)==n
 rng=np.random.default_rng(SEED);train=rng.permutation(train);test=rng.permutation(test);mu=np.concatenate([frames[i] for i in train]).mean(0);std=np.concatenate([frames[i] for i in train]).std(0,ddof=1);std[std<1e-8]=1;frames=[np.round((a-mu)/std,3) for a in frames]
 ids=[f't{v:08x}' for v in rng.choice(2**32,len(test),replace=False)];donor=test.copy()
 if control=='duration_shuffle':
  order=np.argsort([len(frames[i]) for i in test])
  for group in np.array_split(order,4):
   perm=rng.permutation(group);donor[perm]=test[np.roll(perm,1)]
  assert np.all(donor!=test)
 texts=[words(s) for s in d['sentenceText']];assert not set(map(tuple,[texts[i] for i in train]))&set(map(tuple,[texts[i] for i in test]))
 folder=WORK/'agentic_cases'/name;folder.mkdir(parents=True,exist_ok=True);calids=[f'c{i:03d}' for i in range(len(train))];arrays={cid:frames[i] for cid,i in zip(calids,train)}
 arrays.update({qid:(np.zeros_like(frames[i]) if control=='no_signal' else frames[i]) for qid,i in zip(ids,donor)})
 np.savez_compressed(folder/'neural_inputs.npz',**arrays)
 info={'bin_ms_nominal':100,'features':f'{len(ch)} TX and {len(ch)} log1p spike-power channels. Label-free full-block mean correction; calibration-only standardization; no supervised encoder.','calibration':[{'id':cid,'tokens':texts[i],'duration_seconds':round((go[i,1]-go[i,0]+1)*bin_ms/1000,2),'shape':list(frames[i].shape)} for cid,i in zip(calids,train)],'queries':[{'id':qid,'duration_seconds':round((go[i,1]-go[i,0]+1)*bin_ms/1000,2),'shape':list(frames[i].shape)} for qid,i in zip(ids,donor)]}
 save(folder/'inputs.json',info)
 helpers={}
 for f in HELPERS:
  s=(ROOT/'agentic_development_trace'/f).read_text().replace('np.ones(256)',"np.ones(raw[meta['calibration'][0]['id']].shape[1])")
  (folder/f).write_text(s);helpers[f]=hashlib.sha256(s.encode()).hexdigest()
 # Reproduce Sol's calibration-only PCA exactly, without running the exploratory
 # variants.py sweep, which is not needed to expose the selected retrieval.
 (folder/'prepare_pca.py').write_text("from retrieval import *\nX=np.concatenate([raw[c['id']] for c in meta['calibration']]).astype(float)\nmu=X.mean(0)\nw,v=np.linalg.eigh(np.cov(X.T));w=w[::-1];v=v[:,::-1]\nnp.savez('unsupervised_pca.npz',mean=mu,eigenvalues=w,eigenvectors=v)\n")
 python=PYTHON;prompt=f'''Reconstruct the actual ordered words of every query neural recording in inputs.json. All query sentences are new combinations absent from calibration. Use only the labeled calibration recordings and the query neural data in the current directory. No query words, word counts or boundaries are provided. The vocabulary consists of words appearing in calibration. Return your best complete reconstruction for every query.

This is a fresh independent inference call. You may use numerical tools. Python with NumPy/SciPy is {python}. neural_inputs.npz contains trajectories keyed by the random IDs in inputs.json; rows are approximately100 ms. Do not browse, use the network, delegate, or read files outside this directory. No evaluation targets or past predictions are present here.

A prior development call discovered a useful generic workflow: smoothing and calibration-only unsupervised PCA, removal of the first two components, cosine-based retrieval of neural intervals, and slope-constrained subsequence DTW. It interpreted overlapping retrieved calibration transcripts to infer reusable words and phrases rather than copying the closest whole sentence. The reusable numerical helpers here are that call's code, with only neural feature dimension generalized. No neural-to-word weights or word prototypes are supplied. prepare_pca.py prepares the calibration-only PCA; local_retrieval.py and local_dtw.py can generate evidence. Use OPENBLAS_NUM_THREADS=1 for these commands. You may adapt deterministic numerical computation if needed, but avoid unnecessary parameter sweeps. Record the evidence for your reconstruction.

Research boundary: use general-purpose numerical computation freely, including signal processing, PCA, smoothing, statistics, similarity search, retrieval, alignment and other justified algorithms over the supplied examples. No fine-tuning, separately trained or pre-existing specialized neural-to-language decoder/encoder, or external decoded labels/logits may do the decoding for you. All task associations must derive from the supplied calibration within this inference. Keep any data-derived numerical scoring or assembly procedure auditable. If a numerical procedure itself generates a word sequence, save that algorithm-only sequence separately from your final interpreted reconstruction. Distinguish what the tools determine from your own interpretation/composition. Inspect overlapping evidence to resolve word identity and timing. Keep console output concise; save bulky scores/arrays in local files. Do not replace neural evidence with generic grammatical guesses.

Return ONLY a final JSON with predictions (objects with id and tokens as lowercase words), method, and limitations. Keep reproducible analysis scripts and evidence in this directory.'''
 if control=='no_signal':prompt+='\nQuery activity is withheld: query arrays are zero placeholders with the original recording lengths. Predict from available calibration priors only; no query-specific neural evidence exists.\n'
 (ROOT/'prompts'/f'{name}.txt').write_text(prompt);np.savez_compressed(ROOT/'cases'/f'{name}.npz',train=train,test=test,donor=donor)
 save(ROOT/'cases'/f'{name}.json',{'name':name,'source':source,'source_sha256':hashlib.sha256((WORK/'sources'/source).read_bytes()).hexdigest(),'query_ids':ids,'calibration_n':len(train),'query_n':n,'control':control,'sandbox':str(folder),'calibration_blocks':cal_blocks,'query_block':query_block,'query_offset':start,'roi_only':roi,'channel_indices':ch.tolist(),'input_sha256':{f:hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in ['inputs.json','neural_inputs.npz']},'helper_sha256':helpers,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
 print(name,'cal',len(train),'query',n,'channels',len(ch),'control',control,flush=True)

def run(name):
 m=json.loads((ROOT/'cases'/f'{name}.json').read_text());prompt=ROOT/'prompts'/f'{name}.txt';cmd=[os.environ['CODEX_APP_RUNTIME_CLI_PATH'],'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-C',str(resolve_sandbox(m['sandbox'])),'-s','workspace-write','-c','approval_policy="never"','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(ROOT/'responses'/f'{name}.json'),'-'];before=time.time()
 with prompt.open() as stdin,(ROOT/'responses'/f'{name}.events.jsonl').open('w') as stdout,(ROOT/'responses'/f'{name}.stderr.txt').open('w') as stderr:
  try:ret=subprocess.run(cmd,stdin=stdin,stdout=stdout,stderr=stderr,timeout=3600).returncode
  except subprocess.TimeoutExpired:ret=124
 if ret==0:save(ROOT/'responses'/f'{name}.invocation.json',{'name':name,'model':'gpt-6.1-sol','reasoning_effort':'high','returncode':0,'wall_seconds':time.time()-before,'command':cmd,'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest()});print('Completed',name,flush=True)
 else:
  for p in (ROOT/'responses').glob(name+'.*'):p.unlink()
  print('No completed prediction',name,flush=True)

def evaluate(name):
 m=json.loads((ROOT/'cases'/f'{name}.json').read_text());c=np.load(ROOT/'cases'/f'{name}.npz');d=loadmat(WORK/'sources'/m['source'],variable_names=['sentenceText'],simplify_cells=True);text=(ROOT/'responses'/f'{name}.json').read_text().strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 out=json.loads(text);lookup={r['id']:r['tokens'] for r in out['predictions']};assert set(lookup)==set(m['query_ids']);rows=[]
 for q,i,donor in zip(m['query_ids'],c['test'],c['donor']):
  truth=words(d['sentenceText'][i]);dt=words(d['sentenceText'][donor]);pred=lookup[q];rows.append({'id':q,'trial':int(i),'donor_trial':int(donor),'prediction':pred,'truth':truth,'donor_truth':dt,'edit_distance':distance(truth,pred),'donor_edit_distance':distance(dt,pred)})
 out.update(name=name,n=len(rows),exact_sentences=sum(r['prediction']==r['truth'] for r in rows),word_error_rate=sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),donor_word_error_rate=sum(r['donor_edit_distance'] for r in rows)/sum(len(r['donor_truth']) for r in rows),rows=rows)
 save(ROOT/'evaluation'/f'{name}.sol.json',out);print(json.dumps({k:out[k] for k in ['name','n','exact_sentences','word_error_rate','donor_word_error_rate']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','evaluate']);p.add_argument('name');p.add_argument('--source',default='t12.2023.11.28_attempted_50words.mat');p.add_argument('--cal-blocks',nargs='+',type=int,default=[1,4,6]);p.add_argument('--query-block',type=int,default=8);p.add_argument('--start',type=int,default=8);p.add_argument('--n',type=int,default=16);p.add_argument('--control');p.add_argument('--roi',action='store_true');a=p.parse_args()
 if a.action=='prepare':prepare(a.name,a.source,a.cal_blocks,a.query_block,a.start,a.n,a.control,a.roi)
 elif a.action=='run':run(a.name)
 else:evaluate(a.name)
