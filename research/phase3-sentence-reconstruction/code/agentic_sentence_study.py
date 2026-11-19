"""Tool-assisted Sol experiment: neural-only retrieval, no fitted supervised decoder."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,pathlib,hashlib,subprocess,os,time,sys,re
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,SEED,save
from sentence_study import words,distance

NAME='sentences_agentic_dev'
SANDBOX=WORK/'agentic_neural_sandbox'

def prepare():
 source=WORK/'sources/t12.2023.11.28_attempted_50words.mat';d=loadmat(source,simplify_cells=True);go=np.array(d['goTrialEpochs'],int)-1;blocks=d['blockNum'][go[:,0]]
 continuous=np.column_stack([np.asarray(d['binnedTX'],float),np.log1p(d['spikePow'])]);assert continuous.shape[1]==256
 # Generic label-free block correction; no target words enter the transform.
 for block in np.unique(d['blockNum']):
  idx=d['blockNum']==block;continuous[idx]-=continuous[idx].mean(0)
 train=np.flatnonzero(np.isin(blocks,[1,4,6]));test=np.flatnonzero(blocks==8)[:8];rng=np.random.default_rng(SEED);train=rng.permutation(train);test=rng.permutation(test)
 frames=[]
 for a,b in go:
  signal=continuous[a:b+1];frames.append(np.array([v.mean(0) for v in np.array_split(signal,max(1,int(round(len(signal)/5))))]))
 mu=np.concatenate([frames[i] for i in train]).mean(0);sc=np.concatenate([frames[i] for i in train]).std(0,ddof=1);sc[sc<1e-8]=1
 frames=[np.round((f-mu)/sc,3) for f in frames];texts=[words(s) for s in d['sentenceText']]
 SANDBOX.mkdir(exist_ok=True);ids=[f't{v:08x}' for v in rng.choice(2**32,len(test),replace=False)];calids=[f'c{i:03d}' for i in range(len(train))]
 arrays={cid:frames[i] for cid,i in zip(calids,train)};arrays.update({qid:frames[i] for qid,i in zip(ids,test)})
 np.savez_compressed(SANDBOX/'neural_inputs.npz',**arrays)
 info={'bin_ms_nominal':100,'features':'128 TX and 128 log1p spike-power channels (both available arrays); label-free within-block mean correction and calibration-only standardization. No projection/encoder.','calibration':[{'id':cid,'tokens':texts[i],'duration_seconds':round((go[i,1]-go[i,0]+1)*.02,2),'shape':list(frames[i].shape)} for cid,i in zip(calids,train)],'queries':[{'id':qid,'duration_seconds':round((go[i,1]-go[i,0]+1)*.02,2),'shape':list(frames[i].shape)} for qid,i in zip(ids,test)]}
 save(SANDBOX/'inputs.json',info)
 prompt=f'''You are the inference subject in a controlled neural-to-language reconstruction experiment. Reconstruct the actual ordered words of eight cued attempted-speech recordings. The vocabulary is the fifty words occurring in 120 calibration transcripts. Query sentences are unseen combinations, not calibration copies. No query word count, boundaries or target text is provided.

You may use numerical computation and inspect the supplied arrays to overcome the limitations of unaided reading of huge numeric prompts. All access is restricted to your current directory. Read inputs.json for calibration transcripts, random IDs and recording lengths. neural_inputs.npz contains the standardized neural trajectories keyed by those IDs; each row is approximately100 ms, 256 features. Python with NumPy/SciPy is available at {PYTHON}. You may write analysis files in this directory.

Scientific constraints: no browsing, network, delegation, files outside this directory, model fine-tuning, gradient updates, or fitting a separately trained supervised neural-to-word model/encoder. Do not train LDA, regression, neural networks, CTC, HMM emissions or class-labeled neural prototypes. Those would answer a different research question. Allowed: deterministic signal preprocessing; neural-only cosine/Euclidean similarities, nearest-recording/interval retrieval, unsupervised projections, dynamic time warping, and inference-time reasoning over the corresponding labeled calibration examples. Neural retrieval is context selection, not a source of ground truth. Use the examples themselves to infer recurring word relationships and their order. Do not substitute generic grammatical guesses for neural evidence. Be candid when uncertain, but return your best complete reconstruction for every query.

Try to identify local reusable patterns that appear in multiple different calibration utterances, rather than merely copying the transcript of a globally similar trial. Word timing and durations vary. You have autonomy over inference-time computation under the constraints above. No particular algorithm or answer is supplied. The tool trace will be audited.

Return ONLY a final JSON object with predictions (list of objects with id and tokens, lowercase words), method (what computations and neural evidence supported your inference), and limitations. Keep intermediate files as reproducible evidence. Do not read research reports, evaluation labels, other directories, or any other data source.'''
 (ROOT/'prompts'/f'{NAME}.txt').write_text(prompt);np.savez_compressed(ROOT/'cases'/f'{NAME}.npz',train=train,test=test)
 save(ROOT/'cases'/f'{NAME}.json',{'name':NAME,'query_ids':ids,'query_n':len(test),'calibration_n':len(train),'split':'development','input_sha256':{f:hashlib.sha256((SANDBOX/f).read_bytes()).hexdigest() for f in ['inputs.json','neural_inputs.npz']},'preprocessing':info['features'],'tools_permitted':'Neural-only retrieval/alignment and reasoning, no supervised model fitting','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
 print('Prepared',NAME,'calibration',len(train),'queries',len(test),'all256features')

def run():
 cli=os.environ['CODEX_APP_RUNTIME_CLI_PATH'];prompt=ROOT/'prompts'/f'{NAME}.txt';cmd=[cli,'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-C',str(SANDBOX),'-s','workspace-write','-c','approval_policy="never"','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(ROOT/'responses'/f'{NAME}.json'),'-'];start=time.time()
 with prompt.open() as stdin,(ROOT/'responses'/f'{NAME}.events.jsonl').open('w') as stdout,(ROOT/'responses'/f'{NAME}.stderr.txt').open('w') as stderr:
  try:ret=subprocess.run(cmd,stdin=stdin,stdout=stdout,stderr=stderr,timeout=1800).returncode
  except subprocess.TimeoutExpired:ret=124
 if ret==0:save(ROOT/'responses'/f'{NAME}.invocation.json',{'name':NAME,'model':'gpt-6.1-sol','reasoning_effort':'high','returncode':0,'wall_seconds':time.time()-start,'command':cmd,'prompt_sha256':hashlib.sha256(prompt.read_bytes()).hexdigest()});print('Completed',NAME)
 else:
  for p in (ROOT/'responses').glob(NAME+'.*'):p.unlink()
  print('No completed prediction')

def evaluate():
 case=np.load(ROOT/'cases'/f'{NAME}.npz');m=json.loads((ROOT/'cases'/f'{NAME}.json').read_text());d=loadmat(WORK/'sources/t12.2023.11.28_attempted_50words.mat',simplify_cells=True);text=(ROOT/'responses'/f'{NAME}.json').read_text().strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 result=json.loads(text);lookup={x['id']:x['tokens'] for x in result['predictions']};assert set(lookup)==set(m['query_ids']);rows=[]
 for q,i in zip(m['query_ids'],case['test']):
  truth=words(d['sentenceText'][i]);pred=lookup[q];rows.append({'id':q,'trial':int(i),'prediction':pred,'truth':truth,'edit_distance':distance(truth,pred)})
 result.update(name=NAME,n=len(rows),exact_sentences=sum(r['prediction']==r['truth'] for r in rows),word_error_rate=sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),rows=rows)
 save(ROOT/'evaluation'/f'{NAME}.sol.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':{'prepare':prepare,'run':run,'evaluate':evaluate}[sys.argv[1]]()
