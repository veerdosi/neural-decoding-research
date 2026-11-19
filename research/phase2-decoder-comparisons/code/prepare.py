"""Reproduce signal processing, calibration subsets, and conventional comparisons."""
import argparse, hashlib, json, re, urllib.request
import numpy as np
from scipy.io import loadmat
from experiment import *

HF='https://huggingface.co/datasets/raei/Kunz-2025-InnerSpeechMotorCortexImplicationsSpeechNeuroprostheses/resolve/main/'
FILES={
 't12.2023.08.15_imagined1stMotor_raw.mat':'isolatedVerbalBehaviors',
 't16.2024.03.04_imagined1stMotor_raw.mat':'isolatedVerbalBehaviors',
 't12.2024.04.11_interleavedVerbalBehaviors_raw.mat':'interleavedVerbalBehaviors'}

def download():
 (WORK/'sources').mkdir(parents=True,exist_ok=True)
 for file,folder in FILES.items():
  path=WORK/'sources'/file
  if not path.exists():urllib.request.urlretrieve(HF+folder+'/'+file,path)
  reference=PACKAGE/'source_files.json'
  if reference.exists():
   expected=json.loads(reference.read_text())[file]['sha256']
   assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,file

def interleaved():
 d=loadmat(WORK/'sources/t12.2024.04.11_interleavedVerbalBehaviors_raw.mat',simplify_cells=True)
 go=d['goTrialEpochs'].astype(int)-1;block=d['blockNum'][go[:,0]];ch=np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')],int)-1
 cont=np.column_stack([d['binnedTX'][:,ch],np.log1p(np.asarray(d['spikePow'][:,ch],float))]);binms=int(d['binSize'])
 means={b:cont[d['blockNum']==b].mean(axis=0) for b in np.unique(block)}
 x=np.array([cont[a+200//binms:a+1200//binms].mean(axis=0) for a in go[:,0]])
 xc=np.array([v-means[b] for v,b in zip(x,block)]);y=[];behavior=[]
 for t in d['trialCues'].astype(int)-1:
  cue=str(d['cueList'][t]);match=re.match(r'([a-z]+)',cue);word=match.group(1) if match else None
  y.append(WORDS.index(word) if word in WORDS else -1)
  behavior.append('imagined' if 'imagined' in cue else 'attempted' if 'attempted' in cue else 'listening' if word in WORDS else 'rest')
 y=np.array(y);behavior=np.array(behavior);split=np.unique(block)[5]
 np.savez_compressed(ROOT/'data/interleaved_exploratory.npz',raw=x,blockcenter=xc,labels=y,behavior=behavior,blocks=block)
 for correction,features in [('raw',x),('blockcenter',xc)]:
  for source in ['attempted','imagined','listening']:
   for target in ['attempted','imagined','listening']:
    train=np.flatnonzero((block<split)&(behavior==source));test=np.flatnonzero((block>=split)&(behavior==target));name=f'newsession_{correction}_{source}_to_{target}'
    a,b=transform(features[train],features[test],'raw')
    np.savez_compressed(ROOT/'cases'/f'{name}.npz',cal_raw=features[train],query_raw=features[test],cal_x=a,query_x=b,cal_y=y[train],query_y=y[test],cal_blocks=block[train],query_blocks=block[test],train=train,test=test,donor_y=y[test])
    save(ROOT/'cases'/f'{name}.json',{'name':name,'dataset':'t12_newsession','representation':correction,'n_per_word':10,'draw':0,'query_n':len(test),'evaluation_status':'new_session_exploratory_after_baseline_scoring','calibration_behavior':source,'evaluation_behavior':target})
    baseline_case(name)
 for rep in ['raw','blockcenter']:
  old=np.load(ROOT/'data/t12.npz');train=old['train'];test=np.flatnonzero((block>=split)&(behavior=='imagined'));name=f'crosssession_{rep}_no_recalibration'
  oldx=old[rep];newx=x if rep=='raw' else xc;a,b=transform(oldx[train],newx[test],'raw')
  np.savez_compressed(ROOT/'cases'/f'{name}.npz',cal_raw=oldx[train],query_raw=newx[test],cal_x=a,query_x=b,cal_y=old['labels'][train],query_y=y[test],cal_blocks=old['blocks'][train],query_blocks=block[test],train=train,test=test,donor_y=y[test])
  save(ROOT/'cases'/f'{name}.json',{'name':name,'dataset':'t12_crosssession','representation':rep,'n_per_word':14,'draw':0,'query_n':len(test),'evaluation_status':'new_session_exploratory_after_baseline_scoring'})
  baseline_case(name)

def crosssession_sol():
 source='crosssession_blockcenter_no_recalibration';name='dev_crosssession_blockcenter_sol'
 d=np.load(ROOT/'cases'/f'{source}.npz');meta=json.loads((ROOT/'cases'/f'{source}.json').read_text());rng=np.random.default_rng(SEED+777)
 tr=rng.permutation(len(d['cal_y']));qr=rng.permutation(len(d['query_y']));ids=[f'q{v:08x}' for v in rng.choice(2**32,len(qr),replace=False)]
 header=(ROOT/'prompts/dev_blockcenter_n14_sol.txt').read_text().split('Labeled calibration examples, in random order:\n')[0]
 text=header+'Labeled calibration examples, in random order:\n'
 for i in tr:text+=json.dumps({'word':WORDS[d['cal_y'][i]],'x':d['cal_x'][i].tolist()},separators=(',',':'))+'\n'
 text+='\nUnlabeled queries:\n'
 for q,i in zip(ids,qr):text+=json.dumps({'id':q,'x':d['query_x'][i].tolist()},separators=(',',':'))+'\n'
 arrays={k:d[k][tr] if k.startswith('cal_') or k=='train' else d[k][qr] if k.startswith('query_') or k in ['test','donor_y'] else d[k] for k in d.files}
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',**arrays)
 meta.update(name=name,control=None,query_ids=ids,alias_tokens=WORDS,prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
 save(ROOT/'cases'/f'{name}.json',meta);(ROOT/'prompts'/f'{name}.txt').write_text(text);baseline_case(name)

def all_inputs():
 assert not list((ROOT/'cases').glob('*.json')),'Use a fresh NEURAL_PHASE2_OUTPUT_DIR; never overwrite original inputs.'
 download();extract('t12',WORK/'sources/t12.2023.08.15_imagined1stMotor_raw.mat')
 for r in [1,2]:make_case(f'dev_raw_n14_repeat{r}',exact=True)
 for n in [2,4,7,14]:
  for draw in range(20 if n<14 else 1):make_case(f'dev_raw_n{n}_draw{draw}',n=n,draw=draw)
 for rep in ['raw','delay','delta','late','tx','sp','pca','pca_tail','other','blockcenter']:
  make_case(f'dev_{rep}_n14_mechanism',representation=rep)
 extract('t16',WORK/'sources/t16.2024.03.04_imagined1stMotor_raw.mat')
 for n in [2,7]:
  for draw in [0,1]:make_case(f'confirm_raw_n{n}_draw{draw}',dataset='t16',n=n,draw=draw)
 make_case('confirm_raw_n7_shuffled',dataset='t16',n=7,control='shuffled')
 make_case('confirm_raw_n7_no_signal',dataset='t16',n=7,control='no_signal')
 make_case('confirm_blockcenter_n7',dataset='t16',n=7,representation='blockcenter')
 for rep in ['blockcenter','delay','delta']:make_case(f'dev_{rep}_n14_sol',representation=rep)
 make_case('dev_blockcenter_n14_sol_repeat',representation='blockcenter');make_case('dev_raw_n14_alias',alias=True)
 for i in range(7):phase1_chunk(f'dev_raw_n14_chunk{i}',np.arange(i*14,(i+1)*14))
 for path in (ROOT/'cases').glob('*.json'):
  meta=json.loads(path.read_text())
  if meta.get('control') is None:baseline_case(meta['name'])
 interleaved();crosssession_sol()
 for path in (ROOT/'evaluation').glob('*.baseline.json'):conservative_tie_sensitivity(path.name.removesuffix('.baseline.json'))
 for n in [2,4,7,14]:
  for draw in range(20 if n<14 else 1):
   name=f'dev_raw_n{n}_draw{draw}';baseline_case(name,True);conservative_tie_sensitivity(name,True)
 for name in ['dev_raw_n14_repeat1','dev_raw_n14_repeat2']+[f'confirm_raw_n{n}_draw{i}' for n in [2,7] for i in [0,1]]:
  baseline_case(name,True);conservative_tie_sensitivity(name,True)
 rotation_case()
 print('Prepared deterministic inputs and comparison predictions. No Codex calls made.')

if __name__=='__main__':all_inputs()
