"""Exploratory novel-sentence reconstruction from open-loop neural recordings."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,re,hashlib,sys,pathlib
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,SEED,save,run

def words(s):return re.findall('[a-z]+',str(s).lower())

def prepare():
 source=WORK/'sources/t12.2023.11.28_attempted_50words.mat';d=loadmat(source,simplify_cells=True)
 go=np.asarray(d['goTrialEpochs'],int)-1;delay=np.asarray(d['delayTrialEpochs'],int)-1
 ch=np.asarray(d['chanSets'][list(d['chanSetNames']).index('i6v')],int)-1
 continuous=np.column_stack([d['binnedTX'][:,ch],np.log1p(d['spikePow'][:,ch])]);blocks=d['blockNum'][go[:,0]]
 text=[words(s) for s in d['sentenceText']];train=np.flatnonzero(np.isin(blocks,[1,4,6]));dev=np.flatnonzero(blocks==8)[:8];reserved=np.flatnonzero(blocks==10)
 assert len(train)==120 and len(dev)==8 and len(reserved)==40
 assert not set(map(tuple,[text[i] for i in train]))&set(map(tuple,[text[i] for i in dev]))
 assert all(not bool(np.size(v)) for v in d['decodedSentenceText']) # open-loop, no specialized decoder feedback
 trajectories=[]
 for i,(a,b) in enumerate(go):
  baseline=continuous[max(delay[i,0],a-25):a].mean(0)
  signal=continuous[a:b+1]-baseline
  # Deterministic relative-time interpolation; no word boundaries/labels used.
  trajectories.append(np.column_stack([np.interp(np.linspace(0,len(signal)-1,24),np.arange(len(signal)),signal[:,j]) for j in range(128)]))
 trajectories=np.asarray(trajectories);mu=trajectories[train].reshape(-1,128).mean(0);scale=trajectories[train].reshape(-1,128).std(0,ddof=1);scale[scale<1e-8]=1
 z=(trajectories-mu)/scale;_,sv,vt=np.linalg.svd(z[train].reshape(-1,128),full_matrices=False);pcs=vt[:32].copy()
 for p in pcs:
  if p[np.argmax(abs(p))]<0:p*=-1
 feature=np.round(z@pcs.T,2)
 np.savez_compressed(ROOT/'data/sentence_features.npz',features=feature,train=train,development=dev,reserved=reserved,go=go,blocks=blocks,raw_trajectories=trajectories)
 save(ROOT/'data/sentence_text.json',text)
 save(ROOT/'sentence_protocol.json',{'source':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'array':'i6v','source_open_loop':True,'calibration_blocks':[1,4,6],'n_calibration':120,'development_indices':dev.tolist(),'reserved_indices':reserved.tolist(),'preprocessing':'64 TX and 64 log1p SP; subtract each trial last 500 ms of pre-go delay; calibration-only scaling/PCA32; linearly interpolate go interval to24 frames','pca_variance':float(sum(sv[:32]**2)/sum(sv**2)),'calibration_vocabulary':sorted(set(w for i in train for w in text[i])),'test_sentences_unseen':True,'not_used':'decodedSentenceText; trained neural decoder; target word counts or text; query-conditioned labeled retrieval'})
 rng=np.random.default_rng(SEED);train=rng.permutation(train);test=rng.permutation(dev);ids=[f's{v:08x}' for v in rng.choice(2**32,len(test),replace=False)]
 prompt=('Reconstruct attempted spoken sentences from neural activity by learning ONLY from the calibration examples below. Do not use tools, files, browsing, or delegates. No neural-to-language model has been fitted externally. '
 'There are 120 labeled calibration sentences from one participant; query sentences were not among them. The vocabulary is limited to the 50 words occurring in calibration, but the word combinations and lengths vary. Generate the actual ordered words, not a nearest calibration sentence. Do not assume each query has the same length. '
 'Each trajectory has 24 consecutive relative-time samples spanning the actual go interval. Coordinates are 32 unwhitened PCA components of i6v threshold crossings and log spike-band power after subtracting a label-free pre-go baseline. Scaling/PCA use calibration only and no labels. Time alignment is not known, so infer relationships between neural dynamics and words from sentence-level examples. '
 'Duration is the measured go interval, not the number of words. IDs and row order are randomized; no target text, word boundaries, or target word counts are given. '
 'Return ONLY JSON with predictions, a list of objects with id and tokens (list of lowercase words), and method, a brief account of uncertainty.\nCalibration:\n')
 for i in train:prompt+=json.dumps({'tokens':text[i],'duration_seconds':round((go[i,1]-go[i,0]+1)*.02,2),'trajectory':feature[i].tolist()},separators=(',',':'))+'\n'
 prompt+='\nUnlabeled queries:\n'
 for q,i in zip(ids,test):prompt+=json.dumps({'id':q,'duration_seconds':round((go[i,1]-go[i,0]+1)*.02,2),'trajectory':feature[i].tolist()},separators=(',',':'))+'\n'
 name='sentences_pc32_dev';(ROOT/'prompts'/f'{name}.txt').write_text(prompt)
 np.savez_compressed(ROOT/'cases'/f'{name}.npz',train=train,test=test)
 save(ROOT/'cases'/f'{name}.json',{'name':name,'query_ids':ids,'query_n':len(test),'calibration_n':len(train),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'split':'development','task':'unseen sentences'})
 print('sentence prompt bytes',len(prompt),'PCA variance',sum(sv[:32]**2)/sum(sv**2))

def distance(a,b):
 d=np.zeros((len(a)+1,len(b)+1),int);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(a[i-1]!=b[j-1]))
 return int(d[-1,-1])

def evaluate(name='sentences_pc32_dev'):
 meta=json.loads((ROOT/'cases'/f'{name}.json').read_text());case=np.load(ROOT/'cases'/f'{name}.npz');truth=json.loads((ROOT/'data/sentence_text.json').read_text());raw=(ROOT/'responses'/f'{name}.json').read_text().strip()
 if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0]
 response=json.loads(raw);lookup={r['id']:r['tokens'] for r in response['predictions']};assert set(lookup)==set(meta['query_ids'])
 rows=[{'id':q,'trial':int(i),'prediction':lookup[q],'truth':truth[i],'edit_distance':distance(truth[i],lookup[q])} for q,i in zip(meta['query_ids'],case['test'])]
 result={'name':name,'n':len(rows),'exact_sentences':sum(r['prediction']==r['truth'] for r in rows),'word_error_rate':sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),'rows':rows,'method_self_report':response.get('method')}
 save(ROOT/'evaluation'/f'{name}.sol.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':run('sentences_pc32_dev')
 else:evaluate()
