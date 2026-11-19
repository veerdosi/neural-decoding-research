"""Three discovery probes: visual inference, neural-symbol induction, unseen words."""
from __future__ import annotations
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,datetime,hashlib,json,os,pathlib,re,shutil,subprocess,time,textwrap
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from PIL import Image,ImageDraw,ImageFont
from sequence_study import ROOT,WORK,SEED,save
from sentence_study import words,distance

BASE=ROOT/'discovery_probes';BASE.mkdir(exist_ok=True)
HELD_WORDS=['family','computer','comfortable','outside','glasses','music','success','hungry']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_case(name,prompt,meta):
 folder=BASE/name;folder.mkdir(exist_ok=True)
 (folder/'prompt.txt').write_text(prompt);meta.update(name=name,model='gpt-6.1-sol',reasoning_effort='high',prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
 save(folder/'case.json',meta)

def t16_inputs(source_case='sentences_t16_fresh_dev'):
 src=resolve_sandbox(WORK/'agentic_cases'/source_case);meta=json.loads((src/'inputs.json').read_text());raw=np.load(src/'neural_inputs.npz');old=json.loads((ROOT/'cases'/f'{source_case}.json').read_text());case=np.load(ROOT/'cases'/f'{source_case}.npz')
 return meta,{k:raw[k] for k in raw.files},old,case

def basis(raw,ids,n=32):
 x=np.concatenate([gaussian_filter1d(raw[k],1,axis=0) for k in ids]);mu=x.mean(0);_,s,vt=np.linalg.svd(x-mu,full_matrices=False);v=vt[:n].copy()
 for p in v:
  if p[np.argmax(abs(p))]<0:p*=-1
 scale=np.std((x-mu)@v.T,axis=0,ddof=1);scale[scale<1e-8]=1
 return {k:(gaussian_filter1d(a,1,axis=0)-mu)@v.T/scale for k,a in raw.items()},mu,v,scale

def font(size):
 try:return ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',size)
 except OSError:return ImageFont.load_default(size=size)

def atlas(rows,values,out,title,limit):
 from matplotlib import colormaps
 w,h=1760,1020;im=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(im);draw.text((25,12),title,fill='black',font=font(26));cw,ch=870,240
 for j,row in enumerate(rows):
  left=10+(j%2)*880;top=60+(j//2)*ch;label=row['id']
  if 'tokens' in row:label+='  |  '+' '.join(row['tokens'])
  lines=textwrap.wrap(label,width=64)
  for z,line in enumerate(lines[:2]):draw.text((left+8,top+z*27),line,fill='black',font=font(23))
  a=values[row['id']];t=np.linspace(0,len(a)-1,48);b=np.column_stack([np.interp(t,np.arange(len(a)),a[:,k]) for k in range(16)]).T
  rgb=(colormaps['RdBu_r'](np.clip((b+limit)/(2*limit),0,1))[:,:,:3]*255).astype(np.uint8);heat=Image.fromarray(rgb).resize((810,128),Image.Resampling.NEAREST);im.paste(heat,(left+42,top+61))
  draw.text((left+7,top+63),'1',fill='#555555',font=font(16));draw.text((left+2,top+165),'16',fill='#555555',font=font(16));draw.text((left+42,top+195),f'0  ->  {row["duration_seconds"]:.2f} s     /  same 16 PCA coordinates, relative time',fill='#444444',font=font(19))
 im.save(out)

def prepare_visual():
 name='visual_dynamics_t16';folder=BASE/name;folder.mkdir(exist_ok=True);meta,raw,old,case=t16_inputs();ids=[c['id'] for c in meta['calibration']]
 a,mu,v,scale=basis(raw,ids,16);limit=float(np.quantile(np.abs(np.concatenate([a[k] for k in ids])),.98));images=[]
 for i in range(0,len(ids),8):
  p=folder/f'calibration_{i//8+1:02}.png';atlas(meta['calibration'][i:i+8],a,p,f'CALIBRATION {i+1}–{min(i+8,len(ids))} / labelled whole sentences',limit);images.append(p.name)
 p=folder/'queries.png';atlas(meta['queries'],a,p,'QUERY RECORDINGS / target words hidden',limit);images.append(p.name)
 np.savez_compressed(folder/'representation_for_analysis.npz',**a,mean=mu,axes=v,scale=scale)
 prompt='''Decode the actual ordered words of the eight query neural recordings using the attached visual calibration examples. Each calibration panel pairs one whole attempted-speech sentence with its neural trajectory. All panels use the same16 calibration-only PCA coordinates, consistent color scaling and48 relative-time samples. Rows are coordinates; columns progress through the measured go interval. Positive values are red and negative blue; PCA/standardization use calibration only, without word labels. Word boundaries and query word counts are not provided. Query sentences are not complete copies of calibration sentences. Infer recurring visual dynamics and their relationship to words, allowing timing changes.\n\nThis probe tests direct VISUAL in-context learning, not numerical nearest-recording lookup or DTW. Do not use tools, browse, open files, run code or delegate; reason from the attached images only. Calibration images are labelled; the final image contains eight random query IDs with no target text. No neural decoder was trained. Return only JSON with predictions (objects with id and tokens as lowercase words) and method (what visual evidence supports the inference and uncertainty). Do not invent a successful result if the plots are insufficient, but give a best reconstruction for each query.'''
 write_case(name,prompt,{'mechanism':'Direct multimodal visual pattern induction','hypothesis':'Visual temporal motifs let the frozen model infer reusable word structure without explicit neural retrieval.','competing_explanation':'The compression may discard word information or visual ICL may be too weak for continuous sentences.','images':images,'query_ids':old['query_ids'],'calibration_n':80,'query_n':8,'source_case':'sentences_t16_fresh_dev','evaluation_role':'Development probe, same8 recordings; no untouched-confirmation claim','feature_transform':'Calibration-only smoothing/PCA16 and per-PC scaling, 48-point relative-time rendering','image_sha256':{f:digest(folder/f) for f in images},'tool_access':False})
 print('Prepared',name,len(images),'images',flush=True)

def compress(seq):
 result=[]
 for s in seq:
  token=f'S{int(s):02d}'
  if result and result[-1][0]==token:result[-1][1]+=1
  else:result.append([token,1])
 return ' '.join(f'{s}:{n}' for s,n in result)

def prepare_symbols(name='neural_alphabet_t16',source_case='sentences_t16_fresh_dev'):
 folder=BASE/name;folder.mkdir(exist_ok=True);meta,raw,old,case=t16_inputs(source_case);ids=[c['id'] for c in meta['calibration']]
 a,mu,v,scale=basis(raw,ids,32);a={k:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8) for k,x in a.items()}
 km=KMeans(n_clusters=64,n_init=10,random_state=SEED).fit(np.concatenate([a[k] for k in ids]));perm=np.random.default_rng(SEED+5).permutation(64);codes={k:perm[km.predict(x)] for k,x in a.items()}
 prompt='''Infer an unfamiliar neural symbol system from paired examples, then reconstruct the actual ordered words of eight queries. The participant attempted speech. Neural frames were smoothed, projected and clustered using CALIBRATION INPUTS ONLY, with no transcript/word labels used to learn clusters. Each arbitrary symbol S00..S63 is a cluster of neural activity, not a known letter/phoneme or a word classifier output. The number after a colon is consecutive approximately100-ms frames in that state. Symbol IDs were randomly permuted consistently. This is a parallel corpus of neural-symbol strings and whole sentence labels. Word boundaries, query lengths in words and target text are hidden.\n\nLook for reusable symbolic motifs, context-dependent patterns and composition. This is a symbolic induction/cryptanalysis probe, not nearest-neural-recording retrieval or DTW. Do not use tools, files, code, browsing or delegates. Reason from the symbol strings below only. Generate the actual words, not a calibration class. Return only JSON with predictions (id and lowercase tokens), method and uncertainty.\n\nCALIBRATION:\n'''
 for c in meta['calibration']:prompt+=json.dumps({'id':c['id'],'tokens':c['tokens'],'neural_symbols':compress(codes[c['id']])},separators=(',',':'))+'\n'
 prompt+='\nQUERIES:\n'
 for q in meta['queries']:prompt+=json.dumps({'id':q['id'],'neural_symbols':compress(codes[q['id']])},separators=(',',':'))+'\n'
 np.savez_compressed(folder/'symbol_codes_for_analysis.npz',**codes,centroids=km.cluster_centers_,mean=mu,axes=v,scale=scale,symbol_permutation=perm)
 prompt=prompt.replace('eight queries',f'{len(meta["queries"])} queries')
 write_case(name,prompt,{'mechanism':'Unsupervised neural alphabet + in-context grammar induction','hypothesis':'Language-like discrete sequences may expose structure that general symbolic reasoning learns more effectively than raw-vector arithmetic.','competing_explanation':'Unlabeled clustering may reflect generic task phase instead of reusable linguistic units.','query_ids':old['query_ids'],'calibration_n':len(ids),'query_n':len(meta['queries']),'source_case':source_case,'clusters':64,'evaluation_role':('Chronological reserved cohort, frozen representation and unchanged inference instructions' if source_case=='sentences_t16_confirm' else 'Development probe, same8 recordings'),'tool_access':False})
 print('Prepared',name,'prompt bytes',len(prompt),flush=True)

def prepare_phonetic():
 name='unseen_word_phonetics_t12';folder=BASE/name;folder.mkdir(exist_ok=True);source=WORK/'sources/willett2023_tuning/t12.2022.05.03_fiftyWordSet.mat'
 d=loadmat(source,variable_names=['tx1','spikePow','blockNum','goTrialEpochs','trialCues','cueList'],simplify_cells=True);go=d['goTrialEpochs'].astype(int)-1;bl=d['blockNum'][go[:,0]];labels=[str(d['cueList'][int(i)-1]).lower() for i in d['trialCues']]
 train=np.array([i for i,w in enumerate(labels) if bl[i] in [2,4] and w not in HELD_WORDS+['do nothing']]);test=np.array([i for i,w in enumerate(labels) if bl[i]==18 and w in HELD_WORDS]);assert len(train)==168 and len(test)==16
 continuous=np.column_stack([d['tx1'][:,:128].astype(float),np.log1p(d['spikePow'][:,:128])]);del d['tx1'],d['spikePow']
 for b in np.unique(d['blockNum']):
  idx=d['blockNum']==b;continuous[idx]-=continuous[idx].mean(0)
 def frame(i):
  start,end=go[i];sig=continuous[start:end+1];return np.array([p.mean(0) for p in np.array_split(sig,max(1,round(len(sig)/5)))])
 raw={int(i):frame(i) for i in np.r_[train,test]};x=np.concatenate([raw[int(i)] for i in train]);mu=x.mean(0);std=x.std(0,ddof=1);std[std<1e-8]=1
 rng=np.random.default_rng(SEED);train=rng.permutation(train);test=rng.permutation(test);calids=[f'c{i:03d}' for i in range(len(train))];qids=[f'q{v:08x}' for v in rng.choice(2**32,len(test),replace=False)]
 arrays={key:np.round((raw[int(i)]-mu)/std,3) for key,i in zip(calids,train)};arrays.update({key:np.round((raw[int(i)]-mu)/std,3) for key,i in zip(qids,test)});np.savez_compressed(folder/'neural_inputs.npz',**arrays)
 info={'bin_ms_nominal':100,'features':'128 area6v crossing channels +128 log1p spike-power channels; generic label-free block centering and calibration-only scaling. No audio/decoder outputs.','calibration':[{'id':key,'word':labels[int(i)],'shape':list(arrays[key].shape)} for key,i in zip(calids,train)],'queries':[{'id':key,'shape':list(arrays[key].shape)} for key in qids]};save(folder/'inputs.json',info)
 shutil.copy(WORK/'sources/cmudict.dict',folder/'cmudict.dict')
 prompt=f'''Investigate phonological generalization from neural activity. Each query is an attempted SINGLE English word, but its whole word type was NEVER supplied as a calibration example. No query vocabulary/menu is given. Calibration words are examples, not an exhaustive set of possible outputs. Infer any English word; optionally report its ARPABET phonemes as well.\n\nUse general-purpose intelligence and computation to infer structure shared across known words—for example, phonetic/articulatory relationships or a generative account of how word parts affect neural dynamics. This is fundamentally an unseen-word transfer problem, not known-word nearest-neighbor classification. Whole-word lookup cannot solve it. You may use any justified general numerical tools; temporal alignment is allowed as a component if it supports a different compositional/phonological mechanism. Do not load/fine-tune a specialized neural-to-language decoder or pre-existing neural encoder. All neural associations must come from these calibration examples. Document any fitted numerical relationships and their algorithm-only predictions separately from your own conclusions.\n\nRead inputs.json and neural_inputs.npz in this directory. They contain168 neural word examples covering42 known word types, and16 query recordings of unknown word types. Each row is approximately100 ms. Data are binned threshold crossings and log spike power, not audio. cmudict.dict is a generic English pronunciation dictionary with NO neural information or query targets. It may help define linguistic hypotheses. Python with NumPy/SciPy is at {PYTHON}. Access only this directory; no network, outside files or delegates. Keep numerical output concise and save bulky analysis locally.\n\nForm a concrete hypothesis, analyze contrasts among calibration words, and make best predictions for all random query IDs. Do not silently restrict output to the known calibration words. Return final JSON with predictions (objects with id, word, and optional phonemes), method, and limitations. Preserve evidence and any algorithm-only outputs in this directory.'''
 # Evaluation labels/indices are outside the subject's directory.
 save(BASE/f'{name}_ground_truth.json',{'source':str(source),'source_sha256':digest(source),'calibration_trials':train.tolist(),'query_trials':test.tolist(),'query_ids':qids,'truth_words':[labels[int(i)] for i in test],'held_word_types':HELD_WORDS})
 write_case(name,prompt,{'mechanism':'Phonological structure and generative transfer to word types absent from calibration','hypothesis':'Shared articulatory/phonetic structure can support novel lexical decoding rather than choosing known whole-word labels.','competing_explanation':'Neural word patterns may be context-dependent or insufficiently constrained by these lexical examples.','query_ids':qids,'calibration_n':168,'calibration_word_types':42,'query_n':16,'tool_access':True,'source_sha256':digest(source),'input_hashes':{p.name:digest(p) for p in [folder/'inputs.json',folder/'neural_inputs.npz',folder/'cmudict.dict']}})
 print('Prepared',name,'cal',len(train),'query',len(test),flush=True)

def run(name):
 folder=BASE/name;m=json.loads((folder/'case.json').read_text());out=ROOT/'responses'/f'{name}.json';cmd=[os.environ['CODEX_APP_RUNTIME_CLI_PATH'],'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','-C',str(folder),'-s','workspace-write','-c','approval_policy="never"','-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','-o',str(out)]
 for f in m.get('images',[]):cmd.extend(['-i',str(folder/f)])
 cmd.append('-');before=time.time()
 with (folder/'prompt.txt').open() as stdin,(ROOT/'responses'/f'{name}.events.jsonl').open('w') as stdout,(ROOT/'responses'/f'{name}.stderr.txt').open('w') as stderr:
  try:ret=subprocess.run(cmd,stdin=stdin,stdout=stdout,stderr=stderr,timeout=2400).returncode
  except subprocess.TimeoutExpired:ret=124
 if ret==0:save(ROOT/'responses'/f'{name}.invocation.json',{'name':name,'model':'gpt-6.1-sol','reasoning_effort':'high','returncode':0,'wall_seconds':time.time()-before,'command':cmd,'prompt_sha256':digest(folder/'prompt.txt')});print('Completed',name,flush=True)
 else:
  for p in (ROOT/'responses').glob(name+'.*'):p.unlink()
  print('No completed prediction',name,flush=True)

def evaluate(name):
 folder=BASE/name;m=json.loads((folder/'case.json').read_text());text=(ROOT/'responses'/f'{name}.json').read_text().strip()
 if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
 out=json.loads(text);lookup={r['id']:r for r in out['predictions']};assert set(lookup)==set(m['query_ids']);rows=[]
 if name=='unseen_word_phonetics_t12':
  gt=json.loads((BASE/f'{name}_ground_truth.json').read_text());targets={q:[w] for q,w in zip(gt['query_ids'],gt['truth_words'])}
 else:
  source_case=m['source_case'];old=json.loads((ROOT/'cases'/f'{source_case}.json').read_text());case=np.load(ROOT/'cases'/f'{source_case}.npz');d=loadmat(WORK/'sources'/old['source'],variable_names=['sentenceText'],simplify_cells=True);targets={q:words(d['sentenceText'][i]) for q,i in zip(old['query_ids'],case['test'])}
 for q in m['query_ids']:
  r=lookup[q];pred=r.get('tokens',words(r.get('word','')));truth=targets[q];rows.append({'id':q,'prediction':pred,'truth':truth,'edit_distance':distance(truth,pred)})
 events=[json.loads(l) for l in (ROOT/'responses'/f'{name}.events.jsonl').read_text().splitlines()];tool_items=[e.get('item',{}).get('type') for e in events if e.get('item',{}).get('type') not in [None,'reasoning','agent_message']]
 out.update(name=name,n=len(rows),exact=sum(r['prediction']==r['truth'] for r in rows),WER=sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),rows=rows,tool_item_types=tool_items,usage=[e['usage'] for e in events if e.get('type')=='turn.completed'])
 save(ROOT/'evaluation'/f'{name}.sol.json',out);print(json.dumps({k:out[k] for k in ['name','n','exact','WER','tool_item_types']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare_symbols','run','evaluate']);p.add_argument('name',nargs='?');p.add_argument('--source-case',default='sentences_t16_fresh_dev');a=p.parse_args()
 with threadpool_limits(limits=1):
  if a.action=='prepare':prepare_visual();prepare_symbols();prepare_phonetic()
  elif a.action=='prepare_symbols':prepare_symbols(a.name,a.source_case)
  elif a.action=='run':run(a.name)
  else:evaluate(a.name)
