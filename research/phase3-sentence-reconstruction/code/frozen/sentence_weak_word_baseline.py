"""Independent conventional weak-label LDA/forced-alignment sentence decoder.

This is a separately fitted supervised comparison baseline, never an input to
Sol. Calibration-only block CV selects all parameters. Query text is loaded
only by the separate evaluate action after weights/predictions are locked.
"""
import argparse,itertools,json,pathlib,time
import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from sentence_algorithm_comparison import ROOT,OUT,SOURCE,arrays,inputs,weak_labels,language,decode,distance,save,read,sha

DEST=OUT/'weak_word_decoder'
DEST.mkdir(exist_ok=True)

def fit(x,y,vocab,shrink):
 n=len(vocab);counts=np.bincount(y,minlength=n);means=np.zeros((n,x.shape[1]));np.add.at(means,y,x);means/=np.maximum(counts[:,None],1)
 residual=x-means[y];cov=residual.T@residual/len(x);scale=np.trace(cov)/len(cov);cov=(1-shrink)*cov+shrink*scale*np.eye(len(cov))+np.eye(len(cov))*1e-9
 coef=np.linalg.solve(cov,means.T).T;bias=-.5*np.einsum('ij,ij->i',coef,means);bias[counts==0]=-1e9
 return coef,bias

def forced_align(scores,token_indices):
 s=scores[:,token_indices];k=s.shape[1];d=np.full(k,-np.inf);d[0]=s[0,0];back=[]
 for row in s[1:]:
  advance=np.r_[-np.inf,d[:-1]];move=advance>d;d=np.maximum(d,advance)+row;back.append(move)
 state=k-1;path=[state]
 for move in back[::-1]:state-=int(move[state]);path.append(state)
 return np.asarray(token_indices)[path[::-1]]

def train(cal,a,vocab,power,shrink,iterations):
 lookup={w:i for i,w in enumerate(vocab)};ys={c['id']:weak_labels(c,len(a[c['id']]),lookup,power)[2:-2] for c in cal};x=np.concatenate([a[c['id']][2:-2] for c in cal])
 for step in range(iterations+1):
  model=fit(x,np.concatenate([ys[c['id']] for c in cal]),vocab,shrink)
  if step<iterations:
   for c in cal:
    score=a[c['id']][2:-2]@model[0].T+model[1];ys[c['id']]=forced_align(score,[lookup[w] for w in c['tokens']])
 return model,ys

def calibrate():
 before=time.perf_counter();meta,raw=inputs();cal=meta['calibration'];vocab=sorted({w for c in cal for w in c['tokens']});case=np.load(ROOT/'cases/sentences_agentic_dev.npz');source=loadmat(SOURCE,variable_names=['goTrialEpochs','blockNum'],simplify_cells=True);go=source['goTrialEpochs'].astype(int)-1;groups=source['blockNum'][go[:,0]][case['train']]
 records={}
 for block in sorted(set(map(int,groups))):
  tr=[c for c,g in zip(cal,groups) if g!=block];va=[c for c,g in zip(cal,groups) if g==block];a=arrays({c['id']:raw[c['id']] for c in cal},[c['id'] for c in tr]);lm=language(tr,vocab)
  for power,shrink,iterations in itertools.product([0.,1.],[.5,.9,1.],[0,3]):
   model,_=train(tr,a,vocab,power,shrink,iterations);records[block,power,shrink,iterations]=(va,lm,{c['id']:a[c['id']][2:-2]@model[0].T+model[1] for c in va})
 choices=[]
 for power,shrink,iterations,temp,lms,penalty in itertools.product([0.,1.],[.5,.9,1.],[0,3],[.25,1.],[0.,.5],[-2.5,-4.]):
  edits=0;words=0;exact=0
  for block in sorted(set(map(int,groups))):
   va,lm,scores=records[block,power,shrink,iterations]
   for c in va:
    s=scores[c['id']]*temp;e=s-logsumexp(s,axis=1,keepdims=True);pred=decode(e,lm,vocab,lms,penalty,1);edits+=distance(c['tokens'],pred);words+=len(c['tokens']);exact+=pred==c['tokens']
  choices.append({'duration_letter_power':power,'covariance_shrinkage':shrink,'forced_alignment_iterations':iterations,'emission_temperature':temp,'lm_scale':lms,'transition_penalty':penalty,'min_word_frames':1,'calibration_cv_wer':edits/words,'calibration_cv_exact':exact})
 choices.sort(key=lambda c:(c['calibration_cv_wer'],-c['calibration_cv_exact'],c['forced_alignment_iterations'],c['duration_letter_power'],c['covariance_shrinkage']))
 lock={'selected':choices[0],'all_candidates':choices,'vocabulary':vocab,'calibration_only':True,'query_labels_used':False,'status':'Independent trained supervised baseline, not Sol input','weak_labels':'Uniform/letter-proportional word durations initialize labels; optional three monotone transcript-forced alignment/refit iterations; no observed word boundaries','folds':'Leave one calibration block out; neural PCA, word emissions, text priors and alignment use only the other two blocks','frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'wall_seconds_selection':time.perf_counter()-before,'code_sha256':sha(pathlib.Path(__file__))}
 save(DEST/'configuration_lock.json',lock);print('WEAK DECODER LOCKED',json.dumps(choices[0]),'seconds',lock['wall_seconds_selection'],flush=True)

def predict():
 before=time.perf_counter();lock=read(DEST/'configuration_lock.json');s=lock['selected'];meta,raw=inputs();cal=meta['calibration'];vocab=lock['vocabulary'];a=arrays(raw,[c['id'] for c in cal]);model,alignments=train(cal,a,vocab,s['duration_letter_power'],s['covariance_shrinkage'],s['forced_alignment_iterations']);lm=language(cal,vocab)
 np.savez_compressed(DEST/'word_emission_weights.npz',coef=model[0],bias=model[1],vocabulary=np.array(vocab))
 save(DEST/'estimated_calibration_word_alignments.json',{k:[vocab[i] for i in v] for k,v in alignments.items()})
 predictions=[]
 for q in meta['queries']:
  score=(a[q['id']][2:-2]@model[0].T+model[1])*s['emission_temperature'];e=score-logsumexp(score,axis=1,keepdims=True);pred=decode(e,lm,vocab,s['lm_scale'],s['transition_penalty'],s['min_word_frames']);predictions.append({'id':q['id'],'tokens':pred})
 out={'predictions':predictions,'query_labels_used':False,'configuration_sha256':sha(DEST/'configuration_lock.json'),'wall_seconds_fit_and_decode':time.perf_counter()-before};save(DEST/'predictions_locked.json',out);print('WEAK PREDICTIONS LOCKED',json.dumps(out),flush=True)

def evaluate():
 p=read(DEST/'predictions_locked.json');m=read(ROOT/'cases/sentences_agentic_dev.json');c=np.load(ROOT/'cases/sentences_agentic_dev.npz');source=loadmat(SOURCE,variable_names=['sentenceText'],simplify_cells=True)
 import re
 truth={q:re.findall('[a-z]+',str(source['sentenceText'][i]).lower()) for q,i in zip(m['query_ids'],c['test'])};rows=[{'id':r['id'],'prediction':r['tokens'],'truth':truth[r['id']],'edit_distance':distance(truth[r['id']],r['tokens'])} for r in p['predictions']]
 out={'n':len(rows),'exact_sentences':sum(r['prediction']==r['truth'] for r in rows),'edits':sum(r['edit_distance'] for r in rows),'reference_words':sum(len(r['truth']) for r in rows),'wer':sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows),'rows':rows,'protocol':'Secondary independently trained comparison; no Sol receives these weights or predictions; all selection calibration-only','runtime_fit_and_decode_seconds':p['wall_seconds_fit_and_decode'],'runtime_selection_seconds':read(DEST/'configuration_lock.json')['wall_seconds_selection']};save(DEST/'results.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['calibrate','predict','evaluate']);a=p.parse_args()
 with threadpool_limits(limits=1):{'calibrate':calibrate,'predict':predict,'evaluate':evaluate}[a.action]()
