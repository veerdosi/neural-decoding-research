"""Matched discrete HMM comparator; weak word alignment, no Sol predictions used."""
from __future__ import annotations
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,hashlib,itertools,json,time,pathlib
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,save
from sentence_study import distance

BASE=ROOT/'discovery_probes/neural_alphabet_t16'

def align(codes,states,emission):
 score=emission[states][:,codes].T;n,s=score.shape
 prev=np.full(s,-1e30);prev[0]=score[0,0];back=np.zeros((n,s),np.int16)
 for t in range(1,n):
  stay=prev+np.log(.65);advance=np.r_[-1e30,prev[:-1]+np.log(.35)];take=advance>stay;prev=np.maximum(stay,advance)+score[t];back[t]=np.arange(s)-take
 state=s-1;path=np.empty(n,int)
 for t in range(n-1,-1,-1):path[t]=state;state=back[t,state]
 return states[path]

def fit(cal,codes,vocab,nstates,prior,iterations=10):
 wi={w:i for i,w in enumerate(vocab)};s=len(vocab)*nstates+2;start=s-2;end=s-1;counts=np.zeros((s,64));bg=np.bincount(np.concatenate([codes[c['id']] for c in cal]),minlength=64)+1;bg=bg/bg.sum();assigned={};paths={}
 for c in cal:
  path=np.array([start]+[wi[w]*nstates+j for w in c['tokens'] for j in range(nstates)]+[end]);n=len(codes[c['id']]);assert n>=len(path)
  paths[c['id']]=path;assigned[c['id']]=path[np.minimum((np.arange(n)*len(path)/n).astype(int),len(path)-1)]
 for iteration in range(iterations):
  counts.fill(0)
  for c in cal:np.add.at(counts,(assigned[c['id']],codes[c['id']]),1)
  emission=np.log((counts+prior*bg)/(counts.sum(1,keepdims=True)+prior))
  if iteration<iterations-1:
   assigned={c['id']:align(codes[c['id']],paths[c['id']],emission) for c in cal}
 unigram=np.ones(len(vocab));bigram=np.ones((len(vocab),len(vocab)));starts=np.ones(len(vocab));ends=np.ones(len(vocab))
 for c in cal:
  ix=[wi[w] for w in c['tokens']];starts[ix[0]]+=1;ends[ix[-1]]+=1
  for i in ix:unigram[i]+=1
  for a,b in zip(ix,ix[1:]):bigram[a,b]+=1
 bigram=(bigram+3*unigram/unigram.sum());bigram/=bigram.sum(1,keepdims=True)
 return emission,np.log(starts/starts.sum()),np.log(bigram),np.log(ends/ends.sum())

def decode(codes,model,vocab,nstates,lm_scale,penalty):
 em,starts,bigram,ends=model;nw=len(vocab);ns=nw*nstates+2;start=ns-2;end=ns-1;first=np.arange(nw)*nstates;last=first+nstates-1
 trans=np.full((ns,ns),-1e30);np.fill_diagonal(trans,np.log(.65))
 for w in range(nw):
  for j in range(nstates-1):trans[w*nstates+j,w*nstates+j+1]=np.log(.35)
 trans[np.ix_(last,first)]=np.log(.35)+lm_scale*bigram+penalty
 trans[start,first]=np.log(.35)+lm_scale*starts+penalty;trans[last,end]=np.log(.35)+lm_scale*ends
 prev=np.full(ns,-1e30);prev[start]=em[start,codes[0]];back=np.zeros((len(codes),ns),np.int16)
 for t in range(1,len(codes)):
  a=prev[:,None]+trans;back[t]=a.argmax(0);prev=a.max(0)+em[:,codes[t]]
 state=end;path=np.empty(len(codes),int)
 for t in range(len(codes)-1,-1,-1):path[t]=state;state=back[t,state]
 result=[];previous=start
 for state in path:
  if state<start and state%nstates==0 and state!=previous:result.append(vocab[state//nstates])
  previous=state
 return result

def main():
 before=time.time();spec=json.loads((ROOT/'cases/sentences_t16_fresh_dev.json').read_text());meta=json.loads((resolve_sandbox(spec['sandbox'])/'inputs.json').read_text());z=np.load(BASE/'symbol_codes_for_analysis.npz');codes={r['id']:z[r['id']] for r in meta['calibration']+meta['queries']};cal=meta['calibration'];vocab=sorted({w for c in cal for w in c['tokens']});case=np.load(ROOT/'cases/sentences_t16_fresh_dev.npz');src=loadmat(WORK/'sources'/spec['source'],variable_names=['goTrialEpochs','blockNum'],simplify_cells=True);go=src['goTrialEpochs'].astype(int)-1;groups=src['blockNum'][go[:,0]][case['train']];records=[]
 for states,prior in itertools.product([1,2,3],[1.,8.]):
  folds=[]
  for block in sorted(set(groups)):
   tr=[c for c,g in zip(cal,groups) if g!=block];va=[c for c,g in zip(cal,groups) if g==block];folds.append((va,fit(tr,codes,vocab,states,prior)))
  for lm,penalty in itertools.product([0.,.5],[0.,-2.]):
   errors=0;total=0
   for va,model in folds:
    for c in va:errors+=distance(c['tokens'],decode(codes[c['id']],model,vocab,states,lm,penalty));total+=len(c['tokens'])
   records.append({'word_states':states,'dirichlet_prior_frames':prior,'lm_scale':lm,'word_transition_penalty':penalty,'calibration_cv_WER':errors/total})
 records.sort(key=lambda r:(r['calibration_cv_WER'],r['word_states'],r['dirichlet_prior_frames'],r['lm_scale'],r['word_transition_penalty']));s=records[0];model=fit(cal,codes,vocab,s['word_states'],s['dirichlet_prior_frames']);pred=[{'id':q['id'],'tokens':decode(codes[q['id']],model,vocab,s['word_states'],s['lm_scale'],s['word_transition_penalty'])} for q in meta['queries']]
 # Lock predictions before loading evaluation labels or Sol outputs.
 out={'predictions':pred,'selected':s,'calibration_cv':records,'calibration_blocks':sorted(map(int,set(groups))),'method':'Categorical word-state HMM, transcript-constrained Viterbi training, calibration unigram/bigram prior. No DTW or Sol at application.','representation':'Exact same64 cluster symbols and native run durations supplied to Sol. Clusters/PCA fitted on all80 calibration neural recordings; CV holds out transcript labels by block but reuses that unsupervised representation.','wall_seconds':time.time()-before,'query_targets_used_for_selection':False,'baseline_design_scope':'Designed after viewing Sol development outcome; hyperparameters and word alignment use calibration labels only. Not an independently preregistered algorithm family.'}
 save(ROOT/'evaluation/neural_alphabet_hmm_predictions_locked.json',out)
 sol=json.loads((ROOT/'evaluation/neural_alphabet_t16.sol.json').read_text());truth={r['id']:r['truth'] for r in sol['rows']};rows=[{'id':r['id'],'prediction':r['tokens'],'truth':truth[r['id']],'edit_distance':distance(truth[r['id']],r['tokens'])} for r in pred];out.update(rows=rows,exact=sum(r['prediction']==r['truth'] for r in rows),WER=sum(r['edit_distance'] for r in rows)/sum(len(r['truth']) for r in rows));save(ROOT/'evaluation/neural_alphabet_hmm_comparison.json',out);print(json.dumps({k:out[k] for k in ['selected','exact','WER','wall_seconds']},indent=2))

def confirmation():
 before=time.time();folder=ROOT/'discovery_probes/neural_alphabet_t16_confirm';spec=json.loads((ROOT/'cases/sentences_t16_confirm.json').read_text());meta=json.loads((resolve_sandbox(spec['sandbox'])/'inputs.json').read_text());z=np.load(folder/'symbol_codes_for_analysis.npz');dev=np.load(BASE/'symbol_codes_for_analysis.npz');codes={r['id']:z[r['id']] for r in meta['calibration']+meta['queries']};cal=meta['calibration'];assert all(np.array_equal(codes[c['id']],dev[c['id']]) for c in cal)
 lock=ROOT/'evaluation/neural_alphabet_hmm_predictions_locked.json';s=json.loads(lock.read_text())['selected'];vocab=sorted({w for c in cal for w in c['tokens']});model=fit(cal,codes,vocab,s['word_states'],s['dirichlet_prior_frames']);pred=[{'id':q['id'],'tokens':decode(codes[q['id']],model,vocab,s['word_states'],s['lm_scale'],s['word_transition_penalty'])} for q in meta['queries']]
 save(ROOT/'evaluation/neural_alphabet_hmm_confirmation_predictions_locked.json',{'predictions':pred,'selected':s,'development_configuration_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'identical_calibration_symbol_codes':True,'query_targets_read':False,'wall_seconds':time.time()-before});print('Locked32 HMM predictions without reading targets',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--confirm',action='store_true');a=p.parse_args()
 if a.confirm:confirmation()
 else:main()
