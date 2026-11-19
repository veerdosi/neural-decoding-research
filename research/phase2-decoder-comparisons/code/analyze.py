"""Rescore saved model outputs and summarize dependent repeated evaluations."""
import json, pathlib, datetime, collections
import numpy as np
from scipy.stats import binomtest
from experiment import ROOT,PHASE1,WORDS,SEED,score,save

def get(name,suffix):
 p=ROOT/'evaluation'/f'{name}.{suffix}.json'
 return json.loads(p.read_text()) if p.exists() else None

def align(name,prediction):
 d=np.load(ROOT/'cases'/f'{name}.npz');return dict(zip(d['test'].tolist(),np.asarray(prediction).tolist()))

def paired(name,sol,method,mean_align=False):
 b=get(name,'baseline_mean_aligned' if mean_align else 'baseline')
 if not b or method not in b:return None
 d=np.load(ROOT/'cases'/f'{name}.npz');a=np.asarray(sol['prediction'])==d['query_y'];z=np.asarray(b[method]['prediction'])==d['query_y']
 win=int(np.sum(a&~z));lose=int(np.sum(z&~a))
 return {'difference':float(np.mean(a)-np.mean(z)),'sol_only_correct':win,'baseline_only_correct':lose,
         'mcnemar_two_sided_descriptive_p':float(binomtest(win,win+lose,.5).pvalue) if win+lose else 1.,
         'block_difference':{str(k):float(np.mean(a[d['query_blocks']==k])-np.mean(z[d['query_blocks']==k])) for k in np.unique(d['query_blocks'])}}

def permutation_group(names, include_phase1=False):
 preds=[];truth=None;blocks=None;indices=None
 for name in names:
  s=get(name,'sol')
  if not s:return None
  d=np.load(ROOT/'cases'/f'{name}.npz');order=np.argsort(d['test']);p=np.asarray(s['prediction'])[order]
  if truth is None:truth=d['query_y'][order];blocks=d['query_blocks'][order];indices=d['test'][order]
  else:assert np.array_equal(indices,d['test'][order])
  preds.append(p)
 if include_phase1:
  t=json.loads((PHASE1/'evaluation/ground_truth.json').read_text());r=json.loads((PHASE1/'responses/raw128_icl.json').read_text());lookup={q['id']:WORDS.index(q['word']) for q in r['predictions']};bytrial={x['trial_index_zero_based']:lookup[x['id']] for x in t};preds.append(np.array([bytrial[int(i)] for i in indices]))
 preds=np.array(preds);observed=int(np.sum(preds==truth));rng=np.random.default_rng(SEED+199);null=np.zeros(20000,int)
 for b in np.unique(blocks):
  idx=np.flatnonzero(blocks==b);shuffled=np.array([rng.permutation(truth[idx]) for _ in range(20000)])
  null+=(shuffled[:,None,:]==preds[None,:,idx]).sum(axis=(1,2))
 return {'unique_recordings':len(truth),'calls':len(preds),'observed_correct_across_dependent_calls':observed,
         'block_restricted_shared_label_permutation_p':float((1+np.sum(null>=observed))/20001),
         'note':'The same within-block label permutation is shared across all calls, preserving their dependence. Assumes label exchangeability within blocks; not population inference.'}

def main():
 summaries={};pending=[]
 for p in sorted((ROOT/'responses').glob('*.invocation.json')):
  inv=json.loads(p.read_text());name=inv['name']
  if 'returncode' not in inv:pending.append(name);continue
  if inv['returncode']!=0:continue
  s=score(name)
  start=datetime.datetime.fromisoformat(inv['started_utc'].replace('Z','+00:00'));end=datetime.datetime.fromisoformat(inv['finished_utc'].replace('Z','+00:00'))
  meta=json.loads((ROOT/'cases'/f'{name}.json').read_text());s['wall_seconds']=(end-start).total_seconds();s['case_metadata']=meta
  s['paired_baselines']={method:v for method in ['knn','lda','lda_conservative_ties','gnb','gnb_conservative_ties'] if (v:=paired(name,s,method)) is not None}
  s['paired_mean_aligned_baselines']={method:v for method in ['knn','lda','lda_conservative_ties','gnb','gnb_conservative_ties'] if (v:=paired(name,s,method,True)) is not None}
  summaries[name]=s
 curves={}
 for n in [2,4,7,14]:
  names=[f'dev_raw_n{n}_draw{i}' for i in range(3)] if n<14 else ['dev_raw_n14_repeat1','dev_raw_n14_repeat2']
  sols=[summaries[x]['accuracy'] for x in names if x in summaries]
  if n==14:sols=[52/98]+sols
  row={'sol_accuracies':sols,'sol_n_calls':len(sols),'sol_mean':float(np.mean(sols)) if sols else None,
       'sol_range':[float(np.min(sols)),float(np.max(sols))] if sols else None}
  for method in ['knn','lda','lda_conservative_ties','gnb','gnb_conservative_ties']:
   all_names=[f'dev_raw_n{n}_draw{i}' for i in range(20 if n<14 else 1)]
   all_vals=[get(x,'baseline')[method]['accuracy'] for x in all_names if get(x,'baseline') and method in get(x,'baseline')]
   matched_names=[f'dev_raw_n{n}_draw{i}' for i in range(3)] if n<14 else ['dev_raw_n14_repeat1']
   matched=[get(x,'baseline')[method]['accuracy'] for x in matched_names if get(x,'baseline') and method in get(x,'baseline')]
   row[method]={'all_calibration_draws':all_vals,'all_draw_mean':float(np.mean(all_vals)) if all_vals else None,
                'all_draw_sd':float(np.std(all_vals,ddof=1)) if len(all_vals)>1 else None,'matched_draw_values':matched,'matched_mean':float(np.mean(matched)) if matched else None}
   aligned=[get(x,'baseline_mean_aligned')[method]['accuracy'] for x in matched_names if get(x,'baseline_mean_aligned') and method in get(x,'baseline_mean_aligned')]
   row['mean_aligned_'+method]={'matched_draw_values':aligned,'matched_mean':float(np.mean(aligned)) if aligned else None}
  curves[str(n)]=row
 chunks=[f'dev_raw_n14_chunk{i}' for i in range(7)];chunk_result=None
 if all(x in summaries for x in chunks):
  merged={}
  for c in chunks:merged.update(align(c,summaries[c]['prediction']))
  assert len(merged)==98
  base=np.load(ROOT/'cases/dev_raw_n14_repeat1.npz');truth=dict(zip(base['test'].tolist(),base['query_y'].tolist()))
  correct={i:merged[i]==truth[i] for i in merged}
  chunk_result={'unique_recordings':len(merged),'correct':int(sum(correct.values())),'accuracy':float(np.mean(list(correct.values()))),
                'per_chunk_accuracy':[summaries[x]['accuracy'] for x in chunks],
                'note':'One seven-chunk partition, same exact calibration/header and same 98 previously evaluated recordings. Seven calls are not seven new recording cohorts.'}
 result={'sol_calls':summaries,'pending_started_calls':pending,'development_calibration_curve':curves,'chunk_protocol':chunk_result,
         'replication_permutations':{'t12_exact_repeats':permutation_group(['dev_raw_n14_repeat1','dev_raw_n14_repeat2'],True),
                                     't16_frozen_n7':permutation_group(['confirm_raw_n7_draw0','confirm_raw_n7_draw1'])},
         'uncertainty_policy':'Report unique recordings, per-call/per-block outcomes, calibration-draw spread, and descriptive trial intervals separately. Never pool repeated predictions as independent trials or infer population performance from two participants.'}
 save(ROOT/'evaluation/phase2_summary.json',result)
 import csv
 with (ROOT/'evaluation/successful_experiments.csv').open('w',newline='') as file:
  writer=csv.writer(file);writer.writerow(['condition','dataset','evaluation_status','examples_per_word','query_recordings','correct','accuracy','representation','control','input_tokens_completed_turns','output_tokens_including_reasoning','reasoning_output_tokens','wall_seconds'])
  for name,s in summaries.items():
   m=s['case_metadata'];usage=s['usage'];writer.writerow([name,m['dataset'],m['evaluation_status'],m['n_per_word'],s['n'],s['correct'],s['accuracy'],m['representation'],m.get('control'),sum(u['input_tokens'] for u in usage),sum(u['output_tokens'] for u in usage),sum(u.get('reasoning_output_tokens',0) for u in usage),s['wall_seconds']])
 print('Completed model calls',len(summaries),'pending',len(pending))

if __name__=='__main__':main()
