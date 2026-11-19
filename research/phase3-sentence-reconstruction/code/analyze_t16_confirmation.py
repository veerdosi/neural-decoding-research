"""Unblind the reserved T16 cohort only after Sol has completed predictions."""
from __future__ import annotations
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox, resolve_locked
import hashlib,json
import numpy as np
from sequence_study import ROOT,save,SEED
from agentic_cases import evaluate
from sentence_full_evidence_comparison import evaluate_generic

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def scores(rows):
 errors=sum(r['edit_distance'] for r in rows);nwords=sum(len(r['truth']) for r in rows)
 return {'n':len(rows),'edits':errors,'reference_words':nwords,'WER':errors/nwords,'exact':sum(r['prediction']==r['truth'] for r in rows)}

def paired(sol,algorithm):
 a={r['id']:r for r in algorithm};assert set(a)=={r['id'] for r in sol}
 delta=np.array([a[r['id']]['edit_distance']-r['edit_distance'] for r in sol]);length=np.array([len(r['truth']) for r in sol]);rng=np.random.default_rng(SEED)
 samples=rng.integers(0,len(sol),(20000,len(sol)));improvement=delta[samples].sum(1)/length[samples].sum(1)
 # These recordings all come from one block; the interval is conditional on
 # resampling its trials, not a session- or participant-level interval.
 return {'net_edit_reduction_with_sol':int(delta.sum()),'WER_reduction_percentage_points':100*delta.sum()/length.sum(),'descriptive_trial_bootstrap_95pct_percentage_points':(100*np.quantile(improvement,[.025,.975])).tolist(),'sol_lower_error_trials':int((delta>0).sum()),'algorithm_lower_error_trials':int((delta<0).sum()),'equal_error_trials':int((delta==0).sum()),'scope':'Paired within one chronological recording block. Does not account for unknown serial dependencies or support population generalization.'}

def main():
 name='sentences_t16_confirm';response=ROOT/'responses'/f'{name}.json';inv=ROOT/'responses'/f'{name}.invocation.json'
 assert inv.exists() and read(inv)['returncode']==0,'Do not unblind without a completed Sol invocation'
 prediction=read(response);spec=read(ROOT/'cases'/f'{name}.json');assert {p['id'] for p in prediction['predictions']}==set(spec['query_ids'])
 folder=ROOT/'evaluation/algorithm_sentence_baselines/full_evidence';freeze=read(folder/'t16_confirmation_baseline_freeze.json')
 locked=freeze['artifacts']+freeze['code_artifacts']+freeze['shuffled_control_prediction_artifacts'];checks={r['path']:sha(resolve_locked(r['path']))==r['sha256'] for r in locked};assert all(checks.values())
 save(ROOT/'evaluation/t16_preunblinding_integrity.json',{'all_frozen_artifacts_unchanged':True,'checks':checks,'sol_prediction_sha256':sha(response),'sol_invocation_sha256':sha(inv)})
 evaluate(name)
 for tag in ['generic','t16cal']:evaluate_generic(name,tag)
 sol=read(ROOT/'evaluation'/f'{name}.sol.json');meta=read(resolve_sandbox(spec['sandbox'])/'inputs.json');cal=[' '+' '.join(c['tokens'])+' ' for c in meta['calibration']]
 strict={r['id'] for r in sol['rows'] if not any(' '+' '.join(r['truth'])+' ' in s for s in cal)}
 out={'cohort':'32 reserved chronological T16 recordings, calibration80 from two earlier blocks; ROI chosen on separate development8','prediction_sha256':sha(response),'strictly_novel_target_ids':sorted(strict),'sol':{'all':scores(sol['rows']),'strict':scores([r for r in sol['rows'] if r['id'] in strict])},'comparisons':{},'limits':'One participant/session and one evaluation block. Full-block label-free mean correction uses unlabeled recordings. This is within-session generalization, not cross-session or participant transfer. The Sol workflow and numerical inventory were developed before confirmation; no inference about arbitrary language or private thoughts.'}
 for tag in ['generic','t16cal']:
  r=read(folder/f'{tag}_{name}_results.json');out['comparisons'][tag]={'all':scores(r['rows']),'strict':scores([q for q in r['rows'] if q['id'] in strict]),'paired_all':paired(sol['rows'],r['rows']),'paired_strict':paired([q for q in sol['rows'] if q['id'] in strict],[q for q in r['rows'] if q['id'] in strict]),'wall_seconds_algorithm':r['runtime_end_to_end_seconds']}
 events=[json.loads(l) for l in (ROOT/'responses'/f'{name}.events.jsonl').read_text().splitlines()];out['usage']=[e['usage'] for e in events if e['type']=='turn.completed'];out['wall_seconds_sol']=read(inv)['wall_seconds']
 save(ROOT/'evaluation/t16_confirmation_comparison.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['strictly_novel_target_ids']},indent=2))

if __name__=='__main__':main()
