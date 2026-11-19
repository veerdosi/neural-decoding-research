"""Score reserved symbolic decoding and its matched HMM; optional donor control."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib
import numpy as np
from scipy.io import loadmat
from sequence_study import ROOT,WORK,save
from sentence_study import words,distance
from discovery_probes import evaluate
from analyze_t16_confirmation import scores,paired

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 name='neural_alphabet_t16_confirm';inv=ROOT/'responses'/f'{name}.invocation.json';assert inv.exists() and read(inv)['returncode']==0
 evaluate(name);sol=read(ROOT/'evaluation'/f'{name}.sol.json');assert not sol['tool_item_types'],'Tool-free protocol violation must be reported'
 lock=ROOT/'evaluation/neural_alphabet_hmm_confirmation_predictions_locked.json';baseline=read(lock);targets={r['id']:r['truth'] for r in sol['rows']};rows=[{'id':r['id'],'prediction':r['tokens'],'truth':targets[r['id']],'edit_distance':distance(targets[r['id']],r['tokens'])} for r in baseline['predictions']]
 source=read(ROOT/'cases/sentences_t16_confirm.json');meta=read(resolve_sandbox(source['sandbox'])/'inputs.json');cal=[' '+' '.join(r['tokens'])+' ' for r in meta['calibration']];strict={r['id'] for r in sol['rows'] if not any(' '+' '.join(r['truth'])+' ' in s for s in cal)}
 out={'scope':'Frozen symbolic interface,80 calibration sentences and32 chronologically reserved within-session recordings from T16. Symbol IDs arbitrary; PCA and KMeans use calibration neural frames only. No Sol tools, retrieved transcripts, word boundaries or query word counts.','sol':{'all':scores(sol['rows']),'strict':scores([r for r in sol['rows'] if r['id'] in strict]),'usage':sol['usage'],'wall_seconds':read(inv)['wall_seconds']},'matched_HMM':{'all':scores(rows),'strict':scores([r for r in rows if r['id'] in strict]),'selected':baseline['selected'],'wall_seconds':baseline['wall_seconds']},'paired_comparison':paired(sol['rows'],rows),'strictly_novel_target_ids':sorted(strict),'Sol_rows':sol['rows'],'HMM_rows':rows,'prediction_hashes':{'sol':sha(ROOT/'responses'/f'{name}.json'),'HMM':sha(lock)},'limits':'One participant/date and one evaluation block. Conditional trial uncertainty does not establish between-session or population performance. Full-block label-free centering is transductive. HMM is one matched conventional baseline, not the best achievable conventional decoder.'}
 broader={}
 for tag in ['generic','t16cal']:
  p=ROOT/'evaluation/algorithm_sentence_baselines/full_evidence'/f'{tag}_sentences_t16_confirm_results.json'
  if p.exists():
   r=read(p);broader[tag]={'all':scores(r['rows']),'paired_with_symbolic_Sol':paired(sol['rows'],r['rows']),'scope':'Stronger existing workflow on full neural features. Same calibration/evaluation trials, different and richer input representation; not a matched-symbol comparison. T16cal runtime reuses the generic evidence cache.'}
 out['full_feature_contextual_comparisons']=broader
 shuffle='neural_alphabet_t16_confirm_shuffle';shinv=ROOT/'responses'/f'{shuffle}.invocation.json'
 if shinv.exists() and read(shinv)['returncode']==0:
  evaluate(shuffle);control=read(ROOT/'evaluation'/f'{shuffle}.sol.json');c=np.load(ROOT/'cases/sentences_t16_confirm_shuffle.npz');spec=read(ROOT/'cases/sentences_t16_confirm_shuffle.json');d=loadmat(WORK/'sources'/spec['source'],variable_names=['sentenceText'],simplify_cells=True);donors={q:words(d['sentenceText'][i]) for q,i in zip(spec['query_ids'],c['donor'])};donorrows=[{'id':r['id'],'prediction':r['prediction'],'truth':donors[r['id']],'edit_distance':distance(donors[r['id']],r['prediction'])} for r in control['rows']]
  out['duration_matched_recording_shuffle']={'nominal':scores(control['rows']),'donor':scores(donorrows),'tool_item_types':control['tool_item_types'],'donor_rows':donorrows,'scope':'New independent inference on recording strings assigned to different query IDs; donor performance need not equal the original call because model predictions vary.'}
 save(ROOT/'evaluation/neural_alphabet_confirmation_comparison.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['Sol_rows','HMM_rows','strictly_novel_target_ids']},indent=2))

if __name__=='__main__':main()
