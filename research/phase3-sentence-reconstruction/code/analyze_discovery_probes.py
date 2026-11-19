"""Audit the lexical holdout and summarize completed mechanism probes."""
from __future__ import annotations
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib,re
from sequence_study import ROOT,save

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 base=ROOT/'discovery_probes';name='unseen_word_phonetics_t12';folder=base/name
 meta=json.loads((folder/'case.json').read_text());inp=json.loads((folder/'inputs.json').read_text());gt=json.loads((base/(name+'_ground_truth.json')).read_text())
 dictionary={}
 for line in (folder/'cmudict.dict').read_text().splitlines():
  a=line.split()
  if len(a)<2 or a[0].startswith(';;;'):continue
  w=re.sub(r'\(\d+\)$','',a[0]).lower();phones=[re.sub('[012]','',p) for p in a[1:]]
  dictionary.setdefault(w,[]).append(phones)
 calibration_words=sorted({r['word'] for r in inp['calibration']});held=sorted(set(gt['truth_words']))
 calphones={p for w in calibration_words for v in dictionary[w] for p in v}
 coverage={w:{'pronunciations':dictionary[w],'uncovered_phones':sorted({p for v in dictionary[w] for p in v}-calphones)} for w in held}
 assert not set(calibration_words)&set(held)
 assert not set(gt['calibration_trials'])&set(gt['query_trials'])
 assert all(not r['uncovered_phones'] for r in coverage.values())
 assert all('word' not in r and 'tokens' not in r for r in inp['queries'])
 checks={f:sha(folder/f)==h for f,h in meta['input_hashes'].items()}
 assert all(checks.values())
 audit={'calibration_word_types':len(calibration_words),'held_word_types':held,'calibration_trials':len(gt['calibration_trials']),'query_trials':len(gt['query_trials']),'whole_word_types_disjoint':True,'trial_indices_disjoint':True,'all_held_word_phonemes_present_in_calibration':True,'phoneme_coverage':coverage,'query_metadata_has_no_word_labels':True,'input_hashes_match_frozen_case':checks,'scope':'Single participant/date, cued attempted single words; known source vocabulary is a possible model prior. Absence from supplied calibration is not proof of absence from model pretraining. Block centering includes unlabeled data.'}
 save(ROOT/'evaluation/discovery_lexical_holdout_audit.json',audit)
 rows=[]
 for n in ['visual_dynamics_t16','neural_alphabet_t16','unseen_word_phonetics_t12']:
  p=ROOT/'evaluation'/f'{n}.sol.json'
  if not p.exists():continue
  r=json.loads(p.read_text());rows.append({k:r[k] for k in ['name','n','exact','WER','tool_item_types','usage']})
 save(ROOT/'evaluation/discovery_probe_summary.json',{'completed':rows,'lexical_holdout_audit':'discovery_lexical_holdout_audit.json','scope':'Exploratory mechanism probes; visual and symbol calls reuse the eight T16 development recordings. No inference about all possible representations or population performance follows from one call.'})
 print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
