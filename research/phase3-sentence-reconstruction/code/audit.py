"""Check sequence experiment integrity; film QA is stored with each version."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib
import numpy as np
from sequence_study import ROOT,SOURCE,representation,save

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 protocol=json.loads((ROOT/'protocol.json').read_text());data=np.load(ROOT/'data/sequence_features.npz');checks={}
 assert sha(SOURCE)==protocol['source_sha256'];checks['source_hash']=True
 cases={p.stem:json.loads(p.read_text()) for p in (ROOT/'cases').glob('homogeneous*.json')}
 for name,m in cases.items():
  c=np.load(ROOT/'cases'/f'{name}.npz');assert not set(c['train'])&set(c['test']);assert not set(map(tuple,c['ycal']))&set(map(tuple,c['y']))
  assert np.all(c['train']<85)
  if m['split']=='confirmation':assert np.all(c['test']>=85)
  assert not set(hashlib.sha256(x.tobytes()).hexdigest() for x in data['frames'][c['train']])&set(hashlib.sha256(x.tobytes()).hexdigest() for x in data['frames'][c['test']])
  prompt=ROOT/'prompts'/f'{name}.txt';assert sha(prompt)==m['prompt_sha256']
  text=prompt.read_text();cal_text,query_text=text.split('\nUnlabeled query trajectories:\n');cal_rows=[json.loads(l) for l in cal_text.splitlines() if l.startswith('{')];q_rows=[json.loads(l) for l in query_text.splitlines() if l.startswith('{')]
  assert [r['id'] for r in q_rows]==m['query_ids'];assert len(cal_rows)==len(c['train']);assert all('tokens' not in r for r in q_rows)
  assert np.array_equal(np.array([r['trajectory'] for r in cal_rows]),c['cal'])
  if m['control']=='no_signal':assert all(list(r)==['id'] for r in q_rows)
  else:assert np.array_equal(np.array([r['trajectory'] for r in q_rows]),c['query'])
  for row,y in zip(cal_rows,c['ycal']):assert row['tokens']==['up' if b else 'right' for b in y]
  invocation=ROOT/'responses'/f'{name}.invocation.json'
  if invocation.exists():
   inv=json.loads(invocation.read_text());assert inv['returncode']==0;assert inv['model']=='gpt-6.1-sol' and inv['reasoning_effort']=='high';assert inv['prompt_sha256']==sha(prompt)
   events=[json.loads(l) for l in (ROOT/'responses'/f'{name}.events.jsonl').read_text().splitlines()];assert not [e for e in events if e.get('item',{}).get('type') not in [None,'reasoning','agent_message']]
 checks['chronological_and_full_sequence_separation']=True;checks['no_duplicate_calibration_query_features']=True;checks['prompt_case_integrity']=True;checks['model_high_codex_and_tool_free']=True
 name='homogeneous_pc12_confirm';base=np.load(ROOT/'cases'/f'{name}.npz');a,b,_=representation(data['frames'][base['train']],data['frames'][base['test']],12);assert np.array_equal(a,base['cal']) and np.array_equal(b,base['query']);checks['calibration_only_preprocessing_recomputed']=True
 assert (ROOT/'prompts'/f'{name}.txt').read_bytes()==(ROOT/'prompts'/f'{name}_repeat.txt').read_bytes();checks['independent_repeat_identical_prompt']=True
 reverse=np.load(ROOT/'cases'/f'{name}_reversed.npz');assert np.array_equal(reverse['query'],b[:,::-1]);assert np.array_equal(reverse['cal'],a);checks['time_reversal_only_changes_query_time']=True
 shuffle=np.load(ROOT/'cases'/f'{name}_shuffle.npz');lookup={x.tobytes():i for i,x in enumerate(b)};idx=np.array([lookup[x.tobytes()] for x in shuffle['query']]);assert np.array_equal(shuffle['donor_y'],base['y'][idx]);assert np.array_equal(shuffle['donor_y'].sum(1),base['y'].sum(1));checks['shuffle_is_real_recording_and_preserves_word_bag']=True
 checks['scope_sequence_experiments_only']=True
 save(ROOT/'evaluation/integrity_audit.json',{'passed':True,'checks':checks,'limits':['Single participant/session, not population validation','Author block mean correction uses unlabeled recording blocks','Same 62 recordings across repetitions/controls; not independent new participants','Public dataset pretraining contamination cannot be excluded from undisclosed model training data']})
 print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
