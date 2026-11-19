"""Error complementarity, including a truth-informed word-lattice upper bound.

The lattice is built from predictions alone; reference text only scores it.
Neither oracle is a decoder or an achieved fusion result.
"""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json
import numpy as np
from sequence_study import ROOT,save

def align(a,b):
 n,m=len(a),len(b);d=np.zeros((n+1,m+1),int);d[:,0]=np.arange(n+1);d[0]=np.arange(m+1)
 for i in range(1,n+1):
  for j in range(1,m+1):d[i,j]=min(d[i-1,j-1]+(a[i-1]!=b[j-1]),d[i-1,j]+1,d[i,j-1]+1)
 out=[];i,j=n,m
 while i or j:
  if i and j and d[i,j]==d[i-1,j-1]+(a[i-1]!=b[j-1]):out.append((a[i-1],b[j-1]));i-=1;j-=1
  elif i and d[i,j]==d[i-1,j]+1:out.append((a[i-1],None));i-=1
  else:out.append((None,b[j-1]));j-=1
 return out[::-1]

def lattice_edits(columns,truth):
 prev=np.arange(len(truth)+1)
 for pair in columns:
  choices=[]
  for token in set(pair):
   if token is None:choices.append(prev.copy());continue
   row=prev+1
   for j in range(1,len(truth)+1):row[j]=min(row[j],prev[j-1]+(truth[j-1]!=token),row[j-1]+1)
   choices.append(row)
  prev=np.min(choices,axis=0)
 return int(prev[-1])

def main():
 symbolic=json.loads((ROOT/'evaluation/neural_alphabet_t16_confirm.sol.json').read_text())['rows'];folder=ROOT/'evaluation/algorithm_sentence_baselines/full_evidence';methods={'full_numerical':json.loads((folder/'t16cal_sentences_t16_confirm_results.json').read_text())['rows'],'tool_Sol':json.loads((ROOT/'evaluation/sentences_t16_confirm.sol.json').read_text())['rows']}
 for name,other in methods.items():
  byid={r['id']:r for r in other};rows=[]
  for a in symbolic:
   b=byid[a['id']];cols=align(a['prediction'],b['prediction']);lower=lattice_edits(cols,a['truth']);assert lower<=min(a['edit_distance'],b['edit_distance'])
   rows.append({'id':a['id'],'truth':a['truth'],'symbol':a['prediction'],'other':b['prediction'],'symbol_edits':a['edit_distance'],'other_edits':b['edit_distance'],'prediction_alignment':cols,'lattice_oracle_edits':lower})
  out={'other':name,'same_predictions':sum(r['symbol']==r['other'] for r in rows),'symbol_lower_error':sum(r['symbol_edits']<r['other_edits'] for r in rows),'other_lower_error':sum(r['other_edits']<r['symbol_edits'] for r in rows),'symbol_only_exact':sum(r['symbol_edits']==0<r['other_edits'] for r in rows),'other_only_exact':sum(r['other_edits']==0<r['symbol_edits'] for r in rows),'exact_union':sum(min(r['symbol_edits'],r['other_edits'])==0 for r in rows),'oracle_utterance_edits':sum(min(r['symbol_edits'],r['other_edits']) for r in rows),'oracle_word_lattice_edits':sum(r['lattice_oracle_edits'] for r in rows),'oracle_word_lattice_exact':sum(r['lattice_oracle_edits']==0 for r in rows),'reference_words':sum(len(r['truth']) for r in rows),'oracle_saved_edits_vs_other':sum(max(0,r['other_edits']-r['symbol_edits']) for r in rows),'rows':rows,'scope':'Truth-informed complementarity diagnostic only. Oracles are unattainable upper bounds, not decoding results. Word lattice uses deterministic prediction-only Levenshtein alignment and may combine words within a sentence; no new words can be generated.'}
  save(ROOT/'evaluation'/f'symbol_{name}_complementarity.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
