"""Audit utterance novelty, calibration-fragment reuse and verbatim tool outputs."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import json,pathlib,sys
import numpy as np
from sequence_study import ROOT,save
from sentence_study import distance

def fragments(target,calibration):
 n=len(target);best=[10**6]*(n+1);best[0]=0;path=[None]*(n+1);longest=0
 for start in range(n):
  for end in range(start+1,n+1):
   part=target[start:end]
   matches=[i for i,c in enumerate(calibration) if any(c[j:j+len(part)]==part for j in range(len(c)-len(part)+1))]
   if not matches:continue
   longest=max(longest,end-start)
   if best[start]+1<best[end]:best[end]=best[start]+1;path[end]=(start,matches[0])
 pieces=[];end=n
 while end and path[end]:
  start,index=path[end];pieces.append({'tokens':target[start:end],'calibration_index':index});end=start
 return {'complete_target_is_calibration_span':longest==n,'longest_seen_span_words':longest,'minimum_calibration_fragments':best[n] if best[n]<10**6 else None,'one_minimum_cover':list(reversed(pieces))}

def main(name):
 result=json.loads((ROOT/'evaluation'/f'{name}.sol.json').read_text());meta=json.loads((ROOT/'cases'/f'{name}.json').read_text())
 folder=resolve_sandbox(meta.get('sandbox', 'agentic_neural_sandbox'));inputs=json.loads((folder/'inputs.json').read_text());cal=[c['tokens'] for c in inputs['calibration']];rows=[]
 for row in result['rows']:
  y=row['truth'];pred=row['prediction'];novelty=fragments(y,cal);oracle=min(distance(y,c) for c in cal)
  rows.append({'id':row['id'],'trial':row['trial'],'truth':y,'prediction':pred,'edits':row['edit_distance'],'exact':pred==y,'calibration_copy_oracle_edits':oracle,**novelty})
 strict=[r for r in rows if not r['complete_target_is_calibration_span']];out={'name':name,'n':len(rows),'exact_sentences':sum(r['exact'] for r in rows),'word_error_rate':sum(r['edits'] for r in rows)/sum(len(r['truth']) for r in rows),'strict_novel_sequence_n':len(strict),'strict_novel_sequence_exact':sum(r['exact'] for r in strict),'strict_novel_sequence_WER':sum(r['edits'] for r in strict)/sum(len(r['truth']) for r in strict),'whole_calibration_copy_oracle_edits':sum(r['calibration_copy_oracle_edits'] for r in rows),'whole_calibration_copy_oracle_WER':sum(r['calibration_copy_oracle_edits'] for r in rows)/sum(len(r['truth']) for r in rows),'oracle_scope':'Oracle uses true query text only after prediction for descriptive attribution; never a usable decoder or a model input. It cannot be used to tune the method.','rows':rows,'conditional_algorithm_outputs':{}}
 for p in folder.glob('*algorithm_only_sequences.json'):
  x=json.loads(p.read_text());q=x.get('queries',{});edits=[];exact=0;details=[]
  for row in rows:
   choices=q.get(row['id'],[])
   if not choices or not isinstance(choices,list) or 'tokens' not in choices[0]:continue
   pred=choices[0]['tokens'];e=distance(row['truth'],pred);edits.append(e);exact+=int(pred==row['truth']);details.append({'id':row['id'],'prediction':pred,'edits':e,'Sol_edits':row['edits']})
  if len(edits)==len(rows):out['conditional_algorithm_outputs'][p.name]={'n':len(rows),'edits':sum(edits),'WER':sum(edits)/sum(len(r['truth']) for r in rows),'exact':exact,'net_word_edits_reduced_by_Sol':sum(edits)-sum(r['edits'] for r in rows),'scope':'Top numerical path from the algorithm authored/selected by Sol during this call. This already includes Sol method construction and calibration interpretation; it isolates only the remaining final interpretation, not all frontier-model contribution.','method_original':x.get('method'),'rows':details}
 out['provenance_wording']={'calibration_word_boundaries':'Inferred by procedures, not independently observed ground truth.','human_interpretation':'Where model-authored files call the final choices human interpretation, that is inaccurate wording: the Codex event trace records Sol writing those choices; no human supplied the predictions.'}
 save(ROOT/'evaluation'/f'{name}.structure.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['rows','conditional_algorithm_outputs','provenance_wording']},indent=2))

if __name__=='__main__':
 for n in sys.argv[1:]:main(n)
