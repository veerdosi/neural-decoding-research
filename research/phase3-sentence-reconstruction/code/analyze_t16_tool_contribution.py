"""Score every saved numerical-only sequence from the successful T16 call."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib
from sequence_study import ROOT,save
from sentence_study import distance
from analyze_t16_confirmation import scores,paired

def main():
 spec=json.loads((ROOT/'cases/sentences_t16_confirm.json').read_text());folder=resolve_sandbox(spec['sandbox']);sol=json.loads((ROOT/'evaluation/sentences_t16_confirm.sol.json').read_text());truth={r['id']:r['truth'] for r in sol['rows']};out={}
 for f in ['algorithm_only_predictions.json','balanced_algorithm_only_predictions.json','dtw_algorithm_only_predictions.json']:
  a=json.loads((folder/f).read_text());assert set(a)==set(truth);rows=[{'id':q,'prediction':r['tokens'],'truth':truth[q],'edit_distance':distance(truth[q],r['tokens'])} for q,r in a.items()];out[f]={'scores':scores(rows),'paired_final_Sol':paired(sol['rows'],rows),'sha256':hashlib.sha256((folder/f).read_bytes()).hexdigest(),'rows':rows}
 save(ROOT/'evaluation/t16_within_call_algorithm_contribution.json',{'scope':'Numerical-only sequences authored during the same Sol inference. They inherit Sol-selected calibration intervals, adaptive scoring design and orchestration. Not independent baselines and not a causal decomposition of all Sol contributions. All saved sequence procedures are reported; none selected using target labels. Final Sol interpretation may select evidence across procedures.','procedures':out})

if __name__=='__main__':main()
