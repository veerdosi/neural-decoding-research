"""Independent call, matched recording shuffle, and text-only attribution."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib
from sequence_study import ROOT,save
from analyze_t16_confirmation import scores,paired

def read(p):return json.loads(p.read_text())
def main():
 first=read(ROOT/'evaluation/sentences_t16_confirm.sol.json');repeat=read(ROOT/'evaluation/sentences_t16_confirm_repeat.sol.json');shuffle=read(ROOT/'evaluation/sentences_t16_confirm_shuffle.sol.json');text=read(ROOT/'evaluation/neural_text_correction_t16.sol.json');strict=set(read(ROOT/'evaluation/t16_confirmation_comparison.json')['strictly_novel_target_ids']);full=read(ROOT/'evaluation/algorithm_sentence_baselines/full_evidence/t16cal_sentences_t16_confirm_results.json');lookup={r['id']:r for r in repeat['rows']}
 a=ROOT/'prompts/sentences_t16_confirm.txt';b=ROOT/'prompts/sentences_t16_confirm_repeat.txt';assert a.read_bytes()==b.read_bytes();c=read(ROOT/'cases/sentences_t16_confirm.json');d=read(ROOT/'cases/sentences_t16_confirm_repeat.json');assert c['input_sha256']==d['input_sha256']
 donor=[{'id':r['id'],'prediction':r['prediction'],'truth':r['donor_truth'],'edit_distance':r['donor_edit_distance']} for r in shuffle['rows']]
 out={'first':scores(first['rows']),'repeat':scores(repeat['rows']),'repeat_strict':scores([r for r in repeat['rows'] if r['id'] in strict]),'repeat_vs_full_feature_numerical':paired(repeat['rows'],full['rows']),'identical_prompt_and_input_hashes':True,'same_prediction_sequences':sum(r['prediction']==lookup[r['id']]['prediction'] for r in first['rows']),'both_exact':sum(r['prediction']==r['truth'] and lookup[r['id']]['prediction']==r['truth'] for r in first['rows']),'scope':'Two independent model calls on the same32 recordings, not64 independent samples. Adaptive numerical procedures may differ.','shuffled':{'nominal':scores(shuffle['rows']),'donor':scores(donor),'scope':'Independent tool-assisted inference with duration-matched recordings reassigned; no target text supplied.'},'text_only':{'scores':scores(text['rows']),'final_neural_call_vs_text_only':paired(first['rows'],text['rows']),'scope':'Post-inspection development comparison. Given last saved numerical word sequence and calibration transcripts, no query neural evidence. Inherits upstream Sol numerical construction; final-stage comparator only.'}}
 for name in ['sentences_t16_confirm','sentences_t16_confirm_repeat','sentences_t16_confirm_shuffle','neural_text_correction_t16']:
  inv=read(ROOT/'responses'/f'{name}.invocation.json');events=[json.loads(l) for l in (ROOT/'responses'/f'{name}.events.jsonl').read_text().splitlines()];out.setdefault('call_costs',{})[name]={'wall_seconds':inv['wall_seconds'],'usage':[e['usage'] for e in events if e['type']=='turn.completed']}
 save(ROOT/'evaluation/t16_replication_and_controls.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
