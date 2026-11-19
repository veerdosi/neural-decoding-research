"""Separate linguistic correction from final-stage neural evidence inspection."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json,pathlib
from sequence_study import ROOT
from discovery_probes import write_case

def main():
 name='neural_text_correction_t16';source='sentences_t16_confirm';spec=json.loads((ROOT/'cases'/f'{source}.json').read_text());folder=resolve_sandbox(spec['sandbox']);meta=json.loads((folder/'inputs.json').read_text());p=folder/'dtw_algorithm_only_predictions.json';raw=json.loads(p.read_text())
 prompt='''This is a TEXT-ONLY language-correction comparator for neural speech reconstruction. You receive80 calibration transcripts and32 unverified automatic word sequences. Each query's preliminary sequence was produced by a numerical neural interval decoder. You receive NO query neural recording, neural symbols, neural similarity scores, true sentence length, final model reconstruction, or evaluation targets.

Infer the most likely actual word sequence using the preliminary words, the calibration corpus and general linguistic knowledge. This is transcription recovery, not copyediting: unusual but plausible wording should be preserved. The actual vocabulary consists of words in calibration. Calibration whole sentences are examples, not a target menu; targets may be new combinations. Correct likely omissions, substitutions and word-order errors when justified. Return your best reconstruction for every ID. No output is designated correct and no accuracy feedback is provided.

Do not use tools, files, code, browsing or delegates. Reason only from the text below. Return only JSON with predictions (id and lowercase tokens), method and limitations. This is explicitly not an independent neural-decoding experiment: the numerical input already inherits the earlier Sol call's calibration-reference construction and computation.

CALIBRATION TRANSCRIPTS:
'''
 for c in meta['calibration']:prompt+=json.dumps({'id':c['id'],'tokens':c['tokens']},separators=(',',':'))+'\n'
 prompt+='\nUNVERIFIED AUTOMATIC WORD SEQUENCES:\n'
 for q in spec['query_ids']:prompt+=json.dumps({'id':q,'preliminary_tokens':raw[q]['tokens']},separators=(',',':'))+'\n'
 write_case(name,prompt,{'mechanism':'Text-only correction of the last saved numerical sequence stage','hypothesis':'Ordinary language correction may explain the final-stage improvement without additional neural inspection.','competing_explanation':'Neural evidence beyond the preliminary words is needed to resolve omitted or ambiguous content.','source_case':source,'query_ids':spec['query_ids'],'calibration_n':80,'query_n':32,'tool_access':False,'evaluation_role':'Post-inspection development attribution probe, same32 recordings; no target text or correctness feedback supplied.','boundary':'No raw neural evidence is supplied. Inherits Sol-authored numerical decoder upstream; not a Sol-only direct-decoding experiment.','numeric_stage':'Last saved DTW algorithm-only sequence, not a query-wise oracle choice among variants.','numeric_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 print('Prepared text-only attribution comparison')

if __name__=='__main__':main()
