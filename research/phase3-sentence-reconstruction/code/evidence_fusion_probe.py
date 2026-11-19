"""One exploratory fusion call; preserve exactly which decoders supply words."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import hashlib,json
import numpy as np
from sequence_study import ROOT,save
from discovery_probes import BASE,compress,write_case

def main():
 name='neural_evidence_fusion_t16';source='sentences_t16_confirm';spec=json.loads((ROOT/'cases'/f'{source}.json').read_text());meta=json.loads((resolve_sandbox(spec['sandbox'])/'inputs.json').read_text());z=np.load(BASE/'neural_alphabet_t16_confirm/symbol_codes_for_analysis.npz');symbol=json.loads((ROOT/'responses/neural_alphabet_t16_confirm.json').read_text());numeric_path=ROOT/'evaluation/algorithm_sentence_baselines/full_evidence/t16cal_sentences_t16_confirm_predictions_locked.json';numeric=json.loads(numeric_path.read_text());a={r['id']:r['tokens'] for r in numeric['predictions']};b={r['id']:r['tokens'] for r in symbol['predictions']}
 prompt='''Reconstruct the actual attempted-speech sentences by integrating complementary evidence. You receive80 labelled calibration neural-symbol strings,32 unlabelled query symbol strings, and two preliminary word sequences for each query. The symbols were obtained by calibration-only smoothing/PCA and unlabeled clustering: S00..S63 are arbitrary neural states, not predecoded phonemes or letters. A colon gives consecutive approximately100-ms frames. Word boundaries and true query word counts are not supplied.

Proposal A is an automatic decoder using the full neural features, weak word alignment, interval retrieval and a calibration text prior. Proposal B is an independent language-model inference from these neural symbols. Both are unverified hypotheses, NOT labels. Neither is designated correct. Their errors may differ. Use the calibration relationships and query motifs to decide which words or combinations the neural evidence supports. Do not just pick the more fluent sentence or mechanically vote. You may retain one proposal, compose parts of both, or infer another reconstruction. Keep an existing proposal when the added evidence does not justify changing it.

This is an evidence-fusion comparison: a conventional numerical decoder explicitly supplies Proposal A. It is not an experiment claiming that you independently performed all neural decoding. Do not use tools, files, code, browsing or delegates. Reason only from the material below. Return only JSON with predictions (id and lowercase tokens), method, and limitations. No targets, method accuracy or correct-proposal indicators are supplied.

CALIBRATION:
'''
 for c in meta['calibration']:prompt+=json.dumps({'id':c['id'],'tokens':c['tokens'],'neural_symbols':compress(z[c['id']])},separators=(',',':'))+'\n'
 prompt+='\nQUERIES AND UNVERIFIED PROPOSALS:\n'
 for q in meta['queries']:prompt+=json.dumps({'id':q['id'],'neural_symbols':compress(z[q['id']]),'proposal_A':a[q['id']],'proposal_B':b[q['id']]},separators=(',',':'))+'\n'
 write_case(name,prompt,{'mechanism':'Neural-symbol evidence integration with independent full-feature numerical proposals','hypothesis':'Different error locations permit reasoning over complementary evidence to improve on either proposal stream.','competing_explanation':'The streams are redundant or reasoning favors fluency and introduces unsupported words.','source_case':source,'query_ids':spec['query_ids'],'calibration_n':80,'query_n':32,'tool_access':False,'evaluation_role':'Exploratory fusion on previously scored32 recordings; development, NOT untouched confirmation. No target text or method correctness supplied to model.','boundary':'Conventional decoder output is supplied explicitly. This is a fusion comparator, not a direct-decoding Sol-only claim.','proposal_hashes':{'numeric':hashlib.sha256(numeric_path.read_bytes()).hexdigest(),'symbol':hashlib.sha256((ROOT/'responses/neural_alphabet_t16_confirm.json').read_bytes()).hexdigest()},'strongest_existing_comparator':'Separate tool-assisted Sol call on same32 recordings; comparison must include it, not only the weaker two proposal streams.'})
 print('Prepared frozen fusion comparison, no query targets in inputs')

if __name__=='__main__':main()
