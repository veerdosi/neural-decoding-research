"""Save the final HUMAN-INTERPRETED reconstruction separately from every
algorithm-only assembly. Evidence spans are approximate row indices, not
provided token boundaries. Alternatives and scores remain in separate files.
"""
import json
from pathlib import Path
meta=json.load(open('inputs.json'))
choices={
 'tda989373':('they need what','low',[(2,10,'they','c104:2-6; c101:18-22; c061:13-17'),(8,16,'need','c009:7-11; c055:5-9; competing do in c079/c080'),(14,24,'what','c062:10-18; c098:7-16')]),
 'tdd25032b':('tell my family i do not feel thirsty','medium',[(2,22,'tell my family','c114:2-19; c027:5-22; c078:5-20'),(20,36,'i do','c079:8-20; c083:12-25'),(32,45,'not','c079:17-29; c101:37-46'),(38,55,'feel','c015:6-16; c020:9-20; c001:17-25'),(53,67,'thirsty','c072:16-30; optional very unresolved')]),
 't5ee44a7f':('my music is here','medium',[(2,15,'my music','c016:2-14; c093:4-12'),(11,24,'is','c100:5-18; c047:18-26; estimated transition to here'),(17,30,'here','c058:29-39; c047:22-31; c093:14-21; c108:29-37')]),
 'tdf8f9fae':('i hope you have','low',[(2,16,'i hope','c001:2-15; c005:2-14; c097:2-14'),(14,22,'you','c097:15-21; competing they in c025:11-22'),(20,25,'have','c066:19-23; c097:17-23; weak terminal word')]),
 't4f16b000':('is my faith','low',[(2,15,'is my','c091:3-14; c039:4-12'),(11,21,'faith','c112:15-24; c003:2-10; music/family remain alternatives; candidate score favors faith under full features')]),
 't40a0a424':('no they do not feel very comfortable','low',[(2,15,'no they','c018:2-14; competing no i in c079:2-15'),(8,24,'do not','c083:16-31; c080:6-21; c079:9-25'),(20,33,'feel','c079:31-43; c020:10-19'),(26,39,'very','c079:35-50; phrase boundary uncertain'),(32,46,'comfortable','c017:29-41; c114:37-47; c103:18-30; c039:29-38')]),
 'tf256b8ed':('tell my nurse is very good','medium-low',[(2,18,'tell my nurse','c011:3-18; c072:7-18; c049:3-12'),(11,24,'nurse is','c049:6-20; c100:4-18; is inclusion uncertain'),(23,36,'very good','c008:15-28; c049:23-37'),(29,39,'good','c008:22-29; c091:24-33; c049:30-37')]),
 't12be1627':('i like it','low',[(2,15,'i like','c041:2-13; competing they at onset in c009'),(8,21,'like it','c036:3-16; c088:18-31 supports competing like that; terminal it is uncertain')]),
 't42cfafec':('i hope you have faith','medium-low',[(2,15,'i hope','c001:2-14; c097:2-15; c099:3-15'),(8,21,'hope you','c097:10-22; competing i in c117:13-21'),(17,30,'have faith','c002:6-19; c003:2-15; c112:11-23'),(23,31,'faith','c003:11-15; c002:13-20; c004:22-26')]),
 'te3715f88':('tell my family good','medium-low',[(2,15,'tell my','c114:2-10; c061:19-27; c109:7-15; competing you need'),(11,27,'my family','c101:5-18; c108:15-28; c091:13-25'),(20,34,'good','c091:18-32; c049:23-38; c008:15-28')]),
 'tb8083852':('my family is good','low',[(2,13,'my family','c027:13-24; c101:13-25; c108:19-29'),(8,21,'is','c090:14-25; competing do/not-like fragments'),(17,30,'good','c048:12-22; c091:22-34; c049:28-37; c008:22-30; optional not unresolved')]),
 't9024802d':('do not need it','low',[(2,18,'do not','c111:5-18; c080:7-20'),(14,24,'need','c083:27-38; competing is that/like that'),(20,28,'it','c083:34-41; competing that in c088:23-29; candidate comparisons favor need it')]),
 't2b08e042':('my faith is good','medium-low',[(2,15,'my faith','c031:2-12; c003:3-18; competing my nurse in c049:2-15'),(11,21,'is','weak short function word; c056:12-23 gives alternative they are'),(17,28,'good','c008:20-29; c049:28-41; c048:12-20')]),
 't62254eda':('they feel very comfortable','medium-low',[(2,15,'they','c025:11-21; c015:2-14; competing that is'),(8,21,'feel','c015:7-19; c079:31-45; c101:40-50'),(14,25,'very','c079:37-50; c039:24-34; competing not'),(20,31,'comfortable','c103:20-29; c079:44-55; c039:29-41; c114:41-52')]),
 't9db27598':('is my nurse comfortable','medium',[(2,19,'is my nurse','c072:2-19; c011:5-20; c049:3-15'),(17,33,'comfortable','c017:29-42; c114:37-53; c103:16-24; possible intervening not/very unresolved')]),
 'tef25d0d3':('how do i have faith','low',[(2,13,'how do','c007:4-12; c082:4-9; c061:4-9; competing they do'),(14,24,'i have','c117:18-25; c004:8-14; competing not in c079:21-31'),(23,34,'faith','c003:7-18; c004:20-29; c061:30-37 gives competing family')])
}
pred=[{'id':q['id'],'tokens':choices[q['id']][0].split()} for q in meta['queries']]
vocab={w for c in meta['calibration'] for w in c['tokens']};known={tuple(c['tokens']) for c in meta['calibration']}
assert len(pred)==len(choices)==16
assert all(set(p['tokens'])<=vocab and tuple(p['tokens']) not in known for p in pred)
method='Calibration-only PCA after smoothing, removal of the first two components, cosine interval retrieval, and slope-constrained subsequence DTW. Overlapping calibration transcripts were interpreted to compose each reconstruction. Deterministic calibration interval alignment, exemplar assemblies, and candidate comparisons are saved separately from these final interpreted tokens.'
limitations='Word counts and boundaries were inferred. Short words, terminal words, and several short queries remain ambiguous. Reconstructions are best estimates, not verified transcripts. Evidence, approximate spans, alternatives, and reproducible scripts are saved in this directory; algorithm-only sequences are separate.'
result={'predictions':pred,'method':method,'limitations':limitations}
Path('final_reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
evidence={'provenance':'Only inputs.json and neural_inputs.npz in this directory were used. No network, external labels, external weights, or delegated agents. Final choices are human interpretation, not the raw numerical assembly. Calibration cuts are inferred and can be inaccurate.','query_evidence':[{'id':qid,'tokens':sentence.split(),'confidence':confidence,'overlapping_support':[{'query_rows':[s,e],'interpreted_fragment':frag,'calibration_matches':support} for s,e,frag,support in spans]} for qid,(sentence,confidence,spans) in choices.items()],'reproduction':['OPENBLAS_NUM_THREADS=1 /Users/veerdosi/Documents/Codex/2026-11-18/go/work/venv/bin/python prepare_pca.py','local_retrieval.py drop2 pca32drop2','local_dtw.py drop2','align_templates.py','assemble_exemplars.py','align_temporal_exemplars.py','assemble_temporal.py','assemble_phrases.py','score_candidates.py','score_candidates_pca32.py','finalize_reconstruction.py'],'numerical_evidence':['local_drop2.json','local_pca32drop2.json','local_dtw_drop2.json','edge_drop2.json','global_drop2_evidence.json','calibration_alignment.json','temporal_calibration_alignment.json','interpreted_candidate_scores.json','interpreted_candidate_scores_pca32.json'],'automatic_outputs':['algorithm_only_sequences.json','temporal_algorithm_only_sequences.json','phrase_algorithm_only_sequences.json']}
Path('reconstruction_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('Saved 16 interpreted reconstructions and supporting evidence; verified query IDs, lowercase calibration vocabulary, and novel combinations.')
