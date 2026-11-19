"""Human interpretation/composition of saved, calibration-only numerical evidence.
These explicit decisions are not an algorithm-only decoder output. The independent
unrestricted numerical sequences are retained in *_algorithm_only.json and
algorithm_only_predictions.json. Candidate lists in compare_candidates.py were
human proposed from retrieval evidence, not supplied query labels.
"""
from compare_candidates import *
FINAL_TEXT={
 'tda989373':'tell my faith',
 'tdd25032b':'no i do not feel comfortable',
 't5ee44a7f':'my faith is good',
 'tdf8f9fae':'they do that',
 't4f16b000':'i like help',
 't40a0a424':'tell my nurse very good',
 'tf256b8ed':'they do not have my faith',
 't12be1627':'i hope they have',
 't42cfafec':'that is very comfortable',
 'te3715f88':'is my nurse comfortable',
 'tb8083852':'do not need it',
 't9024802d':'my music is here',
 't2b08e042':'my family is not good',
 't62254eda':'tell need my family good',
 't9db27598':'i hope i have faith',
 'tef25d0d3':'tell my family i do not feel very thirsty',
}
NOTES={
 'tda989373':'Low confidence. Individual-word retrieval favors tell-my-faith. The prefix also matches is-my in c091 and are in c043. The final interval matches faith in c112 and c003. Kept tell rather than normalizing the unusual phrase.',
 'tdd25032b':'The overlapping c079/c083/c080 matches identify no, i, do, not; c017/c114/c103 agree on comfortable. The middle is a single feel-like interval. A separate very before comfortable scores poorly; the full c079 transcript is not copied.',
 't5ee44a7f':'My-faith is supported by c031, c003 and c112; good by c049/c008/c091. Is is a human interpretation of the intervening short interval and the continuing my-faith-is motif in c031; numerical alternatives include they, it and are.',
 'tdf8f9fae':'They and do are consistent with c104/c009/c101/c080 overlaps. The ending is weak: that and not are close, with word DTW slightly favoring that. No whole calibration sentence was adopted.',
 't4f16b000':'Low confidence at the final word. I-like is supported by c041 and c036. Help is the best lexical DTW candidate at the end; up, tired and it remain plausible. Chose help on numerical support rather than substituting a more familiar phrase.',
 't40a0a424':'Tell-my-nurse is supported by c011, c072 and c049; very-good by c008/c049. The numerical alignment favors five words. Inserting i-am or is lowers the interval agreement; retained the telegraphic phrase.',
 'tf256b8ed':'They-do-not is supported by c101/c080/c079/c083. The latter part combines have from c117/c004, my from c080/c027 and faith from c003/c004. Have versus feel and whether a distinct my is present are uncertain. The six-word version has strong pooled-exemplar support.',
 't12be1627':'I-hope overlaps c001/c005/c097. They is anchored by c025, and the final short interval matches have in c066. I and you are alternative pronouns. The four-word assembly is supported by phrase and word DTW.',
 't42cfafec':'The initial short interval matches that-is in c103/c107 and what-is in c087. Very is anchored in c115/c079, and comfortable agrees across c039/c103/c078/c114. Chose that-is from the overlapping prefix evidence, despite the numerical my alternative.',
 'te3715f88':'The initial is-my-nurse phrase has the strongest long-prefix match to c072, reinforced by c011/c049. The ending overlaps comfortable in c017/c103/c114. Standalone first-word retrieval also suggests you or tell; longer phrase evidence supports is.',
 'tb8083852':'Do-not is anchored by c111. The ending overlaps need-it in c083 and it in c104; need is short and uncertain. Alternatives include a redundant is and like. Retained the two observed phrases do-not and need-it.',
 't9024802d':'My-music is supported by c016/c093. Is-here overlaps c100/c047/c093 and several right-here endings. Is is selected from phrase agreement; numerical word retrieval also suggests it or they.',
 't2b08e042':'The early noun interval overlaps family in c027/c101/c108. Is-not has a strong direct interval match to c090; good agrees across c048/c091/c008. A very short initial my is plausible but uncertain; numerical duration-penalized assembly merges the early region into family and the middle into do.',
 't62254eda':'Low confidence. Tell is supported by c114, need by c110/c063/c059, and my by c039/c101/c061. The later interval overlaps family-good in c091, with support for family from c109/c108. Very is the main alternative to family; going is an alternative to good. Retained the unusual tell-need order because both intervals have direct evidence.',
 't9db27598':'I-hope agrees across c001/c097/c040. The inner i-have motif overlaps c117; have-faith is supported by c002/c003/c112. My is a strong numerical alternative to have, but the longer have-faith phrase supports have. I and you are close.',
 'tef25d0d3':'Tell-my-family agrees across c114/c027/c078; i-do-not across c079/c083/c101. The final region overlaps feel in c015/c020, very in c049/c079, and thirsty in c072/c099. The very boundary is uncertain; an eight-word version without very is competitive. Thirsty is favored over hungry.',
}
if __name__=='__main__':
 predictions=[{'id':q['id'],'tokens':FINAL_TEXT[q['id']].split()} for q in meta['queries']]
 cal={tuple(c['tokens']) for c in meta['calibration']};vocab=set(WORDS)
 assert len(predictions)==len(meta['queries'])==16
 assert all(set(p['tokens'])<=vocab for p in predictions)
 assert all(tuple(p['tokens']) not in cal for p in predictions)
 scores=np.load('word_dtw_scores.npz');dur=np.array([json.load(open('calibration_interval_exemplars.json'))['duration_frames'][w] for w in WORDS]);local=json.load(open('local_dtw_drop2.json'));word_candidates=json.load(open('word_dtw_candidate_comparison.json'));ev={}
 for q,p in zip(meta['queries'],predictions):
  qid=p['id'];spans=scores[qid+'_spans'];ss=scores[qid+'_scores'];value,alignment=force(spans,ss,p['tokens'],q['shape'][0],dur)
  ev[qid]={'final_interpreted_tokens':p['tokens'],'interpretation':NOTES[qid],'forced_word_DTW_alignment':alignment,'forced_score':value,'candidate_comparison':word_candidates[qid],'overlapping_phrase_evidence':[r for r in local[qid] if r['end']-r['start']==10]}
 Path('reconstruction_evidence.json').write_text(json.dumps(ev,indent=2))
 result={'predictions':predictions,'method':'Calibration-only PCA, Gaussian smoothing, removal of the first two components, cosine interval retrieval and slope-constrained subsequence DTW; calibration exemplar alignment and overlapping phrase interpretation. Scripts and evidence are saved locally, including reconstruction_evidence.json. Unrestricted numerical sequences are saved separately in algorithm_only_predictions.json and *_algorithm_only.json.','limitations':['Word counts, boundaries and several short words remain uncertain; these are best reconstructions, not verified transcripts.','Lowest confidence: tda989373, t4f16b000 and t62254eda. Some unusual phrases were retained because their words have direct neural support.']}
 Path('final_predictions.json').write_text(json.dumps(result,indent=2))
 print('Saved 16 interpreted reconstructions; IDs, vocabulary, and novel-combination constraint checked.')
