import json
from pathlib import Path
m=json.loads(Path('inputs.json').read_text())
chosen=[
('tda989373','they need what'),
('tdd25032b','tell my family i do not feel thirsty'),
('t5ee44a7f','my music is here'),
('tdf8f9fae','i hope they have'),
('t4f16b000','goodbye coming'),
('t40a0a424','no they do not feel comfortable'),
('tf256b8ed','tell my nurse is very good'),
('t12be1627','i like it'),
('t42cfafec','i hope you have faith'),
('te3715f88','need my family good'),
('tb8083852','family is not good'),
('t9024802d','do not need it'),
('t2b08e042','my faith they are good'),
('t62254eda','that is very comfortable'),
('t9db27598','is my nurse not comfortable'),
('tef25d0d3','they do not have faith')]
notes={
 'tda989373':('low','Initial they has multiple matches (c104, c101, c009). The 2:17 they-need alignment to c009 narrowly exceeds they-do to c101. Terminal what has c082/c062 suffix support; that is close.'),
 'tdd25032b':('moderate','Tell-my-family is supported by c114/c027/c078 at query 2:22. I-do-not by c083/c079/c101 at 20:40. Feel by c020/c015/c001 around 40:56; thirsty by c072/c099 at 53:68. The repeated feel in the automatic unigram output is interpreted as one stretched word.'),
 't5ee44a7f':('moderate','My-music prefix matches c016 at 2:15, with music also localized using c041/c093. Is/here overlaps c047/c100/c093. Here is strongly supported by numerous suffix references. Bring is a confusable singleton; full prefix favors my music.'),
 'tdf8f9fae':('moderate','I-hope prefix c001/c005/c097; they specifically c025 9:15 and c001 13:18 at query 11:18. Terminal have is the strongest reverse-edge match to c066, 0.26 for five bins. No extra object inferred.'),
 't4f16b000':('very low','Goodbye prefix c062 is the strongest longer anchored match. Coming is supported by c047 internal interval and scores best as an extracted exemplar; faith and please have competing suffix evidence. This is a weak two-word interpretation.'),
 't40a0a424':('moderate','No-they c018 is slightly stronger than no-I c079 at 2:11. Do-not c080/c083/c079 overlaps 11:22. Feel c020 at 22:31 and comfortable c017/c114/c039 at 31:47. Adding very lowers the composed evidence score; short-word ambiguity remains.'),
 'tf256b8ed':('moderate','Tell-my-nurse matches c011 at 2:16, supplemented by c049/c072 nurse matches. Query 16:21 matches is in c071; 21:39 matches very-good in c008/c049. No sufficiently supported I-am is inserted.'),
 't12be1627':('low','I-like is retrieved as a contiguous prefix from c041 and like independently from c036/c111/c088. Terminal it has c065/c083 suffix support. Initial they is a close competing interpretation, largely driven by c009.'),
 't42cfafec':('moderate','I-hope from c001/c040; hope-you from c097; you-have-faith from c002 and have-faith from c003/c112. Have is brief and was swallowed by the faith interval in some automatic assemblies. Repeated phrase overlaps support retaining it.'),
 'te3715f88':('low','Initial need matches c110 6:12 and c063 5:11, narrowly exceeding tell in c114; my matches c061/c027/c078, family c101/c039/c108, and good c091/c008. Need-my-family-good has the best scored supported composition. Tell-my-family-good remains close.'),
 'tb8083852':('very low','First interval matches medial family in c027/c108/c101; an additional initial my has little support. Is-not is strong in c090 at query 7:20, and good in c048/c091 at 20:30. Thirsty and my can produce higher isolated-template scores at the beginning, but are not consistent across the original interval retrievals.'),
 't9024802d':('low','Do-not is a strong medial match to c111 5:17, at query 2:15. Need-it matches c083 29:42 in the second half; final it has c104/c083 support. Like-that and right-here are competing but weaker compositions.'),
 't2b08e042':('very low','My-faith matches c031, with faith independently supported by c003/c112. Nurse in c049 is confusable. Middle they-are matches c056 13:23 and are in c005/c006; good has c008/c049 suffix support. Retain the two short middle words based on overlapping phrase evidence despite nonstandard syntax; is and are alone remain alternatives.'),
 't62254eda':('moderate','Initial that/is is supported by c103/c107/c087 edges. Very-comfortable overlaps c079/c039/c115, with comfortable independently strong in c103/c114/c078. My is a brief competing local match; repeated prefix evidence favors is.'),
 't9db27598':('moderate','Is-my-nurse is strongly supported by c072 2:17, plus c011/c049 nurse intervals. Query 14:24 matches not-comfortable in c103. Comfortable has c017/c114 support. You at the beginning and absence of not remain close alternatives.'),
 'tef25d0d3':('low','They-do matches c101 18:31 at query 2:12; not c079 at 12:20; have c117 at 20:24; faith c003 at 24:35, supplemented by c004/c112. How-do prefix and need in c075 are confusable; they-do-not has the stronger longer overlapping phrase evidence than selecting those words independently.')
}
ranks=json.loads(Path('candidate_comparison_scores.json').read_text())
evidence=[]
for cid,s in chosen:
 rs=ranks[cid];sel=next(v for v in rs if v['tokens']==s.split())
 evidence.append({'id':cid,'final_interpreted_tokens':s.split(),'subjective_confidence':notes[cid][0],'interpretation':notes[cid][1],'selected_candidate_numerical_evidence':sel,'top_scored_alternatives':rs[:4]})
Path('interpretation_evidence.json').write_text(json.dumps(evidence,indent=2))
result={
 'predictions':[{'id':cid,'tokens':s.split()} for cid,s in chosen],
 'method':{
  'analysis':'Calibration-only PCA, removal of the first two components, Gaussian smoothing, cosine interval retrieval, and slope-constrained DTW. Calibration transcript overlaps and extracted exemplars informed manual word identity, timing, and composition.',
  'distinction':'Numerical assemblies and candidate scores are evidence; final predictions are interpreted reconstructions. No external decoder, language model scores, or query labels were used.',
  'evidence':'interpretation_evidence.json',
  'algorithm_outputs':['algorithm_only_sequences.json','exemplar_assembly_sequences.json','phrase_assembly_sequences.json'],
  'reproduction':'run_analysis.sh'
 },
 'limitations':['Word boundaries were inferred and remain uncertain. Similar short-word trajectories can change token identity or count.','Lowest confidence: t4f16b000, tb8083852, and t2b08e042. Other close alternatives and supporting intervals are recorded in interpretation_evidence.json.']
}
vocab={w for c in m['calibration'] for w in c['tokens']}
assert [p['id'] for p in result['predictions']]==[q['id'] for q in m['queries']]
assert all(p['tokens'] and all(w in vocab and w==w.lower() for w in p['tokens']) for p in result['predictions'])
assert all(p['tokens']!=c['tokens'] for p in result['predictions'] for c in m['calibration'])
Path('final_reconstruction.json').write_text(json.dumps(result,indent=2))
print('Saved and validated 16 interpreted reconstructions and evidence records.')
