import json
from pathlib import Path
meta=json.load(open('inputs.json'))
predictions=[
 {'id':'td133fe66','tokens':['what','is','that']},
 {'id':'t91b28a91','tokens':['tell','my','family','i','need','my','computer']},
 {'id':'ta8a65773','tokens':['what','is','right']},
 {'id':'tcd92c110','tokens':['i','have','no','music']},
 {'id':'t7d52b265','tokens':['is','my','nurse','coming','here']},
 {'id':'t50600ea0','tokens':['music','is','here']},
 {'id':'t53cbb8ea','tokens':['how','do','you','have','faith']},
 {'id':'tf75891fc','tokens':['you','do','not','feel','hungry']}
]
notes={
 'td133fe66':{'evidence':['Query 2:17 retrieves c010 2:10 (what is that like); query 11:19 retrieves c010 6:11 at cosine .357 after removing two unsupervised PCs.','Later query 14:24 retrieves c098 6:16 (here is that), providing independent is/that context.'], 'uncertainty':'Length and boundaries of the first two words remain approximate.'},
 't91b28a91':{'evidence':['Early and middle intervals retrieve c078, c027, c109, c114 and c101 at successive portions of tell my family.','Query 35:45 retrieves c081 3:11 and c022 5:11, consistent with i need; query 41:56 retrieves c054 10:23 and c080 21:29, followed by later computer-containing intervals.'], 'uncertainty':'First word and the transition after family are less distinct than the recurring family and computer intervals.'},
 'ta8a65773':{'evidence':['Prefix retrieval selects c087 and c010 (what...) for query 2:7; middle intervals select is-containing portions of c103 and c090.','c043 (are they right) ranks first under all three full-dimensional alternative preprocessings; query 20:25 retrieves c043 15:19 and c091 29:33.'], 'uncertainty':'Low confidence in the first word and the identity of the final short adjective; good and clean also have local support.'},
 'tcd92c110':{'evidence':['Initial 2:7 retrieves c083 and c059 at their initial i; query 5:15 and 8:18 independently retrieve have/no portions of c051 and c004.','Query 20:30 retrieves c086 16:24, c041 14:21 and c081 12:21, all containing music at these late positions.'], 'uncertainty':'No time boundaries supplied; no is inferred from the overlapping have no phrase correspondences.'},
 't7d52b265':{'evidence':['Prefix 2:18 retrieves c072 2:18 (is my nurse thirsty) and c011 2:17 (tell my nurse i am not hungry).','Query 17:27 and 20:30 retrieve c029, c009 and c012 in coming-containing portions; query 26:36 retrieves c047 16:25, followed by here matches in c047 and c060.'], 'uncertainty':'Low confidence: tell my nurse i am right here is a competing interpretation. An extra right before here is possible but lacks a distinct interval beyond the shared here trajectory.'},
 't50600ea0':{'evidence':['Initial 2:7 retrieves c028 2:6 (music please); query 2:12 also retrieves c016 5:14 (my music is my hope).','Query 11:21 and 14:24 retrieve is/here intervals of c100 (nurse is here) and c107 (that is here), while c075 supports the here ending.'], 'uncertainty':'Initial music has fewer examples than the ending is here.'},
 't53cbb8ea':{'evidence':['Initial 2:7 prefix retrieval ranks c082 and c007 (how do you...) among its best matches; middle query windows retrieve do-containing intervals of c083 and c080.','Query 11:21 retrieves c075 5:15 and c021 14:20, providing short-pronoun context; query 20:30 retrieves c004 8:15 and c097 18:27 around have.','Query 29:39 retrieves c002 13:21 and c003 10:20, independently supporting faith.'], 'uncertainty':'Low confidence in the initial three words; they do not have faith is a competing reading. These retrievals do not establish a uniquely identifiable prefix.'},
 'tf75891fc':{'evidence':['Prefix 2:14 retrieves c080 2:14 (you do not like my computer), and remains the strongest prefix match through length 16.','Middle query 14:24 and 17:27 retrieve c101 37:45 and 40:49, around feel; query 23:33 and 26:36 retrieve hungry portions of c015, c101 and c115.'], 'uncertainty':'Prefix subject is weaker than do not and the feel hungry ending.'}
}
vocab=set(t for c in meta['calibration'] for t in c['tokens'])
assert len(vocab)==50
assert [p['id'] for p in predictions]==[q['id'] for q in meta['queries']]
assert all(t in vocab and t==t.lower() for p in predictions for t in p['tokens'])
assert all(p['tokens']!=c['tokens'] for p in predictions for c in meta['calibration'])
result={
 'predictions':predictions,
 'method':'Applied Gaussian temporal smoothing, framewise cosine normalization, calibration-only unsupervised PCA, and alternative component removal/whitening. Retrieved whole recordings and short intervals using cosine similarity, linear time resampling and slope-constrained subsequence DTW. Inferred words and order from overlapping retrieved calibration transcripts, including have/no plus music, music plus is/here, tell/my/family plus need/my/computer, and do/not plus feel/hungry. No supervised model, labeled neural prototypes, gradient updates or fine-tuning were used. Scripts, scores, paths and query-specific evidence are retained in this directory, including inference_evidence.json.',
 'limitations':'Best-effort reconstructions, not verified target text. Shared onset/offset responses, weak local similarities and variable timing make boundaries and short words ambiguous. ta8a65773, t7d52b265 and t53cbb8ea are particularly uncertain; the latter two admit competing prefixes or an additional word. Retrieval similarity is contextual evidence, not ground truth.'
}
Path('inference_evidence.json').write_text(json.dumps({'predictions':predictions,'notes':notes,'constraints':'All numerical representations and alignments computed without calibration transcript labels; labels interpreted only after retrieval. No class templates were averaged or trained.'},indent=2))
Path('reconstruction.json').write_text(json.dumps(result,indent=2))
print('Saved reconstruction.json and inference_evidence.json. Validated 8 query IDs, 50-word vocabulary membership, lowercase tokens, and unseen transcript combinations.')
