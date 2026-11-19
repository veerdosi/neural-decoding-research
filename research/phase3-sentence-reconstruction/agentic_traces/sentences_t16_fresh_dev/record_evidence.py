"""Write the assistant's interpreted reconstruction and its retrieval evidence.

This script does not decode: INTERPRETATIONS are explicit human/model judgments.
Numerical routines retrieve source intervals, never emit a new word sequence.
Intervals below are overlapping evidence windows, NOT estimated word boundaries.
"""
import json
from pathlib import Path

INTERPRETATIONS = [
 ('te7fda903', 'do you have that', [
   ('do',2,14,'c055/c004 starts; targeted do top-two mean 0.364'),
   ('you',14,24,'c039 31:38 and c070 10:16; targeted you 0.242'),
   ('have',17,26,'targeted c061 33:41 = 0.262; c058 29:40 = 0.248; like alternatives 0.168/0.164'),
   ('that',23,36,'c037 102:112, c051 23:32 and c036 43:51')],
   'Moderate confidence. Have is favored by multiple calibration intervals over like; exact short-word boundaries are unresolved.'),
 ('t196443a7', 'are you closer', [
   ('are',2,13,'c066 5:13 and are intervals in c034/c023/c052; targeted are 0.527'),
   ('you',10,21,'c007 12:22, c012 30:41, c070 5:17; targeted you 0.323'),
   ('closer',23,39,'c062 32:47; computer in c058 is a competing neural match')],
   'Closer has one labeled calibration occurrence, so its identity has less independent support.'),
 ('t0ba95c98', 'you are bad', [
   ('you',2,15,'c072 12:27, c010 10:18 and c074 9:19'),
   ('are',8,19,'c072 20:28, c059 19:29 and c074 13:22'),
   ('bad',14,30,'c018 18:32, c034 30:40 and c036 37:47; that is a weaker competing match')],
   'The initial you/are windows overlap. Bad is supported best by c018 and by endings of the other bad sentences.'),
 ('te1c6177b', 'no i am not hungry', [
   ('no',2,11,'c076 initial interval and c043 initial interval; targeted no 0.297'),
   ('i',10,18,'c055/c054 shared i intervals, c020 22:30; targeted i 0.320'),
   ('am',15,23,'c048 29:43, c060 17:25, c020 24:33; targeted am 0.419'),
   ('not',21,30,'c068 40:49 and c047/c058 not intervals; targeted not 0.300'),
   ('hungry',29,43,'c020 71:80, c016 30:44, c069 39:53; targeted hungry 0.481')],
   'The not interpretation covers a separate interval between am and hungry and is supported by negated calibration sentences.'),
 ('td4202301', 'hope you like my music', [
   ('hope',2,18,'c041 prefix 2:21; c041 has no initial i'),
   ('you',11,24,'c041 14:24, c002 27:37, c039 27:35 and c028 25:36'),
   ('like',17,27,'c028 28:40, c045 10:20, c002 34:45'),
   ('my',20,30,'c035 60:69 and c039 44:55, before the music interval'),
   ('music',23,40,'c039 48:63 and c035 62:78')],
   'The shared c028/c039 phrase is interpreted together with the shorter hope-you prefix in c041; no initial i is added.'),
 ('ta84f22ff', 'you have here', [
   ('you',2,12,'c008 2:12, c045 2:13, c007 9:19; targeted you 0.450'),
   ('have',9,21,'targeted c061 33:41 = 0.295 and c071 29:41 = 0.271 for query 11:21; like 0.186/0.131'),
   ('here',19,34,'c046 44:52, c024 48:57, c078 71:80, c063 59:68, c077 53:60')],
   'Lowest confidence on count. You-have-it-here is a plausible alternative, but no distinct it interval is supported across multiple it examples. The final reconstruction keeps the three better-supported words, despite the awkward wording.'),
 ('tc14fc0db', 'i hope they are comfortable', [
   ('i',2,12,'c046 3:14, c071 4:9 and other i-hope prefixes'),
   ('hope',5,21,'c046 7:26, c002 15:29, c062 8:23, c005 20:29'),
   ('they',14,27,'c046 17:33, c062 17:30 and c042 10:19'),
   ('are',20,33,'c042 12:24, c062 22:34, c072 25:34 and c053 27:34'),
   ('comfortable',26,45,'c029 65:81, c013 34:53, c010 23:41; local cosine similarity peaks around 0.75')],
   'The i-hope-they-are prefix is recombined with the comfortable ending, independently repeated in three calibration recordings.'),
 ('t49dd1ce6', 'i hope they do not need my family', [
   ('i',2,12,'c046 3:14, c078 8:22 and c071 4:18'),
   ('hope',5,21,'c062 7:21, c046 6:22 and c035 11:24'),
   ('they',14,27,'c014 16:30 and c050 40:53'),
   ('do',20,30,'c050 46:57, c058 17:27 and c035 32:46'),
   ('not',26,36,'c050 53:63, c056 35:44 and c035 42:54'),
   ('need',29,42,'c056 37:48, c035 46:62, c015 23:32 and c051 15:23'),
   ('my',35,45,'c078 30:38 and c039 43:51; weaker than adjacent word evidence'),
   ('family',41,57,'c071 60:73, c043 19:30, c047 14:25, c065 10:21 and c078 42:53')],
   'The central do-not-need phrase and family ending have overlapping support. My is supported less strongly than the other words.'),
]

def main():
    meta=json.loads(Path('inputs.json').read_text())
    local=json.loads(Path('local_drop2.json').read_text())
    dtw=json.loads(Path('local_dtw_drop2.json').read_text())
    cal={c['id']:c for c in meta['calibration']}
    vocab={w for c in cal.values() for w in c['tokens']}
    predictions=[];evidence=[]
    for qid,sentence,windows,note in INTERPRETATIONS:
        tokens=sentence.split(); assert all(w in vocab for w in tokens)
        assert not any(tokens==c['tokens'] for c in cal.values())
        predictions.append({'id':qid,'tokens':tokens})
        entries=[]
        for word,s,e,interpretation in windows:
            entries.append({'word':word,'query_evidence_window':[s,e],
                            'interpretation':interpretation,
                            'nearby_retrieval_rows': [r for r in local[qid] if abs(r['start']-s)<=2 and r['end']-r['start']==8],
                            'nearby_dtw_rows': [r for r in dtw[qid] if abs(r['start']-s)<=2 and r['end']-r['start']==10]})
        evidence.append({'id':qid,'tokens':tokens,'note':note,'words':entries})
    assert [p['id'] for p in predictions]==[q['id'] for q in meta['queries']]
    Path('final_interpreted_predictions.json').write_text(json.dumps({'predictions':predictions},indent=2)+'\n')
    Path('reconstruction_evidence.json').write_text(json.dumps({
        'provenance':'Only inputs.json and neural_inputs.npz plus the supplied local numerical helpers were used. No external labels, network, trained encoder, or trained neural-to-word weights.',
        'numerical_results':'Smoothing, calibration-only PCA, removal of components 1 and 2, cosine interval retrieval, and slope-constrained subsequence DTW determine numerical similarities and alignment paths, not words or counts.',
        'interpretation':'All assembled predictions and word associations are assistant interpretations of overlapping labeled calibration evidence. No numerical procedure generated a novel token sequence; therefore there is no algorithm-only decoded sequence to save.',
        'targeted_comparison':'targeted_evidence.py compares complete approximate word intervals manually associated from calibration overlap. Its exemplar boundaries are interpretation, not true boundaries. Targeted scores are conditional on that limited exemplar set and are not probabilities.',
        'interval_convention':'Zero-based row indices, end exclusive; overlapping evidence windows are not word boundaries.',
        'queries':evidence},indent=2)+'\n')
    print('Saved 8 interpreted predictions and structured evidence; IDs, vocabulary, and novelty checks passed.')

if __name__=='__main__': main()
