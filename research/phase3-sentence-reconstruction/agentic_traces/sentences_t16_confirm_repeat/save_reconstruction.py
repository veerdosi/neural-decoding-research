"""Final HUMAN INTERPRETATION of overlapping evidence; not a numerical decoder.
The strings below are the assistant's composition after reviewing saved
retrieval reports, calibration interval matches, and constrained DTW paths.
Numerical-only outputs remain in algorithm_only*_predictions.json.
"""
import json
from pathlib import Path
meta=json.load(open('inputs.json'))
# Ordered query list is taken directly from inputs.json.
reconstructions=[
 ('it music it','2-11 it; 11-18 music; 18-23 it. The final short token is uncertain; c041/c014 support it, while music has a continuing tail.'),
 ('do you like music','2-13 do; 13-21 you; 21-33 like; 33-45 music. Initial do versus they is ambiguous: c004 and c075 both retrieve strongly. The overlapping c022/c038 and c002/c039 fragments establish the remainder.'),
 ('where is that','2-14 where; 14-19 is; 19-30 that. c033/c079 establish the prefix, and c051/c037 establish the final that.'),
 ('i am good here','2-11 i; 11-20 am; 20-27 good; 27-34 here. Middle am also retrieves a c008 nurse interval and is uncertain. Good is supported by overlapping c023/c019/c074 endings, and here by c005/c024/c048.'),
 ('you going','2-12 you; 12-23 going. c005/c008/c074 establish you; c008/c068 support going. No distinct intervening are interval was resolved.'),
 ('i hope they do not music','2-9 i; 9-14 hope; 14-19 they; 19-25 do; 25-32 not; 32-45 music. Short they is supported by c035. There is insufficient separate timing evidence to insert need before music.'),
 ('i like it here','2-11 i; 11-20 like; 20-28 it; 28-35 here. The numerical assembly merges the first two words into my. Short overlapping windows instead support i (c051) and like (c039), followed by the c045 it-here suffix.'),
 ('they like it','2-12 they; 12-21 like; 21-29 it. c064/c024 support they, c002/c039/c031 support like, c041/c014 support it.'),
 ('my family is not very tired','2-12 my; 12-24 family; 24-29 is; 29-38 not; 38-46 very; 46-57 tired. Family-is-not is supported by c043/c047; very by c071/c009/c066; tired by c042. The numerical trailing are is unsupported by consistent overlapping evidence.'),
 ('my hope is my success','2-12 my; 12-18 hope; 18-24 is; 24-32 my; 32-46 success. c067 supplies my-is-my-success structure, while c002/c028/c006 establish hope in place of faith. The weak final bins were not assigned another word.'),
 ('hungry it','2-17 hungry; 17-25 it. The longer first interval matches c016:29-41, so the weak numerical initial do/you alternative is not kept. c014 supports final it, with competing glasses/hope matches.'),
 ('yes nurse are you','2-10 yes; 10-21 nurse; 21-28 are; 28-36 you. Initial yes versus i remains weak. Nurse is supported by c068/c029/c005, are by c074/c052/c053, and you by c012/c008/c044.'),
 ('do they have my glasses','2-14 do; 14-22 they; 22-27 have; 27-33 my; 33-47 glasses. c030 do-they-have overlaps c061 have-my-glasses; c073 independently supports glasses. Have is brief and missed by the unconstrained assembly.'),
 ('you hope my family is good','2-17 you; 17-22 hope; 22-25 my; 25-34 family; 34-40 is; 40-49 good. c008/c005 supply you-hope-my and c000/c078/c043 supply my-family; c019/c023/c032 establish good.'),
 ('my nurse is very tired','2-13 my; 13-23 nurse; 23-28 is; 28-35 very; 35-46 tired. c005/c008/c068 support my-nurse-is. Is versus it is locally ambiguous. c009/c066 establish very and c042 establishes tired.'),
 ('where is my music','2-20 where; 20-24 is; 24-31 my; 31-46 music. The entire first interval follows c079 where, avoiding a spurious split into are/very. c079 supplies where-is-my, c039/c035 supply my-music.'),
 ('i hope you have family','2-19 i; 19-26 hope; 26-30 you; 30-35 have; 35-50 family. c071 supplies i-hope-you-have and c078/c071/c065/c047 establish family. You-have versus my is a weak, compressed transition.'),
 ('i hope they need my music are','2-16 i; 16-21 hope; 21-24 they; 24-32 need; 32-38 my; 38-43 music; 43-53 are. The brief they interval matches c064/c052; need matches c064/c075/c035; my-music matches c035/c039; final are matches c066/c072/c053/c046. The unusual composition follows these intervals.'),
 ('they need feel thirsty','2-9 they; 9-16 need; 16-23 feel; 23-34 thirsty. Need is weak and competes with it/okay; c015/c075 provide supporting intervals. c050/c029 support feel and c066/c012/c050 establish thirsty.'),
 ('it need','2-13 it; 13-22 need. Initial it versus i is uncertain. c026/c027 support it and c077 supports need, with am/my competitors. Both numerical assemblies return it need.'),
 ('i do not have','2-9 i; 9-16 do; 16-22 not; 22-35 have. c058 supplies the full overlapping phrase. The numerical trailing do covers a weak final part of have and was not retained.'),
 ('i tell no family','2-9 i; 9-17 tell; 17-27 no; 27-43 family. Tell is strongly supported by c012. No competes closely with hope (c028/c002) and is (c065), while c074/c076 support no. c071/c078/c065 establish family.'),
 ('how tired are they','2-11 how; 11-23 tired; 23-30 are; 30-37 they. The longer 10-24 window matches c042:19-29, identifying tired rather than splitting it into are/very. c052/c053/c066 support the final are-they phrase.'),
 ('it is not you do','Approximately 0-19 is a weak leading interval with no reliable word assignment; 19-33 it; 33-40 is; 40-51 not; 51-57 you; 57-65 do. c026/c036/c013 support it-is, c076/c068 support not, c063/c028/c071 support you, c070/c011 support do. The leading interval remains a limitation.'),
 ('i like help comfortable i am','2-12 i; 12-20 like; 20-25 help; 25-38 comfortable; 38-42 i; 42-50 am. c028/c039 establish like; short help matches c063/c064, competing with feel. c013/c029/c010 establish comfortable. The final i-am interpretation is weak but has c020/c001/c048 support.'),
 ('yes i am very hungry','2-13 yes; 13-19 i; 19-22 am; 22-31 very; 31-43 hungry. c060 supports yes-i-am-very; hungry is supported independently by c069/c016/c020. Brief am is missed by minimum-five-bin assembly.'),
 ('how do they do right','2-18 how; 18-23 do; 23-28 they; 28-38 do; 38-50 right. c038 supports how-do; c030/c011 support they-do; c014/c041 support the later do; c021 supports right. Two do occurrences are retained as separate intervals.'),
 ('that is okay','2-13 that; 13-18 is; 18-31 okay. c018 establishes that, c065/c079 establish is, and c047/c072/c065 establish okay.'),
 ('it is bad','2-12 it; 12-19 is; 19-29 bad. c027/c026 support initial it; c067/c008 support is; c034/c018/c036 support bad. The first token competes with is in some short windows.'),
 ('i hope i am very good family','2-13 i; 13-23 hope; 23-29 i; 29-39 am; 39-50 very; 50-59 good; 59-77 family. The whole pattern resembles c071, but 23-28 retrieves c071 initial i, and 29-39 retrieves c060 am/c062 are. The central i-am versus you-have remains uncertain. Very-good-family has strong c071/c009 and c065/c078 overlap.'),
 ('how do do please','2-11 how; 11-16 do; 16-25 do; 25-37 please. Earlier do matches c031/c014; later do matches c041/c050. The second numerical assembly resolves two do intervals, while interpolation merges them. Final please is supported by c004. The repeated do remains uncertain.'),
 ('it is hope you are comfortable','2-19 it; 19-26 is; 26-38 hope; 38-44 you; 44-49 are; 49-66 comfortable. c026/c027 support it-is, c024/c002/c028 establish hope-you, c042/c062/c074 establish are, c013/c010/c029 establish comfortable. Initial it versus i is ambiguous; no grammatical substitution was made.')
]
assert len(reconstructions)==len(meta['queries'])
vocab={w for c in meta['calibration'] for w in c['tokens']}
cal={tuple(c['tokens']) for c in meta['calibration']}
pred=[];ev=[]
local=json.load(open('local_drop2.json'));dtw=json.load(open('local_dtw_drop2.json'))
for q,(sentence,note) in zip(meta['queries'],reconstructions):
 tokens=sentence.split();assert set(tokens)<=vocab;assert tuple(tokens) not in cal
 pred.append({'id':q['id'],'tokens':tokens})
 ev.append({'id':q['id'],'tokens':tokens,'interpretation_and_approximate_bins':note,'unrestricted_cosine_evidence':local[q['id']],'unrestricted_subsequence_dtw_evidence':dtw[q['id']]})
result={'predictions':pred,'method':'Calibration-only PCA with the first two components removed, Gaussian smoothing, cosine interval retrieval, and slope-constrained subsequence DTW. Final sequences are manual interpretations of overlapping calibration evidence. Reproducible scripts, calibration_interval_ledger.json, reconstruction_evidence.json, and separate algorithm_only*_predictions.json are saved locally.','limitations':'Word boundaries and counts were inferred. Brief function words, repeats, and some recording ends remain ambiguous; timing annotations are approximate and several sequences have low-confidence tokens.'}
Path('final_reconstruction.json').write_text(json.dumps(result,indent=2))
Path('reconstruction_evidence.json').write_text(json.dumps(ev,indent=2))
print('Saved 32 interpreted predictions; all IDs, vocabulary items, and calibration-absence constraints verified.')
