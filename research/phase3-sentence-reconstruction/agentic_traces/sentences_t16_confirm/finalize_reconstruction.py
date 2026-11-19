"""Save the human-interpreted reconstruction and its evidence, separately
from every algorithm-only hypothesis. Re-running does not perform inference.
"""
import json
from pathlib import Path
m=json.load(open('inputs.json'))
sequences=[
'is it',
'do you like music',
'where is that',
'i am good',
'you going',
'i hope they do not need music',
'i like it here',
 'they like it',
 'my family is not very tired',
 'my hope is my success',
 'hungry',
 'how thirsty are you',
 'do they have my glasses',
 'you hope my family is good',
 'my nurse is very tired',
 'where is my music',
 'i hope you have family',
 'i hope they need my music are',
 'they feel thirsty',
 'i am',
 'i do not have',
 'i hope family',
 'how tired are they',
 'is it you',
 'i like comfortable family',
 'yes i am very hungry',
 'how do they do right',
 'that is okay',
 'it is bad',
 'i hope i have very good family',
 'how do please',
 'it is hope you are comfortable',
]
notes=[
'Initial it/is is ambiguous. c003 and c036 supply is-it prefix evidence; final interval also retrieves c014/c041 terminal it. Interpreted is it rather than numerical it it. Low confidence.',
'First word retrieves c004 do and second c022/c038 you; c002/c028/c039 jointly support like; c002/c035/c039 support music. Refined numerical assembly also selects do you like music.',
'where agrees across c033/c052/c079; middle is agrees c033/c079; ending that agrees c051/c037.',
'Initial i agrees c001/c051/c078; middle am overlaps c001/c016/c012, but need/tell/have remain competing readings; final good agrees c019/c023/c074. Low confidence.',
'Initial you agrees c005/c008/c034/c074. Ending most consistently retrieves c008 going, with weaker c068 agreement. you alone and an unmodeled ending remain possible. Low confidence.',
'i hope is shared across c002/c039/c046/c062. Internal they-do-not is supported by overlapping c035/c050/c006/c058 intervals. Need is tentative from c056, before c035/c039 music. Short function-word timing is uncertain.',
'Initial i competes with my. The overlapping later trajectory follows like-it-here in c045; terminal here is independently strong in c024/c044/c077. Direct assembly merges the initial two words into my and confuses it with not. Low to moderate confidence.',
'Initial they agrees c024/c027/c064, like agrees c002/c031/c039/c045, final it agrees c014/c041.',
'my-family-is-not matches c019/c043/c047/c078 overlapping intervals; very matches c009/c066/c071; tired matches c042.',
'my-hope overlaps c002/c028/c046 and c067; is-my-success is a continuous match in c067. Discard weak numerical trailing they.',
'The lexical core retrieves hungry in c016/c020/c069, especially c016 rows30-42. Extra trailing it/hope in the numerical assemblies is weak and could be shared ending activity. Word count is uncertain.',
'Initial how weakly agrees c014/c038. Middle nurse/thirsty is ambiguous; thirsty is supported by c000/c066 and overlaps the r/s portion of nurse in c005/c029/c068. are is strong across c052/c074; terminal you agrees c012/c008. Low confidence on thirsty.',
'The central sequence follows do-they-have in c030, with c058 have and c055 do; my-glasses agrees c061/c073. Numerical prefix that is weak and treated as start activity.',
'you-hope agrees c008; short my is tentative from c058/c000; family agrees c000/c019/c043/c047/c078; is and good have independent overlapping support. My has only a short interval.',
'my-nurse agrees c005/c008/c029/c068; is agrees c008/c068 though it wins the crude template score; very agrees c009/c066/c071; tired agrees c042.',
'where-is-my agrees c079 continuously, with c052/c033 for where and c035/c039/c058 for my; music agrees c002/c035/c039.',
'i-hope agrees c002/c035/c039/c078; you-have is supported by c071 rows25-39, and family by c000/c047/c065/c071/c078. Have is shorter than the automatic assembly permits comfortably.',
'i-hope is stable. they-need overlaps c064 rows13-24/c075/c015. my/music interval31-43 strongly retrieves c035 rows57-71 and c039 rows43-56. The final43-53 interval strongly retrieves are in c066/c072/c053. Kept the awkward ending because computer, here, and other grammatical substitutes lack comparable neural support. Lowest-confidence complete reconstruction.',
'Initial they agrees c015/c064/c075; feel agrees c017/c029/c050; thirsty agrees c012/c050/c066.',
'Whole query2-22 retrieves the initial i-am interval in c0162-23, and initial i portions of c032/c077; short-word scores separately favor it/my. Interpreted i am using the overlapping whole interval. Low confidence.',
'i-do-not-have follows overlapping consecutive c058 intervals9-39 with independent do/not support in c035/c050; ending have is weaker than the phrase evidence and competes with up/is.',
'Initial i is strong; middle hope agrees c002/c028/c046 and overlaps okay onset in c047/c065; family agrees c071/c078/c019/c047/c065. tell/have/no are competing middle-word interpretations. Low confidence.',
'how agrees c038/c057; tired agrees c042; are agrees c052/c066/c074; terminal they has c052/c053 overlap, while you is competitive in direct templates.',
'Long leading interval has weak lexical matches and is left unassigned. Main is-it interval19-55 agrees c003/c036 with c013 support for is. Final you agrees c012/c063/c071, but do in c070/c011 is competitive. Low confidence.',
'i-like agrees c028/c039, comfortable is strong in c010/c013/c029. Final family is weak but supported by c07156-67/c00017-35 and the variance-balanced assembly; am/have/help remain alternatives.',
'yes agrees c020/c060; i/am are short and overlap c001/c048/c060; very agrees c009/c066/c071; hungry agrees c016/c020/c069. Am is weak and shorter than the numerical minimum duration.',
'how-do prefix agrees c031/c038; they-do agrees c014/c030/c041; right agrees c021/c024/c048. A weak initial do can alternatively be omitted by the assembly.',
'That agrees c018, is agrees c079/c065/c019, and okay agrees c047/c065/c072.',
'Initial it agrees c026/c027; middle is agrees c008/c067/c018; bad agrees c018/c034/c036. Is/it confusion is present in retrieval but ordered overlap favors this reading.',
'i-hope is stable across c002/c035/c039/c062/c078. Internal i/have are ambiguous with my/am. c055 supplies short i evidence and c071 supplies have; very-good-family is a long continuous match to c071, with independent c009/c060/c019/c065 support.',
'Initial how agrees c014/c038. Middle do agrees c014/c031/c041; direct assembly repeats do over overlapping portions of this interval, treated as one stretched word. Final please agrees c004, with some clean confusion. Low confidence.',
'Initial it-is agrees c026/c027 with is-it competition in c003/c036. Hope-you-are agrees overlapping c024/c002/c044/c062 intervals; comfortable strongly agrees c010/c013/c029. Retained it-is prefix despite awkward composition; i-hope is an alternative with poorer interval support.',
]
assert len(sequences)==len(m['queries'])==len(notes)
vocab={w for c in m['calibration'] for w in c['tokens']};cal={tuple(c['tokens']) for c in m['calibration']}
local=json.load(open('local_drop2.json'));dtw=json.load(open('fast_local_dtw_drop2.json'))
alg=json.load(open('dtw_algorithm_only_predictions.json'));ref=json.load(open('refined_algorithm_only_predictions.json'));bal=json.load(open('balanced_algorithm_only_predictions.json'))
predictions=[];evidence=[]
for q,seq,note in zip(m['queries'],sequences,notes):
 tokens=seq.split();assert set(tokens)<=vocab;assert tuple(tokens) not in cal
 predictions.append({'id':q['id'],'tokens':tokens})
 evidence.append({'id':q['id'],'final_interpreted_tokens':tokens,'interpretation':note,'direct_interval_evidence':local[q['id']],'slope_constrained_dtw_evidence':dtw[q['id']],'uninterpreted_numerical_hypotheses':{'initial':[x for x in alg[q['id']]['tokens']],'refined':[x for x in ref[q['id']]['tokens']],'variance_balanced':[x for x in bal[q['id']]['tokens']]}})
method='Calibration-only PCA and smoothing; removal of the first two components; cosine interval retrieval and slope-constrained DTW. Final tokens are manual interpretations of overlapping calibration evidence. Reproducible scripts, scores, algorithm-only sequences, and final_reconstruction_evidence.json are saved locally.'
limitations=['No query labels or boundaries were used. Word counts and timing are inferred; short function words remain ambiguous.','Several reconstructions have low confidence, especially ta8a65775. Rough calibration word intervals and shared onset/ending activity can merge, split, or confuse words.']
Path('final_reconstruction_evidence.json').write_text(json.dumps({'method':method,'limitations':limitations,'queries':evidence},indent=2))
Path('final_predictions.json').write_text(json.dumps({'predictions':predictions,'method':method,'limitations':limitations},indent=2))
print('Saved 32 complete, vocabulary-valid, novel reconstructions and per-query evidence.')
