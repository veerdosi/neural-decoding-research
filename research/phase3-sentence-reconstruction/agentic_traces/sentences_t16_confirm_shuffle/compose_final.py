"""Final analyst composition from saved interval and DTW evidence.
These choices are distinct from the saved algorithm_only_* outputs.
Run after the evidence scripts to reproduce the final JSON and decision log.
"""
import json
from pathlib import Path

# id: (interpreted words, approximate query-bin spans, decision notes)
D={
 'tb0b19ec8':('you going',[(2,13),(13,25)],'You agrees across c005/c008/c034/c074; going is supported by c008/c033/c076. Okay remains a weaker final-word alternative.'),
 't59dcd856':('i hope you have family',[(2,18),(18,24),(24,29),(29,36),(36,51)],'Overlapping i-hope sources; 18:36 matches hope-you-have in c071; family agrees across c000/c047/c065/c071/c078. The short you/have split is less certain.'),
 'tcd92c113':('that is okay',[(2,12),(12,17),(17,33)],'Initial that matches c018; is matches c019/c065/c079; okay matches c047/c065/c072.'),
 't1fe83bc3':('how do you do please',[(2,10),(10,14),(14,18),(18,26),(26,38)],'How agrees across c014/c031/c038/c057. The middle overlaps do and you-do in c041/c058. Please best matches c004. The short do-you split is uncertain; clean is a competing late transition match.'),
 'td28e3558':('my it',[(2,15),(15,26)],'Initial my best matches c073, with competing hungry-like local fragments in c016/c020/c069. Final it matches c014/c015/c041; glasses is a competing suffix. This is a low-confidence composition.'),
 't6f0fdfe0':('where is my music',[(2,19),(19,26),(26,32),(32,49)],'Where/is agree with c033/c052/c079. My agrees across c013/c035/c039/c058/c079. Music agrees across c002/c035/c039.'),
 't4539ffbe':('hope they feel thirsty',[(2,9),(9,15),(15,23),(23,38)],'Early hope has cross-position matches in c024/c041/c078; subsequent they is brief. Feel matches c017/c029/c050 and thirsty matches c000/c012/c050/c066.'),
 'ta2aa0fcb':('is it',[(2,14),(14,24)],'It-is and is-it templates are close; repeated is/it overlap supports both words. Exact it-is equals calibration c026 and is excluded by the supplied novelty constraint; order is therefore interpreted as is-it.'),
 'tf75891ff':('i hope need my music are',[(2,14),(14,23),(23,30),(30,37),(37,44),(44,54)],'Need-my-music matches c035; final are agrees across c024/c053/c066/c072. No extra you was inserted: 21:31 need clearly outscored you in the targeted comparison. Unusual order preserved.'),
 't514a6795':('is it what you do',[(18,37),(37,42),(42,50),(50,58),(58,66)],'First approximately 18 bins have weak, inconsistent matches and are interpreted as leading pause. Is-it / it-is-what overlap appears in c003/c026/c027/c036; what has c004/c011/c069 interval support. Final you-do agrees with c011/c031/c063/c070/c071. What versus that and initial is/it order are uncertain.'),
 't9ce92bac':('it is bad',[(2,13),(13,19),(19,30)],'Initial it-is best matches c027; bad agrees across c018/c034/c036.'),
 't3afd73b4':('yes i am very hungry',[(2,15),(15,19),(19,22),(22,33),(33,45)],'Initial yes-i-am best matches c060 and has c020 overlap; very agrees across c009/c060/c066/c071; hungry agrees across c016/c020/c069.'),
 'tef9d94a6':('you do like music',[(2,16),(16,24),(24,35),(35,47)],'You-do has a stronger full-prefix match to c056 than do-you to c063, although the ordering is ambiguous. Like and music agree across c002/c028/c035/c039.'),
 't52b24bf6':('do they have my glasses',[(2,18),(18,23),(23,26),(26,32),(32,49)],'Do-they overlaps c011/c030; have is supported by c027/c030/c058; my by c028/c058/c078; glasses by c061/c073. The brief have span was merged by some algorithm-only paths.'),
 't5f2d20a0':('you hope my family is good',[(2,15),(15,21),(21,25),(25,34),(34,39),(39,51)],'You-hope matches c008; short my agrees across c008/c039/c058; family agrees across c000/c019/c043/c065/c078; is and good are independently supported.'),
 't44bb2095':('my nurse is very tired',[(2,14),(14,23),(23,27),(27,36),(36,49)],'My-nurse-is overlaps c005/c008/c013/c068; very matches c009/c060/c066/c071; tired has strong c042 matches. Here is a weaker tail alternative in transferred votes.'),
 'te366b91c':('i tell no family',[(2,11),(11,18),(18,27),(27,46)],'Targeted 10:18 tell matched c012 more strongly than hope; 18:27 no matched c074, with they/hope as alternatives. Family agrees across many recordings. The broader vote path incorrectly merged tell/no into hope; unusual sequence retained.'),
 'ta8a65775':('i hope i am very good family',[(2,13),(13,23),(23,30),(30,37),(37,51),(51,60),(60,80)],'I-hope and final very-good-family strongly overlap c071 and other sources. Middle i-am has two independent phrase matches in c048/c060, stronger than you-have, they-are or you-are. The i/am split is approximate.'),
 't8485b6b9':('how tired are they',[(2,10),(10,21),(21,30),(30,38)],'How matches c014/c038/c057; tired has overlapping c042 matches around 14:25; are and they match c052/c053/c066. A simple how-are-they path merges an internal span and is also an exact calibration sentence, excluded by novelty.'),
 't41450074':('they like it',[(2,12),(12,20),(20,30)],'They agrees across c024/c064; like across c002/c028/c031/c039/c045; final it across c014/c017/c041. Up remains a weaker final-word alternative.'),
 't53cbb8eb':('i nurse are you',[(2,10),(10,19),(19,27),(27,41)],'Initial i is weak but supported by c012/c032/c039. Nurse agrees across c005/c029/c068; are across c052/c053/c072/c074; you best matches c012/c008/c044. No grammatical repair applied.'),
 'tbfeb0ff2':('i hope they do not need music',[(2,8),(8,16),(16,20),(20,27),(27,32),(32,39),(39,47)],'Prefix i-hope-they-do-not overlaps c035 and do-not overlaps c006/c050/c058. Need has c056/c075/c077 interval support. Music has c002/c035/c039 support. No my is added without a separate interval.'),
 'te3243945':('i am good here',[(2,11),(11,21),(21,27),(27,35)],'Targeted prefix i-am from c001 outscored i-hope, i-need and i-tell. Good matches c003/c019/c023/c032/c074; here agrees across c005/c024/c044/c045/c048.'),
 'tca769afd':('it is hope you are comfortable',[(2,20),(20,27),(27,37),(37,43),(43,49),(49,66)],'Initial it/is order remains uncertain. Is has internal c008/c067/c079 support. Hope-you-are overlaps c024/c044/c062; comfortable agrees strongly across c010/c013/c029. Weak trailing bins are treated as offset.'),
 'td133fe68':('my family is not very tired',[(2,16),(16,27),(27,32),(32,37),(37,47),(47,61)],'My-family agrees across c000/c019/c043/c047/c065/c078. Targeted 26:38 is-not best matches c043. Very agrees across c009/c060/c066/c071. Tired has strong c042 matches; here is a competing tail alternative.'),
 't588781f6':('my not here',[(2,19),(19,25),(25,35)],'Initial my best matches c073. Not agrees with c050/c058; like-it and do-not are alternatives for the middle interval. Here is very strongly supported by c024/c044/c048/c077/c078. Right was not supported by fixed-interval matching.'),
 'tced8ecfc':('i like my comfortable have',[(2,12),(12,18),(18,24),(24,40),(40,52)],'Initial i and like have strong local matches. 12:23 like-my best matches c028 with c039 support. Comfortable agrees across c010/c013/c029. Final have best matches c030 at 41:52, with help/am alternatives. The my and final word are low confidence; unusual ordering preserved.'),
 't848cf4da':('i',[(2,23)],'A single stretched i best matches c077/c016/c032. Internal my-like fragments and i-am are alternatives, but the extra word is not resolved. No count was supplied; one word is the conservative best match.'),
 'td6e48564':('where is that',[(2,15),(15,21),(21,31)],'Where/is overlap c033/c052/c079. Final that strongly matches c051 and c037.'),
 't0575fbc6':('my hope is my success',[(2,11),(11,20),(20,25),(25,32),(32,51)],'Initial my best matches c073. Hope has c002/c028/c046 support. Is-my-success matches c067, with internal my agreement from other recordings. Success occurs in only one calibration.'),
 't6453e6ae':('i do not have',[(2,10),(10,16),(16,24),(24,36)],'I-do-not-have overlaps c058 and do-not across c006/c035/c050/c056. Have versus is/up at the end is uncertain; multiple transferred paths support have.'),
 't4d2d7415':('how do they do right',[(2,20),(20,24),(24,30),(30,38),(38,52)],'How best matches c053/c038; a brief do appears before they. They-do overlaps c011/c014/c030/c041. Right best matches c021 with c024/c048 support. First how versus what is less certain.'),
}

def main():
 m=json.loads(Path('inputs.json').read_text());vocab={w for c in m['calibration'] for w in c['tokens']};cal={tuple(c['tokens']) for c in m['calibration']};pred=[];notes=[]
 for q in m['queries']:
  qid=q['id'];phrase,spans,note=D[qid];tokens=phrase.split();assert len(tokens)==len(spans);assert all(w in vocab and w==w.lower() for w in tokens);assert tuple(tokens) not in cal;assert all(0<=s<e<=q['shape'][0] for s,e in spans)
  pred.append(dict(id=qid,tokens=tokens));notes.append(dict(id=qid,tokens=tokens,intervals=[dict(token=w,start=s,end=e) for w,(s,e) in zip(tokens,spans)],interpretation=note))
 result=dict(predictions=pred,method='Calibration-only PCA with the first two components removed, Gaussian smoothing, cosine interval retrieval and slope-constrained subsequence DTW. Calibration interval templates and overlapping matches informed manual composition. Reproducible scripts, decision notes and scores are saved locally; algorithm-only sequences are saved separately in algorithm_only_*_predictions.json.',limitations=['Word boundaries and counts were inferred. Short function words, leading pauses and several initial/final words remain ambiguous.','Final sequences are interpreted reconstructions, not uniquely determined tool outputs; unusual word orders were preserved when supported.','Particularly uncertain queries: td28e3558, t514a6795, t53cbb8eb, t588781f6, tced8ecfc and t848cf4da.'])
 Path('final_predictions.json').write_text(json.dumps(result,indent=2)+'\n');Path('interpreted_evidence.json').write_text(json.dumps(notes,indent=2)+'\n');print('Saved 32 predictions, interpretation notes, and validation checks passed.')
if __name__=='__main__':main()
