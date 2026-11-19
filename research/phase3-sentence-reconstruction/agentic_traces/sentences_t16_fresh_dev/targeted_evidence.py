"""Fixed-interval cosine/DTW comparisons; no parameter fitting or sequence decoder.

The word associations below are human interpretations of calibration-to-calibration
overlap reports. Boundaries are approximate, not supplied ground truth. Scores
compare complete intervals, allowing the same 1:1, 1:2, 2:1 steps as local_dtw.
"""
from local_retrieval import *

EXAMPLES = {
    'do': [('c055',2,14),('c058',14,24),('c050',46,55),('c037',72,82),('c031',14,23)],
    'you': [('c008',3,14),('c039',28,36),('c070',5,15),('c007',11,21),('c037',82,92)],
    'like': [('c039',35,44),('c028',32,42),('c031',26,38),('c037',92,102),('c045',12,20)],
    'have': [('c061',33,41),('c058',29,40),('c071',29,41),('c030',37,48)],
    'it': [('c033',25,34),('c045',19,24),('c017',24,34),('c041',32,41)],
    'right': [('c077',40,50),('c046',33,44),('c048',40,50),('c078',61,71)],
    'are': [('c072',25,34),('c034',14,22),('c066',4,15),('c062',24,33)],
    'my': [('c061',40,46),('c058',40,48),('c039',43,51),('c035',58,65)],
    'need': [('c051',12,22),('c064',17,27),('c056',38,49),('c035',49,58)],
    'feel': [('c017',15,24),('c010',18,26),('c029',58,67)],
    'here': [('c045',23,36),('c063',56,71),('c077',50,64),('c046',44,54)],
    'that': [('c037',102,112),('c051',22,33),('c036',30,39),('c034',21,29)],
    'not': [('c058',23,30),('c035',41,49),('c056',32,40),('c068',40,49),('c047',32,41),('c050',54,61)],
    'no': [('c076',2,14),('c043',2,12)],
    'i': [('c055',14,24),('c048',25,34),('c020',20,28),('c001',2,12)],
    'am': [('c048',34,40),('c020',28,38),('c001',12,19),('c060',18,25)],
    'hungry': [('c016',30,46),('c020',67,81),('c069',39,54)],
}

def complete_score(a,b):
    S=a@b.T; n,m=S.shape
    D=np.full((n+1,m+1),-1e9);D[0,0]=0
    for i in range(1,n+1):
        for j in range(1,m+1):
            vals=[D[i-1,j-1]+S[i-1,j-1]]
            if i>=2: vals.append(D[i-2,j-1]+S[i-2,j-1]+S[i-1,j-1]-.035)
            if j>=2: vals.append(D[i-1,j-2]+.5*(S[i-1,j-2]+S[i-1,j-1])-.035)
            D[i,j]=max(vals)
    return float(D[n,m]/n)

if __name__=='__main__':
    A=make_arrays('drop2');out={}
    spans={'te7fda903':[(2,14),(10,19),(14,24),(17,26),(23,35)],
           'ta84f22ff':[(2,12),(9,19),(11,21),(14,22),(19,33),(10,16),(16,22)],
           'te1c6177b':[(2,11),(10,18),(15,23),(21,30),(24,31),(29,43)],
           't196443a7':[(2,12),(10,20)]}
    for qid,intervals in spans.items():
        out[qid]=[]
        for s,e in intervals:
            ranks=[]
            for word,ex in EXAMPLES.items():
                scores=[{'id':cid,'start':l,'end':r,'score':complete_score(A[qid][s:e],A[cid][l:r])} for cid,l,r in ex]
                scores.sort(key=lambda x:-x['score'])
                ranks.append({'word':word,'best':scores[0]['score'],'top2_mean':float(np.mean([v['score'] for v in scores[:2]])),'examples':scores})
            ranks.sort(key=lambda x:-x['top2_mean'])
            out[qid].append({'start':s,'end':e,'scores':ranks})
            print(qid,s,e,' | '.join(f"{r['word']} {r['top2_mean']:.3f}" for r in ranks[:6]))
    Path('targeted_evidence.json').write_text(json.dumps({'boundary_origin':'human interpretation of calibration overlap; approximate','queries':out},indent=2))
