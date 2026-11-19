"""Manually interpreted calibration timing estimates, based on shared-phrase
retrieval in calibration_interval_evidence.txt and the labeled transcripts.
These are uncertain inferred boundaries, not supplied word timings.
Original automatic segmentation and sequences remain saved separately.
"""
from interval_assembly import *
overrides={
'c001':[2,6,13,18,27,38], 'c002':[2,6,11,21], 'c003':[2,7,21], 'c004':[2,7,13,20,31],
'c005':[2,6,13,19,23,35], 'c006':[2,7,12,22], 'c008':[2,7,13,22,30],
'c011':[2,6,9,19,26,32,40,58], 'c014':[2,5,15,23,31],
'c016':[2,5,12,17,22,33], 'c017':[2,5,8,15,23,29,44],
'c018':[2,6,12,17,23,35], 'c019':[2,7,12,22,33],
'c020':[2,7,12,22,32], 'c022':[2,6,11,16,27],
'c025':[2,9,15,27,39], 'c027':[2,6,9,18,26,31,45],
'c031':[2,5,12,17,25,33], 'c032':[2,5,13,18,23,34,44],
'c035':[2,7,14,19,29,44], 'c036':[2,8,14,28],
'c039':[2,5,8,19,28,42], 'c041':[2,5,12,24], 'c042':[2,8,13,20,30,39],
'c046':[2,6,11,16,20,24,33], 'c049':[2,5,15,20,30,42],
'c050':[2,6,12,20,27], 'c054':[2,6,13,19,42], 'c056':[2,6,13,17,23,33],
'c058':[2,5,12,17,26,33,41], 'c059':[2,6,13,17,23,28,37],
'c061':[2,7,11,16,21,26,39], 'c062':[2,10,20], 'c063':[2,5,11,15,23,32,40],
'c066':[2,7,11,17,25], 'c071':[2,7,14,19,28,39], 'c072':[2,5,8,17,35],
'c075':[2,5,10,16,26], 'c076':[2,5,13,19,27,35],
'c078':[2,5,8,18,23,30,45], 'c079':[2,6,12,18,27,33,43,56],
'c080':[2,6,13,19,28,33,46], 'c083':[2,5,11,17,23,29,35,42],
'c088':[2,7,12,24,35], 'c090':[2,5,13,20,27,36],
'c091':[2,5,8,23,34], 'c092':[2,5,14,19,25,35,45],
'c097':[2,6,13,20,27,38], 'c099':[2,6,13,17,23,34],
'c101':[2,5,8,18,23,31,39,49,63], 'c105':[2,9,19],
'c108':[2,6,13,18,28,33,41], 'c109':[2,7,10,26,43],
'c110':[2,6,12,17,28], 'c111':[2,5,11,17,23,29,40],
'c112':[2,10,17,25], 'c114':[2,6,9,20,27,33,40,53],
'c115':[2,5,15,19,25,33,48], 'c117':[2,6,13,20,25,30,47],
}
bs={k:np.array(v) for k,v in json.load(open('calibration_boundaries_3.json')).items()};bs.update({k:np.array(v) for k,v in overrides.items()})
for c in cal:assert len(bs[c['id']])==len(c['tokens'])+1
Path('interpreted_calibration_boundaries.json').write_text(json.dumps({k:v.tolist() for k,v in bs.items()},indent=2))
T,L,S=get_templates(bs)
keep=np.array([v[0] not in ['c013','c034','c069'] for v in S]);T=T[keep];L=L[keep];S=[v for v,k in zip(S,keep) if k]
out={}
for q in meta['queries']:
 cid=q['id'];sc=score_intervals(cid,T,L,S);np.savez_compressed('refined_scores_'+cid+'.npz',spans=candidates[cid],scores=sc,words=words)
 paths=decode(cid,sc);out[cid]=[dict(score=v,tokens=[p[0] for p in ps],intervals=ps) for v,ps in paths]
 print(cid,' | '.join(' '.join(p[0] for p in ps) for v,ps in paths[:3]))
Path('exemplar_assembly_sequences.json').write_text(json.dumps(out,indent=2))
Path('interpreted_exemplar_sources.json').write_text(json.dumps(S,indent=2))
