from local_retrieval import *
A=make_arrays('drop2');out={}
for q in meta['queries']:
 qid=q['id'];matches=[]
 for c in meta['calibration']:
  score,path,C=dtw(A[qid][2:-2],A[c['id']][2:-2]);matches.append({'id':c['id'],'tokens':c['tokens'],'similarity':1-float(score),'path':[(i+2,j+2) for i,j in path]})
 matches.sort(key=lambda x:-x['similarity']);out[qid]=matches
 print('\n'+qid+'\n'+'\n'.join(f"{x['similarity']:.3f} {x['id']} "+' '.join(x['tokens']) for x in matches[:6]))
Path('global_drop2_evidence.json').write_text(json.dumps(out))
