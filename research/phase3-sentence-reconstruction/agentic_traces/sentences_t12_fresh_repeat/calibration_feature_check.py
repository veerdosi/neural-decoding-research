"""One calibration-only diagnostic of spectral truncation, using known first/last
word identities and fixed eight-bin edge matching. No query scores select features.
"""
from fast_evidence import *
results={}
for tag in ['drop2','pca64drop2']:
 A=make_arrays(tag);rows=[]
 for c in meta['calibration']:
  cid=c['id'];out={'id':cid}
  for end in [False,True]:
   a=A[cid][2:10] if not end else A[cid][-10:-2][::-1]
   ranked=[]
   for d in meta['calibration']:
    if d['id']==cid:continue
    b=A[d['id']][2:-2] if not end else A[d['id']][2:-2][::-1]
    r=match(a,b,True);ranked.append((r['similarity'],d['id'],d['tokens'][-1 if end else 0]))
   ranked.sort(reverse=True);out['suffix' if end else 'prefix']={'true':c['tokens'][-1 if end else 0],'matches':ranked[:5]}
  rows.append(out)
 def acc(key):
  elig=[r[key] for r in rows if sum(d['tokens'][-1 if key=='suffix' else 0]==r[key]['true'] for d in meta['calibration'])>1]
  return sum(r['matches'][0][2]==r['true'] for r in elig)/len(elig)
 results[tag]={'prefix_accuracy':acc('prefix'),'suffix_accuracy':acc('suffix'),'rows':rows}
 print(tag,'prefix',round(acc('prefix'),3),'suffix',round(acc('suffix'),3))
Path('calibration_feature_check.json').write_text(json.dumps(results))
