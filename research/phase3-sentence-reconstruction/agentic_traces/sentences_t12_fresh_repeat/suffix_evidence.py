from fast_evidence import *
A=make_arrays('drop2');out={};cs={c['id']:c for c in meta['calibration']}
with open('suffix_evidence.txt','w') as f:
 for q in meta['queries']:
  qid=q['id'];print('\nQUERY',qid,file=f);rs=[]
  for n in [5,8,11,15]:
   if len(A[qid])<n+4:continue
   ms=[]
   for c in meta['calibration']:
    r=match(A[qid][-2-n:-2][::-1],A[c['id']][2:-2][::-1],True);r['id']=c['id'];r['last']=c['tokens'][-1];r['start_original']=len(A[c['id']])-2-r['end'];ms.append(r)
   ms.sort(key=lambda r:-r['similarity']);rs.append(dict(length=n,matches=ms))
   print(n,' | '.join(f"{v['similarity']:.2f} {v['id']}:{v['start_original']}-{len(A[v['id']])-2} "+' '.join(cs[v['id']]['tokens']) for v in ms[:7]),file=f)
  out[qid]=rs
Path('suffix_matches.json').write_text(json.dumps(out));print(open('suffix_evidence.txt').read())
