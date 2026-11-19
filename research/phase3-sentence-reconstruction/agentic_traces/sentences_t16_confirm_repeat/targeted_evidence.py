from fast_helpers import *
A=make_arrays('drop2');labs={c['id']:' '.join(c['tokens']) for c in meta['calibration']};out={}
cases=[('td133fe68',37,51),('tbfeb0ff2',15,28),('t8485b6b9',2,16),('te3243945',10,24),('te3243945',20,37),('t0575fbc6',22,40),('t1fe83bc3',10,22),('t1fe83bc3',20,34),('tb0b19ec8',10,24),('t9ce92bac',2,17),('t9ce92bac',16,26),('ta8a65775',30,52),('t6453e6ae',10,27),('tca769afd',18,40),('tca769afd',40,65)]
for q,s,e in cases:
 rows=[]
 for c in meta['calibration']:
  r=subsequence(A[q][s:e],A[c['id']][2:-2]);r['id']=c['id'];r['start']+=2;r['end']+=2;rows.append(r)
 rows.sort(key=lambda r:-r['similarity']);out[f'{q}:{s}-{e}']=rows[:12]
 print('\n',q,s,e)
 for r in rows[:6]:print(f"{r['similarity']:.3f} {r['id']}:{r['start']}-{r['end']} {labs[r['id']]}")
Path('targeted_evidence.json').write_text(json.dumps(out))
