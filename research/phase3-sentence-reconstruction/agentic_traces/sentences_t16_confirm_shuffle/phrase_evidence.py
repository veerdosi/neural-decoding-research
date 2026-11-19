from template_reconstruction import *
from fast_evidence import match
A=make_arrays('drop2');T=[];V=[]
for c in meta['calibration']:
 cid=c['id']
 if cid not in B:continue
 for i in range(len(c['tokens'])):
  for j in range(i+1,min(i+4,len(c['tokens']))+1):
   s,e=B[cid][i],B[cid][j]
   T.append(dict(id=cid,start=s,end=e,tokens=c['tokens'][i:j]));V.append(interval_vector(A[cid],s,e,12))
V=np.array(V,np.float32)
spans={
 'tb0b19ec8':[(2,25)],'t59dcd856':[(18,36),(23,36)],'tcd92c113':[(2,17),(2,33)],
 't1fe83bc3':[(2,17),(10,26),(14,27),(23,38)],'td28e3558':[(2,16),(2,26)],
 'ta2aa0fcb':[(2,24)],'tf75891ff':[(19,31),(22,44),(37,53)],
 't514a6795':[(2,20),(18,38),(18,50),(35,55),(44,66),(50,66)],
 't9ce92bac':[(2,19)],'t3afd73b4':[(2,22),(10,22)],'tef9d94a6':[(2,24),(10,25)],
 't52b24bf6':[(2,19),(10,26)],'t5f2d20a0':[(18,27),(30,40)],
 'te366b91c':[(2,19),(10,28),(17,28)],'ta8a65775':[(20,37),(27,38)],
 't8485b6b9':[(8,30)],'t53cbb8eb':[(2,20),(26,40)],
 'tbfeb0ff2':[(12,22),(29,46)],'te3243945':[(10,23),(17,35)],
 'tca769afd':[(2,26)],'td133fe68':[(26,38)],
 't588781f6':[(2,17),(13,25)],'tced8ecfc':[(9,24),(37,52)],
 't848cf4da':[(2,23)],'t0575fbc6':[(2,12),(30,51)],'t6453e6ae':[(22,36)],
 't4d2d7415':[(2,21),(15,31),(34,52)]}
out={}
with open('phrase_readable.txt','w') as f:
 for qid,qs in spans.items():
  out[qid]=[];f.write('\n'+qid+'\n')
  for s,e in qs:
   vec=interval_vector(A[qid],s,e,12);scores=V@vec;ix=np.argsort(scores)[::-1][:100];rows=[]
   for k in ix:
    t=T[k];ratio=(e-s)/(t['end']-t['start'])
    if ratio<.45 or ratio>2.2:continue
    r=match(A[qid][s:e],A[t['id']][t['start']:t['end']],2)
    rows.append(dict(t,similarity=r['similarity'],path=r['path']))
   rows.sort(key=lambda r:-r['similarity']);seen=set();best=[]
   for r in rows:
    phrase=' '.join(r['tokens'])
    if phrase in seen:continue
    seen.add(phrase);best.append(r)
    if len(best)==8:break
   out[qid].append(dict(start=s,end=e,matches=best));f.write(f'{s}-{e} '+' | '.join(f"{r['similarity']:.2f} {r['id']}:{r['start']}-{r['end']} "+' '.join(r['tokens']) for r in best[:6])+'\n')
Path('phrase_evidence.json').write_text(json.dumps(out))
