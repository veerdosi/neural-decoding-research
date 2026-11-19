from local_retrieval import *
import sys
name=sys.argv[1];origins=sys.argv[2:];A=make_arrays(name)
cand=[];V=[]
for c in meta['calibration']:
 a=A[c['id']]
 for length in range(4,17):
  for start in range(2,len(a)-length-1):
   cand.append((c['id'],start,start+length));V.append(interval_vector(a,start,start+length))
V=np.asarray(V,dtype=np.float32)
outs={}
for origin in origins:
 a=A[origin];spans=[(s,s+6) for s in range(2,len(a)-7,3)];qs=np.asarray([interval_vector(a,*s) for s in spans],dtype=np.float32);sims=qs@V.T;out=[]
 print('\n',origin)
 for span,sim in zip(spans,sims):
  ix=np.argsort(sim)[::-1];seen={origin};rows=[]
  for j in ix:
   cid,s,e=cand[j]
   if cid in seen:continue
   seen.add(cid);rows.append({'id':cid,'start':s,'end':e,'similarity':float(sim[j])})
   if len(rows)==6:break
  out.append({'start':span[0],'end':span[1],'matches':rows})
  print(span,' | '.join(f"{r['similarity']:.2f} {r['id']}:{r['start']}-{r['end']} "+' '.join(meta['calibration'][int(r['id'][1:])]['tokens']) for r in rows[:4]))
 outs[origin]=out
Path('inspect_'+name+'_'+'_'.join(origins)+'.json').write_text(json.dumps(outs))
