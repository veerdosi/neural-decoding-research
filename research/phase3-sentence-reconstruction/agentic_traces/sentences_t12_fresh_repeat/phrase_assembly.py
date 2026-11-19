from fast_evidence import *
A=make_arrays('drop2');bs=json.load(open('interpreted_calibration_boundaries.json'))
phrases=[];V=[]
for c in meta['calibration']:
 cid=c['id']
 if cid in ['c013','c034','c069']:continue
 for i in range(len(c['tokens'])):
  for j in range(i+1,min(i+4,len(c['tokens']))+1):
   s,e=bs[cid][i],bs[cid][j]
   phrases.append(dict(id=cid,start=s,end=e,tokens=c['tokens'][i:j]));V.append(interval_vector(A[cid],s,e))
V=np.array(V,np.float32)
def global_match(a,b):
 s=np.ascontiguousarray(a@b.T);r=np.zeros(4);p=np.zeros((len(a)+len(b),2),np.int32)
 lib.subdtw(s,len(a),len(b),.035,r,p,2)
 return float(r[0])
outs={}
for q in meta['queries']:
 cid=q['id'];a=A[cid];n=len(a)
 spans=[(s,e) for s in range(2,n-4) for e in range(s+3,min(s+37,n-2)+1)]
 U=np.array([interval_vector(a,s,e) for s,e in spans],np.float32);sims=U@V.T
 lengths=np.array([p['end']-p['start'] for p in phrases])
 scores=[]
 for (s,e),sim in zip(spans,sims):
  sim=sim-.025*np.log((e-s)/lengths)**2
  ix=np.argsort(sim)[-24:];cands=[]
  for k in ix:
   p=phrases[k]
   sc=global_match(a[s:e],A[p['id']][p['start']:p['end']])-.025*np.log((e-s)/lengths[k])**2
   cands.append(dict(**p,similarity=sc))
  cands.sort(key=lambda p:-p['similarity']);scores.append(cands[:8])
 D=[[] for _ in range(n+1)];D[2]=[(0.,[],[])]
 for ((s,e),cands) in zip(spans,scores):
  if not D[s]:continue
  for r in cands:
   for total,tokens,path in D[s]:
    D[e].append((total+r['similarity']*(e-s)-.20,tokens+r['tokens'],path+[dict(query_start=s,query_end=e,**r)]))
  seen=set();best=[]
  for v,t,p in sorted(D[e],key=lambda r:-r[0]):
   key=tuple(t)
   if key in seen:continue
   seen.add(key);best.append((v,t,p))
   if len(best)>=12:break
  D[e]=best
 outs[cid]=[dict(score=v,tokens=t,matches=p) for v,t,p in D[n-2]]
 print(cid,' | '.join(' '.join(t) for v,t,p in D[n-2][:4]),flush=True)
Path('phrase_assembly_sequences.json').write_text(json.dumps(outs,indent=2))
Path('phrase_library.json').write_text(json.dumps(phrases))
