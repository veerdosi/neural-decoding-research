from local_retrieval import *
import sys
A=make_arrays('drop2');cal=meta['calibration'];cand=[];V=[]
for c in cal:
 for length in [4,6,8,10,13,16]:
  for s in range(2,len(A[c['id']])-length-1):
   cand.append((c['id'],s,s+length));V.append(interval_vector(A[c['id']],s,s+length))
V=np.asarray(V,dtype=np.float32)
ids=sys.argv[1:] or [c['id'] for c in cal]
allout={}
with open('calibration_overlap.txt','w') as f:
 for cid in ids:
  c=next(c for c in cal if c['id']==cid);out=[]
  f.write('\n%s %d %s\n'%(cid,len(A[cid]),' '.join(c['tokens'])))
  spans=[(s,s+6) for s in range(2,len(A[cid])-7,3)]
  sims=np.asarray([interval_vector(A[cid],*span) for span in spans],np.float32)@V.T
  for (s,e),sim in zip(spans,sims):
   rows=[];seen={cid}
   for ix in np.argsort(sim)[::-1]:
    cc,ss,ee=cand[ix]
    if cc in seen:continue
    seen.add(cc);rows.append({'id':cc,'start':ss,'end':ee,'similarity':float(sim[ix])})
    if len(rows)==8:break
   out.append({'start':s,'end':e,'matches':rows})
   f.write('%2d:%2d '%(s,e)+' | '.join('%.2f %s %d:%d %s'%(r['similarity'],r['id'],r['start'],r['end'],' '.join(cal[int(r['id'][1:])]['tokens'])) for r in rows[:4])+'\n')
  allout[cid]=out
Path('calibration_overlap.json').write_text(json.dumps(allout))
