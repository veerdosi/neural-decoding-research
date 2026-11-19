from local_retrieval import *
import ctypes
lib=ctypes.CDLL(str(Path('fast_dtw.dylib').resolve()));fn=lib.match
ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
fn.argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,ip,ip];fn.restype=ctypes.c_float

def match(a,b,anchored=False):
 s=np.ascontiguousarray(a@b.T,dtype=np.float32);n,m=s.shape;start=np.zeros(1,np.int32);end=start.copy();path=np.zeros(n,np.int32)
 sc=fn(s,n,m,int(anchored),start,end,path)
 return dict(similarity=float(sc),start=int(start[0]),end=int(end[0]),path=path.tolist())

if __name__=='__main__':
 A=make_arrays('drop2');out={};cal={c['id']:c for c in meta['calibration']}
 for q in meta['queries']:
  qid=q['id'];a=A[qid];out[qid]=[]
  spans=[(2,len(a)-2,False)]+[(2,min(2+n,len(a)-2),True) for n in [8,12,16]]+[(s,s+n,False) for n in [8,12,16] for s in range(2,len(a)-n-1,3)]
  for st,en,anchor in spans:
   rows=[]
   for c in meta['calibration']:
    r=match(a[st:en],A[c['id']][2:-2],anchor);r['id']=c['id'];r['start']+=2;r['end']+=2;r['path']=[v+2 for v in r['path']];rows.append(r)
   rows.sort(key=lambda r:-r['similarity']);out[qid].append(dict(start=st,end=en,anchor=anchor,matches=rows[:10]))
 Path('dtw_evidence.json').write_text(json.dumps(out))
 with open('dtw_edges_readable.txt','w') as f:
  for qid,rows in out.items():
   f.write('\n'+qid+'\n')
   for row in rows[:4]:
    f.write(f"{row['start']}-{row['end']} "+' | '.join(f"{r['similarity']:.2f} {r['id']}:{r['start']}-{r['end']} "+' '.join(cal[r['id']]['tokens']) for r in row['matches'][:5])+'\n')
 print('Saved DTW interval and edge evidence')
