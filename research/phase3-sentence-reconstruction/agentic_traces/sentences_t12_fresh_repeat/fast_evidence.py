from local_retrieval import *
import ctypes
lib=ctypes.CDLL('./fast_dtw.so')
lib.subdtw.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_double,np.ctypeslib.ndpointer(dtype=np.float64),np.ctypeslib.ndpointer(dtype=np.int32),ctypes.c_int]
def match(a,b,anchored=False):
 s=np.ascontiguousarray(a@b.T);r=np.zeros(4);p=np.zeros((len(a)+len(b),2),np.int32)
 lib.subdtw(s,len(a),len(b),.035,r,p,int(anchored))
 return dict(similarity=float(r[0]),start=int(r[1]),end=int(r[2]),path=p[:int(r[3])][::-1].tolist())
if __name__=='__main__':
 A=make_arrays('drop2');out={};edges={}
 for q in meta['queries']:
  a=A[q['id']];rows=[];ers=[]
  for length in [6,10,16]:
   for start in range(2,len(a)-length-1,3):
    ms=[]
    for c in meta['calibration']:
     r=match(a[start:start+length],A[c['id']][2:-2]);r['id']=c['id'];r['start']+=2;r['end']+=2;r['path']=[[x+start,y+2] for x,y in r['path']];ms.append(r)
    rows.append(dict(start=start,end=start+length,matches=sorted(ms,key=lambda r:-r['similarity'])[:12]))
  for n in [5,8,12,16]:
   if len(a)<n+4:continue
   ms=[]
   for c in meta['calibration']:
    r=match(a[2:2+n],A[c['id']][2:-2],True);r['id']=c['id'];r['start']+=2;r['end']+=2;ms.append(r)
   ers.append(dict(length=n,matches=sorted(ms,key=lambda r:-r['similarity'])[:12]))
  out[q['id']]=rows;edges[q['id']]=ers
 Path('local_dtw_drop2.json').write_text(json.dumps(out));Path('edges_drop2.json').write_text(json.dumps(edges))
 cs={c['id']:c for c in meta['calibration']}
 with open('dtw_evidence.txt','w') as f:
  for qid,rows in out.items():
   print('\nQUERY',qid,file=f)
   for row in rows:
    if row['end']-row['start']!=10:continue
    print(f"{row['start']}:{row['end']}",' | '.join(f"{v['similarity']:.2f} {v['id']}:{v['start']}-{v['end']} "+' '.join(cs[v['id']]['tokens']) for v in row['matches'][:4]),file=f)
 with open('edge_evidence.txt','w') as f:
  for qid,rows in edges.items():
   print('\nQUERY',qid,file=f)
   for row in rows[:3]:
    print(row['length'],' | '.join(f"{v['similarity']:.2f} {v['id']}:{v['start']}-{v['end']} "+' '.join(cs[v['id']]['tokens']) for v in row['matches'][:5]),file=f)
 print('saved DTW and edge evidence')
