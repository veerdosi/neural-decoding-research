from local_retrieval import *
import ctypes
lib=ctypes.CDLL('./interval_dtw.so');f=lib.subseq
f.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
A=make_arrays('drop2');out={}
for q in meta['queries']:
 a=A[q['id']];rows=[]
 for length in [6,10,16]:
  for s in range(2,len(a)-length-1,3):
   matches=[]
   for c in meta['calibration']:
    b=A[c['id']][2:-2];S=np.ascontiguousarray(a[s:s+length]@b.T);res=np.zeros(3);f(S,length,len(b),res)
    matches.append({'id':c['id'],'similarity':float(res[0]),'start':int(res[1]+2),'end':int(res[2]+2)})
   matches.sort(key=lambda r:-r['similarity']);rows.append({'start':s,'end':s+length,'matches':matches[:12]})
 out[q['id']]=rows
Path('fast_local_dtw_drop2.json').write_text(json.dumps(out))
with open('fast_dtw_report.txt','w') as fp:
 for qid,rows in out.items():
  fp.write('\n'+qid+'\n')
  for r in rows:
   if r['end']-r['start']!=10:continue
   fp.write('%2d:%2d '%(r['start'],r['end'])+' | '.join('%.2f %s %d:%d %s'%(v['similarity'],v['id'],v['start'],v['end'],' '.join(meta['calibration'][int(v['id'][1:])]['tokens'])) for v in r['matches'][:3])+'\n')
print('saved fast_local_dtw_drop2.json and fast_dtw_report.txt')
