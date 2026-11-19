from local_dtw import *
import ctypes
lib=ctypes.CDLL('./interval_dtw.so');f=lib.subseq
f.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
A=make_arrays('drop2');specs={'ta8a65775':[(20,31),(31,43),(38,53),(31,53)],'t1fe83bc3':[(9,20),(17,25),(23,34)],'t6f0fdfe0':[(20,31),(29,36)],'t41450074':[(2,22)],'td133fe68':[(37,50)],'tb0b19ec8':[(2,23)],'td28e3558':[(2,24)],'t9ce92bac':[(2,25)],'t4d2d7415':[(4,26)]}
out={}
for qid,spans in specs.items():
 print('\n'+qid);out[qid]=[]
 for s,e in spans:
  rows=[]
  for c in meta['calibration']:
   b=A[c['id']][2:-2];S=np.ascontiguousarray(A[qid][s:e]@b.T);r=np.zeros(3);f(S,e-s,len(b),r);rows.append({'id':c['id'],'start':int(r[1]+2),'end':int(r[2]+2),'similarity':float(r[0])})
  rows.sort(key=lambda r:-r['similarity']);out[qid].append({'start':s,'end':e,'matches':rows})
  print(s,e,' | '.join('%.2f %s:%d-%d %s'%(r['similarity'],r['id'],r['start'],r['end'],' '.join(meta['calibration'][int(r['id'][1:])]['tokens'])) for r in rows[:6]))
  if qid=='ta8a65775': print('Computer source:',next(r for r in rows if r['id']=='c058'))
Path('focused_match_evidence.json').write_text(json.dumps(out,indent=2))
