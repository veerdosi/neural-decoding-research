from local_retrieval import *
import ctypes
lib=ctypes.CDLL(str(Path('fast_dtw.so').resolve()))
lib.subdtw.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_void_p,ctypes.c_void_p]
def subsequence(a,b,penalty=.035):
 S=np.ascontiguousarray(a@b.T,dtype=np.float64);n,m=S.shape
 D=np.empty((n+1,m+1));P=np.zeros((n+1,m+1),np.int8)
 lib.subdtw(S.ctypes.data,n,m,penalty,D.ctypes.data,P.ctypes.data)
 j=int(np.argmax(D[n]));score=D[n,j]/n;end=j;i=n;path=[]
 while i>0:
  p=P[i,j];path.append((i-1,j-1))
  if p==0:i-=1;j-=1
  elif p==1:path.append((i-2,j-1));i-=2;j-=1
  else:path.append((i-1,j-2));i-=1;j-=2
 return {'similarity':float(score),'start':j,'end':end,'path':path[::-1]}
if __name__=='__main__':
 A=make_arrays('drop2');out={}
 for q in meta['queries']:
  qid=q['id'];rows=[]
  for length in [6,10,16]:
   for start in range(2,len(A[qid])-length-1,3):
    matches=[]
    for c in meta['calibration']:
     r=subsequence(A[qid][start:start+length],A[c['id']][2:-2]);r['id']=c['id'];r['start']+=2;r['end']+=2
     r['path']=[(int(p[0]+start),int(p[1]+2)) for p in r['path']];matches.append(r)
    matches.sort(key=lambda r:-r['similarity']);rows.append({'start':start,'end':start+length,'matches':matches[:12]})
  out[qid]=rows
 Path('local_dtw_drop2.json').write_text(json.dumps(out))
 print('wrote DTW evidence for',len(out),'queries')
