from local_retrieval import *

def subsequence(a,b,penalty=.035):
 S=a@b.T;n,m=S.shape
 D=np.full((n+1,m+1),-1e9);D[0,:]=0
 P=np.zeros((n+1,m+1),np.int8)
 for i in range(1,n+1):
  for j in range(1,m+1):
   cand=[D[i-1,j-1]+S[i-1,j-1]]
   cand.append(D[i-2,j-1]+S[i-2,j-1]+S[i-1,j-1]-penalty if i>=2 else -1e9)
   cand.append(D[i-1,j-2]+.5*(S[i-1,j-2]+S[i-1,j-1])-penalty if j>=2 else -1e9)
   p=np.argmax(cand);D[i,j]=cand[p];P[i,j]=p
 j=np.argmax(D[n,:]);score=D[n,j]/n;end=int(j);i=n;path=[]
 while i>0:
  p=P[i,j];path.append((i-1,j-1))
  if p==0:i-=1;j-=1
  elif p==1:path.append((i-2,j-1));i-=2;j-=1
  else:path.append((i-1,j-2));i-=1;j-=2
 return {'similarity':float(score),'start':int(j),'end':end,'path':path[::-1]}

if __name__=='__main__':
 import sys
 tag=sys.argv[1] if len(sys.argv)>1 else 'drop2';A=make_arrays(tag);out={}
 for q in meta['queries']:
  qid=q['id'];a=A[qid];rows=[]
  for length in [6,10,16]:
   for start in range(2,len(a)-length-1,3):
    matches=[]
    for c in meta['calibration']:
     b=A[c['id']][2:-2]
     r=subsequence(a[start:start+length],b);r['id']=c['id'];r['start']+=2;r['end']+=2
     r['path']=[(int(p[0]+start),int(p[1]+2)) for p in r['path']];matches.append(r)
    matches.sort(key=lambda r:-r['similarity'])
    rows.append({'start':start,'end':start+length,'matches':matches[:12]})
  out[qid]=rows;print(qid,flush=True)
 Path('local_dtw_'+tag+'.json').write_text(json.dumps(out))
