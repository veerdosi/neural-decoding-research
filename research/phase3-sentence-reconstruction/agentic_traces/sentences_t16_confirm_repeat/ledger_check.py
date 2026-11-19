from local_retrieval import *
A=make_arrays('drop2');led=json.load(open('calibration_interval_ledger.json'));V=np.array([interval_vector(A[t['id']],t['start'],t['end']) for t in led]);S=V@V.T
for i,t in enumerate(led):
 S[i,[j for j,x in enumerate(led) if x['id']==t['id']]]=-1
 ix=np.argsort(S[i])[-3:][::-1]
 if led[ix[0]]['word']!=t['word']:
  print(f"{t['id']} {t['word']} {t['start']}-{t['end']} "+' | '.join(f"{led[j]['word']} {S[i,j]:.2f} {led[j]['id']}:{led[j]['start']}-{led[j]['end']}" for j in ix))
