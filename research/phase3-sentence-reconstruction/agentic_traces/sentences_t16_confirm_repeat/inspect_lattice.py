import json,sys,numpy as np
ex=json.load(open('exemplars.json'))
for arg in sys.argv[1:]:
 qid,rest=arg.split(':');z=np.load('lattice_'+qid+'.npz');spans=z['spans'];scores=z['scores'];words=z['words'];which=z['which']
 print('\n'+qid)
 for spec in rest.split(','):
  s,e=map(int,spec.split('-'));ix=np.flatnonzero((spans[:,0]==s)&(spans[:,1]==e))[0]
  ii=np.argsort(scores[ix])[-8:][::-1]
  print(spec,' | '.join(f"{words[j]} {scores[ix,j]:.2f} {ex[which[ix,j]]['id']}:{ex[which[ix,j]]['start']}-{ex[which[ix,j]]['end']}" for j in ii))
