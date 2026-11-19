from interval_assembly import *
T,L,S=get_templates({k:np.array(b) for k,b in json.load(open('calibration_boundaries_3.json')).items()})
for arg in sys.argv[1:]:
 cid=arg.split(':')[0];a=A[cid]
 print('\n',cid)
 ps=[tuple(map(int,x.split('-'))) for x in arg.split(':')[1:]] if ':' in arg else [(s,min(s+7,len(a)-2)) for s in range(2,len(a)-7,3)]
 for s,e in ps:
  v=interval_vector(a,s,e);sim=v@T.T
  print(s,e,' | '.join(f'{words[k]} {v:.2f}' for k,v in sorted([(k,np.sort(sim[L==k])[-min(3,sum(L==k)):].mean()) for k in range(len(words))],key=lambda p:-p[1])[:8]))
  ix=np.argsort(sim)[::-1][:6]
  print('  '+' | '.join(f'{words[L[j]]}:{S[j][0]}:{S[j][2]}-{S[j][3]} {sim[j]:.2f}' for j in ix))
