from local_retrieval import *
ledger=json.load(open('calibration_interval_ledger.json'))
report={}
for tag in ['drop2','pca64drop2','pca32drop2','white']:
 A=make_arrays(tag);V=np.array([interval_vector(A[t['id']],t['start'],t['end']) for t in ledger]);S=V@V.T
 for i,t in enumerate(ledger):
  S[i,[j for j,x in enumerate(ledger) if x['id']==t['id']]]=-1
 ix=S.argmax(1);valid=[i for i,t in enumerate(ledger) if sum(x['word']==t['word'] for x in ledger)>1]
 acc=np.mean([ledger[i]['word']==ledger[ix[i]]['word'] for i in valid])
 report[tag]={'accuracy':float(acc),'n':len(valid)}
 print(tag,round(acc,3))
Path('calibration_representation_validation.json').write_text(json.dumps(report,indent=2))
