from fast_helpers import *
A=make_arrays('drop2');led=json.load(open('calibration_interval_ledger.json'));out={}
for q in meta['queries']:
 qid=q['id'];rows=[]
 for c in meta['calibration']:
  r=subsequence(A[qid][2:-3],A[c['id']][2:-2]);r['id']=c['id'];r['tokens']=c['tokens'];r['start']+=2;r['end']+=2
  path=[(i+2,j+2) for i,j in r['path']];r['path']=path
  r['mapped_words']=[]
  for t in [x for x in led if x['id']==c['id']]:
   times=[i for i,j in path if t['start']<=j<t['end']]
   if times:r['mapped_words'].append({'word':t['word'],'start':min(times),'end':max(times)+1})
  rows.append(r)
 rows.sort(key=lambda r:-r['similarity']);out[qid]=rows
Path('whole_phrase_evidence.json').write_text(json.dumps(out))
for q,rs in out.items():
 print('\n'+q)
 for r in rs[:5]:print(f"{r['similarity']:.3f} {r['id']} "+' '.join(f"{t['word']}({t['start']}-{t['end']})" for t in r['mapped_words']))
