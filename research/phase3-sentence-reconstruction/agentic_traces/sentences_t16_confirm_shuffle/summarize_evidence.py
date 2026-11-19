import json
m=json.load(open('inputs.json'));d=json.load(open('template_evidence.json'))
with open('template_readable.txt','w') as f:
 for q in m['queries']:
  qid=q['id'];v=d[qid];f.write('\n'+qid+'\n')
  for seg in v['segments']:
   f.write(f"SEG {seg['start']}-{seg['end']} "+' | '.join(f"{r['word']} {r['similarity']:.2f} {r['id']}:{r['start']}-{r['end']}" for r in seg['matches'][:5])+'\n')
  for s in range(2,q['shape'][0]-8,4):
   r=next(r for r in v['interval_candidates'] if r['start']==s and r['end']==s+8)
   f.write(f"WIN {s}-{s+8} "+' | '.join(f"{r['word']} {r['similarity']:.2f} {r['id']}:{r['start']}-{r['end']}" for r in r['matches'])+'\n')
