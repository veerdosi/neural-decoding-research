from retrieval import *
import sys
R=json.load(open('local_drop2.json'));C={c['id']:c for c in meta['calibration']}
for qid in sys.argv[1:]:
 print('\n'+qid)
 for row in R[qid]:
  if row['end']-row['start']!=5:continue
  print(f"{row['start']}-{row['end']} "+' | '.join(f"{m['similarity']:.2f} {m['id']}:{m['start']}-{m['end']} "+' '.join(C[m['id']]['tokens']) for m in row['matches'][:6]))
