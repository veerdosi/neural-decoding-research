from retrieval import *
import sys
local=json.load(open('local_drop2.json'))
cal={c['id']:c for c in meta['calibration']}
for q in meta['queries']:
 if len(sys.argv)>1 and q['id'] not in sys.argv[1:]:continue
 print('\n'+q['id']+' '+str(q['shape'][0]))
 for r in local[q['id']]:
  if r['end']-r['start']!=8: continue
  print('%2d:%2d '%(r['start'],r['end'])+' | '.join('%.2f %s %d:%d %s'%(s['similarity'],s['id'],s['start'],s['end'],' '.join(cal[s['id']]['tokens'])) for s in r['matches'][:3]))
