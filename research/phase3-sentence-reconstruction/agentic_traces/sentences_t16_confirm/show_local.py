from retrieval import *
import sys
name=sys.argv[1] if len(sys.argv)>1 else 'drop2'
qid=sys.argv[2] if len(sys.argv)>2 else meta['queries'][0]['id']
q=json.load(open('local_'+name+'.json'))[qid]
for row in q:
 if row['end']-row['start']!=8:continue
 print('\nQUERY',qid,row['start'],row['end'])
 for r in row['matches'][:6]:
  c=meta['calibration'][int(r['id'][1:])]
  print(round(r['similarity'],3),r['id'],[r['start'],r['end']],len(raw[r['id']]),' '.join(c['tokens']))
