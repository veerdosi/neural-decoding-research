from retrieval import *
r=json.load(open('local_pca32drop2.json'));cs={c['id']:c for c in meta['calibration']}
lines=[]
for q in meta['queries']:
 lines.append('\n'+q['id']+' '+str(q['shape'][0]))
 for row in r[q['id']]:
  if row['end']-row['start']!=8:continue
  lines.append(f"{row['start']:2}-{row['end']:2} "+' | '.join(f"{m['similarity']:.2f} {m['id']}:{m['start']}-{m['end']} "+' '.join(cs[m['id']]['tokens']) for m in row['matches'][:4]))
Path('retrieval_pca32_summary.txt').write_text('\n'.join(lines))
