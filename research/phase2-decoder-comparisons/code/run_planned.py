"""Run the actual Sol conditions through authenticated Codex at high effort."""
import json
from experiment import ROOT,PACKAGE,bounded_queue

manifest=json.loads((PACKAGE/'planned_calls.json').read_text())
names=[name for group in manifest.values() for name in group]
bounded_queue(names)
completed=[]
for name in names:
 path=ROOT/'responses'/f'{name}.invocation.json'
 if path.exists() and json.loads(path.read_text()).get('returncode')==0:completed.append(name)
print(f'Completed {len(completed)} of {len(names)} model conditions.')
