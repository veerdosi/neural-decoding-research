"""Reproduce local numerical evidence, then apply explicit human interpretations.
Run with the supplied Python and OPENBLAS_NUM_THREADS=1. No network or external
labels are used. Each subordinate program reads this directory's data/artifacts.
"""
import os,sys,subprocess
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1'
programs=[['prepare_pca.py'],['local_retrieval.py','drop2','pca32drop2'],['evidence_dtw.py'],['template_inference.py'],['consensus_evidence.py'],['phrase_mosaic.py'],['compare_candidates.py'],['word_dtw_evidence.py'],['finalize_reconstruction.py']]
for args in programs:
 with open(Path(args[0]).stem+'_run.log','w') as log:
  subprocess.run([sys.executable,*args],check=True,stdout=log,stderr=subprocess.STDOUT)
 print('Completed',args[0],flush=True)
with open('word_evidence.txt','w') as report:
 subprocess.run([sys.executable,'inspect_words.py'],check=True,stdout=report)
