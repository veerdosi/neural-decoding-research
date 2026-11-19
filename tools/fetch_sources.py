"""Download original study sources separately from the Git repository."""
import argparse,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['phase1','phase2','phase3'],default='phase3')
    args=parser.parse_args()
    if args.dataset=='phase1':
        c=json.loads((ROOT/'research/phase1-word-decoding/config.json').read_text())
        files=[{'filename':c['source_url'].rsplit('/',1)[1],'download_url':c['source_url'],'sha256':c['source_sha256']}]
    elif args.dataset=='phase2':
        d=json.loads((ROOT/'research/phase2-decoder-comparisons/source_files.json').read_text())
        files=[{'filename':name,'download_url':v['url'],'sha256':v['sha256']} for name,v in d.items()]
    else:
        d=json.loads((ROOT/'research/phase3-sentence-reconstruction/source_registry.json').read_text())
        files=[v for v in d['sources'] if 'filename' in v and 'download_url' in v]
    dest=ROOT/'local/sources';dest.mkdir(parents=True,exist_ok=True)
    for f in files:
        path=dest/f['filename']
        if not path.exists():
            print('Downloading',f['filename'],flush=True)
            temp=path.with_suffix(path.suffix+'.partial')
            urllib.request.urlretrieve(f['download_url'],temp)
            assert hashlib.sha256(temp.read_bytes()).hexdigest()==f['sha256'],f['filename']
            temp.replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'],f['filename']
        print('Verified',f['filename'],flush=True)
    if args.dataset=='phase3':
        print('The separate unseen-word experiment uses the Willett tuning archive; see source_registry.json and code/fetch_calibration_units.py.')

if __name__=='__main__':main()
