"""Stream selected public archive members with resumable HTTP ranges.

The archive is never stored locally. Only the selected source MAT file is kept.
"""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import argparse,concurrent.futures,hashlib,http.client,io,json,pathlib,re,tarfile,time,urllib.request

class RangeStream(io.RawIOBase):
 def __init__(self,url,chunk=1024*1024,workers=8):
  self.url=url;self.chunk=chunk;self.position=0;self.buffer=b'';self.total=None;self.last_report=0
  self.pool=concurrent.futures.ThreadPoolExecutor(workers);self.futures={};self.next_index=0;self.next_schedule=0;self.workers=workers
  for _ in range(workers):self.schedule()
 def schedule(self):
  start=self.next_schedule*self.chunk
  if self.total is None or start<self.total:
   self.futures[self.next_schedule]=self.pool.submit(self.fetch,start)
  self.next_schedule+=1
 def fetch(self,start):
  end=start+self.chunk-1;position=start;pieces=[];empty=0;total=None
  while position<=end:
   try:
    request=urllib.request.Request(self.url,headers={'Range':f'bytes={position}-{end}'})
    with urllib.request.urlopen(request,timeout=30) as response:
     match=re.match(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
     if match:
      if int(match[1])!=position:raise OSError('Byte-range start mismatch')
      total=int(match[3]);end=min(end,total-1)
     elif position:raise OSError('Remote server did not honor requested byte range')
     else:total=int(response.headers.get('Content-Length','0')) or None
     try:data=response.read(end-position+1)
     except http.client.IncompleteRead as ex:data=ex.partial
     if not data:raise OSError('Empty transfer segment')
     pieces.append(data);position+=len(data);empty=0
   except Exception:
    empty+=1
    if empty>=4:raise
    time.sleep(min(2**empty,8))
  return b''.join(pieces),total
 def readable(self):return True
 def read(self,n=-1):
  if n<0:n=self.chunk
  out=bytearray()
  while len(out)<n:
   if not self.buffer:
    if self.total is not None and self.position>=self.total:break
    self.buffer,total=self.futures.pop(self.next_index).result();self.total=total or self.total;self.next_index+=1;self.schedule()
   take=min(n-len(out),len(self.buffer));out+=self.buffer[:take];self.buffer=self.buffer[take:];self.position+=take
   if self.position-self.last_report>=32*1024*1024:
    print(f'compressed transfer {self.position//1048576} MiB',flush=True);self.last_report=self.position
  return bytes(out)
 def close(self):
  self.pool.shutdown(wait=False,cancel_futures=True);super().close()

def main(kind):
 root=PROJECT/'local/sources'/f'willett2023_{kind}';root.mkdir(exist_ok=True)
 filename='tuningTasks.tar.gz' if kind=='tuning' else 'sentences.tar.gz';url='https://zenodo.org/records/8047896/files/'+filename+'?download=1'
 target='2022.05.03';remote=RangeStream(url);partial=None
 try:
  with tarfile.open(fileobj=remote,mode='r|gz') as archive:
   for member in archive:
    if not member.isfile():continue
    print('member',member.name,member.size,flush=True)
    if target not in member.name:continue
    dest=root/pathlib.PurePosixPath(member.name).name;partial=dest.with_suffix('.partial');digest=hashlib.sha256()
    with archive.extractfile(member) as src,partial.open('wb') as dst:
     while data:=src.read(1024*1024):dst.write(data);digest.update(data)
    assert partial.stat().st_size==member.size;partial.rename(dest);partial=None
    receipt={'source_url':url,'archive_member':member.name,'size':member.size,'sha256':digest.hexdigest(),'compressed_bytes_consumed':remote.position,'release':'Zenodo June2023 mirror. Compare current Dryad release for known later corrections; May03 is not the June28 session listed in the timing correction.','not_used':'RNN outputs, weights, audio features or language model archives'}
    dest.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2));print('SAVED',dest,digest.hexdigest(),flush=True);return
  raise RuntimeError('Selected recording not found')
 finally:
  if partial and partial.exists():partial.unlink()
  remote.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['tuning','sentences']);main(p.parse_args().kind)
