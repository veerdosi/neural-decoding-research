"""Render the reviewed Markdown source as a self-contained reading document."""
from paths import PROJECT, PUBLICATION, PYTHON, resolve_sandbox
import base64,datetime,hashlib,html as html_module,json,mimetypes,re
from pathlib import Path
import markdown
from sequence_study import ROOT,save

def data_uri(p):
 mime=mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
 return 'data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()

def main():
 source=PUBLICATION/'essay.md';text=source.read_text();assert '<!--' not in text and 'TODO' not in text and 'PENDING' not in text
 body=markdown.markdown(text,extensions=['tables','fenced_code','toc']);images=[]
 def image(m):
  p=PUBLICATION/m.group(1);assert p.is_file();images.append({'path':str(p.resolve().relative_to(PROJECT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return 'src="'+data_uri(p)+'"'
 body=re.sub(r'src="([^"]+)"',image,body)
 fonts=PUBLICATION/'fonts'
 css=''
 for family,weight,name in [('Plex',400,'ibm-plex-sans-latin-400-normal.woff2'),('Plex',600,'ibm-plex-sans-latin-600-normal.woff2'),('PlexMono',400,'ibm-plex-mono-latin-400-normal.woff2')]:
  css+=f'@font-face{{font-family:{family};font-style:normal;font-weight:{weight};src:url("{data_uri(fonts/name)}") format("woff2");font-display:swap;}}'
 css+='''
 :root{color-scheme:light;--paper:#f6f4ee;--ink:#242420;--muted:#62625b;--accent:#bd4d27;--rule:#c9c6bd}
 *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:20px/1.68 Plex,Arial,sans-serif;-webkit-font-smoothing:antialiased}
 .folio{max-width:1060px;margin:0 auto;padding:30px 48px 20px;border-bottom:1px solid var(--rule);display:flex;justify-content:space-between;gap:20px;font:12px/1.4 PlexMono,monospace;letter-spacing:.04em;color:var(--muted)}
 main{max-width:1060px;margin:0 auto;padding:52px 48px 72px}h1{max-width:850px;font-size:clamp(38px,5vw,64px);font-weight:600;line-height:1.07;letter-spacing:-.035em;margin:0 0 22px}h2{font-size:30px;line-height:1.2;letter-spacing:-.02em;margin:2.4em 0 .8em;max-width:780px}p,ul,ol{max-width:760px;margin:0 0 1.1em}strong{font-weight:600}a{color:inherit;text-decoration-color:var(--accent);text-underline-offset:.18em;text-decoration-thickness:1px}a:hover{color:var(--accent)}em{font-style:italic}code{font: .79em/1.5 PlexMono,monospace;background:#eae7de;padding:2px 4px}pre{max-width:100%;overflow:auto;border-left:2px solid var(--rule);padding:14px 20px;font-size:15px}pre code{background:none;padding:0}
 p:has(>img){max-width:100%;margin:36px 0 13px}img{display:block;width:100%;height:auto}p:has(>img)+p{max-width:850px;color:var(--muted);font-size:15px;line-height:1.5;margin-bottom:2.4em}p:has(>img)+p em{font-style:normal}
 table{border-collapse:collapse;width:100%;font-size:16px;line-height:1.4;margin:30px 0 28px;font-variant-numeric:tabular-nums}th,td{padding:13px 14px 13px 0;border-bottom:1px solid var(--rule);vertical-align:top}th{text-align:left;font-weight:600;border-top:2px solid var(--ink)}td:first-child{max-width:390px}th:not(:first-child),td:not(:first-child){text-align:right;white-space:nowrap}hr{border:0;border-top:1px solid var(--rule);margin:54px 0 25px}hr~p{font-size:15px;line-height:1.6;color:var(--muted)}.footer{max-width:1060px;margin:0 auto;padding:20px 48px 40px;border-top:1px solid var(--rule);font:11px/1.6 PlexMono,monospace;color:var(--muted)}
 @media(max-width:700px){body{font-size:18px}.folio{padding:24px 22px 16px;font-size:10px}.folio span:last-child{display:none}main{padding:38px 22px 50px}h2{font-size:27px}table{font-size:12px}th,td{padding-right:8px}.footer{padding:20px 22px 35px}}
 @media print{body{background:white;font-size:11pt}main{padding:20px}h1{font-size:32pt}h2{font-size:19pt;break-after:avoid}img,table{break-inside:avoid}.folio,.footer{font-size:8pt}a{color:inherit}}
 '''
 title=html_module.escape(text.splitlines()[0].lstrip('# '));licenses='\n\n'.join((fonts/n).read_text() for n in ['IBM-Plex-sans-LICENSE.txt','IBM-Plex-mono-LICENSE.txt']);license_html='<template id="embedded-font-licenses"><pre>'+html_module.escape(licenses)+'</pre></template>'
 html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A technical research account of GPT-6.1 Sol learning from neural recordings at inference time, with comparisons, controls and failures."><title>'+title+'</title><style>'+css+'</style></head><body><div class="folio"><span>NEURAL / LANGUAGE — RESEARCH ESSAY</span><span>19 NOVEMBER 2026 · RECORDED DATA / OFFLINE INFERENCE</span></div><main>'+body+'</main><div class="footer">Article source: essay.md · Experimental outputs and reproduction code accompany this document.<br>IBM Plex fonts: SIL Open Font License; license texts are embedded and retained in fonts/.</div>'+license_html+'</body></html>'
 target=PUBLICATION/'essay.html';target.write_text(html)
 save(ROOT/'evaluation/essay_build.json',{'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'html_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'figures':images,'images_embedded':True,'fonts_embedded':True,'build_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Markdown_version':markdown.__version__});print('Built self-contained essay.html',len(html),'bytes')

if __name__=='__main__':main()
