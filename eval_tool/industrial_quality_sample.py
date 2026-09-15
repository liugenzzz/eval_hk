"""Quality controlled bilingual industrial evaluation sampler."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
from .industrial_labels import build_label_mapping

def _load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def _yaml(p):
    try:
        import yaml; return yaml.safe_load(Path(p).read_text(encoding='utf-8')) or {}
    except ImportError:
        return json.loads(Path(p).read_text(encoding='utf-8'))
def sample_quality(zh,en,mapping,out_dir,zh_n=500,en_n=500,seed=42,object_min=1,defect_min=1):
    z,e=_load(zh),_load(en); doc=_yaml(mapping); lm=build_label_mapping(e,z)
    rows=[]; rejected=[]; seen=set()
    for lang,recs in [('zh',z),('en',e)]:
        for i,r in enumerate(recs):
            try: m=lm.resolve(r,lang); key=m['image_key']
            except Exception as ex: rejected.append({'language':lang,'index':i,'reason':str(ex)}); continue
            if m['status']!='resolved': rejected.append({'language':lang,'index':i,'reason':m['status']}); continue
            if (lang,key) in seen: continue
            seen.add((lang,key)); rows.append((lang,key,r,m))
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for lang,n in [('zh',zh_n),('en',en_n)]:
        avail=[x for x in rows if x[0]==lang]; objs=defaultdict(list)
        for x in avail: objs[x[3]['object_category']].append(x)
        if any(len(v)<object_min for v in objs.values()): continue
        rank=lambda x: hashlib.sha256(f'{seed}:{lang}:{x[1]}'.encode()).hexdigest()
        pick=[]
        for v in objs.values(): pick += sorted(v,key=rank)[:object_min]
        pick += [x for x in sorted(avail,key=rank) if x not in pick]
        if len(pick)>=n: selected += pick[:n]
    cov={'requested':{'zh':zh_n,'en':en_n},'selected':{'zh':sum(x[0]=='zh' for x in selected),'en':sum(x[0]=='en' for x in selected)},'rejected':rejected}
    (out/'coverage_report.json').write_text(json.dumps(cov,ensure_ascii=False,indent=2),encoding='utf-8')
    if cov['selected']!={'zh':zh_n,'en':en_n}: return cov
    with (out/'industrial_eval.jsonl').open('w',encoding='utf-8') as f:
        for lang,key,r,m in selected:
            rr=dict(r); rr.setdefault('metadata',{}).update(language=lang,image_key=key,object_category=m['object_category'],defect_type=m['defect_type'],mapping_version=doc.get('version','unknown')); f.write(json.dumps(rr,ensure_ascii=False)+'\n')
    (out/'manifest.json').write_text(json.dumps({'seed':seed,'mapping_version':doc.get('version','unknown'),'count':len(selected)},ensure_ascii=False,indent=2),encoding='utf-8'); return cov
def main():
    p=argparse.ArgumentParser(); [p.add_argument(x,required=True) for x in ('--zh','--en','--mapping','--out-dir')]; p.add_argument('--zh-n',type=int,default=500); p.add_argument('--en-n',type=int,default=500); p.add_argument('--seed',type=int,default=42); p.add_argument('--object-min',type=int,default=1); p.add_argument('--defect-min',type=int,default=1); a=p.parse_args(); sample_quality(a.zh,a.en,a.mapping,a.out_dir,a.zh_n,a.en_n,a.seed,a.object_min,a.defect_min)
if __name__=='__main__': main()
