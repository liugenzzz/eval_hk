import json
import pytest
import yaml
from eval_tool.industrial_mapping import scan, freeze, FrozenMapping, file_sha256


def rec(obj='aircraft', defect='crack', image='a.png', lang='en'):
    answer = f'Category: {obj}; Anomaly type: {defect}; Description: visible' if lang == 'en' else f'类别：{obj}；异常类型：{defect}；描述：可见'
    return {'images': ['/code/defect-detect-images/aircraft/'+image], 'conversations':[{'from':'human','value':'<image>q'},{'from':'gpt','value':answer}]}


def dataset(tmp_path, en=None, zh=None):
    ep, zp = tmp_path/'en.jsonl', tmp_path/'zh.jsonl'
    for path, rows in [(ep, en or [rec()]),(zp, zh or [rec('飞机','裂纹',lang='zh')])]:
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    scan(ep,zp,tmp_path/'scan')
    return ep,zp,tmp_path/'scan'/'candidate.yaml'


def reviews(candidate, transform=None):
    queue = [json.loads(line) for line in (candidate.parent/'review_queue.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for e in queue:
        row={'entry_id':e['id'],'decision':'accepted','canonical':e['canonical_candidates'][0] if e['canonical_candidates'] else 'new_label',
             'defect_status':'abnormal' if e['kind']=='defect' else None,'reason':'semantic check','provider':'private','model':'test','candidate_sha256':file_sha256(candidate)}
        if transform: transform(row)
        rows.append(row)
    path=candidate.parent/'reviews.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return path


def test_scan_freeze_resolve(tmp_path):
    en,zh,c=dataset(tmp_path); r=reviews(c); out=tmp_path/'frozen.yaml'
    with pytest.raises(ValueError,match='frozen'): FrozenMapping.load(c)
    freeze(c,r,out); m=FrozenMapping.load(out)
    assert m.validate_sources(en,zh)
    assert m.resolve(rec(), 'en')['defect_status']=='abnormal'
    assert m.resolve(rec('飞机','裂纹',lang='zh'),'zh')['object_category']=='aircraft'
    assert m.resolve(rec('new object'),'en')['status']=='unresolved'
    assert m.resolve(rec(defect='new defect'),'en')['status']=='unresolved'
    en.write_text('{}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='sha256'): m.validate_sources(en,zh)


@pytest.mark.parametrize('field,value,pattern', [('candidate_sha256','wrong','sha256'),('canonical','invented','candidates'),('defect_status','unknown','normal'),('decision','maybe','decision'),('provider','','provider')])
def test_invalid_review(tmp_path,field,value,pattern):
    _,_,c=dataset(tmp_path); r=reviews(c,lambda row:row.update({field:value}))
    with pytest.raises(ValueError,match=pattern):freeze(c,r,tmp_path/'frozen.yaml')


def test_missing_review_blocks_even_partial(tmp_path):
    _,_,c=dataset(tmp_path); r=c.parent/'empty.jsonl';r.write_text('',encoding='utf-8')
    with pytest.raises(ValueError,match='missing'):freeze(c,r,tmp_path/'frozen.yaml',True)


def test_partial_rejection_explicit(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c,lambda row:row.update(decision='uncertain',canonical=None))
    freeze(c,r,tmp_path/'frozen.yaml')
    result = FrozenMapping.load(tmp_path/'frozen.yaml').resolve(rec(),'en')
    assert result['status']=='unresolved'
    assert result['exclusion_reason']=='review_excluded'


def test_image_conflict_excluded(tmp_path):
    _,_,c=dataset(tmp_path,en=[rec(),rec(defect='scratch')]);r=reviews(c)
    freeze(c,r,tmp_path/'frozen.yaml')
    assert FrozenMapping.load(tmp_path/'frozen.yaml').resolve(rec(),'en')['status']=='ambiguous'


def test_unknown_normal_substring_requires_review(tmp_path):
    _,_,c=dataset(tmp_path,en=[rec(defect='abnormal')])
    doc=yaml.safe_load(c.read_text(encoding='utf-8'))
    e=next(e for e in doc['entries'] if e['kind']=='defect' and e['language']=='en')
    assert e['defect_status']=='unknown' and e['status']=='uncertain'


def test_conflicting_rule_rejected(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c);doc=freeze(c,r,tmp_path/'frozen.yaml')
    e=dict(doc['entries'][0]);e['id']='another-id';e['canonical']='different';doc['entries'].append(e)
    with pytest.raises(ValueError,match='conflicting'):FrozenMapping(doc)


def test_retry_success_is_idempotent(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c)
    r.write_text(r.read_text()+r.read_text().splitlines()[0]+'\n',encoding='utf-8')
    freeze(c,r,tmp_path/'frozen.yaml')
    assert FrozenMapping.load(tmp_path/'frozen.yaml').resolve(rec(),'en')['status']=='resolved'


def test_network_error_is_audited_and_excluded(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c,lambda row:row.update(status='error'))
    doc=freeze(c,r,tmp_path/'frozen.yaml')
    assert doc['freeze_audit']['review_error_entry_ids']
    assert FrozenMapping(doc).resolve(rec(),'en')['status']=='unresolved'


@pytest.mark.parametrize('field,value', [('sources', {}), ('version', ''), ('entries', None), ('excluded_images', 'image')])
def test_frozen_schema_validation(tmp_path,field,value):
    _,_,c=dataset(tmp_path);r=reviews(c);doc=freeze(c,r,tmp_path/'frozen.yaml')
    doc[field]=value
    with pytest.raises(ValueError):FrozenMapping(doc)


def test_source_change_during_scan_rejected(tmp_path,monkeypatch):
    from eval_tool import industrial_mapping as module
    en=tmp_path/'en.jsonl';zh=tmp_path/'zh.jsonl'
    en.write_text(json.dumps(rec())+'\n');zh.write_text(json.dumps(rec('飞机','裂纹',lang='zh'))+'\n')
    real=module.iter_jsonl
    def mutating(path):
        yield from real(path)
        if path==en:
            with en.open('a') as f:f.write('\n')
    monkeypatch.setattr(module,'iter_jsonl',mutating)
    with pytest.raises(ValueError,match='changed during scan'):scan(en,zh,tmp_path/'scan')
    assert not (tmp_path/'scan'/'candidate.yaml').exists()

def test_canonical_defect_status_conflict_rejected(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c)
    rows=[json.loads(line) for line in r.read_text(encoding='utf-8').splitlines()]
    rows[0]['defect_status']='normal'
    r.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    with pytest.raises(ValueError,match='conflicting canonical defect status'):
        freeze(c,r,tmp_path/'frozen.yaml')


def test_ambiguous_object_binding_is_not_crosswired(tmp_path):
    en=[rec('aircraft','crack','a.png'),rec('bottle','scratch','b.png')]
    zh=[rec('物体','缺陷','a.png','zh'),rec('物体','缺陷','b.png','zh')]
    _,_,c=dataset(tmp_path,en=en,zh=zh);r=reviews(c)
    rows=[json.loads(line) for line in r.read_text(encoding='utf-8').splitlines()]
    doc=yaml.safe_load(c.read_text(encoding='utf-8'))
    entries={e['id']:e for e in doc['entries']}
    for row in rows:
        e=entries[row['entry_id']]
        if e['language']=='zh':
            row['canonical']='aircraft' if e['kind']=='object' else 'scratch'
    r.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    with pytest.raises(ValueError,match='binding'):
        freeze(c,r,tmp_path/'frozen.yaml')


def test_scan_streams_inputs_without_read_text(tmp_path,monkeypatch):
    from pathlib import Path
    en=tmp_path/'en.jsonl';zh=tmp_path/'zh.jsonl'
    en.write_text(json.dumps(rec())+'\n');zh.write_text(json.dumps(rec('飞机','裂纹',lang='zh'))+'\n')
    def fail(*a,**kw):raise AssertionError('whole-file read forbidden')
    monkeypatch.setattr(Path,'read_text',fail)
    scan(en,zh,tmp_path/'scan')


def test_latest_network_error_revokes_prior_success(tmp_path):
    _,_,c=dataset(tmp_path);r=reviews(c)
    prior=json.loads(r.read_text(encoding='utf-8').splitlines()[0])
    prior.update(status='error',decision='uncertain')
    with r.open('a',encoding='utf-8') as f:f.write(json.dumps(prior)+'\n')
    doc=freeze(c,r,tmp_path/'frozen.yaml')
    assert prior['entry_id'] in doc['freeze_audit']['review_error_entry_ids']


def test_all_chinese_aliases_require_semantic_review(tmp_path):
    _,_,c=dataset(tmp_path,en=[rec(defect='normal')],zh=[rec('飞机','正常',lang='zh')])
    doc=yaml.safe_load(c.read_text(encoding='utf-8'))
    assert all(e['status']=='uncertain' for e in doc['entries'] if e['language']=='zh')
    assert all(e['status']=='accepted' for e in doc['entries'] if e['language']=='en')
