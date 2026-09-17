import json
import hashlib
from pathlib import Path
import pytest
import yaml
from PIL import Image
from eval_tool.industrial_quality_sample import sample_quality


def fixture_data(tmp_path, defects=('good', 'crack'), count=2):
    rows = []
    domain = tmp_path/'defect-detect-images'/'parts'
    domain.mkdir(parents=True)
    for d in defects:
        for i in range(count):
            image = domain/f'{d}-{i}.png'
            Image.new('RGB',(8,8),(len(rows) % 256, len(rows) // 256, 127)).save(image)
            rows.append({'images':[str(image)],'conversations':[{'from':'human','value':'<image>Inspect?'},{'from':'gpt','value':f'Category: bolt; Anomaly type: {d}; Description: visible {d}.'}]})
    paths = {}
    for lang in ('en','zh'):
        p=tmp_path/f'{lang}.jsonl'; p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8'); paths[lang]=p
    entries=[]
    for lang in ('en','zh'):
        entries.append(dict(kind='object',source_domain='parts',language=lang,alias='bolt',canonical='bolt',status='accepted'))
        for d in defects:
            entries.append(dict(kind='defect',source_domain='parts',language=lang,alias=d,canonical=d,object_category='bolt',defect_status='normal' if d=='good' else 'abnormal',status='accepted'))
    for i,e in enumerate(entries): e['id']=str(i)
    doc=dict(schema_version=1,version='test_v1',status='frozen',entries=entries,sources={lang:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for lang,p in paths.items()})
    mapping=tmp_path/'mapping.yaml'; mapping.write_text(yaml.safe_dump(doc),encoding='utf-8')
    return paths,mapping,rows,doc


def run(paths,mapping,out,**kw):
    return sample_quality(paths['zh'],paths['en'],mapping,out,zh_n=kw.pop('zh_n',2),en_n=kw.pop('en_n',2),quality_mode=kw.pop('quality_mode','structural'),**kw)


def test_solver_timeout_still_writes_auditable_failure(tmp_path, monkeypatch):
    import eval_tool.industrial_quality_sample as module
    paths, mapping, _, _ = fixture_data(tmp_path)
    def timeout(*args, **kwargs):
        raise ValueError('quota solver did not prove feasibility: time limit')
    monkeypatch.setattr(module, '_solve', timeout)
    with pytest.raises(ValueError, match='solver_unproven'):
        run(paths, mapping, tmp_path/'timeout')
    report = json.loads((tmp_path/'timeout/coverage_report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'blocked'
    assert not (tmp_path/'timeout/industrial_eval.jsonl').exists()


def test_changed_image_successful_reaudit_replaces_prior_verdict(tmp_path):
    from eval_tool.industrial_review import sample_input_hash
    paths, mapping, rows, _ = fixture_data(tmp_path)
    mapping_sha = hashlib.sha256(mapping.read_bytes()).hexdigest()
    cache = []
    for lang in ('zh', 'en'):
        for record in rows:
            base = dict(language=lang, input_hash=sample_input_hash(record,lang,mapping_sha),
                        mapping_sha256=mapping_sha, decision='usable', status='success')
            cache.append(dict(base, cache_key='old', image_sha256='old-image'))
            cache.append(dict(base, cache_key='new', image_sha256=hashlib.sha256(Path(record['images'][0]).read_bytes()).hexdigest()))
    reviews = tmp_path/'reviews.jsonl'
    reviews.write_text(''.join(json.dumps(row)+'\n' for row in cache), encoding='utf-8')
    assert run(paths, mapping, tmp_path/'ok', quality_mode='semantic', quality_review=reviews)['status'] == 'complete'


def test_plan_replay_and_original_preservation(tmp_path):
    p,m,rows,_=fixture_data(tmp_path)
    plan=run(p,m,tmp_path/'plan',plan_only=True)
    assert not (tmp_path/'plan'/'industrial_eval.jsonl').exists()
    report=run(p,m,tmp_path/'sample',plan=tmp_path/'plan'/'coverage_plan.json')
    assert report['selected_ids']==plan['selected_ids']
    selected=[json.loads(x) for x in (tmp_path/'sample'/'industrial_eval.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(selected)==4
    assert {r['metadata']['defect_type'] for r in selected}=={'good','crack'}
    assert all(r['conversations'] in [o['conversations'] for o in rows] for r in selected)
    assert report['minimum_budget']=={'zh':2,'en':2}


def test_budget_failure_emits_no_eval(tmp_path):
    p,m,_,_=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='Coverage blocked'):
        run(p,m,tmp_path/'out',zh_n=1,en_n=1)
    r=json.loads((tmp_path/'out'/'coverage_plan.json').read_text(encoding='utf-8'))
    assert r['minimum_budget']=={'zh':2,'en':2}
    assert not (tmp_path/'out'/'industrial_eval.jsonl').exists()


def test_defect_min_and_status_min(tmp_path):
    p,m,_,_=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='Coverage blocked'):
        run(p,m,tmp_path/'out',defect_min=2)
    report=run(p,m,tmp_path/'ok',zh_n=4,en_n=4,status_min=2)
    assert report['minimum_budget']=={'zh':4,'en':4}


def test_bad_images_and_unknown_labels_visible(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path)
    for row in rows:
        if 'crack' in row['images'][0]: Path(row['images'][0]).write_bytes(b'bad image')
    with pytest.raises(ValueError,match='Coverage blocked'): run(p,m,tmp_path/'out')
    report=json.loads((tmp_path/'out'/'coverage_plan.json').read_text(encoding='utf-8'))
    assert any(c['value']=='crack' and c['available']==0 for c in report['coverage'])
    doc['entries']=[e for e in doc['entries'] if e.get('canonical')!='crack']
    m.write_text(yaml.safe_dump(doc))
    report = run(p,m,tmp_path/'unknown', zh_n=1, en_n=1)
    assert report['status'] == 'complete'
    audit = [json.loads(line) for line in
             (tmp_path/'unknown'/'quality_audit.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(row.get('reason') == 'unresolved_mapping:unresolved' for row in audit)


def test_candidate_rejected_and_source_binding(tmp_path):
    p,m,_,doc=fixture_data(tmp_path)
    doc['status']='candidate'; m.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='frozen'): run(p,m,tmp_path/'out')
    doc['status']='frozen'; m.write_text(yaml.safe_dump(doc))
    p['en'].write_text(p['en'].read_text(encoding='utf-8')+'\n')
    with pytest.raises(ValueError,match='sha256'): run(p,m,tmp_path/'out')


def test_semantic_requires_review(tmp_path):
    p,m,_,_=fixture_data(tmp_path)
    with pytest.raises(ValueError,match='Coverage blocked'): run(p,m,tmp_path/'out',quality_mode='semantic')
    assert 'not_reviewed' in (tmp_path/'out'/'quality_audit.jsonl').read_text(encoding='utf-8')


def test_changed_plan_parameters_rejected(tmp_path):
    p,m,_,_=fixture_data(tmp_path)
    run(p,m,tmp_path/'plan',plan_only=True)
    with pytest.raises(ValueError,match='fingerprint'):
        run(p,m,tmp_path/'out',seed=1,plan=tmp_path/'plan'/'coverage_plan.json')


def test_workpiece_overlap_exact_minimum(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path)
    for i,r in enumerate(rows): r['metadata']={'workpiece_id':i%2}
    for lang,path in p.items():
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); doc['sources'][lang]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    m.write_text(yaml.safe_dump(doc))
    result=run(p,m,tmp_path/'out')
    assert result['minimum_budget']=={'zh':2,'en':2}
    with pytest.raises(ValueError,match='Coverage blocked'): run(p,m,tmp_path/'blocked',zh_n=3,en_n=3)


def test_frozen_mapping_actually_controls_categories(tmp_path):
    p,m,_,doc=fixture_data(tmp_path)
    for e in doc['entries']:
        if e['kind']=='object': e['canonical']='fastener'
        else: e['object_category']='fastener'
    m.write_text(yaml.safe_dump(doc))
    report=run(p,m,tmp_path/'out')
    assert {c['object_category'] for c in report['coverage']}=={'fastener'}


def test_semantic_review_success_retry_and_image_binding(tmp_path):
    from eval_tool.industrial_review import sample_input_hash
    p,m,rows,_=fixture_data(tmp_path)
    mh=hashlib.sha256(m.read_bytes()).hexdigest()
    reviews=[]
    for lang in ('en','zh'):
        for r in rows:
            base=dict(language=lang,input_hash=sample_input_hash(r,lang,mh),mapping_sha256=mh,
                      image_sha256=hashlib.sha256(Path(r['images'][0]).read_bytes()).hexdigest())
            reviews.extend([dict(base,status='error',decision='insufficient'),dict(base,status='success',decision='usable')])
    review=tmp_path/'review.jsonl'; review.write_text(''.join(json.dumps(r)+'\n' for r in reviews),encoding='utf-8')
    assert run(p,m,tmp_path/'ok',quality_mode='semantic',quality_review=review)['status']=='complete'
    for r in rows:
        if 'crack' in r['images'][0]: Image.new('RGB',(8,8),'black').save(r['images'][0])
    with pytest.raises(ValueError,match='Coverage blocked'):
        run(p,m,tmp_path/'bad',quality_mode='semantic',quality_review=review)
    assert 'image_changed' in (tmp_path/'bad'/'quality_audit.jsonl').read_text(encoding='utf-8')


def test_duplicate_questions_and_order_reproducibility(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path)
    alternate=json.loads(json.dumps(rows[0]));alternate['conversations'][0]['value']='<image>What defect?'
    rows.append(alternate)
    def save(items):
        for lang,path in p.items():
            path.write_text(''.join(json.dumps(r)+'\n' for r in items),encoding='utf-8')
            doc['sources'][lang]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        m.write_text(yaml.safe_dump(doc),encoding='utf-8')
    save(rows)
    first=run(p,m,tmp_path/'first')
    save(list(reversed(rows)))
    second=run(p,m,tmp_path/'second')
    assert first['selected_ids']==second['selected_ids']
    selected=[json.loads(x) for x in (tmp_path/'first'/'industrial_eval.jsonl').read_text(encoding='utf-8').splitlines()]
    for lang in ('en','zh'):
        keys=[r['metadata']['source_image'] for r in selected if r['metadata']['language']==lang]
        assert len(keys)==len(set(keys))


def test_no_artifact_overwrite(tmp_path):
    p,m,_,_=fixture_data(tmp_path)
    run(p,m,tmp_path/'out')
    with pytest.raises(FileExistsError):run(p,m,tmp_path/'out')

def _save_fixture_rows(paths, mapping, rows, doc):
    for lang,path in paths.items():
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        doc['sources'][lang]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    mapping.write_text(yaml.safe_dump(doc),encoding='utf-8')


def test_identical_image_bytes_different_paths_cannot_duplicate(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path,count=1)
    copied=Path(rows[0]['images'][0]).with_name('copied-good.png')
    copied.write_bytes(Path(rows[0]['images'][0]).read_bytes())
    duplicate=json.loads(json.dumps(rows[0]));duplicate['images']=[str(copied)]
    rows.append(duplicate)
    _save_fixture_rows(p,m,rows,doc)
    with pytest.raises(ValueError,match='Coverage blocked'):
        run(p,m,tmp_path/'blocked',zh_n=3,en_n=3)
    result=run(p,m,tmp_path/'ok')
    assert result['status']=='complete'
    selected=[json.loads(x) for x in (tmp_path/'ok'/'industrial_eval.jsonl').read_text(encoding='utf-8').splitlines()]
    for lang in ('en','zh'):
        hashes=[hashlib.sha256(Path(r['images'][0]).read_bytes()).hexdigest() for r in selected if r['metadata']['language']==lang]
        assert len(hashes)==len(set(hashes))==2


def test_identical_image_bytes_conflicting_labels_rejected(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path,count=1)
    Path(rows[1]['images'][0]).write_bytes(Path(rows[0]['images'][0]).read_bytes())
    _save_fixture_rows(p,m,rows,doc)
    with pytest.raises(ValueError,match='Coverage blocked'):
        run(p,m,tmp_path/'blocked')
    assert not (tmp_path/'blocked'/'industrial_eval.jsonl').exists()
    assert 'conflict' in (tmp_path/'blocked'/'quality_audit.jsonl').read_text(encoding='utf-8')


def test_workpiece_ids_scoped_to_source_domain(tmp_path):
    p,m,rows,doc=fixture_data(tmp_path,count=1)
    other=tmp_path/'defect-detect-images'/'other_parts'
    other.mkdir()
    moved=other/'crack.png'
    Path(rows[1]['images'][0]).rename(moved)
    rows[1]['images']=[str(moved)]
    for row in rows:row['metadata']={'workpiece_id':'same-id-in-distinct-datasets'}
    additional=[]
    for entry in doc['entries']:
        copy=dict(entry,source_domain='other_parts',id='other-'+entry['id'])
        additional.append(copy)
    doc['entries'].extend(additional)
    _save_fixture_rows(p,m,rows,doc)
    assert run(p,m,tmp_path/'ok')['status']=='complete'
