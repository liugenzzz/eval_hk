import json
import yaml
from industrial_mapping_build import build


def test_candidate_mapping_is_utf8_and_auditable(tmp_path):
    def rec(lang):
        answer = 'Category: aircraft; Anomaly type: crack; Description: visible' if lang == 'en' else '类别：飞机；异常类型：裂纹；描述：可见'
        return {'images': ['/code/defect-detect-images/aircraft/a.png'], 'conversations': [{'from':'human','value':'<image>q'},{'from':'gpt','value':answer}]}
    en, zh = tmp_path/'en.jsonl', tmp_path/'zh.jsonl'
    en.write_text(json.dumps(rec('en'))+'\n', encoding='utf-8'); zh.write_text(json.dumps(rec('zh'), ensure_ascii=False)+'\n', encoding='utf-8')
    y, a, r = tmp_path/'mapping.yaml', tmp_path/'audit.json', tmp_path/'review.jsonl'
    doc, _ = build(en, zh, y, a, r)
    assert doc['status'] == 'candidate' and doc['review_required']
    assert yaml.safe_load(y.read_text(encoding='utf-8'))['version'] == 'industrial_labels_v1'
    assert a.exists() and r.exists()
    rows = [json.loads(line) for line in r.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 3
    assert any(row['kind'] == 'object' and row['language'] == 'zh' for row in rows)
