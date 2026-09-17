"""Versioned bilingual industrial mapping; no model or inference calls at runtime."""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import yaml
from .industrial_labels import extract_labels, image_identity, normalize, _canonical


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path):
    with open(path, encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            try: record = json.loads(line)
            except (ValueError, TypeError) as exc: raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if not isinstance(record, dict): raise ValueError(f'{path}:{line_no}: expected object')
            yield line_no, record


def _write_yaml(path, doc):
    Path(path).write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')


def scan(en, zh, out_dir, version='industrial_labels_v1'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = [out/'candidate.yaml', out/'review_queue.jsonl', out/'scan_report.json']
    if any(p.exists() for p in paths): raise ValueError('refusing to overwrite mapping artifacts')
    groups = {'en': defaultdict(set), 'zh': defaultdict(set)}
    contexts = {}; invalid = []; counts = {}
    sources = {}
    for language, path in [('en', en), ('zh', zh)]:
        sources[language] = {'path': str(Path(path)), 'sha256': file_sha256(path)}
        count = 0
        for line_no, record in iter_jsonl(path):
            count += 1
            try:
                image, domain = image_identity(record); obj, defect = extract_labels(record)
            except (ValueError, TypeError, AttributeError) as exc:
                invalid.append({'language': language, 'line': line_no, 'reason': str(exc)}); continue
            obj, defect = normalize(obj), normalize(defect)
            groups[language][(image, domain)].add((obj, defect))
            contexts.setdefault((language, domain, obj, defect), {'image_key': image, 'line': line_no,
                'raw_object': obj, 'raw_defect': defect})
        counts[language] = count
    conflicts = {image for lang in groups.values() for (image, domain), pairs in lang.items() if len(pairs) != 1}
    candidates = defaultdict(set)
    bindings = defaultdict(set)
    for identity, zh_pairs in groups['zh'].items():
        if identity[0] in conflicts: continue
        en_pairs = groups['en'].get(identity, set())
        if len(en_pairs) != 1 or len(zh_pairs) != 1: continue
        eo, ed = next(iter(en_pairs)); zo, zd = next(iter(zh_pairs)); domain = identity[1]
        if _canonical(eo, True): candidates[('object', domain, zo, None)].add(_canonical(eo, True))
        if _canonical(ed): candidates[('defect', domain, zd, zo)].add(_canonical(ed))
        if _canonical(ed) and _canonical(eo, True):
            bindings[(domain, zd, zo)].add((_canonical(eo, True), _canonical(ed)))
    entries = []; seen = set()
    for language in ('en', 'zh'):
        for (image, domain), pairs in sorted(groups[language].items()):
            for obj, defect in sorted(pairs):
                for kind, alias, object_alias in [('object', obj, None), ('defect', defect, obj)]:
                    key = (kind, domain, language, alias, object_alias)
                    if key in seen: continue
                    seen.add(key)
                    values = ([_canonical(alias, kind == 'object')] if language == 'en'
                              else sorted(candidates[(kind, domain, alias, object_alias)]))
                    values = [v for v in values if v]
                    canonical = values[0] if len(values) == 1 else None
                    # Only exact normal labels are safe; all abnormal semantics need review.
                    normal = normalize(alias) in {'normal', 'good', 'ok', '正常', '无缺陷', '无异常'}
                    status = 'accepted' if language == 'en' and canonical and (kind == 'object' or normal) else 'uncertain'
                    oid = (_canonical(obj, True) if language == 'en' else
                           next(iter(candidates[('object', domain, obj, None)]), None)
                           if len(candidates[('object', domain, obj, None)]) == 1 else None)
                    entry = {'id': hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:24],
                        'kind': kind, 'source_domain': domain, 'language': language, 'alias': alias,
                        'object_category': oid if kind == 'defect' else None, 'canonical': canonical,
                        'defect_status': ('normal' if normal else 'unknown') if kind == 'defect' else None,
                        'status': status, 'decision_source': 'english' if language == 'en' else 'shared_image',
                        'canonical_candidates': values, 'evidence': {'object_alias': object_alias,
                        'canonical_bindings': [list(pair) for pair in sorted(bindings[(domain, alias, object_alias)])] if language == 'zh' and kind == 'defect' else [],
                        'example': contexts[(language, domain, obj, defect)]}}
                    entries.append(entry)
    for language, path in [('en', en), ('zh', zh)]:
        if file_sha256(path) != sources[language]['sha256']:
            raise ValueError(f'{language} source changed during scan')
    doc = {'schema_version': 1, 'version': version, 'status': 'candidate', 'sources': sources,
           'entries': entries, 'excluded_images': sorted(conflicts), 'review_required': True}
    _write_yaml(paths[0], doc)
    candidate_hash = file_sha256(paths[0])
    queue = [dict(e, entry_id=e['id'], candidate_sha256=candidate_hash) for e in entries if e['status'] != 'accepted']
    paths[1].write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in queue), encoding='utf-8')
    report = {'sources': sources, 'records': counts, 'entry_count': len(entries), 'review_count': len(queue),
              'invalid_records': invalid, 'excluded_images': sorted(conflicts), 'candidate_sha256': candidate_hash}
    paths[2].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return doc, report


def _valid_id(value):
    return isinstance(value, str) and re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', value) is not None


def freeze(candidate, review_jsonl, out_yaml, allow_partial=False):
    if Path(out_yaml).exists(): raise ValueError('refusing to overwrite frozen mapping')
    doc = yaml.safe_load(Path(candidate).read_text(encoding='utf-8'))
    if doc.get('status') != 'candidate' or doc.get('schema_version') != 1: raise ValueError('expected candidate schema 1')
    candidate_hash = file_sha256(candidate)
    entries = {e['id']: e for e in doc['entries']}
    if len(entries) != len(doc['entries']): raise ValueError('duplicate entry id')
    reviewed = set()
    failed = set()
    successful = {}
    for _, decision in iter_jsonl(review_jsonl):
        eid = decision.get('entry_id')
        if eid not in entries: raise ValueError('unknown review entry_id')
        if decision.get('candidate_sha256') != candidate_hash: raise ValueError('review candidate_sha256 mismatch')
        if decision.get('status', 'success') == 'error':
            successful.pop(eid, None)
            failed.add(eid)
            reviewed.add(eid)
            continue
        failed.discard(eid)
        if decision.get('status', 'success') != 'success': raise ValueError('invalid review status')
        successful[eid] = decision
    for eid, decision in successful.items():
        verdict = decision.get('decision')
        if verdict not in {'accepted', 'rejected', 'uncertain'}: raise ValueError('invalid review decision')
        if not all(isinstance(decision.get(k), str) and decision[k].strip() for k in ('reason', 'provider', 'model')):
            raise ValueError('review requires reason, provider and model')
        e = entries[eid]; canonical = decision.get('canonical')
        if verdict == 'accepted':
            if not _valid_id(canonical): raise ValueError('canonical must be ASCII snake_case id')
            if e['canonical_candidates'] and canonical not in e['canonical_candidates']: raise ValueError('canonical outside candidates')
            if e['kind'] == 'defect' and decision.get('defect_status') not in {'normal', 'abnormal'}:
                raise ValueError('accepted defect needs normal or abnormal status')
        e.update(status=verdict, canonical=canonical if verdict == 'accepted' else None,
                 defect_status=decision.get('defect_status') if e['kind'] == 'defect' else None,
                 decision_source='model', review=decision)
        reviewed.add(eid)
    required = {e['id'] for e in doc['entries'] if e.get('decision_source') != 'model' and e['status'] != 'accepted'}
    if required - reviewed: raise ValueError('missing reviews for candidate entries')
    objects = {(e['source_domain'], e['language'], normalize(e['alias'])): e['canonical']
               for e in doc['entries'] if e['kind'] == 'object' and e['status'] == 'accepted'}
    for e in doc['entries']:
        if e['kind'] == 'defect':
            e['object_category'] = objects.get((e['source_domain'], e['language'], normalize(e['evidence']['object_alias'])))
            if not e['object_category'] and e['status'] == 'accepted':
                e['status'] = 'uncertain'; e['exclusion_reason'] = 'object_unresolved'
            bindings = e['evidence'].get('canonical_bindings', [])
            if e['status'] == 'accepted' and bindings and [e['object_category'], e['canonical']] not in bindings:
                raise ValueError('reviewed defect canonical is inconsistent with reviewed object binding')
    unresolved = [e['id'] for e in doc['entries'] if e['status'] != 'accepted']
    doc.update(status='frozen', review_required=False, candidate_sha256=candidate_hash,
               freeze_audit={'allow_partial': allow_partial, 'excluded_entry_ids': unresolved,
                             'review_error_entry_ids': sorted(failed)})
    FrozenMapping(doc)
    _write_yaml(out_yaml, doc)
    return doc


class FrozenMapping:
    def __init__(self, doc):
        if not isinstance(doc, dict) or doc.get('status') != 'frozen' or doc.get('schema_version') != 1:
            raise ValueError('a frozen schema 1 mapping is required')
        if not isinstance(doc.get('version'), str) or not doc['version'].strip():
            raise ValueError('mapping version is required')
        if not isinstance(doc.get('entries'), list) or not isinstance(doc.get('sources'), dict):
            raise ValueError('mapping entries and sources are required')
        for language in ('en', 'zh'):
            source = doc['sources'].get(language)
            if not isinstance(source, dict) or not isinstance(source.get('path'), str) or not source['path'] or not isinstance(source.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', source['sha256']):
                raise ValueError('mapping source path and sha256 are required')
        excluded = doc.get('excluded_images', [])
        if not isinstance(excluded, list) or not all(isinstance(x, str) for x in excluded):
            raise ValueError('excluded_images must be strings')
        self.doc = doc; self.entries = {}; self.excluded_images = set(excluded)
        review_error_ids = set(doc.get('freeze_audit', {}).get('review_error_entry_ids', []))
        excluded_ids = set(doc.get('freeze_audit', {}).get('excluded_entry_ids', []))
        self.excluded_keys = {}
        ids = set()
        canonical_statuses = {}
        for e in doc['entries']:
            if not isinstance(e, dict) or not isinstance(e.get('id'), str) or not e['id'] or e['id'] in ids:
                raise ValueError('invalid or duplicate entry id')
            ids.add(e['id'])
            if e.get('status') not in {'accepted', 'rejected', 'uncertain'}:
                raise ValueError('invalid mapping status')
            if not isinstance(e.get('source_domain'), str) or not isinstance(e.get('alias'), str) or not e['alias'].strip():
                raise ValueError('mapping source_domain and alias must be strings')
            if e['id'] in excluded_ids:
                key = (e['kind'], e['source_domain'], e['language'],
                       normalize(e['alias']), e.get('object_category'))
                self.excluded_keys[key] = ('review_error' if e['id'] in review_error_ids
                                           else 'review_excluded')
            if e['status'] != 'accepted': continue
            if e['kind'] not in {'object', 'defect'} or e['language'] not in {'en', 'zh'} or not _valid_id(e['canonical']):
                raise ValueError('invalid accepted mapping rule')
            if e['kind'] == 'defect':
                if not _valid_id(e['object_category']) or e['defect_status'] not in {'normal', 'abnormal'}:
                    raise ValueError('defect requires object_category and explicit normal/abnormal')
                status_key = (e['source_domain'], e['object_category'], e['canonical'])
                if status_key in canonical_statuses and canonical_statuses[status_key] != e['defect_status']:
                    raise ValueError('conflicting canonical defect status')
                canonical_statuses[status_key] = e['defect_status']
            key = (e['kind'], e['source_domain'], e['language'], normalize(e['alias']), e['object_category'] if e['kind'] == 'defect' else None)
            if key in self.entries and self.entries[key]['canonical'] != e['canonical']: raise ValueError('conflicting mapping rules')
            self.entries[key] = e

    @classmethod
    def load(cls, path):
        return cls(yaml.safe_load(Path(path).read_text(encoding='utf-8')))

    def validate_sources(self, en=None, zh=None, **paths):
        for language, path in [('en', en), ('zh', zh)]:
            if path is not None and file_sha256(path) != self.doc['sources'][language]['sha256']:
                raise ValueError(f'{language} source sha256 mismatch')
        return True

    def resolve(self, record, lang):
        if lang not in {'en', 'zh'}: raise ValueError('language must be en or zh')
        image, domain = image_identity(record); obj, defect = extract_labels(record)
        oe = self.entries.get(('object', domain, lang, normalize(obj), None))
        canonical_obj = oe['canonical'] if oe else None
        de = self.entries.get(('defect', domain, lang, normalize(defect), canonical_obj))
        object_exclusion = self.excluded_keys.get(('object', domain, lang, normalize(obj), None))
        defect_exclusion = self.excluded_keys.get(('defect', domain, lang, normalize(defect), canonical_obj))
        return {'image_key': image, 'source_domain': domain, 'object_category': canonical_obj,
                'defect_type': de['canonical'] if de else None, 'raw_object_category': obj, 'raw_defect_type': defect,
                'mapping_source': 'frozen', 'defect_status': de['defect_status'] if de else 'unknown',
                'status': 'ambiguous' if image in self.excluded_images else 'resolved' if oe and de else 'unresolved',
                'exclusion_reason': object_exclusion or defect_exclusion}


def validate_sources(mapping, en, zh):
    return mapping.validate_sources(en, zh)
