"""Frozen-label, audited and quota-constrained industrial sampling."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _keys(row):
    m = row['meta']
    obj, defect, status = m['object_category'], m['defect_type'], m.get('defect_status', 'unknown')
    return [('object', obj, ''), ('defect', obj, defect), ('status', obj, status), ('source_domain', obj, m.get('source_domain', 'unknown'))]


def _solve(rows, expected, minima, count=None, seed=42):
    """Exact integer feasibility handles overlapping workpiece and coverage constraints."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    required = {key: minima[key[0]] for key in expected if key[0] in minima}
    if not rows:
        return None
    constraints = list(sorted(required))
    indexes = {key: i for i, key in enumerate(constraints)}
    groups = sorted({('workpiece', r['group']) for r in rows}
                    | {('content', r['image_sha256']) for r in rows})
    group_index = {g: len(constraints) + i for i, g in enumerate(groups)}
    rr, cc, vv = [], [], []
    for j, row in enumerate(rows):
        for key in _keys(row):
            if key in indexes:
                rr.append(indexes[key]); cc.append(j); vv.append(1.)
        for group in (('workpiece', row['group']), ('content', row['image_sha256'])):
            rr.append(group_index[group]); cc.append(j); vv.append(1.)
    low = [required[k] for k in constraints] + [0] * len(groups)
    high = [np.inf] * len(constraints) + [1] * len(groups)
    if count is not None:
        for j in range(len(rows)):
            rr.append(len(low)); cc.append(j); vv.append(1.)
        low.append(count); high.append(count)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(low), len(rows))).tocsc()
    # Ranking makes ties deterministic, while the first solve minimizes total budget.
    rank_cost = np.array([int(_hash([seed, r['id']])[:12], 16) / 16**12 for r in rows])
    objective = np.ones(len(rows)) + rank_cost / (len(rows) + 1) if count is None else rank_cost
    result = milp(objective, integrality=np.ones(len(rows)), bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, low, high), options={'time_limit': 120})
    if not result.success:
        raise ValueError('quota solver did not prove feasibility: ' + result.message) if result.status != 2 else _Infeasible()
    return [row for row, x in zip(rows, result.x) if x > .5]


class _Infeasible(Exception):
    pass


def sample_quality(zh, en, mapping, out_dir, zh_n=500, en_n=500, seed=42,
                   object_min=1, defect_min=1, status_min=1, plan_only=False,
                   plan=None, image_root=None, quality_review=None,
                   quality_mode='semantic', allow_partial_coverage=False):
    from PIL import Image
    from .industrial_mapping import FrozenMapping
    from .industrial_labels import extract_labels, image_identity
    from .industrial_review import sample_input_hash
    for value in (zh_n, en_n, object_min, defect_min, status_min):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError('counts and minima must be positive integers')
    if quality_mode not in ('semantic', 'structural'):
        raise ValueError('quality_mode must be semantic or structural')
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {out}')
    labels = FrozenMapping.load(mapping)
    labels.validate_sources(en=en, zh=zh)
    mapping_hash = _file_hash(mapping)
    paths = {'zh': Path(zh), 'en': Path(en)}
    source_hashes = {lang: _file_hash(path) for lang, path in paths.items()}
    reviews = {}
    if quality_review:
        with Path(quality_review).open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    r = json.loads(line)
                    key = (r.get('language'), r.get('input_hash'))
                    if (key in reviews and reviews[key] != r
                        and reviews[key].get('status') == r.get('status') == 'success'
                        and reviews[key].get('cache_key') == r.get('cache_key')):
                        raise ValueError('conflicting duplicate quality review')
                    # A new prompt/model/image audit replaces the old verdict. A failed
                    # re-audit cannot silently inherit an earlier successful decision.
                    reviews[key] = r
    params = dict(zh_n=zh_n, en_n=en_n, seed=seed, object_min=object_min,
                  defect_min=defect_min, status_min=status_min, quality_mode=quality_mode,
                  allow_partial_coverage=allow_partial_coverage,
                  image_root=str(Path(image_root).resolve()) if image_root else None)
    fingerprint = {'sources': source_hashes, 'mapping_sha256': mapping_hash,
                   'quality_review_sha256': _file_hash(quality_review) if quality_review else None,
                   'parameters': params}
    saved_plan = json.loads(Path(plan).read_text(encoding='utf-8')) if plan else None
    if saved_plan and saved_plan.get('fingerprint') != fingerprint:
        raise ValueError('plan fingerprint differs: source, mapping, review or parameters changed')
    expected = {'zh': set(), 'en': set()}
    pools = {'zh': [], 'en': []}
    audit, blockers = [], []
    for lang, path in paths.items():
        by_image = defaultdict(list)
        with path.open('rb') as stream:
            line_number = 0
            while True:
                offset = stream.tell(); raw = stream.readline()
                if not raw: break
                line_number += 1
                if not raw.strip(): continue
                issue = {'language': lang, 'source_line': line_number}
                try:
                    record = json.loads(raw.decode('utf-8-sig'))
                    key, _ = image_identity(record)
                    raw_obj, raw_defect = extract_labels(record)
                    issue.update(raw_object_category=raw_obj, raw_defect_type=raw_defect, image_key=key)
                    meta = labels.resolve(record, lang)
                    if meta.get('status') != 'resolved':
                        raise ValueError('unresolved_mapping:' + str(meta.get('status')))
                    row = {'meta': meta}
                    conv = record.get('conversations')
                    if (not isinstance(conv, list) or len(conv) != 2 or
                        [c.get('from') for c in conv] != ['human', 'gpt'] or
                        any(not isinstance(c.get('value'), str) or not c['value'].replace('<image>', '').strip() for c in conv) or
                        conv[0]['value'].count('<image>') > 1):
                        raise ValueError('invalid_single_turn_conversation')
                    if not isinstance(record.get('metadata', {}), dict):
                        raise ValueError('invalid_metadata')
                    input_hash = sample_input_hash(record, lang, mapping_hash)
                    if quality_mode == 'semantic':
                        review = reviews.get((lang, input_hash), {})
                        if review.get('decision') != 'usable' or review.get('status') != 'success' or review.get('mapping_sha256') != mapping_hash:
                            raise ValueError('quality_review:' + str(review.get('decision', 'not_reviewed')))
                    # Only successful model audits enter coverage. Provider/schema
                    # failures are audited and skipped; later data/image failures
                    # still leave a visible coverage gap.
                    expected[lang].update(_keys(row))
                    images = record.get('images', record.get('image'))
                    image_path = images if isinstance(images, str) else images[0]
                    actual = Path(image_root) / key if image_root else path.parent / image_path
                    with Image.open(actual) as image:
                        image.verify()
                    image_hash = _file_hash(actual)
                    if quality_mode == 'semantic' and review.get('image_sha256') != image_hash:
                        raise ValueError('quality_review:image_changed')
                    identity = lang + '_' + _hash([key, conv])
                    workpiece = record.get('metadata', {}).get('workpiece_id')
                    row.update(id=identity, image_key=key,
                               group='workpiece:' + _hash([meta.get('source_domain'), str(workpiece)])
                                     if workpiece not in (None, '') else 'image:' + key,
                               offset=offset, source_line=line_number, language=lang, input_hash=input_hash,
                               image_sha256=image_hash, actual_image=str(actual.resolve()))
                    by_image[key].append(row)
                except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
                    issue['reason'] = str(exc); audit.append(issue)
        for key, rows in by_image.items():
            signatures = {(r['meta']['object_category'], r['meta']['defect_type'], r['meta'].get('defect_status', 'unknown')) for r in rows}
            if len(signatures) > 1:
                audit.append({'language': lang, 'image_key': key, 'reason': 'conflicting_image_labels'})
            else:
                pools[lang].append(min(rows, key=lambda r: (_hash([seed, r['id']]), r['id'])))
        content_groups = defaultdict(list)
        for row in pools[lang]:
            content_groups[row['image_sha256']].append(row)
        pools[lang] = []
        for image_sha, content_rows in content_groups.items():
            signatures = {(r['meta']['object_category'], r['meta']['defect_type'], r['meta']['defect_status'])
                          for r in content_rows}
            if len(signatures) > 1:
                audit.append({'language': lang, 'image_sha256': image_sha,
                              'reason': 'conflicting_image_content_labels',
                              'image_keys': [r['image_key'] for r in content_rows]})
            else:
                pools[lang].extend(content_rows)
        pools[lang].sort(key=lambda r: r['id'])
    if audit and any('raw_object_category' not in a and a.get('reason') not in
                     {'conflicting_image_labels', 'conflicting_image_content_labels'} for a in audit):
        blockers.append('unclassified_rejected_records: complete label coverage cannot be established')
    minima = {'object': object_min, 'defect': defect_min, 'status': status_min}
    selections, minimum_budget, reports = {}, {}, []
    for lang, target in [('zh', zh_n), ('en', en_n)]:
        rows = pools[lang]
        counts = Counter(k for r in rows for k in _keys(r))
        requirements = expected[lang] if not allow_partial_coverage else set(counts)
        try:
            base = _solve(rows, requirements, minima, seed=seed)
            minimum_budget[lang] = len(base) if base is not None else None
            if base is None or len(base) > target:
                raise _Infeasible()
            # Fill in deterministic object -> defect -> domain order, respecting workpieces.
            chosen = list(base); used = {r['group'] for r in chosen}; chosen_ids = {r['id'] for r in chosen}
            used_content = {r['image_sha256'] for r in chosen}
            while len(chosen) < target:
                remaining = [r for r in rows if r['group'] not in used
                             and r['image_sha256'] not in used_content and r['id'] not in chosen_ids]
                if not remaining: break
                freq = Counter(k for r in chosen for k in _keys(r))
                def rank(r):
                    keys = _keys(r)
                    return (freq[keys[0]], freq[keys[1]], freq[keys[3]], _hash([seed, r['id']]))
                item = min(remaining, key=rank); chosen.append(item); used.add(item['group']); chosen_ids.add(item['id'])
                used_content.add(item['image_sha256'])
            if len(chosen) != target:
                # Overlapping workpiece/content constraints can trap greedy fill.
                # Prove exact-size feasibility instead of reporting a false shortage.
                chosen = _solve(rows, requirements, minima, count=target, seed=seed)
                if chosen is None:
                    raise _Infeasible()
            selections[lang] = sorted(chosen, key=lambda r: r['id'])
        except _Infeasible:
            selections[lang] = []
            minimum_budget.setdefault(lang, None)
            blockers.append(f'{lang}: budget/capacity/workpiece constraints cannot meet coverage and target={target}')
        except ValueError as exc:
            selections[lang] = []
            minimum_budget.setdefault(lang, None)
            blockers.append(f'{lang}: solver_unproven: {exc}')
        selected_counts = Counter(k for r in selections[lang] for k in _keys(r))
        for key in sorted(expected[lang]):
            required = minima.get(key[0], 0)
            selected = selected_counts[key]
            reports.append({'language': lang, 'dimension': key[0], 'object_category': key[1], 'value': key[2],
                            'available': counts[key], 'minimum': required, 'planned': selected, 'selected': 0 if plan_only else selected,
                            'gap': max(0, required-selected),
                            'reason': 'no_eligible_samples' if not counts[key] else ('coverage_or_budget_infeasible' if selected < required else '')})
    if allow_partial_coverage:
        blockers = [b for b in blockers if not b.startswith('unclassified_rejected_records')]
    chosen = selections['zh'] + selections['en']
    planned_ids = [r['id'] for r in chosen]
    image_hashes = {r['id']: r['image_sha256'] for r in chosen}
    if saved_plan and (saved_plan.get('selected_ids') != planned_ids or saved_plan.get('image_hashes') != image_hashes):
        raise ValueError('planned selection or image content changed')
    for lang, path in paths.items():
        if _file_hash(path) != source_hashes[lang]: raise ValueError('source changed during construction')
    if _file_hash(mapping) != mapping_hash: raise ValueError('mapping changed during construction')
    prepared = []
    if not plan_only and not blockers:
        for row in chosen:
            with paths[row['language']].open('rb') as source:
                source.seek(row['offset']); original = json.loads(source.readline().decode('utf-8-sig'))
            if sample_input_hash(original, row['language'], mapping_hash) != row['input_hash']:
                raise ValueError('selected source changed')
            metadata = original.setdefault('metadata', {})
            for key in ('category','object_category','defect_type','language'):
                if key in original: metadata.setdefault('original_top_level_labels', {})[key] = original.pop(key)
            if 'id' in original: metadata['original_id'] = original['id']
            metadata.update(row['meta'])
            metadata.update(language=row['language'], category=row['meta']['object_category'],
                            source_image=row['image_key'], source_image_key=row['image_key'], source_id=_hash(row['image_key']),
                            source_line=row['source_line'], source_jsonl=str(paths[row['language']].resolve()),
                            mapping_sha256=mapping_hash, mapping_version=labels.doc['version'], selection_source='training_regression')
            original['id'] = row['id']
            original_images = original.get('images', original.get('image'))
            original_image = original_images if isinstance(original_images, str) else original_images[0]
            if image_root or not Path(original_image).is_absolute():
                metadata['original_images'] = original.get('images', original.get('image'))
                original['images'] = [row['actual_image']]
                original.pop('image', None)
            if _file_hash(row['actual_image']) != row['image_sha256']:
                raise ValueError('image changed before extraction')
            prepared.append(json.dumps(original, ensure_ascii=False)+'\n')
    for lang, path in paths.items():
        if _file_hash(path) != source_hashes[lang]: raise ValueError('source changed before output')
    if quality_review and _file_hash(quality_review) != fingerprint['quality_review_sha256']:
        raise ValueError('quality review changed during construction')
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'blocked' if blockers else ('planned' if plan_only else 'complete'), 'fingerprint': fingerprint,
              'minimum_budget': minimum_budget, 'requested': {'zh': zh_n, 'en': en_n},
              'available': {lang: len(rows) for lang, rows in pools.items()}, 'blockers': blockers,
              'coverage': reports, 'selected_ids': planned_ids, 'image_hashes': image_hashes,
              'quality_checked': quality_mode, 'selection_source': 'training_regression',
              'mapping_version': getattr(labels, 'doc', {}).get('version', 'unknown')}
    _write(out/'coverage_plan.json', report)
    _write(out/'coverage_report.json', report)
    with (out/'coverage_plan.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        fields = ['language','dimension','object_category','value','available','minimum','planned','selected','gap','reason']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(reports)
    with (out/'quality_audit.jsonl').open('x', encoding='utf-8') as stream:
        for item in audit: stream.write(json.dumps(item, ensure_ascii=False)+'\n')
    if blockers:
        _write(out/'manifest.json', report)
        raise ValueError('Coverage blocked; see coverage_plan.json and quality_audit.jsonl: ' + '; '.join(blockers))
    if not plan_only:
        temporary = out/'industrial_eval.jsonl.pending'
        with temporary.open('x', encoding='utf-8') as output:
            output.writelines(prepared)
        temporary.rename(out/'industrial_eval.jsonl')
        report['dataset_sha256'] = _file_hash(out/'industrial_eval.jsonl')
    _write(out/'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('zh','en','mapping','out-dir'): parser.add_argument('--'+key, required=True)
    for key, default in [('zh-n',500),('en-n',500),('seed',42),('object-min',1),('defect-min',1),('status-min',1)]: parser.add_argument('--'+key,type=int,default=default)
    for key in ('plan','image-root','quality-review'): parser.add_argument('--'+key)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--allow-partial-coverage',action='store_true')
    parser.add_argument('--quality-mode',choices=['semantic','structural'],default='semantic')
    args = vars(parser.parse_args())
    try: sample_quality(**args)
    except (ValueError, FileExistsError) as exc: parser.exit(2, str(exc)+'\n')


if __name__ == '__main__': main()
