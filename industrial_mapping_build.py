"""Scan bilingual industrial labels and freeze reviewed YAML mapping."""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
from eval_tool.industrial_mapping import scan, freeze


def build(en_path, zh_path, out_yaml, audit_json=None, review_jsonl=None, version='industrial_labels_v1'):
    targets = [Path(p) for p in (out_yaml, audit_json, review_jsonl) if p is not None]
    if any(p.exists() for p in targets): raise ValueError('refusing to overwrite mapping artifacts')
    with tempfile.TemporaryDirectory() as tmp:
        doc, report = scan(en_path, zh_path, tmp, version)
        shutil.copyfile(Path(tmp)/'candidate.yaml', out_yaml)
        if audit_json: shutil.copyfile(Path(tmp)/'scan_report.json', audit_json)
        if review_jsonl: shutil.copyfile(Path(tmp)/'review_queue.jsonl', review_jsonl)
    return doc, report


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    if args and args[0] in {'scan', 'freeze'}:
        subs = p.add_subparsers(dest='command', required=True)
        s = subs.add_parser('scan')
        s.add_argument('--en', required=True, type=Path); s.add_argument('--zh', required=True, type=Path)
        s.add_argument('--out-dir', required=True, type=Path); s.add_argument('--version', default='industrial_labels_v1')
        f = subs.add_parser('freeze')
        f.add_argument('--candidate', required=True, type=Path)
        f.add_argument('--reviews', '--review-jsonl', dest='reviews', required=True, type=Path)
        f.add_argument('--out-yaml', '--mapping-yaml', dest='out_yaml', required=True, type=Path)
        f.add_argument('--allow-partial', action='store_true')
        a = p.parse_args(args)
        if a.command == 'scan': scan(a.en, a.zh, a.out_dir, a.version)
        else: freeze(a.candidate, a.reviews, a.out_yaml, a.allow_partial)
    else:
        for flag in ('en', 'zh', 'mapping-yaml', 'audit-json', 'review-jsonl'):
            p.add_argument('--'+flag, required=True, type=Path)
        p.add_argument('--version', default='industrial_labels_v1')
        a = p.parse_args(args)
        build(a.en, a.zh, a.mapping_yaml, a.audit_json, a.review_jsonl, a.version)

if __name__ == '__main__': main()
