"""Build an auditable candidate bilingual industrial label mapping."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from eval_tool.industrial_labels import build_label_mapping

def build(en_path, zh_path, out_yaml, audit_json=None, review_jsonl=None, version="industrial_labels_v1"):
    read = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    audit = build_label_mapping(read(en_path), read(zh_path)).audit()
    doc = {"version": version, "status": "candidate", "objects": audit.get("objects", []),
           "defects": audit.get("defects", []), "manual_overrides": audit.get("manual_overrides", {"objects": [], "defects": []}),
           "review_required": True}
    Path(out_yaml).write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit_json: Path(audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if review_jsonl:
        queue = list(audit.get("unresolved", [])) + [{"image_key": x, "reason": "conflict"} for x in audit.get("conflict_images", [])]
        Path(review_jsonl).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in queue), encoding="utf-8")
    return doc, audit

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("en", "zh", "mapping-yaml", "audit-json", "review-jsonl"): p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--version", default="industrial_labels_v1"); a = p.parse_args(argv)
    if any(x.exists() for x in (a.mapping_yaml, a.audit_json, a.review_jsonl)): raise SystemExit("refusing to overwrite mapping artifacts")
    build(a.en, a.zh, a.mapping_yaml, a.audit_json, a.review_jsonl, a.version)
if __name__ == "__main__": main()
