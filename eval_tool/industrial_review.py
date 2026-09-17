"""Resumable private-model audits, separated from evaluation and frozen mapping."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from itertools import zip_longest
from pathlib import Path
import yaml
from .private_review_client import PrivateReviewClient
from .industrial_mapping import FrozenMapping, file_sha256, iter_jsonl

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_input_hash(record, lang, mapping_sha256):
    return _hash({"record": record, "language": lang, "mapping_sha256": mapping_sha256})


def _json(content):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _existing(out, resume):
    if Path(out).exists() and not resume:
        raise ValueError("output exists; use --resume")
    return {r.get("cache_key") for _, r in iter_jsonl(out) if r.get("status") == "success"} if Path(out).exists() else set()


def _append(out, row):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _progress(stage, completed, total, row=None):
    status = ""
    if row is not None:
        status = f" {row.get('status', '')}/{row.get('decision', '')} provider={row.get('provider', '')}"
    print(f"[industrial-review:{stage}] {completed}/{total}{status}", file=sys.stderr, flush=True)


def _mapping_decision(value, entry):
    if not {"entry_id", "decision", "canonical", "defect_status", "reason"}.issubset(value):
        raise ValueError("missing schema fields")
    if value.get("entry_id") != entry["entry_id"]:
        raise ValueError("entry id mismatch")
    if value.get("decision") not in {"accepted", "rejected", "uncertain"}:
        raise ValueError("invalid decision")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("missing reason")
    if value.get("defect_status") not in {None, "normal", "abnormal", "unknown"}:
        raise ValueError("invalid defect status")
    if value["decision"] == "accepted":
        canonical = value.get("canonical")
        if not isinstance(canonical, str) or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", canonical):
            raise ValueError("invalid canonical")
        if entry.get("canonical_candidates") and canonical not in entry["canonical_candidates"]:
            raise ValueError("canonical outside candidate list")
        if entry["kind"] == "defect" and value.get("defect_status") not in {"normal", "abnormal"}:
            raise ValueError("defect status required")
    return {k: value.get(k) for k in ("entry_id", "decision", "canonical", "defect_status", "reason")}


def mapping_review(candidate, queue, out, *, max_items=None, resume=False, provider=None, client=None):
    client = client or PrivateReviewClient(provider)
    digest = file_sha256(candidate)
    doc = yaml.safe_load(Path(candidate).read_text(encoding="utf-8"))
    if doc.get("status") != "candidate":
        raise ValueError("mapping must be candidate")
    entries = {e["id"]: e for e in doc["entries"]}
    prompt = (PROMPTS / "industrial_mapping_review_v1.txt").read_text(encoding="utf-8")
    done = _existing(out, resume); seen = set(); count = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).touch(exist_ok=True)
    queue_rows = [entry for _, entry in iter_jsonl(queue)]
    total = len(queue_rows)
    completed = 0
    for entry in queue_rows:
        eid = entry.get("entry_id")
        if entry.get("candidate_sha256") != digest or eid not in entries:
            raise ValueError("queue does not match candidate")
        expected = dict(entries[eid], entry_id=eid, candidate_sha256=digest)
        if entry != expected:
            raise ValueError("queue entry differs from candidate")
        if eid in seen: continue
        seen.add(eid)
        key = _hash([digest, prompt, entry, client.cache_settings])
        if key in done:
            completed += 1; _progress("mapping", completed, total, {"status": "cached", "provider": "cache"}); continue
        if max_items is not None and count >= max_items: break
        row = {"entry_id": eid, "candidate_sha256": digest, "cache_key": key,
               "decision": "uncertain", "canonical": None, "defect_status": "unknown",
               "provider": "unavailable", "model": "unavailable", "status": "error"}
        try:
            result = client.review(prompt + "\nINPUT:\n" + json.dumps(entry, ensure_ascii=False))
            row.update(provider=result["provider"], model=result["model"], status="success")
            try: row.update(_mapping_decision(_json(result["content"]), entry))
            except (ValueError, TypeError, KeyError):
                row.update(status="error", reason="Invalid model response schema; requires further review")
        except Exception as exc:
            row["reason"] = "Review request failed: " + type(exc).__name__
        _append(out, row); count += 1; completed += 1; _progress("mapping", completed, total, row)
    remaining = max(0, total - completed)
    return {"reviewed": count, "completed": completed, "total": total,
            "remaining": remaining, "candidate_sha256": digest}


def sample_review(zh, en, mapping, out, *, image_root=None, max_items=None, resume=False, provider=None, client=None):
    client = client or PrivateReviewClient(provider)
    frozen = FrozenMapping.load(mapping); frozen.validate_sources(en=en, zh=zh)
    digest = file_sha256(mapping)
    prompt = (PROMPTS / "industrial_sample_review_v1.txt").read_text(encoding="utf-8")
    done = _existing(out, resume); seen = set(); count = 0; unreviewed = 0; skipped_excluded = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).touch(exist_ok=True)
    def records(lang, path):
        for line, record in iter_jsonl(path): yield lang, path, line, record
    totals = sum(1 for _ in iter_jsonl(zh)) + sum(1 for _ in iter_jsonl(en))
    completed = 0
    for pair in zip_longest(records("zh", zh), records("en", en)):
        for item in pair:
            if item is None: continue
            language, path, line, record = item
            ih = sample_input_hash(record, language, digest)
            labels = None
            try:
                labels = frozen.resolve(record, language)
                image_key = labels["image_key"]
                if labels.get("status") != "resolved":
                    skipped_excluded += 1
                    completed += 1
                    continue
                raw_image = record.get("images", record.get("image"))
                raw_image = raw_image if isinstance(raw_image, str) else raw_image[0]
                current_image = Path(image_root) / image_key if image_root else Path(path).parent / raw_image
                image_sha = file_sha256(current_image) if current_image.is_file() else None
            except (ValueError, TypeError, IndexError, OSError): image_sha = None
            key = _hash([ih, prompt, client.cache_settings, str(image_root), image_sha])
            if ih in seen: continue
            seen.add(ih)
            if key in done:
                completed += 1; _progress("samples", completed, totals, {"status": "cached", "provider": "cache"}); continue
            if max_items is not None and count >= max_items:
                unreviewed += 1; continue
            row = {"input_hash": ih, "cache_key": key, "language": language, "source_id": str(path),
                   "line": line, "mapping_sha256": digest, "image_sha256": image_sha, "decision": "insufficient", "status": "error",
                   "provider": "unavailable", "model": "unavailable"}
            try:
                labels = labels or frozen.resolve(record, language)
                if labels["status"] != "resolved": raise ValueError("unresolved mapping")
                raw = record.get("images", record.get("image"))
                raw = raw if isinstance(raw, str) else raw[0]
                image = Path(image_root) / labels["image_key"] if image_root else Path(path).parent / raw
                if not image.is_file(): raise FileNotFoundError("missing image")
                turns = record.get("conversations", [])
                payload = {"language": language, "labels": labels, "conversations": turns}
                result = client.review(prompt + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False), [image])
                row.update(provider=result["provider"], model=result["model"], status="success")
                value = _json(result["content"])
                if value.get("decision") not in {"usable", "conflict", "insufficient"} or not isinstance(value.get("reason"), str) or not value["reason"].strip():
                    raise ValueError("invalid response schema")
                row.update(decision=value["decision"], reason=value["reason"])
            except Exception as exc:
                detail = str(exc).strip()
                reason = "Sample audit failed: " + type(exc).__name__
                if detail:
                    reason += ": " + detail[:200]
                row.update(status="error", decision="insufficient", reason=reason)
            _append(out, row); count += 1; completed += 1; _progress("samples", completed, totals, row)
    return {"reviewed": count, "completed": completed, "total": totals,
            "not_reviewed": unreviewed, "skipped_excluded": skipped_excluded,
            "mapping_sha256": digest}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("mapping", "samples"):
        c = sub.add_parser(command)
        for flag in (("candidate", "queue") if command == "mapping" else ("zh", "en", "mapping")):
            c.add_argument("--" + flag, required=True)
        c.add_argument("--out", required=True); c.add_argument("--provider")
        c.add_argument("--max-items", type=int); c.add_argument("--resume", action="store_true")
        if command == "samples": c.add_argument("--image-root")
    args = vars(p.parse_args(argv)); command = args.pop("command")
    if args["max_items"] is not None and args["max_items"] < 0: p.error("max-items must be nonnegative")
    print(json.dumps((mapping_review if command == "mapping" else sample_review)(**args)))


if __name__ == "__main__": main()
