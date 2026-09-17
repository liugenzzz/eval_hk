#!/usr/bin/env python3
"""Probe one real industrial mapping review request and validate its response."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_tool import _private_model_router as router  # noqa: E402
from eval_tool.industrial_mapping import file_sha256, iter_jsonl  # noqa: E402
from eval_tool.industrial_review import _json, _mapping_decision  # noqa: E402
from eval_tool.private_review_client import PrivateReviewClient  # noqa: E402


def _redact(value: Any, config: dict[str, Any] | None = None) -> str:
    text = str(value)
    secrets = {"sk-local"}
    for provider in (config or {}).get("vlm_pool", {}).get("providers", []):
        env_name = provider.get("api_key_env")
        if env_name and os.environ.get(env_name):
            secrets.add(os.environ[env_name])
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)Bearer\s+[^\s,;\"']+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)(authorization|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
    return text[:2000]


def _diagnostic_http_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        response_body = exc.read(2048).decode("utf-8", errors="replace")
        raise router.RouterError(f"HTTP {exc.code} from provider; response body: {response_body}") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise router.RouterError(f"Request failed for {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:2048].decode("utf-8", errors="replace")
        raise router.RouterError(f"Non-JSON response from provider; response body: {preview}") from exc


@contextmanager
def _diagnostic_transport():
    original = router.http_json
    router.http_json = _diagnostic_http_json
    try:
        yield
    finally:
        router.http_json = original


def _run_with_diagnostics(args: Namespace, config: dict[str, Any]) -> dict[str, Any]:
    with _diagnostic_transport():
        return router.run_chat(args, config)


def _load_entry(build_dir: Path, index: int) -> tuple[dict[str, Any], str]:
    candidate = build_dir / "mapping_scan" / "candidate.yaml"
    queue = build_dir / "mapping_scan" / "review_queue.jsonl"
    if not candidate.is_file():
        raise ValueError(f"candidate not found: {candidate}")
    if not queue.is_file():
        raise ValueError(f"review queue not found: {queue}")

    digest = file_sha256(candidate)
    doc = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("status") != "candidate":
        raise ValueError("mapping must be a candidate")
    entries = {entry["id"]: entry for entry in doc.get("entries", [])}
    rows = [row for _, row in iter_jsonl(queue)]
    if not rows:
        raise ValueError("review queue is empty")
    if index < 0 or index >= len(rows):
        raise ValueError(f"entry index {index} outside queue range 0..{len(rows) - 1}")

    entry = rows[index]
    entry_id = entry.get("entry_id")
    if entry.get("candidate_sha256") != digest or entry_id not in entries:
        raise ValueError("queue does not match candidate")
    expected = dict(entries[entry_id], entry_id=entry_id, candidate_sha256=digest)
    if entry != expected:
        raise ValueError("queue entry differs from candidate")
    return entry, digest


def probe(
    build_dir: str | Path,
    provider: str,
    entry_index: int = 0,
    *,
    runner: Callable[[Namespace, dict[str, Any]], dict[str, Any]] | None = None,
    client_factory: Callable[..., PrivateReviewClient] = PrivateReviewClient,
) -> dict[str, Any]:
    client = client_factory(provider=provider)
    config = client.config
    try:
        entry, digest = _load_entry(Path(build_dir), entry_index)
        prompt_path = ROOT / "prompts" / "industrial_mapping_review_v1.txt"
        prompt = prompt_path.read_text(encoding="utf-8")
        args = Namespace(
            prompt=prompt + "\nINPUT:\n" + json.dumps(entry, ensure_ascii=False),
            prompt_file=None,
            image=[],
            task_type="industrial_review",
            provider=provider,
            system=None,
            dry_run=False,
            raw=False,
        )
        result = (runner or _run_with_diagnostics)(args, config)
        content = result.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("private model returned empty or non-text content")
        parsed = _json(content)
        decision = _mapping_decision(parsed, entry)
        return {
            "ok": True,
            "provider": result.get("provider"),
            "model": result.get("model"),
            "entry_index": entry_index,
            "entry_id": entry["entry_id"],
            "candidate_sha256": digest,
            "raw_content": _redact(content, config),
            "parsed": parsed,
            "schema_valid": True,
            "decision": decision,
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "entry_index": entry_index,
            "error_type": type(exc).__name__,
            "error": _redact(exc, config),
        }


def main(argv: list[str] | None = None, **dependencies: Any) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="data/industrial_quality_v2")
    parser.add_argument("--provider", default="qwen3.6-27b_v2")
    parser.add_argument("--entry-index", type=int, default=0)
    args = parser.parse_args(argv)
    result = probe(args.build_dir, args.provider, args.entry_index, **dependencies)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
