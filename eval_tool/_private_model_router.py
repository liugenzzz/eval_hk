#!/usr/bin/env python3
"""Route private OpenAI-compatible VLM requests without dependencies."""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).with_name("private_model_providers.json")
DEFAULT_STATE = Path.home() / ".codex" / "private-model-router-state.json"


class RouterError(RuntimeError):
    pass


def load_config() -> dict[str, Any]:
    path = Path(os.environ.get("PRIVATE_MODEL_ROUTER_CONFIG", DEFAULT_CONFIG)).expanduser()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RouterError(f"Cannot load router config {path}: {exc}") from exc
    providers = config.get("vlm_pool", {}).get("providers")
    if not isinstance(providers, list) or not providers:
        raise RouterError("Router config requires a nonempty vlm_pool.providers list")
    names = [str(provider.get("name", "")) for provider in providers if isinstance(provider, dict)]
    if len(names) != len(providers) or any(not name for name in names):
        raise RouterError("Every provider requires a nonempty name")
    if len(names) != len(set(name.lower() for name in names)):
        raise RouterError("Router config contains a duplicate provider name")
    return config


@contextlib.contextmanager
def locked_state() -> Iterator[dict[str, Any]]:
    path = Path(os.environ.get("PRIVATE_MODEL_ROUTER_STATE", DEFAULT_STATE)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write("{}")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            try:
                state = json.load(handle)
            except json.JSONDecodeError:
                state = {}
            yield state
            handle.seek(0)
            handle.truncate()
            json.dump(state, handle, ensure_ascii=False)
            handle.flush()
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_api_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                if value:
                    return str(value)
        except (FileNotFoundError, OSError):
            pass
    return None


def provider_available(provider: dict[str, Any]) -> tuple[bool, str | None]:
    key_env = provider.get("api_key_env")
    if key_env and not read_api_key(key_env):
        return False, f"missing environment variable {key_env}"
    return True, None


def eligible_providers(
    config: dict[str, Any], task_type: str, capability: str, requested: str | None = None
) -> list[dict[str, Any]]:
    providers = config["vlm_pool"]["providers"]
    names = [str(provider.get("name", "")) for provider in providers]
    if any(not name for name in names) or len(names) != len(set(name.lower() for name in names)):
        raise RouterError("Provider names must be nonempty and unique")
    requested_lower = requested.lower() if requested else None
    exact_name = any(name.lower() == requested_lower for name in names) if requested else False
    if requested and not exact_name:
        model_matches = [provider for provider in providers
                         if str(provider.get("model", "")).lower() == requested_lower]
        if len(model_matches) > 1:
            choices = ", ".join(str(provider["name"]) for provider in model_matches)
            raise RouterError(f"Model name {requested!r} is ambiguous; choose provider: {choices}")
    matches: list[dict[str, Any]] = []
    for index, provider in enumerate(providers):
        if requested:
            target = str(provider["name"] if exact_name else provider["model"]).lower()
            if target != requested_lower:
                continue
        if capability not in provider.get("capabilities", []):
            continue
        task_types = provider.get("task_types", [])
        if task_types and task_type not in task_types:
            continue
        available, _ = provider_available(provider)
        if not available:
            continue
        item = dict(provider)
        item["_index"] = index
        matches.append(item)
    if requested and not matches:
        raise RouterError(f"Requested provider/model is unavailable or ineligible: {requested}")
    if not matches:
        raise RouterError(
            f"No available provider supports task_type={task_type!r}, capability={capability!r}"
        )
    return matches


def reserve_provider(
    candidates: list[dict[str, Any]], excluded: set[str] | None = None
) -> dict[str, Any]:
    excluded = excluded or set()
    now = time.time()
    with locked_state() as state:
        provider_state = state.setdefault("providers", {})
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for provider in candidates:
            name = provider["name"]
            if name in excluded:
                continue
            current = provider_state.setdefault(name, {"active": 0, "cooldown_until": 0})
            if float(current.get("cooldown_until", 0)) > now:
                continue
            active = max(0, int(current.get("active", 0)))
            stale_after = max(60, int(provider.get("timeout", 240)) * 2)
            if active and now - float(current.get("active_updated_at", 0)) > stale_after:
                active = 0
                current["active"] = 0
            maximum = max(1, int(provider.get("max_concurrency", 1)))
            if active >= maximum:
                continue
            weight = max(0.001, float(provider.get("weight", 1)))
            score = (active + 1) / (maximum * weight)
            ranked.append((score, int(provider["_index"]), provider))
        if not ranked:
            raise RouterError("All eligible providers are busy, cooling down, or excluded")
        _, _, selected = min(ranked, key=lambda row: (row[0], row[1]))
        current = provider_state[selected["name"]]
        current["active"] = max(0, int(current.get("active", 0))) + 1
        current["active_updated_at"] = now
        return selected


def release_provider(name: str, failed: bool, cooldown_seconds: int) -> None:
    with locked_state() as state:
        current = state.setdefault("providers", {}).setdefault(
            name, {"active": 0, "cooldown_until": 0}
        )
        current["active"] = max(0, int(current.get("active", 0)) - 1)
        current["active_updated_at"] = time.time()
        if failed:
            current["cooldown_until"] = time.time() + cooldown_seconds
        else:
            current["cooldown_until"] = 0


def http_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> Any:
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
        raise RouterError(f"HTTP {exc.code} from provider") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RouterError(f"Request failed for {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterError(f"Non-JSON response from {url}") from exc


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def chat_payload(
    provider: dict[str, Any], prompt: str, system: str | None, images: list[Path]
) -> dict[str, Any]:
    user_content: Any = prompt
    if images:
        user_content = [{"type": "text", "text": prompt}]
        user_content.extend(image_part(path) for path in images)
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    payload: dict[str, Any] = {
        "model": provider["model"],
        "messages": messages,
        "stream": bool(provider.get("stream", False)),
        "temperature": provider.get("temperature", 0.6),
        "max_tokens": provider.get("max_tokens", 8192),
    }
    if "chat_template_kwargs" in provider:
        payload["chat_template_kwargs"] = provider["chat_template_kwargs"]
    return payload


def extract_content(response: Any) -> Any:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RouterError("VLM response has no assistant content") from exc
    if content in (None, "", []):
        raise RouterError("VLM response contains empty assistant content")
    return content


def run_chat(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    if not prompt:
        raise RouterError("Provide --prompt or --prompt-file")
    images = [Path(path).resolve() for path in args.image]
    for path in images:
        if not path.is_file():
            raise RouterError(f"Image does not exist: {path}")
    capability = "image" if images else "text"
    candidates = eligible_providers(config, args.task_type, capability, args.provider)
    pool = config["vlm_pool"]
    fallback = pool.get("fallback", {})
    attempts = 1 if args.provider or not fallback.get("enabled", True) else int(
        fallback.get("max_attempts", 2)
    )
    cooldown = int(fallback.get("cooldown_seconds", 300))
    tried: set[str] = set()
    errors: list[str] = []
    for _ in range(max(1, attempts)):
        provider = reserve_provider(candidates, tried)
        tried.add(provider["name"])
        payload = chat_payload(provider, prompt, args.system, images)
        if args.dry_run:
            release_provider(provider["name"], False, cooldown)
            return {
                "dry_run": True,
                "provider": provider["name"],
                "model": provider["model"],
                "task_type": args.task_type,
                "capability": capability,
                "image_count": len(images),
            }
        key_env = provider.get("api_key_env")
        headers = {}
        if key_env:
            api_key = read_api_key(key_env)
            if not api_key:
                raise RouterError(f"Missing environment variable {key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        failed = True
        try:
            response = http_json(provider["url"], payload, headers, int(provider["timeout"]))
            content = extract_content(response)
            failed = False
            result = {
                "provider": provider["name"],
                "model": provider["model"],
                "content": content,
            }
            if args.raw:
                result["response"] = response
            return result
        except RouterError as exc:
            errors.append(f"{provider['name']}: {exc}")
        finally:
            release_provider(provider["name"], failed, cooldown)
    raise RouterError("All attempted providers failed: " + " | ".join(errors))


def list_models(config: dict[str, Any]) -> dict[str, Any]:
    providers = []
    for provider in config["vlm_pool"]["providers"]:
        available, reason = provider_available(provider)
        providers.append(
            {
                "name": provider["name"],
                "model": provider["model"],
                "available": available,
                "reason": reason,
                "capabilities": provider.get("capabilities", []),
                "task_types": provider.get("task_types", []),
            }
        )
    return {"vlm_providers": providers}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List configured endpoints without revealing keys")

    choose = subparsers.add_parser("choose", help="Select a provider without making a request")
    choose.add_argument("--task-type", default="chat")
    choose.add_argument("--capability", choices=["text", "image"], default="text")
    choose.add_argument("--provider")

    chat = subparsers.add_parser("chat", help="Call an OpenAI-compatible VLM")
    chat.add_argument("--prompt")
    chat.add_argument("--prompt-file")
    chat.add_argument("--system")
    chat.add_argument("--image", action="append", default=[])
    chat.add_argument("--task-type", default="chat")
    chat.add_argument("--provider")
    chat.add_argument("--dry-run", action="store_true")
    chat.add_argument("--raw", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config()
        if args.command == "list":
            result = list_models(config)
        elif args.command == "choose":
            candidates = eligible_providers(config, args.task_type, args.capability, args.provider)
            provider = reserve_provider(candidates)
            release_provider(provider["name"], False, 0)
            result = {"provider": provider["name"], "model": provider["model"]}
        elif args.command == "chat":
            result = run_chat(args, config)
        else:
            raise RouterError(f"Unknown command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (RouterError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
