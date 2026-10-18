"""Read-only setup and index diagnostics."""

from __future__ import annotations

import json
import os
import sqlite3
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

from .ask import (
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_GEMINI_BASE_URL,
    _load_dotenv,
)
from .config import DEFAULT_MODEL, DEFAULT_PROVIDER, config_path, load_config
from .tracker import TrackerError, find_repo_root


def run_diagnostics(path: Path) -> list[tuple[bool, str]]:
    root = find_repo_root(path.resolve())
    dotenv = _load_dotenv(root / ".env")
    saved = load_config()
    provider = (
        os.environ.get("CODEMATE_PROVIDER") or saved.get("provider")
        or dotenv.get("CODEMATE_PROVIDER") or DEFAULT_PROVIDER
    ).lower()
    model = os.environ.get("CODEMATE_MODEL") or saved.get("model") or dotenv.get("CODEMATE_MODEL")
    if not model and provider == "gemini":
        model = DEFAULT_MODEL
    base_url = (
        os.environ.get("CODEMATE_BASE_URL") or saved.get("base_url")
        or dotenv.get("CODEMATE_BASE_URL")
        or (DEFAULT_GEMINI_BASE_URL if provider == "gemini" else DEFAULT_CHAT_BASE_URL)
    )
    api_key = os.environ.get("GEMINI_API_KEY") or saved.get("api_key") or dotenv.get("GEMINI_API_KEY")

    results: list[tuple[bool, str]] = [(True, f"Repository: {root}")]
    results.append((config_path().is_file(), f"User configuration: {config_path()}"))
    results.append((bool(model), f"AI provider/model: {provider} / {model or 'not configured'}"))
    results.append((bool(api_key), f"API key: {'configured' if api_key else 'missing'}"))

    database = root / ".index" / "vectors.sqlite3"
    if database.is_file():
        try:
            with sqlite3.connect(database) as connection:
                chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                settings_row = connection.execute(
                    "SELECT value FROM metadata WHERE key='index_settings'"
                ).fetchone()
            settings = json.loads(settings_row[0]) if settings_row else {}
            results.append((chunks > 0, f"Vector index: {chunks} chunks ({settings.get('model', 'unknown model')})"))
        except (sqlite3.Error, json.JSONDecodeError, TypeError):
            results.append((False, "Vector index: unreadable or invalid"))
    else:
        results.append((False, "Vector index: missing; run `codemate index`"))

    if provider not in {"gemini", "openai-compatible"}:
        results.append((False, f"API connection: unsupported provider {provider!r}"))
    elif not api_key or not model:
        results.append((False, "API connection: skipped until provider, model, and API key are configured"))
    else:
        endpoint = (
            f"{base_url.rstrip('/')}/models/{urllib.parse.quote(model, safe='-._')}"
            if provider == "gemini"
            else f"{base_url.rstrip('/')}/models"
        )
        headers = {"Accept": "application/json"}
        headers["x-goog-api-key" if provider == "gemini" else "Authorization"] = (
            api_key if provider == "gemini" else f"Bearer {api_key}"
        )
        try:
            request = urllib.request.Request(endpoint, headers=headers, method="GET")
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=certifi.where())
            with urllib.request.urlopen(request, timeout=12, context=context) as response:
                ok = 200 <= response.status < 300
            results.append((ok, "API connection: reachable" if ok else "API connection: request failed"))
        except (urllib.error.URLError, TimeoutError, OSError, urllib.error.HTTPError):
            results.append((False, "API connection: failed; check network, provider, model, and API key"))
    return results
