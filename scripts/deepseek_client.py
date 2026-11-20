"""Provider-agnostic LLM client for the FinPlan experiments.

Supported providers:

* ``deepseek``: Anthropic-compatible messages endpoint
  (``ANTHROPIC_BASE_URL``, ``ANTHROPIC_MODEL``, ``ANTHROPIC_AUTH_TOKEN``).
* ``glm``: Zhipu / Z.ai OpenAI-compatible chat endpoint
  (``GLM_API_KEY``, default base URL ``https://open.bigmodel.cn/api/paas/v4``,
  default model ``glm-5.3-flash``). GLM 5.x always reasons; we use
  ``reasoning_effort="low"`` so the JSON answer fits in ``max_tokens``.
* ``openai``: any OpenAI-compatible endpoint
  (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``OPENAI_MODEL``).

The API key is read from the environment and never written to disk or printed.
Every response is cached by a SHA-256 digest of the request payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "paper" / "results" / "llm_cache"


class LLMError(RuntimeError):
    pass


# Backwards-compatible name used by the existing runners.
DeepSeekError = LLMError


class LLMClient:
    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        cache_dir: Path = CACHE_DIR,
        timeout: int = 90,
        max_retries: int = 4,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or "deepseek").lower()
        if self.provider not in {"deepseek", "glm", "openai"}:
            raise ValueError(f"unknown provider: {self.provider}")

        if self.provider == "deepseek":
            self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or "").rstrip("/")
            self.model = model or os.environ.get("ANTHROPIC_MODEL") or "deepseek-flash[1m]"
            self.api_key_env = api_key_env or "ANTHROPIC_AUTH_TOKEN"
        elif self.provider == "glm":
            self.base_url = (
                base_url
                or os.environ.get("GLM_BASE_URL")
                or "https://open.bigmodel.cn/api/paas/v4"
            ).rstrip("/")
            self.model = model or os.environ.get("GLM_MODEL") or "glm-5.3-flash"
            self.api_key_env = api_key_env or "GLM_API_KEY"
        else:
            self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
            self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
            self.api_key_env = api_key_env or "OPENAI_API_KEY"

        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise LLMError(f"missing API key environment variable: {self.api_key_env}")
        if not self.base_url:
            raise LLMError(f"missing base URL for provider {self.provider}")
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort or ("low" if self.provider == "glm" else None)
        # Extra request-body fields for OpenAI-compatible servers.  Used by
        # local Qwen3/vLLM to disable the default thinking trace:
        # {"chat_template_kwargs": {"enable_thinking": false}}.
        self.extra_body = dict(extra_body or {})
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        # The local environment has a broken proxy setting; connect directly.
        self.session.trust_env = False
        self._lock = threading.Lock()

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.provider == "deepseek":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            url = f"{self.base_url}/v1/messages"
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/chat/completions"
        return self.session.post(url, headers=headers, json=payload, timeout=self.timeout)

    def _payload(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        seed: int | None,
        thinking: bool,
        thinking_budget: int,
    ) -> dict[str, Any]:
        if self.provider == "deepseek":
            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                payload["system"] = system
            if seed is not None:
                payload["seed"] = seed
            if thinking:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max(1, thinking_budget),
                }
            else:
                payload["thinking"] = {"type": "disabled"}
            return payload

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if self.provider == "glm":
            # GLM 5.x does not allow disabling thinking; low effort keeps the
            # reasoning budget small enough for a JSON answer.
            payload["reasoning_effort"] = self.reasoning_effort or "low"
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.0,
        seed: int | None = None,
        thinking: bool = False,
        thinking_budget: int = 0,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        payload = self._payload(
            prompt, system, max_tokens, temperature, seed, thinking, thinking_budget
        )
        digest = self._digest(
            {"provider": self.provider, "base_url": self.base_url, "payload": payload}
        )
        cache_path = self._cache_path(digest)
        if use_cache and cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                return json.load(handle)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                started = time.time()
                response = self._request(payload)
                latency = time.time() - started
                if response.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code >= 400:
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
                body = response.json()
                record = {
                    "request_digest": digest,
                    "provider": self.provider,
                    "model": self.model,
                    "base_url": self.base_url,
                    "payload": {k: v for k, v in payload.items() if k not in {"messages", "system"}},
                    "response": body,
                    "latency_seconds": latency,
                    "created_at": time.time(),
                }
                with self._lock:
                    cache_path.write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                return record
            except Exception as error:  # noqa: BLE001
                last_error = error
                time.sleep(min(30.0, 2.0 ** attempt))
        raise LLMError(f"request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def text_of(record: dict[str, Any]) -> str:
        body = record.get("response", {})
        if "content" in body:
            chunks = []
            for block in body.get("content", []):
                if block.get("type") == "text":
                    chunks.append(block.get("text", ""))
            return "\n".join(chunks).strip()
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()

    @staticmethod
    def usage_of(record: dict[str, Any]) -> dict[str, int]:
        usage = record.get("response", {}).get("usage", {}) or {}
        return {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(
                usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            ),
        }


# Backwards-compatible alias for existing runners.
DeepSeekClient = LLMClient
