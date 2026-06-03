"""
llm_client.py
-------------
LLM client for the Gazebo mission executor.

Uses the same model registry, API backends, and retry policy as
experiment_utils.py so that executor inference is identical to offline
eval inference.  The only difference is that query_llm() returns a plain
string rather than a (text, latency) tuple - the executor does not record
per-call latency.

Models are served via OpenRouter (Gemini, Qwen3, DeepSeek) or the OpenAI
API (o4-mini).  Set the active model via the LLM_MODEL_KEY environment
variable or by editing the constant below.

Available model keys (identical to experiment_utils.py ALL_MODELS):
  "gemini-2.5-flash-thinking"   google/gemini-2.5-flash, reasoning=medium
  "gemini-2.5-flash-base"       google/gemini-2.5-flash, reasoning=none
  "qwen-235b-thinking"          qwen/qwen3-235b-a22b-thinking-2507, reasoning=medium
  "qwen-235b-instruct"          qwen/qwen3-235b-a22b-2507, no reasoning
  "deepseek-r1"                 deepseek/deepseek-r1, reasoning=medium
  "o4-mini"                     o4-mini via OpenAI, reasoning_effort=medium

Required environment variables:
  OPENROUTER_API_KEY  - all models except o4-mini
  OPENAI_API_KEY      - o4-mini only

Retry policy (mirrors experiment_utils.py):
  All errors are retried indefinitely with exponential backoff.
  Rate-limit / transient errors: wait RETRY_WAIT_S, double up to MAX_SINGLE_WAIT_S.
  Other errors: wait RETRY_WAIT_S, retry without increasing the delay.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MODEL CONFIG  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "gemini-2.5-flash-thinking": {
        "backend":   "openrouter",
        "model_id":  "google/gemini-2.5-flash",
        "reasoning": "medium",
    },
    "gemini-2.5-flash-base": {
        "backend":   "openrouter",
        "model_id":  "google/gemini-2.5-flash",
        "reasoning": "none",
    },
    "qwen-235b-thinking": {
        "backend":   "openrouter",
        "model_id":  "qwen/qwen3-235b-a22b-thinking-2507",
        "reasoning": "medium",
    },
    "qwen-235b-instruct": {
        "backend":   "openrouter",
        "model_id":  "qwen/qwen3-235b-a22b-2507",
        "reasoning": None,
    },
    "deepseek-r1": {
        "backend":   "openrouter",
        "model_id":  "deepseek/deepseek-r1",
        "reasoning": "medium",
    },
    "o4-mini": {
        "backend":   "openai",
        "model_id":  "o4-mini",
        "reasoning": None,   # handled via reasoning_effort="medium" in API call
    },
}

# ---------------------------------------------------------------------------
# ACTIVE MODEL  - override with the LLM_MODEL_KEY environment variable
# ---------------------------------------------------------------------------

LLM_MODEL_KEY: str = os.environ.get("LLM_MODEL_KEY", "gemini-2.5-flash-thinking")

# ---------------------------------------------------------------------------
# API KEYS
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENAI_API_KEY:     str = os.environ.get("OPENAI_API_KEY",     "")

# ---------------------------------------------------------------------------
# RETRY CONFIG  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

RETRY_WAIT_S:      int = 15
MAX_SINGLE_WAIT_S: int = 120

# ---------------------------------------------------------------------------
# CLIENT CACHE  (lazy singletons)
# ---------------------------------------------------------------------------

_openrouter_client = None
_openai_client     = None


def _get_openrouter():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI as _OpenAI
        _openrouter_client = _OpenAI(
            api_key  = OPENROUTER_API_KEY,
            base_url = "https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI as _OpenAI
        _openai_client = _OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# RATE-LIMIT DETECTION  (verbatim from experiment_utils.py)
# ---------------------------------------------------------------------------

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg for kw in (
            "429", "quota", "rate", "resource_exhausted", "too many requests",
            "503", "service unavailable", "overloaded", "high traffic",
            "try again", "temporarily unavailable", "server error",
            "internal error", "connection", "timeout", "timed out",
        )
    )


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def model_label() -> str:
    """Human-readable label for log messages."""
    cfg = MODELS.get(LLM_MODEL_KEY, {})
    return f"{LLM_MODEL_KEY} ({cfg.get('model_id', '?')})"


def query_llm(
    prompt:     str,
    system:     str,
    model_key:  str | None = None,
    max_tokens: int = 4096,
) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    model_key defaults to LLM_MODEL_KEY.  Pass it explicitly to make a
    one-off call with a different model without changing the global setting.

    Retry policy mirrors experiment_utils.py: all errors are retried
    indefinitely; rate-limit / transient errors use exponential backoff.
    """
    key = model_key or LLM_MODEL_KEY
    cfg = MODELS[key]
    backend = cfg["backend"]

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt if prompt else "Respond now."},
    ]

    # ------------------------------------------------------------------
    # OpenRouter - Gemini, Qwen3, DeepSeek
    # ------------------------------------------------------------------
    if backend == "openrouter":
        wait_s  = RETRY_WAIT_S
        attempt = 0

        extra_body: dict = {}
        if cfg.get("reasoning") is not None:
            extra_body["reasoning"] = {"effort": cfg["reasoning"]}

        while True:
            attempt += 1
            try:
                kwargs: dict[str, Any] = dict(
                    model      = cfg["model_id"],
                    messages   = messages,
                    max_tokens = max_tokens,
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = _get_openrouter().chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except Exception as exc:
                exc_kind = "Rate-limit" if _is_rate_limit_error(exc) else "Unexpected error"
                log.warning(
                    f"[LLM/{key}] {exc_kind} on attempt {attempt}: "
                    f"{exc!r:.120}. Waiting {wait_s}s before retry..."
                )
                time.sleep(wait_s)
                if _is_rate_limit_error(exc):
                    wait_s = min(wait_s * 2, MAX_SINGLE_WAIT_S)

    # ------------------------------------------------------------------
    # OpenAI - o4-mini
    # reasoning_effort="medium"; temperature intentionally omitted.
    # ------------------------------------------------------------------
    if backend == "openai":
        wait_s  = RETRY_WAIT_S
        attempt = 0
        while True:
            attempt += 1
            try:
                response = _get_openai().chat.completions.create(
                    model                 = cfg["model_id"],
                    messages              = messages,
                    max_completion_tokens = max_tokens,
                    reasoning_effort      = "medium",
                )
                return response.choices[0].message.content

            except Exception as exc:
                exc_kind = "Rate-limit" if _is_rate_limit_error(exc) else "Unexpected error"
                log.warning(
                    f"[LLM/{key}] {exc_kind} on attempt {attempt}: "
                    f"{exc!r:.120}. Waiting {wait_s}s before retry..."
                )
                time.sleep(wait_s)
                if _is_rate_limit_error(exc):
                    wait_s = min(wait_s * 2, MAX_SINGLE_WAIT_S)

    raise ValueError(f"Unknown backend: {backend!r} for model_key={key!r}")
