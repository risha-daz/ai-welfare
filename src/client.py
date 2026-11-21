"""Thin OpenAI-compatible chat client that returns the full raw exchange."""
import time
from datetime import datetime, timezone

import openai
from openai import OpenAI

from .config import resolve_api_key


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_client(model_cfg, run_cfg):
    return OpenAI(
        base_url=model_cfg["base_url"],
        api_key=resolve_api_key(model_cfg),
        timeout=run_cfg["request_timeout_s"],
        max_retries=0,  # all retries are handled (and logged) in call_chat
    )


def _error_info(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
    }


def _retry_after(exc):
    resp = getattr(exc, "response", None)
    try:
        return float(resp.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def call_chat(client, model_cfg, run_cfg, messages, seed):
    """Send one chat completion. Returns a dict containing the exact request, the
    full raw response (model_dump), and a log of every attempt. Never raises for
    API errors; failure is reported via ok=False and the attempts log."""
    request = {
        "model": model_cfg["model"],
        "messages": messages,
        "temperature": run_cfg["temperature"],
        "max_tokens": run_cfg["max_tokens"],
        "seed": seed,
    }
    kwargs = dict(request)
    if model_cfg.get("extra_body"):
        kwargs["extra_body"] = model_cfg["extra_body"]
        request["extra_body"] = model_cfg["extra_body"]

    attempts = []
    generic_failures = 0
    rate_limit_failures = 0
    t_total = time.monotonic()

    while True:
        t0 = time.monotonic()
        started = now_iso()
        try:
            resp = client.chat.completions.create(**kwargs)
            attempts.append({"started": started, "latency_s": time.monotonic() - t0, "ok": True})
            return {
                "ok": True,
                "request": request,
                "response": resp.model_dump(mode="json"),
                "attempts": attempts,
                "latency_s": time.monotonic() - t_total,
            }
        except openai.RateLimitError as exc:
            rate_limit_failures += 1
            attempts.append({"started": started, "latency_s": time.monotonic() - t0,
                             "ok": False, "rate_limited": True, "error": _error_info(exc)})
            if rate_limit_failures > run_cfg["rate_limit_max_retries"]:
                break
            time.sleep(_retry_after(exc) or min(60, 2 ** rate_limit_failures))
        except (openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError) as exc:
            generic_failures += 1
            attempts.append({"started": started, "latency_s": time.monotonic() - t0,
                             "ok": False, "error": _error_info(exc)})
            if generic_failures > run_cfg["retry_max"]:
                break
            time.sleep(2)

    return {
        "ok": False,
        "request": request,
        "response": None,
        "attempts": attempts,
        "latency_s": time.monotonic() - t_total,
    }


def reply_text(call):
    """Assistant message content from a call result ('' if missing)."""
    try:
        return call["response"]["choices"][0]["message"].get("content") or ""
    except (TypeError, KeyError, IndexError):
        return ""
