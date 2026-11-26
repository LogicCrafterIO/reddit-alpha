"""llm — Ollama Cloud chat client with honest token accounting.

The two models this workspace has on that endpoint are reasoning models: they
write a `thinking` field first and the `content` field second. A small
`num_predict` therefore returns an empty answer with `done_reason: length`, which
looks exactly like a broken API. This client budgets for the thinking, retries
once with a bigger budget, and — only as a last resort — reports that it salvaged
JSON out of the reasoning field, so a caller never mistakes thinking for output.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

import config as C

USAGE: dict[str, Any] = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0,
                         "thinking_chars": 0, "by_tag": {}, "started_at": None}


class LLMError(RuntimeError):
    pass


def _account(tag: str, tin: int, tout: int, thinking: int = 0) -> None:
    USAGE["calls"] += 1
    USAGE["tokens_in"] += tin
    USAGE["tokens_out"] += tout
    USAGE["thinking_chars"] += thinking
    row = USAGE["by_tag"].setdefault(tag, {"calls": 0, "in": 0, "out": 0})
    row["calls"] += 1
    row["in"] += tin
    row["out"] += tout


def usage_snapshot(reset: bool = False) -> dict:
    snap = {k: (dict(v) if isinstance(v, dict) else v) for k, v in USAGE.items()
            if k != "by_tag"}
    snap["by_tag"] = {k: dict(v) for k, v in USAGE["by_tag"].items()}
    if reset:
        USAGE.update({"calls": 0, "tokens_in": 0, "tokens_out": 0, "errors": 0,
                      "thinking_chars": 0, "by_tag": {}})
    return snap


def configured() -> bool:
    return bool(C.OLLAMA_API_KEY)


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {C.OLLAMA_API_KEY}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()[:400].decode(errors="replace")
        raise LLMError(f"HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc


def chat(messages: list[dict], *, model: str | None = None, tag: str = "chat",
         json_mode: bool = False, num_predict: int | None = None,
         temperature: float | None = None, think: bool | None = None,
         attempts: int = 2) -> dict:
    """One completion. Returns {text, thinking, model, tokens_in, tokens_out, salvaged}."""
    if not configured():
        raise LLMError("no OLLAMA_API_KEY in .env")
    model = model or C.LLM_MODEL
    budget = num_predict or C.LLM_NUM_PREDICT
    want_think = C.LLM_THINK if think is None else bool(think)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": want_think,
        "options": {"num_predict": budget,
                    "temperature": C.LLM_TEMPERATURE if temperature is None else temperature},
    }
    if json_mode:
        payload["format"] = "json"

    last_err: Exception | None = None
    for attempt in range(attempts):
        started = time.time()
        try:
            data = _post(f"{C.OLLAMA_BASE_URL}/api/chat", payload)
        except LLMError as exc:
            last_err = exc
            if attempt + 1 < attempts:
                budget = int(budget * 1.5)
                payload["options"]["num_predict"] = budget
                time.sleep(1.5)
                continue
            USAGE["errors"] += 1
            raise

        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        thinking = msg.get("thinking") or ""
        tin = int(data.get("prompt_eval_count") or 0)
        tout = int(data.get("eval_count") or 0)
        _account(tag, tin, tout, len(thinking))
        if USAGE["started_at"] is None:
            USAGE["started_at"] = time.time()

        if text:
            return {"text": text, "thinking": thinking, "model": data.get("model", model),
                    "tokens_in": tin, "tokens_out": tout, "salvaged": False,
                    "think": want_think, "elapsed_s": round(time.time() - started, 1)}

        # Empty content means the budget went on the reasoning field. Switching
        # reasoning off recovers the answer far more cheaply than doubling a
        # budget that the reasoning will simply consume again.
        if attempt + 1 < attempts:
            if want_think:
                want_think = False
                payload["think"] = False
            else:
                budget = max(int(budget * 2), 4000)
                payload["options"]["num_predict"] = budget
            continue
        last_err = LLMError(
            f"{model} returned empty content (done_reason={data.get('done_reason')!r}, "
            f"{len(thinking)} chars of thinking, eval={tout})")
    raise last_err or LLMError("empty response")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise LLMError("empty text")
    for candidate in (text,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # last resort: the outermost object or array in the string
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"no JSON in response: {text[:200]!r}")


def _balanced_objects(text: str) -> list[str]:
    """The outermost complete {...} spans, even when an enclosing brace never closed.

    A truncated `{"ideas": [{...}, {...}, {"headl` has no complete root, but its
    finished children are intact — so spans are ranked by nesting depth and the
    shallowest complete level is returned.
    """
    stack, spans = [], []
    for i, ch in enumerate(text or ""):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            spans.append((start, i + 1, len(stack)))
    if not spans:
        return []
    shallowest = min(depth for _s, _e, depth in spans)
    return [text[s:e] for s, e, depth in spans if depth == shallowest]


def _salvage_partial(text: str) -> Any:
    """Recover the finished objects from a response the token budget cut mid-string.

    A model that writes three ideas and runs out of room mid-third leaves invalid
    JSON and two perfectly good ideas. Discarding both is the wrong trade: parse
    the complete objects and keep them, and let the caller know it was truncated.
    """
    objs = []
    for chunk in _balanced_objects(text):
        try:
            objs.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    if not objs:
        raise LLMError("nothing recoverable")
    for key in ("ideas", "picks"):
        gathered = [x for o in objs if isinstance(o, dict) and isinstance(o.get(key), list)
                    for x in o[key]]
        if gathered:
            return {key: gathered}
    ideas = [o for o in objs if isinstance(o, dict) and ("headline" in o or "thesis" in o)]
    if ideas:
        return {"ideas": ideas}
    return objs[0]


def chat_json(messages: list[dict], *, model: str | None = None, tag: str = "json",
              num_predict: int | None = None, think: bool | None = None) -> Any:
    """Chat, then parse JSON — repair a truncated answer rather than lose it."""
    out = chat(messages, model=model, tag=tag, json_mode=True, num_predict=num_predict,
               think=think)

    def parse(text: str, *, partial_ok: bool = True) -> Any:
        try:
            return _extract_json(text)
        except LLMError:
            if not partial_ok:
                raise
        parsed = _salvage_partial(text)
        USAGE["truncated"] = USAGE.get("truncated", 0) + 1
        return parsed

    try:
        return parse(out["text"])
    except LLMError:
        pass

    # The answer ran out of budget mid-sentence: one retry with more room and no
    # reasoning costs less than throwing away the call that was already paid for.
    try:
        bigger = max(4200, int((num_predict or C.LLM_NUM_PREDICT) * 1.8))
        retry = chat(messages, model=model, tag=f"{tag}+retry", json_mode=True,
                     num_predict=bigger, think=False)
        return parse(retry["text"])
    except LLMError:
        pass

    # Last resort: some models write the JSON into the reasoning field.
    if out.get("thinking"):
        try:
            parsed = _extract_json(out["thinking"])
            USAGE["salvaged"] = USAGE.get("salvaged", 0) + 1
            return parsed.get("ideas", parsed) if isinstance(parsed, dict) else parsed
        except LLMError:
            pass
    raise LLMError(f"no JSON in response: {(out.get('text') or '')[:200]!r}")

