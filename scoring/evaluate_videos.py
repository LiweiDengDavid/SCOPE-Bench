"""Annotate videos with an LLM-based Cognitive Depth Score (0-6 or null).

Pipeline:
1. Read items from JSON/JSONL (from convert_csv_to_json.py).
2. Build system+user prompt (prompts.py).
3. Call chat completions API (SiliconFlow, OpenAI, or OpenRouter) in stream mode.
4. Parse strict JSON output:
   score, level_name, reason, evidence, confidence
5. Append one JSON record per sample to output JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ModuleNotFoundError:
    requests = None

from prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    is_asr_cn_missing,
    resolve_scoring_inputs,
)


DEFAULT_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_API_KEY_ENVS = {
    "siliconflow": [
        "SILICONFLOW_API_KEY",
        "API_KEY",
        "MINIMAX_API_KEY",
        "ANTHROPIC_API_KEY",
    ],
    "openai": [
        "OPENAI_API_KEY",
        "API_KEY",
    ],
    "openrouter": [
        "OPENROUTER_API_KEY",
        "API_KEY",
    ],
}

OPENROUTER_MODEL_IDS = {
    "qwen/qwen3.7-max",
    "xiaomi/mimo-v2.5-pro",
    "google/gemini-3.1-pro-preview",
}

SCORE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "video_cognitive_depth_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0, "maximum": 6},
                        {"type": "null"},
                    ]
                },
                "level_name": {"type": "string"},
                "reason": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": [
                "score",
                "level_name",
                "reason",
                "evidence",
                "confidence",
            ],
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "properties": {"score": {"type": "null"}},
                        "required": ["score"],
                    },
                    "then": {
                        "properties": {
                            "level_name": {
                                "const": "insufficient_information"
                            }
                        }
                    },
                }
            ],
        },
    },
}


def _video_id_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def iter_items(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(4096)

    if prefix.lstrip().startswith("["):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must contain a list: {path}")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Each input item must be a JSON object: {path}")
            yield item
        return

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Line {lineno} is not a JSON object: {path}")
            yield item


def load_items(path: Path) -> List[Dict[str, Any]]:
    return list(iter_items(path))


def select_items(
    path: Path,
    done_ids: set[str],
    start: int,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    scanned = 0
    skipped_done = 0
    skipped_start = 0
    selected_ids: set[str] = set()

    if limit == 0:
        return selected, {
            "scanned": 0,
            "skipped_done": 0,
            "skipped_start": 0,
            "selected": 0,
        }

    for item in iter_items(path):
        scanned += 1
        video_id = _video_id_key(item.get("video_id"))
        if video_id is None:
            raise ValueError(f"Input item #{scanned} has no video_id")
        if video_id in done_ids:
            skipped_done += 1
            continue
        if skipped_start < start:
            skipped_start += 1
            continue
        if video_id in selected_ids:
            raise ValueError(f"Duplicate video_id in selected input: {video_id}")
        selected_ids.add(video_id)
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break

    return selected, {
        "scanned": scanned,
        "skipped_done": skipped_done,
        "skipped_start": skipped_start,
        "selected": len(selected),
    }


def load_done_ids(path: Path, include_errors: bool = False) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Failed API calls and malformed model output are retried by
            # default. Some large continuation jobs may explicitly choose to
            # keep those rows and skip past them.
            if (rec.get("error") or rec.get("parse_error")) and not include_errors:
                continue
            vid = _video_id_key(rec.get("video_id"))
            if vid is not None:
                done.add(vid)
    return done


LEVEL_NAME_BY_SCORE = {
    0: "Affect",
    1: "Point",
    2: "Concept",
    3: "Procedure",
    4: "Mechanism",
    5: "Judgment",
    6: "Model",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
INSUFFICIENT_ASR_LEVEL_NAME = "insufficient_information"
INSUFFICIENT_ASR_REASON = (
    "The Chinese ASR information is missing or null in the input, so the actual "
    "informational depth provided by the video cannot be judged reliably."
)


def _normalize_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not value.is_integer():
            return None
        value = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.upper() == "NAN":
            return None
        try:
            as_float = float(text)
        except ValueError:
            return None
        if math.isnan(as_float) or not as_float.is_integer():
            return None
        value = int(as_float)
    elif isinstance(value, int):
        pass
    else:
        return None

    if 0 <= value <= 6:
        return int(value)
    return None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    # Drop the opening fence (e.g. ```json)
    lines = lines[1:]
    # Drop trailing fences if present.
    while lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _extract_first_json_object(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    if not text:
        raise ValueError("empty response")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no valid JSON object found in model output")


def _extract_segment(text: str, key: str, next_keys: List[str]) -> Optional[str]:
    start_match = re.search(rf'"{re.escape(key)}"\s*:', text, flags=re.S)
    if not start_match:
        return None
    start = start_match.end()
    end = len(text)
    for next_key in next_keys:
        m = re.search(
            rf',\s*"{re.escape(next_key)}"\s*:',
            text[start:],
            flags=re.S,
        )
        if m:
            end = min(end, start + m.start())
    return text[start:end].strip()


def _unwrap_json_string(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.endswith(","):
        s = s[:-1].rstrip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    # Common quote artifacts in malformed model JSON.
    s = s.replace('\\"', '"').replace('""', '"')
    return s.strip()


def _parse_lenient_payload(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fence(raw_text)
    parsed: Dict[str, Any] = {
        "score": None,
        "level_name": None,
        "reason": "",
        "evidence": [],
        "confidence": "low",
        "parse_error": None,
    }

    score_raw = _extract_segment(text, "score", ["level_name", "reason", "evidence", "confidence"])
    level_raw = _extract_segment(text, "level_name", ["reason", "evidence", "confidence"])
    reason_raw = _extract_segment(text, "reason", ["evidence", "confidence"])
    evidence_raw = _extract_segment(text, "evidence", ["confidence"])
    confidence_raw = _extract_segment(text, "confidence", [])

    score = _normalize_score(_unwrap_json_string(score_raw))
    level_name = _unwrap_json_string(level_raw)
    reason = _unwrap_json_string(reason_raw)

    evidence_items: List[str] = []
    if evidence_raw:
        evidence_body = evidence_raw.strip()
        if evidence_body.startswith("[") and evidence_body.endswith("]"):
            evidence_inner = evidence_body[1:-1]
        else:
            evidence_inner = evidence_body
        for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', evidence_inner, flags=re.S):
            item = m.group(1).replace('\\"', '"').strip()
            if item:
                evidence_items.append(item)
        # Fallback for malformed array items that failed regex.
        if not evidence_items:
            for line in evidence_inner.splitlines():
                line = line.strip().strip(",")
                if not line:
                    continue
                evidence_items.append(_unwrap_json_string(line))
    evidence_items = [x for x in evidence_items if x][:3]

    conf_match = re.search(
        r'"confidence"\s*:\s*"?([A-Za-z]+)"?',
        text,
        flags=re.S,
    )
    if conf_match:
        confidence = conf_match.group(1).strip().lower()
    else:
        confidence = _unwrap_json_string(confidence_raw).lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    parsed["score"] = score
    parsed["level_name"] = (
        LEVEL_NAME_BY_SCORE[score]
        if score in LEVEL_NAME_BY_SCORE
        else level_name or "N/A"
    )
    parsed["reason"] = reason
    parsed["evidence"] = evidence_items
    parsed["confidence"] = confidence
    if score is None and not level_name and not reason and not evidence_items:
        parsed["parse_error"] = "no valid JSON object found in model output"
    return parsed


def parse_score_payload(raw_text: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "score": None,
        "level_name": None,
        "reason": None,
        "evidence": [],
        "confidence": "low",
        "parse_error": None,
    }
    try:
        obj = _extract_first_json_object(raw_text)
    except ValueError as exc:
        # Fallback: tolerate malformed JSON (e.g. unescaped quotes in reason).
        lenient = _parse_lenient_payload(raw_text)
        if lenient.get("score") is not None or lenient.get("reason") or lenient.get("evidence"):
            return lenient
        parsed["parse_error"] = str(exc)
        return parsed

    score = _normalize_score(obj.get("score"))
    level_name = obj.get("level_name")
    reason = obj.get("reason")
    evidence = obj.get("evidence")
    confidence = obj.get("confidence")

    if score in LEVEL_NAME_BY_SCORE:
        # Keep the serialized level name identical to the paper rubric even if
        # a provider returns a translated or legacy label.
        level_name = LEVEL_NAME_BY_SCORE[score]
    elif not isinstance(level_name, str) or not level_name.strip():
        level_name = "N/A"
    else:
        level_name = level_name.strip()

    if not isinstance(reason, str) or not reason.strip():
        reason = ""
    else:
        reason = reason.strip()

    evidence_items: List[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    evidence_items.append(cleaned)
    elif isinstance(evidence, str):
        cleaned = evidence.strip()
        if cleaned:
            evidence_items.append(cleaned)
    evidence_items = evidence_items[:3]

    conf = str(confidence).strip().lower() if confidence is not None else ""
    if conf not in VALID_CONFIDENCE:
        conf = "low"

    parsed.update(
        {
            "score": score,
            "level_name": level_name,
            "reason": reason,
            "evidence": evidence_items,
            "confidence": conf,
        }
    )
    return parsed


@dataclass
class ApiConfig:
    model: str
    api_key: str = ""
    base_url: str = ""
    provider: str = "auto"
    max_tokens_param: str = "auto"
    max_tokens: int = 1024
    temperature: float = 1.0
    seed: Optional[int] = None
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 180.0
    structured_output: bool = True
    max_parse_attempts: int = 2
    prompt_cache: bool = True
    session_id: str = ""


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "auto").strip().lower()
    if normalized not in {"auto", "siliconflow", "openai", "openrouter"}:
        raise ValueError(f"unsupported provider: {provider}")
    return normalized


def _infer_provider(model: str, base_url: str, provider: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized != "auto":
        return normalized

    model_text = (model or "").strip().lower()
    base_text = (base_url or "").strip().lower()

    if "openai.com" in base_text:
        return "openai"
    if "siliconflow.cn" in base_text:
        return "siliconflow"
    if "openrouter.ai" in base_text:
        return "openrouter"

    if model_text in OPENROUTER_MODEL_IDS:
        return "openrouter"

    if model_text.startswith(("gpt-", "o1", "o3", "o4", "o5")):
        return "openai"
    return "siliconflow"


def _resolve_base_url(base_url: str, provider: str) -> str:
    text = (base_url or "").strip()
    if text:
        return text.rstrip("/")
    return DEFAULT_BASE_URLS[provider].rstrip("/")


def _resolve_api_key(explicit_key: str, provider: str) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for env_key in DEFAULT_API_KEY_ENVS[provider]:
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    return ""


def _resolve_max_tokens_param(provider: str, model: str, param: str) -> str:
    normalized = (param or "auto").strip().lower()
    if normalized in {"max_tokens", "max_completion_tokens"}:
        return normalized
    if normalized != "auto":
        raise ValueError(f"unsupported max_tokens_param: {param}")

    model_text = (model or "").strip().lower()
    if provider == "openai" and model_text.startswith(("gpt-5", "o1", "o3", "o4", "o5")):
        return "max_completion_tokens"
    return "max_tokens"


def _build_client(cfg: ApiConfig) -> Any:
    if requests is None:
        raise SystemExit(
            "Missing dependency: requests. Install it with `pip install requests`."
        )
    provider = _infer_provider(model=cfg.model, base_url=cfg.base_url, provider=cfg.provider)
    resolved_base_url = _resolve_base_url(cfg.base_url, provider=provider)
    return {
        "provider": provider,
        "url": f"{resolved_base_url}/chat/completions",
        "headers": {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {cfg.api_key}",
        },
    }


def _extract_delta_content(delta: Dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)
    return ""


def _error_suggests_max_completion_tokens(detail: str) -> bool:
    text = detail.lower()
    return (
        "max_completion_tokens" in text
        and "max_tokens" in text
        and "unsupported parameter" in text
    )


def _error_suggests_max_tokens(detail: str) -> bool:
    text = detail.lower()
    return (
        "max_completion_tokens" in text
        and "unsupported parameter" in text
        and "max_tokens" in text
    )


def _error_suggests_default_temperature_only(detail: str) -> bool:
    text = detail.lower()
    return (
        "temperature" in text
        and ("unsupported value" in text or "unsupported parameter" in text)
        and ("only the default (1) value is supported" in text or "only the default" in text)
    )


def _error_suggests_unsupported_response_format(detail: str) -> bool:
    text = detail.lower()
    return (
        "response_format" in text
        and (
            "unsupported" in text
            or "not supported" in text
            or "invalid parameter" in text
        )
    )


def _error_suggests_unsupported_cache_control(detail: str) -> bool:
    text = detail.lower()
    return "cache_control" in text and (
        "unsupported" in text
        or "not supported" in text
        or "invalid parameter" in text
        or "invalid request" in text
    )


def _use_explicit_prompt_cache(provider: str, model: str) -> bool:
    return provider == "openrouter" and model.strip().lower() == "qwen/qwen3.7-max"


def _use_default_temperature_only(provider: str, model: str) -> bool:
    model_text = (model or "").strip().lower()
    if provider != "openai":
        return False
    return model_text.startswith(("gpt-5", "o1", "o3", "o4", "o5"))


def call_llm(
    client: Any,
    cfg: ApiConfig,
    user_prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    last_err: Optional[Exception] = None
    token_param_mode = _resolve_max_tokens_param(
        provider=client["provider"],
        model=cfg.model,
        param=cfg.max_tokens_param,
    )
    temperature_value = float(cfg.temperature)
    temperature_enabled = True
    structured_output_enabled = (
        cfg.structured_output and client["provider"] == "openrouter"
    )
    prompt_cache_enabled = cfg.prompt_cache and _use_explicit_prompt_cache(
        client["provider"],
        cfg.model,
    )
    if _use_default_temperature_only(client["provider"], cfg.model):
        temperature_value = 1.0

    for attempt in range(1, cfg.max_retries + 1):
        try:
            parts: List[str] = []
            response_id: Optional[str] = None
            response_model: Optional[str] = None
            usage: Dict[str, Any] = {}
            system_content: Any = SYSTEM_PROMPT
            if prompt_cache_enabled:
                system_content = [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            payload = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
            }
            if cfg.session_id and client["provider"] == "openrouter":
                payload["session_id"] = cfg.session_id
            payload[token_param_mode] = cfg.max_tokens
            if temperature_enabled:
                payload["temperature"] = temperature_value
            if cfg.seed is not None:
                payload["seed"] = cfg.seed
            if structured_output_enabled:
                payload["response_format"] = SCORE_RESPONSE_FORMAT
            with requests.post(
                client["url"],
                json=payload,
                headers=client["headers"],
                stream=True,
                timeout=cfg.request_timeout_seconds,
            ) as resp:
                if resp.status_code != 200:
                    detail = resp.text.strip()
                    if (
                        structured_output_enabled
                        and _error_suggests_unsupported_response_format(detail)
                    ):
                        structured_output_enabled = False
                        raise RuntimeError(
                            "server does not support response_format, retrying without it"
                        )
                    if (
                        prompt_cache_enabled
                        and _error_suggests_unsupported_cache_control(detail)
                    ):
                        prompt_cache_enabled = False
                        raise RuntimeError(
                            "server does not support cache_control, retrying without it"
                        )
                    if (
                        token_param_mode == "max_tokens"
                        and _error_suggests_max_completion_tokens(detail)
                    ):
                        token_param_mode = "max_completion_tokens"
                        raise RuntimeError(
                            "server requires max_completion_tokens, retrying with adapted parameter"
                        )
                    if (
                        token_param_mode == "max_completion_tokens"
                        and _error_suggests_max_tokens(detail)
                    ):
                        token_param_mode = "max_tokens"
                        raise RuntimeError(
                            "server requires max_tokens, retrying with adapted parameter"
                        )
                    if temperature_enabled and _error_suggests_default_temperature_only(detail):
                        if abs(temperature_value - 1.0) > 1e-9:
                            temperature_value = 1.0
                            raise RuntimeError(
                                "server requires default temperature=1, retrying with adapted parameter"
                            )
                        temperature_enabled = False
                        raise RuntimeError(
                            "server requires default temperature behavior, retrying without temperature"
                        )
                    if 400 <= resp.status_code < 500 and resp.status_code != 429:
                        raise ValueError(
                            f"HTTP {resp.status_code} from {client['provider']}: "
                            f"{detail[:1000]}"
                        )
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from {client['provider']}: {detail[:1000]}"
                    )
                for chunk in resp.iter_lines():
                    if not chunk:
                        continue
                    chunk_str = chunk.decode("utf-8", errors="ignore").strip()
                    if not chunk_str:
                        continue
                    if chunk_str.startswith("data:"):
                        chunk_str = chunk_str[len("data:") :].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk_data.get("id"), str):
                        response_id = chunk_data["id"]
                    if isinstance(chunk_data.get("model"), str):
                        response_model = chunk_data["model"]
                    if isinstance(chunk_data.get("usage"), dict):
                        usage = chunk_data["usage"]
                    err = chunk_data.get("error")
                    if err:
                        raise RuntimeError(f"{client['provider']} stream error: {err}")
                    choices = chunk_data.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content_text = _extract_delta_content(delta)
                    if content_text:
                        parts.append(content_text)
            output = "".join(parts).strip()
            if not output:
                raise RuntimeError(f"Empty streamed response from {client['provider']}.")
            return output, {
                "id": response_id,
                "model": response_model,
                "usage": usage,
            }
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt == cfg.max_retries:
                break
            delay = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
            time.sleep(delay + random.uniform(0.0, min(1.0, delay * 0.25)))
    raise RuntimeError(f"LLM call failed after {cfg.max_retries} retries: {last_err}")


@dataclass
class Progress:
    total: int
    done: int = 0
    ok: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, success: bool) -> None:
        with self.lock:
            self.done += 1
            if success:
                self.ok += 1
            else:
                self.failed += 1
            elapsed = time.time() - self.started_at
            rate = self.done / elapsed if elapsed > 0 else 0.0
            remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
            eta = f"{remaining/60:.1f}m" if remaining != float("inf") else "?"
            sys.stderr.write(
                f"\r[progress] {self.done}/{self.total} "
                f"ok={self.ok} fail={self.failed} "
                f"rate={rate:.2f}/s eta={eta}    "
            )
            sys.stderr.flush()


def score_one(client: Any, cfg: ApiConfig, item: Dict[str, Any]) -> Dict[str, Any]:
    mapped_inputs = resolve_scoring_inputs(item)
    used_caption = mapped_inputs.get("caption")
    used_category = mapped_inputs.get("category")
    used_asr_text = mapped_inputs.get("asr_text")
    if is_asr_cn_missing(item):
        return build_missing_asr_record(item, cfg.seed, mapped_inputs)
    user_prompt = build_user_prompt(item)
    raw = ""
    parsed: Dict[str, Any] = {}
    response_attempts: List[Dict[str, Any]] = []
    for parse_attempt in range(1, cfg.max_parse_attempts + 1):
        raw, response_meta = call_llm(client, cfg, user_prompt)
        response_attempts.append(response_meta)
        parsed = parse_score_payload(raw)
        if not parsed.get("parse_error"):
            if (
                parsed.get("score") is None
                and parsed.get("level_name") != "insufficient_information"
            ):
                parsed["parse_error"] = (
                    "score=null requires level_name=insufficient_information"
                )
            else:
                break
        if parse_attempt == cfg.max_parse_attempts:
            raise RuntimeError(
                "Model returned malformed JSON after "
                f"{cfg.max_parse_attempts} attempts: {parsed['parse_error']}"
            )
    usage: Dict[str, Any] = {}
    for response_meta in response_attempts:
        attempt_usage = response_meta.get("usage")
        if not isinstance(attempt_usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
            value = attempt_usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
        prompt_details = attempt_usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            target = usage.setdefault("prompt_tokens_details", {})
            for key in ("cached_tokens", "cache_write_tokens"):
                value = prompt_details.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    target[key] = target.get(key, 0) + value

    final_meta = response_attempts[-1] if response_attempts else {}
    return {
        "video_id": item.get("video_id"),
        "caption": used_caption,
        # Keep `title` for compatibility with existing score files.
        "title": used_caption,
        "category": used_category,
        "asr_text": used_asr_text,
        "score": parsed.get("score"),
        "level_name": parsed.get("level_name"),
        "reason": parsed.get("reason"),
        "evidence": parsed.get("evidence"),
        "confidence": parsed.get("confidence"),
        "parse_error": parsed.get("parse_error"),
        "seed": cfg.seed,
        "request_id": final_meta.get("id"),
        "response_model": final_meta.get("model"),
        "usage": usage,
        "model_call_count": len(response_attempts),
        "raw_response": raw,
    }


def build_missing_asr_record(
    item: Dict[str, Any],
    seed: Optional[int] = None,
    mapped_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if mapped_inputs is None:
        mapped_inputs = resolve_scoring_inputs(item)
    used_caption = mapped_inputs.get("caption")
    used_category = mapped_inputs.get("category")
    used_asr_text = mapped_inputs.get("asr_text")
    return {
        "video_id": item.get("video_id"),
        "caption": used_caption,
        "title": used_caption,
        "category": used_category,
        "asr_text": used_asr_text,
        "score": None,
        "level_name": INSUFFICIENT_ASR_LEVEL_NAME,
        "reason": INSUFFICIENT_ASR_REASON,
        "evidence": [],
        "confidence": "low",
        "parse_error": None,
        "seed": seed,
        "request_id": None,
        "response_model": None,
        "usage": {},
        "model_call_count": 0,
        "raw_response": "",
        "skip_reason": "missing_asr",
    }


def write_record(out_path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(
    items: Iterable[Dict[str, Any]],
    out_path: Path,
    cfg: ApiConfig,
    concurrency: int,
    continue_on_preflight_failure: bool = False,
) -> Tuple[int, int]:
    items_list = list(items)
    progress = Progress(total=len(items_list))
    write_lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = _build_client(cfg)

    def worker(item: Dict[str, Any]) -> bool:
        try:
            record = score_one(client, cfg, item)
            write_record(out_path, record, write_lock)
            return True
        except Exception as exc:  # noqa: BLE001
            err_record = {
                "video_id": item.get("video_id"),
                "error": str(exc),
            }
            write_record(out_path, err_record, write_lock)
            return False

    # Fail fast before submitting a large batch when credentials, billing, model
    # availability, or output formatting are broken.
    first_item, remaining_items = items_list[0], items_list[1:]
    try:
        first_record = score_one(client, cfg, first_item)
        write_record(out_path, first_record, write_lock)
        first_success = True
    except Exception as exc:  # noqa: BLE001
        write_record(
            out_path,
            {"video_id": first_item.get("video_id"), "error": str(exc)},
            write_lock,
        )
        sys.stderr.write(f"\n[preflight] failed: {exc}\n")
        first_success = False
    progress.update(success=first_success)
    if not first_success and not continue_on_preflight_failure:
        sys.stderr.write("\n")
        return progress.ok, progress.failed

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, item) for item in remaining_items]
        try:
            for fut in as_completed(futures):
                success = fut.result()
                progress.update(success=success)
        except KeyboardInterrupt:
            print("\n[interrupt] cancelling pending tasks ...", flush=True)
            for future in futures:
                future.cancel()
            raise

    sys.stderr.write("\n")
    return progress.ok, progress.failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/items.json"),
        help="Path to items JSON or JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scores.jsonl"),
        help="Path to output JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-V4-Flash",
        help=(
            "Model name, e.g. deepseek-ai/DeepSeek-V4-Flash, "
            "gpt-5.5-2026-04-23, or qwen/qwen3.7-max."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "siliconflow", "openai", "openrouter"),
        default="auto",
        help="LLM provider. auto infers from --model/--base_url.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for the selected provider. If omitted, provider-specific env vars are used.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="",
        help="Base URL. If omitted, auto uses provider default (SiliconFlow/OpenAI/OpenRouter).",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--max_tokens_param",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default="auto",
        help=(
            "Token budget parameter name. auto selects a compatible field by provider/model "
            "(e.g. GPT-5.x on OpenAI uses max_completion_tokens)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the LLM.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional sampling seed. Determinism depends on model/provider support.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=180.0,
        help="HTTP timeout for one LLM request (seconds).",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional item limit after resume filtering.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index after resume filtering.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore existing output file and score everything.",
    )
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--no_structured_output",
        action="store_true",
        help="Disable OpenRouter response_format JSON schema enforcement.",
    )
    parser.add_argument(
        "--max_parse_attempts",
        type=int,
        default=2,
        help="Maximum model calls when a response cannot be parsed as JSON.",
    )
    parser.add_argument(
        "--no_prompt_cache",
        action="store_true",
        help="Disable explicit prompt caching for supported OpenRouter models.",
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default="",
        help="Optional OpenRouter session ID for sticky provider routing.",
    )
    parser.add_argument(
        "--continue_on_preflight_failure",
        action="store_true",
        help=(
            "Continue scoring the remaining items when the first preflight item fails. "
            "Useful for small retry cohorts whose failures may be item-specific."
        ),
    )
    parser.add_argument(
        "--treat_errors_as_done",
        action="store_true",
        help=(
            "Treat existing error/parse_error rows as completed for resume "
            "selection. This is useful when a large run should skip blocked "
            "items instead of retrying them."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.max_retries < 1:
        raise SystemExit("--max_retries must be at least 1")
    if args.max_parse_attempts < 1:
        raise SystemExit("--max_parse_attempts must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("--max_tokens must be at least 1")

    provider = _infer_provider(model=args.model, base_url=args.base_url, provider=args.provider)
    resolved_base_url = _resolve_base_url(args.base_url, provider=provider)
    api_key = _resolve_api_key(args.api_key, provider=provider)
    if not api_key:
        expected_envs = "/".join(DEFAULT_API_KEY_ENVS[provider])
        raise SystemExit(
            f"Missing API key for provider={provider}. "
            f"Pass --api_key or set one of: {expected_envs}."
        )

    cfg = ApiConfig(
        model=args.model,
        api_key=api_key,
        base_url=resolved_base_url,
        provider=provider,
        max_tokens_param=args.max_tokens_param,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        request_timeout_seconds=args.request_timeout,
        structured_output=not args.no_structured_output,
        max_parse_attempts=args.max_parse_attempts,
        prompt_cache=not args.no_prompt_cache,
        session_id=args.session_id.strip(),
    )
    print(
        f"Provider={provider} model={cfg.model} base_url={cfg.base_url} "
        f"max_tokens_param={_resolve_max_tokens_param(provider, cfg.model, cfg.max_tokens_param)} "
        f"temperature={cfg.temperature} seed={cfg.seed}"
    )
    if _use_default_temperature_only(provider, cfg.model) and abs(cfg.temperature - 1.0) > 1e-9:
        print(
            "[note] This model only supports default temperature=1. "
            "The request will automatically use temperature=1.",
            flush=True,
        )

    done = (
        set()
        if args.no_resume
        else load_done_ids(args.output, include_errors=args.treat_errors_as_done)
    )
    items, selection = select_items(
        args.input,
        done_ids=done,
        start=args.start,
        limit=args.limit,
    )
    print(
        f"Input={args.input} scanned={selection['scanned']} "
        f"resume_skipped={selection['skipped_done']} "
        f"start_skipped={selection['skipped_start']}"
    )
    print(f"Scoring {len(items)} items (concurrency={args.concurrency})")

    if not items:
        print("Nothing to do.")
        return

    ok, failed = run(
        items,
        args.output,
        cfg,
        args.concurrency,
        continue_on_preflight_failure=args.continue_on_preflight_failure,
    )
    print(f"Done. ok={ok} failed={failed} -> {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
