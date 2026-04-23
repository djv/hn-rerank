from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import cast

from aiolimiter import AsyncLimiter
import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from api.cache_utils import atomic_write_json
from api.constants import (
    LLM_CLUSTER_MAX_RETRIES,
    LLM_CLUSTER_MAX_ROUNDS,
    LLM_CLUSTER_MAX_TOKENS,
    LLM_CLUSTER_MAX_TOTAL_SECONDS,
    LLM_CLUSTER_NAME_MODEL_FALLBACK,
    LLM_CLUSTER_NAME_MODEL_PRIMARY,
    LLM_CLUSTER_NAME_MAX_WORDS,
    LLM_CLUSTER_NAME_PROMPT_VERSION,
    LLM_CLUSTER_TITLE_MAX_CHARS,
    LLM_CLUSTER_TITLE_SAMPLES,
    LLM_429_COOLDOWN_BASE,
    LLM_429_COOLDOWN_MAX,
    LLM_HTTP_CONNECT_TIMEOUT,
    LLM_HTTP_POOL_TIMEOUT,
    LLM_HTTP_READ_TIMEOUT,
    LLM_HTTP_USER_AGENT,
    LLM_HTTP_WRITE_TIMEOUT,
    LLM_MIN_REQUEST_INTERVAL,
    LLM_TEMPERATURE,
    LLM_TLDR_BATCH_SIZE,
    LLM_TLDR_MAX_TOKENS,
    LLM_TLDR_MODEL,
    RATE_LIMIT_ERROR_BACKOFF_BASE,
    RATE_LIMIT_ERROR_BACKOFF_MAX,
)
from api.models import StoryDict, StoryForTldr

type ClusterItem = tuple[StoryDict, float]

logger = logging.getLogger(__name__)

def build_messages(contents: object | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, object], item)
            parts = item_dict.get("parts")
            if not isinstance(parts, list):
                continue
            texts: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    part_dict = cast(dict[str, object], part)
                    text_val = part_dict.get("text")
                    if isinstance(text_val, str):
                        texts.append(text_val)
            if texts:
                messages.append({"role": "user", "content": "".join(texts)})
    return messages


def build_payload(
    model: str,
    contents: object | None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": build_messages(contents),
        "temperature": config.get("temperature", LLM_TEMPERATURE)
        if config
        else LLM_TEMPERATURE,
    }

    if config:
        max_tokens = config.get("max_tokens", config.get("max_output_tokens"))
        if isinstance(max_tokens, (int, float)) and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)

    if config and config.get("response_mime_type") == "application/json":
        payload["response_format"] = {"type": "json_object"}

    return payload


class GroqQuotaError(RuntimeError):
    """Raised when Groq returns a non-retryable quota error (e.g., TPD)."""


class GroqRetryableError(RuntimeError):
    """Raised for retryable Groq errors."""

    def __init__(
        self, message: str, cooldown: float | None = None, is_rate_limit: bool = False
    ) -> None:
        super().__init__(message)
        self.cooldown = cooldown
        self.is_rate_limit = is_rate_limit


_LLM_LIMITER: AsyncLimiter = AsyncLimiter(
    1, max(1.0, float(LLM_MIN_REQUEST_INTERVAL))
)


def _parse_retry_after(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    now = datetime.now(dt.tzinfo)
    delta = (dt - now).total_seconds()
    return max(0.0, delta)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not header:
        return None
    return _parse_retry_after(header)


def _extract_groq_error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text.strip()
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str):
                return msg.strip()
    return resp.text.strip()


_RATE_LIMIT_WAIT = wait_random_exponential(
    min=RATE_LIMIT_ERROR_BACKOFF_BASE, max=RATE_LIMIT_ERROR_BACKOFF_MAX
)


_RATE_LIMIT_429_WAIT = wait_random_exponential(
    min=LLM_429_COOLDOWN_BASE, max=LLM_429_COOLDOWN_MAX
)


def _retry_wait(retry_state: RetryCallState) -> float:
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    if isinstance(exc, GroqRetryableError):
        if exc.cooldown is not None:
            return exc.cooldown
        if exc.is_rate_limit:
            return _RATE_LIMIT_429_WAIT(retry_state)
    return _RATE_LIMIT_WAIT(retry_state)


CLUSTER_NAME_CACHE_PATH = Path(".cache/cluster_names.json")


def _cluster_name_cache_key(story_ids: Sequence[str], model: str) -> str:
    key_src = ",".join(story_ids)
    key_src += f"|model={model}|prompt={LLM_CLUSTER_NAME_PROMPT_VERSION}"
    return hashlib.sha256(key_src.encode()).hexdigest()


def _finalize_cluster_name(raw_name: str) -> str | None:
    """Normalize cluster name formatting without altering semantics."""
    cleaned = raw_name.strip()
    if not cleaned or "\n" in cleaned:
        return None
    if any(ch in cleaned for ch in ("{", "}", "[", "]")):
        return None
    cleaned = " ".join(cleaned.split()[:LLM_CLUSTER_NAME_MAX_WORDS])
    cleaned = cleaned.rstrip(" ,&/").rstrip()
    if cleaned.endswith(" and") or cleaned.endswith(" or"):
        cleaned = cleaned.rsplit(" ", 1)[0].rstrip()
    return cleaned or None


def _load_cluster_name_cache() -> dict[str, str]:
    if CLUSTER_NAME_CACHE_PATH.exists():
        try:
            cache = json.loads(CLUSTER_NAME_CACHE_PATH.read_text())
            if isinstance(cache, dict):
                return {
                    str(key): str(val)
                    for key, val in cache.items()
                    if isinstance(val, str) and val.strip()
                }
        except Exception as e:
            logger.warning(f"Failed to load cluster name cache: {e}")
    return {}


def _save_cluster_name_cache(cache: dict[str, str]) -> None:
    CLUSTER_NAME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CLUSTER_NAME_CACHE_PATH, cache)


def _safe_json_loads(text: str) -> dict[str, object]:
    """Safely load JSON, handling potential markdown blocks."""
    if not text:
        return {}

    def _strip_code_fence(src: str) -> str:
        cleaned = src.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _extract_json_substring(src: str) -> str | None:
        first_obj = src.find("{")
        first_arr = src.find("[")
        if first_obj == -1 and first_arr == -1:
            return None
        if first_arr == -1 or (first_obj != -1 and first_obj < first_arr):
            open_ch, close_ch, start = "{", "}", first_obj
        else:
            open_ch, close_ch, start = "[", "]", first_arr

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(src)):
            ch = src[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return src[start : idx + 1]
        return None

    clean_text = _strip_code_fence(text)
    candidates = [clean_text]
    extracted = _extract_json_substring(clean_text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    import ast

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode failed, trying fallback: {e}")
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except Exception:
            continue
    return {}


async def _generate_with_retry(
    model: str = LLM_TLDR_MODEL,
    contents: object | None = None,
    config: dict[str, object] | None = None,
    max_retries: int = 3,
) -> str | None:
    """Call Groq API with exponential backoff retry logic using httpx."""
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, skipping LLM call")
        return None

    payload = build_payload(model=model, contents=contents, config=config)

    timeout = httpx.Timeout(
        connect=LLM_HTTP_CONNECT_TIMEOUT,
        read=LLM_HTTP_READ_TIMEOUT,
        write=LLM_HTTP_WRITE_TIMEOUT,
        pool=LLM_HTTP_POOL_TIMEOUT,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                retry=retry_if_exception_type(GroqRetryableError),
                wait=_retry_wait,
                reraise=True,
            ):
                with attempt:
                    try:
                        async with _LLM_LIMITER:
                            resp = await client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={
                                    "Authorization": f"Bearer {api_key}",
                                    "Content-Type": "application/json",
                                    "User-Agent": LLM_HTTP_USER_AGENT,
                                },
                                json=payload,
                            )
                    except httpx.HTTPError as e:
                        raise GroqRetryableError(str(e)) from e

                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"]

                    if resp.status_code == 429:
                        error_msg = _extract_groq_error_message(resp)
                        msg_lower = error_msg.lower()
                        if "tokens per day" in msg_lower or " tpd" in msg_lower:
                            raise GroqQuotaError(error_msg)
                        if "requests per day" in msg_lower or " rpd" in msg_lower:
                            raise GroqQuotaError(error_msg)
                        retry_after = _retry_after_seconds(resp)
                        raise GroqRetryableError(
                            error_msg,
                            cooldown=retry_after,
                            is_rate_limit=True,
                        )

                    if resp.status_code in {408, 500, 502, 503, 504}:
                        raise GroqRetryableError(
                            f"Groq API error {resp.status_code}"
                        )

                    error_msg = _extract_groq_error_message(resp)
                    logger.error(
                        "Groq API error %d: %s", resp.status_code, error_msg or resp.text
                    )
                    return None
        except GroqQuotaError:
            raise
        except GroqRetryableError as e:
            logger.error("Groq API call failed after %d retries: %s", max_retries, e)
            return None

    return None


async def generate_batch_cluster_names(
    clusters: dict[int, list[ClusterItem]],
    progress_callback: Callable[[int, int], None] | None = None,
    debug_path: Path | None = None,
) -> dict[int, str]:
    """Generate names for clusters one at a time to improve reliability."""
    if not clusters:
        return {}

    cache = _load_cluster_name_cache()
    results: dict[int, str] = {}
    to_generate: dict[int, list[ClusterItem]] = {}
    primary_model = LLM_CLUSTER_NAME_MODEL_PRIMARY
    fallback_model = LLM_CLUSTER_NAME_MODEL_FALLBACK
    active_model = primary_model
    fallback_triggered = False

    for cid, items in clusters.items():
        # Generate cache key based on sorted story IDs
        story_ids = sorted([str(s.get("id", s.get("objectID", ""))) for s, _ in items])
        primary_key = _cluster_name_cache_key(story_ids, primary_model)
        cached_val = cache.get(primary_key)
        if not cached_val and fallback_model:
            fallback_key = _cluster_name_cache_key(story_ids, fallback_model)
            cached_val = cache.get(fallback_key)
        if cached_val and cached_val.strip():
            results[cid] = cached_val
        else:
            to_generate[cid] = items

    if not to_generate:
        if progress_callback:
            progress_callback(len(clusters), len(clusters))
        return {cid: results.get(cid, "") for cid in clusters}

    cid_list = list(to_generate.keys())
    total_batches = len(cid_list)
    max_rounds = LLM_CLUSTER_MAX_ROUNDS
    request_count = 0
    start_time = time.time()
    deadline = start_time + LLM_CLUSTER_MAX_TOTAL_SECONDS
    missing_overall: set[int] = set()

    debug_records: list[dict[str, object]] = []

    payloads: dict[int, list[str]] = {}
    for cid, items in to_generate.items():
        # Use top titles only to keep prompts compact and reduce latency.
        sorted_items = sorted(items, key=lambda x: -x[1])[:LLM_CLUSTER_TITLE_SAMPLES]
        cluster_titles: list[str] = []
        for s, _ in sorted_items:
            title = str(s.get("title", "")).strip()
            if not title:
                continue
            title = " ".join(title.split())
            if len(title) > LLM_CLUSTER_TITLE_MAX_CHARS:
                title = title[:LLM_CLUSTER_TITLE_MAX_CHARS].rstrip()
            cluster_titles.append(title)

        payloads[cid] = cluster_titles

    def _flush_debug() -> None:
        if debug_path is None:
            return
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(debug_records, indent=2))

    def _remap_batch_results(
        batch_results: dict[str, object],
        pending_ids: Sequence[int],
        context: str,
        batch_index: int,
        attempt: int,
        model: str,
    ) -> dict[str, object]:
        if not batch_results or not pending_ids:
            return batch_results
        pending_set = set(pending_ids)
        numeric_keys: list[str] = []
        for k in batch_results:
            if isinstance(k, str) and k.strip().lstrip("-").isdigit():
                numeric_keys.append(k)
        if not numeric_keys:
            return batch_results
        if any(int(k) in pending_set for k in numeric_keys):
            return batch_results
        # No overlap: model ignored requested IDs. Remap by order.
        ordered_items = sorted(
            ((int(k), v) for k, v in batch_results.items() if k in numeric_keys),
            key=lambda x: x[0],
        )
        if len(pending_ids) == 1:
            return {str(pending_ids[0]): ordered_items[0][1]}
        if len(ordered_items) != len(pending_ids):
            return batch_results
        remapped: dict[str, object] = {}
        for cid, (_, value) in zip(sorted(pending_ids), ordered_items):
            remapped[str(cid)] = value
        if debug_path is not None:
            debug_records.append(
                {
                    "event": "remap_keys",
                    "context": context,
                    "batch_index": batch_index,
                    "attempt": attempt,
                    "model": model,
                    "pending_ids": sorted(pending_ids),
                    "original_keys": numeric_keys,
                }
            )
            _flush_debug()
        return remapped

    for batch_index, cid in enumerate(cid_list, start=1):
        batch_cids = [cid]
        pending: set[int] = {cid}

        for attempt in range(1, max_rounds + 1):
            if not pending:
                break
            if time.time() > deadline:
                debug_records.append(
                    {
                        "event": "timeout",
                        "batch_index": batch_index,
                        "pending": sorted(pending),
                        "elapsed": time.time() - start_time,
                    }
                )
                _flush_debug()
                raise RuntimeError(
                    "Groq cluster naming timed out after "
                    f"{LLM_CLUSTER_MAX_TOTAL_SECONDS:.0f}s. "
                    "Try rerunning with fewer clusters/candidates or wait for quota reset."
                )

            batch_prompts: list[str] = []
            for cid in sorted(pending):
                titles = payloads.get(cid, [])
                title_lines = "\n".join(f"- {t}" for t in titles) if titles else "- (no titles)"
                batch_prompts.append(f"Titles:\n{title_lines}")

            full_prompt = f"""
Provide a concise label between two and six words that describes the cluster stories.

Rules:
- Prefer the most specific recurring technical topic in the titles.
- Use 1-2 topics only if the cluster clearly spans two themes.
- Avoid umbrella terms unless the titles are genuinely diverse.
- Use key terms from the titles where possible.
- Use normal spacing and Title Case (e.g., "Large Language Models", not "LargeLanguageModels" or "Large_Language_Models").
- Return ONLY the label text. Do not return JSON, bullets, or extra commentary.

Group:
{chr(10).join(batch_prompts)}
"""

            try:
                pending_before = sorted(pending)
                logger.info(
                    "Groq cluster naming batch %d/%d attempt %d starting (%d pending).",
                    batch_index,
                    total_batches,
                    attempt,
                    len(pending),
                )
                request_count += 1
                t0 = time.time()
                attempt_model = active_model
                text = await _generate_with_retry(
                    model=attempt_model,
                    contents=full_prompt,
                    config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                    },
                    max_retries=LLM_CLUSTER_MAX_RETRIES,
                )
                duration = time.time() - t0

                returned = 0
                batch_results: dict[str, object] | None = None
                if (
                    text is None
                    and attempt_model == primary_model
                    and fallback_model
                    and not fallback_triggered
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "empty_response",
                                "batch_index": batch_index,
                                "attempt": attempt,
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                if text:
                    logger.debug("Groq cluster naming raw response: %s", text)
                    if len(pending) == 1:
                        only_cid = next(iter(pending))
                        final_name = _finalize_cluster_name(text)
                        if final_name:
                            results[only_cid] = final_name
                            items = to_generate[only_cid]
                            story_ids = sorted(
                                [
                                    str(s.get("id", s.get("objectID", "")))
                                    for s, _ in items
                                ]
                            )
                            cache_key = _cluster_name_cache_key(
                                story_ids, attempt_model
                            )
                            cache[cache_key] = final_name
                            returned += 1
                pending = {cid for cid in pending if cid not in results}
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "model": attempt_model,
                            "batch_cids": batch_cids,
                            "pending_before": pending_before,
                            "payloads": {
                                str(cid): payloads.get(cid, {})
                                for cid in sorted(pending_before)
                            },
                            "prompt": full_prompt,
                            "response": text,
                            "parsed": batch_results if text else None,
                            "returned": returned,
                            "pending_after": sorted(pending),
                        }
                    )
                    _flush_debug()
                logger.info(
                    "Groq cluster naming batch %d/%d attempt %d: %d/%d names in %.2fs (pending %d).",
                    batch_index,
                    total_batches,
                    attempt,
                    returned,
                    len(batch_cids),
                    duration,
                    len(pending),
                )

                if pending and attempt < max_rounds:
                    await asyncio.sleep(min(2**attempt, 8))
            except GroqQuotaError as e:
                if (
                    not fallback_triggered
                    and fallback_model
                    and active_model == primary_model
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "quota_error",
                                "batch_index": batch_index,
                                "attempt": attempt,
                                "batch_cids": batch_cids,
                                "pending_before": sorted(pending),
                                "error": str(e),
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                    if attempt < max_rounds:
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "batch_cids": batch_cids,
                            "pending_before": sorted(pending),
                            "error": str(e),
                            "model": active_model,
                        }
                    )
                    _flush_debug()
                raise RuntimeError(f"Groq quota exceeded: {e}") from e
            except Exception as e:
                logger.warning(f"Cluster naming batch failed: {e}")
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "batch",
                            "batch_index": batch_index,
                            "attempt": attempt,
                            "batch_cids": batch_cids,
                            "pending_before": sorted(pending),
                            "error": repr(e),
                        }
                    )
                    _flush_debug()
                if attempt < max_rounds:
                    await asyncio.sleep(min(2**attempt, 8))

        if pending:
            missing_overall.update(pending)

        if progress_callback:
            progress_callback(len(results), len(clusters))

    _save_cluster_name_cache(cache)

    if missing_overall:
        # Final rescue pass: retry missing clusters in small batches.
        missing_list = sorted(missing_overall)
        for rescue_index, cid in enumerate(missing_list, start=1):
            batch_missing = [cid]
            if time.time() > deadline:
                debug_records.append(
                    {
                        "event": "timeout",
                        "batch_index": "rescue",
                        "pending": batch_missing,
                        "elapsed": time.time() - start_time,
                    }
                )
                _flush_debug()
                raise RuntimeError(
                    "Groq cluster naming timed out after "
                    f"{LLM_CLUSTER_MAX_TOTAL_SECONDS:.0f}s. "
                    "Try rerunning with fewer clusters/candidates or wait for quota reset."
                )

            rescue_prompts = []
            for cid in batch_missing:
                titles = payloads.get(cid, [])
                title_lines = "\n".join(f"- {t}" for t in titles) if titles else "- (no titles)"
                rescue_prompts.append(f"Titles:\n{title_lines}")

            rescue_prompt = f"""
Provide a concise label between two and six words that describes the cluster stories.

Rules:
- Prefer specific technical topics from the titles.
- Generic labels are allowed only if the titles are genuinely diverse.
- Use normal spacing and Title Case (e.g., "Large Language Models", not "LargeLanguageModels" or "Large_Language_Models").
- Return ONLY the label text. Do not return JSON, bullets, or extra commentary.

Group:
{chr(10).join(rescue_prompts)}
"""

            try:
                rescue_model = active_model
                request_count += 1
                t0 = time.time()
                try:
                    text = await _generate_with_retry(
                        model=rescue_model,
                        contents=rescue_prompt,
                        config={
                            "temperature": LLM_TEMPERATURE,
                            "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                        },
                        max_retries=1,
                    )
                except GroqQuotaError as e:
                    if (
                        not fallback_triggered
                        and fallback_model
                        and active_model == primary_model
                    ):
                        fallback_triggered = True
                        active_model = fallback_model
                        rescue_model = active_model
                        if debug_path is not None:
                            debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "quota_error_rescue",
                                "batch_index": rescue_index,
                                "batch_cids": batch_missing,
                                "error": str(e),
                                    "from_model": primary_model,
                                    "to_model": fallback_model,
                                }
                            )
                            _flush_debug()
                        request_count += 1
                        t0 = time.time()
                        text = await _generate_with_retry(
                            model=rescue_model,
                            contents=rescue_prompt,
                            config={
                                "temperature": LLM_TEMPERATURE,
                                "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                            },
                            max_retries=1,
                        )
                    else:
                        raise
                duration = time.time() - t0
                if (
                    text is None
                    and rescue_model == primary_model
                    and fallback_model
                    and not fallback_triggered
                ):
                    fallback_triggered = True
                    active_model = fallback_model
                    rescue_model = active_model
                    if debug_path is not None:
                        debug_records.append(
                            {
                                "event": "fallback",
                                "reason": "empty_response_rescue",
                                "batch_index": rescue_index,
                                "batch_cids": batch_missing,
                                "from_model": primary_model,
                                "to_model": fallback_model,
                            }
                        )
                        _flush_debug()
                    request_count += 1
                    t0 = time.time()
                    text = await _generate_with_retry(
                        model=rescue_model,
                        contents=rescue_prompt,
                        config={
                            "temperature": LLM_TEMPERATURE,
                            "max_output_tokens": LLM_CLUSTER_MAX_TOKENS,
                        },
                        max_retries=1,
                    )
                    duration = time.time() - t0

                returned = 0
                batch_results: dict[str, object] | None = None
                if text and len(batch_missing) == 1:
                    cid = batch_missing[0]
                    final_name = _finalize_cluster_name(text)
                    if final_name:
                        results[cid] = final_name
                        items = to_generate[cid]
                        story_ids = sorted(
                            [
                                str(s.get("id", s.get("objectID", "")))
                                for s, _ in items
                            ]
                        )
                        cache_key = _cluster_name_cache_key(
                            story_ids, rescue_model
                        )
                        cache[cache_key] = final_name
                        returned += 1

                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "rescue_batch",
                            "batch_index": rescue_index,
                            "model": rescue_model,
                            "batch_cids": batch_missing,
                            "payloads": {
                                str(cid): payloads.get(cid, {})
                                for cid in batch_missing
                            },
                            "prompt": rescue_prompt,
                            "response": text,
                            "parsed": batch_results if text else None,
                            "returned": returned,
                            "duration": duration,
                        }
                    )
                    _flush_debug()
            except GroqQuotaError as e:
                if debug_path is not None:
                    debug_records.append(
                        {
                            "mode": "rescue_batch",
                            "batch_index": rescue_index,
                            "batch_cids": batch_missing,
                            "error": str(e),
                        }
                    )
                    _flush_debug()
                raise RuntimeError(f"Groq quota exceeded: {e}") from e
            except Exception as e:
                logger.warning(f"Cluster naming rescue batch failed: {e}")

    _flush_debug()

    missing = [cid for cid in clusters if not results.get(cid)]
    if missing:
        elapsed = time.time() - start_time
        raise RuntimeError(
            "Groq cluster naming failed for "
            f"{len(missing)} clusters after {request_count} requests "
            f"({elapsed:.1f}s). Missing IDs: {sorted(missing)}. "
            "Likely rate-limited or incomplete JSON. "
            "Try rerunning, lowering --clusters, or waiting for quota reset."
        )

    return {cid: results.get(cid, "") for cid in clusters}


TLDR_CACHE_PATH = Path(".cache/tldrs.json")


def _load_tldr_cache() -> dict[str, str]:
    """Load TL;DR cache from disk."""
    if TLDR_CACHE_PATH.exists():
        try:
            return json.loads(TLDR_CACHE_PATH.read_text())
        except Exception as e:
            logger.warning(f"Failed to load TLDR cache: {e}")
    return {}


def _save_tldr_cache(cache: dict[str, str]) -> None:
    """Save TL;DR cache to disk."""
    TLDR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(TLDR_CACHE_PATH, cache)


async def generate_batch_tldrs(
    stories: Sequence[StoryForTldr],
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[int, str]:
    """Generate TL;DRs for multiple stories in batches to save API quota."""
    if not stories:
        return {}

    cache = _load_tldr_cache()
    results: dict[int, str] = {}
    to_generate: list[StoryForTldr] = []

    for s in stories:
        sid = int(s["id"])
        cached_val = cache.get(str(sid))
        if cached_val and len(cached_val.strip()) > 0:
            results[sid] = cached_val
        else:
            to_generate.append(s)

    if not to_generate:
        if progress_callback:
            progress_callback(len(stories), len(stories))
        return {
            int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
            for s in stories
        }

    total_to_gen = len(to_generate)
    completed_initial = len(stories) - total_to_gen

    for i in range(0, total_to_gen, LLM_TLDR_BATCH_SIZE):
        original_batch = to_generate[i : i + LLM_TLDR_BATCH_SIZE]
        pending_stories: dict[int, StoryForTldr] = {
            int(s["id"]): s for s in original_batch
        }

        # Try up to 2 times to get all summaries for this batch
        for attempt in range(2):
            if not pending_stories:
                break

            current_batch = list(pending_stories.values())
            stories_formatted = []
            for s in current_batch:
                title = s.get("title", "Untitled")
                comments = s.get("comments", [])
                context = f"ID: {s['id']}\nTitle: {title}"
                if comments:
                    context += "\nComments:\n" + "\n".join(
                        f"- {c[:300]}" for c in comments[:4]
                    )
                stories_formatted.append(context)

            batch_context = "\n\n---\n\n".join(stories_formatted)

            prompt = f"""
Summarize the discussion and technical insights in 2-3 sentences.
CRITICAL: Do NOT repeat the title. The user sees it. Do NOT say "This story is about...".

Focus on:
- Technical implementation details, trade-offs, or benchmarks mentioned in comments
- Significant debates, criticisms, or comparisons to other tools

BAD: "PostgreSQL 17 Released. It features..." (Repeats title)
GOOD: "Praised for incremental backups and optimized vacuuming, though some users warn about update conflicts on legacy systems."

Return JSON with story IDs as keys: {{ "12345": "Summary here." }}

Stories:
{batch_context}

JSON:"""

            try:
                text = await _generate_with_retry(
                    model=LLM_TLDR_MODEL,
                    contents=prompt,
                    config={
                        "temperature": LLM_TEMPERATURE,
                        "max_output_tokens": LLM_TLDR_MAX_TOKENS,
                        "response_mime_type": "application/json",
                    },
                )

                if text:
                    batch_results = _safe_json_loads(text)
                    for sid_str, tldr in batch_results.items():
                        try:
                            sid = int(sid_str)
                            if not isinstance(tldr, str):
                                logger.debug(
                                    f"TLDR value for {sid_str} was not a string"
                                )
                                continue
                            tldr_clean = tldr.strip().strip('"').strip("'")
                            if sid in pending_stories:
                                results[sid] = tldr_clean
                                cache[str(sid)] = tldr_clean
                                del pending_stories[sid]
                        except (ValueError, TypeError) as e:
                            logger.debug(f"Failed to parse TLDR for {sid_str}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"TLDR batch generation failed (attempt {attempt+1}): {e}")

        if progress_callback:
            progress_callback(completed_initial + i + len(original_batch), len(stories))

    _save_tldr_cache(cache)
    return {
        int(s["id"]): results.get(int(s["id"]), cache.get(str(s["id"]), ""))
        for s in stories
    }
