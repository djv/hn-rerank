#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import asyncio
import os
import statistics
import time
from typing import Any

import httpx


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _build_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }


async def _probe(
    *,
    model: str,
    prompt: str,
    samples: int,
    timeout: float,
    min_interval: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    max_tokens: int,
) -> int:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set.")
        return 2

    payload = _build_payload(model, prompt, max_tokens)
    latencies: list[float] = []
    statuses: list[int] = []
    errors: list[str] = []
    cooldowns: list[float] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(samples):
            attempt = 0
            while True:
                if attempt > 0:
                    delay = min(backoff_base * (2** (attempt - 1)), backoff_max)
                    cooldowns.append(delay)
                    await asyncio.sleep(delay)

                await asyncio.sleep(min_interval)
                start = time.perf_counter()
                try:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                except Exception as exc:  # pragma: no cover - runtime diagnostic
                    elapsed = time.perf_counter() - start
                    errors.append(f"{type(exc).__name__}: {exc}")
                    statuses.append(-1)
                    latencies.append(elapsed)
                    break

                elapsed = time.perf_counter() - start
                statuses.append(resp.status_code)
                latencies.append(elapsed)

                if resp.status_code == 200:
                    break

                if resp.status_code == 429 and attempt < max_retries:
                    attempt += 1
                    continue

                errors.append(f"{resp.status_code}: {resp.text[:200]}")
                break

            print(
                f"sample {i + 1}/{samples}: status={statuses[-1]} "
                f"latency={latencies[-1]:.2f}s"
            )

    ok_lat = [lat for lat, status in zip(latencies, statuses) if status == 200]
    if ok_lat:
        median = statistics.median(ok_lat)
        p90 = _percentile(ok_lat, 90)
        p95 = _percentile(ok_lat, 95)
        p99 = _percentile(ok_lat, 99)
        max_lat = max(ok_lat)
        print("\nLatency stats for 200 OK responses:")
        print(f"- median: {median:.2f}s")
        print(f"- p90: {p90:.2f}s")
        print(f"- p95: {p95:.2f}s")
        print(f"- p99: {p99:.2f}s")
        print(f"- max: {max_lat:.2f}s")
    else:
        print("\nNo successful 200 OK responses.")

    if cooldowns:
        print("\nCooldowns applied (due to retries):")
        print(f"- count: {len(cooldowns)}")
        print(f"- max: {max(cooldowns):.2f}s")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"- {err}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Groq latency and errors.")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--prompt", default="Return a short, 3 word label.")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-json-key", default="prompt")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--min-interval", type=float, default=6.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--backoff-base", type=float, default=20.0)
    parser.add_argument("--backoff-max", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as fp:
            raw = fp.read()
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                entry = data[min(args.prompt_index, len(data) - 1)]
                if isinstance(entry, dict):
                    prompt = str(entry.get(args.prompt_json_key, prompt))
                else:
                    prompt = str(entry)
            elif isinstance(data, dict):
                prompt = str(data.get(args.prompt_json_key, prompt))
        except Exception:
            prompt = raw

    return asyncio.run(
        _probe(
            model=args.model,
            prompt=prompt,
            samples=args.samples,
            timeout=args.timeout,
            min_interval=args.min_interval,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            backoff_max=args.backoff_max,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
