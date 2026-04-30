#!/usr/bin/env -S uv run
import asyncio
import json
import time
import httpx
import sys
from pathlib import Path
from typing import List

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.fetching import get_best_stories  # noqa: E402
from api.llm_utils import _build_tldr_prompt  # noqa: E402
from api.config import AppConfig  # noqa: E402

MODELS = [
    "phi4-mini:latest",
    "gemma4:e2b",
]

OLLAMA_URL = "http://localhost:11434/api/chat"

async def check_and_pull(model: str):
    async with httpx.AsyncClient(timeout=None) as client:
        print(f"Checking for model {model}...")
        resp = await client.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            tags = [t["name"] for t in resp.json().get("models", [])]
            if model in tags:
                print(f"  Model {model} already exists.")
                return True
        
        print(f"  Pulling {model}...")
        # We use the pull API to show progress if we were interactive, 
        # but here we'll just wait.
        try:
            async with client.stream("POST", "http://localhost:11434/api/pull", json={"name": model}) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "status" in data and "total" in data:
                        # Print some progress
                        pass
            return True
        except Exception as e:
            print(f"  Failed to pull {model}: {e}")
            return False

async def benchmark_model(model: str, stories_formatted: List[str]):
    prompt = _build_tldr_prompt(stories_formatted)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.2,
        }
    }
    
    print(f"\nBenchmarking {model}...")
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            duration = time.time() - start_time
            
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code} - {resp.text}")
                return None
            
            data = resp.json()
            content = data["message"]["content"]
            
            # Validation
            try:
                parsed = json.loads(content)
                valid_json = True
                summary_count = len(parsed)
            except Exception:
                valid_json = False
                summary_count = 0
            
            tokens = data.get("eval_count", 0)
            tps = tokens / duration if duration > 0 else 0
            
            return {
                "model": model,
                "duration": duration,
                "tokens": tokens,
                "tps": tps,
                "valid_json": valid_json,
                "summaries": summary_count,
                "content": content[:200] + "..."
            }
    except Exception as e:
        print(f"  Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    # 1. Fetch some test data
    print("Fetching test stories...")
    # Get some "best" stories
    stories = await get_best_stories(limit=3, config=AppConfig(no_rss=True))
    
    if not stories:
        print("No stories found.")
        return

    stories_formatted = []
    for s in stories:
        sid = s.id
        title = s.title or "Untitled"
        comments = list(s.comments)
        text_content = s.text_content or ""
        
        context = f"### STORY ID: {sid} ###\nTitle: {title}"
        if text_content:
            context += f"\nContent: {text_content[:500]}"
        if comments:
            context += "\nComments:\n" + "\n".join(f"- {c[:200]}" for c in comments[:3])
        stories_formatted.append(context)

    # 2. Run benchmarks
    results = []
    for model in MODELS:
        # For this test, we only pull if they aren't there. 
        # If pull fails, we skip.
        if await check_and_pull(model):
            res = await benchmark_model(model, stories_formatted)
            if res:
                results.append(res)
    
    # 3. Print Report
    print("\n" + "="*80)
    print(f"{'Model':<20} | {'Time (s)':<10} | {'TPS':<8} | {'JSON':<6} | {'Count'}")
    print("-" * 80)
    for r in results:
        json_ok = "YES" if r["valid_json"] else "NO"
        print(f"{r['model']:<20} | {r['duration']:<10.2f} | {r['tps']:<8.1f} | {json_ok:<6} | {r['summaries']}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
