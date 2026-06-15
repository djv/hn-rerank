# Local LLM via whichllm for TL;DR + Cluster Naming

## Goal

Replace cloud LLM calls (currently Groq: `llama-3.3-70b-versatile` / Mistral: `mistral-small-latest`)
with a local model run via `uvx whichllm@latest run "Qwen/Qwen3-4B-Thinking-2507"`.

## Feasibility assessment

### `whichllm run` is interactive-only

`whichllm run` starts a chat loop (read from stdin line-by-line, print responses). It is
NOT designed for batch/non-interactive use. There is no `--prompt` flag, no JSON-mode switch,
no guaranteed response format. Each invocation generates a temp script, installs deps via
`uv run --no-project`, loads the model, chats, then tears down. Model loading alone takes
seconds to minutes on CPU.

**Verdict: Not viable as a subprocess replacement for `_generate_with_retry()`.**

### Hardware constraints

| Resource | Available | Required (Qwen3-4B FP16) | Required (Q4_K_M GGUF) |
|----------|-----------|--------------------------|-------------------------|
| RAM | 7.6 GB | ~8 GB (weights only) | ~2.4 GB |
| Disk | 20 GB free | ~8 GB (model files) | ~2.4 GB |
| GPU | None | — | — |
| Swap | 5.7 GB free | Will thrash | OK |

CPU-only inference on 4B params: expect **30-90 seconds per response**. The current
TL;DR pipeline batches 2-3 stories per call and runs ~20 calls per dashboard generation.
At 60s/call that's **~20 minutes** just for TL;DR, vs ~60s total with Groq today.

### Model format concerns

Qwen3-4B-Thinking-2507 is a "thinking" model — it outputs `<think>...</think><answer>...</answer>`
blocks. The prompts ask for flat JSON. The thinking block may appear before/instead of JSON,
making parsing unreliable. A non-thinking variant (e.g. `Qwen/Qwen3-4B-Instruct-2507`) or
a standard instruct model would be more appropriate.

### Disk space issue

Earlier test showed: "Not enough free disk space to download the file. The expected file
size is: 3987.45 MB. The target location only has 3189.51 MB free." — `whichllm run` was
trying to download to `/home/dev/.cache/huggingface/hub/` which is on the same 75G
partition with 20G free. This may be a transient issue (other downloads in flight) or a
misconfiguration of HF cache location vs uv's isolated env.

## Options

### Option A: `whichllm snippet` → persistent llama-cpp server (Recommended if going local)

1. Run `uvx whichllm@latest snippet "Qwen/Qwen3-4B-Thinking-2507" --quant Q4_K_M`
   to get a GGUF model ID + Python snippet.
2. Download the GGUF model once.
3. Run `llama-cpp-python` server in OpenAI-compatible mode (persistent process).
4. Point the existing `_generate_with_retry()` at `http://localhost:8080/v1/chat/completions`.

**Pros**: Model loaded once, OpenAI-compatible API, can reuse existing HTTP client code.  
**Cons**: Still slow on CPU (30-90s/response). Qwen3-4B may struggle with structured JSON output.

### Option B: Use a smaller model

Use `whichllm` to find the best model that fits 4-6 GB RAM:

```
uvx whichllm@latest --cpu-only --profile general --top 5
```

This would rank viable CPU-only models. Likely candidates: Phi-3-mini (3.8B), Qwen2.5-1.5B,
or a small MoE model with low active params.

### Option C: Keep cloud LLM, change provider/model

Instead of Groq/Mistral, use a cheaper/smaller cloud model or self-hosted endpoint.
No code changes needed beyond model name in `hn_rerank.toml`.

### Option D: Hybrid — local for cluster names, cloud for TL;DR

Cluster naming: short output (2-4 words), small model sufficient → local.
TL;DR: longer output (3 bullet points), needs quality → cloud.
Split the model selection per task.

## Recommended plan

If local inference is the goal, proceed with **Option A**:

1. **Setup** — Download GGUF quantized model (Q4_K_M, ~2.4 GB) via whichllm snippet guidance.
2. **Server** — Create a `api/local_llm_server.py` that wraps `llama-cpp-python` server
   as a systemd user service (follows existing service pattern in AGENTS.md).
3. **Provider** — Add `"local"` as a new provider in `_generate_with_retry()` at
   `api/llm_utils.py:332`. When `provider == "local"`, point at `http://localhost:8080/v1/chat/completions`.
4. **Config** — Add `local_model` and `local_base_url` to `LLMConfig` in `api/config.py`.
5. **TOML** — Set `provider = "local"` in `hn_rerank.toml`.
6. **Test** — Run a dashboard generation, measure latency and output quality. Compare against
   current Mistral output.

### Required new dependency

`llama-cpp-python` in `pyproject.toml` (for the server, or use `uvx` to run server).

### Expected changes

| File | Change |
|------|--------|
| `api/config.py` | Add `LLMConfig.local_base_url: str`, `LLMConfig.local_model: str` |
| `api/llm_utils.py` | Add `"local"` provider branch in `_generate_with_retry()` |
| `api/local_llm_server.py` | NEW: systemd-managed llama.cpp server |
| `hn_rerank.toml` | Set provider + model |
| `pyproject.toml` | Add `llama-cpp-python` |
| `~/.config/systemd/user/hn_rerank_llm.service` | NEW: systemd unit |

## Open questions

1. Is 30-90s per TL;DR batch acceptable? (Currently ~2-3s with Groq.)
2. Can Qwen3-4B produce well-formed JSON reliably, especially the thinking variant?
3. Would a smaller model (Qwen2.5-1.5B, Phi-3-mini) be sufficient for TL;DR quality?
4. Is disk space actually an issue or was that a one-time conflict?
