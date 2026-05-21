# Technical Debt

## High Priority

### 1. Split Configuration Model

The project currently has two config paths:
- `generate_html.py` parses top-level runtime values from `[hn_rerank]`
- `api/constants.py` separately loads nested tuned parameters from `[hn_rerank.<section>]`

Impact:
- documentation drift is easy
- runtime precedence is hard to explain
- tuning and production execution do not share a single typed config object

Target:
- load config once
- validate it once
- pass an explicit config object through the pipeline

### 2. Global Patching in Evaluation and Promotion

The tuning/promotion path still patches `api.rerank` module globals during evaluation.

Impact:
- harder to reason about with threaded CV and multi-job optimization
- reproducibility depends on implicit global state
- type signatures are harder to keep tight

Target:
- pass scoring and clustering parameters into evaluation and ranking explicitly
- remove `patch.multiple(...)` from the steady-state tuning path

### 3. Runtime Contract Drift Prevention

Recently fixed drift:
- the UI match badge is documented as `max_cluster_score`, while `knn_score` remains a debug diagnostic
- README examples use nested TOML sections for the values loaded by `api/constants.py`
- architecture docs now describe `public/index.html` and `public/clusters.html`, not root-level artifacts
- scheduled open-index usage is documented as `[hn_rerank.archive].open_index_enabled = true`

Impact:
- new contributors get the wrong mental model
- integration tests and docs can fail for non-code reasons

Target:
- treat docs and integration tests as part of the public runtime contract
- keep one source of truth for score naming and config examples
- keep demo docs as executable workflows instead of stale captured output dumps

## Medium Priority

### 4. Operational Portability

Current automation assumes:
- fixed repo path `/home/dev/hn_rerank`
- systemd user services copied by hand
- ad hoc logs and artifacts landing in the repo root

Impact:
- awkward to move between machines
- harder to distinguish source from runtime artifacts

Target:
- centralize artifact/log output
- reduce hardcoded absolute paths
- keep wrappers thin and parameterized

### 5. Mixed Responsibilities in `generate_html.py`

`generate_html.py` is still both:
- composition root
- orchestration layer
- part of the business logic host
- part of the rendering layer

Impact:
- behavior is harder to test in isolation
- config and policy logic are coupled to output concerns

Target:
- push ranking, selection, and rendering helpers into separate modules with narrower interfaces
- longer term, reduce `generate_html.py` to CLI parsing plus pipeline invocation

### 6. Comment and Signal Coverage Gaps

Current gaps:
- favorited comments are still not first-class signals
- hidden/comment-like edge cases still collapse into invalid-story handling
- the public-mode experience is much weaker than the logged-in path

Target:
- support comment-based signals intentionally instead of treating them as invalid stories
- make signal provenance explicit in the model layer

## Lower Priority

### 7. HTML Artifact Independence

The generated HTML pages are static, but they are not fully self-contained because they load Tailwind from a CDN.

Target:
- inline or vendor CSS if true offline artifact portability matters

### 8. Logging Surface

There is a structured logging module, but the repo mostly still uses ad hoc logging/printing paths.

Target:
- converge on one logging approach for CLI runs, scheduled runs, and tuning utilities

### 9. Typed Pipeline and Policy Objects

The clean target is a pipeline with explicit domain policies and adapters:
- a single typed `AppConfig` loaded once
- explicit `RunRequest`, `RunState`, and `RunArtifacts` objects
- selection, clustering, and cache freshness rules represented as testable policy objects
- side effects isolated behind source, model, cache, renderer, and verification adapters

This is the direction for larger refactors, but it should be introduced
incrementally to avoid changing ranking behavior while moving code.

## Security Notes

- Session cookies are stored locally in `.cache/user/cookies.json`.
- This is acceptable for a single-user local workflow, but it should not be treated as secure secret storage.
- Secrets do not belong in markdown, tracked config, or demo files.

## Verification Notes

For user-visible changes, the current minimum bar is:
- relevant `uv run pytest ...` subset, or the full suite for broad behavior changes
- `uv run ruff check .`
- `uv run ty check .`
- relevant Showboat/Rodney verification when output changes are visible in the generated dashboard
