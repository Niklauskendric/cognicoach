# CogniCoach — Bug Fix Log

Issues found and fixed while getting the whole stack running together
(Days 21-26), as opposed to each piece tested in isolation on the day
it was built.

## Fixed during Day 21-23 consolidation ("build it into one file")

### 1. Guardrails built but never wired into the live API
**Symptom:** `evaluator_node`/`critic_node` in every version of
`main.py` from Day 9 through Day 14 never called
`guardrails_client.py` at all. Only the terminal scripts
(`day13_graph.py`, `day14_graph.py`) had real input/output safety
checks — the actual FastAPI backend had none.
**Fix:** Rewired `evaluator_node` to call `check_input()` before
scoring and `check_output()` on generated feedback; `critic_node` to
`check_output()` follow-up questions. Added `input_blocked` through
state → API response → excluded from Neo4j/Pinecone writes.
**Where:** `backend/api/main.py`

### 2. Pinecone zero-vector dimension mismatch
**Symptom:** `history_router.py`'s original plan called for a
1536-dimension zero-vector (OpenAI's embedding size) for a
metadata-only filter query. This project's Pinecone index is
actually 384-dimensional (local `sentence-transformers`). A
1536-length vector against a 384-dim index gets rejected outright —
**every single call** to `GET /user/{id}/sessions/{id}/qa` would have
failed.
**Fix:** Import `EMBED_DIM` directly from `pinecone_client.py`
instead of hardcoding a number.
**Where:** `backend/api/history_router.py`

### 3. Guardrails config path assumed the old standalone folder layout
**Symptom:** `guardrails_client.py` computed its config path as
`Path(__file__).parent.parent.parent / "guardrails"` — correct for
the original Day 13 zip's file layout, wrong for the consolidated
project where `config.yml` sits right next to `guardrails_client.py`.
**Fix:** Path updated to `Path(__file__).parent`. Also renamed
`guardrails_config.yml` → `config.yml`, since NeMo's
`RailsConfig.from_path()` requires that exact filename.
**Where:** `backend/safety/guardrails_client.py`, `backend/safety/`

## Found during Day 26's integration review

### 4. LiteLLM proxy may not see your `.env` file
**Symptom (potential, not yet confirmed against a live run — flag
this if `test_integration.py`'s health check shows
`litellm_gateway_enabled: true` but every LLM call still fails):**
`backend/api/main.py` loads `.env` via `python-dotenv` inside the
FastAPI process. The LiteLLM proxy (`litellm --config
litellm_config.yaml --port 4000`) is a **separate process** — if your
shell doesn't already have `GROQ_API_KEY` and `LITELLM_MASTER_KEY`
exported, and your LiteLLM version doesn't auto-load `.env` from the
current directory, the proxy starts successfully but every request
through it fails with an auth error that's easy to misread as a Groq
API problem instead of an env-loading problem.
**Mitigation:** Before starting the proxy, explicitly export the two
vars it needs so this can't silently depend on version-specific
`.env` auto-loading behavior:

```bash
export $(grep -v '^#' .env | xargs)   # macOS/Linux
litellm --config litellm_config.yaml --port 4000
```

(Windows PowerShell: `Get-Content .env | ForEach-Object { if ($_ -match '^([^#][^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }`)

**Where:** operational — no code change, just a run-order gotcha
worth knowing before you're debugging a confusing auth error at 1am.

## What Day 26's `test_integration.py` actually catches

Running it end-to-end would have caught bugs #2 and #4 above
immediately (a Pinecone dimension error surfaces as a hard failure on
the transcript-fetch check; a LiteLLM env issue surfaces as every
single LLM-dependent check failing at once, which is a distinctive
enough pattern to point straight at the gateway rather than any
individual node). Bug #1 (guardrails not wired in) would have needed
a specific "does a malicious input actually get blocked" check,
which is exactly what section 3 of the script tests — this is
precisely the kind of bug that "everything looks fine because nobody
tried the adversarial case" produces, which is why it's a permanent
fixture in the test now rather than a one-off manual check.

## Not yet run against a live stack

This script was written and syntax-checked but **not executed
against a real running backend** — that requires live Groq/Neo4j/
Pinecone credentials and three running processes, which aren't
available in this environment. Run it yourself and report back
anything that fails — that's exactly what Day 26 is for.

## Fixed during Day 27's adversarial testing

### 5. Swallowed exception in session-lookup (both text and voice answer endpoints)
**Symptom:** `submit_answer` and `submit_voice_answer` both had:
```python
except Exception:
    pass
```
after the checkpoint lookup — any unexpected error (not just "session
not found") was silently discarded, letting execution fall through to
`graph.invoke()` anyway instead of surfacing the real problem. Found
while designing Day 27's invalid-`thread_id` test, by tracing the
actual code path rather than by reproducing a live failure.
**Fix:** Both `except Exception:` blocks now log the real error and
raise a proper `HTTPException(500, ...)` instead of silently
continuing.
**Where:** `backend/api/main.py` — `submit_answer()` and
`submit_voice_answer()`
