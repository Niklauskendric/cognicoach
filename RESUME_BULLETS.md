# CogniCoach — Resume / Portfolio Bullets

No live deployment — every bullet below points at the GitHub repo and
(once recorded) the Day 25 demo video, never a live URL. Pick 2-4 for
a resume, use the fuller versions for a portfolio site or LinkedIn
project post.

## One-liner (resume project title line)

> **CogniCoach** — Adaptive AI interview coach with long-term memory
> and real-time voice input · LangGraph, FastAPI, Neo4j, Pinecone,
> Groq · [GitHub] · [Demo video]

## Resume bullets (pick a few, ~1 line each)

- Built a multi-agent LangGraph system (planner → voice → evaluator →
  critic → graph_updater) with checkpointed state, enabling the
  interview to pause for real human input mid-graph and resume exactly
  where it left off
- Designed a dual-database long-term memory architecture — Neo4j for
  per-user weak-topic tracking, Pinecone for RAG context and full Q&A
  history — so question generation adapts based on prior sessions, not
  just the current one
- Integrated real-time speech-to-text (Groq Whisper) and implemented
  input/output safety guardrails (NeMo Guardrails) that block
  adversarial input before it reaches the scoring pipeline or gets
  written to long-term memory
- Wrote an end-to-end integration test suite covering the full request
  path (LLM gateway → API → graph → databases) plus a dedicated
  adversarial test suite (empty/malformed input, non-English answers,
  retry-loop stress testing), catching and fixing 3 real integration
  bugs invisible to any single component's unit tests
- Shipped a Streamlit dashboard (session interview flow, weak-topics
  visualisation, score-trend history, optional streaming feedback)
  against a FastAPI backend, with all state management handled
  through `st.session_state` across Streamlit's rerun-on-every-
  interaction model

## Longer portfolio-site version (a paragraph)

> CogniCoach is an adaptive AI interview coach that asks real
> technical questions, scores answers across three dimensions, and
> follows up automatically when an answer needs more depth. Under the
> hood it's a LangGraph state machine with a Neo4j knowledge graph
> tracking per-user weak topics across sessions and a Pinecone vector
> store providing RAG context and full session history — so the
> planner genuinely references what you got wrong three sessions ago,
> not just this one. It supports real voice input via Groq Whisper,
> runs every answer through NeMo Guardrails before scoring, and is
> backed by an end-to-end + adversarial test suite that caught three
> real integration bugs during development (documented in
> `BUGFIXES.md`). Frontend is a Streamlit dashboard; backend is
> FastAPI behind a LiteLLM gateway for model fallback. Runs fully
> locally — no cloud deployment, by design, to keep the free-tier
> footprint at zero. [GitHub repo] · [Demo video]

## Interview talking points (things worth being ready to explain)

- **Why LangGraph over a plain loop:** the interview needs to pause
  for human input, potentially for minutes, then resume with full
  prior state — `interrupt_before` + a checkpointer gives that for
  free instead of hand-rolling session state.
- **Why two databases, not one:** Neo4j models the *relationship*
  between a user and their weak topics naturally as a graph; Pinecone
  handles semantic similarity search over free-text Q&A — different
  access patterns, different right tools.
- **The honest streaming limitation:** Day 20's "streaming feedback"
  is a typewriter effect over a complete result, not token-level model
  streaming — because the evaluator uses structured output, which
  returns one parsed object, not a token stream. Know this cold; it's
  a common follow-up question and the honest answer is more impressive
  than a vague overclaim would be.
- **A real bug you found and fixed:** guardrails were built (Day 13)
  but never actually wired into the live API until later consolidation
  — only the terminal test scripts exercised them. Good story about
  the gap between "a feature exists in the codebase" and "a feature is
  actually reachable," and why integration testing (Day 26-27) matters
  even after every individual piece is unit-tested.
- **Why no deployment:** a deliberate scope decision, not a limitation
  you ran out of time for — say so plainly if asked.
