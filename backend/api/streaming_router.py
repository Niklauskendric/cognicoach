"""
╔══════════════════════════════════════════════════════════════════╗
║   COGNICOACH — DAY 20 — backend/api/streaming_router.py           ║
║                                                                    ║
║  ONE new endpoint, purely additive:                                ║
║    POST /session/answer/stream                                    ║
║                                                                    ║
║  HONEST LIMITATION — READ THIS FIRST:                              ║
║  The evaluator node uses                                          ║
║      llm.with_structured_output(EvalRubric, method="function_calling")║
║  Structured-output calls return a parsed Pydantic object, not a    ║
║  token stream — there is no partial EvalRubric to stream. So this  ║
║  endpoint does NOT stream the LLM's actual generation live. It     ║
║  runs the real graph.invoke() exactly like /session/answer (Day 9),║
║  gets the complete rubric, and THEN streams rubric.feedback back   ║
║  to the client word-by-word over Server-Sent Events — a UI         ║
║  "typewriter" effect, not a token-level model stream.               ║
║                                                                    ║
║  IF YOU WANT A TRUE TOKEN STREAM instead: call the evaluator LLM a ║
║  second time WITHOUT structured output (plain .astream(msgs)) just ║
║  to regenerate the feedback text conversationally, and stream      ║
║  those real tokens. That costs one extra LLM call per answer.      ║
║  This file keeps the simpler, single-call version since Day 20 is  ║
║  marked optional in the plan and the UX payoff is the same either  ║
║  way — the typewriter effect is what a user actually notices.       ║
║                                                                    ║
║  HOW TO WIRE THIS IN — add 2 lines to backend/api/main.py:         ║
║    from backend.api.streaming_router import streaming_router       ║
║    app.include_router(streaming_router)                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

logger = logging.getLogger("cognicoach")

streaming_router = APIRouter()

WORD_DELAY_SECONDS = 0.045  # tune for demo pacing — faster/slower typewriter effect


class StreamAnswerRequest(BaseModel):
    thread_id: str
    answer: str = Field(..., min_length=1)


def _sse(payload: dict) -> str:
    """Formats one Server-Sent Events frame. Two trailing newlines are
    required by the SSE spec to mark the end of an event."""
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_answer_events(request: Request, thread_id: str, answer: str):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    yield _sse({"type": "status", "message": "Evaluating your answer..."})

    try:
        # graph.invoke() is synchronous (LangGraph's MemorySaver checkpointer
        # is sync-only here) — run it in a thread so it doesn't block the
        # event loop while other requests are being served.
        state = await asyncio.to_thread(
            graph.invoke,
            Command(resume={"user_answer": answer}),
            config,
        )
    except Exception as e:
        logger.error(f"[/session/answer/stream] Graph error: {e}")
        yield _sse({"type": "error", "detail": f"Failed to process answer: {e}"})
        return

    rubric = state.get("rubric")
    if rubric is None:
        yield _sse({"type": "error", "detail": "Evaluator did not return a rubric"})
        return

    yield _sse({
        "type": "scores",
        "depth_score": rubric.depth_score,
        "accuracy_score": rubric.accuracy_score,
        "communication_score": rubric.communication_score,
        "overall_score": rubric.overall_score,
    })

    # ── typewriter the feedback text word by word ──
    words = rubric.feedback.split(" ")
    for i in range(len(words)):
        partial = " ".join(words[: i + 1])
        yield _sse({"type": "token", "partial_feedback": partial})
        await asyncio.sleep(WORD_DELAY_SECONDS)

    session_complete = state.get("session_complete", False)
    yield _sse({
        "type": "done",
        "rubric": {
            "overall_score": rubric.overall_score,
            "depth_score": rubric.depth_score,
            "accuracy_score": rubric.accuracy_score,
            "communication_score": rubric.communication_score,
            "feedback": rubric.feedback,
            "weak_topics": rubric.weak_topics,
        },
        "needs_followup": state.get("needs_followup", False),
        "retry_count": state.get("retry_count", 0),
        "next_question": None if session_complete else state.get("current_question", ""),
        "question_index": state.get("question_index", 0),
        "session_complete": session_complete,
        "input_blocked": state.get("input_blocked", False),
    })


@streaming_router.post("/session/answer/stream")
async def stream_answer(payload: StreamAnswerRequest, request: Request):
    if not hasattr(request.app.state, "graph"):
        raise HTTPException(status_code=503, detail="Graph not ready yet.")

    return StreamingResponse(
        _stream_answer_events(request, payload.thread_id, payload.answer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx etc.) so tokens arrive live
        },
    )
