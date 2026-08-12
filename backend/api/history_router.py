"""
╔══════════════════════════════════════════════════════════════════╗
║   COGNICOACH — DAY 19 — backend/api/history_router.py             ║
║                                                                    ║
║  Two new endpoints for the Streamlit history + analytics tab:     ║
║    GET /user/{user_id}/sessions                                   ║
║        — session-level list (date, avg_score, weak_topics) from   ║
║          Neo4j's :Session nodes (written since Day 10/11's         ║
║          save_session()).                                          ║
║    GET /user/{user_id}/sessions/{session_id}/qa                    ║
║        — every individual Q&A exchange for one session, pulled     ║
║          from Pinecone (written since Day 11's store_qa_memory()). ║
║                                                                    ║
║  ★ FIX vs the plan doc: the zero-vector used for the Pinecone      ║
║  metadata-only filter query must match EMBED_DIM (384 —            ║
║  sentence-transformers, per pinecone_client.py's own comment       ║
║  explaining it's deliberately NOT OpenAI's 1536). A hardcoded      ║
║  1536-length vector would raise a dimension-mismatch error from    ║
║  Pinecone on every single call — imported directly from            ║
║  pinecone_client so it can never drift out of sync again.          ║
║                                                                    ║
║  NOTE ON IMPLEMENTATION: Neo4jClient and PineconeMemory (Days      ║
║  10-11) don't expose public methods for "list all sessions" or     ║
║  "list all Q&A for one session_id" — only the single-topic /       ║
║  single-similarity-search methods planner_node and                 ║
║  graph_updater_node needed. Rather than editing those classes,     ║
║  this router runs the two read-only queries directly against the  ║
║  same driver/index instances via get_neo4j_client() /              ║
║  get_pinecone_memory() (the existing singletons). If you'd rather  ║
║  keep DB access encapsulated, move query_session_history() and     ║
║  query_session_qa() below into Neo4jClient / PineconeMemory as      ║
║  proper methods — the Cypher/Pinecone-filter logic is unchanged.   ║
║                                                                    ║
║  HOW TO WIRE THIS IN — add 2 lines to backend/api/main.py:         ║
║    from backend.api.history_router import history_router           ║
║    app.include_router(history_router)                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.memory.neo4j_client import get_neo4j_client
from backend.memory.pinecone_client import get_pinecone_memory, EMBED_DIM

history_router = APIRouter()


# ════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════

class SessionSummary(BaseModel):
    session_id: str
    date: str | None = None
    avg_score: float
    total_questions: int
    weak_topics: list[str] = []


class SessionHistoryResponse(BaseModel):
    user_id: str
    sessions: list[SessionSummary]


class QAExchange(BaseModel):
    question: str
    answer: str
    feedback: str
    overall_score: float
    weak_topics: list[str] = []
    timestamp: str | None = None


class SessionQAResponse(BaseModel):
    session_id: str
    exchanges: list[QAExchange]


# ════════════════════════════════════════════════════════════════
# NEO4J — session-level history
# ════════════════════════════════════════════════════════════════

def query_session_history(user_id: str) -> list[dict]:
    """
    Reads every :Session node the user HAD_SESSION, newest first.
    Mirrors the fields written by Neo4jClient.save_session() (Day 10/11).
    """
    client = get_neo4j_client()
    with client._driver.session() as session:  # reusing the existing driver instance
        result = session.run(
            """
            MATCH (u:User {id: $user_id})-[:HAD_SESSION]->(s:Session)
            RETURN s.id AS session_id,
                   toString(s.date) AS date,
                   s.avg_score AS avg_score,
                   s.total_questions AS total_questions,
                   s.weak_topics AS weak_topics
            ORDER BY s.date DESC
            """,
            user_id=user_id,
        )
        return [dict(record) for record in result]


@history_router.get("/user/{user_id}/sessions", response_model=SessionHistoryResponse)
async def get_session_history(user_id: str):
    try:
        rows = query_session_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session history: {e}")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No completed sessions for user '{user_id}' yet.",
        )

    sessions = [
        SessionSummary(
            session_id=r["session_id"],
            date=r["date"],
            avg_score=r["avg_score"],
            total_questions=r["total_questions"],
            weak_topics=r["weak_topics"] or [],
        )
        for r in rows
    ]
    return SessionHistoryResponse(user_id=user_id, sessions=sessions)


# ════════════════════════════════════════════════════════════════
# PINECONE — per-session Q&A detail
# ════════════════════════════════════════════════════════════════

def query_session_qa(user_id: str, session_id: str, top_k: int = 20) -> list[dict]:
    """
    Pinecone's query() always needs a vector, even for a pure metadata
    filter lookup. We pass an all-zero vector of length EMBED_DIM —
    cosine similarity ranking is meaningless here (we only care about
    the `filter`; every Q&A exchange for this session_id/type matches
    regardless of similarity order), but the dimension MUST match the
    index's configured dimension or Pinecone rejects the call outright.
    """
    memory = get_pinecone_memory()
    zero_vector = [0.0] * EMBED_DIM

    try:
        result = memory._index.query(
            vector=zero_vector,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True,
            filter={
                "type": {"$eq": "qa_exchange"},
                "session_id": {"$eq": session_id},
            },
        )
    except Exception as e:
        logger.warning(f"[history_router] Pinecone query failed: {e}")
        return []

    exchanges = []
    for match in result.get("matches", []):
        meta = match.get("metadata", {})
        exchanges.append({
            "question": meta.get("question", ""),
            "answer": meta.get("answer", ""),
            "feedback": meta.get("feedback", ""),
            "overall_score": meta.get("overall_score", 0.0),
            "weak_topics": meta.get("weak_topics", []),
            "timestamp": meta.get("timestamp"),
        })

    # Pinecone doesn't guarantee insertion order back — sort by timestamp
    # so the transcript reads in the order questions were actually asked.
    exchanges.sort(key=lambda e: e["timestamp"] or "")
    return exchanges


@history_router.get(
    "/user/{user_id}/sessions/{session_id}/qa", response_model=SessionQAResponse
)
async def get_session_qa(user_id: str, session_id: str):
    try:
        rows = query_session_qa(user_id, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Q&A detail: {e}")

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No Q&A exchanges found for session '{session_id}'.",
        )

    return SessionQAResponse(
        session_id=session_id,
        exchanges=[QAExchange(**r) for r in rows],
    )
