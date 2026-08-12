"""
╔══════════════════════════════════════════════════════════════════╗
║   COGNICOACH — DAY 18 — backend/api/weak_topics_router.py         ║
║                                                                    ║
║  One new endpoint for the Streamlit weak-topics dashboard:        ║
║    GET /user/{user_id}/topics                                     ║
║                                                                    ║
║  Wraps Neo4jClient.get_user_profile(), which already existed      ║
║  since Day 10/11 for exactly this purpose. No new Neo4j logic —   ║
║  only the HTTP layer was missing.                                 ║
║                                                                    ║
║  HOW TO WIRE THIS IN — add 2 lines to backend/api/main.py:        ║
║                                                                    ║
║    from backend.api.weak_topics_router import weak_topics_router  ║
║    app.include_router(weak_topics_router)                          ║
║                                                                    ║
║  Put those two lines anywhere after `app = FastAPI(...)` is        ║
║  defined (e.g. right after the CORS middleware block).            ║
╚══════════════════════════════════════════════════════════════════╝
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.memory.neo4j_client import get_neo4j_client

weak_topics_router = APIRouter()


class TopicProfile(BaseModel):
    topic: str
    weakness_score: float
    times_tested: int
    last_tested: str | None = None


class UserTopicsResponse(BaseModel):
    user_id: str
    topics: list[TopicProfile]


@weak_topics_router.get("/user/{user_id}/topics", response_model=UserTopicsResponse)
async def get_user_topics(user_id: str):
    """
    Returns every topic in the user's knowledge graph with its current
    weakness_score (0.0 = strong, 1.0 = weak), times_tested count, and
    last_tested timestamp — ordered weakest-first (same ordering
    get_user_profile's Cypher query already does).
    """
    neo4j = get_neo4j_client()
    try:
        profile = neo4j.get_user_profile(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch topic profile: {e}")

    if not profile:
        # New user who hasn't started a session yet — 404 is more honest
        # than returning an empty list, so the frontend can show a
        # "take a session first" message instead of an empty chart.
        raise HTTPException(
            status_code=404,
            detail=f"No topic data for user '{user_id}' yet — start a session first.",
        )

    topics = [
        TopicProfile(
            topic=row["topic"],
            weakness_score=row["weakness_score"],
            times_tested=row["times_tested"],
            last_tested=str(row["last_tested"]) if row.get("last_tested") else None,
        )
        for row in profile
    ]
    return UserTopicsResponse(user_id=user_id, topics=topics)
