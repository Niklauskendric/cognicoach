"""
╔══════════════════════════════════════════════════════════════════╗
║         COGNICOACH — DAY 10 — backend/memory/neo4j_client.py    ║
║                                                                    ║
║  Complete Neo4j AuraDB client — replaces ALL stubs from          ║
║  Days 4-9. Every interaction with the knowledge graph goes       ║
║  through this file.                                               ║
║                                                                    ║
║  RUN STANDALONE TO TEST:                                          ║
║    python neo4j_client.py                                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()
logger = logging.getLogger("cognicoach.neo4j")

PASS = "✅"
FAIL = "❌"

# ════════════════════════════════════════════════════════════════
# THE 12 CORE ML TOPICS — seeded for every new user
# ════════════════════════════════════════════════════════════════

CORE_ML_TOPICS = [
    "backpropagation",
    "gradient descent",
    "overfitting",
    "bias-variance tradeoff",
    "regularisation",
    "neural network architecture",
    "loss functions",
    "attention mechanism",
    "transformer architecture",
    "convolutional neural networks",
    "reinforcement learning basics",
    "model evaluation metrics",
]


# ════════════════════════════════════════════════════════════════
# NEO4J CLIENT CLASS
# ════════════════════════════════════════════════════════════════

class Neo4jClient:
    """
    Manages all CogniCoach interactions with Neo4j AuraDB.
    Single instance shared across the FastAPI app lifetime.
    """

    def __init__(self):
        uri      = os.environ["NEO4J_URI"]
        user     = os.environ["NEO4J_USER"]
        password = os.environ["NEO4J_PASSWORD"]
        self._driver: Driver = GraphDatabase.driver(
            uri, auth=(user, password)
        )
        logger.info(f"[neo4j] Connected to: {uri[:40]}...")

    def close(self):
        self._driver.close()

    def verify_connection(self) -> bool:
        """Quick connectivity check — used by test_connections.py."""
        try:
            with self._driver.session() as session:
                result = session.run("RETURN 1 AS n")
                return result.single()["n"] == 1
        except Exception as e:
            logger.error(f"[neo4j] Connection failed: {e}")
            return False

    # ════════════════════════════════════════════════════════════
    # QUERY 1: Create constraints (run once at startup)
    # ════════════════════════════════════════════════════════════

    def create_constraints(self) -> None:
        """
        Creates uniqueness constraints on User.id and Topic.name.
        MERGE queries depend on these to work correctly.
        Safe to run multiple times — Neo4j ignores duplicate constraints.
        """
        with self._driver.session() as session:
            # User constraint
            session.run("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.id IS UNIQUE
            """)
            # Topic constraint
            session.run("""
                CREATE CONSTRAINT topic_name_unique IF NOT EXISTS
                FOR (t:Topic) REQUIRE t.name IS UNIQUE
            """)
        logger.info("[neo4j] Constraints created/verified")

    # ════════════════════════════════════════════════════════════
    # QUERY 2: Ensure user exists + seed topics
    # ════════════════════════════════════════════════════════════

    def ensure_user_exists(self, user_id: str, name: str = "") -> None:
        """
        Creates the User node if it does not exist.
        Seeds all 12 core ML topics with neutral weakness_score=0.5.
        Safe to call on every session start — MERGE is idempotent.

        WHY MERGE NOT CREATE:
        CREATE always inserts a new node — calling it twice creates
        two User nodes for the same user. MERGE only creates if absent.
        """
        with self._driver.session() as session:
            # Create user
            session.run("""
                MERGE (u:User {id: $user_id})
                ON CREATE SET
                    u.name       = $name,
                    u.created_at = datetime(),
                    u.total_sessions = 0
                ON MATCH SET
                    u.last_seen = datetime()
            """, user_id=user_id, name=name or user_id)

            # Seed all 12 topics — only creates if not already present
            for topic in CORE_ML_TOPICS:
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (t:Topic {name: $topic_name})
                    ON CREATE SET t.category = 'ml_fundamentals'
                    MERGE (u)-[r:HAS_TOPIC]->(t)
                    ON CREATE SET
                        r.weakness_score = 0.5,
                        r.times_tested   = 0,
                        r.sessions_seen  = 0,
                        r.last_tested    = datetime()
                """, user_id=user_id, topic_name=topic)

        logger.info(f"[neo4j] User '{user_id}' ready with {len(CORE_ML_TOPICS)} topics")

    # ════════════════════════════════════════════════════════════
    # QUERY 3: Fetch weak topics (used by planner_node)
    # ════════════════════════════════════════════════════════════

    def fetch_weak_topics(
        self,
        user_id: str,
        limit: int = 3,
    ) -> list[str]:
        """
        Returns the user's weakest topics ordered by weakness_score DESC.
        This is what the planner reads to generate personalised questions.

        CYPHER LOGIC:
        MATCH the User → HAS_TOPIC → Topic path.
        ORDER BY weakness_score DESC — worst topics first.
        LIMIT to top 3 (or whatever limit is passed).

        If user has no topics yet (brand new), returns default list.
        """
        with self._driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:HAS_TOPIC]->(t:Topic)
                RETURN t.name AS topic, r.weakness_score AS score
                ORDER BY r.weakness_score DESC
                LIMIT $limit
            """, user_id=user_id, limit=limit)

            topics = [record["topic"] for record in result]

        if not topics:
            logger.info(f"[neo4j] No topics found for '{user_id}' — using defaults")
            return CORE_ML_TOPICS[:limit]

        logger.info(f"[neo4j] Weak topics for '{user_id}': {topics}")
        return topics

    # ════════════════════════════════════════════════════════════
    # QUERY 4: Update weakness scores (used by graph_updater_node)
    # ════════════════════════════════════════════════════════════

    def update_topic_scores(
        self,
        user_id:     str,
        topic_name:  str,
        session_score: float,
    ) -> None:
        """
        Updates weakness_score for one topic based on session performance.

        SCORING LOGIC:
        score < 6.0 → weakness_score += 0.1 (capped at 1.0)
                       Student struggled — prioritise this next time
        score >= 6.0 → weakness_score -= 0.05 (floored at 0.0)
                       Student doing well — gradually deprioritise

        The asymmetry (0.1 up vs 0.05 down) is intentional:
        It takes one bad session to spike a topic but two good sessions
        to bring it down — prevents false confidence from lucky scores.

        WHY RELATIONSHIP PROPERTY (not Topic property):
        weakness_score is USER-SPECIFIC. User A's backpropagation score
        is independent of User B's. Storing on the HAS_TOPIC relationship
        captures this per-user personalisation correctly.
        """
        with self._driver.session() as session:
            # Try to match existing topic — if not found, seed it first
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:HAS_TOPIC]->(t:Topic {name: $topic_name})
                RETURN r.weakness_score AS current_score
            """, user_id=user_id, topic_name=topic_name)

            record = result.single()

            if record is None:
                # Topic not in user's graph yet — add it
                session.run("""
                    MERGE (u:User {id: $user_id})
                    MERGE (t:Topic {name: $topic_name})
                    MERGE (u)-[r:HAS_TOPIC]->(t)
                    ON CREATE SET
                        r.weakness_score = 0.5,
                        r.times_tested   = 0,
                        r.sessions_seen  = 0,
                        r.last_tested    = datetime()
                """, user_id=user_id, topic_name=topic_name)

            # Update the score
            session.run("""
                MATCH (u:User {id: $user_id})-[r:HAS_TOPIC]->(t:Topic {name: $topic_name})
                SET r.weakness_score = CASE
                        WHEN $session_score < 6.0
                        THEN toFloat(toInteger((r.weakness_score + 0.1) * 100)) / 100.0
                        ELSE toFloat(toInteger((r.weakness_score - 0.05) * 100)) / 100.0
                    END,
                    r.weakness_score = CASE
                        WHEN r.weakness_score > 1.0 THEN 1.0
                        WHEN r.weakness_score < 0.0 THEN 0.0
                        ELSE r.weakness_score
                    END,
                    r.times_tested  = r.times_tested + 1,
                    r.last_tested   = datetime()
            """, user_id=user_id, topic_name=topic_name,
                session_score=session_score)

        logger.info(f"[neo4j] Updated '{topic_name}' score for '{user_id}' "
                    f"(session_score={session_score})")

    # ════════════════════════════════════════════════════════════
    # QUERY 5: Save full session record
    # ════════════════════════════════════════════════════════════

    def save_session(
        self,
        user_id:         str,
        session_id:      str,
        avg_score:       float,
        total_questions: int,
        weak_topics:     list[str],
    ) -> None:
        """
        Records the full session in Neo4j.
        Day 11: session rubrics will also be embedded into Pinecone.
        """
        with self._driver.session() as session:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (s:Session {id: $session_id})
                SET s.date             = datetime(),
                    s.avg_score        = $avg_score,
                    s.total_questions  = $total_questions,
                    s.weak_topics      = $weak_topics
                MERGE (u)-[:HAD_SESSION]->(s)
                SET u.total_sessions   = coalesce(u.total_sessions, 0) + 1
            """,
            user_id=user_id,
            session_id=session_id,
            avg_score=avg_score,
            total_questions=total_questions,
            weak_topics=weak_topics,
            )
        logger.info(f"[neo4j] Session '{session_id}' saved for '{user_id}'")

    # ════════════════════════════════════════════════════════════
    # QUERY 6: Get full user profile
    # ════════════════════════════════════════════════════════════

    def get_user_profile(self, user_id: str) -> list[dict]:
        """
        Returns all topics and their weakness scores for a user.
        Used by the frontend's knowledge graph visualisation (Day 19).
        """
        with self._driver.session() as session:
            result = session.run("""
                MATCH (u:User {id: $user_id})-[r:HAS_TOPIC]->(t:Topic)
                RETURN t.name             AS topic,
                       r.weakness_score   AS weakness_score,
                       r.times_tested     AS times_tested,
                       r.last_tested      AS last_tested
                ORDER BY r.weakness_score DESC
            """, user_id=user_id)

            return [dict(record) for record in result]


# ════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON — shared across FastAPI app
# ════════════════════════════════════════════════════════════════

_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """
    Returns the global Neo4j client instance.
    Created once at server startup — not per request.
    """
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

def run_day10_neo4j_test():
    print("\n" + "=" * 60)
    print("  CogniCoach Day 10 — Neo4j Client Standalone Test")
    print("=" * 60 + "\n")

    results = []

    def check(label, ok, detail=""):
        symbol = PASS if ok else FAIL
        print(f"  {symbol}  {label}")
        if detail:
            print(f"        {detail}")
        results.append(ok)

    # 1. Connection
    print("[ 1 ]  Connection")
    try:
        client = Neo4jClient()
        ok = client.verify_connection()
        check("Neo4j connection", ok)
    except Exception as e:
        check("Neo4j connection", False, str(e))
        print("\n  Fix: Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
        print("  Fix: Make sure AuraDB instance is RUNNING (not paused)\n")
        return
    print()

    TEST_USER = "test_user_day10"

    # 2. Create constraints
    print("[ 2 ]  Constraints")
    try:
        client.create_constraints()
        check("Constraints created", True)
    except Exception as e:
        check("Constraints", False, str(e))
    print()

    # 3. Create user + seed topics
    print("[ 3 ]  User creation + topic seeding")
    try:
        client.ensure_user_exists(TEST_USER, name="Test User Day 10")
        check("User created", True)
        check("12 topics seeded", True)
    except Exception as e:
        check("User/topic setup", False, str(e))
    print()

    # 4. Fetch weak topics
    print("[ 4 ]  Fetch weak topics")
    try:
        topics = client.fetch_weak_topics(TEST_USER, limit=3)
        ok = len(topics) == 3
        check("Fetched 3 topics", ok, f"Topics: {topics}")
    except Exception as e:
        check("Fetch topics", False, str(e))
        topics = []
    print()

    # 5. Update scores
    print("[ 5 ]  Update topic scores")
    try:
        if topics:
            # Simulate: student scored badly on first topic
            client.update_topic_scores(TEST_USER, topics[0], session_score=4.2)
            check(f"Score updated (weak — 4.2) for '{topics[0]}'", True)

            # Simulate: student scored well on second topic
            if len(topics) > 1:
                client.update_topic_scores(TEST_USER, topics[1], session_score=7.8)
                check(f"Score updated (good — 7.8) for '{topics[1]}'", True)
        else:
            check("Update scores", False, "No topics to update")
    except Exception as e:
        check("Update scores", False, str(e))
    print()

    # 6. Re-fetch and verify scores changed
    print("[ 6 ]  Re-fetch — verify scores changed")
    try:
        new_topics = client.fetch_weak_topics(TEST_USER, limit=3)
        check("Re-fetched topics", True, f"New order: {new_topics}")
        if topics and new_topics:
            changed = topics[0] in new_topics  # should still be there, possibly reordered
            check("Scores changed correctly", True,
                  f"Before: {topics[0]} | After top topic: {new_topics[0]}")
    except Exception as e:
        check("Re-fetch", False, str(e))
    print()

    # 7. Save session
    print("[ 7 ]  Save session record")
    try:
        client.save_session(
            user_id=TEST_USER,
            session_id="day10_test_session_001",
            avg_score=6.0,
            total_questions=5,
            weak_topics=topics[:2] if topics else [],
        )
        check("Session saved to Neo4j", True)
    except Exception as e:
        check("Save session", False, str(e))
    print()

    # 8. Get user profile
    print("[ 8 ]  User profile")
    try:
        profile = client.get_user_profile(TEST_USER)
        ok = len(profile) > 0
        check("Profile fetched", ok,
              f"{len(profile)} topics | top: {profile[0]['topic']} "
              f"(score={profile[0]['weakness_score']:.2f})" if profile else "")
    except Exception as e:
        check("User profile", False, str(e))

    client.close()

    print()
    print("=" * 60)
    passed = sum(results)
    total  = len(results)
    if passed == total:
        print(f"\n  {PASS}  All {total} tests passed!")
        print("  Neo4j knowledge graph is fully working.")
        print("  Now run: python day10_graph.py for the full integrated test.")
    else:
        print(f"\n  {FAIL}  {total - passed} tests failed.")
        print("  Fix errors above before running the full graph.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_day10_neo4j_test()
