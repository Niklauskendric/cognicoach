"""
╔══════════════════════════════════════════════════════════════════╗
║         COGNICOACH — DAY 11 — backend/memory/pinecone_client.py  ║
║                                                                    ║
║  Pinecone RAG — episodic session memory.                         ║
║                                                                    ║
║  WHY THIS EXISTS ALONGSIDE NEO4J (Day 10):                       ║
║  Neo4j stores STRUCTURED long-term state — "backpropagation      ║
║  weakness_score = 0.8". It cannot tell you WHAT the candidate     ║
║  actually said, or HOW they were confused. It has no language.   ║
║                                                                    ║
║  Pinecone stores UNSTRUCTURED episodic memory — the actual        ║
║  question, the actual answer, the actual feedback, embedded as    ║
║  vectors. When the planner needs to generate a follow-up          ║
║  session, it can retrieve "here is exactly what happened last     ║
║  time you were asked about gradient descent" and use that real    ║
║  context instead of just a number.                                ║
║                                                                    ║
║  Neo4j  = long-term STRUCTURE (which topics, what score)          ║
║  Pinecone = long-term CONTENT   (what was actually said)          ║
║                                                                    ║
║  ★ EMBEDDINGS: LOCAL, FREE, NO API KEY — sentence-transformers    ║
║  all-MiniLM-L6-v2, runs entirely on your machine (CPU is fine).   ║
║  384 dimensions. Chosen specifically because this project has NO  ║
║  OpenAI budget — Groq (the chat LLM provider) has no embeddings   ║
║  endpoint at all, so this is the $0-forever alternative rather    ║
║  than a paid API. Quality is more than sufficient for semantic    ║
║  similarity search over short Q&A/feedback text like this.        ║
║                                                                    ║
║  FIRST RUN WILL DOWNLOAD THE MODEL (~80MB, one-time, cached       ║
║  locally afterward at ~/.cache/torch/sentence_transformers/).      ║
║                                                                    ║
║  RUN STANDALONE TO TEST:                                          ║
║    python pinecone_client.py                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = logging.getLogger("cognicoach.pinecone")

PASS = "✅"
FAIL = "❌"

EMBED_MODEL = "all-MiniLM-L6-v2"   # local, free, 384 dimensions
EMBED_DIM   = 384                   # NOTE: different from OpenAI's 1536 —
                                     # if you previously created a Pinecone
                                     # index at 1536 dims, you MUST delete
                                     # it or use a new index name, since
                                     # Pinecone indexes have a fixed dim.
INDEX_NAME  = os.environ.get("PINECONE_INDEX_NAME", "cognicoach-sessions")


class LocalEmbeddings:
    """
    Thin wrapper around sentence-transformers matching the same
    .embed_query(text) -> list[float] interface langchain_openai's
    OpenAIEmbeddings has, so the rest of this file (and anyone
    reading it) doesn't need to care which embedding backend is used.

    Loaded ONCE per process — the model stays in memory for the
    lifetime of the app, same lifecycle as the Pinecone client itself.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        logger.info(f"[embeddings] Loading local model '{model_name}' "
                    f"(first run downloads ~80MB, cached after that)...")
        self._model = SentenceTransformer(model_name)
        logger.info("[embeddings] Model loaded")

    def embed_query(self, text: str) -> list[float]:
        # normalize_embeddings=True makes cosine similarity behave
        # correctly with Pinecone's "cosine" metric — same convention
        # OpenAI's embeddings API follows internally.
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class PineconeMemory:
    """
    Manages all CogniCoach episodic (Q&A / feedback) memory in Pinecone.
    Single instance shared across the app lifetime — same pattern as
    Neo4jClient in Day 10.

    Each user gets their own Pinecone NAMESPACE (not a filter) so that
    one user's embeddings never mix into another user's retrieval,
    and per-user data can be wiped independently if needed.
    """

    def __init__(self):
        api_key = os.environ["PINECONE_API_KEY"]
        self._pc = Pinecone(api_key=api_key)
        self._embeddings = LocalEmbeddings()
        self._ensure_index_exists()
        self._index = self._pc.Index(INDEX_NAME)
        logger.info(f"[pinecone] Connected to index '{INDEX_NAME}'")

    # ════════════════════════════════════════════════════════════
    # SETUP — create the serverless index once, safe to call repeatedly
    # ════════════════════════════════════════════════════════════

    def _ensure_index_exists(self) -> None:
        existing = [i["name"] for i in self._pc.list_indexes()]
        if INDEX_NAME in existing:
            return

        logger.info(f"[pinecone] Index '{INDEX_NAME}' not found — creating "
                    f"at {EMBED_DIM} dimensions...")
        self._pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not self._pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        logger.info(f"[pinecone] Index '{INDEX_NAME}' created and ready")

    # ════════════════════════════════════════════════════════════
    # WRITE 1: Embed + store a single Q&A exchange (per question)
    # ════════════════════════════════════════════════════════════

    def store_qa_memory(
        self,
        user_id: str,
        session_id: str,
        question: str,
        answer: str,
        feedback: str,
        overall_score: float,
        weak_topics: list[str],
    ) -> None:
        """
        Called from graph_updater_node, once per answered question —
        exactly where Neo4j's update_topic_scores() is also called.
        """
        text_to_embed = (
            f"Question: {question}\n"
            f"Candidate's answer: {answer}\n"
            f"Feedback given: {feedback}\n"
            f"Weak topics identified: {', '.join(weak_topics) if weak_topics else 'none'}"
        )

        vector = self._embeddings.embed_query(text_to_embed)

        self._index.upsert(
            vectors=[{
                "id": f"qa_{session_id}_{uuid.uuid4().hex[:8]}",
                "values": vector,
                "metadata": {
                    "type": "qa_exchange",
                    "session_id": session_id,
                    "question": question[:500],
                    "answer": answer[:500],
                    "feedback": feedback[:500],
                    "overall_score": overall_score,
                    "weak_topics": weak_topics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }],
            namespace=user_id,
        )
        logger.info(f"[pinecone] Stored Q&A memory for '{user_id}' "
                    f"(score={overall_score})")

    # ════════════════════════════════════════════════════════════
    # WRITE 2: Embed + store a full session summary (end of session)
    # ════════════════════════════════════════════════════════════

    def store_session_summary(
        self,
        user_id: str,
        session_id: str,
        avg_score: float,
        weak_topics: list[str],
        num_questions: int,
    ) -> None:
        """
        Called from graph_updater_node at session end, right next to
        Neo4j's save_session() call.
        """
        text_to_embed = (
            f"Session summary. Average score: {avg_score}/10 across "
            f"{num_questions} questions. Weak topics this session: "
            f"{', '.join(weak_topics) if weak_topics else 'none'}."
        )
        vector = self._embeddings.embed_query(text_to_embed)

        self._index.upsert(
            vectors=[{
                "id": f"session_{session_id}",
                "values": vector,
                "metadata": {
                    "type": "session_summary",
                    "session_id": session_id,
                    "avg_score": avg_score,
                    "weak_topics": weak_topics,
                    "num_questions": num_questions,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }],
            namespace=user_id,
        )
        logger.info(f"[pinecone] Stored session summary for '{user_id}'")

    # ════════════════════════════════════════════════════════════
    # READ: Retrieve relevant past context (used by planner_node)
    # ════════════════════════════════════════════════════════════

    def fetch_relevant_context(
        self,
        user_id: str,
        query_topics: list[str],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Given the weak topics Neo4j just returned, embed them as a
        query and pull back the most semantically similar past Q&A
        exchanges for THIS user — the actual RAG retrieval step,
        feeding into the planner's Groq generation step.

        Returns [] gracefully for brand-new users with no history.
        """
        if not query_topics:
            return []

        query_text = f"Past struggles with: {', '.join(query_topics)}"
        query_vector = self._embeddings.embed_query(query_text)

        try:
            result = self._index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=user_id,
                include_metadata=True,
                filter={"type": {"$eq": "qa_exchange"}},
            )
        except Exception as e:
            logger.warning(f"[pinecone] Query failed (likely empty namespace): {e}")
            return []

        matches = []
        for match in result.get("matches", []):
            meta = match.get("metadata", {})
            matches.append({
                "score":       round(match.get("score", 0.0), 3),
                "question":    meta.get("question", ""),
                "answer":      meta.get("answer", ""),
                "feedback":    meta.get("feedback", ""),
                "overall_score": meta.get("overall_score", 0.0),
            })

        logger.info(f"[pinecone] Retrieved {len(matches)} relevant past "
                    f"exchanges for '{user_id}'")
        return matches


# ════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON
# ════════════════════════════════════════════════════════════════

_pinecone_memory: PineconeMemory | None = None


def get_pinecone_memory() -> PineconeMemory:
    global _pinecone_memory
    if _pinecone_memory is None:
        _pinecone_memory = PineconeMemory()
    return _pinecone_memory


# ════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ════════════════════════════════════════════════════════════════

def run_day11_pinecone_test():
    print("\n" + "=" * 60)
    print("  CogniCoach Day 11 — Pinecone Client Standalone Test")
    print("  (local embeddings — no OpenAI, no API key needed for this part)")
    print("=" * 60 + "\n")

    results = []

    def check(label, ok, detail=""):
        symbol = PASS if ok else FAIL
        print(f"  {symbol}  {label}")
        if detail:
            print(f"        {detail}")
        results.append(ok)

    TEST_USER = "test_user_day11"
    TEST_SESSION = "day11_test_session_001"

    print("[ 1 ]  Connection + index setup + local model load")
    try:
        client = PineconeMemory()
        check("Pinecone connected + index ready + local embed model loaded", True)
    except Exception as e:
        check("Pinecone connection", False, str(e))
        print("\n  Fix: Check PINECONE_API_KEY in .env")
        print("  Fix: Check `pip install sentence-transformers` succeeded\n")
        return
    print()

    print("[ 2 ]  Store Q&A memories")
    try:
        client.store_qa_memory(
            user_id=TEST_USER, session_id=TEST_SESSION,
            question="How does gradient descent update model weights?",
            answer="It uses the derivative of the loss to adjust weights.",
            feedback="Mention the learning rate and direction of steepest descent.",
            overall_score=4.5,
            weak_topics=["gradient descent", "learning rate"],
        )
        client.store_qa_memory(
            user_id=TEST_USER, session_id=TEST_SESSION,
            question="What is overfitting?",
            answer="When a model performs well on train but poorly on test data because it memorised noise.",
            feedback="Good — could mention regularisation as a fix.",
            overall_score=7.5,
            weak_topics=[],
        )
        check("2 Q&A memories stored", True)
    except Exception as e:
        check("Store Q&A memories", False, str(e))
    print()

    print("  (waiting 5s for index to become queryable...)")
    time.sleep(5)
    print()

    print("[ 3 ]  Store session summary")
    try:
        client.store_session_summary(
            user_id=TEST_USER, session_id=TEST_SESSION,
            avg_score=6.0, weak_topics=["gradient descent"], num_questions=2,
        )
        check("Session summary stored", True)
    except Exception as e:
        check("Store session summary", False, str(e))
    print()

    print("[ 4 ]  Retrieve relevant context")
    try:
        results_ctx = client.fetch_relevant_context(
            user_id=TEST_USER,
            query_topics=["gradient descent"],
            top_k=3,
        )
        ok = len(results_ctx) > 0
        check("Retrieved relevant past exchange(s)", ok,
              f"Top match: {results_ctx[0]['question'][:50]}..." if ok else "No matches")
    except Exception as e:
        check("Retrieve context", False, str(e))
    print()

    print("[ 5 ]  Empty namespace handling (new user)")
    try:
        empty_ctx = client.fetch_relevant_context(
            user_id="brand_new_user_never_seen",
            query_topics=["transformers"],
            top_k=3,
        )
        check("Empty namespace returns [] without crashing", empty_ctx == [])
    except Exception as e:
        check("Empty namespace handling", False, str(e))

    print()
    print("=" * 60)
    passed, total = sum(results), len(results)
    if passed == total:
        print(f"\n  {PASS}  All {total} tests passed!")
        print("  Pinecone RAG memory is fully working — $0 spent on embeddings.")
        print("  Now run: python day11_graph.py for the full integrated test.")
    else:
        print(f"\n  {FAIL}  {total - passed} tests failed.")
        print("  Fix errors above before running the full graph.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_day11_pinecone_test()
