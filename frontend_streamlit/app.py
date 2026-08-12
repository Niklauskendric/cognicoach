"""
╔══════════════════════════════════════════════════════════════════╗
║      COGNICOACH — DAY 20 — frontend_streamlit/app.py              ║
║                                                                    ║
║  Adds STREAMING FEEDBACK (optional, per the plan) on top of        ║
║  Day 19's history + analytics. Last of the 6 Streamlit days —      ║
║  Docker (Day 21) and deployment (Days 22-23) come next.            ║
║                                                                    ║
║  TODAY'S GOAL (Day 20 — Streaming feedback, optional):              ║
║    ✅ Sidebar toggle: "✨ Enable streaming feedback"                ║
║    ✅ When ON, answers go to POST /session/answer/stream (SSE)     ║
║       instead of POST /session/answer — scores appear the instant  ║
║       they're computed, then feedback types out word-by-word        ║
║       (see backend_addition/streaming_router.py — wire into        ║
║       main.py, 2 lines; read its docstring for the honest           ║
║       limitation: this is a typewriter effect over the FINAL        ║
║       feedback text, not true token-level model streaming,          ║
║       because the evaluator uses structured output)                 ║
║    ✅ When OFF, falls back to Days 16-19's plain POST /session/answer║
║    ✅ Both paths converge on one shared _apply_answer_result() so   ║
║       history/dashboard refresh and next-question logic is          ║
║       identical either way — no duplicated state logic              ║
║                                                                    ║
║  Everything from Days 16-19 (interview loop, session-state          ║
║  hardening, weak-topics dashboard, history + analytics) unchanged. ║
║                                                                    ║
║  RUN (2 terminals):                                                ║
║    Terminal 1: cd backend && uvicorn api.main:app --reload --port 8000║
║    Terminal 2: cd frontend_streamlit && streamlit run app.py       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json

import pandas as pd
import requests
import streamlit as st

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CogniCoach",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.environ.get("COGNICOACH_API_URL", "http://localhost:8000")

# Mirrors backend/api/main.py's CogniCoachState.max_retries default (=2).
# Purely cosmetic here — the backend is the source of truth for when
# follow-ups actually stop; this just lets the UI show "attempt 2 of 3".
MAX_RETRIES = 2
MIN_ANSWER_CHARS = 10


# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "api_url": DEFAULT_API_URL,
        "connected": False,
        "health": None,
        "last_error": None,
        "user_id": "streamlit_test_user",

        # interview state
        "thread_id": None,
        "current_question": None,
        "question_index": None,
        "total_questions": None,
        "focus_topics": [],
        "difficulty": None,
        "used_past_context": False,
        "is_followup": False,

        # per-answer state
        "last_rubric": None,
        "last_needs_followup": False,
        "retry_count": 0,

        # session progress
        "session_complete": False,
        "answered_log": [],

        # UX guards
        "submitting": False,
        "confirm_reset": False,
        "form_key_suffix": 0,

        # ★ Day 18 — weak topics dashboard
        "topics_data": None,      # list[dict] from GET /user/{id}/topics, or None if never fetched
        "topics_error": None,     # distinguishes "no data yet" (404) from a real failure
        "topics_stale": True,     # forces a fetch next time the tab is opened

        # ★ NEW — Day 19 — history + analytics
        "sessions_data": None,        # list[dict] from GET /user/{id}/sessions
        "sessions_error": None,       # "no_data" (404) vs a real error string
        "sessions_stale": True,       # forces a refetch next time the tab is opened
        "selected_session_id": None,  # which session's transcript is being viewed
        "session_qa_data": None,      # list[dict] from GET /user/{id}/sessions/{id}/qa
        "session_qa_error": None,

        # ★ NEW — Day 20
        "streaming_enabled": False,   # sidebar toggle
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def reset_interview_state():
    """Wipes interview progress, keeps backend connection settings, and
    bumps form_key_suffix so the next answer form starts empty rather
    than showing stale leftover text. Topics data survives a reset —
    it belongs to the user_id, not the session."""
    for key in [
        "thread_id", "current_question", "question_index", "total_questions",
        "focus_topics", "difficulty", "used_past_context", "is_followup",
        "last_rubric", "last_needs_followup", "retry_count",
        "session_complete", "answered_log", "submitting", "confirm_reset",
    ]:
        del st.session_state[key]
    st.session_state.form_key_suffix += 1
    init_session_state()


# ════════════════════════════════════════════════════════════════
# API HELPERS
# ════════════════════════════════════════════════════════════════

def api_get(path: str, timeout: int = 10):
    url = st.session_state.api_url.rstrip("/") + path
    return requests.get(url, timeout=timeout)


def api_post(path: str, json_body: dict, timeout: int = 30):
    url = st.session_state.api_url.rstrip("/") + path
    return requests.post(url, json=json_body, timeout=timeout)


def _extract_detail(http_error: requests.exceptions.HTTPError) -> str:
    try:
        return http_error.response.json().get("detail", "")
    except Exception:
        return ""


def check_health():
    try:
        resp = api_get("/health")
        resp.raise_for_status()
        st.session_state.health = resp.json()
        st.session_state.connected = True
        st.session_state.last_error = None
    except requests.exceptions.ConnectionError:
        st.session_state.connected = False
        st.session_state.last_error = "Could not reach the backend. Is uvicorn running?"
    except Exception as e:
        st.session_state.connected = False
        st.session_state.last_error = f"Unexpected error: {e}"


def start_session():
    try:
        resp = api_post("/session/start", {"user_id": st.session_state.user_id})
        resp.raise_for_status()
        data = resp.json()
        st.session_state.thread_id = data["thread_id"]
        st.session_state.current_question = data["question"]
        st.session_state.question_index = data["question_index"]
        st.session_state.total_questions = data["total_questions"]
        st.session_state.focus_topics = data["focus_topics"]
        st.session_state.difficulty = data["difficulty"]
        st.session_state.used_past_context = data["used_past_context"]
        st.session_state.is_followup = False
        st.session_state.session_complete = False
        st.session_state.answered_log = []
        st.session_state.last_rubric = None
        st.session_state.last_error = None
        st.toast("Session started — good luck! 🎯")
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Could not reach the backend for /session/start."
    except requests.exceptions.HTTPError as e:
        st.session_state.last_error = f"/session/start failed ({e.response.status_code}): {_extract_detail(e)}"
    except Exception as e:
        st.session_state.last_error = f"Unexpected error: {e}"


def _apply_answer_result(data: dict, answer_text: str):
    """★ NEW — shared by both the plain (/session/answer) and streaming
    (/session/answer/stream) paths. Everything after the API call
    returns its final result is identical either way, so this is the
    single place that logic lives — avoids two slowly-diverging copies
    of the same state-update code."""
    st.session_state.answered_log.append({
        "question": st.session_state.current_question,
        "answer": answer_text,
        "is_followup": st.session_state.is_followup,
        "rubric": data["rubric"],
    })

    st.session_state.last_rubric = data["rubric"]
    st.session_state.last_needs_followup = data["needs_followup"]
    st.session_state.retry_count = data["retry_count"]
    st.session_state.question_index = data["question_index"]
    st.session_state.session_complete = data["session_complete"]

    score = data["rubric"]["overall_score"]
    if data.get("input_blocked"):
        st.toast("⚠️ Answer flagged by guardrails — see feedback below.")
    elif data["session_complete"]:
        st.toast(f"Final answer scored {score}/10 — session complete! 🏁")
    elif data["needs_followup"]:
        st.toast(f"Scored {score}/10 — sending a follow-up to dig deeper.")
    else:
        st.toast(f"Scored {score}/10 — nice, moving to the next question. ✅")

    if data["session_complete"]:
        st.session_state.current_question = None
        # A session just completed: Neo4j got a new :Session node and
        # Pinecone got a fresh Q&A batch — refresh both dashboards.
        st.session_state.topics_stale = True
        st.session_state.sessions_stale = True
    else:
        st.session_state.current_question = data["next_question"]
        st.session_state.is_followup = data["needs_followup"]

    st.session_state.last_error = None
    st.session_state.form_key_suffix += 1  # fresh empty textbox for the next question


def submit_answer(answer_text: str):
    """POST /session/answer (non-streaming path). Guarded by
    st.session_state.submitting so a slow network can't let the same
    answer fire twice."""
    if st.session_state.submitting:
        return
    st.session_state.submitting = True
    try:
        resp = api_post("/session/answer", {
            "thread_id": st.session_state.thread_id,
            "answer": answer_text,
        })
        resp.raise_for_status()
        _apply_answer_result(resp.json(), answer_text)
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Could not reach the backend for /session/answer."
    except requests.exceptions.HTTPError as e:
        st.session_state.last_error = f"/session/answer failed ({e.response.status_code}): {_extract_detail(e)}"
    except Exception as e:
        st.session_state.last_error = f"Unexpected error: {e}"
    finally:
        st.session_state.submitting = False


def submit_answer_streaming(answer_text: str, status_placeholder, feedback_placeholder):
    """★ NEW — POST /session/answer/stream (SSE path — Day 20).

    Reads Server-Sent Events off the response body as they arrive and
    updates two Streamlit placeholders live, in the SAME script run —
    no rerun needed per token. Once the 'done' event arrives, hands off
    to the same _apply_answer_result() the non-streaming path uses, so
    downstream state (history, dashboard refresh, next question) is
    identical either way."""
    if st.session_state.submitting:
        return
    st.session_state.submitting = True
    final_payload = None
    try:
        url = st.session_state.api_url.rstrip("/") + "/session/answer/stream"
        with requests.post(
            url,
            json={"thread_id": st.session_state.thread_id, "answer": answer_text},
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                event = json.loads(raw_line[len("data: "):])
                event_type = event.get("type")

                if event_type == "status":
                    status_placeholder.info(event["message"])
                elif event_type == "scores":
                    status_placeholder.empty()
                    # columns() on an st.empty() container works fine —
                    # this replaces the "Evaluating..." message with the
                    # score row as soon as scoring finishes.
                    c1, c2, c3, c4 = status_placeholder.columns(4)
                    c1.metric("Depth", f"{event['depth_score']}/10")
                    c2.metric("Accuracy", f"{event['accuracy_score']}/10")
                    c3.metric("Communication", f"{event['communication_score']}/10")
                    c4.metric("Overall", f"{event['overall_score']}/10")
                elif event_type == "token":
                    feedback_placeholder.info(event["partial_feedback"] + " ▌")
                elif event_type == "error":
                    st.session_state.last_error = event.get("detail", "Unknown streaming error")
                    return
                elif event_type == "done":
                    final_payload = event
                    feedback_placeholder.info(event["rubric"]["feedback"])  # remove the cursor
    except requests.exceptions.ConnectionError:
        st.session_state.last_error = "Could not reach the backend for /session/answer/stream."
        return
    except requests.exceptions.HTTPError as e:
        st.session_state.last_error = f"/session/answer/stream failed ({e.response.status_code}): {_extract_detail(e)}"
        return
    except Exception as e:
        st.session_state.last_error = f"Unexpected error: {e}"
        return
    finally:
        st.session_state.submitting = False

    if final_payload is None:
        st.session_state.last_error = "Stream ended without a final result — check the backend logs."
        return

    _apply_answer_result(final_payload, answer_text)


def fetch_topics():
    """GET /user/{user_id}/topics. A 404 means 'no data yet', which is a
    normal, expected state for a new user — not an error to alarm the
    user about, so it's tracked separately from other failures."""
    try:
        resp = api_get(f"/user/{st.session_state.user_id}/topics")
        if resp.status_code == 404:
            st.session_state.topics_data = []
            st.session_state.topics_error = "no_data"
            st.session_state.topics_stale = False
            return
        resp.raise_for_status()
        st.session_state.topics_data = resp.json()["topics"]
        st.session_state.topics_error = None
        st.session_state.topics_stale = False
    except requests.exceptions.ConnectionError:
        st.session_state.topics_error = "Could not reach the backend for /user/{id}/topics."
        st.session_state.topics_stale = False
    except Exception as e:
        st.session_state.topics_error = f"Unexpected error: {e}"
        st.session_state.topics_stale = False


def fetch_sessions():
    """★ NEW — GET /user/{user_id}/sessions. Same 404-as-'no_data' pattern
    as fetch_topics()."""
    try:
        resp = api_get(f"/user/{st.session_state.user_id}/sessions")
        if resp.status_code == 404:
            st.session_state.sessions_data = []
            st.session_state.sessions_error = "no_data"
            st.session_state.sessions_stale = False
            return
        resp.raise_for_status()
        st.session_state.sessions_data = resp.json()["sessions"]
        st.session_state.sessions_error = None
        st.session_state.sessions_stale = False
    except requests.exceptions.ConnectionError:
        st.session_state.sessions_error = "Could not reach the backend for /user/{id}/sessions."
        st.session_state.sessions_stale = False
    except Exception as e:
        st.session_state.sessions_error = f"Unexpected error: {e}"
        st.session_state.sessions_stale = False


def fetch_session_qa(session_id: str):
    """★ NEW — GET /user/{user_id}/sessions/{session_id}/qa. Loads the
    full transcript for one past session, on demand (only when the
    user actually picks a session to view)."""
    try:
        resp = api_get(f"/user/{st.session_state.user_id}/sessions/{session_id}/qa")
        resp.raise_for_status()
        st.session_state.session_qa_data = resp.json()["exchanges"]
        st.session_state.session_qa_error = None
    except requests.exceptions.HTTPError as e:
        st.session_state.session_qa_data = None
        st.session_state.session_qa_error = f"Could not load transcript ({e.response.status_code}): {_extract_detail(e)}"
    except Exception as e:
        st.session_state.session_qa_data = None
        st.session_state.session_qa_error = f"Unexpected error: {e}"


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 CogniCoach")
    st.caption("Adaptive AI interview coach — Streamlit dashboard")

    st.subheader("Backend connection")
    st.session_state.api_url = st.text_input("FastAPI base URL", value=st.session_state.api_url)
    new_user_id = st.text_input(
        "User ID", value=st.session_state.user_id,
        disabled=st.session_state.thread_id is not None,
        help="Locked once a session has started — start a new session to change it.",
    )
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.topics_stale = True    # different user → different topic profile
        st.session_state.sessions_stale = True  # ★ NEW — and a different session history
        st.session_state.selected_session_id = None
        st.session_state.session_qa_data = None

    if st.button("🔄 Check backend health", use_container_width=True):
        with st.spinner("Pinging backend..."):
            check_health()

    if st.session_state.connected:
        st.success("Connected")
    elif st.session_state.last_error:
        st.error("Not connected")

    st.divider()

    if st.session_state.thread_id:
        st.caption(f"thread_id: `{st.session_state.thread_id[:8]}...`")

        if not st.session_state.session_complete:
            if not st.session_state.confirm_reset:
                if st.button("🔁 Start a new session", use_container_width=True):
                    st.session_state.confirm_reset = True
                    st.rerun()
            else:
                st.warning("You're mid-interview — really start over?")
                c1, c2 = st.columns(2)
                if c1.button("Yes, reset", type="primary", use_container_width=True):
                    reset_interview_state()
                    st.rerun()
                if c2.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()
        else:
            if st.button("🔁 Start a new session", use_container_width=True):
                reset_interview_state()
                st.rerun()

    st.divider()
    st.session_state.streaming_enabled = st.toggle(
        "✨ Enable streaming feedback",
        value=st.session_state.streaming_enabled,
        help="Feedback types out word-by-word instead of appearing all at "
             "once. Requires backend_addition/streaming_router.py wired "
             "into main.py.",
    )

    st.divider()
    st.caption("Day 20 — optional streaming feedback added. "
               "Days 21+ move on to Docker + deployment.")


# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════

st.title("CogniCoach — Interview")

if st.session_state.last_error:
    st.warning(st.session_state.last_error)

tab_interview, tab_history, tab_topics, tab_analytics = st.tabs(
    ["🎤 Interview", "📜 Answers so far", "📊 Weak Topics", "📈 History"]
)

# ──────────────────────────────────────────────────────────────
# TAB 1 — Interview (unchanged from Day 17)
# ──────────────────────────────────────────────────────────────
with tab_interview:

    if st.session_state.thread_id is None:
        st.write(
            "Start a session to get 5 personalised interview questions "
            "targeting your weakest topics."
        )
        if st.button("▶️ Start Session", type="primary"):
            with st.spinner("Planner is generating your questions..."):
                start_session()
            st.rerun()

    elif st.session_state.session_complete:
        st.success("🎉 Session complete!")
        st.balloons()

        log = st.session_state.answered_log
        if log:
            avg_score = sum(e["rubric"]["overall_score"] for e in log) / len(log)
            c1, c2 = st.columns(2)
            c1.metric("Questions answered", len(log))
            c2.metric("Average score", f"{avg_score:.1f} / 10")

        st.caption("See 📜 Answers so far for the recap, 📊 Weak Topics for "
                   "updated topic mastery, or 📈 History to see this session "
                   "added to your score trend — or start another session "
                   "from the sidebar.")

    else:
        progress = st.session_state.question_index / max(st.session_state.total_questions, 1)
        st.progress(
            min(progress, 1.0),
            text=f"Question {st.session_state.question_index + 1} of {st.session_state.total_questions}",
        )

        if st.session_state.focus_topics:
            st.caption("Focus topics: " + ", ".join(st.session_state.focus_topics)
                       + f" · Difficulty: {st.session_state.difficulty}"
                       + (" · using your past sessions 🧠" if st.session_state.used_past_context else ""))

        if st.session_state.last_rubric is not None:
            r = st.session_state.last_rubric
            with st.container(border=True):
                st.markdown("**Last answer scored:**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Depth", f"{r['depth_score']}/10")
                c2.metric("Accuracy", f"{r['accuracy_score']}/10")
                c3.metric("Communication", f"{r['communication_score']}/10")
                c4.metric("Overall", f"{r['overall_score']}/10")
                st.info(r["feedback"])
                if r["weak_topics"]:
                    st.caption("Weak topics: " + ", ".join(r["weak_topics"]))

        if st.session_state.is_followup:
            st.warning(
                f"🔁 Follow-up question — attempt "
                f"{st.session_state.retry_count} of {MAX_RETRIES + 1} "
                f"(your last answer needed more depth):"
            )
        else:
            st.markdown("**Question:**")

        st.markdown(f"### {st.session_state.current_question}")

        form_key = f"answer_form_{st.session_state.form_key_suffix}"
        with st.form(key=form_key, clear_on_submit=False):
            answer_text = st.text_area(
                "Your answer",
                height=150,
                placeholder=f"Type your answer here (min {MIN_ANSWER_CHARS} characters)...",
            )
            char_count = len(answer_text.strip())
            too_short = char_count < MIN_ANSWER_CHARS
            st.caption(f"{char_count} characters"
                       + (f" — need at least {MIN_ANSWER_CHARS}" if too_short else " ✓"))

            submitted = st.form_submit_button(
                "Submit answer",
                type="primary",
                disabled=st.session_state.submitting,
            )

        if submitted:
            if too_short:
                st.error(f"Answer needs at least {MIN_ANSWER_CHARS} characters — add a bit more detail.")
            elif st.session_state.streaming_enabled:
                status_placeholder = st.empty()
                feedback_placeholder = st.empty()
                submit_answer_streaming(answer_text.strip(), status_placeholder, feedback_placeholder)
                st.rerun()
            else:
                with st.spinner("Evaluator is scoring your answer..."):
                    submit_answer(answer_text.strip())
                st.rerun()

# ──────────────────────────────────────────────────────────────
# TAB 2 — Answers so far (unchanged from Day 17)
# ──────────────────────────────────────────────────────────────
with tab_history:
    log = st.session_state.answered_log
    if not log:
        st.info("No answers submitted yet this session.")
    else:
        avg_score = sum(e["rubric"]["overall_score"] for e in log) / len(log)
        st.metric("Running average", f"{avg_score:.1f} / 10")
        for i, entry in enumerate(log, 1):
            label = f"Q{i}" + (" 🔁 follow-up" if entry["is_followup"] else "")
            with st.expander(f"{label} — {entry['rubric']['overall_score']}/10"):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Your answer:** {entry['answer']}")
                st.markdown(f"**Feedback:** {entry['rubric']['feedback']}")
                if entry["rubric"]["weak_topics"]:
                    st.markdown("**Weak topics:** " + ", ".join(entry["rubric"]["weak_topics"]))

# ──────────────────────────────────────────────────────────────
# TAB 3 — ★ NEW — Weak Topics dashboard
# ──────────────────────────────────────────────────────────────
with tab_topics:
    st.subheader(f"Topic mastery — {st.session_state.user_id}")

    refresh_col, _ = st.columns([1, 3])
    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.topics_stale = True

    # Fetch lazily: only when the tab is actually opened for the first
    # time, the user_id changed, or a session just completed — not on
    # every single rerun (e.g. every keystroke in the answer box).
    if st.session_state.topics_stale:
        with st.spinner("Loading topic data..."):
            fetch_topics()

    if st.session_state.topics_error == "no_data":
        st.info(
            "No topic data yet for this user. Complete an interview "
            "session first — Neo4j gets written to at the end of each "
            "question via `graph_updater_node`."
        )
    elif st.session_state.topics_error:
        st.warning(st.session_state.topics_error)
    elif st.session_state.topics_data:
        topics = st.session_state.topics_data
        df = pd.DataFrame(topics).sort_values("weakness_score", ascending=False)

        def _bucket(score: float) -> str:
            if score >= 0.65:
                return "Weak"
            if score >= 0.4:
                return "Developing"
            return "Strong"

        df["status"] = df["weakness_score"].apply(_bucket)
        color_map = {"Weak": "#e05252", "Developing": "#e0b152", "Strong": "#52b788"}

        st.bar_chart(
            df.set_index("topic")[["weakness_score"]],
            color="#e05252",  # st.bar_chart doesn't support per-bar colour —
            # noted as a known limitation, see README for the plotly upgrade path
            height=max(300, 28 * len(df)),
        )

        legend_cols = st.columns(3)
        legend_cols[0].markdown("🔴 **Weak** — score ≥ 0.65")
        legend_cols[1].markdown("🟡 **Developing** — 0.4 ≤ score < 0.65")
        legend_cols[2].markdown("🟢 **Strong** — score < 0.4")

        st.divider()
        st.markdown("**Full breakdown**")
        display_df = df[["topic", "status", "weakness_score", "times_tested", "last_tested"]].rename(columns={
            "topic": "Topic", "status": "Status", "weakness_score": "Weakness score",
            "times_tested": "Times tested", "last_tested": "Last tested",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click 🔄 Refresh to load topic data.")

# ──────────────────────────────────────────────────────────────
# TAB 4 — ★ NEW — History & Analytics (past completed sessions)
# ──────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader(f"Session history — {st.session_state.user_id}")

    refresh_col, _ = st.columns([1, 3])
    with refresh_col:
        if st.button("🔄 Refresh history", use_container_width=True):
            st.session_state.sessions_stale = True

    if st.session_state.sessions_stale:
        with st.spinner("Loading session history..."):
            fetch_sessions()

    if st.session_state.sessions_error == "no_data":
        st.info(
            "No completed sessions yet for this user. Finish a full "
            "interview in the 🎤 Interview tab first — `graph_updater_node` "
            "writes a `:Session` node to Neo4j once all 5 questions are done."
        )
    elif st.session_state.sessions_error:
        st.warning(st.session_state.sessions_error)
    elif st.session_state.sessions_data:
        sessions = st.session_state.sessions_data  # newest-first from Neo4j
        df = pd.DataFrame(sessions)

        # Charts read left-to-right as "time passing" — flip to oldest-first
        # for the trend line, but keep the table newest-first (more useful
        # for scanning "what did I just do").
        chart_df = df.iloc[::-1].reset_index(drop=True).copy()
        chart_df["session_label"] = [f"Session {i + 1}" for i in range(len(chart_df))]

        st.markdown("**Average score trend**")
        st.line_chart(chart_df.set_index("session_label")[["avg_score"]])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sessions completed", len(df))
        c2.metric("Best session", f"{df['avg_score'].max():.1f}/10")
        c3.metric("Overall average", f"{df['avg_score'].mean():.1f}/10")

        st.divider()
        st.markdown("**All sessions**")
        table_df = df.copy()
        table_df["weak_topics"] = table_df["weak_topics"].apply(
            lambda t: ", ".join(t) if t else "—"
        )
        table_df = table_df.rename(columns={
            "session_id": "Session ID", "date": "Date", "avg_score": "Avg score",
            "total_questions": "Questions", "weak_topics": "Weak topics",
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**View a session transcript**")

        session_options = {
            f"{s['date'] or '(no date)'} — avg {s['avg_score']}/10 — {s['session_id'][:8]}...": s["session_id"]
            for s in sessions
        }
        chosen_label = st.selectbox("Pick a session", options=list(session_options.keys()))

        if st.button("Load transcript"):
            chosen_id = session_options[chosen_label]
            st.session_state.selected_session_id = chosen_id
            with st.spinner("Fetching Q&A from Pinecone..."):
                fetch_session_qa(chosen_id)

        if st.session_state.session_qa_error:
            st.warning(st.session_state.session_qa_error)

        if st.session_state.session_qa_data:
            for i, qa in enumerate(st.session_state.session_qa_data, 1):
                with st.expander(f"Q{i} — {qa['overall_score']}/10"):
                    st.markdown(f"**Question:** {qa['question']}")
                    st.markdown(f"**Answer:** {qa['answer']}")
                    st.markdown(f"**Feedback:** {qa['feedback']}")
                    if qa["weak_topics"]:
                        st.caption("Weak topics: " + ", ".join(qa["weak_topics"]))
    else:
        st.info("Click 🔄 Refresh history to load this user's past sessions.")

