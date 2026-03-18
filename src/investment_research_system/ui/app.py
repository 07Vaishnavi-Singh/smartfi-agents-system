"""Streamlit frontend for the Investment Research System.

Run with:
    uv run streamlit run src/investment_research_system/ui/app.py

Requires the FastAPI backend running on localhost:8000:
    uv run uvicorn investment_research_system.api.app:create_app --factory --port 8000
"""

import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
POLL_INTERVAL = 2  # seconds between status polls

FOCUS_AREA_OPTIONS = [
    "Revenue & Earnings",
    "Competition",
    "Market Sentiment",
    "Risk Factors",
    "Valuation",
    "Growth Prospects",
    "Dividends",
    "Technical Analysis",
    "Macro Environment",
    "Management & Leadership",
]

AGENT_DISPLAY = {
    "researcher": {"label": "Researcher", "icon": "🔍", "color": "#4A90D9"},
    "sentiment": {"label": "Sentiment Analyst", "icon": "📊", "color": "#50C878"},
    "analyst": {"label": "Financial Analyst", "icon": "💹", "color": "#FFB347"},
    "risk_assessor": {"label": "Risk Assessor", "icon": "⚠️", "color": "#E74C3C"},
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investment Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Main container */
    .block-container { max-width: 1100px; padding-top: 2rem; }

    /* Agent cards */
    .agent-card {
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }

    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e0e0e0;
        margin: 0.3rem 0 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.8rem;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.2rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-pending { background: #fff3cd; color: #856404; }
    .status-running { background: #cce5ff; color: #004085; }
    .status-completed { background: #d4edda; color: #155724; }
    .status-failed { background: #f8d7da; color: #721c24; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_health() -> dict | None:
    """Ping the API health endpoint."""
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
        return None


def submit_research(query: str, focus_areas: list[str], depth: str) -> dict:
    """POST /research and return the response."""
    r = httpx.post(
        f"{API_BASE}/research",
        json={"query": query, "focus_areas": focus_areas, "depth": depth},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_job_status(job_id: str) -> dict:
    """GET /research/{job_id} and return the response."""
    r = httpx.get(f"{API_BASE}/research/{job_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def confidence_color(score: float) -> str:
    """Return a hex color based on confidence score."""
    if score >= 0.8:
        return "#28a745"
    if score >= 0.6:
        return "#ffc107"
    return "#dc3545"


def render_status_badge(status: str) -> str:
    """Return HTML for a status badge."""
    return f'<span class="status-badge status-{status}">{status.upper()}</span>'


def render_sources(sources: list[dict]) -> None:
    """Display sources as clickable chips."""
    if not sources:
        return
    html_parts = []
    for s in sources:
        title = s.get("title", "Source")
        url = s.get("url")
        score = s.get("reliability_score", 0)
        score_pct = int(score * 100)
        if url:
            html_parts.append(
                f'<a href="{url}" target="_blank" class="source-chip" '
                f'title="Reliability: {score_pct}%">{title[:50]} ({score_pct}%)</a>'
            )
        else:
            html_parts.append(
                f'<span class="source-chip">{title[:50]} ({score_pct}%)</span>'
            )
    st.markdown(" ".join(html_parts), unsafe_allow_html=True)


def render_agent_card(response: dict) -> None:
    """Render a single agent's response as an expandable card."""
    name = response.get("agent_name", "unknown")
    info = AGENT_DISPLAY.get(name, {"label": name, "icon": "🤖", "color": "#888"})
    confidence = response.get("confidence", 0)
    conf_pct = int(confidence * 100)
    cost = response.get("cost_usd", 0)
    tokens = response.get("tokens_used", 0)
    latency = response.get("latency_ms", 0)

    header = f"{info['icon']} {info['label']}  —  Confidence: {conf_pct}%"

    with st.expander(header, expanded=(name == "researcher")):
        # Confidence bar
        color = confidence_color(confidence)
        st.markdown(
            f"""<div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf_pct}%; background:{color};"></div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Cost", f"${cost:.4f}")
        col2.metric("Tokens", f"{tokens:,}")
        col3.metric("Latency", f"{latency / 1000:.1f}s")

        # Content
        st.markdown(response.get("content", "No content."))

        # Sources
        sources = response.get("sources", [])
        if sources:
            st.markdown("**Sources:**")
            render_sources(sources)


def render_report(report: dict) -> None:
    """Render the full research report."""
    # --- Summary metrics ---
    total_cost = report.get("total_cost_usd", 0)
    total_tokens = report.get("total_tokens", 0)
    proc_time = report.get("processing_time_seconds", 0)
    agent_count = len(report.get("agent_responses", []))

    st.markdown(
        f"""<div class="metric-row">
            <div class="metric-card">
                <div class="metric-value">{agent_count}</div>
                <div class="metric-label">Agents</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${total_cost:.4f}</div>
                <div class="metric-label">Total Cost</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_tokens:,}</div>
                <div class="metric-label">Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{proc_time:.1f}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- Executive summary ---
    summary = report.get("summary", "")
    if summary:
        st.markdown("### Executive Summary")
        st.markdown(summary)

    # --- Failed agents ---
    failed_agents = report.get("failed_agents", [])
    if failed_agents:
        st.markdown("### Agent Failures")
        for failure in failed_agents:
            agent_name = failure.get("agent", "unknown")
            error_type = failure.get("error_type", "unknown")
            error_message = failure.get("error_message", "Unknown error")
            info = AGENT_DISPLAY.get(
                agent_name, {"label": agent_name, "icon": "🤖", "color": "#888"},
            )

            # Map error types to user-friendly icons
            error_icons = {
                "timeout": "⏱️",
                "rate_limit": "🚦",
                "refused": "🚫",
                "budget_exceeded": "💰",
                "memory_unavailable": "🗄️",
                "circuit_breaker": "⚡",
                "agent_error": "❌",
                "unexpected": "⚠️",
            }
            error_icon = error_icons.get(error_type, "❌")

            st.warning(
                f"{error_icon} **{info['label']}** — {error_message}",
                icon=None,
            )

    # --- Agent responses ---
    st.markdown("### Agent Analysis")
    agent_order = ["researcher", "sentiment", "analyst", "risk_assessor"]
    responses_by_name = {
        r["agent_name"]: r for r in report.get("agent_responses", [])
    }
    for agent_name in agent_order:
        if agent_name in responses_by_name:
            render_agent_card(responses_by_name[agent_name])
    # Render any agents not in the predefined order
    for r in report.get("agent_responses", []):
        if r["agent_name"] not in agent_order:
            render_agent_card(r)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Investment Research")
    st.caption("Multi-agent AI research system")
    st.divider()

    # Health check
    health = check_health()
    if health:
        st.success("API Connected")
        st.caption(f"Version: {health['version']}")
        tracing = "On" if health.get("tracing_enabled") else "Off"
        st.caption(f"Tracing: {tracing}")
    else:
        st.error("API Offline")
        st.caption(f"Cannot reach {API_BASE}")
        st.caption("Start with: `uv run uvicorn investment_research_system.api.app:create_app --factory --port 8000`")

    st.divider()

    # History
    if "history" in st.session_state and st.session_state.history:
        st.markdown("### Recent Queries")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            status_icon = "✅" if h["status"] == "completed" else "❌" if h["status"] == "failed" else "⏳"
            if st.button(f"{status_icon} {h['query'][:30]}...", key=f"hist_{i}"):
                st.session_state.active_job = h["job_id"]
                st.rerun()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "active_job" not in st.session_state:
    st.session_state.active_job = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📈 Investment Research")
st.caption("Ask any investment question — 4 AI agents will research, analyze sentiment, assess financials, and evaluate risks.")

# --- Input form ---
with st.form("research_form"):
    query = st.text_area(
        "What do you want to research?",
        placeholder="e.g., Should I invest in NVIDIA? Is Tesla overvalued? Compare AMD vs Intel for long-term growth.",
        height=80,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        focus_areas = st.multiselect(
            "Focus areas (optional)",
            options=FOCUS_AREA_OPTIONS,
            default=[],
            help="Select specific areas for the agents to focus on.",
        )
    with col2:
        depth = st.selectbox(
            "Research depth",
            options=["quick", "standard", "deep"],
            index=1,
            help="Quick: fast overview. Standard: balanced. Deep: thorough analysis.",
        )

    submitted = st.form_submit_button("🚀 Start Research", use_container_width=True)

# --- Handle submission ---
if submitted:
    if not query or len(query.strip()) < 3:
        st.error("Please enter a research question (at least 3 characters).")
    elif not health:
        st.error("Cannot submit — API is offline. Start the backend first.")
    else:
        try:
            result = submit_research(query.strip(), focus_areas, depth)
            job_id = result["job_id"]
            st.session_state.active_job = job_id
            st.session_state.history.append({
                "job_id": job_id,
                "query": query.strip(),
                "status": "pending",
            })
            st.rerun()
        except httpx.HTTPStatusError as e:
            st.error(f"API error: {e.response.status_code} — {e.response.text}")
        except httpx.ConnectError:
            st.error("Cannot connect to API. Is the backend running?")
        except httpx.TimeoutException:
            st.error("API timed out while creating the research job. Please retry.")

# --- Active job display ---
if st.session_state.active_job:
    job_id = st.session_state.active_job
    st.divider()

    try:
        job = get_job_status(job_id)
    except httpx.HTTPStatusError:
        st.error(f"Job {job_id} not found.")
        st.session_state.active_job = None
        st.stop()
    except httpx.ConnectError:
        st.error("Lost connection to API.")
        st.stop()
    except httpx.TimeoutException:
        st.warning("API is slow right now. Retrying job status...")
        time.sleep(POLL_INTERVAL)
        st.rerun()

    status = job["status"]

    # Update history entry
    for h in st.session_state.history:
        if h["job_id"] == job_id:
            h["status"] = status

    # --- Pending / Running ---
    if status in ("pending", "running"):
        st.markdown(
            f"**Query:** {job['query']}  {render_status_badge(status)}",
            unsafe_allow_html=True,
        )

        status_messages = {
            "pending": "Queuing research job...",
            "running": "Agents are researching — this typically takes 15-30 seconds...",
        }
        with st.spinner(status_messages.get(status, "Working...")):
            time.sleep(POLL_INTERVAL)
            st.rerun()

    # --- Failed ---
    elif status == "failed":
        st.markdown(
            f"**Query:** {job['query']}  {render_status_badge(status)}",
            unsafe_allow_html=True,
        )
        st.error(f"Research failed: {job.get('error', 'Unknown error')}")

        if st.button("🔄 Retry"):
            st.session_state.active_job = None
            st.rerun()

    # --- Completed ---
    elif status == "completed":
        st.markdown(
            f"**Query:** {job['query']}  {render_status_badge(status)}",
            unsafe_allow_html=True,
        )

        report = job.get("report")
        if report:
            render_report(report)
        else:
            st.warning("Job completed but no report data found.")

        # New research button
        st.divider()
        if st.button("🔍 New Research", use_container_width=True):
            st.session_state.active_job = None
            st.rerun()
