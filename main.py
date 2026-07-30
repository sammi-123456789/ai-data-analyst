"""
main.py
-------
Streamlit UI for the AI Data Analyst Agent.

This file is intentionally "dumb": it only handles page layout, user input,
and displaying results. All the real work happens in:
    - agents.py  (LangChain: LLM calls, prompts, structured output)
    - graph.py   (LangGraph: orchestration of the agent pipeline)
"""

from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

import agents
from graph import run_analysis

st.set_page_config(page_title="AI Data Analyst Agent", page_icon="📊", layout="wide")

EXAMPLE_QUESTIONS = [
    "Which product generated the highest revenue?",
    "Which product sold the most?",
    "Show monthly revenue.",
    "What is the average sales?",
    "Which customer spent the most?",
    "Which city generated the highest revenue?",
    "Show the distribution of sales.",
    "Are there any missing values?",
    "What are the top five products?",
    "Summarize this dataset.",
    "Generate business insights.",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "df": None,
        "file_name": None,
        "result": None,
        "last_chart_dir": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear everything: uploaded data, results, and any temp chart files."""
    if st.session_state.get("last_chart_dir"):
        shutil.rmtree(st.session_state["last_chart_dir"], ignore_errors=True)
    st.session_state["df"] = None
    st.session_state["file_name"] = None
    st.session_state["result"] = None
    st.session_state["last_chart_dir"] = None


def load_csv(uploaded_file) -> pd.DataFrame:
    """Read and validate an uploaded CSV file. Raises ValueError on problems."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read this file as a CSV. Details: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise ValueError("The uploaded CSV is empty.")

    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.header("Setup")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = load_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state["file_name"] = uploaded_file.name
                st.success(f"Loaded `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} columns)")
            except ValueError as exc:
                st.error(str(exc))

        st.divider()
        st.subheader("API Status")
        if agents.is_api_key_configured():
            st.success("Groq API key detected")
        else:
            st.warning("GROQ_API_KEY not set. Add it to a .env file (see .env.example).")

        st.divider()
        if st.button("Reset Session", width="stretch"):
            reset_session()
            st.rerun()


# ---------------------------------------------------------------------------
# Main page sections
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("AI Data Analyst Agent")
    st.markdown(
        "Upload a CSV file, ask a question in plain English, and this agent "
        "will compute statistics, build charts, and generate AI-written "
        "insights and a final report - powered by **LangChain + LangGraph + Groq**."
    )


def render_data_preview() -> None:
    df: pd.DataFrame = st.session_state["df"]
    st.subheader("Data Preview")
    st.dataframe(df.head(10), width="stretch")
    st.caption(f"Showing first 10 of {df.shape[0]} rows - {df.shape[1]} columns.")


def render_question_form() -> str | None:
    st.subheader("Ask a Question")
    with st.expander("Need ideas? Click for example questions"):
        for q in EXAMPLE_QUESTIONS:
            st.markdown(f"- {q}")

    question = st.text_area(
        "Ask anything about your data",
        placeholder="e.g. Which product generated the highest revenue?",
        height=90,
    )

    col1, col2 = st.columns([1, 1])
    analyze_clicked = col1.button("Analyze", type="primary", width="stretch")
    reset_clicked = col2.button("Reset", width="stretch")

    if reset_clicked:
        reset_session()
        st.rerun()

    if analyze_clicked:
        return question
    return None


def render_results() -> None:
    result = st.session_state["result"]
    if not result:
        return

    st.divider()
    st.header("Results")

    if result.get("error"):
        st.warning(f"Part of the pipeline had an issue: {result['error']}")

    stats = result.get("statistics") or {}
    with st.expander("📈 Statistics", expanded=True):
        col1, col2 = st.columns(2)
        col1.metric("Rows", stats.get("row_count", "-"))
        col2.metric("Columns", stats.get("column_count", "-"))

        missing = stats.get("missing_values") or {}
        if missing:
            st.markdown("**Missing values:**")
            st.json(missing)
        else:
            st.markdown("**Missing values:** None")

        numeric_summary = stats.get("numeric_summary") or {}
        if numeric_summary:
            st.markdown("**Numeric summary:**")
            st.dataframe(pd.DataFrame(numeric_summary).T, width="stretch")

        grouped = stats.get("grouped_summary")
        if grouped:
            st.markdown(f"**Grouped by {grouped['groupby_column']} ({grouped['metric_column']}):**")
            st.dataframe(pd.DataFrame(grouped["data"]).T, width="stretch")

    charts = result.get("charts") or []
    with st.expander("📊 Charts", expanded=True):
        if not charts:
            st.info("No charts were needed to answer this question.")
        for chart in charts:
            st.markdown(f"**{chart['title']}**")
            st.image(chart["path"], width="stretch")

    with st.expander("💡 AI Insights", expanded=True):
        st.markdown(result.get("insights") or "No insights generated.")

    with st.expander("📄 Final Report", expanded=True):
        st.markdown(result.get("report") or "No report generated.")
        st.download_button(
            "Download Report (Markdown)",
            data=result.get("report") or "",
            file_name="data_analysis_report.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_session_state()
    render_sidebar()
    render_header()

    if st.session_state["df"] is None:
        st.info("Upload a CSV file from the sidebar to get started.")
        return

    render_data_preview()
    question = render_question_form()

    if question is not None:
        if not question.strip():
            st.warning("Please type a question before clicking Analyze.")
        elif not agents.is_api_key_configured():
            st.error("GROQ_API_KEY is not set. Add it to a .env file, then restart the app.")
        else:
            with st.spinner("Running the multi-agent pipeline (planner → statistics/visualization → insights → report)..."):
                try:
                    result = run_analysis(st.session_state["df"], question.strip())
                    st.session_state["result"] = result
                    st.session_state["last_chart_dir"] = result.get("chart_dir")
                    st.success("Analysis complete.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Analysis failed: {exc}")

    render_results()


if __name__ == "__main__":
    main()
