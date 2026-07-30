"""
main.py
-------
Streamlit UI for the AI Data Analyst Agent.

This file is intentionally "dumb": it only handles page layout, user input,
and displaying results. All the real work happens in:
    - agent/agents.py  (LangChain: LLM calls, prompts, structured output)
    - agent/graph.py   (LangGraph: orchestration of the agent pipeline)
"""

from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

from agent import agents
from agent.graph import run_analysis

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
        "response": None,
        "last_question": None,
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
    st.session_state["response"] = None
    st.session_state["last_question"] = None


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip stray whitespace/quote characters from column names.

    Real-world CSVs often have headers like `"Game Length"` where a space
    before the quote (`, "Game Length"`) means pandas keeps the space and
    quote marks as literal characters in the column name. Every agent (the
    LLM included) would then have to guess the exact messy literal string,
    which is unreliable - cleaning it once here fixes it everywhere.
    """
    cleaned = [str(c).strip().strip('"').strip("'").strip() for c in df.columns]
    if len(set(cleaned)) == len(cleaned):
        df = df.copy()
        df.columns = cleaned
    return df


def load_csv(uploaded_file) -> pd.DataFrame:
    """Read and validate an uploaded CSV file. Raises ValueError on problems."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read this file as a CSV. Details: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise ValueError("The uploaded CSV is empty.")

    df = _clean_column_names(df)

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

    if st.session_state.get("result") is not None:
        st.caption(
            "Statistics / Charts / AI Insights / Final Report were generated once from "
            "your first question and stay the same. Only the Response below updates per question."
        )
        if st.button("🔄 Regenerate full overview", width="stretch"):
            st.session_state["result"] = None

    if reset_clicked:
        reset_session()
        st.rerun()

    if analyze_clicked:
        return question
    return None


def render_results() -> None:
    response = st.session_state.get("response")
    result = st.session_state["result"]

    if not response and not result:
        return

    st.divider()
    st.header("Results")

    if response:
        with st.expander("💬 Response", expanded=True):
            if st.session_state.get("last_question"):
                st.caption(f"Question: {st.session_state['last_question']}")
            st.markdown(response)

    if not result:
        return

    if result.get("error"):
        st.warning(f"Part of the pipeline had an issue: {result['error']}")

    stats = result.get("statistics") or {}
    with st.expander("📈 Statistics", expanded=False):
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

        unique_counts = stats.get("categorical_unique_counts") or {}
        if unique_counts:
            st.markdown("**Distinct value counts (exact):**")
            st.json(unique_counts)

    charts = result.get("charts") or []
    with st.expander("📊 Charts", expanded=False):
        if not charts:
            st.info("No charts were needed to answer this question.")
        for chart in charts:
            st.markdown(f"**{chart['title']}**")
            st.image(chart["path"], width="stretch")

    with st.expander("💡 AI Insights", expanded=False):
        st.markdown(result.get("insights") or "No insights generated.")

    with st.expander("📄 Final Report", expanded=False):
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
            clean_question = question.strip()
            df = st.session_state["df"]
            needs_full_analysis = st.session_state["result"] is None

            spinner_text = (
                "Running the multi-agent pipeline (planner → statistics/visualization → "
                "insights → report) and answering your question..."
                if needs_full_analysis
                else "Answering your question..."
            )
            with st.spinner(spinner_text):
                try:
                    response = agents.run_query_agent(clean_question, df)
                    st.session_state["response"] = response
                    st.session_state["last_question"] = clean_question

                    if needs_full_analysis:
                        result = run_analysis(df, clean_question)
                        st.session_state["result"] = result
                        st.session_state["last_chart_dir"] = result.get("chart_dir")

                    st.success("Done.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Analysis failed: {exc}")

    render_results()


if __name__ == "__main__":
    main()
