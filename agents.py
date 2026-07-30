"""
agents.py
---------
This file holds ALL the LangChain-related code for the AI Data Analyst Agent.

LangChain is used ONLY for:
    - Talking to the Groq LLM (llama-3.3-70b-versatile)
    - Prompt templates
    - Structured output (turning free text into a validated Python object)
    - Output parsing

LangChain is NOT used for orchestration. The graph/flow control lives in
graph.py (that part is LangGraph's job).

This file also contains the pure-Python "Statistics Agent" and
"Visualization Agent" logic. Those two agents do not need an LLM at all -
they just use pandas and matplotlib.
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, required for Streamlit/servers
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
ALLOWED_CHART_TYPES = ["bar", "histogram", "pie", "scatter"]


# ---------------------------------------------------------------------------
# LLM setup (LangChain)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Create (and cache) the LangChain ChatGroq LLM client.

    Raises:
        ValueError: if GROQ_API_KEY is missing from the environment/.env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to a .env file in the project "
            "root (see .env.example)."
        )
    return ChatGroq(model=GROQ_MODEL_NAME, api_key=api_key, temperature=temperature)


def is_api_key_configured() -> bool:
    """Quick check used by the UI to show API status without raising."""
    api_key = os.getenv("GROQ_API_KEY")
    return bool(api_key) and api_key != "your_key_here"


# ---------------------------------------------------------------------------
# Planner Agent (LangChain: structured output)
# ---------------------------------------------------------------------------

class AnalysisPlan(BaseModel):
    """Structured plan produced by the Planner Agent."""

    intent_summary: str = Field(
        description="One short sentence describing what the user wants to know."
    )
    relevant_columns: List[str] = Field(
        default_factory=list,
        description="Dataset column names that are relevant to answering the question.",
    )
    groupby_column: Optional[str] = Field(
        default=None,
        description="A categorical column to group by, if grouping helps answer the question. Otherwise null.",
    )
    metric_column: Optional[str] = Field(
        default=None,
        description="A numeric column to analyze/aggregate, if relevant. Otherwise null.",
    )
    chart_types: List[str] = Field(
        default_factory=list,
        description=(
            "Which chart types would help answer the question. "
            "Choose only from: bar, histogram, pie, scatter. Empty list if no chart is needed."
        ),
    )
    needs_grouped_summary: bool = Field(
        default=False,
        description="True if a group-by statistical summary (sum/mean/count per group) is needed.",
    )


PLANNER_SYSTEM_PROMPT = """You are the Planner Agent inside an AI Data Analyst system.
You will be given the columns of a pandas DataFrame and a user's question.
Your job is to decide what needs to be computed to answer the question well.

Rules:
- Only use column names that actually exist in the dataset.
- chart_types must only contain values from: bar, histogram, pie, scatter.
- If the question is a general request like "summarize this dataset", pick a
  sensible default groupby/metric column and a couple of useful chart types.
- Keep intent_summary to one sentence.
"""


def run_planner_agent(user_question: str, df: pd.DataFrame) -> AnalysisPlan:
    """Ask the LLM to turn the user's question into a structured analysis plan."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(AnalysisPlan)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            (
                "human",
                "Dataset columns and types:\n{columns}\n\n"
                "Sample rows:\n{sample}\n\n"
                "User question: {question}",
            ),
        ]
    )

    columns_desc = "\n".join(f"- {c} ({str(df[c].dtype)})" for c in df.columns)
    sample = df.head(3).to_string()

    chain = prompt | structured_llm
    plan: AnalysisPlan = chain.invoke(
        {"columns": columns_desc, "sample": sample, "question": user_question}
    )

    # Defensive cleanup: drop any hallucinated column names / chart types.
    valid_columns = set(df.columns)
    plan.relevant_columns = [c for c in plan.relevant_columns if c in valid_columns]
    if plan.groupby_column not in valid_columns:
        plan.groupby_column = None
    if plan.metric_column not in valid_columns:
        plan.metric_column = None
    plan.chart_types = [c for c in plan.chart_types if c in ALLOWED_CHART_TYPES]

    return plan


# ---------------------------------------------------------------------------
# Statistics Agent (pure pandas - no LLM needed)
# ---------------------------------------------------------------------------

def _pick_fallback_column(df: pd.DataFrame, dtype_kind: str) -> Optional[str]:
    """Pick a reasonable default column if the planner didn't provide one."""
    if dtype_kind == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cols[0] if cols else None


def compute_statistics(df: pd.DataFrame, plan: AnalysisPlan) -> Dict[str, Any]:
    """
    Compute dataset statistics: shape, missing values, numeric summary,
    and (if relevant) a grouped summary based on the planner's output.
    """
    stats: Dict[str, Any] = {}

    stats["row_count"] = int(df.shape[0])
    stats["column_count"] = int(df.shape[1])
    stats["columns"] = list(df.columns)

    missing = df.isna().sum()
    stats["missing_values"] = {c: int(missing[c]) for c in df.columns if missing[c] > 0}

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        numeric_summary[col] = {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        }
    stats["numeric_summary"] = numeric_summary

    categorical_df = df.select_dtypes(include=["object", "category"])
    categorical_counts = {}
    for col in categorical_df.columns:
        top_values = df[col].value_counts().head(5)
        categorical_counts[col] = {str(k): int(v) for k, v in top_values.items()}
    stats["categorical_top_values"] = categorical_counts

    groupby_col = plan.groupby_column or _pick_fallback_column(df, "categorical")
    metric_col = plan.metric_column or _pick_fallback_column(df, "numeric")

    if plan.needs_grouped_summary and groupby_col and metric_col and groupby_col in df.columns and metric_col in df.columns:
        grouped = (
            df.groupby(groupby_col)[metric_col]
            .agg(["sum", "mean", "count"])
            .round(2)
            .sort_values("sum", ascending=False)
            .head(10)
        )
        stats["grouped_summary"] = {
            "groupby_column": groupby_col,
            "metric_column": metric_col,
            "data": grouped.to_dict(orient="index"),
        }

    return stats


# ---------------------------------------------------------------------------
# Visualization Agent (pure matplotlib - no LLM needed)
# ---------------------------------------------------------------------------

def create_charts(df: pd.DataFrame, plan: AnalysisPlan, output_dir: str) -> List[Dict[str, str]]:
    """
    Create only the charts relevant to the question, based on the planner's
    chart_types list. Returns a list of {"title": ..., "path": ...} dicts.
    """
    os.makedirs(output_dir, exist_ok=True)
    charts: List[Dict[str, str]] = []

    groupby_col = plan.groupby_column or _pick_fallback_column(df, "categorical")
    metric_col = plan.metric_column or _pick_fallback_column(df, "numeric")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    def save_fig(fig, name: str) -> str:
        path = os.path.join(output_dir, f"{name}_{uuid.uuid4().hex[:8]}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    if "bar" in plan.chart_types and groupby_col and metric_col:
        grouped = df.groupby(groupby_col)[metric_col].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_title(f"{metric_col} by {groupby_col}")
        ax.set_ylabel(metric_col)
        ax.set_xlabel(groupby_col)
        plt.xticks(rotation=30, ha="right")
        charts.append({"title": f"{metric_col} by {groupby_col} (Bar Chart)", "path": save_fig(fig, "bar")})

    if "histogram" in plan.chart_types and metric_col:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[metric_col].dropna().plot(kind="hist", bins=15, ax=ax, color="#55A868")
        ax.set_title(f"Distribution of {metric_col}")
        ax.set_xlabel(metric_col)
        charts.append({"title": f"Distribution of {metric_col} (Histogram)", "path": save_fig(fig, "hist")})

    if "pie" in plan.chart_types and groupby_col:
        counts = df[groupby_col].value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
        ax.set_title(f"Share of {groupby_col}")
        charts.append({"title": f"Share of {groupby_col} (Pie Chart)", "path": save_fig(fig, "pie")})

    if "scatter" in plan.chart_types and len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df[x_col], df[y_col], color="#C44E52", alpha=0.7)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        charts.append({"title": f"{y_col} vs {x_col} (Scatter Plot)", "path": save_fig(fig, "scatter")})

    return charts


# ---------------------------------------------------------------------------
# Insight Agent (LangChain: prompt template + LLM + output parser)
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = """You are a friendly senior data analyst.
Explain findings from a dataset in simple, plain English for a business audience.
Be specific and use the numbers you were given - do not invent numbers.

Structure your answer as short markdown bullet points covering:
- Key findings that answer the user's question
- Any notable trends
- Any anomalies or things that stand out
- 2-3 practical business insights or recommendations

Keep it concise (under 200 words).
"""


def run_insight_agent(user_question: str, statistics: Dict[str, Any]) -> str:
    """Use the LLM to turn raw statistics into plain-English insights."""
    llm = get_llm(temperature=0.4)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", INSIGHT_SYSTEM_PROMPT),
            (
                "human",
                "User question: {question}\n\nStatistics (JSON-like):\n{stats}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": user_question, "stats": statistics})


# ---------------------------------------------------------------------------
# Report Agent (LangChain: prompt template + LLM for the summary paragraph)
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """You are a data analyst writing the executive summary
of a short report. Given the user's question and the insights already
generated, write ONE short paragraph (3-4 sentences) that introduces the
report. Do not repeat the bullet points verbatim, just summarize the story.
"""


def _format_statistics_markdown(stats: Dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.append(f"| Rows | {stats.get('row_count', '-')} |")
    lines.append(f"| Columns | {stats.get('column_count', '-')} |")

    missing = stats.get("missing_values") or {}
    missing_str = ", ".join(f"{k}: {v}" for k, v in missing.items()) if missing else "None"
    lines.append(f"| Missing values | {missing_str} |")

    md = "\n".join(lines)

    numeric_summary = stats.get("numeric_summary") or {}
    if numeric_summary:
        md += "\n\n**Numeric Summary**\n\n"
        md += "| Column | Mean | Median | Min | Max |\n|---|---|---|---|---|\n"
        for col, s in numeric_summary.items():
            md += f"| {col} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} |\n"

    grouped = stats.get("grouped_summary")
    if grouped:
        md += f"\n\n**Grouped Summary: {grouped['metric_column']} by {grouped['groupby_column']}**\n\n"
        md += "| Group | Sum | Mean | Count |\n|---|---|---|---|\n"
        for group_name, row in grouped["data"].items():
            md += f"| {group_name} | {row['sum']} | {row['mean']} | {row['count']} |\n"

    return md


def run_report_agent(
    user_question: str,
    statistics: Dict[str, Any],
    insights: str,
    charts: List[Dict[str, str]],
) -> str:
    """
    Combine statistics, chart titles, and insights into one clean markdown
    report. The opening executive summary paragraph is generated by the LLM;
    the rest is assembled deterministically from real numbers so nothing is
    hallucinated.
    """
    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPORT_SYSTEM_PROMPT),
            ("human", "User question: {question}\n\nInsights:\n{insights}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    executive_summary = chain.invoke({"question": user_question, "insights": insights})

    chart_list = "\n".join(f"- {c['title']}" for c in charts) if charts else "No charts were needed for this question."

    report = f"""# Data Analysis Report

**Question:** {user_question}

## Executive Summary
{executive_summary}

## Key Statistics
{_format_statistics_markdown(statistics)}

## Charts Generated
{chart_list}

## AI Insights
{insights}
"""
    return report
