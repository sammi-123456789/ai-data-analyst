# How LangChain Is Used In This Project

This document explains, in simple terms, where and why LangChain is used.

## What LangChain Does Here (and What It Doesn't)

**LangChain is ONLY used for talking to the LLM.** That's it. Things like:

- Sending a well-structured prompt to the Groq model
- Getting the model's reply back in a predictable, validated shape
- Turning the model's raw text reply into a clean string

**LangChain is NOT used to control the flow of the app.** Deciding "run the statistics agent, then the chart agent, then the insight agent" is [LangGraph](LANGGRAPH_USAGE.md)'s job, not LangChain's. Keeping these two responsibilities separate is one of the main things this demo is meant to show.

## Which File Uses LangChain?

All LangChain code lives in **[agents.py](agents.py)**. `main.py` and `graph.py` never import `langchain` directly.

## The Building Blocks Used

### 1. The LLM Client

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=..., temperature=0.2)
```

`ChatGroq` is LangChain's wrapper around the Groq API. It gives us a consistent interface (`.invoke()`, prompt piping with `|`, etc.) no matter which LLM provider we use underneath.

### 2. Prompt Templates

Instead of writing a raw string every time, we use `ChatPromptTemplate` to define reusable prompts with placeholders:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly senior data analyst..."),
    ("human", "User question: {question}\n\nStatistics:\n{stats}"),
])
```

This keeps instructions to the model (the "system" message) separate from the actual data (the "human" message), and makes prompts easy to read and edit.

### 3. Structured Output (Planner Agent)

The Planner Agent needs the model's answer as a *real Python object*, not a paragraph of text. We define what we want using a Pydantic model:

```python
class AnalysisPlan(BaseModel):
    intent_summary: str
    groupby_column: Optional[str]
    metric_column: Optional[str]
    chart_types: List[str]
    needs_grouped_summary: bool

structured_llm = llm.with_structured_output(AnalysisPlan)
```

LangChain handles telling the model exactly what shape of JSON to return, and automatically parses/validates the response into an `AnalysisPlan` object. If the model's reply doesn't fit the schema, LangChain raises an error instead of silently passing along garbage.

### 4. Output Parsing (Insight & Report Agents)

For the Insight and Report agents, we just want clean text back, so we use `StrOutputParser`:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
insights = chain.invoke({"question": question, "stats": stats})
```

The `|` (pipe) syntax is LangChain's "chain" (aka LCEL - LangChain Expression Language). It means: *build the prompt, send it to the LLM, then parse the output* - all in one readable line.

### 5. Messages

Every prompt is built from `system` and `human` messages, mirroring how chat-based LLMs actually work: the system message sets behavior/rules, the human message carries the real request and data.

## Example Flow: The Insight Agent

1. `graph.py` calls `agents.run_insight_agent(question, statistics)`.
2. Inside `agents.py`, a `ChatPromptTemplate` is filled in with the question and stats.
3. The filled prompt is piped into the Groq LLM (`ChatGroq`).
4. The LLM's response is piped into `StrOutputParser` to get plain text.
5. That plain text (markdown bullet points) is returned to the graph, which stores it in the shared state.

That's the entire LangChain responsibility in this project: **prompt in, structured/clean result out.** Everything about *when* and *in what order* that happens is handled by LangGraph.
