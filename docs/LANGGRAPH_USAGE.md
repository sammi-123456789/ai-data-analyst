# How LangGraph Is Used In This Project

This document explains, in simple terms, why LangGraph is used and how the agent pipeline is structured.

## Why LangGraph?

LangChain is great at talking to an LLM, but this app needs more than one LLM/logic step - it needs a **pipeline** of agents that run in a specific order, where some steps can run **at the same time**, and where every step needs access to shared data (the dataframe, the question, the results so far).

LangGraph is built exactly for this: it lets us describe the pipeline as a graph of **nodes** (steps) connected by **edges** (the order they run in), all sharing one **state** object. This is what makes the flow easy to read, easy to extend, and easy to run partially in parallel.

## All LangGraph code lives in [agent/graph.py](../agent/graph.py)

`agent/graph.py` never talks to the Groq API directly - it only calls functions from `agent/agents.py` inside each node. This keeps orchestration (LangGraph) and LLM logic (LangChain) cleanly separated.

## The State

The state is a single `TypedDict` that gets passed between every node. Each node reads what it needs from the state and returns a small dictionary of updates, which LangGraph merges back into the state automatically.

```python
class AgentState(TypedDict, total=False):
    user_question: str
    dataframe: pd.DataFrame
    plan: AnalysisPlan
    statistics: dict
    charts: list
    insights: str
    report: str
    error: str | None
```

Think of the state as a shared clipboard that every agent can read from and write to.

## The Graph Structure

```
              START
                |
                v
            [ planner ]
             /        \
            v          v
     [statistics]  [visualization]     <-- run in PARALLEL
            \          /
             v        v
           [ insight ]                 <-- automatically waits for
                |                          BOTH branches (the "merge")
                v
            [ report ]
                |
                v
               END
```

## Nodes

| Node | What it does | Uses LangChain? |
|---|---|---|
| `planner` | Reads the question + column names, decides what stats/charts are needed | Yes (structured output) |
| `statistics` | Computes rows, columns, missing values, means, medians, group summaries | No (pure pandas) |
| `visualization` | Builds only the charts relevant to the question | No (pure matplotlib) |
| `insight` | Turns statistics into plain-English findings and business insights | Yes (prompt + LLM) |
| `report` | Combines everything into one clean markdown report | Yes (prompt + LLM for the summary) |

## Edges and Parallel Execution

```python
graph.add_edge(START, "planner")
graph.add_edge("planner", "statistics")
graph.add_edge("planner", "visualization")
graph.add_edge("statistics", "insight")
graph.add_edge("visualization", "insight")
graph.add_edge("insight", "report")
graph.add_edge("report", END)
```

Notice that **both** `statistics` and `visualization` have an edge coming out of `planner`. LangGraph sees this as a fan-out: as soon as `planner` finishes, it runs `statistics` and `visualization` in the same execution step (in parallel), since neither depends on the other's output.

Then, notice that **both** `statistics` and `visualization` have an edge going into `insight`. LangGraph automatically waits until *every* incoming edge into a node has completed before running that node. This is the "merge" step - no extra code needed, it's just how the graph is wired.

## Execution Flow (Step by Step)

1. **START → planner**: The planner agent reads the dataframe's columns and the user's question, and produces a structured `AnalysisPlan`.
2. **planner → statistics** and **planner → visualization** (parallel): The statistics agent computes numbers; the visualization agent draws charts. Neither needs the other's output, so LangGraph runs them together.
3. **statistics + visualization → insight**: Once both are done, the insight agent reads the statistics and writes plain-English findings.
4. **insight → report**: The report agent combines statistics, chart titles, and insights into one markdown report.
5. **report → END**: The final state (containing `statistics`, `charts`, `insights`, `report`) is returned to `main.py` for display.

## Why The Response Isn't A Graph Node

The UI also has a **Response** section that directly answers whatever
question you type, every time. It is deliberately NOT part of this
LangGraph pipeline - it's a single LangChain chain (`agents.run_query_agent`)
that runs once per question, independent of the graph above.

Why keep it separate: the pipeline above (planner → statistics/visualization
→ insight → report) is meant to build one general **overview** of the
dataset, and per the product requirement it only needs to run **once**
(from your first question). Re-running that whole graph for every follow-up
question would be slow and pointless. The Response, on the other hand,
needs to change on every question, so `main.py` calls it directly instead of
routing it through the graph. LangGraph is used where there's an actual
multi-step pipeline to orchestrate; a single LLM lookup doesn't need one.

## Why This Matters (Advantages)

- **Clarity**: The pipeline is defined as a graph, not buried in nested if/else logic.
- **Parallelism**: Independent work (stats vs. charts) genuinely runs together instead of one waiting on the other, which is faster and mirrors how a real analyst team would split work.
- **Shared State**: Every node has access to everything computed so far, without manually passing arguments around.
- **Extensibility**: Adding a new agent (e.g. an "Anomaly Detection" node) is as simple as adding one node and rewiring a couple of edges.
- **Separation of Concerns**: LangGraph handles *flow*; LangChain (used inside nodes) handles *talking to the LLM*. Each tool does the one job it's best at.
