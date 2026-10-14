# AI Data Analyst Agent

🔗 **Try it live:** [analysiscsv.streamlit.app](https://analysiscsv.streamlit.app) - upload your own CSV or use one from `data/` and ask it anything.

A simple, working AI agent that analyzes CSV data end-to-end: upload a file, ask a question in plain English, and get statistics, charts, AI-written insights, and a downloadable markdown report. Built with Streamlit, LangChain, LangGraph, and the Groq API.

This project demonstrates how **LangChain** (LLM calls) and **LangGraph** (workflow orchestration) work together in a real agentic pipeline.

## Features

- 📤 Upload any CSV and preview it instantly
- 💬 Ask questions about your data in plain English and get a direct, accurate response grounded in the real data (never guesses - if it's not in the CSV, it says so)
- 📈 Automatic statistics: rows, columns, missing values, mean/median/min/max, grouped summaries, exact distinct-value counts
- 📊 Automatic, question-relevant charts: bar, histogram, pie, scatter
- 💡 AI-generated insights: trends, anomalies, business recommendations
- 📄 A final downloadable markdown report combining everything
- 🛡️ Safe execution: every LLM-generated pandas expression is checked against an AST allow-list (no imports, file/system access, or private attributes) and run with restricted builtins
- 🔁 Self-correction: if a generated query fails, the error is sent back to the LLM for one automatic retry
- ⚡ Powered by Groq's `openai/gpt-oss-120b` for fast responses

## Architecture

```
User Question + CSV
        |
        +-------------------------------------------+
        |                                            |
        v                                            v
 [ Query/Response Agent ]                    [ LangGraph Pipeline ]      <- runs ONCE, first question only
 runs on EVERY question                              |
 (LangChain: writes + executes a                  planner  (LangChain: structured output)
  real pandas query, then phrases              /    \
  the real result - never guesses)      statistics  visualization    <- parallel
        |                                      \    /
        v                                     insight   (LangChain: LLM + prompt)
  Streamlit "Response" section                    |
  (always open, always up to date)              report   (LangChain: LLM + prompt)
                                                    |
                                                    v
                                     Statistics / Charts / Insights / Report
                                          (collapsed, generated once)
```

- **LangChain** ([docs/LANGCHAIN_USAGE.md](docs/LANGCHAIN_USAGE.md)) handles all LLM communication: prompts, structured output, parsing.
- **LangGraph** ([docs/LANGGRAPH_USAGE.md](docs/LANGGRAPH_USAGE.md)) handles the workflow: state, nodes, edges, and parallel execution.
- **Streamlit** (`main.py`) handles the UI.

## Folder Structure

```
project/
├── main.py                     # Streamlit UI
├── agent/
│   ├── agents.py               # LangChain logic (LLM, prompts, stats, charts, query agent)
│   └── graph.py                # LangGraph orchestration (state, nodes, edges)
├── data/
│   └── sample_data.csv         # Example dataset to try the app with
├── docs/
│   ├── INSTALL.md              # Step-by-step setup guide
│   ├── LANGCHAIN_USAGE.md      # Where/why LangChain is used
│   └── LANGGRAPH_USAGE.md      # Where/why LangGraph is used
├── requirements.txt            # Python dependencies
├── .env.example                # Template for your Groq API key
└── README.md                   # This file
```

## How to Run

See [docs/INSTALL.md](docs/INSTALL.md) for full step-by-step instructions. Quick version:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY inside
streamlit run main.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

## Example Questions

Try these with `data/sample_data.csv`:

- Which product generated the highest revenue?
- Which product sold the most?
- Show monthly revenue.
- What is the average sales?
- Which customer spent the most?
- Which city generated the highest revenue?
- Show the distribution of sales.
- Are there any missing values?
- What are the top five products?
- Summarize this dataset.
- Generate business insights.

## Technologies

| Technology | Purpose |
|---|---|
| Streamlit | Web UI |
| LangChain | LLM calls, prompts, structured output |
| LangGraph | Workflow orchestration (state, nodes, parallel execution) |
| Groq (`openai/gpt-oss-120b`) | LLM inference |
| Pandas | Data analysis |
| Matplotlib | Charts |

## Future Improvements

- Run LLM-generated code in a fully isolated sandbox with time and memory limits
- Support Excel (.xlsx) and JSON uploads in addition to CSV
- Let users ask follow-up questions with conversation memory
- Add more chart types (line charts for time series, box plots)
- Cache LLM responses for repeated questions to save API calls
- Export the final report as PDF in addition to markdown
