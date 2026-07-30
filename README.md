# AI Data Analyst Agent

A simple, working demo of an AI agent that analyzes CSV data end-to-end: upload a file, ask a question in plain English, and get statistics, charts, AI-written insights, and a downloadable markdown report - all running locally with **Streamlit**, **LangChain**, **LangGraph**, and the **Groq** API.

This project is built as a clean demonstration of how **LangChain** (LLM calls) and **LangGraph** (agent orchestration) work together in a real multi-agent pipeline.

## Features

- 📤 Upload any CSV and preview it instantly
- 💬 Ask questions about your data in plain English
- 📈 Automatic statistics: rows, columns, missing values, mean/median/min/max, grouped summaries
- 📊 Automatic, question-relevant charts: bar, histogram, pie, scatter
- 💡 AI-generated insights: trends, anomalies, business recommendations
- 📄 A final downloadable markdown report combining everything
- ⚡ Powered by Groq's `llama-3.3-70b-versatile` for fast responses
- 🖥️ 100% local - no Docker, no database, nothing to deploy

## Architecture

```
User Question + CSV
        |
        v
   [ LangGraph Pipeline ]
        |
     planner  (LangChain: structured output)
      /    \
statistics  visualization    <- parallel
      \    /
     insight   (LangChain: LLM + prompt)
        |
      report   (LangChain: LLM + prompt)
        |
        v
  Streamlit UI displays results
```

- **LangChain** ([LANGCHAIN_USAGE.md](LANGCHAIN_USAGE.md)) handles all LLM communication: prompts, structured output, parsing.
- **LangGraph** ([LANGGRAPH_USAGE.md](LANGGRAPH_USAGE.md)) handles the agent pipeline: state, nodes, edges, and parallel execution.
- **Streamlit** ([main.py](main.py)) handles the UI.

## Folder Structure

```
project/
├── main.py                  # Streamlit UI
├── agents.py                 # LangChain logic (LLM, prompts, stats, charts)
├── graph.py                  # LangGraph orchestration (state, nodes, edges)
├── requirements.txt          # Python dependencies
├── .env.example               # Template for your Groq API key
├── sample_data.csv            # Example dataset to try the app with
├── README.md                  # This file
├── INSTALL.md                  # Step-by-step setup guide
├── LANGCHAIN_USAGE.md           # Where/why LangChain is used
└── LANGGRAPH_USAGE.md            # Where/why LangGraph is used
```

## Screenshots

*(Add screenshots here after running the app locally - e.g. `screenshots/upload.png`, `screenshots/report.png`)*

## How to Run

See [INSTALL.md](INSTALL.md) for full step-by-step instructions. Quick version:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY inside
streamlit run main.py
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).

## Example Questions

Try these with `sample_data.csv`:

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
| [Streamlit](https://streamlit.io/) | Web UI |
| [LangChain](https://python.langchain.com/) | LLM calls, prompts, structured output |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-agent orchestration |
| [Groq](https://groq.com/) (`llama-3.3-70b-versatile`) | LLM inference |
| [Pandas](https://pandas.pydata.org/) | Data analysis |
| [Matplotlib](https://matplotlib.org/) | Charts |

## Future Improvements

- Support Excel (`.xlsx`) and JSON uploads in addition to CSV
- Let users ask follow-up questions with conversation memory
- Add more chart types (line charts for time series, box plots)
- Cache LLM responses for repeated questions to save API calls
- Export the final report as PDF in addition to markdown

---

Built as a client demo to showcase LangChain + LangGraph working together in a real, runnable application.
