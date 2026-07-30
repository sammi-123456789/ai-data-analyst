# Installation Guide

This guide walks you through setting up and running the **AI Data Analyst Agent** on your machine, step by step. No prior experience needed.

## 1. Install Python

You need **Python 3.10 or newer**.

Check if you already have it:

```bash
python --version
```

If you don't have Python, download it from [python.org/downloads](https://www.python.org/downloads/) and install it (make sure to check "Add Python to PATH" during setup on Windows).

## 2. Create a Virtual Environment

A virtual environment keeps this project's packages separate from the rest of your system.

### Windows (PowerShell or Command Prompt)

```bash
python -m venv .venv
```

### Linux / macOS

```bash
python3 -m venv .venv
```

## 3. Activate the Virtual Environment

### Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)

```bash
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

You'll know it worked because your terminal prompt will now start with `(.venv)`.

## 4. Install Requirements

```bash
pip install -r requirements.txt
```

This installs Streamlit, LangChain, LangGraph, Pandas, Matplotlib, and everything else the app needs.

## 5. Create Your `.env` File

Copy the example file:

**Windows:**
```bash
copy .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

## 6. Add Your Groq API Key

Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).

Open the `.env` file in any text editor and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Save the file. **Never share this file or commit it to git.**

## 7. Run the App

```bash
streamlit run main.py
```

Your browser should automatically open to `http://localhost:8501`. If it doesn't, copy that URL into your browser manually.

## 8. Try It Out

1. In the sidebar, upload `sample_data.csv` (included in this project).
2. Wait for the green "Loaded" confirmation.
3. Type a question, e.g. *"Which product generated the highest revenue?"*
4. Click **Analyze**.
5. Expand the **Statistics**, **Charts**, **AI Insights**, and **Final Report** sections to see the results.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GROQ_API_KEY is missing` error | Make sure `.env` exists in the project root and contains a valid key, then restart Streamlit. |
| `streamlit: command not found` | Make sure your virtual environment is activated before running the command. |
| PowerShell blocks `Activate.ps1` | Run PowerShell as Administrator and execute: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| CSV upload fails | Make sure the file is a valid, non-empty `.csv` file. |
| Charts don't show up | Make sure `matplotlib` installed correctly (`pip show matplotlib`). |
| Slow or failed AI responses | Check your internet connection and that your Groq API key is valid and has quota remaining. |

## Expected Output

When everything works, you should see:

- A clean web page titled **"AI Data Analyst Agent"**
- A sidebar with upload, API status (green), and reset controls
- A data preview table after uploading a CSV
- After clicking **Analyze**: statistics tables, relevant charts, an AI-written insights section, and a downloadable markdown report
