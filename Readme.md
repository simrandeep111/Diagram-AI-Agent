# Diagram AI Agent

A AI Agent that converts plain-English prompts into Mermaid diagrams using Groq (Llama) via LangChain + LangGraph.

## Features
- `/generate` API that returns JSON: `{ "code": "..." }`
- Mermaid syntax validation
- Post-processing fixes for picky Mermaid formats (class, sankey, journey, etc.)
- Simple web UI for testing

## Setup
```bash
git clone <repo-url>
cd diagram-ai-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt