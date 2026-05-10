<div align="center">

<img src="https://raw.githubusercontent.com/your-username/deploymate-ai/main/assets/logo.png" alt="DeployMate AI Logo" width="120" />

# 🚀 DeployMate AI

### Production-Grade Multi-Agent DevOps Co-Pilot

> _Stop firefighting deployments. Let AI agents handle the chaos._

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3B4D?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3B4D?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented-FF6B35?style=for-the-badge&logo=elasticsearch&logoColor=white)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📌 What It Does

**DeployMate AI** is a multi-agent DevOps co-pilot built on **LangGraph** that automates the most painful parts of the deployment lifecycle — log triage, CI/CD failure diagnosis, infrastructure analysis, and incident response — through a coordinated swarm of specialized AI agents.

Instead of a single bloated LLM prompt, DeployMate uses a **supervisor-worker agent graph** where each agent owns a specific DevOps domain. Agents communicate, delegate, and synthesize — producing actionable insights, not walls of text.

### 🎯 Core Capabilities

| Capability | What Happens |
|---|---|
| 🔍 **Log Analyzer Agent** | Ingests raw logs, pinpoints errors, and ranks root causes by severity |
| 🏗️ **CI/CD Debugger Agent** | Diagnoses pipeline failures, maps them to code changes, and suggests fixes |
| 📊 **Infrastructure Advisor Agent** | Analyzes resource configs and flags bottlenecks, misconfigurations, or cost leaks |
| 🚨 **Incident Responder Agent** | Drafts a structured runbook in real-time during live incidents |
| 🤖 **Supervisor Agent** | Routes tasks, manages inter-agent handoffs, and assembles the final response |
| 📚 **RAG Knowledge Base** | Grounds agent responses in your private docs, wikis, and past post-mortems |

---

## 🏛️ Architecture

```
                        ┌─────────────────────────────────────┐
                        │           USER / CLI / API           │
                        └────────────────┬────────────────────┘
                                         │  Query / Task
                                         ▼
                        ┌─────────────────────────────────────┐
                        │         SUPERVISOR AGENT            │
                        │   (LangGraph StateGraph Orchestrator)│
                        │  • Intent classification            │
                        │  • Agent routing & delegation       │
                        │  • Response synthesis               │
                        └────┬──────────┬──────────┬──────────┘
                             │          │          │
              ┌──────────────▼──┐  ┌────▼──────┐  ┌▼─────────────────┐
              │  Log Analyzer   │  │  CI/CD    │  │  Infra Advisor   │
              │     Agent       │  │  Debugger │  │     Agent        │
              │                 │  │   Agent   │  │                  │
              │ • Error triage  │  │ • Pipeline│  │ • Resource audit │
              │ • Root cause    │  │   parsing │  │ • Cost analysis  │
              │ • Severity rank │  │ • Fix gen │  │ • Misconfig scan │
              └────────┬────────┘  └─────┬─────┘  └────────┬─────────┘
                       │                 │                  │
                       └────────┬────────┘                  │
                                │                           │
                        ┌───────▼───────────────────────────▼────────┐
                        │          INCIDENT RESPONDER AGENT           │
                        │     (Activated on critical severity)        │
                        │   • Live runbook generation                 │
                        │   • Escalation path suggestions             │
                        └───────────────────┬─────────────────────────┘
                                            │
                        ┌───────────────────▼─────────────────────────┐
                        │           RAG KNOWLEDGE BASE                 │
                        │  (FAISS / Chroma + Post-mortems + Wikis)    │
                        └─────────────────────────────────────────────┘
```

### Agent Communication Flow (LangGraph State)

```python
# Simplified state schema
class DeployMateState(TypedDict):
    task: str                    # Original user query
    agent_scratchpad: list       # Inter-agent messages
    retrieved_context: str       # RAG output
    analysis_results: dict       # Per-agent findings
    final_response: str          # Synthesized output
    severity: Literal["low", "medium", "critical"]
```

---

## 🖥️ Demo

> **Replace this section** with your actual GIF/screenshot after recording a demo run.

```
📹  Suggested recording flow:
    1. Run: python main.py --task "Diagnose this GitHub Actions failure" --log ./sample_logs/ci_fail.log
    2. Show the agent graph spinning up in terminal
    3. Show the structured diagnosis output
    4. Record with: ttyrec + ttygif  OR  asciinema
```

**Screenshot placeholder — drop your demo here:**

```
┌─────────────────────────────────────────────────────────┐
│  DeployMate AI  v1.0                    [● LIVE]        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  > Task: Diagnose CI failure on PR #47                  │
│                                                         │
│  [Supervisor]  → Routing to CI/CD Debugger Agent        │
│  [CI/CD Agent] → Parsing pipeline YAML...              │
│  [CI/CD Agent] → Error: ModuleNotFoundError at step 3  │
│  [Log Agent]   → Confirmed: missing dependency in      │
│                  requirements.txt                       │
│  [Supervisor]  → Synthesizing response...              │
│                                                         │
│  ✅ Root Cause: `torch==2.1.0` not pinned in           │
│     requirements.txt. Added in PR branch but not       │
│     base. Fix: merge dependency update first.          │
│                                                         │
│  📋 Confidence: 94%  │  Agents Used: 3  │  Time: 4.2s  │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph 0.2+ | Multi-agent state graph & routing |
| **LLM Framework** | LangChain 0.3+ | Tool calling, prompts, chains |
| **LLM** | Gemini 1.5 Flash | Primary inference (complex agents) |
| **Retrieval** | FAISS / Chroma | Vector store for RAG |
| **Embeddings** | Google text-embedding-004 | Document vectorization |
| **Backend API** | FastAPI | REST interface for agent graph |
| **Containerization** | Docker + Compose | Reproducible deployment |
| **CI/CD** | GitHub Actions | Automated testing & lint |
| **Language** | Python 3.11+ | Core implementation |

---

## ⚡ How to Run

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Google AI API Key (for Gemini) OR OpenAI API Key

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/deploymate-ai.git
cd deploymate-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

```env
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_key_here          # optional fallback

# Vector Store
CHROMA_PERSIST_DIR=./vectorstore
EMBEDDING_MODEL=models/text-embedding-004

# App Settings
LOG_LEVEL=INFO
MAX_AGENT_ITERATIONS=10
SUPERVISOR_MODEL=gemini-1.5-flash
```

### 3. Run with Docker (Recommended)

```bash
docker-compose up --build
```

The API will be live at `http://localhost:8000`.

### 4. Run Locally

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --port 8000

# OR run CLI directly
python main.py --task "Analyze these logs" --log ./sample_data/error.log
```

### 5. Ingest Your Knowledge Base (RAG)

```bash
# Add your docs/post-mortems/wikis
python scripts/ingest.py --source ./docs/ --type markdown

# Supported: markdown, PDF, plain text
```

### 6. API Usage

```bash
# POST a DevOps task
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "task": "My GitHub Actions pipeline is failing at the Docker build step",
    "logs": "ERROR: failed to solve: process \"/bin/sh -c pip install...\" did not complete"
  }'
```

**Response:**

```json
{
  "root_cause": "Dependency resolution failure during Docker build",
  "severity": "medium",
  "fix": "Pin package versions in requirements.txt. Detected conflict: numpy>=1.24 vs tensorflow==2.12 constraint.",
  "agents_used": ["supervisor", "ci_debugger", "log_analyzer"],
  "confidence": 0.91,
  "runbook_url": null
}
```

---

## 📁 Project Structure

```
deploymate-ai/
├── agents/
│   ├── supervisor.py          # Routing + synthesis
│   ├── log_analyzer.py        # Log triage agent
│   ├── cicd_debugger.py       # Pipeline failure agent
│   ├── infra_advisor.py       # Infrastructure agent
│   └── incident_responder.py  # Incident runbook agent
├── graph/
│   └── workflow.py            # LangGraph StateGraph definition
├── rag/
│   ├── ingest.py              # Document ingestion pipeline
│   └── retriever.py           # RAG retrieval chain
├── app/
│   └── main.py                # FastAPI entrypoint
├── scripts/
│   └── ingest.py              # CLI ingestion tool
├── sample_data/               # Sample logs & CI configs for testing
├── tests/                     # Unit + integration tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=agents --cov-report=term-missing
```

---

## 🗺️ Roadmap

- [x] Multi-agent LangGraph orchestration
- [x] RAG knowledge base with FAISS/Chroma
- [x] FastAPI REST interface
- [x] Docker Compose deployment
- [ ] Slack / Teams alert integration
- [ ] GitHub webhook for real-time CI failure interception
- [ ] Agent memory with conversation history
- [ ] Multi-tenant SaaS mode

---

## 👤 Author

**Rohan** — AI/ML Engineer · LangGraph Developer  
MCA @ GL Bajaj Institute of Technology & Management  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/your-username)

---

<div align="center">

**⭐ Star this repo if DeployMate saved your deployment**

</div>

