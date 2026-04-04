# AGENTS.md — DeployMate AI Developer Guide

This file provides guidelines for agentic coding agents working on this codebase.

---

## 1. Build, Test & Development Commands

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run a single test
pytest tests/unit/test_nodes.py::TestIntentDetection::test_router_detects_error_intent -v

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Syntax Check
```bash
python -m py_compile src/config/settings.py
python -m py_compile src/graph/builder.py
```

### Type Checking
```bash
mypy src/ --ignore-missing-imports
```

### Running the Application
```bash
# Start the Streamlit frontend
streamlit run streamlit_frontend.py
```

### Docker
```bash
docker-compose up --build
```

---

## 2. Code Style Guidelines

### Imports

The project uses these key libraries:
- **LangChain/LangGraph**: `from langgraph.graph import StateGraph, START, END`
- **Typing**: Use Python 3.10+ syntax (`str | None`, `list[X]`, `dict[K, V]`)
- **Pydantic**: `from pydantic import BaseModel, Field`
- **LangChain Core**: `from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage`

Order imports by:
1. Standard library (`os`, `tempfile`, `typing`)
2. Third-party (`langchain`, `pydantic`, `pytest`)
3. Local application (`src.*`)

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Add blank lines between logical sections
- Use consistent spacing: `config = {}` not `config={}`

### Type Hints

Always use type hints for function parameters and return types:

```python
from __future__ import annotations

def is_error_message(message: str) -> bool:
    ...

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `is_error_message`, `db_uri`)
- **Classes**: `PascalCase` (e.g., `ChatState`, `GradeScore`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DANGEROUS_KEYWORDS`)
- **Files**: `snake_case` with descriptive names

### Error Handling

- Use try/except blocks for operations that may fail
- Catch specific exceptions from `src.exceptions`
- Use `structlog` for structured logging
- Re-raise caught exceptions as custom exceptions

### Documentation

Use docstrings for all public functions:

```python
def is_error_message(message: str) -> bool:
    """Detect if message contains error-related keywords.

    Args:
        message: User message to analyze.

    Returns:
        True if error keywords found, False otherwise.
    """
```

### Testing Guidelines

- Test files go in `tests/` directory
- Use `conftest.py` for shared fixtures
- Use descriptive test names: `test_<function>_<expected_behavior>`
- Test both positive and negative cases
- Mock external dependencies (LLM, database)

---

## 3. Project Structure

```
DeployMate_AI/
├── src/
│   ├── config/settings.py          # Pydantic BaseSettings
│   ├── agents/                     # Agent nodes
│   │   ├── chat.py                 # chat_node
│   │   ├── error_analyzer.py       # error_analyzer_node
│   │   ├── deployment.py          # deployment_guide_node
│   │   ├── code_review.py         # code_review_node
│   │   └── router.py               # route_message + intent detection
│   ├── tools/                      # Utilities
│   │   ├── rag.py                  # FAISS retrieval + PDF ingestion
│   │   └── memory.py               # User memory operations
│   ├── graph/builder.py            # LangGraph construction
│   └── exceptions.py               # Custom exception hierarchy
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/test_nodes.py          # Unit tests
│   └── integration/test_graph.py   # Integration tests
├── streamlit_frontend.py
├── requirements.txt
├── docker-compose.yml
└── dockerfile
```

---

## 4. Key Technologies

- **LLM**: Ollama (qwen2.5-coder:0.5b)
- **Embeddings**: Ollama (nomic-embed-text)
- **Frontend**: Streamlit
- **Database**: PostgreSQL + FAISS (vector store)
- **Framework**: LangGraph (agent orchestration)

---

## 5. Configuration (src/config/settings.py)

All configuration centralized in Pydantic BaseSettings:

| Variable | Default | Description |
|----------|---------|-------------|
| `database_url` | postgresql://... | PostgreSQL connection |
| `llm_model` | qwen2.5-coder:0.5b | Ollama chat model |
| `llm_base_url` | http://host.docker.internal:11434 | Ollama API URL |
| `embeddings_model` | nomic-embed-text | Ollama embeddings |
| `langchain_api_key` | (empty) | LangSmith API key (optional) |
| `tavily_api_key` | (empty) | Tavily search API (optional) |

---

## 6. Common Patterns

### Importing the chatbot
```python
from src.graph.builder import chatbot
```

### Using settings
```python
from src.config.settings import settings
llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url)
```

### Custom exceptions
```python
from src.exceptions import MemoryReadError, PDFIngestionError

try:
    result = get_user_memory(store, user_id)
except MemoryReadError as e:
    logger.error("memory_read_failed", user_id=e.user_id)
```

### Creating a new agent node
```python
def my_agent_node(state: MyState, config: RunnableConfig) -> dict:
    return {"messages": [response]}
```

---

## 7. Notes

- The backend uses LangGraph's interrupt mechanism for human-in-the-loop
- Dangerous commands are intercepted using keyword matching (`is_dangerous`)
- Message intent is detected via keyword analysis (error, deploy, code review)
- Use `structlog` for structured logging throughout
- All nodes should have Google-style docstrings
