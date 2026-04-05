# DeployMate AI — Agent Guide

## Quick Commands
```bash
pip install -r requirements.txt          # Install deps
pytest tests/ -v --tb=short              # Run tests
mypy src/ --ignore-missing-imports       # Type check
streamlit run streamlit_frontend.py       # Run app
docker-compose up --build                # Full stack
```

## Common Pitfalls

**Import store in agent nodes that use memory:**
```python
from src.graph.builder import store  # Required, not optional
```

**Router config is optional:**
```python
def route_message(state: dict, config: dict | None = None) -> str:  # Can be None
```

**Memory tools return {} on failure** (don't raise):
```python
get_user_memory(store, user_id)  # Returns {} if user_id empty or read fails
save_user_memory(store, user_id, info)  # No-op if user_id empty
```

## Architecture

- **Entry point**: `src/graph/builder.py` → builds LangGraph with 6 agent nodes
- **Agents**: `chat`, `error_analyzer`, `deploy_guide`, `code_review`, `github_connector`, `docker_generator`
- **State**: Uses `ChatState` TypedDict with `messages`, `session_id`, `debug_history`, `intent`
- **Intent detection**: Keyword-based in `src/agents/router.py` (`is_error_message`, `is_deployment_request`, etc.)
- **Memory**: PostgresStore via `src/graph/builder.store` (global, lazy-initialized)
- **Checkpointer**: PostgresSaver for conversation persistence

## Docker Development

- Ollama runs on port 11435 (mapped to 11434 inside container)
- Streamlit needs `LLM_BASE_URL=http://ollama:11434` and `EMBEDDINGS_BASE_URL=http://ollama:11434`
- Add `extra_hosts: - "host.docker.internal:host-gateway"` for local model access

## Code Style

- 4 spaces, max 100 chars, snake_case functions/variables, PascalCase classes
- Use Python 3.10+ type hints: `str | None`, `list[X]`
- Use `structlog` for logging
- All public functions need docstrings

## Key Files

| File | Purpose |
|------|---------|
| `src/graph/builder.py` | Main LangGraph definition |
| `src/agents/router.py` | Intent detection & routing |
| `src/config/settings.py` | Pydantic BaseSettings |
| `src/tools/memory.py` | User memory (read/write) |
| `src/tools/rag.py` | FAISS + PDF ingestion |
| `streamlit_frontend.py` | Streamlit UI |
