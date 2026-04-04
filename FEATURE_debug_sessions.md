# Feature Specification: Multi-Turn Debugging Sessions with Memory Continuity

## Feature Overview
**Category**: Agent Capability  
**Complexity**: Medium (4-6 hours)  
**Problem Solves**: Junior developers run iterative debugging sessions, but stateless agents forget context across turns, forcing users to repeat error messages.

---

## Implementation Requirements

### 1. Extend ChatState

Modify `src/graph/builder.py` to include debug session fields:

```python
class ChatState(dict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str | None
    debug_history: list[dict]
    intent: str | None
```

### 2. Create DebugSessionManager

Create `src/tools/debug_session.py`:

- `DebugSession` TypedDict with: `session_id`, `user_id`, `errors`, `iteration`
- `init_debug_session(state: ChatState) -> dict`: Generate session_id if not present
- `add_error_to_session(session_id: str, error: str, resolution: str | None = None) -> None`
- `get_session_errors(session_id: str, limit: int = 5) -> list[dict]`
- `detect_follow_up(current_error: str, prior_errors: list[dict]) -> bool`: Use LLM to detect if current error is related to prior ones

### 3. Create Debug Table

Add to `src/graph/builder.py` `_init_db()`:

```python
def _init_db():
    # ... existing code ...
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS debug_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW(),
                iteration_count INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS debug_errors (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) REFERENCES debug_sessions(session_id),
                error_text TEXT,
                resolution TEXT,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """)
```

### 4. Modify Router

In `src/agents/router.py`, add logic to:
- Check if message contains error patterns
- Pass session_id to error_analyzer_node

### 5. Update error_analyzer_node

In `src/agents/error_analyzer.py`:

```python
def error_analyzer_node(state: ChatState, config: RunnableConfig) -> dict:
    prior_errors = state.get("debug_history", [])
    session_id = state.get("session_id")
    current_msg = state["messages"][-1].content
    
    # Build context from prior errors
    context = f"Session error history ({len(prior_errors)} prior errors):\n"
    for e in prior_errors[-3:]:
        context += f"- {e.get('error', 'N/A')}\n"
    
    # Check if this is a follow-up error using LLM
    is_follow_up = detect_follow_up(current_msg, prior_errors)
    
    # Invoke LLM with context
    response = llm.invoke(f"{context}\nCurrent error: {current_msg}")
    
    # Update debug history
    new_history = prior_errors + [{"error": current_msg, "iteration": len(prior_errors) + 1}]
    
    return {
        "messages": [response],
        "debug_history": new_history
    }
```

### 6. Surface Related Errors in Response

Add metadata to responses indicating prior session errors:

```python
response.response_metadata = {
    "related_errors": prior_errors[-3:],
    "is_follow_up": is_follow_up,
    "session_id": session_id
}
```

---

## Code Patterns to Follow

- Use `structlog` for logging (see `src/tools/memory.py`)
- Use Pydantic-style TypedDict with proper annotations
- Follow import ordering: stdlib → third-party → local
- Use 4 spaces indentation, max 100 char line length
- Use `src.exceptions` for custom exceptions

---

## Integration Points

| Component | Location | Changes |
|-----------|----------|---------|
| ChatState | `src/graph/builder.py:31` | Add session_id, debug_history |
| DebugSessionManager | `src/tools/debug_session.py` | New file |
| Router | `src/agents/router.py` | Pass session_id to nodes |
| error_analyzer | `src/agents/error_analyzer.py` | Use prior context |
| _init_db | `src/graph/builder.py:69` | Create debug tables |

---

## Testing

- Unit tests in `tests/unit/test_debug_session.py`
- Test session creation, error tracking, follow-up detection
- Integration test for multi-turn debugging flow
