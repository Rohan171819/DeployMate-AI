"""Chat agent node."""

from __future__ import annotations

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.router import is_dangerous, extract_user_info
from src.config.settings import settings
from src.exceptions import DangerousCommandError
from src.tools.memory import get_user_memory, save_user_memory
from src.tools.rag import get_rag_context_with_fallback

logger = structlog.get_logger()


# System prompt for main chat
SYSTEM_PROMPT = SystemMessage(
    content="""
You are DeployMate AI — an expert DevOps co-pilot for junior developers.
You specialize in:
- Docker errors and containerization
- CI/CD pipeline setup and failures
- Production error log debugging
- Cloud deployment (AWS, Railway, Render, VPS)
- Code review for security and performance

Always provide clear, step-by-step, beginner-friendly guidance.
When user shares an error, identify root cause first, then provide exact fix.
"""
)


def chat_node(state: dict, config: RunnableConfig) -> dict:
    """Process user chat message with RAG and memory context.

    Handles:
    - Long-term memory retrieval and persistence
    - User profile extraction (tech stack, experience level)
    - RAG context retrieval from uploaded PDFs
    - Dangerous command detection with HITL

    Args:
        state: Current chat state containing messages.
        config: RunnableConfig with thread_id and other metadata.

    Returns:
        Updated state with LLM response added to messages.

    Raises:
        DangerousCommandError: When command requires human approval.
    """
    from langgraph.types import interrupt

    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    logger.info("chat_node_started", thread_id=thread_id, message_prefix=query[:50])

    user_memory = get_user_memory(store, thread_id)

    new_info = extract_user_info(query)
    if new_info:
        save_user_memory(store, thread_id, new_info)
        user_memory.update(new_info)

    memory_context = ""
    if user_memory:
        memory_context = f"""
USER PROFILE (Remember this):
- Tech Stack: {user_memory.get("tech_stack", "Unknown")}
- Experience: {user_memory.get("experience", "Unknown")}
"""

    if is_dangerous(query):
        logger.warning("dangerous_command_detected", thread_id=thread_id)
        human_decision = interrupt(
            {
                "type": "dangerous_command",
                "message": "Dangerous command detected!",
                "suggested_response": f"User asked for dangerous command: {query}",
            }
        )
        if not human_decision.get("approved"):
            logger.info("dangerous_command_rejected", thread_id=thread_id)
            return {
                "messages": [
                    SystemMessage(
                        content="""
I recommend NOT running this command — it can be destructive!
Safer alternatives:
- `du -sh /var/log/*` — check log sizes first
- `journalctl --vacuum-size=100M` — safely clean logs
- `find /var/log -name "*.gz" -delete` — only compressed logs
"""
                    )
                ]
            }

    crag = get_rag_context_with_fallback(thread_id, query)
    rag_context = crag.get("context", "")

    dynamic_prompt = SystemMessage(
        content=f"""
{SYSTEM_PROMPT.content}
{memory_context if memory_context else ""}
{f"DOCUMENT CONTEXT (Answer based on this): {rag_context}" if rag_context else ""}
"""
    )

    messages = [dynamic_prompt] + state["messages"]
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        streaming=True,
    )
    response = llm.invoke(messages)

    logger.info("chat_node_completed", thread_id=thread_id)
    return {"messages": [response]}
