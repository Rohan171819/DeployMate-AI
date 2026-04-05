"""Error analyzer agent node."""

from __future__ import annotations

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.router import is_dangerous
from src.config.settings import settings
from src.exceptions import DangerousCommandError
from src.graph.builder import store
from src.tools.debug_session import (
    add_error_to_session,
    detect_follow_up,
    init_debug_session,
)
from src.tools.memory import get_user_memory
from src.tools.rag import get_rag_context_with_fallback

logger = structlog.get_logger()

ERROR_ANALYZER_PROMPT = SystemMessage(
    content="""
You are the Error Analyzer Agent inside DeployMate AI.
When given an error, ALWAYS respond in this exact structure:

## Root Cause
Explain what caused this error in simple language.

## Why It Happened
Explain the reason behind this error.

## Exact Fix
Provide the exact fix with code snippet.

## Prevention
How to avoid this error in future.

Be beginner-friendly, clear, and precise.
"""
)


def error_analyzer_node(state: dict, config: RunnableConfig) -> dict:
    """Analyze and provide solution for error messages.

    Handles:
    - RAG context retrieval from uploaded PDFs
    - User memory context (tech stack, experience level)
    - Debug session context for multi-turn debugging
    - Dangerous command detection with HITL

    Args:
        state: Current chat state with error message in messages.
        config: RunnableConfig with thread_id and other metadata.

    Returns:
        Updated state with error analysis response added to messages.

    Raises:
        DangerousCommandError: When command requires human approval.
    """
    from langgraph.types import interrupt
    from langchain_core.messages import AIMessage

    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    state_with_session = init_debug_session(state)
    session_id = state_with_session.get("session_id")
    prior_errors = state.get("debug_history", [])

    logger.info(
        "error_analyzer_started",
        thread_id=thread_id,
        session_id=session_id,
        message_prefix=query[:50],
    )

    context = ""
    is_follow_up = False

    if prior_errors:
        context = f"Session error history ({len(prior_errors)} prior errors):\n"
        for e in prior_errors[-3:]:
            context += f"- {e.get('error', 'N/A')}\n"
        is_follow_up = detect_follow_up(query, prior_errors)
        if is_follow_up:
            context += (
                "\n⚠️ This appears to be a follow-up error related to prior issues.\n"
            )

    crag = get_rag_context_with_fallback(thread_id, query)
    rag_context = crag.get("context", "")

    user_memory = get_user_memory(store, thread_id)
    memory_context = ""
    if user_memory:
        memory_context = f"""
USER PROFILE:
- Tech Stack: {user_memory.get("tech_stack", "Unknown")}
- Experience: {user_memory.get("experience", "Unknown")}
"""

    if is_dangerous(query):
        logger.warning("dangerous_command_in_error", thread_id=thread_id)
        human_decision = interrupt(
            {
                "type": "dangerous_command",
                "message": "Dangerous command detected!",
                "suggested_response": f"User asked for dangerous command: {query}",
            }
        )
        if human_decision.get("approved"):
            prompt = SystemMessage(
                content=f"""
{ERROR_ANALYZER_PROMPT.content}
{context if context else ""}
{memory_context if memory_context else ""}
{f"DOCUMENT CONTEXT: {rag_context}" if rag_context else ""}
"""
            )
            messages = [prompt] + state["messages"]
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
            )
            response = llm.invoke(messages)

            if session_id:
                add_error_to_session(session_id, query, response.content)

            logger.info("error_analysis_completed_approved", thread_id=thread_id)
            return {"messages": [response]}
        else:
            logger.info("dangerous_command_rejected_in_error", thread_id=thread_id)
            return {
                "messages": [
                    SystemMessage(
                        content="""
I recommend NOT running this command as it can be destructive.
Safer alternatives:
- Use `du -sh /var/log/*` to check log sizes first
- Use `journalctl --vacuum-size=100M` to safely clean logs
- Use `find /var/log -name "*.gz" -delete` for compressed logs only
"""
                    )
                ]
            }

    prompt = SystemMessage(
        content=f"""
{ERROR_ANALYZER_PROMPT.content}
{context if context else ""}
{memory_context if memory_context else ""}
{f"DOCUMENT CONTEXT: {rag_context}" if rag_context else ""}
"""
    )

    messages = [prompt] + state["messages"]
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    response = llm.invoke(messages)

    new_history = prior_errors + [{"error": query, "iteration": len(prior_errors) + 1}]

    if session_id:
        add_error_to_session(session_id, query, response.content)

    logger.info(
        "error_analysis_completed",
        thread_id=thread_id,
        session_id=session_id,
        is_follow_up=is_follow_up,
    )

    return {
        "messages": [response],
        "debug_history": new_history,
        "session_id": session_id,
    }
