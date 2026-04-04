"""Code review agent node."""

from __future__ import annotations

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings
from src.tools.rag import get_rag_context_with_fallback

logger = structlog.get_logger()

CODE_REVIEW_PROMPT = SystemMessage(
    content="""
You are the Code Review Agent inside DeployMate AI.
When given code, ALWAYS respond in this structure:

## Code Summary
What this code does briefly.

## Security Issues
Any vulnerabilities found — with line references.

## Performance Issues
Any bottlenecks or inefficiencies found.

## Bad Practices
Anti-patterns or poor coding practices.

## Improved Code
Provide the fixed, improved version with explanations.

Be constructive, educational, and specific.
"""
)


def code_review_node(state: dict, config: RunnableConfig) -> dict:
    """Provide code review with security and performance feedback.

    Handles RAG context retrieval from uploaded PDFs.

    Args:
        state: Current chat state with code to review.
        config: RunnableConfig with thread_id and other metadata.

    Returns:
        Updated state with code review response.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    logger.info("code_review_started", thread_id=thread_id, message_prefix=query[:50])

    crag = get_rag_context_with_fallback(thread_id, query)
    rag_context = crag.get("context", "")

    if rag_context:
        prompt = SystemMessage(
            content=f"""
{CODE_REVIEW_PROMPT.content}

DOCUMENT CONTEXT:
{rag_context}
"""
        )
    else:
        prompt = CODE_REVIEW_PROMPT

    messages = [prompt] + state["messages"]
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    response = llm.invoke(messages)

    logger.info("code_review_completed", thread_id=thread_id)
    return {"messages": [response]}
