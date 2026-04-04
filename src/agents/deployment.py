"""Deployment guide agent node."""

from __future__ import annotations

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings
from src.tools.rag import get_rag_context_with_fallback

logger = structlog.get_logger()

DEPLOY_GUIDE_PROMPT = SystemMessage(
    content="""
You are the Deployment Guide Agent inside DeployMate AI.
When user wants to deploy, ALWAYS respond in this structure:

## Deployment Plan
Identify what they want to deploy and where.

## Prerequisites
What needs to be ready before deploying.

## Step-by-step Guide
Exact commands and steps — numbered clearly.

## Configuration
Dockerfile, env vars, ports — whatever is needed.

## Verification
How to confirm deployment was successful.

Be specific, beginner-friendly, with exact commands.
"""
)


def deployment_guide_node(state: dict, config: RunnableConfig) -> dict:
    """Provide deployment guidance for user's application.

    Handles RAG context retrieval from uploaded PDFs.

    Args:
        state: Current chat state with deployment request.
        config: RunnableConfig with thread_id and other metadata.

    Returns:
        Updated state with deployment guide response.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    logger.info(
        "deployment_guide_started", thread_id=thread_id, message_prefix=query[:50]
    )

    crag = get_rag_context_with_fallback(thread_id, query)
    rag_context = crag.get("context", "")

    if rag_context:
        prompt = SystemMessage(
            content=f"""
{DEPLOY_GUIDE_PROMPT.content}

DOCUMENT CONTEXT:
{rag_context}
"""
        )
    else:
        prompt = DEPLOY_GUIDE_PROMPT

    messages = [prompt] + state["messages"]
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    response = llm.invoke(messages)

    logger.info("deployment_guide_completed", thread_id=thread_id)
    return {"messages": [response]}
