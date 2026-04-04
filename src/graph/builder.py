"""Graph builder - LangGraph StateGraph construction."""

from __future__ import annotations

import os
from typing import Annotated
from typing import TypedDict

import structlog
import psycopg
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from src.config.settings import settings

load_dotenv()

logger = structlog.get_logger()

# Configure LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing else "false"
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


# TypedDict for state
class ChatState(TypedDict):
    """Chat state for the main graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str | None
    debug_history: list[dict]
    intent: str | None


class ErrorAnalysisState(TypedDict):
    """Error analysis subgraph state."""

    messages: Annotated[list[BaseMessage], add_messages]
    error_type: str
    severity: str
    has_fix: bool


class DeploymentState(TypedDict):
    """Deployment subgraph state."""

    messages: Annotated[list[BaseMessage], add_messages]
    tech_stack: str
    target_platform: str
    has_dockerfile: bool


class CodeReviewState(TypedDict):
    """Code review subgraph state."""

    messages: Annotated[list[BaseMessage], add_messages]
    language: str
    has_security_issue: bool
    has_performance_issue: bool


# Global store for memory
store = None
checkpointer = None


def _init_db():
    """Initialize database connections."""
    global store, checkpointer

    if store is not None:
        return

    logger.info("initializing_database", db_uri_prefix=settings.database_url[:30])

    conn = psycopg.connect(settings.database_url, autocommit=True)
    checkpointer = PostgresSaver(conn=conn)
    checkpointer.setup()

    store_conn = psycopg.connect(settings.database_url, autocommit=True)
    store = PostgresStore(store_conn)
    store.setup()

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

    logger.info("database_initialized")


def get_store():
    """Get the PostgresStore instance."""
    _init_db()
    return store


def get_checkpointer():
    """Get the PostgresSaver checkpointer."""
    _init_db()
    return checkpointer


# Subgraph builders


def _error_parser_node(state: ErrorAnalysisState, config: RunnableConfig) -> dict:
    """Identify what type of error this is."""
    from src.agents.router import is_error_message

    query = state["messages"][-1].content

    if "docker" in query.lower():
        error_type = "Docker"
    elif "pipeline" in query.lower() or "ci/cd" in query.lower():
        error_type = "CI/CD"
    elif "traceback" in query.lower():
        error_type = "Python"
    else:
        error_type = "General"

    return {"error_type": error_type}


def _severity_checker_node(state: ErrorAnalysisState, config: RunnableConfig) -> dict:
    """Check how critical this error is."""
    query = state["messages"][-1].content

    critical_keywords = ["production", "down", "crash", "data loss"]
    severity = (
        "Critical"
        if any(kw in query.lower() for kw in critical_keywords)
        else "Warning"
    )

    return {"severity": severity}


def _solution_finder_node(state: ErrorAnalysisState, config: RunnableConfig) -> dict:
    """Find the solution based on error type and severity."""
    from langchain_core.messages import SystemMessage
    from langchain_ollama import ChatOllama

    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    from src.tools.rag import get_rag_context

    rag_context = get_rag_context(thread_id, query)

    prompt = SystemMessage(
        content=f"""
You are the Error Analyzer Agent inside DeployMate AI.
Error Type: {state["error_type"]}
Severity: {state["severity"]}

{"⚠️ CRITICAL ERROR — Provide immediate fix first!" if state["severity"] == "Critical" else ""}

ALWAYS respond in this structure:
## Root Cause
## Why It Happened
## Exact Fix
## Prevention

{f"DOCUMENT CONTEXT: {rag_context}" if rag_context else ""}
"""
    )

    messages = [prompt] + state["messages"]
    llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url)
    response = llm.invoke(messages)
    return {"messages": [response], "has_fix": True}


def _fix_validator_node(state: ErrorAnalysisState, config: RunnableConfig) -> dict:
    """Validate if the fix is complete and safe."""
    from langchain_core.messages import SystemMessage
    from langchain_ollama import ChatOllama
    from langgraph.types import interrupt
    from src.agents.router import is_dangerous

    if is_dangerous(state["messages"][-1].content):
        human_decision = interrupt(
            {
                "type": "dangerous_command",
                "message": "Dangerous command detected!",
                "suggested_response": state["messages"][-1].content,
            }
        )
        if not human_decision.get("approved"):
            llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url)
            safe_response = llm.invoke(
                [
                    SystemMessage(content="Provide a SAFER alternative."),
                    *state["messages"],
                ]
            )
            return {"messages": [safe_response]}

    return state


def build_error_analysis_subgraph():
    """Build and return the error analysis subgraph."""
    subgraph = StateGraph(ErrorAnalysisState)

    subgraph.add_node("error_parser", _error_parser_node)
    subgraph.add_node("severity_checker", _severity_checker_node)
    subgraph.add_node("solution_finder", _solution_finder_node)
    subgraph.add_node("fix_validator", _fix_validator_node)

    subgraph.add_edge(START, "error_parser")
    subgraph.add_edge("error_parser", "severity_checker")
    subgraph.add_edge("severity_checker", "solution_finder")
    subgraph.add_edge("solution_finder", "fix_validator")
    subgraph.add_edge("fix_validator", END)

    return subgraph.compile()


# Deployment subgraph


def _stack_detector_node(state: DeploymentState, config: RunnableConfig) -> dict:
    """Detect user's tech stack from message."""
    query = state["messages"][-1].content.lower()

    stacks = {
        "python": ["python", "flask", "django", "fastapi"],
        "nodejs": ["node", "express", "npm", "javascript"],
        "react": ["react", "nextjs", "vite"],
        "docker": ["docker", "dockerfile", "container"],
    }

    detected = "general"
    for stack, keywords in stacks.items():
        if any(kw in query for kw in keywords):
            detected = stack
            break

    return {"tech_stack": detected}


def _platform_selector_node(state: DeploymentState, config: RunnableConfig) -> dict:
    """Detect target deployment platform."""
    query = state["messages"][-1].content.lower()

    platforms = {
        "railway": ["railway"],
        "aws": ["aws", "ec2", "s3"],
        "render": ["render"],
        "vps": ["vps", "digitalocean", "linode"],
    }

    detected = "railway"
    for platform, keywords in platforms.items():
        if any(kw in query for kw in keywords):
            detected = platform
            break

    return {"target_platform": detected}


def _config_generator_node(state: DeploymentState, config: RunnableConfig) -> dict:
    """Generate deployment configuration."""
    from langchain_core.messages import SystemMessage
    from langchain_ollama import ChatOllama

    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    from src.tools.rag import get_rag_context

    rag_context = get_rag_context(thread_id, query)

    prompt = SystemMessage(
        content=f"""
You are the Deployment Guide Agent inside DeployMate AI.
Tech Stack Detected: {state["tech_stack"]}
Target Platform: {state["target_platform"]}

Provide deployment configuration specific to {state["tech_stack"]} on {state["target_platform"]}.

ALWAYS respond in this structure:

## Deployment Plan
What will be deployed and where.

## Prerequisites
What needs to be ready before deploying.

## Configuration
Dockerfile, env vars, ports for {state["tech_stack"]}.

## Step-by-step Guide
Exact commands for {state["target_platform"]}.

## Verification
How to confirm deployment was successful.

{f"DOCUMENT CONTEXT: {rag_context}" if rag_context else ""}
"""
    )

    messages = [prompt] + state["messages"]
    llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url)
    response = llm.invoke(messages)
    return {"messages": [response]}


def _steps_generator_node(state: DeploymentState, config: RunnableConfig) -> dict:
    """Generate final deployment checklist."""
    return state


def build_deployment_subgraph():
    """Build and return the deployment pipeline subgraph."""
    subgraph = StateGraph(DeploymentState)

    subgraph.add_node("stack_detector", _stack_detector_node)
    subgraph.add_node("platform_selector", _platform_selector_node)
    subgraph.add_node("config_generator", _config_generator_node)
    subgraph.add_node("steps_generator", _steps_generator_node)

    subgraph.add_edge(START, "stack_detector")
    subgraph.add_edge("stack_detector", "platform_selector")
    subgraph.add_edge("platform_selector", "config_generator")
    subgraph.add_edge("config_generator", "steps_generator")
    subgraph.add_edge("steps_generator", END)

    return subgraph.compile()


# Code review subgraph


def _language_detector_node(state: CodeReviewState, config: RunnableConfig) -> dict:
    """Detect programming language from code."""
    query = state["messages"][-1].content.lower()

    languages = {
        "python": ["def ", "import ", "print(", ".py", "django", "flask"],
        "javascript": ["function", "const ", "let ", "var ", "=>", ".js"],
        "java": ["public class", "void ", "system.out", ".java"],
        "sql": ["select ", "insert ", "update ", "delete ", "drop "],
        "bash": ["#!/bin/bash", "chmod", "sudo", "apt-get"],
        "dockerfile": ["from ", "run ", "cmd ", "expose", "workdir"],
    }

    detected = "general"
    for lang, keywords in languages.items():
        if any(kw in query for kw in keywords):
            detected = lang
            break

    return {"language": detected}


def _security_scanner_node(state: CodeReviewState, config: RunnableConfig) -> dict:
    """Scan code for security vulnerabilities."""
    query = state["messages"][-1].content.lower()

    security_red_flags = [
        "sql",
        "select *",
        "exec(",
        "eval(",
        "password",
        "secret",
        "api_key",
        "shell=true",
        "subprocess",
        "md5",
        "sha1",
    ]

    has_issue = any(flag in query for flag in security_red_flags)
    return {"has_security_issue": has_issue}


def _performance_scanner_node(state: CodeReviewState, config: RunnableConfig) -> dict:
    """Scan code for performance issues."""
    query = state["messages"][-1].content.lower()

    performance_red_flags = [
        "for i in range",
        "while true",
        "select *",
        "time.sleep",
        "global ",
        "nested for",
        ".append(",
        "string +",
    ]

    has_issue = any(flag in query for flag in performance_red_flags)
    return {"has_performance_issue": has_issue}


def _review_generator_node(state: CodeReviewState, config: RunnableConfig) -> dict:
    """Generate comprehensive code review."""
    from langchain_core.messages import SystemMessage
    from langchain_ollama import ChatOllama

    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    from src.tools.rag import get_rag_context

    rag_context = get_rag_context(thread_id, query)

    security_note = (
        "⚠️ SECURITY ISSUES DETECTED — Review carefully!"
        if state["has_security_issue"]
        else "No obvious security issues detected."
    )
    performance_note = (
        "⚠️ PERFORMANCE ISSUES DETECTED — Optimize!"
        if state["has_performance_issue"]
        else "No obvious performance issues detected."
    )

    prompt = SystemMessage(
        content=f"""
You are the Code Review Agent inside DeployMate AI.
Language Detected: {state["language"]}
Security Pre-scan: {security_note}
Performance Pre-scan: {performance_note}

ALWAYS respond in this structure:

## Code Summary
What this code does briefly.

## Security Issues
{"Focus heavily here — issues were pre-detected!" if state["has_security_issue"] else "Check for any vulnerabilities."}

## Performance Issues
{"Focus heavily here — issues were pre-detected!" if state["has_performance_issue"] else "Check for bottlenecks."}

## Bad Practices
Anti-patterns specific to {state["language"]}.

## Improved Code
Provide complete fixed version with explanations.

{f"DOCUMENT CONTEXT: {rag_context}" if rag_context else ""}
"""
    )

    messages = [prompt] + state["messages"]
    llm = ChatOllama(model=settings.llm_model, base_url=settings.llm_base_url)
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_code_review_subgraph():
    """Build and return the code review pipeline subgraph."""
    subgraph = StateGraph(CodeReviewState)

    subgraph.add_node("language_detector", _language_detector_node)
    subgraph.add_node("security_scanner", _security_scanner_node)
    subgraph.add_node("performance_scanner", _performance_scanner_node)
    subgraph.add_node("review_generator", _review_generator_node)

    subgraph.add_edge(START, "language_detector")
    subgraph.add_edge("language_detector", "security_scanner")
    subgraph.add_edge("security_scanner", "performance_scanner")
    subgraph.add_edge("performance_scanner", "review_generator")
    subgraph.add_edge("review_generator", END)

    return subgraph.compile()


# Main graph builder


def build_graph():
    """Build and compile the main LangGraph."""
    from src.agents.chat import chat_node
    from src.agents.router import route_message
    from src.agents.error_analyzer import error_analyzer_node
    from src.agents.deployment import deployment_guide_node
    from src.agents.code_review import code_review_node
    from src.agents.github_agent import github_connector_node

    logger.info("building_main_graph")

    error_analysis_subgraph = build_error_analysis_subgraph()
    deployment_subgraph = build_deployment_subgraph()
    code_review_subgraph = build_code_review_subgraph()

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("error_analyzer_node", error_analysis_subgraph)
    graph.add_node("deploy_guide_node", deployment_subgraph)
    graph.add_node("code_review_node", code_review_subgraph)
    graph.add_node("github_connector_node", github_connector_node)

    graph.add_conditional_edges(
        START,
        route_message,
        {
            "error_analyzer_node": "error_analyzer_node",
            "chat_node": "chat_node",
            "deploy_guide_node": "deploy_guide_node",
            "code_review_node": "code_review_node",
            "github_connector_node": "github_connector_node",
        },
    )

    graph.add_edge("error_analyzer_node", END)
    graph.add_edge("deploy_guide_node", END)
    graph.add_edge("code_review_node", END)
    graph.add_edge("chat_node", END)
    graph.add_edge("github_connector_node", END)

    _init_db()
    chatbot = graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=[],
    )

    logger.info("graph_compiled_successfully")
    return chatbot


chatbot = build_graph()


def retrieve_all_threads() -> list[str]:
    """Retrieve all thread IDs from PostgreSQL checkpointer."""
    all_threads: set[str] = set()

    _init_db()
    try:
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
            if thread_id:
                all_threads.add(thread_id)
    except Exception as e:
        logger.error("thread_retrieval_failed", error=str(e))

    return list(all_threads)
