"""Pytest fixtures for DeployMate AI tests."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def mock_llm():
    """Mock LLM for fast unit tests."""
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Mocked response")
    llm.with_structured_output.return_value.invoke.return_value = MagicMock(score=8)
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for RAG tests."""
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1] * 768]
    emb.embed_query.return_value = [0.1] * 768
    return emb


@pytest.fixture
def mock_store():
    """Mock PostgresStore for memory tests."""
    store = MagicMock()
    store.get.return_value = MagicMock(value={"tech_stack": "python"})
    return store


@pytest.fixture
def mock_checkpointer():
    """Mock PostgresSaver for checkpoint tests."""
    checkpointer = MagicMock()
    checkpointer.list.return_value = []
    return checkpointer


@pytest.fixture
def sample_chat_state():
    """Sample ChatState for tests."""
    return {"messages": [HumanMessage(content="Hello, how do I deploy to AWS?")]}


@pytest.fixture
def sample_error_state():
    """Sample state with Docker error message."""
    return {"messages": [HumanMessage(content="my docker container is failing")]}


@pytest.fixture
def sample_deploy_state():
    """Sample state with deployment question."""
    return {"messages": [HumanMessage(content="I want to deploy my app on railway")]}


@pytest.fixture
def sample_code_review_state():
    """Sample state with code review request."""
    return {
        "messages": [
            HumanMessage(content="review my code please:\ndef foo():\n    pass")
        ]
    }


@pytest.fixture
def mock_runnable_config():
    """Sample RunnableConfig for tests."""
    return {"configurable": {"thread_id": "test-thread-123", "run_id": "test-run-456"}}


@pytest.fixture
def mock_interrupt():
    """Mock LangGraph interrupt for dangerous command tests."""
    return {"approved": False}


@pytest.fixture
def mock_rag_context():
    """Mock RAG context response."""
    return {
        "context": "Retrieved context from PDF",
        "source": "pdf",
        "needs_fallback": False,
    }
