"""Custom exception hierarchy for DeployMate AI.

This module defines a structured exception hierarchy for different failure
modes in the application, enabling more precise error handling and debugging.

Exception Hierarchy:
    - MemoryError
        - MemoryReadError
        - MemoryWriteError
    - RAGError
        - PDFIngestionError
        - RetrievalError
    - AgentError
        - RouterError
        - AnalysisError
    - DangerousCommandError
"""

from __future__ import annotations


class DeployMateError(Exception):
    """Base exception for all DeployMate AI errors."""

    pass


class MemoryError(DeployMateError):
    """Base exception for memory-related operations."""

    pass


class MemoryReadError(MemoryError):
    """Raised when reading from memory store fails."""

    def __init__(self, message: str, user_id: str | None = None) -> None:
        self.user_id = user_id
        super().__init__(message)


class MemoryWriteError(MemoryError):
    """Raised when writing to memory store fails."""

    def __init__(self, message: str, user_id: str | None = None) -> None:
        self.user_id = user_id
        super().__init__(message)


class RAGError(DeployMateError):
    """Base exception for RAG-related operations."""

    pass


class PDFIngestionError(RAGError):
    """Raised when PDF ingestion fails."""

    def __init__(self, message: str, filename: str | None = None) -> None:
        self.filename = filename
        super().__init__(message)


class RetrievalError(RAGError):
    """Raised when document retrieval fails."""

    def __init__(self, message: str, thread_id: str | None = None) -> None:
        self.thread_id = thread_id
        super().__init__(message)


class AgentError(DeployMateError):
    """Base exception for agent-related errors."""

    pass


class RouterError(AgentError):
    """Raised when message routing fails."""

    pass


class AnalysisError(AgentError):
    """Raised when agent analysis fails."""

    def __init__(self, message: str, agent_type: str | None = None) -> None:
        self.agent_type = agent_type
        super().__init__(message)


class DangerousCommandError(DeployMateError):
    """Raised when a dangerous command is detected requiring human approval.

    This exception is used by the Human-in-the-Loop (HITL) interrupt mechanism
    to pause execution and request user approval before proceeding.
    """

    def __init__(
        self,
        message: str,
        command: str | None = None,
        suggested_response: str | None = None,
    ) -> None:
        self.command = command
        self.suggested_response = suggested_response
        super().__init__(message)
