"""Debug session management for multi-turn debugging."""

from __future__ import annotations

import uuid
from typing import TypedDict

import structlog
import psycopg
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings

logger = structlog.get_logger()


class DebugSession(TypedDict):
    """Debug session data structure."""

    session_id: str
    user_id: str
    errors: list[dict]
    iteration: int


class DebugSessionManager:
    """Manages debug sessions for multi-turn debugging with memory continuity."""

    def __init__(self) -> None:
        """Initialize debug session manager."""
        self._conn = None

    def _get_conn(self):
        """Get database connection."""
        if self._conn is None:
            self._conn = psycopg.connect(settings.database_url, autocommit=True)
        return self._conn

    def init_debug_session(self, state: dict) -> dict:
        """Initialize or retrieve debug session from state.

        Args:
            state: Chat state containing session_id if exists.

        Returns:
            Updated state with session_id.
        """
        session_id = state.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info("debug_session_created", session_id=session_id)

            conn = self._get_conn()
            thread_id = ""
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO debug_sessions (session_id, user_id, iteration_count) "
                    "VALUES (%s, %s, 0) ON CONFLICT (session_id) DO NOTHING",
                    (session_id, thread_id),
                )

        return {"session_id": session_id}

    def add_error_to_session(
        self, session_id: str, error: str, resolution: str | None = None
    ) -> None:
        """Add error to debug session.

        Args:
            session_id: Debug session ID.
            error: Error text to add.
            resolution: Resolution text if available.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO debug_errors (session_id, error_text, resolution) "
                "VALUES (%s, %s, %s)",
                (session_id, error, resolution),
            )
            cur.execute(
                "UPDATE debug_sessions SET iteration_count = iteration_count + 1 "
                "WHERE session_id = %s",
                (session_id,),
            )
        logger.info("error_added_to_session", session_id=session_id, error=error[:50])

    def get_session_errors(self, session_id: str, limit: int = 5) -> list[dict]:
        """Get errors from debug session.

        Args:
            session_id: Debug session ID.
            limit: Maximum number of errors to retrieve.

        Returns:
            List of error dictionaries.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT error_text, resolution, timestamp 
                FROM debug_errors 
                WHERE session_id = %s 
                ORDER BY timestamp DESC 
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()
            return [
                {
                    "error": row[0],
                    "resolution": row[1],
                    "timestamp": row[2].isoformat() if row[2] else None,
                }
                for row in rows
            ]

    def get_iteration_count(self, session_id: str) -> int:
        """Get iteration count for debug session.

        Args:
            session_id: Debug session ID.

        Returns:
            Iteration count.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT iteration_count FROM debug_sessions WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def detect_follow_up(current_error: str, prior_errors: list[dict]) -> bool:
    """Detect if current error is related to prior errors.

    Uses simple keyword matching to determine if the current error
    is related to any prior errors in the session.

    Args:
        current_error: Current error message.
        prior_errors: List of prior error dictionaries.

    Returns:
        True if current error appears related to prior errors.
    """
    if not prior_errors:
        return False

    current_lower = current_error.lower()
    error_keywords = set()

    for prior in prior_errors:
        error_text = prior.get("error", "").lower()
        words = error_text.split()
        error_keywords.update(w for w in words if len(w) > 4)

    matches = sum(1 for kw in error_keywords if kw in current_lower)
    is_follow_up = matches >= 2

    logger.info(
        "follow_up_detection",
        is_follow_up=is_follow_up,
        matching_keywords=matches,
        prior_errors_count=len(prior_errors),
    )
    return is_follow_up


_debug_session_manager = DebugSessionManager()


def init_debug_session(state: dict) -> dict:
    """Initialize debug session from state.

    Args:
        state: Chat state.

    Returns:
        Updated state with session_id.
    """
    return _debug_session_manager.init_debug_session(state)


def add_error_to_session(
    session_id: str, error: str, resolution: str | None = None
) -> None:
    """Add error to debug session.

    Args:
        session_id: Debug session ID.
        error: Error text.
        resolution: Resolution text if available.
    """
    _debug_session_manager.add_error_to_session(session_id, error, resolution)


def get_session_errors(session_id: str, limit: int = 5) -> list[dict]:
    """Get errors from debug session.

    Args:
        session_id: Debug session ID.
        limit: Maximum errors to retrieve.

    Returns:
        List of error dictionaries.
    """
    return _debug_session_manager.get_session_errors(session_id, limit)
