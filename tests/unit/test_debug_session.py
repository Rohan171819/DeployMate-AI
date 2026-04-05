"""Unit tests for debug session feature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.debug_session import (
    DebugSessionManager,
    detect_follow_up,
    init_debug_session,
)


class TestDebugSessionManager:
    """Tests for DebugSessionManager class."""

    @patch("src.tools.debug_session.psycopg")
    def test_init_debug_session_creates_new_session(self, mock_psycopg):
        """Test that new session is created when none exists."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_psycopg.connect.return_value = mock_conn

        manager = DebugSessionManager()
        manager._conn = mock_conn

        state = {"messages": []}
        result = manager.init_debug_session(state)

        assert "session_id" in result
        assert result["session_id"] is not None

    @patch("src.tools.debug_session.psycopg")
    def test_init_debug_session_uses_existing_session(self, mock_psycopg):
        """Test that existing session_id is preserved."""
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value = mock_conn

        manager = DebugSessionManager()
        manager._conn = mock_conn

        existing_session_id = "existing-session-123"
        state = {"messages": [], "session_id": existing_session_id}
        result = manager.init_debug_session(state)

        assert result["session_id"] == existing_session_id


class TestDetectFollowUp:
    """Tests for follow-up detection."""

    def test_detect_follow_up_no_prior_errors(self):
        """Test that no follow-up is detected when there are no prior errors."""
        current_error = "ModuleNotFoundError: No module named 'requests'"
        prior_errors = []

        result = detect_follow_up(current_error, prior_errors)

        assert result is False

    def test_detect_follow_up_with_related_errors(self):
        """Test that follow-up is detected for related errors."""
        current_error = "ModuleNotFoundError: No module named 'requests'"
        prior_errors = [
            {"error": "ModuleNotFoundError: No module named 'flask'", "iteration": 1},
            {"error": "ImportError: cannot import name 'app'", "iteration": 2},
        ]

        result = detect_follow_up(current_error, prior_errors)

        assert result is True

    def test_detect_follow_up_with_unrelated_errors(self):
        """Test that follow-up is not detected for unrelated errors."""
        current_error = "ValueError: invalid literal for int()"
        prior_errors = [
            {"error": "docker permission denied", "iteration": 1},
            {"error": "connection refused to localhost", "iteration": 2},
        ]

        result = detect_follow_up(current_error, prior_errors)

        assert result is False


class TestInitDebugSessionFunction:
    """Tests for init_debug_session helper function."""

    @patch("src.tools.debug_session._debug_session_manager")
    def test_init_debug_session_wrapper(self, mock_manager):
        """Test wrapper function calls manager."""
        mock_manager.init_debug_session.return_value = {"session_id": "test-123"}

        state = {}
        result = init_debug_session(state)

        assert "session_id" in result
        mock_manager.init_debug_session.assert_called_once()


class TestDebugSessionIntegration:
    """Integration-style tests with mocked database."""

    @patch("src.tools.debug_session.psycopg")
    def test_add_error_to_session(self, mock_psycopg):
        """Test adding error to session."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_psycopg.connect.return_value = mock_conn

        manager = DebugSessionManager()
        manager._conn = mock_conn

        manager.add_error_to_session("session-123", "ModuleNotFoundError: requests")

        mock_cursor.execute.assert_called()

    @patch("src.tools.debug_session.psycopg")
    def test_get_session_errors(self, mock_psycopg):
        """Test retrieving session errors."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        from datetime import datetime

        mock_cursor.fetchall.return_value = [
            ("error1", "resolution1", datetime.now()),
            ("error2", "resolution2", datetime.now()),
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_psycopg.connect.return_value = mock_conn

        manager = DebugSessionManager()
        manager._conn = mock_conn

        errors = manager.get_session_errors("session-123", limit=5)

        assert len(errors) == 2
        assert "error" in errors[0]
        assert "resolution" in errors[0]
