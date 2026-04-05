"""Unit tests for agent nodes and utilities."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.router import (
    is_error_message,
    is_deploy_message,
    is_code_review_message,
    is_dangerous,
    extract_user_info,
    route_message,
)
from src.tools.memory import get_user_memory, save_user_memory


class TestIntentDetection:
    """Tests for intent detection functions."""

    def test_router_detects_error_intent(self):
        """Test that error messages are correctly detected."""
        assert is_error_message("my docker container is failing") is True
        assert is_error_message("Traceback most recent call last") is True
        assert is_error_message("ModuleNotFoundError: No module named") is True
        assert is_error_message("how are you doing today") is False

    def test_router_detects_deployment_intent(self):
        """Test that deployment messages are correctly detected."""
        assert is_deploy_message("I want to deploy my app") is True
        assert is_deploy_message("how to host on aws ec2") is True
        assert is_deploy_message("deploy on railway platform") is True
        assert is_deploy_message("what is python") is False

    def test_router_detects_code_review_intent(self):
        """Test that code review messages are correctly detected."""
        assert is_code_review_message("review my code please") is True
        assert is_code_review_message("```python print('hello')```") is True
        assert is_code_review_message("def my_function():") is True
        assert is_code_review_message("what is docker") is False

    def test_dangerous_command_detection(self):
        """Test that dangerous commands are correctly detected."""
        assert is_dangerous("run rm -rf /var/log") is True
        assert is_dangerous("drop database mydb") is True
        assert is_dangerous("sudo rm se files delete karo") is True
        assert is_dangerous("restart the nginx server") is False

    def test_extract_user_info_python(self):
        """Test tech stack extraction for Python."""
        result = extract_user_info("I'm using Flask with Python")
        assert result.get("tech_stack") == "python"

    def test_extract_user_info_nodejs(self):
        """Test tech stack extraction for Node.js."""
        result = extract_user_info("I built an Express app")
        assert result.get("tech_stack") == "nodejs"

    def test_extract_user_info_react(self):
        """Test tech stack extraction for React."""
        result = extract_user_info("My NextJS app needs deployment")
        assert result.get("tech_stack") == "react"

    def test_extract_user_info_junior(self):
        """Test experience level extraction for junior."""
        result = extract_user_info("I'm a beginner learning Docker")
        assert result.get("experience") == "junior"

    def test_extract_user_info_senior(self):
        """Test experience level extraction for senior."""
        result = extract_user_info("I'm an expert in AWS")
        assert result.get("experience") == "senior"


class TestRouter:
    """Tests for the router function."""

    @patch("src.tools.debug_session._debug_session_manager")
    def test_route_to_error_analyzer(self, mock_manager):
        """Test routing to error analyzer for error messages."""
        mock_manager.init_debug_session.return_value = {"session_id": "test-123"}
        state = {"messages": [HumanMessage(content="Docker container failing")]}
        result = route_message(state, {})
        assert result == "error_analyzer_node"

    def test_route_to_deployment_guide(self):
        """Test routing to deployment guide for deploy messages."""
        state = {"messages": [HumanMessage(content="deploy to railway")]}
        result = route_message(state, {})
        assert result == "deploy_guide_node"

    def test_route_to_code_review(self):
        """Test routing to code review for review requests."""
        state = {"messages": [HumanMessage(content="review my code please")]}
        result = route_message(state, {})
        assert result == "code_review_node"

    def test_route_to_chat_node_default(self):
        """Test routing to chat node for general questions."""
        state = {"messages": [HumanMessage(content="what is Python?")]}
        result = route_message(state, {})
        assert result == "chat_node"


class TestMemory:
    """Tests for memory utilities."""

    def test_get_user_memory_returns_empty_on_miss(self, mock_store):
        """Test that missing memory returns empty dict."""
        mock_store.get.return_value = None
        result = get_user_memory(mock_store, "nonexistent-user")
        assert result == {}

    def test_get_user_memory_returns_profile(self, mock_store):
        """Test that existing profile is returned."""
        mock_store.get.return_value = MagicMock(value={"tech_stack": "python"})
        result = get_user_memory(mock_store, "test-user")
        assert result == {"tech_stack": "python"}

    def test_get_user_memory_returns_empty_on_db_error(self, mock_store):
        """Test that DB errors return empty dict."""
        mock_store.get.side_effect = Exception("Connection failed")
        result = get_user_memory(mock_store, "test-user")
        assert result == {}

    def test_save_user_memory_empty_info(self, mock_store):
        """Test that empty info skips save."""
        save_user_memory(mock_store, "test-user", {})
        mock_store.put.assert_not_called()

    def test_save_user_memory_new_profile(self, mock_store):
        """Test saving new user profile."""
        mock_store.get.return_value = None
        save_user_memory(mock_store, "test-user", {"tech_stack": "python"})
        mock_store.put.assert_called_once()

    def test_save_user_memory_no_op_on_db_error(self, mock_store):
        """Test that DB errors are silently handled."""
        mock_store.get.return_value = MagicMock(value={"tech_stack": "python"})
        mock_store.put.side_effect = Exception("Connection failed")
        save_user_memory(mock_store, "test-user", {"tech_stack": "python"})
        mock_store.put.assert_called_once()


class TestRAG:
    """Tests for RAG utilities."""

    def test_ingest_pdf_raises_on_empty_bytes(self):
        """Test that empty file bytes raises PDFIngestionError."""
        from src.tools.rag import ingest_pdf
        from src.exceptions import PDFIngestionError

        with pytest.raises(PDFIngestionError):
            ingest_pdf(b"", "thread-123")

    @pytest.mark.skip(reason="Requires real PDF file - tested manually")
    def test_ingest_pdf_success(self):
        """Test successful PDF ingestion - skipped as requires valid PDF file."""
        pass

    def test_retrieve_relevant_docs_no_retriever(self):
        """Test retrieval returns empty when no retriever."""
        from src.tools.rag import retrieve_relevant_docs

        result = retrieve_relevant_docs("nonexistent-thread", "query")
        assert result == []


class TestDangerousCommand:
    """Tests for dangerous command handling."""

    def test_dangerous_command_triggers_interrupt(self):
        """Test that dangerous commands return True from is_dangerous."""
        dangerous_queries = [
            "run rm -rf /var/logs",
            "drop database production",
            "sudo rm -rf /",
            "format disk",
        ]
        for query in dangerous_queries:
            assert is_dangerous(query) is True

    def test_safe_command_no_interrupt(self):
        """Test that safe commands return False."""
        safe_queries = [
            "restart nginx",
            "check docker status",
            "list files in directory",
        ]
        for query in safe_queries:
            assert is_dangerous(query) is False
