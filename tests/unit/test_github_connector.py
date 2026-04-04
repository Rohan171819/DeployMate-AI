"""Unit tests for GitHub connector feature."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.github_agent import github_connector_node
from src.exceptions import GitHubAuthError, GitHubFetchError, GitHubURLParseError
from src.tools.github_connector import GitHubConnector, extract_github_url


class TestParsePRURL:
    """Tests for parsing PR URLs."""

    def test_parse_pr_url_valid(self):
        """Test parsing a valid PR URL."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url(
            "https://github.com/langchain-ai/langgraph/pull/42"
        )

        assert result["type"] == "pull"
        assert result["pr_number"] == 42
        assert result["owner"] == "langchain-ai"
        assert result["repo"] == "langgraph"

    def test_parse_pr_url_with_slash(self):
        """Test parsing PR URL with trailing slash."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url("https://github.com/owner/repo/pull/123/")

        assert result["type"] == "pull"
        assert result["pr_number"] == 123


class TestParseFileURL:
    """Tests for parsing file URLs."""

    def test_parse_file_url_valid(self):
        """Test parsing a valid file URL."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url(
            "https://github.com/user/repo/blob/main/src/agents/chat.py"
        )

        assert result["type"] == "file"
        assert result["branch"] == "main"
        assert result["path"] == "src/agents/chat.py"
        assert result["owner"] == "user"
        assert result["repo"] == "repo"

    def test_parse_file_url_deep_path(self):
        """Test parsing file URL with deep path."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url(
            "https://github.com/owner/repo/blob/feature/very/deep/path/file.ts"
        )

        assert result["type"] == "file"
        assert result["branch"] == "feature"
        assert result["path"] == "very/deep/path/file.ts"


class TestParseRepoURL:
    """Tests for parsing repository URLs."""

    def test_parse_repo_url_valid(self):
        """Test parsing a valid repo URL."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url("https://github.com/user/repo")

        assert result["type"] == "repo"
        assert result["owner"] == "user"
        assert result["repo"] == "repo"

    def test_parse_repo_url_with_trailing_slash(self):
        """Test parsing repo URL with trailing slash."""
        connector = GitHubConnector("fake_token")
        result = connector.parse_github_url("https://github.com/owner/repo/")

        assert result["type"] == "repo"


class TestParseInvalidURL:
    """Tests for invalid URL parsing."""

    def test_parse_invalid_url_raises(self):
        """Test that invalid URL raises GitHubURLParseError."""
        connector = GitHubConnector("fake_token")

        with pytest.raises(GitHubURLParseError):
            connector.parse_github_url("https://notgithub.com/something")

    def test_parse_non_github_url_raises(self):
        """Test that non-GitHub URL raises error."""
        connector = GitHubConnector("fake_token")

        with pytest.raises(GitHubURLParseError):
            connector.parse_github_url("https://gitlab.com/owner/repo")


class TestMissingToken:
    """Tests for missing token handling."""

    def test_missing_token_raises_auth_error(self):
        """Test that empty token raises GitHubAuthError."""
        with pytest.raises(GitHubAuthError):
            GitHubConnector("")


class TestGetPRDiff:
    """Tests for getting PR diff."""

    @patch("src.tools.github_connector.Github")
    def test_get_pr_diff_caps_at_8000_chars(self, mock_github):
        """Test that PR diff is capped at 8000 characters."""
        mock_repo = MagicMock()
        mock_pr = MagicMock()

        mock_files = []
        for i in range(10):
            mock_file = MagicMock()
            mock_file.filename = f"test{i}.py"
            mock_file.status = "modified"
            mock_file.additions = 500
            mock_file.deletions = 300
            mock_file.patch = "x" * 1000
            mock_files.append(mock_file)

        mock_pr.get_files.return_value = iter(mock_files)
        mock_repo.get_pull.return_value = mock_pr
        mock_github.return_value.get_repo.return_value = mock_repo

        connector = GitHubConnector("fake_token")
        result = connector.get_pr_diff("owner", "repo", 1)

        assert len(result) <= 8050
        assert "[Output truncated to 8000 chars]" in result


class TestExtractGitHubURL:
    """Tests for extracting GitHub URLs from messages."""

    def test_extract_github_url_from_message(self):
        """Test extracting URL from user message."""
        message = "hey can you review https://github.com/user/repo/pull/5 for me"
        result = extract_github_url(message)

        assert result == "https://github.com/user/repo/pull/5"

    def test_extract_github_url_from_plain_message(self):
        """Test that plain message returns None."""
        message = "how do I deploy to AWS?"
        result = extract_github_url(message)

        assert result is None

    def test_extract_github_url_file_url(self):
        """Test extracting file URL from message."""
        message = "check this file https://github.com/owner/repo/blob/main/src/main.py"
        result = extract_github_url(message)

        assert result == "https://github.com/owner/repo/blob/main/src/main.py"

    def test_extract_github_url_multiple_urls(self):
        """Test extracting first URL when multiple present."""
        message = "see https://github.com/owner/repo/pull/1 and https://github.com/other/repo/pull/2"
        result = extract_github_url(message)

        assert result == "https://github.com/owner/repo/pull/1"


class TestGitHubConnectorNode:
    """Tests for the github_connector_node function."""

    @patch("src.agents.github_agent.settings")
    @patch("src.agents.github_agent.GitHubConnector")
    def test_github_connector_node_no_url(self, mock_connector_class, mock_settings):
        """Test node returns helpful message when no URL found."""
        mock_settings.github_token = "fake_token"
        state = {"messages": [MagicMock(content="hello world")]}
        config = {"configurable": {"thread_id": "test"}}

        result = github_connector_node(state, config)

        assert len(result["messages"]) == 1
        assert "paste a github url" in result["messages"][0].content.lower()

    @patch("src.agents.github_agent.settings")
    def test_github_connector_node_no_token(self, mock_settings):
        """Test node returns error when no token configured."""
        mock_settings.github_token = ""
        state = {"messages": [MagicMock(content="https://github.com/owner/repo")]}
        config = {"configurable": {"thread_id": "test"}}

        result = github_connector_node(state, config)

        assert len(result["messages"]) == 1
        assert "GITHUB_TOKEN" in result["messages"][0].content


class TestGitHubConnectorIntegration:
    """Integration-style tests with mocked API calls."""

    @patch("src.tools.github_connector.Github")
    def test_get_file_contents(self, mock_github):
        """Test fetching file contents."""
        mock_repo = MagicMock()
        mock_content = MagicMock()
        mock_content.decoded_content = b"print('hello world')"
        mock_repo.get_contents.return_value = mock_content
        mock_github.return_value.get_repo.return_value = mock_repo

        connector = GitHubConnector("fake_token")
        result = connector.get_file_contents("owner", "repo", "main", "hello.py")

        assert result == "print('hello world')"

    @patch("src.tools.github_connector.Github")
    def test_get_repo_tree(self, mock_github):
        """Test fetching repository tree."""
        mock_repo = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "README.md"
        mock_file.type = "file"
        mock_dir = MagicMock()
        mock_dir.name = "src"
        mock_dir.type = "dir"
        mock_repo.get_contents.side_effect = [mock_file, []]
        mock_github.return_value.get_repo.return_value = mock_repo

        connector = GitHubConnector("fake_token")
        result = connector.get_repo_tree("owner", "repo")

        assert "owner/" in result or "repo/" in result
