"""GitHub agent node for connecting to GitHub repositories."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agents.code_review import code_review_node
from src.config.settings import settings
from src.exceptions import GitHubAuthError, GitHubFetchError
from src.tools.github_connector import GitHubConnector, extract_github_url

logger = structlog.get_logger()


def github_connector_node(state: dict, config: RunnableConfig) -> dict:
    """Fetch GitHub content and pass to code review node.

    Extracts GitHub URL from user message, fetches the content (PR diff,
    file, or repo tree), and chains into the code_review_node for analysis.

    Args:
        state: Chat state containing messages.
        config: RunnableConfig with thread metadata.

    Returns:
        Updated state with AI response from code review.
    """
    logger.info(
        "github_connector_node_started",
        thread_id=config.get("configurable", {}).get("thread_id"),
    )

    last_message = state["messages"][-1].content
    url = extract_github_url(last_message)

    if not url:
        logger.info("no_github_url_in_message")
        return {
            "messages": [
                AIMessage(
                    content="I couldn't find a GitHub URL in your message. "
                    "Please paste a GitHub URL in one of these formats:\n"
                    "- PR: https://github.com/owner/repo/pull/123\n"
                    "- File: https://github.com/owner/repo/blob/branch/path/to/file.py\n"
                    "- Repo: https://github.com/owner/repo"
                )
            ]
        }

    if not settings.github_token:
        logger.warning("github_token_not_configured")
        return {
            "messages": [
                AIMessage(
                    content="GitHub token not configured. Please add GITHUB_TOKEN to your .env file. "
                    "The token needs 'repo' and 'read:user' scopes for private repo access."
                )
            ]
        }

    try:
        connector = GitHubConnector(settings.github_token)
    except GitHubAuthError as e:
        logger.error("github_auth_error", error=str(e))
        return {"messages": [AIMessage(content=str(e))]}

    try:
        parsed = connector.parse_github_url(url)
    except GitHubFetchError as e:
        logger.error("github_url_parse_error", error=str(e))
        return {
            "messages": [AIMessage(content=f"Failed to parse GitHub URL: {str(e)}")]
        }

    try:
        if parsed["type"] == "pull":
            content = connector.get_pr_diff(
                parsed["owner"], parsed["repo"], parsed["pr_number"]
            )
        elif parsed["type"] == "file":
            content = connector.get_file_contents(
                parsed["owner"], parsed["repo"], parsed["branch"], parsed["path"]
            )
        else:
            content = connector.get_repo_tree(parsed["owner"], parsed["repo"])
    except GitHubFetchError as e:
        logger.error("github_fetch_error", error=str(e))
        return {
            "messages": [AIMessage(content=f"Failed to fetch GitHub content: {str(e)}")]
        }

    context_string = (
        f"=== GitHub Content ===\nURL: {url}\nType: {parsed['type']}\n\n{content}"
    )
    logger.info(
        "github_content_fetched",
        url=url,
        type=parsed["type"],
        content_length=len(context_string),
    )

    review_state = {
        "messages": state["messages"] + [SystemMessage(content=context_string)]
    }
    return code_review_node(review_state, config)
