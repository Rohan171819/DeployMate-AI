"""GitHub connector tool for fetching repository content."""

from __future__ import annotations

import re
from typing import Any

import structlog
from github import Github
from github.GithubException import GithubException

from src.exceptions import GitHubAuthError, GitHubFetchError, GitHubURLParseError

logger = structlog.get_logger()

GITHUB_URL_PATTERN = re.compile(
    r"(https?://)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/(?P<type>pull|blob|tree)/(?P<ref>[^/]+)/?(?P<path>.*))?"
)
MAX_CONTENT_LENGTH = 8000


def extract_github_url(text: str) -> str | None:
    """Extract first GitHub URL from free-form user message.

    Args:
        text: User message to extract URL from.

    Returns:
        First GitHub URL found, or None if no URL found.
    """
    import re

    url_match = re.search(
        r"https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(?:/pull/\d+|/blob/[^/]+/.+)?/?",
        text,
    )
    if url_match:
        url = url_match.group(0).rstrip("/")
        logger.info("github_url_extracted", url=url)
        return url
    logger.debug("no_github_url_found", text=text[:50])
    return None


class GitHubConnector:
    """GitHub connector for fetching repository content.

    Provides methods to fetch PR diffs, file contents, and repository trees
    from GitHub repositories using the PyGithub library.
    """

    def __init__(self, token: str) -> None:
        """Initialize GitHub connector.

        Args:
            token: GitHub personal access token.

        Raises:
            GitHubAuthError: If token is empty.
        """
        if not token:
            logger.error("github_token_missing")
            raise GitHubAuthError(
                "GitHub token is required. Set GITHUB_TOKEN in .env file."
            )
        from github import Auth

        self.client = Github(auth=Auth.Token(token))
        logger.info("github_client_initialized")

    def parse_github_url(self, url: str) -> dict[str, Any]:
        """Parse GitHub URL to extract owner, repo, type, and refs.

        Args:
            url: GitHub URL to parse (PR, file, or repo).

        Returns:
            Dictionary with owner, repo, type, pr_number, branch, and path.

        Raises:
            GitHubURLParseError: If URL pattern does not match.
        """
        logger.info("parsing_github_url", url=url)

        url = url.rstrip("/")

        pr_match = re.match(
            r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(\d+)/?$", url
        )
        if pr_match:
            result = {
                "owner": pr_match.group(1),
                "repo": pr_match.group(2),
                "type": "pull",
                "pr_number": int(pr_match.group(3)),
                "branch": None,
                "path": None,
            }
            logger.info("github_url_parsed", type="pull", pr_number=result["pr_number"])
            return result

        file_match = re.match(
            r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$",
            url,
        )
        if file_match:
            result = {
                "owner": file_match.group(1),
                "repo": file_match.group(2),
                "type": "file",
                "pr_number": None,
                "branch": file_match.group(3),
                "path": file_match.group(4),
            }
            logger.info(
                "github_url_parsed",
                type="file",
                branch=result["branch"],
                path=result["path"],
            )
            return result

        repo_match = re.match(
            r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/?$", url
        )
        if repo_match:
            result = {
                "owner": repo_match.group(1),
                "repo": repo_match.group(2),
                "type": "repo",
                "pr_number": None,
                "branch": None,
                "path": None,
            }
            logger.info("github_url_parsed", type="repo")
            return result

        logger.error("github_url_parse_failed", url=url)
        raise GitHubURLParseError(f"Could not parse GitHub URL: {url}")

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch PR diff from GitHub.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.

        Returns:
            Formatted diff string with file changes.

        Raises:
            GitHubFetchError: If API call fails.
        """
        logger.info("fetching_pr_diff", owner=owner, repo=repo, pr_number=pr_number)

        try:
            full_repo = self.client.get_repo(f"{owner}/{repo}")
            pr = full_repo.get_pull(pr_number)
            files = pr.get_files()

            diff_parts = []
            for file in files:
                file_info = (
                    f"File: {file.filename}\n"
                    f"Status: {file.status}\n"
                    f"Changes: +{file.additions} -{file.deletions}\n"
                )
                if file.patch:
                    patch = file.patch[:2000]
                    file_info += f"```\n{patch}\n```\n"
                diff_parts.append(file_info)

            result = "\n\n".join(diff_parts)

            total_length = sum(len(part) for part in diff_parts)
            if total_length > MAX_CONTENT_LENGTH:
                truncate_msg = "\n\n[Output truncated to 8000 chars]"
                result = result[: MAX_CONTENT_LENGTH - len(truncate_msg)] + truncate_msg
                logger.warning("pr_diff_truncated", original_length=total_length)

            logger.info("pr_diff_fetched", pr_number=pr_number, length=len(result))
            return result

        except GithubException as e:
            logger.error("github_api_error", error=str(e))
            raise GitHubFetchError(f"Failed to fetch PR: {str(e)}") from e

    def get_file_contents(self, owner: str, repo: str, branch: str, path: str) -> str:
        """Fetch file contents from GitHub.

        Args:
            owner: Repository owner.
            repo: Repository name.
            branch: Branch name.
            path: File path.

        Returns:
            Decoded file content.

        Raises:
            GitHubFetchError: If API call fails.
        """
        logger.info(
            "fetching_file_contents", owner=owner, repo=repo, branch=branch, path=path
        )

        try:
            full_repo = self.client.get_repo(f"{owner}/{repo}")
            content = full_repo.get_contents(path, ref=branch)
            decoded = content.decoded_content.decode("utf-8")

            if len(decoded) > MAX_CONTENT_LENGTH:
                decoded = (
                    decoded[:MAX_CONTENT_LENGTH]
                    + "\n\n[Output truncated to 8000 chars]"
                )
                logger.warning("file_content_truncated", original_length=len(decoded))

            logger.info("file_contents_fetched", path=path, length=len(decoded))
            return decoded

        except GithubException as e:
            logger.error("github_api_error", error=str(e))
            raise GitHubFetchError(f"Failed to fetch file: {str(e)}") from e

    def get_repo_tree(self, owner: str, repo: str) -> str:
        """Fetch repository file tree.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            Formatted file tree string.

        Raises:
            GitHubFetchError: If API call fails.
        """
        logger.info("fetching_repo_tree", owner=owner, repo=repo)

        try:
            full_repo = self.client.get_repo(f"{owner}/{repo}")
            contents = full_repo.get_contents("")
            tree_lines = [f"{repo}/"]

            for item in contents:
                if item.type == "dir":
                    tree_lines.append(f"  {item.name}/")
                    if item.name not in [
                        "node_modules",
                        "vendor",
                        "__pycache__",
                        ".git",
                    ]:
                        try:
                            sub_contents = full_repo.get_contents(item.name)
                            for sub_item in sub_contents[:5]:
                                prefix = "    "
                                if sub_item.type == "dir":
                                    tree_lines.append(f"{prefix}{sub_item.name}/")
                                else:
                                    tree_lines.append(f"{prefix}{sub_item.name}")
                        except GithubException:
                            pass
                else:
                    tree_lines.append(f"  {item.name}")

            result = "\n".join(tree_lines[:50])
            logger.info("repo_tree_fetched", repo=repo, lines=len(tree_lines))
            return result

        except GithubException as e:
            logger.error("github_api_error", error=str(e))
            raise GitHubFetchError(f"Failed to fetch repo tree: {str(e)}") from e
