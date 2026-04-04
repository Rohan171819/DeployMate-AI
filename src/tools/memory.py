"""Tools module for memory operations."""

from __future__ import annotations

import structlog
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings
from src.exceptions import MemoryReadError, MemoryWriteError

logger = structlog.get_logger()

_THREAD_MEMORY_CACHE: dict[str, dict[str, object]] = {}


def get_user_memory(store, user_id: str) -> dict:
    """Retrieve user information from long term memory.

    Args:
        store: PostgresStore instance for persistent memory.
        user_id: Unique identifier for the user.

    Returns:
        Dictionary containing user profile information (tech_stack, experience).
        Returns empty dict if no profile exists.

    Raises:
        MemoryReadError: On database connection failures.
    """
    namespace = ("user_profiles", user_id)
    try:
        result = store.get(namespace, "profile")
        if result is not None:
            logger.debug("memory_retrieved", user_id=user_id)
            return result.value
        logger.debug("no_memory_found", user_id=user_id)
        return {}
    except Exception as e:
        logger.error("memory_read_failed", user_id=user_id, error=str(e))
        raise MemoryReadError(
            f"Failed to read user memory: {e}", user_id=user_id
        ) from e


def save_user_memory(store, user_id: str, info: dict) -> None:
    """Save user information to long term memory.

    Args:
        store: PostgresStore instance for persistent memory.
        user_id: Unique identifier for the user.
        info: Dictionary containing profile information to save.

    Raises:
        MemoryWriteError: On database write failures.
    """
    if not info:
        logger.debug("empty_info_skipping_save", user_id=user_id)
        return

    namespace = ("user_profiles", user_id)
    try:
        existing = store.get(namespace, "profile")
        if existing:
            profile = existing.value
            if isinstance(profile, dict):
                profile.update(info)
            else:
                profile = info
        else:
            profile = info

        store.put(namespace, "profile", profile)
        logger.info("memory_saved", user_id=user_id, keys=list(info.keys()))
    except Exception as e:
        logger.error("memory_write_failed", user_id=user_id, error=str(e))
        raise MemoryWriteError(
            f"Failed to save user memory: {e}", user_id=user_id
        ) from e


def get_thread_id_from_config(config: RunnableConfig) -> str:
    """Extract thread_id from RunnableConfig.

    Args:
        config: LangGraph RunnableConfig with configurable metadata.

    Returns:
        Thread ID string, or empty string if not found.
    """
    return config.get("configurable", {}).get("thread_id", "")
