"""Agent router and intent detection."""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

# Intent detection keywords
ERROR_KEYWORDS = [
    "error",
    "traceback",
    "exception",
    "failed",
    "exit code",
    "cannot",
    "unable to",
    "not found",
    "permission denied",
    "connection refused",
    "modulenotfounderror",
    "syntaxerror",
    "typeerror",
    "valueerror",
    "importerror",
    "runtimeerror",
    "docker",
    "container",
    "pipeline",
    "deployment failed",
    "rm -rf",
    "sudo rm",
    "drop database",
    "delete all",
    "format",
    "truncate",
    "chmod 777",
    "iptables",
    "disk full",
    "storage full",
    "clean up",
    "free space",
]

DEPLOY_KEYWORDS = [
    "deploy",
    "deployment",
    "publish",
    "host",
    "hosting",
    "aws",
    "railway",
    "render",
    "vps",
    "ec2",
    "go live",
    "production",
    "server",
    "cloud",
    "dockerfile",
    "docker compose",
    "nginx",
]

CODE_REVIEW_KEYWORDS = [
    "review my code",
    "check my code",
    "code review",
    "is this code good",
    "improve my code",
    "optimize",
    "security issue",
    "bad practice",
    "refactor",
    "```",
    "def ",
    "class ",
    "function",
    "import ",
]

DANGEROUS_KEYWORDS = [
    "rm -rf",
    "drop database",
    "delete all",
    "format",
    "truncate",
    "sudo rm",
    "chmod 777",
    "iptables -F",
    "dd if=",
    "> /dev/sda",
    "mkfs",
    "fdisk",
]


def is_error_message(message: str) -> bool:
    """Detect if message contains error-related keywords.

    Args:
        message: User message to analyze.

    Returns:
        True if error keywords found, False otherwise.
    """
    message_lower = message.lower()
    result = any(keyword in message_lower for keyword in ERROR_KEYWORDS)
    logger.debug("intent_detection", intent="error", result=result)
    return result


def is_deploy_message(message: str) -> bool:
    """Detect if message is about deployment.

    Args:
        message: User message to analyze.

    Returns:
        True if deployment keywords found, False otherwise.
    """
    result = any(keyword in message.lower() for keyword in DEPLOY_KEYWORDS)
    logger.debug("intent_detection", intent="deploy", result=result)
    return result


def is_code_review_message(message: str) -> bool:
    """Detect if message is a code review request.

    Args:
        message: User message to analyze.

    Returns:
        True if code review keywords found, False otherwise.
    """
    result = any(keyword in message.lower() for keyword in CODE_REVIEW_KEYWORDS)
    logger.debug("intent_detection", intent="code_review", result=result)
    return result


def is_dangerous(command: str) -> bool:
    """Detect if command contains dangerous keywords.

    Args:
        command: Command string to check.

    Returns:
        True if dangerous keywords found, False otherwise.
    """
    result = any(kw in command.lower() for kw in DANGEROUS_KEYWORDS)
    if result:
        logger.warning("dangerous_command_detected", command_prefix=command[:50])
    return result


def extract_user_info(message: str) -> dict:
    """Extract user preferences from message.

    Detects tech stack (python, nodejs, react) and experience level
    (junior, senior) from the message content.

    Args:
        message: User message to analyze.

    Returns:
        Dictionary with tech_stack and/or experience keys.
    """
    info: dict = {}

    stacks = {
        "python": ["python", "flask", "django", "fastapi"],
        "nodejs": ["node", "express", "javascript"],
        "react": ["react", "nextjs"],
    }
    for stack, keywords in stacks.items():
        if any(kw in message.lower() for kw in keywords):
            info["tech_stack"] = stack
            break

    if any(
        kw in message.lower() for kw in ["junior", "beginner", "learning", "new to"]
    ):
        info["experience"] = "junior"
    elif any(kw in message.lower() for kw in ["senior", "experienced", "expert"]):
        info["experience"] = "senior"

    if info:
        logger.info("user_info_extracted", info=info)

    return info


def route_message(state: dict, config: dict) -> str:
    """Route message to appropriate agent based on intent detection.

    Args:
        state: Chat state with messages.
        config: RunnableConfig with thread metadata.

    Returns:
        Node name to route to: error_analyzer_node, deploy_guide_node,
        code_review_node, or chat_node.
    """
    last_message = state["messages"][-1].content
    logger.info("routing_message", message_prefix=last_message[:50])

    if is_error_message(last_message):
        logger.info("route_selected", node="error_analyzer_node")
        return "error_analyzer_node"
    elif is_deploy_message(last_message):
        logger.info("route_selected", node="deploy_guide_node")
        return "deploy_guide_node"
    elif is_code_review_message(last_message):
        logger.info("route_selected", node="code_review_node")
        return "code_review_node"

    logger.info("route_selected", node="chat_node")
    return "chat_node"
