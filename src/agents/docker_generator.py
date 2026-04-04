"""Docker generator agent node."""

from __future__ import annotations

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings

logger = structlog.get_logger()

DOCKER_EXTRACTION_PROMPT = SystemMessage(
    content="""You are a Docker Configuration Extractor. Analyze the user's request and extract the following details for generating Docker configuration:

1. Framework/Technology (e.g., Python/Flask, Node/Express, React, Go, etc.)
2. Ports (e.g., 3000, 5000, 8000)
3. Dependencies (required packages, libraries, environment variables)
4. Entrypoint (how the application starts, main file)

Respond in this format exactly:
Framework: <extracted framework or "unknown">
Ports: <extracted ports or "8080">
Dependencies: <extracted dependencies or "none">
Entrypoint: <extracted entrypoint or "python app.py or npm start">"""
)

DOCKER_GENERATION_PROMPT = SystemMessage(
    content="""You are the Docker Generator Agent inside DeployMate AI.
Generate production-ready Dockerfile and docker-compose.yml based on the following project details.

PROJECT DETAILS:
- Framework: {framework}
- Ports: {ports}
- Dependencies: {dependencies}
- Entrypoint: {entrypoint}

FEW-SHOT EXAMPLES:

Example 1 (Python Flask):
Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]
```

docker-compose.yml:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: always
```

Example 2 (Node.js/Express):
Dockerfile:
```dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
```

docker-compose.yml:
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    restart: always
```

Now generate the Dockerfile and docker-compose.yml for the user's project. Start with "## Generated Docker Configuration" and include both files clearly labeled."""
)


def _extract_project_details(query: str, llm) -> dict:
    """Extract project details from user query using LLM.

    Args:
        query: User message with project description.
        llm: ChatOllama instance.

    Returns:
        Dictionary with framework, ports, dependencies, entrypoint.
    """
    messages = [DOCKER_EXTRACTION_PROMPT, SystemMessage(content=query)]
    response = llm.invoke(messages)

    result = {
        "framework": "Python",
        "ports": "8000",
        "dependencies": "none",
        "entrypoint": "python app.py",
    }

    content = response.content
    for line in content.split("\n"):
        if line.startswith("Framework:"):
            result["framework"] = line.split(":", 1)[1].strip()
        elif line.startswith("Ports:"):
            result["ports"] = line.split(":", 1)[1].strip()
        elif line.startswith("Dependencies:"):
            result["dependencies"] = line.split(":", 1)[1].strip()
        elif line.startswith("Entrypoint:"):
            result["entrypoint"] = line.split(":", 1)[1].strip()

    logger.info("project_details_extracted", details=result)
    return result


def _generate_docker_configs(project_details: dict, llm) -> dict:
    """Generate Dockerfile and docker-compose.yml.

    Args:
        project_details: Extracted project details.
        llm: ChatOllama instance.

    Returns:
        Dictionary with dockerfile and docker_compose content.
    """
    prompt_content = DOCKER_GENERATION_PROMPT.content.format(
        framework=project_details.get("framework", "Python"),
        ports=project_details.get("ports", "8000"),
        dependencies=project_details.get("dependencies", "none"),
        entrypoint=project_details.get("entrypoint", "python app.py"),
    )

    messages = [SystemMessage(content=prompt_content)]
    response = llm.invoke(messages)

    content = response.content

    dockerfile = ""
    docker_compose = ""
    in_dockerfile = False
    in_compose = False

    lines = content.split("\n")
    for line in lines:
        if "dockerfile" in line.lower() and "```" in line:
            in_dockerfile = True
            in_compose = False
            continue
        elif "docker-compose" in line.lower() and "```" in line:
            in_compose = True
            in_dockerfile = False
            continue
        elif "```" in line and (in_dockerfile or in_compose):
            in_dockerfile = False
            in_compose = False
            continue

        if in_dockerfile:
            dockerfile += line + "\n"
        elif in_compose:
            docker_compose += line + "\n"

    return {
        "dockerfile": dockerfile.strip(),
        "docker_compose": docker_compose.strip(),
    }


def _save_docker_config(
    thread_id: str, framework: str, dockerfile: str, docker_compose: str
) -> None:
    """Save generated Docker config to database.

    Args:
        thread_id: Thread ID for tracking.
        framework: Detected framework.
        dockerfile: Generated Dockerfile content.
        docker_compose: Generated docker-compose content.
    """
    from src.graph.builder import get_checkpointer

    checkpointer = get_checkpointer()
    conn = checkpointer.conn

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO docker_configs (thread_id, framework, dockerfile, docker_compose)
                VALUES (%s, %s, %s, %s)
            """,
                (thread_id, framework, dockerfile, docker_compose),
            )
        logger.info("docker_config_saved", thread_id=thread_id, framework=framework)
    except Exception as e:
        logger.error("docker_config_save_failed", error=str(e))


def docker_generator_node(state: dict, config: RunnableConfig) -> dict:
    """Generate production-ready Dockerfile and docker-compose.yml.

    Args:
        state: Current chat state with Docker config request.
        config: RunnableConfig with thread_id and other metadata.

    Returns:
        Updated state with generated Docker configs and artifacts.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "")
    query = state["messages"][-1].content

    logger.info(
        "docker_generator_started", thread_id=thread_id, message_prefix=query[:50]
    )

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )

    project_details = _extract_project_details(query, llm)
    docker_configs = _generate_docker_configs(project_details, llm)

    _save_docker_config(
        thread_id,
        project_details.get("framework", "unknown"),
        docker_configs["dockerfile"],
        docker_configs["docker_compose"],
    )

    response_content = f"""## Generated Docker Configuration

### Dockerfile
```
{docker_configs["dockerfile"]}
```

### docker-compose.yml
```yaml
{docker_configs["docker_compose"]}
```

You can apply these files to your project by clicking the "Apply" button in the UI.
"""

    from langchain_core.messages import AIMessage

    response = AIMessage(content=response_content)

    logger.info("docker_generator_completed", thread_id=thread_id)

    return {
        "messages": [response],
        "generated_artifacts": {
            "dockerfile": docker_configs["dockerfile"],
            "docker_compose": docker_configs["docker_compose"],
        },
    }
