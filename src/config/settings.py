from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for DeployMate AI.

    All configuration is loaded from environment variables with fallback
    defaults for local development. For production, ensure all sensitive
    values are set via environment variables.

    Attributes:
        database_url: PostgreSQL connection string for checkpointer and store.
        llm_model: Ollama model name for chat completions.
        llm_base_url: Base URL for Ollama API (including port).
        embeddings_model: Ollama model for text embeddings.
        embeddings_base_url: Base URL for Ollama embeddings API.
        langchain_api_key: Optional API key for LangSmith tracing.
        langchain_project: Project name for LangSmith.
        langchain_tracing: Enable LangChain tracing v2.
        tavily_api_key: Optional API key for Tavily web search.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5442/postgres",
        description="PostgreSQL connection string",
    )

    llm_model: str = Field(
        default="qwen2.5-coder:0.5b",
        description="Ollama model for chat completions",
    )

    llm_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Ollama API base URL",
    )

    embeddings_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model for text embeddings",
    )

    embeddings_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Ollama embeddings API base URL",
    )

    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key for tracing (optional)",
    )

    langchain_project: str = Field(
        default="DeployMate-AI",
        description="LangSmith project name",
    )

    langchain_tracing: bool = Field(
        default=True,
        description="Enable LangChain tracing v2",
    )

    tavily_api_key: str = Field(
        default="",
        description="Tavily API key for web search (optional)",
    )

    github_token: str = Field(
        default="",
        description="GitHub PAT (read-only scope)",
    )


settings = Settings()
