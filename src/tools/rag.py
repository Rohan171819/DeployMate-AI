"""Tools module - RAG and PDF ingestion."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import structlog
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.exceptions import PDFIngestionError, RetrievalError

logger = structlog.get_logger()

# Global thread retriever storage
_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict[str, Any]] = {}
_RAG_CACHE: dict[str, str] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str | None = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Args:
        file_bytes: Raw PDF file content as bytes.
        thread_id: Unique identifier for the chat thread.
        filename: Optional original filename for metadata.

    Returns:
        Summary dictionary with filename, document count, and chunk count.

    Raises:
        PDFIngestionError: If file processing fails.
    """
    if not file_bytes:
        raise PDFIngestionError("No file bytes provided for PDF ingestion")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        logger.info("loading_pdf", thread_id=thread_id, filename=filename)
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        from src.tools.rag import get_embeddings

        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_file_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        logger.info(
            "pdf_ingested",
            thread_id=thread_id,
            filename=filename,
            documents=len(docs),
            chunks=len(chunks),
        )

        return {
            "filename": filename or os.path.basename(temp_file_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    except PDFIngestionError:
        raise
    except Exception as e:
        logger.error("pdf_ingestion_failed", thread_id=thread_id, error=str(e))
        raise PDFIngestionError(f"Failed to ingest PDF: {e}", filename=filename) from e
    finally:
        if temp_file_path:
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def _get_retriever(thread_id: str) -> Any:
    """Get retriever for a specific thread."""
    return _THREAD_RETRIEVERS.get(str(thread_id))


def retrieve_relevant_docs(thread_id: str, query: str) -> list[str]:
    """Retrieve relevant documents from FAISS vector store for a query.

    Args:
        thread_id: Unique identifier for the chat thread.
        query: Search query string.

    Returns:
        List of document content strings.

    Raises:
        RetrievalError: If retrieval fails.
    """
    cache_key = f"{thread_id}:{query[:50]}"

    if cache_key in _RAG_CACHE:
        logger.debug("rag_cache_hit", thread_id=thread_id, query_prefix=query[:30])
        return _RAG_CACHE[cache_key].split("\n\n") if _RAG_CACHE[cache_key] else []

    retriever = _get_retriever(thread_id)
    if not retriever:
        logger.warning("no_retriever_found", thread_id=thread_id)
        return []

    try:
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        _RAG_CACHE[cache_key] = context
        logger.info("docs_retrieved", thread_id=thread_id, doc_count=len(docs))
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error("retrieval_failed", thread_id=thread_id, error=str(e))
        raise RetrievalError(
            f"Failed to retrieve documents: {e}", thread_id=thread_id
        ) from e


def get_rag_context(thread_id: str, query: str) -> str:
    """Get RAG context string for a query.

    Args:
        thread_id: Unique identifier for the chat thread.
        query: Search query string.

    Returns:
        Context string from retrieved documents, or empty string if none found.
    """
    docs = retrieve_relevant_docs(thread_id, query)
    return "\n\n".join(docs)


def get_rag_context_with_fallback(thread_id: str, query: str) -> dict:
    """Get RAG context with web search fallback.

    Args:
        thread_id: Unique identifier for the chat thread.
        query: Search query string.

    Returns:
        Dictionary with context, source (pdf/web/none), and needs_fallback flag.
    """
    from langchain_community.tools import TavilySearchResults

    cache_key = f"crag:{thread_id}:{query[:80]}"
    if cache_key in _RAG_CACHE:
        cached = _RAG_CACHE[cache_key]
        if isinstance(cached, dict):
            return cached

    retriever = _get_retriever(thread_id)
    if not retriever:
        return {
            "context": "",
            "source": "none",
            "needs_fallback": True,
        }

    try:
        docs = retriever.invoke(query)
    except Exception as e:
        logger.warning(
            "retrieval_fallback_triggered", thread_id=thread_id, error=str(e)
        )
        return _web_fallback(query)

    filtered_docs = []
    for doc in docs:
        is_relevant = _grade_single_document(query, doc.page_content)
        if is_relevant:
            filtered_docs.append(doc)

    needs_fallback = len(filtered_docs) == 0

    if needs_fallback:
        return _web_fallback(query)

    context = "\n\n".join([doc.page_content for doc in filtered_docs])
    result = {
        "context": context,
        "source": "pdf",
        "needs_fallback": False,
    }

    _RAG_CACHE[cache_key] = result
    return result


def _web_fallback(query: str) -> dict:
    """Fallback to web search when no local docs found."""
    from langchain_community.tools import TavilySearchResults

    try:
        web_search_tool = TavilySearchResults(k=3)
        web_docs = web_search_tool.invoke({"query": query})
        web_text = "\n\n".join([d["content"] for d in web_docs]) if web_docs else ""
        logger.info(
            "web_search_fallback", query_prefix=query[:30], results=len(web_docs)
        )
        return {
            "context": web_text,
            "source": "web",
            "needs_fallback": True,
        }
    except Exception as e:
        logger.warning("web_fallback_failed", error=str(e))
        return {
            "context": "",
            "source": "none",
            "needs_fallback": True,
        }


def _grade_single_document(question: str, document: str) -> bool:
    """Grade if a document is relevant to the question using LLM.

    Args:
        question: User question.
        document: Document content to grade.

    Returns:
        True if relevant, False otherwise.
    """
    from pydantic import BaseModel

    class GradeScore(BaseModel):
        score: int

    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    structured_llm = llm.with_structured_output(GradeScore)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document "
                "to a user question about DevOps/deployment errors. "
                "If the document contains keyword(s) or semantic meaning related to the question, "
                "grade it as relevant. Give a binary 'yes' or 'no' score.",
            ),
            ("human", "Retrieved document:\n{document}\n\nUser question: {question}"),
        ]
    )

    try:
        result = structured_llm.invoke(
            grade_prompt.format_messages(question=question, document=document)
        )
        return result.score >= 5
    except Exception:
        return True  # Default to relevant if grading fails


# Lazy initialization for embeddings
_embeddings_instance = None


def get_embeddings():
    """Get or create the embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_ollama import OllamaEmbeddings

        _embeddings_instance = OllamaEmbeddings(
            model=settings.embeddings_model,
            base_url=settings.embeddings_base_url,
        )
    return _embeddings_instance
