from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
import os
import tempfile
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

load_dotenv() 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY","")
os.environ["LANGCHAIN_PROJECT"] = "DeployMate-AI"

# Local Ollama Model

llm = ChatOllama(
    model="llama3.2:3b",
    base_url="http://host.docker.internal:11434"  
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"  
)

#---------------------Prompts-----------------------
system_prompt = SystemMessage(content="""
You are DeployMate AI — an expert DevOps co-pilot for junior developers.
You specialize in:
- Docker errors and containerization
- CI/CD pipeline setup and failures  
- Production error log debugging
- Cloud deployment (AWS, Railway, Render, VPS)
- Code review for security and performance

Always provide clear, step-by-step, beginner-friendly guidance.
When user shares an error, identify root cause first, then provide exact fix.
""")

error_analyzer_prompt = SystemMessage(content="""
You are the Error Analyzer Agent inside DeployMate AI.
When given an error, ALWAYS respond in this exact structure:

## 🔍 Root Cause
Explain what caused this error in simple language.

## 💥 Why It Happened  
Explain the reason behind this error.

## ✅ Exact Fix
Provide the exact fix with code snippet.

## 🛡️ Prevention
How to avoid this error in future.

Be beginner-friendly, clear, and precise.
""")

fix_suggester_prompt = SystemMessage(content="""
You are the Fix Suggester Agent inside DeployMate AI.
Your job is to provide precise, working code fixes.

ALWAYS:
1. Detect the user's tech stack automatically from context
2. Provide complete, copy-paste ready fix
3. Explain what each fix does line by line
4. Show before vs after code comparison
5. Make sure developer LEARNS, not just copy-pastes

Format your response clearly with code blocks.
""")

deploy_guide_prompt = SystemMessage(content="""
You are the Deployment Guide Agent inside DeployMate AI.
When user wants to deploy, ALWAYS respond in this structure:

## 🎯 Deployment Plan
Identify what they want to deploy and where.

## 📋 Prerequisites
What needs to be ready before deploying.

## 🚀 Step-by-Step Guide
Exact commands and steps — numbered clearly.

## ⚙️ Configuration
Dockerfile, env vars, ports — whatever is needed.

## ✅ Verification
How to confirm deployment was successful.

Be specific, beginner-friendly, with exact commands.
""")

code_review_prompt = SystemMessage(content="""
You are the Code Review Agent inside DeployMate AI.
When given code, ALWAYS respond in this structure:

## 🔍 Code Summary
What this code does briefly.

## 🚨 Security Issues
Any vulnerabilities found — with line references.

## ⚡ Performance Issues  
Any bottlenecks or inefficiencies found.

## ❌ Bad Practices
Anti-patterns or poor coding practices.

## ✅ Improved Code
Provide the fixed, improved version with explanations.

Be constructive, educational, and specific.
""")




class ChatState(TypedDict):
    messages : Annotated[List[BaseMessage],add_messages]


# ─── ERROR DETECTOR ───────────────────────────────────────
_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}

_RAG_CACHE = {}

# HITL Only for the Dangerous keywords..
DANGEROUS_KEYWORDS = [
    "rm -rf", "drop database", "delete all",
    "format", "truncate", "sudo rm",
    "chmod 777", "iptables -F","rm -rf", "drop database", "delete all",
    "format", "truncate", "sudo rm",
    "chmod 777", "iptables -F", "dd if=",
    "> /dev/sda", "mkfs", "fdisk"
]


# Helper functions to detect message intent based on keywords.
def is_error_message(message: str) -> bool:
    error_keywords = [
        "error", "traceback", "exception", "failed",
        "exit code", "cannot", "unable to", "not found",
        "permission denied", "connection refused",
        "modulenotfounderror", "syntaxerror", "typeerror",
        "valueerror", "importerror", "runtimeerror",
        "docker", "container", "pipeline", "deployment failed",
        "rm -rf", "sudo rm", "drop database",
        "delete all", "format", "truncate",
        "chmod 777", "iptables", "disk full",
        "storage full", "clean up", "free space"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in error_keywords)


def is_deploy_message(message: str) -> bool:
    deploy_keywords = [
        "deploy", "deployment", "publish", "host", "hosting",
        "aws", "railway", "render", "vps", "ec2",
        "go live", "production", "server", "cloud",
        "dockerfile", "docker compose", "nginx"
    ]
    return any(keyword in message.lower() for keyword in deploy_keywords)


def is_code_review_message(message: str) -> bool:
    code_keywords = [
        "review my code", "check my code", "code review",
        "is this code good", "improve my code", "optimize",
        "security issue", "bad practice", "refactor",
        "```", "def ", "class ", "function", "import "
    ]
    return any(keyword in message.lower() for keyword in code_keywords)


def ingest_pdf(file_bytes: bytes,thread_id: str,filename: Optional[str] = None)-> dict:
    """Build  a FAISS retriever for the uploded PDF and store it for the thread.
        Returns a summart dict that can be surfaced in he UI.
    """
    if not file_bytes:
        raise ValueError("No file received for PDF ingestion.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100,separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks,embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_file_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return{
            "filename": filename or os.path.basename(temp_file_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps the copies of the text,so the temp file is safe to remove.
        try:
            os.remove(temp_file_path)
        except OSError:
            pass


def _get_retriever(thread_id):
    return _THREAD_RETRIEVERS.get(str(thread_id))

def get_rag_context(thread_id: str, query:str)-> str:
    cache_key = f"{thread_id}:{query[:50]}"

    # if the context is in the cache to add directly in Cache.
    if cache_key in _RAG_CACHE:
        return _RAG_CACHE[cache_key]
    
    retriever = _get_retriever(thread_id)
    if not retriever:
        return ""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    #Storing the data in the cache.
    _RAG_CACHE[cache_key] = context
    return context


def get_thread_id_from_config(config: RunnableConfig) -> str:
    return config.get("configurable", {}).get("thread_id", "")


def is_dangerous(response_text: str) -> bool:
    return any(kw in response_text.lower() for kw in DANGEROUS_KEYWORDS)


# ─── NODES ──────────────────────────────────────────────

def chat_node(state: ChatState, config: RunnableConfig):
    thread_id = get_thread_id_from_config(config)
    query = state["messages"][-1].content

    if is_dangerous(query):
        human_decision = interrupt({
            "type": "dangerous_command",
            "message": "⚠️ Dangerous command detected!",
            "suggested_response": f"User asked for dangerous command: {query}",
        })
        if not human_decision.get("approved"):
            return {'messages': [SystemMessage(content="""
I recommend NOT running this command — it can be destructive!

Safer alternatives:
- `du -sh /var/log/*` — check log sizes first
- `journalctl --vacuum-size=100M` — safely clean logs  
- `find /var/log -name "*.gz" -delete` — only compressed logs
""")]}

    rag_context = get_rag_context(thread_id, query)
    
    # Context ko system prompt mein inject karo
    if rag_context:
        dynamic_prompt = SystemMessage(content=f"""
{system_prompt.content}

DOCUMENT CONTEXT (Answer based on this):
{rag_context}
""")
    else:
        dynamic_prompt = system_prompt

    messages = [dynamic_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def error_analyzer_node(state: ChatState, config: RunnableConfig):
    thread_id = get_thread_id_from_config(config)
    query = state["messages"][-1].content
    rag_context = get_rag_context(thread_id, query)

    # ─── HITL — User input pe check karo PEHLE ───
    if is_dangerous(query):
        human_decision = interrupt({
            "type": "dangerous_command",
            "message": "⚠️ Dangerous command detected!",
            "suggested_response": f"User asked for dangerous command: {query}",
        })

        if human_decision.get("approved"):
            # ✅ Approved → Original response do
            if rag_context:
                prompt = SystemMessage(content=f"{error_analyzer_prompt.content}\nDOCUMENT CONTEXT:\n{rag_context}")
            else:
                prompt = error_analyzer_prompt
            messages = [prompt] + state['messages']
            response = llm.invoke(messages)
            return {'messages': [response]}
        else:
            # ❌ Rejected → Safe alternative do
            return {'messages': [SystemMessage(content="""
I recommend NOT running this command as it can be destructive.

Here are safer alternatives:
- Use `du -sh /var/log/*` to check log sizes first
- Use `journalctl --vacuum-size=100M` to safely clean logs
- Use `find /var/log -name "*.gz" -delete` for compressed logs only
""")]}

    # ─── Normal Flow — No dangerous command ───────
    if rag_context:
        prompt = SystemMessage(content=f"{error_analyzer_prompt.content}\nDOCUMENT CONTEXT:\n{rag_context}")
    else:
        prompt = error_analyzer_prompt

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def fix_suggester_node(state: ChatState, config: RunnableConfig):
    thread_id = get_thread_id_from_config(config)
    
    query = state["messages"][-1].content
    rag_context = get_rag_context(thread_id, query)

    if rag_context:
        prompt = SystemMessage(content=f"""
{fix_suggester_prompt.content}

DOCUMENT CONTEXT:
{rag_context}
""")
    else:
        prompt = fix_suggester_prompt

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def route_message(state: ChatState,config: RunnableConfig):
    last_message = state['messages'][-1].content
    if is_error_message(last_message):
        return "error_analyzer_node"
    elif is_deploy_message(last_message):
        return "deploy_guide_node"
    elif is_code_review_message(last_message):
        return "code_review_node"
        
    # General question
    else:
        return "chat_node"


def deploy_guide_node(state: ChatState, config: RunnableConfig):
    thread_id = get_thread_id_from_config(config)
    
    query = state["messages"][-1].content
    rag_context = get_rag_context(thread_id, query)

    if rag_context:
        prompt = SystemMessage(content=f"""
{deploy_guide_prompt.content}

DOCUMENT CONTEXT:
{rag_context}
""")
    else:
        prompt = deploy_guide_prompt

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def code_review_node(state: ChatState, config: RunnableConfig):
    thread_id = get_thread_id_from_config(config)
    
    query = state["messages"][-1].content
    rag_context = get_rag_context(thread_id, query)

    if rag_context:
        prompt = SystemMessage(content=f"""
{code_review_prompt.content}

DOCUMENT CONTEXT:
{rag_context}
""")
    else:
        prompt = code_review_prompt

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """ Retrieve the relevant information from the uploded file for this chat thread.
    Always include the thread_id when calling this tool."""

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF First.",
            "query": query,}
    
    result = retriever.invoke(query) 
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


conn = sqlite3.connect(database = 'chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(ChatState)

# addinng nodes..
graph.add_node('chat_node', chat_node)
graph.add_node('error_analyzer_node', error_analyzer_node)
graph.add_node('fix_suggester_node', fix_suggester_node)
graph.add_node('deploy_guide_node', deploy_guide_node)
graph.add_node('code_review_node', code_review_node)

# Conditional routing — START to either error analyzer or regular chat based on message content.
graph.add_conditional_edges(START, route_message, {
    "error_analyzer_node": "error_analyzer_node",
    "chat_node": "chat_node",
    "deploy_guide_node": "deploy_guide_node",
    "code_review_node": "code_review_node",
})

#adding edges.
graph.add_edge('error_analyzer_node', END)

# Both fix suggester and regular chat lead to END, allowing the conversation to conclude after either path.
graph.add_edge('fix_suggester_node', END)
graph.add_edge('chat_node', END)
graph.add_edge('deploy_guide_node', END)
graph.add_edge('code_review_node', END)

chatbot = graph.compile(checkpointer=checkpointer, interrupt_before=[])

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)