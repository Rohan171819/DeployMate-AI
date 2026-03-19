from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg2
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

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5442/postgres")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY","")
os.environ["LANGCHAIN_PROJECT"] = "DeployMate-AI"

# Local Ollama Model

llm = ChatOllama(
    model="llama3.2:3b",
    #base_url="http://host.docker.internal:11434",
    streaming=True
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    #base_url="http://host.docker.internal:11434"  
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

class ErrorAnalysisState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    error_type: str        # Docker/CI/Python etc
    severity: str          # Critical/Warning/Info
    has_fix: bool          # Fix mila ya nahi

class DeploymentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tech_stack: str        # Python/Node/React etc
    target_platform: str   # AWS/Railway/Render/VPS
    has_dockerfile: bool   # Dockerfile needed?

class CodeReviewState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    language: str          # Python/JS/Java etc
    has_security_issue: bool
    has_performance_issue: bool

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


#--------------------------SubGraph Nodes-------------------------------------------

def error_parser_node(state: ErrorAnalysisState, config: RunnableConfig):
    """Identify what type of error this is."""
    query = state['messages'][-1].content
    
    # Error type detect karo
    if "docker" in query.lower():
        error_type = "Docker"
    elif "pipeline" in query.lower() or "ci/cd" in query.lower():
        error_type = "CI/CD"
    elif "traceback" in query.lower():
        error_type = "Python"
    else:
        error_type = "General"
    
    return {"error_type": error_type}

def severity_checker_node(state: ErrorAnalysisState, config: RunnableConfig):
    """Check how critical this error is."""
    query = state['messages'][-1].content
    
    critical_keywords = ["production", "down", "crash", "data loss"]
    severity = "Critical" if any(kw in query.lower() for kw in critical_keywords) else "Warning"
    
    return {"severity": severity}

def solution_finder_node(state: ErrorAnalysisState, config: RunnableConfig):
    """Find the solution based on error type and severity."""
    thread_id = get_thread_id_from_config(config)
    query = state['messages'][-1].content
    rag_context = get_rag_context(thread_id, query)
    
    # Dynamic prompt based on error type and severity
    prompt = SystemMessage(content=f"""
You are the Error Analyzer Agent inside DeployMate AI.
Error Type: {state['error_type']}
Severity: {state['severity']}

{'⚠️ CRITICAL ERROR — Provide immediate fix first!' if state['severity'] == 'Critical' else ''}

ALWAYS respond in this structure:
## 🔍 Root Cause
## 💥 Why It Happened
## ✅ Exact Fix
## 🛡️ Prevention

{f'DOCUMENT CONTEXT: {rag_context}' if rag_context else ''}
""")
    
    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response], 'has_fix': True}

def fix_validator_node(state: ErrorAnalysisState, config: RunnableConfig):
    """Validate if the fix is complete and safe."""
    # HITL check — dangerous hai?
    if is_dangerous(state['messages'][-1].content):
        human_decision = interrupt({
            "type": "dangerous_command",
            "message": "⚠️ Dangerous command detected!",
            "suggested_response": state['messages'][-1].content,
        })
        if not human_decision.get("approved"):
            safe_response = llm.invoke([
                SystemMessage(content="Provide a SAFER alternative."),
                *state['messages']
            ])
            return {'messages': [safe_response]}
    
    return state

def build_error_analysis_subgraph():
    """Build and return the error analysis subgraph."""
    subgraph = StateGraph(ErrorAnalysisState)
    
    #Adding the Nodes..
    subgraph.add_node("error_parser", error_parser_node)
    subgraph.add_node("severity_checker", severity_checker_node)
    subgraph.add_node("solution_finder", solution_finder_node)
    subgraph.add_node("fix_validator", fix_validator_node)

    # Edges add karo
    subgraph.add_edge(START, "error_parser")
    subgraph.add_edge("error_parser", "severity_checker")
    subgraph.add_edge("severity_checker", "solution_finder")
    subgraph.add_edge("solution_finder", "fix_validator")
    subgraph.add_edge("fix_validator", END)

    return subgraph.compile()

#----------------------------------------------------------
def stack_detector_node(state: DeploymentState, config: RunnableConfig):
    """Detect user's tech stack from message."""
    query = state['messages'][-1].content.lower()
    
    stacks = {
        "python": ["python", "flask", "django", "fastapi"],
        "nodejs": ["node", "express", "npm", "javascript"],
        "react": ["react", "nextjs", "vite"],
        "docker": ["docker", "dockerfile", "container"],
    }
    
    detected = "general"
    for stack, keywords in stacks.items():
        if any(kw in query for kw in keywords):
            detected = stack
            break
    
    return {"tech_stack": detected}

def platform_selector_node(state: DeploymentState, config: RunnableConfig):
    """Detect target deployment platform."""
    query = state['messages'][-1].content.lower()
    
    platforms = {
        "railway": ["railway"],
        "aws": ["aws", "ec2", "s3"],
        "render": ["render"],
        "vps": ["vps", "digitalocean", "linode"],
    }
    
    detected = "railway"  # Default — easiest for beginners
    for platform, keywords in platforms.items():
        if any(kw in query for kw in keywords):
            detected = platform
            break
    
    return {"target_platform": detected}

def config_generator_node(state: DeploymentState, config: RunnableConfig):
    """Generate deployment configuration."""
    thread_id = get_thread_id_from_config(config)
    query = state['messages'][-1].content
    rag_context = get_rag_context(thread_id, query)

    prompt = SystemMessage(content=f"""
You are the Deployment Guide Agent inside DeployMate AI.
Tech Stack Detected: {state['tech_stack']}
Target Platform: {state['target_platform']}

Provide deployment configuration specific to {state['tech_stack']} on {state['target_platform']}.

ALWAYS respond in this structure:

## 🎯 Deployment Plan
What will be deployed and where.

## 📋 Prerequisites
What needs to be ready before deploying.

## ⚙️ Configuration
Dockerfile, env vars, ports for {state['tech_stack']}.

## 🚀 Step-by-Step Guide
Exact commands for {state['target_platform']}.

## ✅ Verification
How to confirm deployment was successful.

{f'DOCUMENT CONTEXT: {rag_context}' if rag_context else ''}
""")

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def steps_generator_node(state: DeploymentState, config: RunnableConfig):
    """Generate final deployment checklist."""
    # Just pass through — config_generator already complete response deta hai
    return state

def build_deployment_subgraph():
    """Build and return the deployment pipeline subgraph."""
    subgraph = StateGraph(DeploymentState)

    subgraph.add_node("stack_detector", stack_detector_node)
    subgraph.add_node("platform_selector", platform_selector_node)
    subgraph.add_node("config_generator", config_generator_node)
    subgraph.add_node("steps_generator", steps_generator_node)

    subgraph.add_edge(START, "stack_detector")
    subgraph.add_edge("stack_detector", "platform_selector")
    subgraph.add_edge("platform_selector", "config_generator")
    subgraph.add_edge("config_generator", "steps_generator")
    subgraph.add_edge("steps_generator", END)

    return subgraph.compile()
#--------------------------------------------------------------------------

def language_detector_node(state: CodeReviewState, config: RunnableConfig):
    """Detect programming language from code."""
    query = state['messages'][-1].content.lower()
    
    languages = {
        "python": ["def ", "import ", "print(", ".py", "django", "flask"],
        "javascript": ["function", "const ", "let ", "var ", "=>", ".js"],
        "java": ["public class", "void ", "system.out", ".java"],
        "sql": ["select ", "insert ", "update ", "delete ", "drop "],
        "bash": ["#!/bin/bash", "chmod", "sudo", "apt-get"],
        "dockerfile": ["from ", "run ", "cmd ", "expose", "workdir"],
    }
    
    detected = "general"
    for lang, keywords in languages.items():
        if any(kw in query for kw in keywords):
            detected = lang
            break
    
    return {"language": detected}

def security_scanner_node(state: CodeReviewState, config: RunnableConfig):
    """Scan code for security vulnerabilities."""
    query = state['messages'][-1].content.lower()
    
    security_red_flags = [
        "sql", "select *", "exec(", "eval(",
        "password", "secret", "api_key",
        "shell=true", "subprocess",
        "md5", "sha1",
    ]
    
    has_issue = any(flag in query for flag in security_red_flags)
    return {"has_security_issue": has_issue}

def performance_scanner_node(state: CodeReviewState, config: RunnableConfig):
    """Scan code for performance issues."""
    query = state['messages'][-1].content.lower()
    
    performance_red_flags = [
        "for i in range", "while true",
        "select *", "time.sleep",
        "global ", "nested for",
        ".append(", "string +",
    ]
    
    has_issue = any(flag in query for flag in performance_red_flags)
    return {"has_performance_issue": has_issue}

def review_generator_node(state: CodeReviewState, config: RunnableConfig):
    """Generate comprehensive code review."""
    thread_id = get_thread_id_from_config(config)
    query = state['messages'][-1].content
    rag_context = get_rag_context(thread_id, query)

    # Dynamic prompt based on detected issues
    security_note = "⚠️ SECURITY ISSUES DETECTED — Review carefully!" if state['has_security_issue'] else "No obvious security issues detected."
    performance_note = "⚠️ PERFORMANCE ISSUES DETECTED — Optimize!" if state['has_performance_issue'] else "No obvious performance issues detected."

    prompt = SystemMessage(content=f"""
You are the Code Review Agent inside DeployMate AI.
Language Detected: {state['language']}
Security Pre-scan: {security_note}
Performance Pre-scan: {performance_note}

ALWAYS respond in this structure:

## 🔍 Code Summary
What this code does briefly.

## 🚨 Security Issues
{'Focus heavily here — issues were pre-detected!' if state['has_security_issue'] else 'Check for any vulnerabilities.'}

## ⚡ Performance Issues
{'Focus heavily here — issues were pre-detected!' if state['has_performance_issue'] else 'Check for bottlenecks.'}

## ❌ Bad Practices
Anti-patterns specific to {state['language']}.

## ✅ Improved Code
Provide complete fixed version with explanations.

{f'DOCUMENT CONTEXT: {rag_context}' if rag_context else ''}
""")

    messages = [prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def build_code_review_subgraph():
    """Build and return the code review pipeline subgraph."""
    subgraph = StateGraph(CodeReviewState)

    subgraph.add_node("language_detector", language_detector_node)
    subgraph.add_node("security_scanner", security_scanner_node)
    subgraph.add_node("performance_scanner", performance_scanner_node)
    subgraph.add_node("review_generator", review_generator_node)

    subgraph.add_edge(START, "language_detector")
    subgraph.add_edge("language_detector", "security_scanner")
    subgraph.add_edge("security_scanner", "performance_scanner")
    subgraph.add_edge("performance_scanner", "review_generator")
    subgraph.add_edge("review_generator", END)

    return subgraph.compile()

#_________________________________________________________________________________

error_analysis_Subgraph = build_error_analysis_subgraph()
deployment_Subgraph = build_deployment_subgraph()
code_review_Subgraph = build_code_review_subgraph()



conn = psycopg2.connect(DB_URI)
checkpointer = PostgresSaver(conn = conn)
checkpointer.setup()

graph = StateGraph(ChatState)

# addinng nodes..
graph.add_node('chat_node', chat_node)
graph.add_node('error_analyzer_node', error_analysis_Subgraph)
graph.add_node('deploy_guide_node', deployment_Subgraph)
graph.add_node('code_review_node', code_review_Subgraph)

# Conditional routing — START to either error analyzer or regular chat based on message content.
graph.add_conditional_edges(START, route_message, {
    "error_analyzer_node": "error_analyzer_node",
    "chat_node": "chat_node",
    "deploy_guide_node": "deploy_guide_node",
    "code_review_node": "code_review_node",
})

#adding edges.
graph.add_edge('error_analyzer_node', END)
graph.add_edge('deploy_guide_node', END)
graph.add_edge('code_review_node', END)
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer, interrupt_before=[])


def retrieve_all_threads():
    """Retrieve all thread IDs from PostgreSQL checkpointer."""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)