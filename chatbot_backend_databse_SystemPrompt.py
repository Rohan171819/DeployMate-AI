from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
load_dotenv() 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "DeployMate-AI"

# Local Ollama Model
llm = ChatOllama(model="llama3.2:3b")

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

# Helper functions to detect message intent based on keywords.
def is_error_message(message: str) -> bool:
    error_keywords = [
        "error", "traceback", "exception", "failed",
        "exit code", "cannot", "unable to", "not found",
        "permission denied", "connection refused",
        "modulenotfounderror", "syntaxerror", "typeerror",
        "valueerror", "importerror", "runtimeerror",
        "docker", "container", "pipeline", "deployment failed"
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
    return any(keyword in message for keyword in deploy_keywords)

def is_code_review_message(message: str) -> bool:
    code_keywords = [
        "review my code", "check my code", "code review",
        "is this code good", "improve my code", "optimize",
        "security issue", "bad practice", "refactor",
        "```", "def ", "class ", "function", "import "
    ]
    return any(keyword in message for keyword in code_keywords)

# ─── NODES ──────────────────────────────────────────────
def chat_node(state:ChatState):
    #taking user query from the state.
    messages = [system_prompt] +state['messages']
    #send to llm
    response = llm.invoke(messages)
    #store to state.
    return {'messages' : [response]}

def error_analyzer_node(state: ChatState):
    messages = [error_analyzer_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def fix_suggester_node(state: ChatState):
    messages = [fix_suggester_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def route_message(state: ChatState):
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
    
def deploy_guide_node(state: ChatState):
    messages = [deploy_guide_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

def code_review_node(state: ChatState):
    messages = [code_review_prompt] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}



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
graph.add_edge('error_analyzer_node', 'fix_suggester_node')

# Both fix suggester and regular chat lead to END, allowing the conversation to conclude after either path.
graph.add_edge('fix_suggester_node', END)
graph.add_edge('chat_node', END)
graph.add_edge('deploy_guide_node', END)
graph.add_edge('code_review_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)