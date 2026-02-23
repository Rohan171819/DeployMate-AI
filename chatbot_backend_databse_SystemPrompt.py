
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

load_dotenv() 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "DeployMate-AI"

# Local Ollama Model
llm = ChatOllama(model="llama3.2:3b")

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



from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages : Annotated[List[BaseMessage],add_messages]


def chat_node(state:ChatState):
    #taking user query from the state.
    messages = [system_prompt] +state['messages']
    #send to llm
    response = llm.invoke(messages)
    #store to state.
    return {'messages' : [response]}



conn = sqlite3.connect(database = 'chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(ChatState)

# addinng nodes..
graph.add_node('chat_node',chat_node)

#adding edges.
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)