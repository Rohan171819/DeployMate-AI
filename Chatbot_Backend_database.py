
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


# Local Ollama Model
llm = ChatOllama(model="llama3.2:3b")


from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages : Annotated[List[BaseMessage],add_messages]


def chat_node(state:ChatState):
    #taking user query from the state.
    messages = state['messages']
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