
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
import os
from langgraph.checkpoint.memory import MemorySaver


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


checkpinter = MemorySaver()
graph = StateGraph(ChatState)

# addinng nodes..
graph.add_node('chat_node',chat_node)

#adding edges.
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpinter)


