import streamlit as st
from src.graph.builder import chatbot
from src.config.settings import settings
from langchain_core.messages import HumanMessage
import uuid


# -----------------------------utility functions-----------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    st.session_state["thread_names"][str(thread_id)] = "Chat"


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversations(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return messages
    except Exception:
        return []  # Return an empty list if there's an error


def get_thread_name(thread_id):
    return st.session_state["thread_names"].get(str(thread_id), "Chat")


def set_thread_name(thread_id, first_message):
    # Pehle 30 characters se naam banao
    name = first_message[:30] + "..." if len(first_message) > 30 else first_message
    st.session_state["thread_names"][str(thread_id)] = name


# --------------------Session Setup--------------------------

message_history = []

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {}

# -------------------Sidebar UI-------------------------------
st.sidebar.image("deploymate-logo.svg", use_container_width=True)
st.sidebar.title("DeployMate AI")
st.sidebar.caption("Your AI DevOps Co-Pilot")

if settings.github_token:
    st.sidebar.success("✅ GitHub Connected")
else:
    st.sidebar.warning("⚠️ GitHub token not set — add GITHUB_TOKEN to .env")

if st.sidebar.button("New Chat"):
    reset_chat()

for thread_id in st.session_state["chat_threads"][
    ::-1
]:  # Display threads in reverse order (latest first)
    thread_name = get_thread_name(thread_id)
    if st.sidebar.button(thread_name, key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversations(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"

            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state["message_history"] = temp_messages

# --------------------Main UI-------------------------------
# loading the message history.
for message in st.session_state["message_history"]:
    with st.chat_message(
        message["role"], avatar="👨‍💻" if message["role"] == "user" else "🚀"
    ):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    # First add the message in message history.
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    current_thread = str(st.session_state["thread_id"])
    if st.session_state["thread_names"].get(current_thread) in [None, "Chat"]:
        set_thread_name(current_thread, user_input)

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(user_input)

    # Streaming implemented..

    with st.chat_message("assistant", avatar="🚀"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
