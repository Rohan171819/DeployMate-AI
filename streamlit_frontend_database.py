import streamlit as st
from chatbot_backend_database_SystemPrompt import chatbot,retrieve_all_threads, ingest_pdf
from langchain_core.messages import HumanMessage
import uuid


#-----------------------------utility functions-----------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return (thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['thread_names'][str(thread_id)] = "Chat"
    st.session_state['uploaded_file_name'] = None

def add_thread(thread_id):
    if st.session_state['chat_threads'] is None:
        st.session_state['chat_threads'] = []
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversations(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])  
        return messages
    except Exception:
        return []  # Return an empty list if there's an error
    
def get_thread_name(thread_id):
    return st.session_state['thread_names'].get(str(thread_id), "Chat")

def set_thread_name(thread_id, first_message):
    # Pehle 30 characters se naam banao
    name = first_message[:30] + "..." if len(first_message) > 30 else first_message
    st.session_state['thread_names'][str(thread_id)] = name


#--------------------Session Setup--------------------------

message_history =[]

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

#CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
          "metadata": {"thread_id": st.session_state['thread_id']},
          "run_name": "chat_run",}

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads() or []

add_thread(st.session_state['thread_id'])

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

if 'uploaded_file_name' not in st.session_state:  
    st.session_state['uploaded_file_name'] = None

#-------------------Sidebar UI-------------------------------
st.sidebar.image("deploymate-logo.svg", use_container_width=True)
st.sidebar.title("DeployMate AI")
st.sidebar.caption("Your AI DevOps Co-Pilot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()
    st.rerun()
st.sidebar.divider()

# ─── PDF UPLOAD — SIDEBAR MEIN 👈 ────────────────────────
st.sidebar.markdown("### 📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF for RAG",
    type=["pdf"],
    help="Upload any DevOps doc, error log PDF, or guide"
)
if uploaded_file:
    if st.session_state['uploaded_file_name'] != uploaded_file.name:
        with st.spinner("⚡ Processing PDF in background..."):
            result = ingest_pdf(
                file_bytes=uploaded_file.read(),
                thread_id=str(st.session_state['thread_id']),
                filename=uploaded_file.name
            )
        st.session_state['uploaded_file_name'] = uploaded_file.name
        # Success info dikhao
        st.sidebar.success(
                f"**{result['filename']}**\n\n"
                f"Pages: {result['documents']} | "
                f"Chunks: {result['chunks']}"
            )
    else:
        # Already uploaded — sirf naam dikhao
        st.sidebar.success(f"✅ **{st.session_state['uploaded_file_name']}** loaded")

st.sidebar.divider()


st.sidebar.markdown("### 💬 Conversations")
for thread_id in st.session_state['chat_threads'][::-1]: # Display threads in reverse order (latest first)
    thread_name = get_thread_name(thread_id)
    if st.sidebar.button(thread_name,key = str(thread_id)):
        st.session_state['thread_id'] = thread_id
        st.session_state['uploaded_file_name'] = None
        messages = load_conversations(thread_id)

        temp_messages =[]
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"

            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state['message_history'] = temp_messages
        st.rerun()

#--------------------Main UI-------------------------------
if st.session_state['uploaded_file_name']:
    st.info(f"📄 **{st.session_state['uploaded_file_name']}** is active — Ask anything about this document!")


# loading the message history.
for message in st.session_state['message_history']:
    with st.chat_message(message["role"], avatar="👨‍💻" if message["role"] == "user" else "🚀"):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message or ask about uploaded or ask about uploaded PDF...")

if user_input:

    # First add the message in message history.
    st.session_state['message_history'].append({"role": "user", "content": user_input}) 

    current_thread = str(st.session_state['thread_id'])
    if st.session_state['thread_names'].get(current_thread) in [None, "Chat"]:
        set_thread_name(current_thread, user_input)
        
    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(user_input)

    # Streaming implemented..

    with st.chat_message("assistant", avatar="🚀"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode= "messages"
            )
        )
    
    st.session_state['message_history'].append({"role": "assistant", "content":ai_message }) 