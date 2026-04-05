import streamlit as st
from src.graph.builder import chatbot
from src.config.settings import settings
from langchain_core.messages import HumanMessage
import uuid
import os

st.set_page_config(
    page_title="DeployMate AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = """
<style>
    :root {
        --primary: #6C63FF;
        --secondary: #00D9FF;
        --dark: #0D0D0D;
        --card: #1A1A2E;
        --text: #E8E8E8;
        --accent: #FF6B6B;
    }
    
    .stApp {
        background-color: #0D0D0D;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1A1A2E !important;
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }
    
    .stChatInput input {
        background-color: #1A1A2E !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        color: #E8E8E8 !important;
        border-radius: 12px !important;
    }
    
    .stChatInput input::placeholder {
        color: #888 !important;
    }
    
    .stChatInput input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
    }
    
    div[data-testid="chat-message-user"] {
        background-color: rgba(108, 99, 255, 0.15) !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 16px !important;
    }
    
    div[data-testid="chat-message-assistant"] {
        background-color: #1A1A2E !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 16px !important;
    }
    
    .stMarkdown {
        color: #E8E8E8 !important;
    }
    
    .stButton > button {
        background-color: #1A1A2E !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        color: #E8E8E8 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        border-color: #6C63FF !important;
        background-color: rgba(108, 99, 255, 0.1) !important;
        transform: translateY(-2px);
    }
    
    .stButton > button[kind="primary"] {
        background-color: #FF6B6B !important;
        border-color: #FF6B6B !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #FF5252 !important;
        border-color: #FF5252 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        border: none !important;
        color: #E8E8E8 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(108, 99, 255, 0.15) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #E8E8E8 !important;
    }
    
    p, span, label {
        color: #E8E8E8 !important;
    }
    
    .stSuccess {
        background-color: rgba(39, 201, 63, 0.15) !important;
        border: 1px solid rgba(39, 201, 63, 0.3) !important;
        color: #27C93F !important;
    }
    
    .stWarning {
        background-color: rgba(255, 189, 46, 0.15) !important;
        border: 1px solid rgba(255, 189, 46, 0.3) !important;
    }
    
    .stError {
        background-color: rgba(255, 107, 107, 0.15) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        color: #FF6B6B !important;
    }
    
    .stTextInput > div > div {
        background-color: #1A1A2E !important;
        border: 1px solid rgba(108, 99, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
    }
    
    hr {
        border-color: rgba(108, 99, 255, 0.2) !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1A2E;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6C63FF;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7C73FF;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown > div {
        animation: fadeIn 0.3s ease;
    }
    
    /* Typing Animation */
    @keyframes typingDot {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #6C63FF;
        border-radius: 50%;
        margin: 0 2px;
        animation: typingDot 1.4s infinite ease-in-out both;
    }
    
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0s; }
    
    .typing-container {
        background-color: #1A1A2E !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 16px !important;
        display: inline-flex;
        gap: 4px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF 0%, #00D9FF 100%) !important;
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background-color: #1A1A2E !important;
        color: #E8E8E8 !important;
        border-radius: 8px !important;
    }
    
    /* Suggestion chips */
    .suggestion-chip {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.2) 0%, rgba(0, 217, 255, 0.2) 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #E8E8E8;
    }
    
    .suggestion-chip:hover {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.4) 0%, rgba(0, 217, 255, 0.4) 100%);
        transform: translateY(-2px);
        border-color: #6C63FF;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
"""

st.markdown(COLORS, unsafe_allow_html=True)


def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    st.session_state["thread_names"][str(thread_id)] = "Chat"
    st.session_state["context_suggestions"] = get_suggestions("")


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversations(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        return messages
    except Exception:
        return []


def get_thread_name(thread_id):
    return st.session_state["thread_names"].get(str(thread_id), "Chat")


def set_thread_name(thread_id, first_message):
    name = first_message[:30] + "..." if len(first_message) > 30 else first_message
    st.session_state["thread_names"][str(thread_id)] = name


def get_suggestions(last_message):
    """Get contextual suggestions based on conversation"""
    suggestions = []
    last_msg_lower = last_message.lower()

    if "docker" in last_msg_lower or "container" in last_msg_lower:
        suggestions = [
            "Explain exit code 137",
            "How to debug OOM errors?",
            "Docker compose not starting",
            "Port already in use error",
        ]
    elif "deploy" in last_msg_lower or "deployment" in last_msg_lower:
        suggestions = [
            "Deploy to Railway",
            "Deploy to Render",
            "AWS ECS deployment",
            "Dockerize my Node.js app",
        ]
    elif "error" in last_msg_lower or "fail" in last_msg_lower:
        suggestions = [
            "Analyze this error log",
            "Why is my CI failing?",
            "Python import errors",
            "Connection refused",
        ]
    elif "code" in last_msg_lower or "review" in last_msg_lower:
        suggestions = [
            "Review my PR",
            "Security issues in code",
            "Performance optimization",
            "Best practices check",
        ]
    else:
        suggestions = [
            "My Docker container keeps crashing",
            "How to deploy to Railway?",
            "Review my code for errors",
            "Explain this error message",
        ]

    return suggestions


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

if "context_suggestions" not in st.session_state:
    st.session_state["context_suggestions"] = get_suggestions("")


# --------------------Sidebar UI-------------------------------
st.sidebar.markdown(
    """
<style>
    .sidebar-title {
        font-size: 24px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""",
    unsafe_allow_html=True,
)

try:
    st.sidebar.image("deploymate-logo.svg", width="stretch")
except Exception:
    st.sidebar.markdown("🚀")

st.sidebar.markdown(
    '<p class="sidebar-title">DeployMate AI</p>', unsafe_allow_html=True
)
st.sidebar.caption("Your AI DevOps Co-Pilot")

st.sidebar.markdown("---")

if settings.github_token:
    st.sidebar.success("✅ GitHub Connected")
else:
    st.sidebar.warning("⚠️ GitHub token not set")

st.sidebar.markdown("### 💬 Chats")

if st.sidebar.button("➕ New Chat", width="stretch"):
    reset_chat()

st.sidebar.markdown("---")

for thread_id in st.session_state["chat_threads"][::-1]:
    thread_name = get_thread_name(thread_id)
    is_active = str(thread_id) == str(st.session_state["thread_id"])

    if st.sidebar.button(
        f"💬 {thread_name}", key=f"thread_{thread_id}", width="stretch"
    ):
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


# --------------------Page Tabs-------------------------------
tab_chat, tab_home, tab_about, tab_contact = st.tabs(
    ["💬 Chat", "🏠 Home", "ℹ️ About", "📞 Contact"]
)

with tab_home:
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 48px; margin-bottom: 20px;">
            <span style="background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;">🚀 DeployMate AI</span>
        </h1>
        <h2 style="color: #888; font-weight: 400;">Your AI DevOps Co-Pilot</h2>
        <p style="font-size: 18px; color: #aaa; margin-top: 20px;">
            Instantly solve Docker errors, get deployment guides, and review your code — 
            all in one place!
        </p>
        <div style="margin-top: 40px;">
            <h3 style="color: #6C63FF;">🔍 What can I help you with?</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-top: 20px;">
                <span style="background: rgba(108, 99, 255, 0.2); padding: 10px 20px; border-radius: 20px;">🐳 Docker Errors</span>
                <span style="background: rgba(0, 217, 255, 0.2); padding: 10px 20px; border-radius: 20px;">☁️ Deployment</span>
                <span style="background: rgba(255, 107, 107, 0.2); padding: 10px 20px; border-radius: 20px;">🔒 Code Review</span>
                <span style="background: rgba(39, 201, 63, 0.2); padding: 10px 20px; border-radius: 20px;">🐙 GitHub Integration</span>
                <span style="background: rgba(255, 189, 46, 0.2); padding: 10px 20px; border-radius: 20px;">🐳 Docker Config</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.info("💬 Go to the **Chat** tab to start asking questions!")


with tab_about:
    st.markdown("---")
    st.markdown(
        """
    <div style="padding: 20px;">
        <h1 style="color: #6C63FF;">ℹ️ About DeployMate AI</h1>
        
        <div style="background: rgba(108, 99, 255, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h3>🎯 Our Mission</h3>
            <p style="color: #aaa;">Making DevOps accessible to everyone with the power of AI.</p>
        </div>
        
        <div style="background: rgba(0, 217, 255, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h3>⚡ Features</h3>
            <ul style="color: #aaa; line-height: 2;">
                <li>🐳 <strong>Docker Error Analysis</strong> - Debug container issues instantly</li>
                <li>☁️ <strong>Deployment Guides</strong> - Deploy to Railway, Render, AWS, and more</li>
                <li>🔒 <strong>Code Review</strong> - Get security and performance suggestions</li>
                <li>🐙 <strong>GitHub Integration</strong> - Analyze PRs and repositories</li>
                <li>🐳 <strong>Docker Config Generator</strong> - Auto-generate Dockerfiles and docker-compose</li>
            </ul>
        </div>
        
        <div style="background: rgba(255, 107, 107, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h3>🛠️ Built With</h3>
            <p style="color: #aaa;">
                <strong>Backend:</strong> LangGraph, LangChain, PostgreSQL, Ollama (LLM)<br>
                <strong>Frontend:</strong> Streamlit<br>
                <strong>AI:</strong> Qwen2.5-Coder (Local LLM)
            </p>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #666;">
            <p>🚀 DeployMate AI © 2026 | Made with ❤️</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


with tab_contact:
    st.markdown("---")
    st.markdown(
        """
    <div style="padding: 20px; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #6C63FF;">📞 Contact Us</h1>
        
        <div style="background: rgba(108, 99, 255, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h3>💬 Get in Touch</h3>
            <p style="color: #aaa;">Have questions or feedback? We'd love to hear from you!</p>
            
            <div style="margin-top: 20px;">
                <p><strong>📧 Email:</strong> support@deploymate.ai</p>
                <p><strong>🐙 GitHub:</strong> github.com/Rohan171819/DeployMate-AI</p>
            </div>
        </div>
        
        <div style="background: rgba(0, 217, 255, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0;">
            <h3>🤝 Contribute</h3>
            <p style="color: #aaa;">DeployMate AI is open source! Contributions are welcome.</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


with tab_chat:
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, rgba(108, 99, 255, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(108, 99, 255, 0.2);">
        <h1 style="margin: 0; font-size: 28px;">
            <span style="background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;">🚀 DeployMate AI</span>
        </h1>
        <p style="margin: 8px 0 0 0; color: #888;">
            Your AI DevOps Co-Pilot — Ask me about Docker errors, deployment guides, or code reviews
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    for message in st.session_state["message_history"]:
        avatar = "👨‍💻" if message["role"] == "user" else "🚀"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

st.markdown(
    """
<div style="background: linear-gradient(135deg, rgba(108, 99, 255, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(108, 99, 255, 0.2);">
    <h1 style="margin: 0; font-size: 28px;">
        <span style="background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;">🚀 DeployMate AI</span>
    </h1>
    <p style="margin: 8px 0 0 0; color: #888;">
        Your AI DevOps Co-Pilot — Ask me about Docker errors, deployment guides, or code reviews
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

for message in st.session_state["message_history"]:
    avatar = "👨‍💻" if message["role"] == "user" else "🚀"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if st.session_state["message_history"]:
    last_msg = (
        st.session_state["message_history"][-1]["content"]
        if st.session_state["message_history"]
        else ""
    )
    st.session_state["context_suggestions"] = get_suggestions(last_msg)

if not st.session_state["message_history"]:
    st.markdown(
        """
    <div class="info-box">
        <h4 style="margin: 0 0 8px 0; color: #6C63FF;">💡 Quick Start</h4>
        <p style="margin: 0; color: #888; font-size: 14px;">
            Try one of these examples or ask me anything about Docker, deployment, or code reviews!
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    quick_starts = [
        "My Docker container keeps crashing with exit code 137",
        "How do I deploy my Node.js app to Railway?",
        "Review my code for security issues",
        "Explain this error: command not found",
    ]

    for i, suggestion in enumerate(quick_starts):
        col = cols[i % 2]
        with col:
            if st.button(f"💭 {suggestion}", key=f"quick_{i}"):
                st.session_state["message_history"].append(
                    {"role": "user", "content": suggestion}
                )
                st.rerun()

st.markdown("---")

if prompt := st.chat_input(
    "💭 Ask me about your Docker errors, deployment issues, or get code reviews..."
):
    st.session_state["message_history"].append({"role": "user", "content": prompt})

    current_thread = str(st.session_state["thread_id"])
    if st.session_state["thread_names"].get(current_thread) in [None, "Chat"]:
        set_thread_name(current_thread, prompt)

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🚀"):
        response_content = ""

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.markdown(
            """
        <div class="typing-container">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            <span style="color: #888; margin-left: 8px;">Thinking...</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        total_chunks = 0
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            response_content += message_chunk.content
            total_chunks += 1
            progress = min(total_chunks / 20, 1.0)
            progress_bar.progress(progress)

        progress_bar.empty()
        status_text.empty()

        st.markdown(response_content)

    state = chatbot.get_state(config=CONFIG)
    if state and state.values:
        artifacts = state.values.get("generated_artifacts")

    if artifacts and (artifacts.get("dockerfile") or artifacts.get("docker_compose")):
        st.session_state["last_artifacts"] = artifacts

        st.markdown("---")
        st.markdown("### 🐳 Generated Docker Config")

        col1, col2 = st.columns(2)
        with col1:
            target_dir = st.text_input("📁 Target directory:", value=os.getcwd())
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Apply Docker Config", type="primary"):
                if artifacts.get("dockerfile"):
                    dockerfile_path = os.path.join(target_dir, "Dockerfile")
                    with open(dockerfile_path, "w") as f:
                        f.write(artifacts["dockerfile"])
                    st.success(f"✅ Dockerfile saved to {dockerfile_path}")

                if artifacts.get("docker_compose"):
                    compose_path = os.path.join(target_dir, "docker-compose.yml")
                    with open(compose_path, "w") as f:
                        f.write(artifacts["docker_compose"])
                    st.success(f"✅ docker-compose.yml saved to {compose_path}")

    st.session_state["message_history"].append(
        {"role": "assistant", "content": response_content}
    )

    st.session_state["context_suggestions"] = get_suggestions(response_content)

if st.session_state.get("context_suggestions") and st.session_state["message_history"]:
    st.markdown("---")
    st.markdown("### 💡 Suggested Follow-ups")

    suggestion_cols = st.columns(2)
    for i, suggestion in enumerate(st.session_state["context_suggestions"][:4]):
        col = suggestion_cols[i % 2]
        with col:
            if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                st.session_state["message_history"].append(
                    {"role": "user", "content": suggestion}
                )
                st.rerun()
