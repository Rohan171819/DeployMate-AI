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
    .stApp { background-color: #0D0D0D; }
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
    .stChatInput input::placeholder { color: #888 !important; }
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
    .stMarkdown { color: #E8E8E8 !important; }
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
    section[data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        border: none !important;
        color: #E8E8E8 !important;
        text-align: left !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(108, 99, 255, 0.15) !important;
    }
    /* Active nav button */
    .nav-active button {
        background-color: rgba(108, 99, 255, 0.25) !important;
        border-left: 3px solid #6C63FF !important;
        color: #6C63FF !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #E8E8E8 !important; }
    p, span, label { color: #E8E8E8 !important; }
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
    hr { border-color: rgba(108, 99, 255, 0.2) !important; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1A1A2E; }
    ::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 4px; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown > div { animation: fadeIn 0.3s ease; }
    @keyframes typingDot {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    .typing-indicator span {
        display: inline-block;
        width: 8px; height: 8px;
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
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6C63FF 0%, #00D9FF 100%) !important;
    }
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


# ─── Helper functions ────────────────────────────────────────────────────────


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["thread_names"][str(thread_id)] = "Chat"
    st.session_state["context_suggestions"] = get_suggestions("")


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversations(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception:
        return []


def get_thread_name(thread_id):
    return st.session_state["thread_names"].get(str(thread_id), "Chat")


def set_thread_name(thread_id, first_message):
    name = first_message[:30] + "..." if len(first_message) > 30 else first_message
    st.session_state["thread_names"][str(thread_id)] = name


def get_suggestions(last_message):
    lm = last_message.lower()
    if "docker" in lm or "container" in lm:
        return [
            "Explain exit code 137",
            "How to debug OOM errors?",
            "Docker compose not starting",
            "Port already in use error",
        ]
    elif "deploy" in lm or "deployment" in lm:
        return [
            "Deploy to Railway",
            "Deploy to Render",
            "AWS ECS deployment",
            "Dockerize my Node.js app",
        ]
    elif "error" in lm or "fail" in lm:
        return [
            "Analyze this error log",
            "Why is my CI failing?",
            "Python import errors",
            "Connection refused",
        ]
    elif "code" in lm or "review" in lm:
        return [
            "Review my PR",
            "Security issues in code",
            "Performance optimization",
            "Best practices check",
        ]
    else:
        return [
            "My Docker container keeps crashing",
            "How to deploy to Railway?",
            "Review my code for errors",
            "Explain this error message",
        ]


# ─── Session state init ───────────────────────────────────────────────────────

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "💬 Chat"

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {}

if "context_suggestions" not in st.session_state:
    st.session_state["context_suggestions"] = get_suggestions("")

add_thread(st.session_state["thread_id"])
CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

st.sidebar.markdown(
    """
<style>
.sidebar-title {
    font-size: 22px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.nav-label {
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 12px 0 4px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

try:
    st.sidebar.image("deploymate-logo.svg", width=200)
except Exception:
    pass

st.sidebar.markdown(
    '<p class="sidebar-title">🚀 DeployMate AI</p>', unsafe_allow_html=True
)
st.sidebar.caption("Your AI DevOps Co-Pilot")
st.sidebar.markdown("---")

# GitHub status
if settings.github_token:
    st.sidebar.success("✅ GitHub Connected")
else:
    st.sidebar.warning("⚠️ GitHub token not set")

st.sidebar.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
st.sidebar.markdown('<p class="nav-label">Navigation</p>', unsafe_allow_html=True)

pages = ["💬 Chat", "🏠 Home", "ℹ️ About", "📞 Contact"]
for page in pages:
    is_active = st.session_state["current_page"] == page
    if is_active:
        st.sidebar.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.sidebar.button(page, key=f"nav_{page}", use_container_width=True):
        st.session_state["current_page"] = page
        st.rerun()
    if is_active:
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

# ── Chat history (only relevant on Chat page) ──────────────────────────────
st.sidebar.markdown('<p class="nav-label">💬 Chats</p>', unsafe_allow_html=True)

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.session_state["current_page"] = "💬 Chat"
    st.rerun()

st.sidebar.markdown("")

for thread_id in st.session_state["chat_threads"][::-1]:
    thread_name = get_thread_name(thread_id)
    if st.sidebar.button(
        f"💬 {thread_name}", key=f"thread_{thread_id}", use_container_width=True
    ):
        st.session_state["thread_id"] = thread_id
        st.session_state["current_page"] = "💬 Chat"
        messages = load_conversations(thread_id)
        temp_messages = []
        for message in messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": message.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE RENDERING  — based on current_page
# ═══════════════════════════════════════════════════════════════════════════

current_page = st.session_state["current_page"]

# ─── HOME PAGE ───────────────────────────────────────────────────────────────
if current_page == "🏠 Home":
    st.markdown(
        """
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="font-size: 48px; margin-bottom: 16px;">
            <span style="background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;">
                🚀 DeployMate AI
            </span>
        </h1>
        <h2 style="color: #888; font-weight: 400; font-size: 22px;">
            Your AI DevOps Co-Pilot
        </h2>
        <p style="font-size: 18px; color: #aaa; margin-top: 16px; max-width: 600px; margin: 16px auto 0;">
            Instantly solve Docker errors, get deployment guides, and review 
            your code — powered by local AI that never sends your code to the cloud.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    features = [
        (
            "🔍",
            "Error Analyzer",
            "Paste any Docker, Python, or CI/CD error. Get root cause + fix instantly.",
        ),
        (
            "🚀",
            "Deployment Guide",
            "Describe your app. Get tailored instructions for AWS, Railway, Render, VPS.",
        ),
        (
            "👨‍💻",
            "Code Review",
            "Paste code or a GitHub PR link. Get security & performance feedback.",
        ),
        (
            "🧠",
            "Long-Term Memory",
            "Remembers your tech stack and past errors across all sessions.",
        ),
        (
            "🔒",
            "100% Local & Private",
            "Powered by Ollama on your machine. Your code never leaves your computer.",
        ),
        (
            "📄",
            "PDF Knowledge Base",
            "Upload internal docs or runbooks. DeployMate learns from them.",
        ),
    ]

    for idx, (icon, title, desc) in enumerate(features):
        col = [col1, col2, col3][idx % 3]
        with col:
            st.markdown(
                f"""
            <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                        border-radius: 12px; padding: 20px; margin-bottom: 16px;
                        transition: all 0.3s;">
                <h3 style="margin: 0 0 8px 0;">{icon} {title}</h3>
                <p style="color: #aaa; margin: 0; font-size: 14px;">{desc}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; padding: 20px;">
        <p style="color: #555; font-size: 13px;">
            Built with &nbsp;
            <strong style="color:#6C63FF">LangGraph</strong> &nbsp;·&nbsp;
            <strong style="color:#00D9FF">LangChain</strong> &nbsp;·&nbsp;
            <strong style="color:#6C63FF">Ollama</strong> &nbsp;·&nbsp;
            <strong style="color:#00D9FF">PostgreSQL</strong> &nbsp;·&nbsp;
            <strong style="color:#6C63FF">FAISS</strong> &nbsp;·&nbsp;
            <strong style="color:#00D9FF">Docker</strong>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.info("💬 Click **💬 Chat** in the sidebar to start asking questions!")


# ─── ABOUT PAGE ──────────────────────────────────────────────────────────────
elif current_page == "ℹ️ About":
    st.markdown(
        """
    <div style="padding: 10px 0 30px 0;">
        <h1 style="color: #6C63FF;">ℹ️ About DeployMate AI</h1>
        <p style="color: #aaa; font-size: 16px;">
            Built by a developer, for developers — because DevOps shouldn't be a wall.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
        <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                    border-radius: 12px; padding: 24px; margin-bottom: 16px;">
            <h3>🎯 The Mission</h3>
            <p style="color: #aaa; line-height: 1.8;">
                As an MCA student building real AI systems, I constantly hit the same wall — DevOps.
                Docker containers crashing with cryptic exit codes. CI/CD pipelines failing on step 7 of 12.
                AWS deployment guides written for engineers with 5 years of experience.
            </p>
            <p style="color: #aaa; line-height: 1.8;">
                I didn't want another generic chatbot. I wanted an agent that knows MY stack,
                remembers what I've tried, and gives me the exact command to run — not a 10-page tutorial.
            </p>
            <p style="color: #aaa; line-height: 1.8;">
                DeployMate AI is that agent. Built on LangGraph, running 100% locally via Ollama,
                with long-term memory that learns your project over time.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style="background: rgba(0,217,255,0.08); border: 1px solid rgba(0,217,255,0.2);
                    border-radius: 12px; padding: 24px; margin-bottom: 16px;">
            <h3>⚡ Features</h3>
            <ul style="color: #aaa; line-height: 2.2; margin: 0; padding-left: 20px;">
                <li>🐳 <strong>Docker Error Analysis</strong> — Debug container issues instantly</li>
                <li>☁️ <strong>Deployment Guides</strong> — Railway, Render, AWS, VPS and more</li>
                <li>🔒 <strong>Code Review</strong> — Security and performance suggestions</li>
                <li>🐙 <strong>GitHub Integration</strong> — Analyze PRs and repositories</li>
                <li>🐳 <strong>Docker Generator</strong> — Auto-generate Dockerfiles and docker-compose</li>
                <li>🧠 <strong>Long-Term Memory</strong> — Remembers your stack across sessions</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style="background: rgba(255,107,107,0.08); border: 1px solid rgba(255,107,107,0.2);
                    border-radius: 12px; padding: 24px; margin-bottom: 16px; text-align: center;">
            <div style="width: 80px; height: 80px; border-radius: 50%;
                        background: linear-gradient(135deg, #6C63FF, #00D9FF);
                        display: flex; align-items: center; justify-content: center;
                        font-size: 36px; margin: 0 auto 16px;">R</div>
            <h3 style="margin: 0;">Rohan</h3>
            <p style="color: #aaa; font-size: 13px; margin: 4px 0;">AI/ML Engineer & MCA Student</p>
            <p style="color: #666; font-size: 12px;">GL Bajaj College, Greater Noida</p>
            <div style="margin-top: 16px; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center;">
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">LangGraph</span>
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">LangChain</span>
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">RAG</span>
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">Python</span>
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">Docker</span>
                <span style="background: rgba(108,99,255,0.2); padding: 3px 10px; border-radius: 12px; font-size: 11px;">FAISS</span>
            </div>
        </div>

        <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                    border-radius: 12px; padding: 24px;">
            <h3>🛠️ Built With</h3>
            <p style="color: #aaa; font-size: 14px; line-height: 2;">
                <strong style="color:#6C63FF">Brain:</strong> Ollama + LangGraph + LangChain<br>
                <strong style="color:#00D9FF">Memory:</strong> PostgreSQL + FAISS<br>
                <strong style="color:#6C63FF">DevOps:</strong> Docker + GitHub Actions<br>
                <strong style="color:#00D9FF">Interface:</strong> Streamlit + LangSmith<br>
                <strong style="color:#6C63FF">Tunnel:</strong> Cloudflare (public access)
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div style="text-align: center; margin-top: 30px; color: #444;">
        <p>🚀 DeployMate AI © 2026 | Made with ❤️ | Open Source</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─── CONTACT PAGE ────────────────────────────────────────────────────────────
elif current_page == "📞 Contact":
    st.markdown(
        """
    <div style="padding: 10px 0 30px 0;">
        <h1 style="color: #6C63FF;">📞 Let's Build Something Together</h1>
        <p style="color: #aaa; font-size: 16px;">
            Questions, collaborations, bug reports, or job opportunities — I'd love to hear from you.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
        <div style="background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.2);
                    border-radius: 12px; padding: 24px; margin-bottom: 16px;">
            <h3>💬 Get in Touch</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        contact_name = st.text_input("Full Name", placeholder="Your name")
        contact_email = st.text_input("Email Address", placeholder="you@example.com")
        contact_subject = st.selectbox(
            "Subject",
            [
                "General Inquiry",
                "Bug Report",
                "Feature Request",
                "Collaboration",
                "Job Opportunity",
                "Other",
            ],
        )
        contact_message = st.text_area(
            "Message", placeholder="Describe your question or idea...", height=150
        )

        if st.button("Send Message →", type="primary", use_container_width=True):
            if contact_name and contact_email and contact_message:
                st.success("✅ Message sent! I'll get back to you within 24 hours.")
                st.balloons()
            else:
                st.error("⚠️ Please fill in all fields before sending.")

    with col2:
        st.markdown(
            """
        <div style="background: rgba(0,217,255,0.08); border: 1px solid rgba(0,217,255,0.2);
                    border-radius: 12px; padding: 24px; margin-bottom: 16px;">
            <h3>🔗 Other Ways to Reach Me</h3>
            <div style="margin-top: 16px;">
                <p style="margin: 12px 0;">
                    <strong>📧 Email</strong><br>
                    <span style="color: #aaa; font-size: 14px;">support@deploymate.ai</span>
                </p>
                <p style="margin: 12px 0;">
                    <strong>🐙 GitHub</strong><br>
                    <span style="color: #aaa; font-size: 14px;">github.com/Rohan171819/DeployMate-AI</span>
                </p>
                <p style="margin: 12px 0;">
                    <strong>📍 Location</strong><br>
                    <span style="color: #aaa; font-size: 14px;">Greater Noida, UP, India 🇮🇳</span>
                </p>
            </div>
        </div>

        <div style="background: rgba(39,201,63,0.08); border: 1px solid rgba(39,201,63,0.2);
                    border-radius: 12px; padding: 24px;">
            <h3>🟢 Available For</h3>
            <p style="color: #27C93F; font-size: 14px; line-height: 2.2; margin: 0;">
                ✅ Freelance AI/ML Projects<br>
                ✅ Open Source Collaboration<br>
                ✅ Junior AI Engineer Roles<br>
                ⚡ Typical response: &lt; 24 hours
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ─── CHAT PAGE ───────────────────────────────────────────────────────────────
else:  # "💬 Chat"
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, rgba(108,99,255,0.1) 0%, rgba(0,217,255,0.1) 100%);
                padding: 20px; border-radius: 12px; margin-bottom: 20px;
                border: 1px solid rgba(108,99,255,0.2);">
        <h1 style="margin: 0; font-size: 28px;">
            <span style="background: linear-gradient(135deg, #6C63FF 0%, #00D9FF 100%);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;">
                🚀 DeployMate AI
            </span>
        </h1>
        <p style="margin: 8px 0 0 0; color: #888;">
            Your AI DevOps Co-Pilot — Ask me about Docker errors, deployment guides, or code reviews
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Render message history
    for message in st.session_state["message_history"]:
        avatar = "👨‍💻" if message["role"] == "user" else "🚀"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Quick start buttons (empty state)
    if not st.session_state["message_history"]:
        st.markdown(
            """
        <div class="info-box">
            <h4 style="margin: 0 0 8px 0; color: #6C63FF;">💡 Quick Start</h4>
            <p style="margin: 0; color: #888; font-size: 14px;">
                Try one of these examples or ask anything about Docker, deployment, or code reviews!
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        quick_starts = [
            "My Docker container keeps crashing with exit code 137",
            "How do I deploy my Node.js app to Railway?",
            "Review my code for security issues",
            "Explain this error: command not found",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(quick_starts):
            with cols[i % 2]:
                if st.button(
                    f"💭 {suggestion}", key=f"quick_{i}", use_container_width=True
                ):
                    st.session_state["message_history"].append(
                        {"role": "user", "content": suggestion}
                    )
                    st.rerun()

    # Suggested follow-ups
    if (
        st.session_state.get("context_suggestions")
        and st.session_state["message_history"]
    ):
        st.markdown("---")
        st.markdown("### 💡 Suggested Follow-ups")
        suggestion_cols = st.columns(2)
        for i, suggestion in enumerate(st.session_state["context_suggestions"][:4]):
            with suggestion_cols[i % 2]:
                if st.button(
                    f"💭 {suggestion}", key=f"suggestion_{i}", use_container_width=True
                ):
                    st.session_state["message_history"].append(
                        {"role": "user", "content": suggestion}
                    )
                    st.rerun()

    st.markdown("---")

    # ── chat_input at TOP LEVEL (not inside tabs) ──────────────────────────
    if prompt := st.chat_input(
        "💭 Ask me about Docker errors, deployment issues, or code reviews..."
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
                progress_bar.progress(min(total_chunks / 20, 1.0))

            progress_bar.empty()
            status_text.empty()
            st.markdown(response_content)

        # Docker artifacts
        artifacts = None
        try:
            state = chatbot.get_state(config=CONFIG)
            if state and state.values:
                artifacts = state.values.get("generated_artifacts")
        except Exception:
            pass

        if artifacts and (
            artifacts.get("dockerfile") or artifacts.get("docker_compose")
        ):
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
                        path = os.path.join(target_dir, "Dockerfile")
                        with open(path, "w") as f:
                            f.write(artifacts["dockerfile"])
                        st.success(f"✅ Dockerfile saved to {path}")
                    if artifacts.get("docker_compose"):
                        path = os.path.join(target_dir, "docker-compose.yml")
                        with open(path, "w") as f:
                            f.write(artifacts["docker_compose"])
                        st.success(f"✅ docker-compose.yml saved to {path}")

        st.session_state["message_history"].append(
            {"role": "assistant", "content": response_content}
        )
        st.session_state["context_suggestions"] = get_suggestions(response_content)
        st.rerun()
