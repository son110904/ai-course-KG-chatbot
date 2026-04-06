"""
NEU Knowledge Graph Chatbot — Streamlit UI
Giao diện chat phong cách ChatGPT, tích hợp script3a.py
"""

import streamlit as st
import uuid
import datetime
import sys
import os

# ── Page config (phải là lệnh Streamlit đầu tiên) ────────────────────────────
st.set_page_config(
    page_title="NEU AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tùy chỉnh — giao diện tối kiểu ChatGPT ───────────────────────────────
st.markdown("""
<style>
/* ── Reset & font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Ẩn các phần mặc định của Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

/* ── Nền tổng thể ── */
.stApp {
    background-color: #212121;
    color: #ececec;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown p {
    color: #ececec !important;
}

/* ── Logo NEU ── */
.neu-logo-container {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 20px 16px 8px 16px;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 12px;
}
.neu-logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 16px; color: white;
    flex-shrink: 0;
}
.neu-logo-text { line-height: 1.2; }
.neu-logo-text .title { font-size: 15px; font-weight: 700; color: #fff; }
.neu-logo-text .subtitle { font-size: 11px; color: #8e8ea0; }

/* ── Nút New Chat ── */
.new-chat-btn button {
    background: #2a2a2a !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: background 0.2s !important;
}
.new-chat-btn button:hover {
    background: #333 !important;
}

/* ── Tiêu đề History ── */
.history-label {
    font-size: 11px;
    font-weight: 600;
    color: #8e8ea0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 12px 16px 4px;
}

/* ── Item lịch sử ── */
.history-item {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: #c5c5d2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background 0.15s;
    margin: 1px 0;
}
.history-item:hover { background: #2a2a2a; color: #fff; }
.history-item.active { background: #2a2a2a; color: #fff; }

/* ── Main chat area ── */
.main-chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 820px;
    margin: 0 auto;
    padding: 0 20px;
}

/* ── Welcome screen ── */
.welcome-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px 40px;
    text-align: center;
}
.welcome-icon {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px; margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(26,86,219,0.3);
}
.welcome-title {
    font-size: 32px; font-weight: 700; color: #fff;
    margin-bottom: 8px;
}
.welcome-sub {
    font-size: 15px; color: #8e8ea0; margin-bottom: 40px;
}

/* ── Suggested questions ── */
.suggest-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    width: 100%;
    max-width: 680px;
    margin-bottom: 24px;
}
.suggest-card {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
}
.suggest-card:hover {
    background: #313131;
    border-color: #4a4a4a;
    transform: translateY(-1px);
}
.suggest-card .icon { font-size: 18px; margin-bottom: 6px; }
.suggest-card .text { font-size: 13px; color: #c5c5d2; line-height: 1.4; }

/* ── Messages ── */
.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px 0;
    scrollbar-width: thin;
    scrollbar-color: #3a3a3a transparent;
}

.message-row {
    display: flex;
    gap: 14px;
    padding: 12px 0;
    align-items: flex-start;
}
.message-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700;
    flex-shrink: 0;
}
.avatar.user {
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    color: white;
}
.avatar.bot {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
}

.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.65;
    word-break: break-word;
}
.bubble.user {
    background: #1a56db;
    color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: #2a2a2a;
    color: #ececec;
    border-bottom-left-radius: 4px;
    border: 1px solid #3a3a3a;
}

.msg-time {
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
    text-align: right;
}

/* ── Typing indicator ── */
.typing-indicator {
    display: flex; gap: 5px; align-items: center;
    padding: 14px 16px;
}
.typing-dot {
    width: 8px; height: 8px;
    background: #8e8ea0;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;
}
.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
}

/* ── Input area ── */
.input-wrapper {
    padding: 16px 0 24px;
    background: transparent;
}
.stChatInput > div {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 14px !important;
}
.stChatInput textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 15px !important;
}
.stChatInput textarea::placeholder { color: #6b7280 !important; }
.stChatInput button {
    background: #1a56db !important;
    border-radius: 10px !important;
    color: white !important;
}
.stChatInput button:hover { background: #1447b8 !important; }

/* ── Stats pills trong sidebar ── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 12px;
    color: #c5c5d2;
    margin: 3px;
}
.stat-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}

/* ── Neo4j status dot ── */
.status-row {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 16px; font-size: 13px; color: #8e8ea0;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.online { background: #10b981; box-shadow: 0 0 6px #10b981; }
.status-dot.offline { background: #ef4444; }
.status-dot.checking { background: #f59e0b; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
</style>
""", unsafe_allow_html=True)

# ── Khởi tạo session state ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = str(uuid.uuid4())
if "driver" not in st.session_state:
    st.session_state.driver = None
if "ai_client" not in st.session_state:
    st.session_state.ai_client = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "db_status" not in st.session_state:
    st.session_state.db_status = "checking"  # checking | online | offline
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Load script3a ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_backend():
    """Import script3a và khởi tạo driver + AI client."""
    try:
        # Thêm thư mục hiện tại vào path nếu cần
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        import script3a as s3
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()

        ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        driver = s3.get_driver()
        # Kiểm tra kết nối
        with driver.session() as sess:
            sess.run("RETURN 1").single()
        # Khởi tạo communities (bỏ qua nếu đã tồn tại)
        s3.initialize_communities(driver, force_rebuild=False)
        return s3, ai_client, driver, "online"
    except Exception as e:
        return None, None, None, f"offline:{e}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="neu-logo-container">
        <div class="neu-logo-icon">N</div>
        <div class="neu-logo-text">
            <div class="title">NEU Assistant</div>
            <div class="subtitle">Knowledge Graph Chatbot</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✏️  Cuộc trò chuyện mới", use_container_width=True):
        # Lưu cuộc trò chuyện hiện tại nếu có
        if st.session_state.messages:
            first_q = next(
                (m["content"][:45] for m in st.session_state.messages if m["role"] == "user"),
                "Cuộc trò chuyện"
            )
            st.session_state.conversations.insert(0, {
                "id": st.session_state.current_conv_id,
                "title": first_q + ("..." if len(first_q) == 45 else ""),
                "messages": st.session_state.messages.copy(),
                "ts": datetime.datetime.now(),
            })
        st.session_state.messages = []
        st.session_state.current_conv_id = str(uuid.uuid4())
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # DB Status
    s3_mod, ai_cl, drv, status_str = load_backend()
    if status_str == "online":
        st.session_state.db_status = "online"
        st.session_state.driver = drv
        st.session_state.ai_client = ai_cl
        st.session_state.initialized = True
        db_dot = "online"
        db_label = "Neo4j — Đã kết nối"
    else:
        st.session_state.db_status = "offline"
        db_dot = "offline"
        db_label = "Neo4j — Mất kết nối"

    st.markdown(f"""
    <div class="status-row">
        <div class="status-dot {db_dot}"></div>
        <span>{db_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown('<div style="padding:8px 16px 4px">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:4px;padding:0 0 12px 0">
        <span class="stat-pill"><span class="dot" style="background:#6366f1"></span>37 Ngành</span>
        <span class="stat-pill"><span class="dot" style="background:#0ea5e9"></span>802 Môn học</span>
        <span class="stat-pill"><span class="dot" style="background:#10b981"></span>27 Nghề</span>
        <span class="stat-pill"><span class="dot" style="background:#f59e0b"></span>5217 Kỹ năng</span>
        <span class="stat-pill"><span class="dot" style="background:#ef4444"></span>695 Giảng viên</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lịch sử cuộc trò chuyện
    if st.session_state.conversations:
        st.markdown('<div class="history-label">Lịch sử</div>', unsafe_allow_html=True)
        for conv in st.session_state.conversations[:20]:
            is_active = conv["id"] == st.session_state.current_conv_id
            cls = "history-item active" if is_active else "history-item"
            if st.button(
                f"💬 {conv['title']}",
                key=f"conv_{conv['id']}",
                use_container_width=True,
            ):
                st.session_state.messages = conv["messages"].copy()
                st.session_state.current_conv_id = conv["id"]
                st.rerun()



# ── Main area ─────────────────────────────────────────────────────────────────
SUGGESTED = [
    {"icon": "🎓", "text": "Ngành Kinh tế học có những môn học nào?"},
    {"icon": "💼", "text": "Nghề Data Analyst yêu cầu những kỹ năng gì?"},
    {"icon": "👨‍🏫", "text": "Giảng viên nào dạy môn Kinh tế lượng?"},
    {"icon": "🔗", "text": "Môn nào được dạy trong tất cả các ngành?"},
]

# ── Xử lý suggested question click ──────────────────────────────────────────
if st.session_state.pending_question:
    user_msg = st.session_state.pending_question
    st.session_state.pending_question = None
    # Thêm vào messages và xử lý
    st.session_state.messages.append({
        "role": "user",
        "content": user_msg,
        "ts": datetime.datetime.now().strftime("%H:%M"),
    })

    if st.session_state.initialized and s3_mod is not None:
        qid = "q" + uuid.uuid4().hex[:6]
        with st.spinner(""):
            try:
                result = s3_mod.ask(
                    st.session_state.driver,
                    st.session_state.ai_client,
                    user_msg,
                    query_id=qid,
                )
                answer = result.get("generated_answer", "Xin lỗi, không tìm được câu trả lời.")
            except Exception as e:
                answer = f"⚠️ Đã xảy ra lỗi: {str(e)}"
    else:
        answer = "⚠️ Chưa kết nối được với cơ sở dữ liệu. Vui lòng kiểm tra file `.env` và thử lại."

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "ts": datetime.datetime.now().strftime("%H:%M"),
    })
    st.rerun()

# ── Welcome screen (không có tin nhắn) ───────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-screen">
        <div class="welcome-icon">🎓</div>
        <div class="welcome-title">Xin chào, tôi là NEU AI</div>
        <div class="welcome-sub">
            Trợ lý thông minh về chương trình đào tạo, môn học,<br>
            giảng viên và định hướng nghề nghiệp tại NEU.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions dạng 2 cột
    col1, col2 = st.columns(2)
    for i, s in enumerate(SUGGESTED):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(
                f"{s['icon']} {s['text']}",
                key=f"suggest_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_question = s["text"]
                st.rerun()

else:
    # ── Hiển thị lịch sử tin nhắn ────────────────────────────────────────────
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("ts", "")

        if role == "user":
            st.markdown(f"""
            <div class="message-row user">
                <div class="avatar user">T</div>
                <div>
                    <div class="bubble user">{content}</div>
                    <div class="msg-time">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Dùng st.markdown cho nội dung bot để render markdown đúng
            with st.container():
                col_av, col_msg = st.columns([0.06, 0.94])
                with col_av:
                    st.markdown("""
                    <div class="avatar bot" style="margin-top:8px">N</div>
                    """, unsafe_allow_html=True)
                with col_msg:
                    st.markdown(f"""
                    <div style="background:#2a2a2a;border:1px solid #3a3a3a;
                                border-radius:16px;border-bottom-left-radius:4px;
                                padding:14px 18px;color:#ececec;font-size:14.5px;
                                line-height:1.7;margin-top:4px">
                    {content}
                    </div>
                    <div class="msg-time" style="text-align:left;padding-left:4px">{ts}</div>
                    """, unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Hỏi về ngành học, môn học, giảng viên, nghề nghiệp...",
    key="chat_input",
)

st.markdown("""
<div style="text-align:center;font-size:11px;color:#4a4a5a;padding:4px 0 2px;
            white-space:nowrap">
    © 2025 Trường Đại học Kinh tế Quốc dân &nbsp;·&nbsp;
    <span style="color:#3a3a4a">NEU Knowledge Graph v9</span>
</div>
""", unsafe_allow_html=True)

if user_input and user_input.strip():
    question = user_input.strip()

    # Thêm tin nhắn người dùng
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "ts": datetime.datetime.now().strftime("%H:%M"),
    })

    # Gọi backend
    if st.session_state.initialized and s3_mod is not None:
        qid = "q" + uuid.uuid4().hex[:6]
        with st.spinner("NEU AI đang suy nghĩ..."):
            try:
                result = s3_mod.ask(
                    st.session_state.driver,
                    st.session_state.ai_client,
                    question,
                    query_id=qid,
                )
                answer = result.get("generated_answer", "Xin lỗi, không tìm được câu trả lời.")
            except Exception as e:
                answer = f"⚠️ Đã xảy ra lỗi: {str(e)}"
    else:
        answer = (
            "⚠️ Chưa kết nối được với cơ sở dữ liệu.\n\n"
            "Vui lòng kiểm tra file `.env` (DB_URL, DB_USER, DB_PASSWORD, OPENAI_API_KEY) "
            "và đảm bảo Neo4j đang chạy."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "ts": datetime.datetime.now().strftime("%H:%M"),
    })
    st.rerun()