"""
Streamlit前端 - DeepSeek风格
智能问答 + 知识库管理（含删除功能）
"""
import streamlit as st
import requests
import time
import pandas as pd

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="RAG知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DeepSeek风格CSS ====================
st.markdown("""
<style>
    #MainMenu, footer, .stDeployButton {display: none !important;}
    header {visibility: hidden !important;}

    * {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}

    .chat-scroll {
        height: calc(100vh - 280px);
        overflow-y: auto;
        padding: 0 24px;
        scroll-behavior: smooth;
    }
    .chat-scroll::-webkit-scrollbar {width: 6px;}
    .chat-scroll::-webkit-scrollbar-thumb {background: #e0e0e0; border-radius: 3px;}
    .chat-scroll::-webkit-scrollbar-track {background: transparent;}

    .welcome-box {
        text-align: center;
        padding: 80px 20px 40px;
    }
    .welcome-box .icon {font-size: 48px; margin-bottom: 16px;}
    .welcome-box .title {font-size: 24px; font-weight: 600; color: #1a1a1a; margin-bottom: 8px;}
    .welcome-box .subtitle {font-size: 14px; color: #999;}

    .user-row {
        display: flex;
        justify-content: flex-end;
        margin: 16px 0;
        animation: fadeIn 0.3s ease;
    }
    .user-bubble {
        background: #f0f4ff;
        color: #1a1a1a;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        max-width: 75%;
        font-size: 15px;
        line-height: 1.6;
        word-break: break-word;
        white-space: pre-wrap;
    }

    .assistant-row {
        display: flex;
        justify-content: flex-start;
        margin: 16px 0;
        animation: fadeIn 0.3s ease;
    }
    .assistant-bubble {
        background: #ffffff;
        color: #1a1a1a;
        padding: 12px 18px;
        border-radius: 16px 16px 16px 4px;
        max-width: 85%;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid #eee;
        word-break: break-word;
        white-space: pre-wrap;
    }
    .assistant-meta {
        font-size: 11px;
        color: #bbb;
        margin-top: 6px;
        padding-left: 4px;
    }

    .toolbar-line {
        border-top: 1px solid #eee;
        margin: 0 24px;
        padding: 10px 0;
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    .input-box {
        padding: 0 24px 16px;
    }

    .clear-btn {
        text-align: center;
        padding: 8px;
    }

    [data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #eee;
    }

    @keyframes fadeIn {from {opacity:0;transform:translateY(8px);} to {opacity:1;transform:translateY(0);}}

    .kb-container {max-width: 900px; margin: 0 auto; padding: 20px;}

    /* 删除按钮样式 */
    .delete-btn button {
        background: none !important;
        border: 1px solid #ffcccc !important;
        color: #cc0000 !important;
        font-size: 12px !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 常量 ====================
API = "http://localhost:5000/api"

# ==================== 会话状态 ====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = {}
if 'page' not in st.session_state:
    st.session_state.page = 'qa'
if 'msgs' not in st.session_state:
    st.session_state.msgs = []
if 'cookies' not in st.session_state:
    st.session_state.cookies = {}
if 'web_on' not in st.session_state:
    st.session_state.web_on = False
if 'kb_refresh' not in st.session_state:
    st.session_state.kb_refresh = 0

# ==================== API（带重试和连接池）====================
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=5,
    max_retries=2,
    pool_block=False
)
session.mount('http://', adapter)


def api(method, endpoint, data=None, files=None, timeout=300):
    """统一API调用"""
    url = f"{API}{endpoint}"

    for attempt in range(3):
        try:
            if method == 'GET':
                r = session.get(url, cookies=st.session_state.cookies, timeout=timeout)
            elif method == 'DELETE':
                r = session.delete(url, cookies=st.session_state.cookies, timeout=timeout)
            elif files:
                r = session.post(url, files=files, data=data or {}, cookies=st.session_state.cookies, timeout=timeout)
            else:
                r = session.post(url, json=data or {}, cookies=st.session_state.cookies, timeout=timeout)

            if r.cookies:
                st.session_state.cookies.update(r.cookies.get_dict())

            if r.status_code == 401:
                st.session_state.logged_in = False
                st.rerun()

            if r.ok:
                try:
                    return r.json()
                except:
                    return {'error': '无效的响应格式'}
            else:
                try:
                    return r.json()
                except:
                    return {'error': f'HTTP {r.status_code}'}

        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(2)
                continue
            return {'error': '连接失败，请确认后端服务已启动'}
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2)
                continue
            return {'error': '请求超时'}
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
                continue
            return {'error': f'网络错误: {str(e)}'}

    return {'error': '多次重试后仍失败'}


# ==================== 登录页 ====================
def login_page():
    st.title("📚 RAG知识库问答系统")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login"):
            u = st.text_input("用户名")
            p = st.text_input("密码", type="password")
            if st.form_submit_button("登录", width="stretch"):
                r = api('POST', '/login', {'username': u, 'password': p})
                if 'error' not in r:
                    st.session_state.logged_in = True
                    st.session_state.user = r['user']
                    st.rerun()
                else:
                    st.error(r['error'])
        st.caption("默认: admin / 123456")


# ==================== 智能问答页 ====================
def qa_page():
    # 对话滚动区
    st.markdown('<div class="chat-scroll" id="chat-scroll">', unsafe_allow_html=True)

    if not st.session_state.msgs:
        st.markdown("""
        <div class="welcome-box">
            <div class="icon">📚</div>
            <div class="title">RAG知识库问答</div>
            <div class="subtitle">基于本地知识库 + 联网搜索，提供精准回答</div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.msgs:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div></div>',
                        unsafe_allow_html=True)
        else:
            meta = msg.get('meta', '')
            meta_html = f'<div class="assistant-meta">{meta}</div>' if meta else ''
            st.markdown(
                f'<div class="assistant-row"><div class="assistant-bubble">{msg["content"]}{meta_html}</div></div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 自动滚动
    st.markdown("""
    <script>
        var el = window.parent.document.querySelector('.chat-scroll');
        if(el) el.scrollTop = el.scrollHeight;
    </script>
    """, unsafe_allow_html=True)

    # 分割线 + 联网开关
    st.markdown('<div class="toolbar-line">', unsafe_allow_html=True)
    c1, c2 = st.columns([8, 2])
    with c1:
        pass
    with c2:
        st.session_state.web_on = st.toggle(
            "🌐 联网搜索",
            st.session_state.web_on,
            help="开启后，本地知识不足时自动搜索网络"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # 输入框
    st.markdown('<div class="input-box">', unsafe_allow_html=True)

    q = st.chat_input("输入您的问题，按回车发送...")

    if q and q.strip():
        st.session_state.msgs.append({'role': 'user', 'content': q.strip()})

        status_placeholder = st.empty()
        status_placeholder.info("⏳ 正在处理，请稍候...")

        r = api('POST', '/qa/ask', {
            'question': q.strip(),
            'enable_web': st.session_state.web_on
        }, timeout=300)

        status_placeholder.empty()

        if r and 'error' not in r:
            answer = r.get('answer', '')
            parts = []
            if r.get('response_time'):
                parts.append(f"⏱️ {r['response_time']}")
            if r.get('web_used'):
                parts.append("🌐 已联网")
            if not r.get('local_sufficient', True):
                parts.append("📚 本地不足")
            meta = " · ".join(parts) if parts else ""

            st.session_state.msgs.append({
                'role': 'assistant',
                'content': answer,
                'meta': meta
            })
        else:
            err = r.get('error', '未知错误') if r else '无法连接'
            st.session_state.msgs.append({
                'role': 'assistant',
                'content': f'❌ {err}',
                'meta': '处理失败'
            })

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # 清空按钮
    if st.session_state.msgs:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ 清空对话", width="content"):
            st.session_state.msgs = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ==================== 知识库管理页 ====================
def kb_page():
    st.markdown('<div class="kb-container">', unsafe_allow_html=True)
    st.title("📁 知识库管理")

    # ===== 上传区域 =====
    st.subheader("📤 上传文档")
    with st.form("upload", clear_on_submit=True):
        f = st.file_uploader(
            "选择文档（PDF/Word/Markdown/TXT）",
            type=['pdf', 'docx', 'doc', 'md', 'txt'],
            label_visibility="collapsed"
        )
        if st.form_submit_button("🚀 上传到知识库", width="stretch", type="primary") and f:
            with st.spinner("处理中...（可能需要1-2分钟）"):
                r = api('POST', '/knowledge/upload',
                        files={'file': (f.name, f.getvalue(), f.type)}, timeout=300)
            if 'error' in r:
                st.error(r['error'])
            else:
                st.success(f"✅ {r.get('file_name', '')} - {r.get('chunk_count', 0)}个文本块")
                st.session_state.kb_refresh += 1
                st.rerun()

    st.divider()

    # ===== 文档列表（含删除按钮）=====
    st.subheader("📚 已上传文档")

    kb = api('GET', '/knowledge/list')

    if isinstance(kb, list) and kb:
        for i, doc in enumerate(kb):
            doc_id = doc.get('id', i)
            doc_name = doc.get('name', '未知')
            doc_type = doc.get('file_type', '?')
            doc_chunks = doc.get('chunk_count', 0)
            doc_time = doc.get('created_at', '')

            col1, col2, col3, col4 = st.columns([4, 1, 2, 1])
            with col1:
                st.markdown(f"**{doc_name}**")
            with col2:
                st.caption(f"{doc_type}")
            with col3:
                st.caption(f"{doc_chunks}块 · {doc_time[:10] if doc_time else ''}")
            with col4:
                # 删除按钮 - 使用唯一key
                if st.button("🗑️", key=f"del_{doc_id}", help=f"删除 {doc_name}"):
                    # 确认对话框
                    st.session_state[f'confirm_delete_{doc_id}'] = True

            # 确认删除
            if st.session_state.get(f'confirm_delete_{doc_id}'):
                st.warning(f"⚠️ 确认删除 **{doc_name}**？此操作不可撤销。")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ 确认删除", key=f"confirm_{doc_id}", width="stretch"):
                        r = api('DELETE', f'/knowledge/delete/{doc_id}', timeout=30)
                        if r and 'error' not in r:
                            st.success(f"✅ {r.get('message', '删除成功')}")
                            st.session_state.pop(f'confirm_delete_{doc_id}')
                            st.session_state.kb_refresh += 1
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(r.get('error', '删除失败') if r else '连接失败')
                with c2:
                    if st.button("❌ 取消", key=f"cancel_{doc_id}", width="stretch"):
                        st.session_state.pop(f'confirm_delete_{doc_id}')
                        st.rerun()

            st.divider()
    else:
        st.info("📭 知识库为空，请上传文档")

    # ===== 统计 =====
    stats = api('GET', '/knowledge/stats')
    if stats:
        st.metric("📄 向量总数", stats.get('total_documents', 0))

    st.markdown('</div>', unsafe_allow_html=True)


# ==================== 侧边栏 ====================
def sidebar():
    with st.sidebar:
        st.markdown("### 📚 RAG系统")

        if st.session_state.logged_in:
            u = st.session_state.user
            st.markdown(f"👤 **{u.get('username', '?')}**")
            st.caption(f"角色: {u.get('role', '?')}")
            st.divider()

            pg = st.radio(
                "导航",
                ["💬 智能问答", "📁 知识库管理"],
                index=0 if st.session_state.page == 'qa' else 1,
                label_visibility="collapsed"
            )
            st.session_state.page = 'qa' if '问答' in pg else 'kb'

            st.divider()
            if st.button("🚪 退出登录", width="stretch"):
                api('POST', '/logout')
                st.session_state.logged_in = False
                st.session_state.user = {}
                st.session_state.msgs = []
                st.rerun()
        else:
            st.session_state.page = 'login'


# ==================== 主函数 ====================
def main():
    sidebar()

    if not st.session_state.logged_in:
        login_page()
    elif st.session_state.page == 'qa':
        qa_page()
    elif st.session_state.page == 'kb':
        kb_page()
    else:
        qa_page()


if __name__ == "__main__":
    main()