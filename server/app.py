"""
Flask主应用
RAG企业内部知识库问答Agent系统
"""
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
import os
import time
import gc
import json
import threading
from datetime import datetime

from config import Config
from auth import login_user, register_user, logout_user, login_required, admin_required, db_manager
from document_processor import DocumentProcessor
from text_cleaner import TextCleaner
from status_manager import status_manager

# ==================== 创建Flask应用 ====================

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
CORS(app)

# ==================== 初始化核心组件（不加载模型）====================

document_processor = DocumentProcessor()
text_cleaner = TextCleaner(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)

hybrid_rag = None
hybrid_rag_lock = threading.Lock()


def get_hybrid_rag():
    """获取HybridRAG实例（线程安全，延迟初始化）"""
    global hybrid_rag
    if hybrid_rag is None:
        with hybrid_rag_lock:
            if hybrid_rag is None:
                from hybrid_rag import HybridRAG
                print("🔄 首次初始化HybridRAG系统...")
                hybrid_rag = HybridRAG()
                print("✅ HybridRAG系统就绪")
    return hybrid_rag


# ==================== 健康检查接口 ====================

@app.route('/api/health', methods=['GET'])
def api_health():
    """系统健康检查"""
    return jsonify({
        'status': 'ok',
        'message': 'RAG知识库系统运行正常',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'llm_model': Config.OLLAMA_LLM_MODEL,
            'embedding_model': Config.OLLAMA_EMBEDDING_MODEL,
            'chunk_size': Config.CHUNK_SIZE,
            'rag_top_k': Config.RAG_TOP_K,
            'score_threshold': Config.SCORE_THRESHOLD
        }
    })


# ==================== 用户认证接口 ====================

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录"""
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400

    if login_user(username, password):
        status_manager.update_status('idle', f'用户 {username} 已登录', 0)
        return jsonify({
            'message': '登录成功',
            'user': {
                'id': session['user_id'],
                'username': session['username'],
                'role': session['role']
            }
        })
    else:
        return jsonify({'error': '用户名或密码错误'}), 401


@app.route('/api/register', methods=['POST'])
@admin_required
def api_register():
    """注册新用户（仅管理员）"""
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'user')

    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400

    if len(password) < 6:
        return jsonify({'error': '密码长度至少6位'}), 400

    if register_user(username, password, role):
        return jsonify({'message': f'用户 {username} 创建成功'})
    else:
        return jsonify({'error': '用户名已存在'}), 400


@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """用户登出"""
    username = session.get('username', '未知')
    logout_user()
    return jsonify({'message': f'用户 {username} 已退出登录'})


@app.route('/api/user/info', methods=['GET'])
@login_required
def api_user_info():
    """获取当前用户信息"""
    return jsonify({
        'id': session['user_id'],
        'username': session['username'],
        'role': session['role']
    })


# ==================== 用户管理接口（管理员）====================

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def api_get_users():
    """获取所有用户列表"""
    users = db_manager.get_all_users()
    return jsonify(users)


# ==================== 知识库管理接口 ====================

@app.route('/api/knowledge/upload', methods=['POST'])
@login_required
def api_upload_document():
    """上传文档到知识库"""
    if 'file' not in request.files:
        return jsonify({'error': '请选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    if file_ext not in allowed:
        return jsonify({'error': f'不支持的文件类型: {file_ext}'}), 400

    try:
        status_manager.update_status('processing', f'正在处理文档: {file.filename}', 10)

        file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        status_manager.update_status('processing', '正在解析文档...', 30)
        raw_documents = document_processor.process_file(file_path)

        status_manager.update_status('processing', '正在清洗和分块...', 60)
        cleaned_documents = text_cleaner.clean_documents(raw_documents)
        chunks = text_cleaner.smart_chunk(cleaned_documents)

        status_manager.update_status('processing', '正在生成向量嵌入...', 80)
        rag = get_hybrid_rag()
        added_count = rag.process_document(chunks)

        description = request.form.get('description', '')
        kb_id = db_manager.add_knowledge_base(
            name=file.filename,
            description=description,
            file_path=file_path,
            file_type=file_ext,
            chunk_count=added_count,
            created_by=session['user_id']
        )

        db_manager.add_log(
            user_id=session['user_id'],
            action='上传文档',
            details=f'上传 {file.filename}，{added_count}个文本块'
        )

        status_manager.update_status('idle', f'文档 {file.filename} 处理完成', 100)

        return jsonify({
            'message': '文档上传成功',
            'kb_id': kb_id,
            'file_name': file.filename,
            'chunk_count': added_count
        })

    except Exception as e:
        status_manager.update_status('error', f'处理失败: {str(e)}', 0)
        print(f"❌ 文档上传失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'文档处理失败: {str(e)}'}), 500


@app.route('/api/knowledge/list', methods=['GET'])
@login_required
def api_knowledge_list():
    """获取知识库列表"""
    try:
        knowledge_bases = db_manager.get_all_knowledge_bases()
        return jsonify(knowledge_bases)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge/stats', methods=['GET'])
@login_required
def api_knowledge_stats():
    """获取知识库统计"""
    try:
        rag = get_hybrid_rag()
        stats = rag.kb_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'total_documents': 0})


# ==================== RAG问答接口 ====================

@app.route('/api/qa/ask', methods=['POST'])
@login_required
def api_ask_question():
    """
    RAG问答接口

    RAG模式（联网关闭）:
    BGE检索 → 释放BGE → Qwen打分 → Qwen生成 → 释放Qwen

    RAG+联网模式（联网开启）:
    BGE检索 → 释放BGE → Qwen打分 → 如本地不足则联网 → Qwen生成 → 释放Qwen
    """
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    enable_web = data.get('enable_web', False)

    if not question:
        return jsonify({'error': '问题不能为空'}), 400

    if len(question) > 2000:
        return jsonify({'error': '问题长度不能超过2000字'}), 400

    try:
        mode_name = 'RAG + 联网搜索混合' if enable_web else 'RAG模式'
        print(f"\n{'=' * 60}")
        print(f"💬 问题: {question[:80]}...")
        print(f"🌐 模式: {mode_name}")

        status_manager.update_status(
            'loading_bge', '正在加载嵌入模型...', 10,
            bge_loaded=False, llm_loaded=False, web_enabled=enable_web
        )

        rag = get_hybrid_rag()

        status_manager.update_status(
            'searching', '正在检索本地知识库...', 30,
            bge_loaded=True, llm_loaded=False, web_enabled=enable_web
        )

        # 执行查询
        result = rag.query(question, enable_web=enable_web)

        status_manager.update_status(
            'generating', 'AI正在生成回答...', 70,
            bge_loaded=False, llm_loaded=True, web_enabled=enable_web
        )

        if result.get('success'):
            db_manager.add_qa_stat(
                user_id=session['user_id'],
                question=question,
                answer=result.get('answer', '')[:500],
                response_time=result.get('response_time', 0)
            )

            search_type = 'RAG+联网' if result.get('web_used') else 'RAG'
            db_manager.add_log(
                user_id=session['user_id'],
                action=f'知识问答({search_type})',
                details=f'提问: {question[:50]}... 耗时: {result.get("response_time", 0):.1f}s'
            )

            status_manager.update_status(
                'done', '回答完成！', 100,
                bge_loaded=False, llm_loaded=False
            )

            return jsonify({
                'answer': result['answer'],
                'local_sufficient': result.get('local_sufficient', True),
                'web_used': result.get('web_used', False),
                'response_time': f"{result.get('response_time', 0):.2f}秒",
                'search_mode': mode_name,
                'scores': result.get('scores', []),
                'success': True
            })
        else:
            status_manager.update_status('error', '处理失败', 0)
            return jsonify({
                'answer': result.get('answer', '处理失败'),
                'success': False,
                'error': result.get('error', '未知错误')
            }), 500

    except Exception as e:
        status_manager.update_status('error', f'错误: {str(e)[:50]}', 0)
        print(f"❌ 问答失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'问答失败: {str(e)}'}), 500


# ==================== 系统状态接口 ====================

@app.route('/api/status/current', methods=['GET'])
def api_current_status():
    """获取当前系统状态"""
    return jsonify(status_manager.get_current_status())


# ==================== 管理后台接口 ====================

@app.route('/api/admin/statistics', methods=['GET'])
@admin_required
def api_statistics():
    """获取系统统计数据"""
    try:
        stats = db_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/logs', methods=['GET'])
@admin_required
def api_logs():
    """获取操作日志"""
    limit = request.args.get('limit', 50, type=int)
    logs = db_manager.get_logs(limit)
    return jsonify(logs)


# ==================== 模型测试接口 ====================

@app.route('/api/test/models', methods=['GET'])
@login_required
def api_test_models():
    """测试模型状态"""
    try:
        import requests as req

        result = {
            'ollama_status': False,
            'models': [],
            'llm_model': {'name': Config.OLLAMA_LLM_MODEL, 'status': False},
            'embedding_model': {'name': Config.OLLAMA_EMBEDDING_MODEL, 'status': False},
        }

        try:
            resp = req.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                result['ollama_status'] = True
                result['models'] = [
                    {'name': m['name'], 'size': m.get('size', 'N/A')}
                    for m in resp.json().get('models', [])
                ]
        except:
            pass

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 删除文档接口 ====================

@app.route('/api/knowledge/delete/<int:kb_id>', methods=['DELETE'])
@login_required
def api_delete_document(kb_id):
    """删除知识库文档"""
    try:
        # 获取文档信息
        kb_info = db_manager.get_knowledge_base_by_id(kb_id)
        if not kb_info:
            return jsonify({'error': '文档不存在'}), 404

        file_name = kb_info['name']

        # 从Chroma向量数据库删除
        rag = get_hybrid_rag()
        deleted_chunks = rag.delete_document(file_name)

        # 从SQLite数据库删除
        db_manager.delete_knowledge_base(kb_id)

        # 记录日志
        db_manager.add_log(
            user_id=session['user_id'],
            action='删除文档',
            details=f'删除文档 {file_name}，移除 {deleted_chunks} 个向量块'
        )

        return jsonify({
            'message': f'文档 {file_name} 已删除',
            'deleted_chunks': deleted_chunks
        })

    except Exception as e:
        print(f"❌ 删除失败: {str(e)}")
        return jsonify({'error': f'删除失败: {str(e)}'}), 500


# ==================== 启动应用 ====================

if __name__ == '__main__':
    Config.init_directories()

    print("\n" + "=" * 60)
    print("🚀 RAG企业内部知识库问答系统")
    print("=" * 60)
    print(f"🤖 LLM模型:       {Config.OLLAMA_LLM_MODEL} (打分+生成)")
    print(f"📊 嵌入模型:      {Config.OLLAMA_EMBEDDING_MODEL} (检索)")
    print(f"💾 数据库:        {Config.SQLITE_DB_PATH}")
    print(f"🗂️  向量库:        {Config.CHROMA_DB_PATH}")
    print(f"📂 上传目录:      {Config.UPLOAD_FOLDER}")
    print(f"📏 分块大小:      {Config.CHUNK_SIZE}")
    print(f"🔍 检索Top-K:     {Config.RAG_TOP_K}")
    print(f"🎯 打分阈值:      score >= {Config.SCORE_THRESHOLD}")
    print(f"👤 管理员:        {Config.ADMIN_USERNAME} / 123456")
    print("=" * 60)
    print("💡 模型调度: BGE检索 → 释放 → Qwen打分 → Qwen生成 → 释放")
    print("💡 两模型严格互斥，4GB显存友好")
    print("💡 打分: Qwen3-4B + JSON Prompt (0-3分)")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)