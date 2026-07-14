"""
系统配置文件
"""
import os


class Config:
    """系统配置类"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # SQLite数据库
    SQLITE_DB_PATH = os.path.join(BASE_DIR, 'data', 'rag_system.db')

    # Chroma向量数据库
    CHROMA_DB_PATH = os.path.join(BASE_DIR, 'data', 'chroma_db')

    # 文档存储
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')

    # ========== Ollama配置 ==========
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_LLM_MODEL = "qwen3-4b:latest"
    OLLAMA_EMBEDDING_MODEL = "bge-large-zh:latest"

    # Flask配置
    SECRET_KEY = 'your-secret-key-change-in-production'

    # 管理员
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD_MD5 = 'e10adc3949ba59abbe56e057f20f883e'  # 123456

    # 文档处理
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # RAG配置
    RAG_TOP_K = 8  # BGE检索候选文档数
    SCORE_THRESHOLD = 6  # 文档相关性阈值（>=6认为相关，满分10）

    # 本地充足判断阈值（不同模式）
    MIN_RELEVANT_DOCS_RAG = 4  # RAG模式：>=4个高分文档认为充足
    MIN_RELEVANT_DOCS_HYBRID = 6  # 混合模式：>=6个高分文档认为充足

    FINAL_TOP_K = 6  # 最终保留文档数

    # 网络搜索
    MAX_WEB_RESULTS = 5
    WEB_SEARCH_TIMEOUT = 15

    # LLM生成
    MAX_TOKENS = 1024
    TEMPERATURE = 0.7

    @classmethod
    def init_directories(cls):
        for d in [
            os.path.dirname(cls.SQLITE_DB_PATH),
            cls.CHROMA_DB_PATH,
            cls.UPLOAD_FOLDER
        ]:
            os.makedirs(d, exist_ok=True)