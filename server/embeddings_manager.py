"""
BGE嵌入模型管理模块
与Qwen模型互斥，按需加载
"""
import gc
import time
import requests
from typing import List
from config import Config


class EmbeddingsManager:
    """BGE嵌入模型管理器"""

    def __init__(self, lazy_load=True):
        self.embeddings = None
        self.is_loaded = False
        self.llm_manager = None

        if not lazy_load:
            self.load_model()

    def set_llm_manager(self, llm_manager):
        self.llm_manager = llm_manager

    def _unload_from_ollama(self, model_name: str):
        """通过Ollama API卸载模型"""
        try:
            requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={'model': model_name, 'prompt': '', 'keep_alive': 0},
                timeout=10
            )
        except:
            pass
        try:
            requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/chat",
                json={'model': model_name, 'messages': [], 'keep_alive': 0},
                timeout=10
            )
        except:
            pass
        time.sleep(1)

    def load_model(self):
        """加载BGE模型，先卸载Qwen"""
        if self.is_loaded:
            return True

        if self.llm_manager and self.llm_manager.is_loaded:
            print("🔄 先卸载Qwen模型...")
            self.llm_manager.unload_model()
            time.sleep(2)

        try:
            print(f"🔄 正在加载BGE模型: {Config.OLLAMA_EMBEDDING_MODEL}")
            from langchain_ollama import OllamaEmbeddings

            self.embeddings = OllamaEmbeddings(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL,
            )

            self.is_loaded = True
            print(f"✅ BGE模型加载成功")
            return True
        except Exception as e:
            print(f"❌ BGE加载失败: {str(e)}")
            return False

    def unload_model(self):
        """卸载BGE模型"""
        if not self.is_loaded:
            return

        try:
            print("🗑️ 正在卸载BGE模型...")
            self.embeddings = None
            self.is_loaded = False
            self._unload_from_ollama(Config.OLLAMA_EMBEDDING_MODEL)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            time.sleep(1)
            print("✅ BGE已卸载")
        except Exception as e:
            print(f"⚠️ 卸载警告: {str(e)}")
            self.is_loaded = False

    def ensure_loaded(self):
        """确保模型已加载（供外部调用）"""
        if not self.is_loaded:
            return self.load_model()
        return True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        if not self.ensure_loaded():
            raise RuntimeError("BGE模型未加载")
        try:
            return self.embeddings.embed_documents(texts)
        except:
            return [self.embeddings.embed_query(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        if not self.ensure_loaded():
            raise RuntimeError("BGE模型未加载")
        try:
            return self.embeddings.embed_query(query)
        except:
            return []

    def test_connection(self) -> bool:
        """测试连接"""
        if not self.ensure_loaded():
            return False
        try:
            v = self.embed_query("测试")
            return bool(v and len(v) > 0)
        except:
            return False