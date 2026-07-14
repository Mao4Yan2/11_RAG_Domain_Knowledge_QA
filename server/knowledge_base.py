"""
知识库管理模块
"""
from typing import List, Tuple
from langchain_chroma import Chroma
from config import Config
from embeddings_manager import EmbeddingsManager


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self):
        self.embeddings_manager = EmbeddingsManager(lazy_load=True)
        self.vector_store = None

    def _init_vector_store(self):
        """初始化向量数据库"""
        try:
            self.embeddings_manager.ensure_loaded()
            self.vector_store = Chroma(
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embeddings_manager.embeddings,
                collection_name="knowledge_base"
            )
            print("✅ 向量数据库初始化完成")
        except Exception as e:
            print(f"⚠️ 向量数据库初始化失败: {str(e)}")
            self.vector_store = None

    def add_documents(self, documents: List) -> int:
        """添加文档"""
        if not documents:
            return 0
        if not self.embeddings_manager.ensure_loaded():
            raise RuntimeError("BGE模型加载失败")
        if self.vector_store is None:
            self._init_vector_store()

        try:
            batch_size = 50
            added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_store.add_documents(batch)
                added += len(batch)
            return added
        except Exception as e:
            print(f"❌ 添加失败: {str(e)}")
            return self._add_one_by_one(documents)

    def _add_one_by_one(self, documents: List) -> int:
        """逐个添加"""
        added = 0
        for doc in documents:
            try:
                self.vector_store.add_documents([doc])
                added += 1
            except:
                pass
        return added

    def delete_by_source(self, source_name: str) -> int:
        """
        按源文件名删除文档

        Args:
            source_name: 文件名
        Returns:
            int: 删除数量
        """
        if self.vector_store is None:
            self._init_vector_store()
        if self.vector_store is None:
            return 0

        try:
            collection = self.vector_store._collection
            # 获取所有文档
            all_docs = collection.get()

            if not all_docs or not all_docs.get('ids'):
                return 0

            # 找到匹配的文档ID
            ids_to_delete = []
            for i, meta in enumerate(all_docs.get('metadatas', [])):
                if meta and meta.get('source') == source_name:
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(f"✅ 已删除 {len(ids_to_delete)} 个文档块 (来源: {source_name})")
                return len(ids_to_delete)

            return 0
        except Exception as e:
            print(f"❌ 删除失败: {str(e)}")
            return 0

    def search_with_scores(self, query: str, k: int = 5) -> List[Tuple]:
        """搜索文档"""
        if not self.embeddings_manager.ensure_loaded():
            return []
        if self.vector_store is None:
            self._init_vector_store()
        if self.vector_store is None:
            return []

        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except:
            return []

    def get_stats(self):
        """获取统计"""
        try:
            if self.vector_store:
                return {'total_documents': self.vector_store._collection.count()}
            return {'total_documents': 0}
        except:
            return {'total_documents': 0}