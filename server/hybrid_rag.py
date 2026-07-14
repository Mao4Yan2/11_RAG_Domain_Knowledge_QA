"""
Hybrid RAG策略模块
"""
from typing import Dict, List
import time
from config import Config
from embeddings_manager import EmbeddingsManager
from llm_manager import LLMManager
from knowledge_base import KnowledgeBaseManager
from web_search import WebSearchManager


class HybridRAG:
    """混合RAG系统"""

    def __init__(self):
        print("🚀 初始化Hybrid RAG系统...")

        self.llm_manager = LLMManager()
        self.embeddings_manager = EmbeddingsManager(lazy_load=True)

        self.embeddings_manager.set_llm_manager(self.llm_manager)
        self.llm_manager.set_embeddings_manager(self.embeddings_manager)

        self.kb_manager = KnowledgeBaseManager()
        self.kb_manager.embeddings_manager = self.embeddings_manager

        self.web_search = WebSearchManager(max_results=Config.MAX_WEB_RESULTS)

        print("✅ Hybrid RAG系统初始化完成")

    def query(self, question: str, enable_web: bool = True) -> Dict:
        """
        执行查询

        RAG模式: >=4个文档score>=6 → 本地充足
        混合模式: >=6个文档score>=6 → 本地充足，否则联网
        """
        start_time = time.time()
        min_docs = Config.MIN_RELEVANT_DOCS_HYBRID if enable_web else Config.MIN_RELEVANT_DOCS_RAG

        try:
            print("\n" + "=" * 60)
            print(f"🔄 查询: {question[:60]}...")
            print(f"🌐 模式: {'RAG+联网' if enable_web else 'RAG'}")
            print(f"📊 充足阈值: >={min_docs}个文档score>={Config.SCORE_THRESHOLD}")

            # 阶段1: BGE检索
            print("\n📊 [1/4] BGE检索...")
            if not self.embeddings_manager.load_model():
                return {'answer': '嵌入模型加载失败', 'success': False}

            raw_results = self.kb_manager.search_with_scores(question, k=Config.RAG_TOP_K)

            print("🗑️ 释放BGE...")
            self.embeddings_manager.unload_model()
            time.sleep(1)

            if not raw_results:
                if enable_web:
                    return self._web_only_query(question)
                return {'answer': '知识库为空，请先上传文档。', 'success': True, 'local_sufficient': False}

            local_docs = [doc for doc, _ in raw_results]
            print(f"📄 BGE检索: {len(local_docs)}个文档")

            # 阶段2: Qwen打分
            print(f"\n🎯 [2/4] Qwen打分(0-10)...")
            if not self.llm_manager.load_model():
                return {'answer': 'LLM加载失败', 'success': False}

            scored_local = []
            for i, doc in enumerate(local_docs):
                score = self.llm_manager.score_document(question, doc.page_content)
                scored_local.append({
                    'content': doc.page_content,
                    'source': 'local',
                    'title': f"本地文档{i + 1}",
                    'score': score,
                    'metadata': doc.metadata
                })
                print(f"  文档{i + 1}: 分数={score}")

            high_score_count = sum(1 for d in scored_local if d['score'] >= Config.SCORE_THRESHOLD)
            local_sufficient = high_score_count >= min_docs

            print(f"📊 高分文档(>={Config.SCORE_THRESHOLD}): {high_score_count}/{len(scored_local)}")
            print(f"📊 本地充足: {'是' if local_sufficient else '否'}")

            # 阶段3: 联网（如需要）
            web_docs = []
            web_used = False

            if not local_sufficient and enable_web:
                print("🌐 触发双引擎搜索...")
                web_result = self.web_search.search(question)

                if web_result['success'] and web_result['results']:
                    for r in web_result['results']:
                        web_docs.append({
                            'content': f"{r['title']}\n{r['snippet']}",
                            'source': 'web',
                            'title': r.get('title', ''),
                            'link': r.get('link', ''),
                            'score': 0
                        })
                    web_used = True
                    print(f"🌐 获取{len(web_docs)}个网络结果")
                else:
                    print("⚠️ 网络搜索失败，回退纯本地")
                    local_sufficient = True

            # 阶段4: 合并排序
            pool = scored_local + web_docs

            if len(pool) > Config.FINAL_TOP_K:
                print(f"\n📊 [3/4] 统一排序(池:{len(pool)})...")
                final_docs = self.llm_manager.rank_pool(question, pool)
            else:
                final_docs = sorted(pool, key=lambda x: x.get('score', 0), reverse=True)

            formatted = self._format_pool(final_docs)

            # 阶段5: 生成
            print(f"\n🤖 [4/4] Qwen生成...")
            answer = self.llm_manager.generate_answer(question, formatted)

            print("🗑️ 释放Qwen...")
            self.llm_manager.unload_model()

            elapsed = time.time() - start_time
            print(f"⏱️ 总耗时: {elapsed:.1f}s")
            print("=" * 60 + "\n")

            return {
                'answer': answer,
                'local_sufficient': local_sufficient,
                'web_used': web_used,
                'response_time': round(elapsed, 2),
                'high_score_count': high_score_count,
                'total_docs': len(final_docs),
                'success': True
            }

        except Exception as e:
            print(f"❌ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                self.embeddings_manager.unload_model()
            except:
                pass
            try:
                self.llm_manager.unload_model()
            except:
                pass
            return {'answer': f'处理失败: {str(e)}', 'success': False}

    def _web_only_query(self, question: str) -> Dict:
        """纯网络查询"""
        web_result = self.web_search.search(question)
        if not web_result['success']:
            return {'answer': '知识库为空且网络搜索失败。', 'success': True}

        web_docs = [{
            'content': f"{r['title']}\n{r['snippet']}",
            'source': 'web',
            'title': r.get('title', ''),
            'score': 0
        } for r in web_result['results']]

        if not self.llm_manager.load_model():
            return {'answer': 'LLM加载失败', 'success': False}

        formatted = self._format_pool(web_docs)
        answer = self.llm_manager.generate_answer(question, formatted)
        self.llm_manager.unload_model()

        return {'answer': answer, 'local_sufficient': False, 'web_used': True, 'success': True}

    def process_document(self, chunks: List) -> int:
        """处理文档上传"""
        try:
            if not self.embeddings_manager.load_model():
                raise RuntimeError("BGE加载失败")
            self.kb_manager._init_vector_store()
            added = self.kb_manager.add_documents(chunks)
            self.embeddings_manager.unload_model()
            return added
        except Exception as e:
            self.embeddings_manager.unload_model()
            raise e

    def delete_document(self, source_name: str) -> int:
        """删除文档"""
        return self.kb_manager.delete_by_source(source_name)

    def _format_pool(self, docs: List[Dict]) -> str:
        """格式化文档池"""
        parts = []
        for i, doc in enumerate(docs, 1):
            source_tag = "本地" if doc.get('source') == 'local' else "网络"
            title = doc.get('title', '')
            content = doc.get('content', '')
            score = doc.get('score', 0)
            parts.append(f"[{i}] [{source_tag}] (相关度:{score}) 《{title}》\n{content}")
        return "\n\n".join(parts)