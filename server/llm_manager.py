"""
Qwen LLM模型管理模块
打分(0-10) + 回答生成
"""
import gc
import time
import re
import requests
from typing import List, Dict
from config import Config


class LLMManager:
    """Qwen LLM模型管理器"""

    def __init__(self):
        self.llm = None
        self.is_loaded = False
        self.embeddings_manager = None

    def set_embeddings_manager(self, embeddings_manager):
        self.embeddings_manager = embeddings_manager

    def _unload_from_ollama(self):
        """通过Ollama API卸载模型"""
        try:
            requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={'model': Config.OLLAMA_LLM_MODEL, 'prompt': '', 'keep_alive': 0},
                timeout=10
            )
        except:
            pass
        try:
            requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/chat",
                json={'model': Config.OLLAMA_LLM_MODEL, 'messages': [], 'keep_alive': 0},
                timeout=10
            )
        except:
            pass
        time.sleep(2)

    def load_model(self):
        """加载Qwen模型"""
        if self.is_loaded:
            return True

        if self.embeddings_manager and self.embeddings_manager.is_loaded:
            print("🔄 先卸载BGE模型...")
            self.embeddings_manager.unload_model()
            time.sleep(2)

        try:
            print(f"🔄 正在加载Qwen模型: {Config.OLLAMA_LLM_MODEL}")
            from langchain_ollama import OllamaLLM

            self.llm = OllamaLLM(
                base_url=Config.OLLAMA_BASE_URL,
                model=Config.OLLAMA_LLM_MODEL,
                temperature=Config.TEMPERATURE,
                num_predict=Config.MAX_TOKENS,
                top_p=0.9,
                top_k=40,
                verbose=False,
                keep_alive=0
            )

            self.is_loaded = True
            print(f"✅ Qwen模型加载成功")
            return True
        except Exception as e:
            print(f"❌ Qwen加载失败: {str(e)}")
            return False

    def unload_model(self):
        """卸载Qwen模型"""
        if not self.is_loaded:
            return

        try:
            print("🗑️ 正在卸载Qwen模型...")
            self.llm = None
            self.is_loaded = False
            self._unload_from_ollama()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            time.sleep(1)
            print("✅ Qwen已卸载")
        except Exception as e:
            print(f"⚠️ 卸载警告: {str(e)}")
            self.is_loaded = False

    def score_document(self, query: str, document: str) -> int:
        """
        使用Qwen3-4B打分 (0-10)
        """
        if not self.is_loaded:
            if not self.load_model():
                return 0

        doc_text = document[:800]

        prompt = f"""判断文档与查询的相关性。仅输出JSON {{"score": 0-10的整数}}

评分标准：
- 10: 完全匹配，直接完整回答问题
- 8-9: 高度相关，包含大部分关键信息
- 6-7: 相关，包含部分关键信息
- 4-5: 部分相关，涉及相关主题但信息不完整
- 2-3: 弱相关，仅有少量关联
- 0-1: 基本无关或完全无关

查询: {query}
文档: {doc_text}

仅输出: {{"score": 数字}}"""

        try:
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    'model': Config.OLLAMA_LLM_MODEL,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0, 'num_predict': 50}
                },
                timeout=60
            )

            if response.status_code == 200:
                text = response.json().get('response', '')
                match = re.search(r'"score"\s*:\s*(\d+)', text)
                if match:
                    score = int(match.group(1))
                    return max(0, min(10, score))
                nums = re.findall(r'\b(\d+)\b', text)
                if nums:
                    score = int(nums[0])
                    return max(0, min(10, score))

            return 0

        except Exception as e:
            print(f"   ⚠️ 打分失败: {str(e)}")
            return 0

    def rank_pool(self, query: str, pool: List[Dict]) -> List[Dict]:
        """
        对文档池统一排序，按分数取Top
        """
        if not pool:
            return []

        print(f"🔄 排序 {len(pool)} 个文档...")

        # 已有分数的保持，没有的重新打分
        for item in pool:
            if item.get('score', 0) == 0 and item.get('source') == 'web':
                item['score'] = self.score_document(query, item.get('content', ''))

        # 按分数降序
        pool.sort(key=lambda x: x.get('score', 0), reverse=True)
        top = pool[:Config.FINAL_TOP_K]

        local_count = sum(1 for d in top if d.get('source') == 'local')
        web_count = sum(1 for d in top if d.get('source') == 'web')
        print(f"📊 Top-{len(top)}: 本地{local_count} + 网络{web_count}")

        return top

    def generate_answer(self, query: str, formatted_docs: str) -> str:
        """
        使用Qwen3-4B生成最终回答
        """
        if not self.is_loaded:
            if not self.load_model():
                return "模型加载失败，请重试"

        prompt = f"""基于以下参考文档回答用户问题。
规则：
1. 仅使用参考文档中的信息，禁止编造或引入外部知识
2. 回答要准确、简洁，可分点陈述
3. 在关键事实后标注来源编号，如[1][2]
4. 如果参考文档确实完全不包含相关信息，才回复"根据现有资料无法回答"
5. 回答结束后，必须另起一行添加"参考资料："，并按格式分段列出所有被引用的文档：[编号]《标题》

参考文档：
{formatted_docs}

用户问题：{query}
回答："""

        try:
            response = requests.post(
                f"{Config.OLLAMA_BASE_URL}/api/generate",
                json={
                    'model': Config.OLLAMA_LLM_MODEL,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': Config.TEMPERATURE,
                        'num_predict': Config.MAX_TOKENS
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get('response', '').strip()
            return "生成回答失败"

        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return f"生成回答时出错: {str(e)}"

    def __del__(self):
        self.unload_model()