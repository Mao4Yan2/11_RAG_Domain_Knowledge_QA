"""
重排序模型管理模块
使用Xinference RESTfulClient管理qwen3-reranker模型
按需加载，用完即释放，与Ollama模型互斥
"""
import gc
import time
from typing import List, Tuple, Dict
from xinference.client import RESTfulClient
from config import Config


class RerankerManager:
    """重排序模型管理器，通过Xinference RESTfulClient自动管理模型生命周期"""

    def __init__(self):
        self.is_loaded = False
        self.model_uid = None
        self.client = None
        self.model_name = Config.XINFERENCE_RERANKER_MODEL
        self.model_path = Config.XINFERENCE_RERANKER_PATH

    def _get_client(self) -> RESTfulClient:
        """获取Xinference客户端连接"""
        if self.client is None:
            self.client = RESTfulClient(Config.XINFERENCE_BASE_URL)
        return self.client

    def load_model(self):
        """
        加载重排序模型
        使用RESTfulClient自动注册并启动模型
        """
        if self.is_loaded and self.model_uid:
            return True

        try:
            print(f"🔄 正在通过Xinference加载重排序模型: {self.model_name}")

            client = self._get_client()

            # 检查模型是否已注册
            try:
                existing_models = client.list_models()
                for model in existing_models:
                    if model.get('model_name') == self.model_name:
                        self.model_uid = model.get('model_uid')
                        print(f"✅ 模型已存在: {self.model_name} (uid={self.model_uid})")
                        self.is_loaded = True
                        return True
            except:
                pass

            # 注册并启动模型
            print(f"📦 注册模型: {self.model_name}")
            print(f"📂 模型路径: {self.model_path}")

            # 使用LLM类型注册reranker模型
            model_uid = client.launch_model(
                model_name=self.model_name,
                model_format="gguf",
                model_path=self.model_path,
                model_type="LLM",  # Xinference中reranker作为LLM类型运行
            )

            self.model_uid = model_uid
            self.is_loaded = True

            print(f"✅ 重排序模型加载成功: {self.model_name} (uid={model_uid})")
            return True

        except Exception as e:
            print(f"❌ 重排序模型加载失败: {str(e)}")
            # 尝试获取已运行的模型
            try:
                client = self._get_client()
                models = client.list_models()
                for model in models:
                    if model.get('model_name') == self.model_name:
                        self.model_uid = model.get('model_uid')
                        self.is_loaded = True
                        print(f"✅ 使用已运行的模型: uid={self.model_uid}")
                        return True
            except:
                pass
            return False

    def unload_model(self):
        """卸载重排序模型，释放显存"""
        if not self.is_loaded:
            return

        try:
            print("🗑️ 正在卸载重排序模型...")

            if self.model_uid:
                client = self._get_client()
                client.terminate_model(self.model_uid)
                print(f"✅ 模型已终止: uid={self.model_uid}")

            self.model_uid = None
            self.is_loaded = False
            self.client = None

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            time.sleep(1)
            print("✅ 重排序模型已完全卸载")

        except Exception as e:
            print(f"⚠️ 卸载警告: {str(e)}")
            self.is_loaded = False
            self.model_uid = None

    def _call_reranker(self, question: str, document: str) -> float:
        """
        调用reranker模型对单个文档评分

        Args:
            question: 用户问题
            document: 文档内容
        Returns:
            float: 相关性分数 0-1
        """
        # 截断文档
        content = document[:800]

        # 构建reranker prompt
        prompt = f"""<|im_start|>system
你是一个文档相关性评估专家。请评估文档与问题的相关程度，只输出0到1之间的分数。

评分标准：
- 1.0: 完全相关，直接回答问题
- 0.8: 高度相关，包含关键信息
- 0.5: 部分相关，涉及相关主题
- 0.2: 弱相关，仅有少量关联
- 0.0: 完全不相关

只输出一个数字，不要其他内容。
<|im_end|>
<|im_start|>user
问题：{question}

文档：{content}

相关度分数（0-1）：<|im_end|>
<|im_start|>assistant
"""

        try:
            client = self._get_client()
            model = client.get_model(self.model_uid)

            # 调用模型生成评分
            response = model.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            # 提取响应文本
            if isinstance(response, dict):
                text = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                text = str(response)

            text = text.strip()

            # 提取数字
            import re
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))

            return 0.0

        except Exception as e:
            print(f"   ⚠️ 评分失败: {str(e)}")
            return 0.0

    def rerank(self, question: str, documents: List, top_k: int = 5) -> List[Tuple]:
        """
        对文档列表进行重排序

        Args:
            question: 用户问题
            documents: Document列表
            top_k: 返回数量
        Returns:
            List[Tuple]: [(document, score), ...] 按分数降序
        """
        if not documents:
            return []

        if not self.is_loaded:
            if not self.load_model():
                print("⚠️ 无法加载重排序模型，返回原始排序")
                return [(doc, 0.3) for doc in documents[:top_k]]

        print(f"🔄 正在重排序 {len(documents)} 个文档...")

        scores = []
        for i, doc in enumerate(documents):
            score = self._call_reranker(question, doc.page_content)
            scores.append(score)
            print(f"  文档 {i + 1}/{len(documents)}: 分数={score:.3f}")
            if (i + 1) % 5 == 0:
                print(f"  进度: {i + 1}/{len(documents)}")

        # 组合并排序
        pairs = list(zip(documents, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top = pairs[:top_k]
        avg = sum(s for _, s in top) / len(top) if top else 0
        print(f"✅ 重排序完成, Top{len(top)}平均分: {avg:.3f}")

        return top

    def batch_rerank(self, question: str, documents: List, top_k: int = 5) -> Tuple[List[Tuple], Dict]:
        """
        批量重排序并返回决策信息

        Returns:
            Tuple: (重排序结果, 决策信息)
        """
        reranked = self.rerank(question, documents, top_k)

        if not reranked:
            return [], {
                'sufficient': False,
                'max_score': 0,
                'avg_score': 0,
                'all_scores': [],
                'threshold': Config.RERANKER_THRESHOLD,
                'reason': '无文档可排序'
            }

        scores = [s for _, s in reranked]
        max_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        sufficient = max_score >= Config.RERANKER_THRESHOLD

        decision = {
            'sufficient': sufficient,
            'max_score': round(max_score, 4),
            'avg_score': round(avg_score, 4),
            'all_scores': [round(s, 4) for s in scores],
            'threshold': Config.RERANKER_THRESHOLD,
            'reason': f"最高分{max_score:.3f}{'≥' if sufficient else '<'}阈值{Config.RERANKER_THRESHOLD}"
        }

        print(f"📊 重排序决策: {decision['reason']}")
        print(f"   所有分数: {decision['all_scores']}")

        if sufficient:
            filtered = [(doc, s) for doc, s in reranked if s >= Config.RERANKER_THRESHOLD]
            return filtered, decision
        else:
            return [], decision

    def __del__(self):
        self.unload_model()