"""
模型测试脚本
测试Ollama中的模型是否正常工作
"""
import sys
import os

sys.path.append(os.path.dirname(__file__))

from config import Config
from embeddings_manager import EmbeddingsManager
from langchain_community.llms import Ollama


def test_models():
    """测试所有模型"""
    print("=" * 60)
    print("🔍 模型测试开始")
    print("=" * 60)

    # 1. 测试Ollama连接
    print("\n📡 测试Ollama服务连接...")
    try:
        import requests
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama服务正常，已装载 {len(models)} 个模型:")
            for model in models:
                print(f"   - {model['name']} ({model['size']})")
        else:
            print("❌ Ollama服务响应异常")
    except Exception as e:
        print(f"❌ 无法连接到Ollama服务: {str(e)}")
        return

    # 2. 测试嵌入模型
    print(f"\n📊 测试嵌入模型: {Config.OLLAMA_EMBEDDING_MODEL}")
    embeddings_manager = EmbeddingsManager()

    test_texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的重要技术",
        "深度学习推动了AI的发展"
    ]

    try:
        vectors = embeddings_manager.embed_documents(test_texts)
        if vectors and len(vectors) == 3:
            print(f"✅ 嵌入模型工作正常")
            print(f"   - 向量维度: {len(vectors[0])}")

            # 测试相似度计算
            sim = embeddings_manager.compute_similarity(vectors[0], vectors[1])
            print(f"   - 相关文本相似度: {sim:.3f}")

            sim2 = embeddings_manager.compute_similarity(vectors[0], vectors[2])
            print(f"   - 相关文本相似度: {sim2:.3f}")
        else:
            print("❌ 嵌入模型返回结果异常")
    except Exception as e:
        print(f"❌ 嵌入模型测试失败: {str(e)}")

    # 3. 测试LLM模型
    print(f"\n🤖 测试LLM模型: {Config.OLLAMA_LLM_MODEL}")
    try:
        llm = Ollama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0.7,
            num_predict=100
        )

        test_prompt = "请用一句话介绍人工智能。"
        print(f"   测试提示: {test_prompt}")
        response = llm(test_prompt)
        print(f"✅ LLM模型工作正常")
        print(f"   响应: {response[:200]}...")
    except Exception as e:
        print(f"❌ LLM模型测试失败: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ 模型测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_models()