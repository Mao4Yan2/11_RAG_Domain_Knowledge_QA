"""测试重排序模型"""
import requests

API = "http://localhost:11434/api"
MODEL = "bge-reranker-base"  # ← 改成你的实际模型名

print(f"测试模型: {MODEL}")

# 测试generate
print("\n1. 测试 generate API...")
r = requests.post(f"{API}/generate", json={
    'model': MODEL,
    'prompt': 'hello',
    'stream': False,
    'options': {'num_predict': 5}
}, timeout=10)
print(f"   状态码: {r.status_code}")
print(f"   响应: {r.text[:200]}")

# 测试chat
print("\n2. 测试 chat API...")
r2 = requests.post(f"{API}/chat", json={
    'model': MODEL,
    'messages': [{'role': 'user', 'content': 'hello'}],
    'stream': False,
    'options': {'num_predict': 5}
}, timeout=10)
print(f"   状态码: {r2.status_code}")
print(f"   响应: {r2.text[:200]}")

# 查看已安装模型
print("\n3. 已安装模型:")
r3 = requests.get(f"{API}/tags", timeout=5)
for m in r3.json().get('models', []):
    print(f"   - {m['name']}")