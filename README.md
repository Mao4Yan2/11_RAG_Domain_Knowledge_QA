### 注：
1. 我电脑是11代i5+3050笔记本，显存只有4g，而一个qwen3-b模型就2g起步。所以我采取了“按需加载与释放” 策略，导致rag运行有点慢。
2. 我为了方便，是用Qwen3-4B-Instruct + JSON Prompt来给“问题与数据库的相关性”打分的，没用reranker模型。
3. 这是个人项目
4. 账号：admin  密码：123456
---

## 安装依赖并启动
1. 下载 Ollama 软件
2. 下载 Qwen3-4B-Instruct模型、bge-large-zh模型
3. 创建 Modelfile 文件（参考Models文件夹里面的文件）
4. 在 Models 文件夹内打开 cmd ，输入创建模型的命令：
```ollama create qwen3-4b -f Modelfile.qwen```
```ollama create bge-large-zh -f Modelfile.bge_large```
5. 创建新虚拟环境，在虚拟环境中批量安装 requirements.txt 的所有依赖包，指令为：  ```pip install -r requirements.txt```
6. 启动后端```python ./server/app.py```（建议在编辑器上启动）
7. 启动前端 ```streamlit run ./client/app.py```（或双击start.bat）
8. 最后在浏览器上打开```http://localhost:8501```
9. 登录账号：admin  密码：123456
---

## 目录结构描述
```
_Project > 11_RAG_Domain_Knowledge_QA
├── Models/                  # 用于参考（无模型）
├── client/ 
│ ├── app.py                 # 前端代码
│ ├── start.bat              # 前端启动文件
│ └── requirements.txt       # 依赖
├──server/
│ ├── __init__.py
│ ├── app.py                 # Flask主应用
│ ├── config.py              # 配置
│ ├── models.py              # SQLite数据库
│ ├── auth.py                # 认证
│ ├── document_processor.py  # 文档处理
│ ├── text_cleaner.py        # 文本清洗
│ ├── embeddings_manager.py  # BGE嵌入管理
│ ├── llm_manager.py         # Qwen LLM管理（打分+生成）
│ ├── knowledge_base.py      # Chroma知识库
│ ├── hybrid_rag.py          # RAG策略
│ ├── web_search.py          # 网络搜索
│ ├── status_manager.py      # 状态管理
│ └── requirements.txt       # 依赖
└── README.md                # 项目说明文档
```
---
## 代码说明

1. client是前端；server是后端
2. Python开发，前端使用Streamlit；后端使用Flask框架
3. Chroma + SQLite 混合存储方案。SQLite存用户、角色、操作日志、统计；Chroma存文档向量、知识库
4. 用Qwen3-4B-Instruct + JSON Prompt来给“问题与数据库的相关性”打分。 

---
## Prompt 模板
**Qwen3-4B 打分 Prompt 模板为：**
```
prompt = f"""判断文档与查询的相关性。仅输出JSON {{"score": 0/1/2/3}}
3=直接完整回答 2=部分相关 1=边缘相关 0=无关
查询:{query}
文档:{doc}"""
```

**RAG 回答 Prompt 模板为：**
```
answer_prompt = f"""基于以下参考文档回答用户问题。
规则：
1. 仅使用参考文档中的信息，禁止编造或引入外部知识
2. 若文档信息不足以回答问题，直接回复"根据现有资料无法回答"
3. 回答需简洁准确，必要时可分点陈述
4. 在关键事实后标注来源编号，如[1][2]

参考文档：
{formatted_docs}

用户问题：{query}
回答："""
```
---
## 回答流程
“RAG + 联网搜索混合”的步骤是：
```
0.用户未上传问题前，所有模型是关闭状态
1. 用户提问
2. 启用 bge-large-zh 将问题转为向量
3. 用向量在本地知识库中检索相关文档 (RAG)，得到 Top-K 候选文档 (建议 K=8，太多会拖慢打分)
4. 显式卸载bge-large-zh模型，启用qwen3-4b，逐个文档串行打分
5. 根据重排序分数做决策：如果 >= 6 个文档分数 >= 6，说明本地知识充足，走纯本地回答; 如果 < 6 个文档分数 >= 6，说明本地知识不足，触发联网
6. 如果本地知识不足,触发联网搜索（如果开启）或直接返回"根据现有资料无法回答"
7. 同时向 Brave 和 DuckDuckGo 发起搜索请求；如果有一个搜索失败或超时，则将其结果置空；如果两个都失败，则回退到纯本地知识库
8. 获取两个来源的结果后，按标题或链接进行去重，并对联网搜索结果打分
9. 使用concurrent.futures技术合并 RAG 检索结果 + 两个联网搜索结果成一个更大的候选文档池，再对合并后的文档池 pool 进行按分数排序，取分数最高的 Top-6 个。
10. 再用 qwen3-4b 生成最终回答
11. 生成完后显式卸载qwen3-4b模型，准备将下一个问题
```

“RAG模式”的步骤是：
```
0.用户未上传问题前，所有模型是关闭状态
1. 用户提问
2. 启用 bge-large-zh 将问题转为向量
3. 用向量在本地知识库中检索相关文档 (RAG)，得到 Top-K 候选文档 (建议 K=8，太多会拖慢打分)
4. 显式卸载bge-large-zh模型，启用qwen3-4b，逐个文档串行打分
5. 根据重排序分数做决策：如果 >= 4 个文档分数 >= 6，说明本地知识充足，走纯本地回答; 如果 < 4 个文档分数 >= 6，说明本地知识不足，回复"根据现有资料无法回答"
6. 再用 qwen3-4b 生成最终回答
7.生成完后显式卸载qwen3-4b模型，准备将下一个问题
```
---
## 自用ollama指令

查看模型，有模型就不用创建，直接run启动
```ollama list```

创建模型的命令
```ollama create qwen3-4b -f Modelfile.qwen```
```ollama create bge-large-zh -f Modelfile.bge_large```

启动了模型
```ollama run qwen3-4b ```
```ollama run bge-large-zh```

卸载模型
```ollama stop qwen3-4b```
```ollama stop bge-large-zh```


移除已创建的模型
```ollama rm qwen3-4b```
```ollama rm bge-large-zh```

显示创建模型时使用的完整 Modelfile 配置
```ollama show bge-large-zh --modelfile```
```ollama show qwen3-4b --modelfile```
