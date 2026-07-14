"""
文本清洗与智能分块模块
提供文本清洗、去重、格式化以及智能分块功能
"""
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import jieba


class TextCleaner:
    """文本清洗器，负责文本预处理和智能分块"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文本清洗器

        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # 多个空行
                "\n\n",  # 段落分隔
                "\n",  # 换行
                "。",  # 句号
                "！",  # 感叹号
                "？",  # 问号
                "；",  # 分号
                "，",  # 逗号
                "、",  # 顿号
                " ",  # 空格
                ""  # 字符级分割
            ]
        )

    def clean_text(self, text: str) -> str:
        """
        清洗文本内容

        Args:
            text: 原始文本
        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        # 1. 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)

        # 2. 移除特殊控制字符（保留换行符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 3. 统一标点符号
        # 英文标点转中文标点
        punctuation_map = {
            ',': '，',
            ';': '；',
            ':': '：',
            '!': '！',
            '?': '？',
            '(': '（',
            ')': '）',
            '[': '【',
            ']': '】'
        }
        for eng, chn in punctuation_map.items():
            # 只在中文上下文中转换
            text = re.sub(f'([\u4e00-\u9fff]){re.escape(eng)}([\u4e00-\u9fff])',
                          f'\\1{chn}\\2', text)

        # 4. 移除重复的标点符号
        text = re.sub(r'([，。！？；、])\1+', r'\1', text)

        # 5. 处理连续换行（最多保留两个换行）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 6. 去除首尾空白
        text = text.strip()

        return text

    def clean_documents(self, documents: List[Dict]) -> List[Document]:
        """
        清洗文档列表并转换为LangChain Document格式

        Args:
            documents: 原始文档列表
        Returns:
            List[Document]: 清洗后的LangChain文档列表
        """
        cleaned_docs = []

        for doc in documents:
            # 清洗文本
            cleaned_text = self.clean_text(doc['text'])

            if len(cleaned_text) < 10:  # 跳过太短的文本
                continue

            # 创建LangChain Document
            langchain_doc = Document(
                page_content=cleaned_text,
                metadata={
                    **doc.get('metadata', {}),
                    'original_length': len(doc['text']),
                    'cleaned_length': len(cleaned_text),
                    'has_table': '|' in cleaned_text  # 标记是否包含表格
                }
            )
            cleaned_docs.append(langchain_doc)

        return cleaned_docs

    def smart_chunk(self, documents: List[Document]) -> List[Document]:
        """
        智能分块：根据文档结构进行分块

        Args:
            documents: 清洗后的文档列表
        Returns:
            List[Document]: 分块后的文档列表
        """
        chunks = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata

            # 根据文档类型选择不同的分块策略
            if metadata.get('has_table'):
                # 包含表格的文档，保持表格完整性
                chunks.extend(self._chunk_with_table(text, metadata))
            elif len(text) > self.chunk_size * 2:
                # 长文档使用语义分块
                chunks.extend(self._semantic_chunk(text, metadata))
            else:
                # 短文档使用标准分块
                doc_chunks = self.text_splitter.split_documents([doc])
                chunks.extend(doc_chunks)

        return chunks

    def _chunk_with_table(self, text: str, metadata: Dict) -> List[Document]:
        """
        处理包含表格的文档分块，保持表格完整性

        Args:
            text: 文档文本
            metadata: 文档元数据
        Returns:
            List[Document]: 分块列表
        """
        chunks = []

        # 分离表格和普通文本
        parts = re.split(r'(\n(?:[^|\n]+\|)+[^|\n]+\n)', text)

        current_chunk = ""
        for part in parts:
            if '|' in part and '\n' in part:
                # 这是一个表格
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={**metadata, 'chunk_type': 'text'}
                    ))
                chunks.append(Document(
                    page_content=part.strip(),
                    metadata={**metadata, 'chunk_type': 'table'}
                ))
                current_chunk = ""
            else:
                if len(current_chunk) + len(part) > self.chunk_size:
                    if current_chunk:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={**metadata, 'chunk_type': 'text'}
                        ))
                    current_chunk = part
                else:
                    current_chunk += part

        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={**metadata, 'chunk_type': 'text'}
            ))

        return chunks

    def _semantic_chunk(self, text: str, metadata: Dict) -> List[Document]:
        """
        语义分块：尝试在句子边界处分割

        Args:
            text: 文档文本
            metadata: 文档元数据
        Returns:
            List[Document]: 分块列表
        """
        # 先按句子分割
        sentences = re.split(r'([。！？\n])', text)

        # 重组句子（保留分隔符）
        full_sentences = []
        current_sentence = ""
        for i, part in enumerate(sentences):
            current_sentence += part
            if i % 2 == 1:  # 分隔符
                full_sentences.append(current_sentence)
                current_sentence = ""
        if current_sentence:
            full_sentences.append(current_sentence)

        # 组合句子成块
        chunks = []
        current_chunk = ""

        for sentence in full_sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={**metadata, 'chunk_type': 'semantic'}
                    ))
                current_chunk = sentence
            else:
                current_chunk += sentence

        if current_chunk:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={**metadata, 'chunk_type': 'semantic'}
            ))

        return chunks