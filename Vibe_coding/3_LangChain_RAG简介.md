# LangChain RAG 简介

## 1. 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合**信息检索**与**大语言模型生成**的技术，核心目的是：

- 向模型提供**私有知识/外部知识**，弥补模型训练数据的时效性和覆盖面不足
- 给出模型**可溯源的参考资料**，有效减少模型幻觉（hallucination）
- 无需微调模型，即可让模型"知道"训练截止后的新知识或企业内部文档

整体分为两个阶段：**离线处理** 和 **在线处理**。

---

## 2. 离线处理（知识入库）

离线阶段负责将私有知识文档预先处理并存入向量数据库。

### 2.1 文档加载（Document Loaders）
- 从多种来源加载文档：PDF、Markdown、Word、网页、数据库、Notion 等
- LangChain 提供了丰富的 Document Loader（如 `PyPDFLoader`、`TextLoader`、`WebBaseLoader`）

### 2.2 文本分割（Text Splitting）
- 将长文档按语义边界切分为较小的**文本块（chunk）**
- 常见策略：
  - 按字符数/Token 数固定大小切分（`CharacterTextSplitter`、`RecursiveCharacterTextSplitter`）
  - 保留重叠区域（chunk_overlap），避免关键信息被切断
  - 按句子、段落等语义边界切分，保持语义完整性

### 2.3 向量嵌入（Embedding）
- 将每个 chunk 通过嵌入模型（如 OpenAI `text-embedding-ada-002`、HuggingFace 模型、本地模型等）转换为**向量（embedding）**
- 语义相近的文本，其向量在空间中距离也相近

### 2.4 向量存储（Vector Store）
- 将 embedding 连同原始文本（或元数据）一起存入向量数据库
- 常见向量库：Chroma、FAISS、Pinecone、Weaviate、Milvus、PGVector 等
- 支持高效的**近似最近邻搜索（ANN）**，在大规模向量中快速找到最相似的 top-k 结果

---

## 3. 在线处理（问答流程）

当用户提问时，在线阶段实时执行以下流程：

### 3.1 用户问题向量化
- 使用**与离线阶段相同的嵌入模型**将用户问题转为向量

### 3.2 相似性检索（Retrieval）
- 在向量库中执行相似度匹配（通常用余弦相似度或欧氏距离）
- 返回与问题最相关的 top-k 个文档 chunk

### 3.3 组装提示词（Prompt Assembly）
- 将检索到的参考资料和用户原始问题拼装成一个增强版 prompt
- 典型模板：

  ```
  请根据以下参考资料回答用户问题。如果参考资料中没有相关信息，请如实说明。

  参考资料：
  {context}

  用户问题：
  {question}
  ```

### 3.4 大模型生成（Generation）
- 将组装好的 prompt 发送给 LLM（如 GPT-4、Claude 等）
- LLM 基于参考资料生成答案，而非仅凭内部知识"编造"

---

## 4. LangChain 中的典型 RAG 链路

```
文档加载 → 文本分割 → 嵌入向量化 → 存入向量库（离线）
                                              ↓
用户提问 → 问题向量化 → 向量检索 → 拼装 prompt → LLM 生成答案（在线）
```

核心组件对应：
| 步骤           | LangChain 组件              |
| -------------- | --------------------------- |
| 文档加载       | `DocumentLoader`            |
| 文本分割       | `TextSplitter`              |
| 嵌入           | `Embeddings`                |
| 向量存储/检索  | `VectorStore`               |
| 组装+生成      | `RetrievalQA` / `Chain`     |

---

## 5. 进阶话题

- **多路检索（Hybrid Search）**：结合向量检索 + 关键词检索（BM25），兼顾语义和精确匹配
- **重排序（Re-ranking）**：对初步检索结果用更强的模型重新排序，提升相关性
- **对话式 RAG**：将历史对话上下文纳入检索和生成，支持多轮问答
- **Agentic RAG**：让 Agent 自主决定何时检索、检索什么、如何利用检索结果
