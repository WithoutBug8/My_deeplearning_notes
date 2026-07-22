# LangChain 与 RAG 学习说明文档

这里用于记录学习 LangChain 和 RAG（Retrieval-Augmented Generation）的笔记、示例和实践经验。

---

### RAG（检索增强生成）

LLM 的知识截止于训练数据的时间点，且无法访问企业内部文档、私有数据库或实时信息。RAG 的思路是：**先检索相关信息，再让模型基于这些信息生成答案**，把 LLM 从"闭卷考试"变成"开卷考试"。

典型应用场景：
- **企业知识库问答**：让 AI 能回答公司内部文档、规章制度、产品手册中的问题
- **客服系统**：基于产品文档和历史工单自动回复用户
- **法律/医疗辅助**：在海量法条或医学文献中找到相关条目再给出建议
- **研究助手**：对学术论文库进行检索式问答

### LangChain

LangChain 是一个 LLM 应用开发框架，核心思路是把 LLM 与外部工具、数据源、记忆等**链接（Chain）**起来。它提供了一套标准化的抽象（Chains、Agents、Tools、Memory 等），让开发者不必从零写胶水代码。

典型能力：
- **多步推理链**：把复杂任务拆成多个 LLM 调用步骤
- **Agent 智能体**：让 LLM 自主决定调用什么工具、按什么顺序解决问题
- **Memory 对话记忆**：管理多轮对话的上下文
- **工具集成**：对接搜索引擎、数据库、API、代码解释器等

---

## 目录

### 学习笔记

1. [环境变量保护 API Key](./Notes/01_环境变量保护APIKEY.md)
2. [Ollama 简介](./Notes/02_Ollama简介.md)
3. [OpenAI 库的调用](./Notes/03_OpenAI库的调用.md)
4. [OpenAI 流式输出](./Notes/04_OpenAI流式输出.md)
5. [OpenAI 调用附带历史消息](./Notes/05_OpenAI调用附带历史消息.md)
6. [大模型提示词工程指南](./Notes/06_大模型提示词工程指南.md)
7. [JSON 数据格式](./Notes/07_JSON数据格式.md)
8. [LangChain 与 RAG 简介](./Notes/08_LangChain&RAG简介.md)
9. [RAG 中的向量和余弦相似度](./Notes/09_RAG中的向量和余弦相似度.md)
10. [LangChain 调用](./Notes/10_LangChain调用.md)
11. [Chain 链](./Notes/11_Chain链.md)
12. [Memory 历史会话记忆管理](./Notes/12_Memory历史会话记忆管理.md)
13. [向量存储技术](./Notes/13_向量存储技术.md)
14. [Agent 智能体](./Notes/14_Agent智能体.md)

### 示例代码

#### 模型与提示词

1. [OpenAI 库的调用](./Codes/01_OpenAI库的调用.py)
2. [提示词工程实战](./Codes/02_提示词工程实战.ipynb)
3. [LangChain 调用 API 模型](./Codes/03_langChain调用API方式.py)
4. [LangChain 调用本地模型](./Codes/04_langChain调用本地模型.py)
5. [LangChain 流式输出](./Codes/05_langchain的流式输出.py)
6. [LangChain 聊天模型](./Codes/06_langchain的聊天模型.py)
7. [LangChain 文本嵌入模型](./Codes/07_langchain的文本嵌入模型.py)
8. [通用提示词模板](./Codes/08_langchain的通用提示词模板.py)
9. [Few-shot 提示词模板](./Codes/09_langchain的通用提示词模板few_shot.py)
10. [Chat 提示词模板](./Codes/10_langchain的通用提示词模板Chat.py)

#### Chain 与记忆

11. [Chain 基础使用](./Codes/11_Chain的基础使用.py)
12. [StrOutputParser 解析器](./Codes/12_StrOutputParser解析器.py)
13. [JSONOutputParser 解析器](./Codes/13_JSONOutputParser解析器.py)
14. [RunnableLambda 基础使用](./Codes/14_RunnableLambda的基础使用.py)
15. [临时会话记忆](./Codes/15_临时会话记忆.py)

#### 文档加载与 RAG

16. [外部向量持久化存储](./Codes/16_外部向量持久化存储.py)
17. [Document Loaders 文档加载器](./Codes/17_Document_loaders文档加载器.py)
18. [JSONLoader 文档加载器](./Codes/18_JSONLoader文档加载器.py)
19. [文档加载器和分割器](./Codes/19_文档加载器和分割器.py)
20. [PyPDFLoader 的使用](./Codes/20_PyPDFLoader的使用.py)
21. [内存向量存储](./Codes/21_内存向量存储.py)
22. [内存向量的外部持久化](./Codes/22_内存向量的外部持久化存储.py)
23. [向量检索构建提示词](./Codes/22_向量检索构建提示词.py)
24. [RunnablePassthrough 的使用](./Codes/23_RunnablePassThrough的使用.py)

#### Agent 智能体

25. [Agent 智能体初体验](./Codes/24_Agent智能体初体验.py)
26. [Agent 流式输出](./Codes/25_Agent的流式输出.py)
27. [ReAct 框架](./Codes/26_ReAct框架.py)
28. [Middleware 中间件](./Codes/27_middleware中间件.py)

### 实践项目

- [RAG 知识库问答项目](./RAG_Project/Readme.md)：包含文档入库、向量检索、会话历史和问答应用。
- [Agent + RAG 项目](./AgentAI_Project/README.md)：结合智能体、工具调用和 RAG 的完整示例。

## 注意

部分示例使用的包或接口可能已经废弃，请根据当前 LangChain 版本调整。
