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




## 注意
很多包已经被废弃了，如果代码运行不起来的话，属于正常现象
