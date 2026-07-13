# LangChain 调用笔记

LangChain 的作用是把不同大模型、Embedding 模型、提示词、检索器等组件统一成一套调用方式。

## 1. 两种模型调用方式

### 云端 API

云端模型通过服务商 API 调用，例如通义千问、OpenAI、Gemini 等。优点是效果稳定、无需本地显卡；缺点是需要 API Key，并且会产生调用费用。

### 本地 Ollama

本地模型通过 Ollama 调用，例如 `qwen3:8b`。优点是数据不出本机、适合练习；缺点是效果和速度受本地硬件影响。


## 2. 聊天消息 Messages

聊天模型不是只接收一段字符串，也可以接收一组“带身份的消息”。常见角色有三种：

| 类型 | 含义 | 例子 |
| --- | --- | --- |
| `SystemMessage` / `system` | 设定模型身份和规则 | 你是一个 Python 老师 |
| `HumanMessage` / `human` | 用户输入 | 请解释列表推导式 |
| `AIMessage` / `ai` | AI 之前的回复 | 列表推导式用于快速生成列表 |



重点：`system` 控制整体风格和规则，`human` 是当前用户问题，`ai` 可以把历史回答传回去，让模型知道上下文。

## 3. PromptTemplate：静态提示词 vs 动态提示词

普通字符串是静态提示词，内容写死；`PromptTemplate` 或 `ChatPromptTemplate` 可以用 `{变量}` 占位，运行时再注入具体内容。


## 4. 流式输出 Stream

普通 `invoke()` 会等模型完整生成后一次性返回；`stream()` 会边生成边返回，适合聊天界面。


如果使用的是聊天模型，流式返回的可能是 `AIMessageChunk`，通常需要取其中的文本字段；如果使用的是传统 LLM，可能直接返回字符串。实际打印方式要看具体模型封装。

## 5. Embedding Models

Embedding 模型的作用是把文本变成向量，也就是一串浮点数。向量可以用于相似度计算、语义搜索和 RAG 检索。

## 6. 通用Prompt模板

提示词优化在模型应用中非常重要，LangChain中提供了一个PromptTemplate的类，用来协助优化提示词。
- 举个例子来说，你构建了一个通用的提示词模板，里面有一些内容是变量，你可以进行变量的注入，更具不同的变量生成不同的提示词
- 有[PromptTemplate](../Codes/08_langchain的通用提示词模板.py)，[FewShotPromptTemplate](../Codes/09_langchain的通用提示词模板few_shot%20copy.py), [ChatPromptTemplate](../Codes/10_langchain的通用提示词模板Chat.py),具体请参考代码示例
- 模板类都可以调用`format`和`invoke`方法

### PromptTemplate（提示词模板）简要总结

* **Prompt 优化对大模型应用非常重要**，LangChain 提供 `PromptTemplate` 类帮助管理和复用提示词。
* PromptTemplate 可以创建**带变量的通用提示词模板**，通过变量注入，根据不同输入生成不同 Prompt。
* 例如：

```python
template = "你是一名{role}，请回答：{question}"
```

传入不同变量即可生成不同提示词。

---

### PromptTemplate 常用方法

| 方法         | 作用                                            |
| ---------- | --------------------------------------------- |
| `format()` | 填充模板变量，返回普通字符串                                |
| `invoke()` | 使用 LangChain Runnable 调用方式填充变量，返回 PromptValue |

### format 和 invoke 的相同点

* 都可以根据输入变量填充 PromptTemplate 中的占位符。
* 都会生成最终的提示词内容。
* 都支持传入字典形式的变量。


### format 和 invoke 的区别

| 对比   | format()       | invoke()                  |
| ---- | -------------- | ------------------------- |
| 返回类型 | 普通字符串 (`str`)  | LangChain 的 `PromptValue` |
| 使用场景 | 查看或手动处理 Prompt | 推荐用于 LangChain Chain 调用   |
| 接入模型 | 需要手动传入 LLM     | 可以直接连接后续 Runnable         |
| 调用方式 | Python 字符串格式化  | LangChain 标准调用接口          |

