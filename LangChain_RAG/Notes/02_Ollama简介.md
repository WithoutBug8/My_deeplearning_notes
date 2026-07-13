# Ollama 简介

## 什么是 Ollama？

Ollama 是一个让你在**本地**运行大语言模型（LLM）的工具。只需一条命令，就能下载并运行 Llama、Mistral、Qwen 等开源模型，无需 GPU 也能跑（使用 CPU 推理）。

核心价值：**把 LLM 从云端搬到本地**，数据不出本机。

## 为什么需要 Ollama？

蒸馏模型是大模型的学生，标准的大模型在本机是跑不起来的——参数量动辄几百 GB，消费级硬件根本无法加载。Ollama 解决的就是这个问题：

| 问题 | Ollama 的方案 |
|------|--------------|
| 模型太大，显存放不下 | 使用 GGUF 量化格式，把模型压缩到 4-bit / 8-bit，精度换空间 |
| 部署复杂，环境难配 | 一键安装，内置推理引擎（llama.cpp），自动管理依赖 |
| 数据隐私担忧 | 全部在本地运行，API Key、对话数据不出本机 |
| 想用开源模型但不方便 | 内置模型库，`ollama pull` 即下即用 |

## 核心概念

### 1. 量化（Quantization）

把模型的浮点参数（FP16/FP32）压缩成低精度整数（如 4-bit），大幅减少体积和显存占用。

```
FP16 模型: 参数 × 2 字节 → 70B 模型 ≈ 140 GB
Q4 模型:   参数 × 0.5 字节 → 70B 模型 ≈ 35 GB
```

代价是轻微的精度损失，但对大多数场景影响很小。

### 2. 蒸馏模型（Distilled Model）

蒸馏模型是大模型的学生——用大模型（教师）的输出去训练小模型（学生），让小模型学会大模型的"思维方式"。小模型参数少、跑得快，但效果接近大模型。Ollama 支持的很多模型（如 `qwen2.5:0.5b`）就是蒸馏产物。

### 3. GGUF 格式

GGUF（GPT-Generated Unified Format）是 llama.cpp 项目的模型文件格式，Ollama 底层就用的它。特点：
- 单文件分发，包含权重 + 配置 + tokenizer
- 支持多种量化级别（Q2 ~ Q8）
- 可在 CPU 上高效推理

## 安装

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
# 或从 https://ollama.com 下载 .app
```

安装后 Ollama 以后台服务运行，默认监听 `localhost:11434`。

## 常用命令

```bash
# 下载并运行模型（没有会先自动下载）
ollama run llama3.2

# 只下载模型，不运行
ollama pull qwen2.5:7b

# 查看已下载的模型
ollama list

# 查看模型详情（参数量、量化级别、大小等）
ollama show qwen2.5:7b

# 删除模型
ollama rm qwen2.5:7b

# 查看运行状态
ollama ps

# 停止正在运行的模型
ollama stop qwen2.5:7b
```

### 模型标签说明

```bash
ollama pull qwen2.5:7b        # 7B 参数版本（默认 Q4_K_M 量化）
ollama pull qwen2.5:0.5b      # 0.5B 蒸馏小模型
ollama pull llama3.2:latest   # 最新版
```

## 对话示例

```bash
$ ollama run qwen2.5:0.5b
>>> 用一句话介绍 Python
Python 是一种简洁易学、功能强大的高级编程语言。

>>> /bye
```

### 对话内命令

| 命令 | 作用 |
|------|------|
| `/bye` | 退出对话 |
| `/show` | 查看当前模型信息 |
| `/load <model>` | 切换模型 |

## REST API

Ollama 启动后提供 HTTP API，其他程序可以通过 API 调用：

### 生成（单轮）

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:0.5b",
  "prompt": "为什么天空是蓝色的？",
  "stream": false
}'
```

### 聊天（多轮）

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:0.5b",
  "messages": [
    {"role": "user", "content": "你好，请介绍一下自己"}
  ],
  "stream": false
}'
```

## 在 LangChain 中使用 Ollama

这是本系列的核心——用 Ollama 提供的本地模型替代 OpenAI 等云端 API：

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# 连接本地 Ollama 模型
llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.7,
)

response = llm.invoke([HumanMessage(content="你好")])
print(response.content)
```

用 Ollama 做 Embedding（用于 RAG 的向量检索）：

```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("这是一段待检索的文本")
```

## 推荐模型

| 模型 | 适用场景 | 大小 |
|------|----------|------|
| `qwen2.5:7b` | 中文对话、通用任务（推荐） | ~4.7 GB |
| `qwen2.5:0.5b` | 轻量测试、快速验证 | ~0.4 GB |
| `llama3.2` | 英文对话、代码生成 | ~2.0 GB |
| `nomic-embed-text` | 文本向量化（Embedding），用于 RAG 检索 | ~0.3 GB |
| `mistral` | 英文推理、逻辑任务 | ~4.1 GB |
| `deepseek-r1:7b` | 深度推理、数学/编程 | ~4.7 GB |

## 自定义模型：Modelfile

可以通过 Modelfile 自定义系统提示词或微调参数：

```dockerfile
# Modelfile
FROM qwen2.5:7b

# 设置系统提示词
SYSTEM "你是一个专业的 Python 编程助手，回答简洁、准确。"

# 调整推理参数
PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

```bash
# 创建自定义模型
ollama create my-coder -f Modelfile

# 使用自定义模型
ollama run my-coder
```


