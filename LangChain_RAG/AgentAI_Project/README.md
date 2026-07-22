# 智扫通机器人智能客服

基于 **LangChain Agent + RAG + Streamlit** 的扫地机器人智能客服。Agent 会根据问题自主调用知识库、天气、用户信息和使用记录等工具，也可以动态切换提示词生成用户使用报告。

## 主要功能

- 基于本地文档回答产品咨询、故障排除和维护问题
- 使用 Chroma 持久化存储文档向量
- Agent 自主选择并调用多个工具
- Middleware 记录模型与工具调用过程
- 根据任务动态切换普通客服和报告生成提示词
- Streamlit 流式聊天界面

## 工作流程

```text
用户问题
  ↓
LangChain Agent 判断任务
  ├─ 知识问答 → RAG 检索 → 模型总结
  ├─ 天气咨询 → 天气与位置工具
  └─ 使用报告 → 用户记录工具 → 动态切换报告提示词
  ↓
流式返回结果
```

## 项目结构

```text
AgentAI_Project/
├── app.py                  # Streamlit 应用入口
├── agent/
│   ├── react_agent.py      # Agent 创建与流式执行
│   └── tools/
│       ├── agent_tools.py  # Agent 工具
│       └── middleware.py   # 日志监控与动态提示词
├── rag/
│   ├── rag_service.py      # 检索与总结服务
│   └── vector_store.py     # 文档切分、入库与检索
├── model/factory.py        # 对话模型与 Embedding 模型
├── config/                 # 模型、向量库和路径配置
├── prompts/                # 系统、RAG 和报告提示词
├── data/                   # 知识库与模拟用户数据
├── utils/                  # 配置、文件、日志等工具
├── chroma_db/              # Chroma 持久化数据
└── logs/                   # 运行日志
```

## 环境准备

建议使用 Python 3.10 及以上版本。

```bash
pip install streamlit langchain langchain-community langchain-chroma \
  langchain-text-splitters dashscope pyyaml pypdf
```

项目使用阿里云百炼的通义模型，需要设置 API Key：

```bash
export DASHSCOPE_API_KEY="你的 API Key"
```

默认模型配置位于 `config/rag.yml`：

```yaml
chat_model_name: qwen3-max
embedding_model_name: text-embedding-v4
```

## 运行项目

所有命令均在 `AgentAI_Project` 目录下执行。

### 1. 构建知识库

将 `.txt` 或 `.pdf` 文件放入 `data/`，然后执行：

```bash
python -m rag.vector_store
```

程序会切分文档并写入 Chroma；`md5.text` 用于避免重复导入相同文件。

### 2. 启动客服界面

```bash
streamlit run app.py
```

打开终端显示的本地地址，即可进行对话。

## 核心工具

| 工具 | 作用 |
| --- | --- |
| `rag_summarize` | 检索知识库并生成回答 |
| `get_weather` | 获取指定城市天气 |
| `get_user_location` | 获取用户所在城市 |
| `get_user_id` | 获取用户 ID |
| `get_current_month` | 获取当前月份 |
| `fetch_external_data` | 获取用户月度使用记录 |
| `fill_context_for_report` | 标记报告场景并触发提示词切换 |

当前天气、位置、用户和月份数据均为模拟数据，仅用于演示 Agent 的工具调用流程。

## 配置说明

- `config/rag.yml`：对话模型与 Embedding 模型
- `config/chroma.yml`：向量库路径、分片参数和召回数量
- `config/prompts.yml`：提示词文件路径
- `config/agent.yml`：外部用户数据路径

修改配置后重新启动应用即可生效；修改知识库文档后需要再次执行入库命令。
