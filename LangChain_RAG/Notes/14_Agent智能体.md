# LangChain Agent 智能体

Agent 是由大模型驱动的任务执行器。它会根据用户目标自主选择工具，并结合工具返回结果继续决策，直到得到最终答案。

## 1. Agent 与 Chain 的区别

| Chain | Agent |
| --- | --- |
| 执行流程由代码预先确定 | 执行流程由模型动态决定 |
| 适合固定、标准化任务 | 适合复杂、多步骤任务 |
| 不会自主选择工具 | 可以自主选择和多次调用工具 |

## 2. 基本用法

使用 `@tool` 定义工具，再通过 `create_agent()` 创建智能体：

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool(description="查询指定城市的天气")
def get_weather(city: str) -> str:
    return f"{city}今天晴天"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个聊天助手，请根据需要调用工具。",
)

res = agent.invoke({
    "messages": [{"role": "user", "content": "深圳天气如何？"}]
})

print(res["messages"][-1].content)
```

工具名称、参数类型和描述要清楚，因为模型会据此判断何时调用工具以及传入什么参数。

## 3. 流式输出

`stream()` 可以逐步返回 Agent 的消息和工具调用过程：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "深圳天气如何？"}]},
    stream_mode="values",
):
    latest_message = chunk["messages"][-1]
    latest_message.pretty_print()
```

## 4. ReAct 框架

ReAct 表示 **Reasoning + Acting**，核心循环是：

```text
分析任务 → 调用工具 → 观察结果 → 继续决策 → 最终回答
```

例如计算 BMI 时，Agent 可以先获取身高，再获取体重，最后根据两个工具的结果完成计算。

通常不需要在提示词中强制模型输出完整思考过程；展示工具调用和最终结论即可。

## 5. Middleware 中间件

Middleware 通过 Hook 拦截 Agent 的执行过程，常用于日志、监控、权限校验、重试和修改请求。

| Hook | 执行时机 |
| --- | --- |
| `before_agent` / `after_agent` | Agent 开始前 / 结束后 |
| `before_model` / `after_model` | 每次调用模型前 / 后 |
| `wrap_model_call` | 包裹并控制模型调用 |
| `wrap_tool_call` | 包裹并控制工具调用 |

```python
from langchain.agents.middleware import before_agent, wrap_tool_call

@before_agent
def log_start(state, runtime):
    print(f"Agent 启动，共 {len(state['messages'])} 条消息")

@wrap_tool_call
def monitor_tool(request, handler):
    print("调用工具：", request.tool_call["name"])
    return handler(request)  # 继续执行原工具

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[log_start, monitor_tool],
)
```

普通 Hook 适合在调用前后执行逻辑；`wrap_*` 可以决定是否继续调用，并可修改请求、结果或处理异常。


