# OpenAI 调用附带历史消息

## 为什么需要传递历史消息？

将历史消息填入请求，让模型更好地知晓对话上下文，实现真正的多轮对话。

## 消息结构

OpenAI API 的 messages 参数是一个消息列表，每条消息包含：

| 字段 | 说明 |
|------|------|
| `role` | 消息角色：`system`、`user`、`assistant`、`tool` |
| `content` | 消息内容（字符串） |

## 简单示例

### Python 示例

```python
from openai import OpenAI

client = OpenAI()

# 第一轮对话
messages = [
    {
        "role": "system",
        "content": "你是一个友善的AI助手，乐于帮助用户解决问题。"
    },
    {
        "role": "user",
        "content": "你好！"
    }
]

response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)
assistant_reply_1 = response1.choices[0].message.content

# 将助手的回复添加到历史消息中
messages.append({
    "role": "assistant",
    "content": assistant_reply_1
})

# 第二轮对话（基于历史上下文）
messages.append({
    "role": "user",
    "content": "什么是Python？"
})

response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

print(response2.choices[0].message.content)
```

### 该方法的局限性
当前的消息是存放在messages的list中，只能修改一次代码，运行一次结果；
在生产系统中需要把消息存放在数据库等持久化工具中，需要时按需取用，在Langchain中会有`长期记忆`和`短期记忆`的使用方法