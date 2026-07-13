# OpenAI 流式输出

## 概述
可以设定结果为stream模式，获得更好的使用体验，避免长时间等待完整响应。

## 基本使用
1. 开始流式输出的方法：`client.chat.completions.create()` 调用模型时设定参数 `stream=True`
2. 通过 for 循环遍历 response 对象，在循环中实时输出内容

## 示例代码

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "写一首关于春天的诗"}],
    stream=True  # 启用流式输出
)

# 逐块输出内容
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 关键说明

| 参数 | 说明 |
|------|------|
| `stream=True` | 启用流式输出模式 |
| `chunk.choices[0].delta.content` | 每个流式块中的文本内容 |
| `flush=True` | 立即输出缓冲区内容 |
| `end=" "` | 每一段之间以空格分割 |

## 优势
- 实时显示响应内容，提升用户体验
- 适用于长文本生成场景
- 可提前终止接收（break 循环）

## 注意事项
- 流式输出无法使用 `usage` 字段获取token消耗信息
- 需要手动处理文本拼接（如需要完整内容）