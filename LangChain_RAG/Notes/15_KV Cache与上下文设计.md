# KV Cache与上下文设计

## 1. 什么是 KV Cache？

### 1.1 Transformer 自回归生成的基本过程

Transformer 模型在推理时是**自回归**的：每次只生成一个 token，然后将新生成的 token 拼接到输入序列末尾，再送入模型生成下一个 token。

```
Step 1: [A, B, C] → 模型 → token D
Step 2: [A, B, C, D] → 模型 → token E
Step 3: [A, B, C, D, E] → 模型 → token F
...
```

每一步都需要对整个序列重新计算注意力（Attention），这会导致 $O(n^2)$ 的计算复杂度，随着序列变长，推理成本急剧上升。

### 1.2 KV Cache 的核心思想

在 Transformer 的注意力机制中，每个 token 会计算出三个向量：

- **Q（Query，查询）**：当前 token 想去"查"什么信息
- **K（Key，键）**：当前 token "拥有"什么信息标签
- **V（Value，值）**：当前 token 实际携带的信息内容

注意力计算的公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

关键洞察：**已生成 token 的 K 和 V 在后续步骤中不会改变**。因为 Transformer 使用因果掩码（causal mask），每个 token 只能看到它前面的 token，所以之前 token 的 K、V 计算只依赖于它们自身和更早的 token。

因此，我们可以在第一次计算后把 K、V **缓存**起来：

```
Step 1: [A, B, C] → 计算并缓存 K_A,V_A, K_B,V_B, K_C,V_C → token D
Step 2: 复用缓存的 K_A..K_C, V_A..V_C，只计算 K_D, V_D → token E  ← 大幅减少计算
Step 3: 复用缓存的 K_A..K_D, V_A..V_D，只计算 K_E, V_E → token F
...
```

KV Cache 就是这个缓存的 **Key 和 Value 矩阵**。有了它，每步只需计算新 token 的 Q、K、V，然后拿新 Q 去和所有历史 K 做注意力即可。

### 1.3 KV Cache 的内存占用

KV Cache 的大小是推理部署的核心瓶颈之一：

$$\text{KV Cache 大小} = 2 \times \text{层数} \times \text{头数} \times \text{每头维度} \times \text{序列长度} \times \text{精度字节数}$$

以 Llama 2 70B 为例（层数 80，头数 64，每头维度 128，FP16 精度）：
- 每个 token 的 KV Cache 约 1.25 MB
- 2048 个 token 约 2.5 GB
- 这就是为什么长上下文推理需要大显存显卡

---

## 2. Prompt Caching（提示缓存）

### 2.1 是什么？

Prompt Caching 是 KV Cache 思想在 API 层面的应用。多家 LLM 服务商（Anthropic、OpenAI 等）都提供了类似功能：

**核心机制**：服务端把 Prompt 前缀的 KV Cache 保留一段时间（通常 5 分钟），后续请求如果前缀相同，就不需要重新计算这部分。

### 2.2 为什么能加速？

```
请求 1: [System Prompt + Tool Defs + User Msg 1] → 服务端缓存 KV
请求 2: [System Prompt + Tool Defs + ...]          → 命中缓存，免费/快速
```

第一次请求：完整计算，产生完整 KV Cache。
后续请求（前缀命中）：只计算新增部分的 KV，已有部分直接复用。

### 2.3 各家方案对比

| 服务商 | 缓存粒度 | 最低缓存长度 | 缓存 TTL | 命中价格 |
|--------|---------|-------------|---------|---------|
| Anthropic | 前缀自动缓存 | 1024 tokens (Claude 3.5) / 2048 tokens (Claude 5) | ~5 分钟 | 写入价格 1.25×，读取价格 0.1× |
| OpenAI | 前缀自动缓存 | 1024 tokens | 5-10 分钟 | 读取 50% 折扣 |
| DeepSeek | 前缀自动缓存 | 前 10% 不缓存 | 5 分钟 | 读取 50% 折扣 |
| Google Gemini | 上下文缓存 API | 32,000 tokens | 由 TTL 参数控制 | 大幅折扣 |

Anthropic 的方案在价格设计上很有特点：写入时多收 25%，命中时只收 10%，鼓励把稳定内容放在前面。

---

## 3. KV Cache 友好的上下文设计原则

这些原则直接源于 Prompt Caching 的工作机制——缓存命中依赖于**前缀匹配**。

### 3.1 🔒 系统提示词和工具定义一旦确定就不要修改

这是最重要的原则。系统提示词和工具定义通常是长且稳定的内容，放在上下文最前面。如果每次请求都微调系统提示词，缓存永远无法命中。

**实践建议**：
- 系统提示词写成模板，动态部分通过变量插入到末尾
- 工具定义的 schema 不要按需删减，始终发送完整集合
- 如果需要"动态系统指令"，把它们作为 User 消息追加到对话末尾，而不是修改 System Prompt

### 3.2 📌 动态信息永远要增加到末尾

缓存从前往后匹配。任何插入到中间的内容都会导致之后的所有内容缓存失效：

```
❌ 错误做法（缓存全部失效）：
[System Prompt] [User Msg 1] [新插入的上下文] [User Msg 2] [User Msg 3]
                   ↑ 这里之前可以缓存    ↑ 插入导致后续全部重算

✅ 正确做法（只有新增部分需要计算）：
[System Prompt] [User Msg 1] [User Msg 2] [User Msg 3] [新插入的上下文]
                   ↑ 全部命中缓存                       ↑ 只算这部分
```

**实践建议**：
- RAG 检索到的文档放在消息列表末尾
- 新对话轮次追加到末尾
- 少样本示例（few-shot examples）考虑：如果它们很长且稳定，放在 System Prompt 中靠前；如果频繁变化，接受缓存策略（放在末尾）

### 3.3 📐 使用标准的 API 格式，请勿自己修改拼接信息

LLM API 的消息格式（System/User/Assistant/Tool）不仅仅是语义上的区分，也影响服务端如何切分和管理缓存。自行拼接或转换格式可能破坏缓存逻辑。

**实践建议**：
- 使用 SDK 的原生消息格式（如 LangChain 的 `SystemMessage`、`HumanMessage`、`AIMessage`）
- 不要自行把多轮对话拼接成单条消息
- 如果使用第三方框架，确认它的消息序列化方式与服务端期望一致

### 3.4 📊 额外的实用技巧

**长前缀起步**：Anthropic Claude 5 要求前缀至少 2048 tokens 才会被缓存。如果系统提示词较短，可以考虑加入一些稳定的参考文档或格式说明来达到这个阈值。

**缓存断点（Cache Breakpoints）**：Anthropic 支持手动指定缓存断点（最多 4 个），允许在 4 个位置分别设置断点。例如：

```
[System Prompt (cache)]
--- breakpoint ---
[Tool Results (cache)]
--- breakpoint ---
[Latest User Message (不缓存)]
```

**监控缓存命中率**：在 API 响应的 `usage` 字段中查看 `cache_creation_input_tokens` 和 `cache_read_input_tokens`，了解缓存效果。

**少样本示例的位置权衡**：
- 如果示例很长且稳定 → 放在 System Prompt 中（放在前面，稳定缓存）
- 如果示例频繁变化 → 接受放在末尾（首次需重算，后续几轮命中）
- 如果是少量示例且工具定义已很长 → 放在 User 消息末尾（微调时不影响工具定义缓存）

---

## 4. 为什么这些原则重要？

### 4.1 成本

以 Claude Opus 5 为例（输入 $15/MTok，缓存写入 $18.75/MTok，缓存读取 $1.5/MTok）：

| 场景 | 系统提示词 | 对话历史 | 新消息 | 缓存命中情况 | 大致成本 |
|------|-----------|---------|--------|-------------|---------|
| 没有缓存 | 全额 | 全额 | 全额 | 0% | 基准 |
| 前缀命中 | 1.5 (10%) | 1.5 (10%) | 15 (100%) | 系统+历史 | ~1/7 |
| 中间插入 | 15 (100%) | 15 (100%) | 15 (100%) | 0% | 全额 |

缓存命中的价格只有原始价格的 **1/10**。

### 4.2 延迟

不需要重新计算 KV Cache 的 token 直接跳过矩阵运算，推理速度显著提升。对于长系统提示词（如包含大量工具定义的 Agent），延迟差异可达数倍。

### 4.3 内存

服务端内存也是有限的。良好的缓存命中意味着服务端可以用更少的 GPU 内存服务更多的并发请求。

---

## 5. KV Cache 的局限与发展

### 5.1 局限

- **严格的顺序依赖**：必须前缀匹配，顺序稍有不同即失效
- **TTL 限制**：缓存通常只在 5 分钟内有效
- **服务端不保证**：缓存是 best-effort，高负载时可能被驱逐
- **工具调用格式**：不同框架对工具调用 results 的序列化方式不同，可能影响缓存命中

### 5.2 发展方向

- **Multi-Query Attention (MQA)** 和 **Grouped-Query Attention (GQA)**：减少 K、V 头数，直接缩小 KV Cache 体积
- **KV Cache 量化**：对缓存的 K、V 进行低精度量化（如 KIVI、KVQuant）
- **滑动窗口注意力**：只保留最近的 N 个 token 的 KV Cache（如 Mistral）
- **多层共享 KV**：跨层复用 KV（如 MiniMax-01 的 Lightning Attention）

---

## 6. 总结

> **KV Cache** 是 Transformer 推理优化的基石。理解它的工作机制后，上下文设计就很简单：
>
> **把确定的内容放前面，把变化的内容放最后。**

| 原则 | 原因 | 实践 |
|------|------|------|
| 固定内容在前 | 前缀命中缓存 | 系统提示词、工具定义固定 |
| 动态内容在后 | 避免打断缓存链 | RAG 文档、新用户消息追加到末尾 |
| 标准 API 格式 | 服务端缓存逻辑依赖 | 使用 SDK，不自行拼接 |
| 监控命中率 | 验证优化效果 | 查看 `cache_read_input_tokens` |