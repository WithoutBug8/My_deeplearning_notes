# 环境变量保护 API Key

## 为什么要这样做？

API Key 属于敏感信息，明文写在代码里会带来严重的安全隐患：
- 代码一旦提交到 Git 等版本控制系统，Key 就会永久暴露在历史记录中
- 他人可以轻易获取你的 Key，造成盗用或账单损失
- 更换 Key 后需要修改所有硬编码的地方，维护成本高

## 如何使用环境变量隐藏 API Key？

### 1. 设置环境变量

在 `~/.zshrc`（或 `~/.bashrc`）中添加：

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
```

然后执行 `source ~/.zshrc` 使其立即生效。

### 2. 在代码中读取环境变量

Python 示例：

```python
import os

api_key = os.getenv("OPENAI_API_KEY")
# 或
api_key = os.environ["OPENAI_API_KEY"]
```

### 3. 确保 `.env` 或配置文件不被提交

在 `.gitignore` 中添加：

```
.env
```

## 总结

| 方式 | 安全性 | 推荐 |
|------|--------|------|
| 明文写在代码里 | ❌ 极差 | 绝对不推荐 |
| 环境变量 | ✅ 安全 | 推荐 |
| `.env` 文件 + `.gitignore` | ✅ 安全 | 团队协作推荐 |
