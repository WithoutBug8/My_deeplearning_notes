# Hooks 拦截工具的调用

## 1. 核心概念

Hooks 是一种**拦截机制**：在特定事件发生前后自动执行脚本，通过返回值决定**放行**还是**阻止**。零信任原则：不满足条件就阻止。

---

## 2. Git Hooks（理解 Hook 模式的最佳入口）

### 2.1 常见类型

| Hook | 触发时机 | 典型用途 |
|------|---------|---------|
| `pre-commit` | commit 输 message 之前 | 代码检查、测试 |
| `commit-msg` | 输完 message 之后 | 校验 commit message 格式 |
| `pre-push` | push 之前 | 集成测试 |

Hooks 位置：`.git/hooks/`，或通过 `git config core.hooksPath` 自定义。

### 2.2 示例：pre-commit 跑测试

```bash
#!/bin/bash
# .git/hooks/pre-commit

npx eslint src/ --max-warnings 0 || { echo "❌ ESLint 未通过"; exit 1; }
npm test || { echo "❌ 测试未通过"; exit 1; }
echo "✅ 通过，允许 commit"
```

```bash
chmod +x .git/hooks/pre-commit
```

规则：`exit 0` 放行，`exit 非0` 阻止。**.git/hooks/ 不会被 git 跟踪**，团队共享用 **Husky**（前端）或 **pre-commit**（Python）。

---

## 3. Claude Code 的 Hooks 机制

在 `settings.json` 中配置，拦截 AI Agent 的工具调用和用户交互。

### 3.1 可用事件

| Hook 事件 | 触发时机 |
|-----------|---------|
| `PreToolUse` | 工具执行**之前**，可阻止 |
| `PostToolUse` | 工具执行**之后** |
| `PreUserPromptSubmit` | 用户提交 prompt 之前 |
| `PostUserPromptSubmit` | 用户提交 prompt 之后 |
| `SessionStart` / `SessionEnd` | 会话开始/结束时 |

### 3.2 配置结构

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/claude-hooks/check-dangerous-cmd.py"
          }
        ]
      }
    ]
  }
}
```

- `matcher`：匹配工具名（`Bash`/`Write`/`Edit`），不写或 `""` 匹配所有。大小写敏感
- hook 脚本从 **stdin** 接收 JSON（含 `tool_input` 等上下文），**stdout** 输出决策

### 3.3 示例：PreToolUse 拦截危险的 rm 命令

```python
#!/usr/bin/env python3
"""check-dangerous-cmd.py"""
import json, sys

hook_input = json.load(sys.stdin)
command = hook_input.get("tool_input", {}).get("command", "")

if "rm " in command:
    print(json.dumps({
        "decision": "block",
        "reason": f"⚠️ 危险命令已阻止: {command}"
    }))
else:
    print(json.dumps({"decision": "allow"}))
```

**返回值格式**：
```json
{"decision": "allow"}                           // 放行
{"decision": "block", "reason": "原因说明"}       // 阻止
{"decision": "allow", "silent": true}            // 静默放行
```

### 3.4 调试：Hook 没生效？

1. **matcher 大小写不对** — `"bash"` ❌ → `"Bash"` ✅
2. **if 条件太严，匹配不上** — 用 `in` 而不是 `==` 做宽松匹配
3. **脚本没返回合法 JSON** — stdout 必须是决策对象
4. **脚本没执行权限** — `chmod +x`
5. **settings.json 语法错误** — 检查逗号、括号

**建议**：调试时先把 `matcher` 设为 `""` 匹配全部工具，确认能触发后再收紧条件。
