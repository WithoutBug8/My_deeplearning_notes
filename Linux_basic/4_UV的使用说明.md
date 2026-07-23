# UV 快速入门

> `uv` 是一个现代 Python 包管理工具，可以替代 `pip + venv + virtualenv + pip-tools + pyenv` 的大部分功能。

---

# 1. 创建新项目

```bash
uv init myproject
cd myproject
```

生成：

```
myproject/
├── pyproject.toml    # 项目配置
├── uv.lock           # 依赖锁文件
├── .python-version   # Python版本
└── src/
```

---

# 2. 安装 Python

查看可用版本：

```bash
uv python list
```

安装 Python：

```bash
uv python install 3.12
```

固定项目 Python 版本：

```bash
uv python pin 3.12
```

---

# 3. 创建/同步虚拟环境

```bash
uv sync
```

作用：

- 创建 `.venv`
- 安装项目所有依赖
- 根据 `uv.lock` 恢复环境

通常第一次 clone 项目后第一件事就是执行：

```bash
uv sync
```

---

# 4. 安装依赖

安装：

```bash
uv add numpy
```

安装多个：

```bash
uv add numpy pandas matplotlib
```

开发依赖：

```bash
uv add --dev pytest
```

删除依赖：

```bash
uv remove numpy
```

---

# 5. 运行程序

无需激活虚拟环境：

```bash
uv run python main.py
```

运行模块：

```bash
uv run python -m app.main
```

运行测试：

```bash
uv run pytest
```

运行 FastAPI：

```bash
uv run uvicorn app:app --reload
```

---

# 6. 查看依赖

```bash
uv tree
```

查看已安装包：

```bash
uv pip list
```

---

# 7. 导出 requirements.txt

```bash
uv export -o requirements.txt
```

---

# 8. 临时运行工具

无需安装即可使用：

```bash
uvx ruff check .
```

```bash
uvx black .
```

```bash
uvx mypy .
```

---

# 9. 常见文件说明

## pyproject.toml

项目配置文件。

记录：

- 项目名称
- Python版本要求
- 项目依赖

例如：

```toml
dependencies = [
    "numpy",
    "pandas"
]
```

---

## uv.lock

依赖锁文件。

记录：

- 所有依赖的**精确版本**
- 所有间接依赖

保证：

> 每个人安装出来的环境完全一致。

建议提交到 Git。

---

## .venv

虚拟环境目录。

里面是真正安装的 Python 和所有依赖。

**不要提交到 Git。**

`.gitignore`

```text
.venv/
```

---

# 10. 推荐工作流

## 新建项目

```bash
uv init
uv add requests
uv run python main.py
```

---

## 克隆别人项目

```bash
git clone ...
cd project
uv sync
uv run python main.py
```

---

## 添加新依赖

```bash
uv add fastapi
git add pyproject.toml uv.lock
git commit
```

---

# 11. Conda / Pip 对照

| Conda / Pip | uv |
|------------|----|
| `conda create -n env python=3.12` | `uv init` + `uv python pin 3.12` |
| `conda activate env` | 无需 activate，直接 `uv run` |
| `pip install numpy` | `uv add numpy` |
| `pip uninstall numpy` | `uv remove numpy` |
| `pip install -r requirements.txt` | `uv sync` |
| `python main.py` | `uv run python main.py` |
| `pip list` | `uv pip list` |
| `pip freeze` | `uv export` |

---

# 12. 最常用命令（记住这几个就够了）

```bash
uv init                 # 创建项目

uv sync                 # 创建/同步虚拟环境

uv add requests         # 安装依赖

uv remove requests      # 删除依赖

uv run python main.py   # 运行程序

uv tree                 # 查看依赖关系

uv python pin 3.12      # 固定 Python 版本
```

---

# 一句话理解三个核心文件

```
pyproject.toml
    ↓
告诉 uv：项目需要哪些依赖

uv.lock
    ↓
锁定所有依赖的具体版本

uv sync
    ↓
创建 .venv，并安装这些依赖

uv run
    ↓
使用 .venv 中的 Python 运行程序
```