# JSON 数据格式

## 为什么使用 JSON？

| 格式 | 问题 |
|------|------|
| Text | 非结构化，抽取信息不便 |
| CSV | 数据不含 schema，有一定风险 |

## JSON 结构

JSON 主要有两种结构：**JSON 对象** 和 **JSON 数组**

### JSON 对象

```json
{
  "key": value,
  "key": value
}
```

- **key**: 必须是字符串
- **value**: 可以是数字、字符串、列表、JSON 对象或 JSON 数组

**示例**：

```json
{
  "age": 11,
  "name": "周杰轮",
  "hobby": ["唱", "跳", "rap"],
  "other": {
    "身高": 172,
    "体重": 68
  }
}
```

### JSON 数组

```json
[{}, {}, {}]
```

## 与 Python 对应

| JSON | Python |
|------|--------|
| JSON 对象 | 字典 (dict) |
| JSON 数组 | 列表 (list) |

## Python JSON 操作

### `json.dump()` - 写入 JSON 文件

将 Python 对象序列化为 JSON 并写入文件。

```python
import json

data = {"name": "张三", "age": 25}
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
```

**参数说明**：
- `ensure_ascii=False`：保留中文字符，不转义为 Unicode
- `indent=2`：格式化缩进，提高可读性

### `json.load()` - 读取 JSON 文件

从 JSON 文件反序列化为 Python 对象。

```python
import json

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(data)  # {'name': '张三', 'age': 25}
```