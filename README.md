# SimpleRAG

轻量级本地 RAG（检索增强生成）系统，支持文档摄入、向量化存储和 LLM 智能问答。

## 特性

- **完全本地化 Embedding**：使用 sentence-transformers + Qwen3-Embedding-0.6B，GPU 加速
- **OpenAI 兼容 API**：支持任意兼容 API（DeepSeek、Ollama、OpenAI 等）
- **多策略分块**：paragraph + small_window + sliding 三种分块策略并行处理
- **精确关键词匹配**：查询时对关键词执行全文扫描，确保专有名词不错漏
- **轻量向量库**：Chroma 本地存储，零配置
- **Tool Calling**：LLM Agent 自主决定何时查询知识库

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置

编辑 `config.yaml`：

```yaml
llm:
  base_url: "https://api.minimaxi.com/v1"
  model: "MiniMax-M2.7"
  api_key: "your-api-key"

embedding:
  model_path: "D:/Resources/Models/Qwen3-Embedding-0.6B"
  dimension: 1024

vectorstore:
  persist_dir: "./data/chroma"

search:
  default_top_k: 5
  hard_cap: 50.0
  filter_delta: 0.3
  post_process: true      # 开启关键词匹配、Boost、去重、过滤；关闭则输出原始语义相似度

chunking:
  chunk_size: 500
  chunk_overlap: 50
  use_paragraph: true     # 段落分块（默认开启）
  use_small_window: true  # 小窗口分块（200字），利于精确匹配专有名词
  use_sliding: true      # 滑动窗口分块（400字，步长200），增加边界多样性

documents:
  input_dir: "./documents"
```

### 3. 放入文档

```bash
mkdir -p documents
# 放入 .txt, .md, .json 文件
```

### 4. 摄入文档

```bash
uv run python main.py ingest
```

### 5. 启动聊天

```bash
uv run python main.py chat
```

交互命令：
- `exit` - 退出
- `clear` - 清除对话历史

## 项目结构

```
simpleRAG/
├── config.yaml           # 配置文件
├── main.py               # 入口
├── src/
│   ├── config.py         # 配置加载
│   ├── chunker.py        # 多策略分块
│   ├── embeddings.py    # Embedding 服务
│   ├── vectorstore.py    # Chroma 封装 + 搜索算法
│   ├── tools.py          # Tool Calling
│   ├── agent.py          # LLM Agent
│   └── ingest.py         # 摄入流水线
├── tests/                # 测试
├── documents/            # 文档目录
└── data/chroma/         # 向量库存储（不入库）
```

## 搜索算法

查询流程（六步）：

1. **语义搜索**：向量化查询，返回 top_k × 5 条候选
2. **关键词提取**：从查询中提取 ≥3 字符的英文/数字词
3. **精确匹配扫描**：对全部文档做关键词全文扫描
4. **合并排序**：精确匹配的结果获得距离大幅折扣（0.15× 全匹配，0.5× 部分匹配）
5. **关键词 Boost**：每个命中的关键词将距离乘以 0.4^n（n = 命中词数）
6. **去重 + 过滤**：按内容去重，按 `min_dist + filter_delta` 相对阈值过滤

阈值说明：
- `hard_cap`：绝对距离上限，超过此值直接丢弃（默认 50.0）
- `filter_delta`：相对阈值窗口，仅保留距离在最优结果 + delta 内的结果

## 技术栈

| 模块 | 技术 |
|------|------|
| Embedding | sentence-transformers + Qwen3-Embedding-0.6B |
| Vector Store | Chroma |
| LLM | OpenAI 兼容 API |
| 分块 | paragraph + small_window + sliding 三策略 |

## 命令

```bash
uv run python main.py ingest                      # 摄入文档
uv run python main.py chat                        # 启动聊天
uv run python main.py query <搜索内容>              # 直接搜索向量库
uv run python main.py query <搜索内容> --top 10    # 指定返回条数
uv run python main.py query <搜索内容> --no-post-process  # 关闭后处理，输出原始语义距离
uv run python main.py help                         # 帮助
```

## 开发

```bash
# 运行测试
uv run pytest tests/ -v
```
