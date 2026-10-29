# 来源披露与代码溯源（Origin Disclosure & Provenance）

> 本文档供 Agent Memory Leaderboard 准入申请 / 代码审核使用。
> 说明本参测系统（`AML Memory Adapter`，仓库 `memory_20261022`）与公开基线
> `vanilla-rag-memory`（Vanilla RAG Memory Benchmark v0.6.0）的派生关系、
> 复用范围与新增改动。

---

## 1. 系统概述

本参测系统（AML Memory Adapter）是基于公开基线 **Vanilla RAG Memory Benchmark
（v0.6.0，仓库名 `vanilla-rag-memory`）** 的派生与扩展实现，对原仓库做了封装与
增强，并按 Agent Memory Leaderboard 契约对齐了 Add / Search 接口。原仓库核心实现
已在源码头注释中逐文件标注来源（file:line），复用的部分见第 3 节。

---

## 2. 来源披露（Origin / Disclosure）

本参测系统是公开基线 **Vanilla RAG Memory Benchmark（v0.6.0，`vanilla-rag-memory`）**
的派生与扩展实现，对原仓库做了封装与增强，并按 Agent Memory Leaderboard 契约对齐了
Add/Search 接口。原仓库核心实现已在源码头注释中逐文件标注来源（file:line），复用的
部分包括：

- **分词与切块**：`TOKEN_PATTERN`（CJK/英文/符号统一切分正则）与 `lexical_tokens()`
  BM25 词袋逻辑（对齐 vanilla `chunking.py` / `memory_store.py`）；
- **持久化与检索核心**：`SQLiteMemoryStore` 的表结构、`_payload_hash` 幂等、
  `add()`/`search()` 的 Dense + BM25/RRF 融合、rerank 候选、result_window 相邻上下文
  扩展（`memory_store.py`）；
- **Embedding**：`SentenceTransformerEmbedder` 的 `identity` 串与 BGE 英文检索查询前缀
  判定（`embeddings.py`）；本系统新增 `APIEmbedder` 后端（阿里百炼 `text-embedding-v4`），
  **提交版本即使用该 API 后端**，不依赖本地 sentence-transformers。
- **服务层**：FastAPI 的 `/health` `/add` `/search` 路由、`require_api_key`
  （hmac 恒定时间比较 + 未配密钥返 503 + `ALLOW_UNAUTHENTICATED` 旁路）与 422 校验
  处理器（对应 vanilla `api.py`，本系统重命名为 `app.py`）。

在基线之上，本系统新增/修改的模块包括：

1. `config.py` + YAML 配置体系（替代纯环境变量）；
2. `embedder.py` 新增 `APIEmbedder` 后端（OpenAI 兼容 / 阿里百炼，含批次上限、退避重试、
   超长截断）；
3. `chunking.py` 新增相邻消息打包 `pack_and_split`、句级切分、英文词干还原；
4. `store.py` 大幅扩展：邻接索引、上下文分组、时间增强与相对时间解析、查询扩展、
   多检索模式、embedding 身份校验；
5. `app.py` 新增 409 冲突处理、`{"detail":{"reason":...}}` 错误信封、debug 模式。

合规声明：被复用代码仅用于本地生成评测证据；未将公开/官方评测数据用于训练、微调或
对外传播；本系统自行部署在线 Add/Search 接口，符合"不接受仅提交仓库后代为部署"的要求。

- 原仓库地址：<https://github.com/wenxiaof345-ctrl/vanilla-rag-memory>
- 原作者 / 组织：GitHub 账号 `wenxiaof345-ctrl`

---

## 3. 代码级溯源清单

按当前项目 `src/` 文件组织，逐文件给出对应的 vanilla 文件:行与复用 / 改动说明。

| 当前文件 | 对应 vanilla 文件:行 | 复用内容 | 改动 / 新增 |
| --- | --- | --- | --- |
| `chunking.py` | `chunking.py:8` | `TOKEN_PATTERN` 正则逐字符相同（CJK/英文/符号统一切分） | 同 |
| `chunking.py` | `memory_store.py:41-46` | `lexical_tokens()` 算法一致（casefold + 过滤纯符号 token） | **新增**英文词干还原（`ENABLE_STEMMING` / `_rule_stem` / `_EN_STEMMER`）打通形态变体 |
| `chunking.py` | `memory_store.py:157-176` | `split_memory_text()` 滑窗算法一致 | 新增参数夹取（`chunk_tokens>=1`、`0<=overlap<chunk_tokens`）+ `drop_short_tail` |
| `chunking.py` | — | 无 | **新增** `pack_and_split()`（相邻消息打包）、`split_sentences()`（句级切分） |
| `schema.py` | `api.py:20-44` | `top_k` 必填且 ≤100、`request_id`/`user_id`/`session_id` 非空等字段约束 | 实现从 dataclass 改为 pydantic；新增 `speaker`/`debug` 字段（契约向后兼容） |
| `embedder.py` | `embeddings.py:24` + `memory_store.py:202-208` | `identity` 串格式 `model\|query_prefix=...`、BGE 前缀判定逻辑 | **新增** `APIEmbedder` 后端（OpenAI 兼容/百炼、批次上限、退避重试、超长截断）、`get_embedder` 单例工厂 |
| `rerank.py` | `reranking.py:13-65` | `CrossEncoderReranker` 参数与 `score()` 逻辑近逐字改写 | 单例 `get_reranker`；去掉自动 cuda dtype（`torch.float16`）；**当前提交部署未启用 rerank**（`rerank_enabled: false`，无 torch 镜像） |
| `store.py` | `memory_store.py`（全文件 614 行） | `SQLiteMemoryStore` 表结构、`_payload_hash` 幂等、`add()`/`search()` 的 Dense+BM25/RRF 融合、rerank 候选、result_window 相邻扩展 | **大幅扩展**（约 1900+ 行，撰写时快照；以当前仓库 `src/store.py` 为准）：邻接索引、上下文分组、时间增强 `dateparse`、查询扩展 `expander`、多检索模式、`_check_identity` 身份校验 |
| `app.py` | `api.py:114-129` | `require_api_key`（`hmac.compare_digest` + 未配密钥返 503 + `ALLOW_UNAUTHENTICATED` 旁路） | 同；新增 509/兜底异常处理器、409 冲突说明 |
| `app.py` | `api.py:166-188` | `/health` `/add` `/search` 路由、`422` 校验处理器、Add/Search 契约模型 | 422 错误体改为 `{"detail":{"reason":...}}`；新增 debug 模式、`load_config()` YAML 配置 |

---

## 4. 当前项目新增、vanilla 中不存在的文件

| 当前文件 | 说明 |
| --- | --- |
| `config.py` | YAML 配置体系（替代 vanilla 的纯环境变量） |
| `schema.py` | 从 vanilla `api.py` 抽出的独立 Add/Search 请求与响应 pydantic 模型 |
| `adjacency.py` | 邻接索引 |
| `context_grouping.py` | 上下文分组 |
| `dateparse.py` | 时间增强与相对时间解析 |
| `expander.py` | 查询扩展（multiword / dense query） |

---

## 5. 当前项目有意未复用的 vanilla 模块

vanilla 的 `cli.py` / `datasets.py` / `evaluation.py` / `generation.py` / `index.py` /
`pipeline.py`（离线基准链路：JSONL → 切块 → FAISS → Answer → 指标）在 AML 赛制下由
平台统一负责 Answer / Eval，故当前项目**只保留赛事服务链路**（Add / Search），未引入
这些离线模块。这说明本次派生是聚焦裁剪，而非整体照搬。

---

## 6. 合规声明

- 被复用的 `vanilla-rag-memory` 代码仅在本地用于生成评测证据；
- 未将公开 / 官方评测数据用于模型训练、微调、产品分析、数据集重建或对外传播；
- 本系统自行部署在线 Add / Search 接口，符合平台"第二期所有榜单均要求参赛方自行部署
  接口，AML 不接受仅提交仓库后代为部署"的规定；
- 已在源码头注释中逐文件标注对 `vanilla-rag-memory` 的复用来源（file:line）。

---

## 7. 外部数据资产（查询扩展用 HyperE 词向量）

查询扩展（`store.py` 的 `expand_enabled`、模块 `expander.py`）使用的 **HyperE WordNet 双曲嵌入**
来自 **HazyResearch / Stanford DAWN**（Gu, Sala, Gunel & Ré，*"HyperE: Hyperbolic Embeddings
for Entities"*，项目页 <https://hazyresearch.stanford.edu/hyperE/>）。该数据自 WordNet 导出
（hypernym / domain topic / member holonym 层级，GloVe 格式，10 维），本项目仅**查表 + 双曲最近邻**
消费，不修改、不训练。

- **许可状态**：上游项目页**未明示任何软件 / 数据许可证**（无 MIT / Apache / CC-BY 等条款）。
  依"无许可即保留权利"原则，本项目**不将其随公开仓库再分发**。
- **处理方式**：公开仓库 `data/hyperE/` 被有意排除；部署方 / 审计方需按上游页面另行下载
  `WordNet.zip`（含 10d 文件）并挂载到容器 `/app/data/hyperE/hyperE.wn.10d`。
- **缺失行为**：数据缺失时扩展器自动禁用、检索从 4 路降级为 2 路（仅 WARNING，不报错），服务照常运行。
