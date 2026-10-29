# AML Memory Adapter —— 公开版（Agent Memory Leaderboard 开源/学术榜）

本目录是提交给 **Agent Memory Leaderboard（开源/学术榜）** 的公开代码快照，对应公开
GitHub 仓库根。它是公开基线 [Vanilla RAG Memory Benchmark (v0.6.0,
`vanilla-rag-memory`)](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) 的派生与
扩展实现，按 AML 契约对齐了**同步 Add / Search** 接口，并做了大量检索增强。

> 完整代码级来源披露与复用范围见 [`doc/DISCLOSURE.md`](doc/DISCLOSURE.md)；评测权限申请
> 填表参考见 [`doc/ACCESS_REQUEST.md`](doc/ACCESS_REQUEST.md)；全部可调参数见
> [`doc/参数字典.md`](doc/参数字典.md)。

---

## 一、学术榜合规要点（与提交部署一致）

- **Embedding（硬性）**：提交版本使用阿里百炼 **API 后端 `text-embedding-v4`**（1024 维，
  `embedding_backend=api`），满足学术榜"Embedding 须用 `text-embedding-v4`"的要求。
  本地开发 / 评测可改用开源 `BAAI/bge-small-en-v1.5`（改 `config.server.yaml` 的
  `embedding_backend` 即可，见 `src/embedder.py`）。
- **LLM 组件**：本系统的 Add / Search 服务链路为**纯检索，不含任何 LLM 生成组件**，
  因此学术榜"LLM 组件须用 `gpt-4o-mini`"的约束在本服务上**不触发**；评测期 Answer
  生成由平台统一负责。
- **Reranker**：代码内置 `CrossEncoder`（`src/rerank.py`，非 LLM，不受 `gpt-4o-mini`
  约束）作为可选增强；**当前提交部署未启用 rerank**（`rerank_enabled: false`，配合无
  torch 的轻量镜像，无需 GPU）。如需开启，改用含 torch 的镜像即可。
- **运行资源**：当前部署无本地模型推理（embedding 走 API、rerank 关），**无需 GPU、
  无需大内存**，CPU + ≥2GiB 内存即可稳定运行（已在 2GiB 实例上验证）。

## 二、目录结构

| 路径 | 内容 |
| --- | --- |
| `src/` | 服务核心代码（vanilla 派生 + 扩展） |
| `deploy/` | 部署脚手架（Dockerfile / docker-compose / Caddyfile / 配置示例 / 自检脚本）—— **不含任何密钥** |
| `configs/` | 配置示例（`20261026best.yaml`，定版检索参数） |
| `doc/` | 来源披露、权限申请参考、参数字典 |

> 本公开仓库**不包含**：`.env`（密钥由部署方自行填）、评测脚本与评测数据
> （`local_eval/` / `data/` / `result/`，赛制由平台统一 Answer/Eval，不公开）、
> 查询扩展用的 HyperE 词向量数据（约 30MB，部署时自行挂卷到容器 `/app/data/hyperE`，缺失则检索退化为 2 路、不影响启动）。

## 三、快速运行

```bash
cd deploy
cp .env.example .env          # 填入 DASHSCOPE_API_KEY（百炼 embedding）与 AML_API_KEY（平台鉴权）
docker compose up -d --build
curl http://127.0.0.1:8000/health     # 期望 200
```

- 公开仓库不含 `.env`；密钥由部署方自行提供，**绝不入库**。
- 查询扩展（4 路融合）依赖 HyperE 词向量数据：把它挂为卷到容器
  `/app/data/hyperE/hyperE.wn.10d`（缺失则自动退化为 2 路，服务仍正常）。公开仓库不含该数据。
- `Caddyfile` 提供自动 HTTPS 反代（需域名 + 安全组放行 80/443），公网地址即
  `https://<你的域名>/add`、`/search`、`/health`。

## 四、接口契约（与平台一致）

- `POST /add`：同步写入记忆，返回 200 前完成持久化与向量化；`request_id` 幂等
  （同 ID 同 payload 幂等，异 payload 返回 **409**）。
- `POST /search`：按 `user_id` 隔离检索，返回 `data` 数组（数量 ≤ `top_k`，正式评测
  `top_k=100`），字段 `id` / `content` / `score` / `created_at`。
- `GET /health`：无需鉴权，返回 2xx。
- 认证：`Authorization: Token <key>`（同时支持 `Bearer` 与 `X-Api-Key`）。

## 五、合规声明

被复用代码仅用于本地生成评测证据，未将公开 / 官方评测数据用于训练、微调或对外传播；
本系统由参赛方自行部署在线 Add / Search 接口，符合平台"参赛方自行部署，AML 不代为部署"
的规定。逐文件溯源见 [`doc/DISCLOSURE.md`](doc/DISCLOSURE.md)。
