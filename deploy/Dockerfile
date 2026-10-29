# AML Memory Adapter —— 生产镜像（无 torch / sentence-transformers）
#
# 依赖为什么能砍掉 torch：20261026best.yaml 用的是
#   embedding_backend=api（百炼 text-embedding-v4），rerank_enabled=false。
#   src/embedder.py 与 src/rerank.py 里 sentence_transformers 都是**函数内延迟导入**，
#   不走这两条路就不会被 import —— 镜像从 ~4GB 降到 ~300MB。
#   ⚠ 若将来改回 local embedding 或开启 rerank，必须在 requirements.server.txt
#      里加 sentence-transformers 并换 cuda/torch 基础镜像。
#
# 目录形为什么必须是 /app/src（不能把 src/*.py 平铺到 /app）：
#   src/expander.py 的 `Expander._resolve()` 用 `Path(__file__).resolve().parents[1]`
#   作为数据根目录基准 —— expander.py 在 /app/src 下时 parents[1] == /app，
#   正好等于 `data/hyperE/...` 的父目录。平铺到 /app 会让 parents[1] 变成 /，
#   HyperE 加载失败 -> 查询扩展静默禁用（只打 WARNING，不报错，指标偷偷下滑）。
#
# 构建（在仓库根执行，build context 必须是仓库根）：
#   docker build -f deploy/Dockerfile -t aml-memory:latest .
#   docker compose -f deploy/docker-compose.yml up -d --build

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 依赖单独一层：改代码不重装依赖
COPY deploy/requirements.server.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt

# 源码
COPY src/ /app/src/

# HyperE 词向量数据（约 30MB，查询扩展用；本公开仓库不含该数据）：
#   - 容器内路径固定为 /app/data/hyperE/hyperE.wn.10d/（对应 config.server.yaml 的 expand_dir）；
#   - 部署时需自行把 HyperE 数据挂为卷，例如：
#       -v /宿主机/hyperE.wn.10d:/app/data/hyperE/hyperE.wn.10d
#     `.txt` 必须存在（expander 用 exists() 定位），`.pickle` 存在时优先加载（秒开）。
#   - 缺了它们 == 查询扩展被静默禁用，检索从 4 路退回 2 路（只打 WARNING，不报错，服务照常）。
#   - 因公开仓库不含 data/，下面这行 COPY 已移除；若从含 data/ 的私有仓库构建可恢复：
#       COPY data/hyperE/hyperE.wn.10d/ /app/data/hyperE/hyperE.wn.10d/

# 服务配置快照（db_path / expand_dir 已改成容器内绝对路径）
COPY deploy/config.server.yaml /app/config.server.yaml

# 外部密钥（secrets）通过环境变量 / compose env_file 注入，不入镜像：
#   AML_API_KEY        —— 不给则 /add /search 一律 503（app.py 的 require_api_key）
#   DASHSCOPE_API_KEY  —— 缺失时 MemoryStore 构造期的 _check_identity() 会直接抛错，容器起不来
ENV MEMORY_CONFIG=/app/config.server.yaml \
    MEMORY_DB_PATH=/data/memories.db \
    PORT=8000

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# --workers 1 是硬约束，不是默认值选择：
#   store.MemoryStore 是单进程单例，进程内的 {snapshot 缓存 / expander 索引 /
#   阴极 forget 墓碑缓存} 都不跨进程同步；多 worker 会让同一 user_id 随机落到
#   不同进程 -> 同一道题两次 Search 结果不一致（不可复现）。
#   SQLite 侧虽然开了 WAL + busy_timeout=60s，仍建议单进程。
# --proxy-headers / --forwarded-allow-ips 是硬要求，不是可选优化：
#   Caddy 以明文 HTTP 转发给 uvicorn，uvicorn 需信任 X-Forwarded-Proto 才能把
#   request.url.scheme 记成 https。否则 FastAPI 的 redirect_slashes 生成的 307
#   Location 会写成 http://…（HTTPS 被降级），POST 场景还会被客户端改写成 GET 丢 body。
#   容器只监听内网，--forwarded-allow-ips * 是安全的。
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*", \
     "--timeout-keep-alive", "120"]
