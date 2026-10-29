"""Embedder: 文本 -> 向量，供 Add（向量化）与 Search（查询向量化）后续调用。

两种后端，按配置切换（`ServerConfig.embedding_backend` / 环境变量 `EMBEDDING_BACKEND`）：

1. ``local``（默认）—— 进程内 `sentence-transformers`，默认模型
   `BAAI/bge-small-en-v1.5`（384 维），由 `EMBEDDING_MODEL` 切换；
2. ``api`` —— OpenAI 兼容的 ``POST {base}/embeddings``（硅基流动 / 阿里百炼 /
   OpenAI 都能用），模型名由 ``EMBEDDING_MODEL`` 给（例 ``text-embedding-v4``），
   端点/密钥由 ``EMBEDDING_API_BASE`` / ``EMBEDDING_API_KEY`` 给。

   阿里百炼（DashScope）走兼容模式的最小配置：

   .. code-block:: text

       embedding_backend: api
       embedding_model: text-embedding-v4
       embedding_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
       embedding_api_dimensions: 1024        # 默认 1024；可选 2048/1536/768/512/256/128/64
       embedding_api_batch_size: 10          # 百炼硬上限 10 条/请求（超限直接 400）
       # 密钥放 local_eval/.env：DASHSCOPE_API_KEY=sk-xxx

   百炼的兼容模式**只支持** ``model`` / ``input`` / ``dimensions`` /
   ``encoding_format`` 这一组参数；``text_type``（query/document 非对称编码）与
   ``instruct`` 只有 DashScope 原生 SDK/API 才有，兼容模式下传了也会被忽略，
   所以本后端的"查询/文档差异"只能靠 ``embedding_query_prefix`` 表达（v4 默认不加）。

接口对齐 `doc/我的思路.txt` 里的 `Embedder.encode_documents()` / `encode_queries()`：
- Add 侧对切块文本调用 `encode_documents`（文档不加前缀）；
- Search 侧对 query 调用 `encode_queries`（BGE 英文检索加固定前缀，否则掉分）；
- 两个后端都做 L2 归一化，使点积 == 余弦相似度，便于后续加权融合打分。

`identity` 是"真实编码配置"的序列化（后端 + 模型名 + 端点 + 维度 + 查询前缀），
`store._check_identity()` 直接用它固化到 `db_meta`，不再自己拼装 —— 保证"身份串 ==
实际编码行为"，杜绝改了配置但身份没变（或反之）的静默错配（对齐
vanilla-rag-memory embeddings.py:24 + memory_store.py:202-208）。
**换后端 / 换模型 / 换端点 / 换维度都必须重灌库**：旧库会被 `_check_identity()`
直接拒绝打开（不是静默错配）。

用法：
    from embedder import get_embedder
    emb = get_embedder()                          # 进程内单例；query_prefix=None -> BGE 英文模型自动加官方前缀
    emb_no_prefix = get_embedder(query_prefix="") # 显式空串 = 不加任何查询前缀（关掉）
    doc_vecs = emb.encode_documents(["text"])    # (n, dim) float32，已归一化
    q_vec   = emb.encode_queries(["query"])      # 加当前 embedder 配置的查询前缀
    dim     = emb.dimension                      # 384（local）/ 由端点返回（api）
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# BGE 英文检索固定查询前缀（官方建议，检索类任务必加，否则明显掉分）
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# sentence-transformers 默认 max_seq_length 由模型配置决定（bge-small 为 512）。
# 超过该长度的 chunk 会被**静默截断**，向量只覆盖前 512 个 subword；此处显式固定
# 并可由 EMBEDDING_MAX_SEQ_LENGTH 调整，配合切块参数一起评估。
DEFAULT_MAX_SEQ_LENGTH = 512

# ----- API 后端默认值 -----
#: 每批送多少条文本；太大容易撞端点的单请求 token / 条数上限
DEFAULT_API_BATCH_SIZE = 32
#: 单次 HTTP 请求超时（秒）
DEFAULT_API_TIMEOUT = 120.0
#: 429 / 5xx 的重试次数（指数退避）
API_RETRIES = 4
#: 单条文本送 API 编码前的字符上限（DashScope text-embedding-v4 单条输入上限
#: 16000；取保守值 8000，只截"本来就发不出去"的超长文本）
_API_MAX_CHARS = 8000

#: 服务商"单次请求最多几条文本"的硬上限：(端点子串, 上限, 说明)。
#: 阿里百炼 text-embedding-v4：**一次最多 10 条**（北京/新加坡同），单条 ≤8192
#: token，整批 ≤33000 token（北京）/ 8192 token（新加坡）。超限是直接 400，
#: 不会自动降级，所以按端点把 batch 夹到这个上限内（见 APIEmbedder.__init__）。
API_MAX_BATCH_BY_HOST: tuple[tuple[str, int, str], ...] = (
    ("dashscope.aliyuncs.com", 10, "阿里百炼 DashScope（text-embedding-v4 单请求上限 10 条）"),
    ("dashscope-intl.aliyuncs.com", 10, "阿里百炼 DashScope 新加坡（单请求上限 10 条）"),
    (".maas.aliyuncs.com", 10, "阿里百炼业务空间专属域名（单请求上限 10 条）"),
)


def _load_env_files() -> None:
    """best-effort 从仓库根 `.env` / `local_eval/.env` 补齐环境变量。

    只填充**当前进程里还没有**的键（不覆盖已 export 的值），解析失败静默跳过。
    目的：评测脚本从任意 cwd 启动时，`EMBEDDING_API_KEY` 之类也能自动取到，
    不必手动 export 或依赖 python-dotenv。`src/` 单独部署到平台时这两个文件
    不存在，等价于 no-op。
    """
    root = Path(__file__).resolve().parent.parent
    for cand in (root / ".env", root / "local_eval" / ".env"):
        try:
            if not cand.exists():
                continue
            for line in cand.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            continue


def _env(*names: str) -> str:
    """按顺序取第一个非空环境变量；都没有则返回空串。"""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val.strip()
    return ""


def _auto_query_prefix(model_name: str) -> str:
    """未显式给前缀时，仅对 BGE 英文模型套官方前缀（对齐 vanilla embeddings.py:21-22）。"""
    if "bge" in model_name.lower() and "-en-" in model_name.lower():
        return BGE_QUERY_PREFIX
    return ""


def _provider_batch_cap(api_base: str) -> tuple[int, str] | None:
    """按端点判断"单请求最多几条文本"；没有已知硬上限时返回 None。"""
    host = (api_base or "").lower()
    for pattern, cap, label in API_MAX_BATCH_BY_HOST:
        if pattern in host:
            return cap, label
    return None


def _is_dashscope(api_base: str) -> bool:
    """端点是否阿里百炼（决定密钥取哪个环境变量）。"""
    return _provider_batch_cap(api_base) is not None


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化（零向量保持零，不产生 NaN）。"""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class Embedder:
    """封装 ``sentence_transformers``，把文本批量转成归一化向量（本地后端）。"""

    backend = "local"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        query_prefix: str | None = None,
        device: str | None = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ) -> None:
        # 延迟导入：无 torch 环境 import 本模块不会直接崩
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        # 未显式给前缀时，仅对 BGE 英文模型套官方前缀（对齐 vanilla embeddings.py:21-22）
        if query_prefix is None:
            query_prefix = _auto_query_prefix(model_name)
        self.query_prefix = query_prefix or ""
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length and int(max_seq_length) > 0:
            self.model.max_seq_length = int(max_seq_length)

    @property
    def identity(self) -> str:
        """真实编码配置的序列化，供 embedding 身份固化。"""
        return f"{self._model_name}|query_prefix={self.query_prefix}"

    @property
    def model_name(self) -> str:
        """解析后的模型名（含环境变量覆盖），用于日志/兼容旧调用。

        必须与 `identity` / `dimension` 同为 property：写成普通方法时
        `emb.model_name` 拿到的是 bound method —— 不报错、只静默出错
        （例如拼进日志或拼进身份串）。
        """
        return self._model_name

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """文档侧编码：不加查询前缀，归一化。返回 ``(n, dim)`` float32。"""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=64, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """查询侧编码：加 BGE 前缀，归一化。返回 ``(n, dim)`` float32。"""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prefixed = [f"{self.query_prefix}{t}" for t in texts]
        vecs = self.model.encode(prefixed, batch_size=64, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        # get_sentence_embedding_dimension 在新版 sentence-transformers 已弃用
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()


class _FatalAPIError(RuntimeError):
    """不可重试的端点错误（鉴权失败 / 模型不存在 / 单条就超限）。

    单独成类只为在 ``_request`` 的重试循环里区分"该退避重试"与"立刻放弃"。
    """


class APIEmbedder:
    """OpenAI 兼容 ``/embeddings`` 后端（硅基流动 / 阿里百炼 / OpenAI）。

    与 `Embedder` 接口完全一致（`encode_documents` / `encode_queries` /
    `dimension` / `identity` / `model_name`），store 侧无需区分后端。

    - 批次：内部按 ``batch_size`` 切分，一次 HTTP 请求一批；``batch_size`` 会先被
      服务商硬上限夹住（百炼 = 10 条/请求，见 ``API_MAX_BATCH_BY_HOST``）；
    - 容错：429 / 5xx / 超时指数退避重试；单批 400/413（多半是超长或超条数）
      自动二分成两条更小的批，避免一条坏 chunk 让整次 Add 失败；
    - 归一化：返回值统一 L2 归一化（端点是否已归一化都无影响）；
    - 维度：``dimensions`` 显式给就信它；不给则首次编码时探测并缓存。
    """

    backend = "api"

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        query_prefix: str | None = None,
        dimensions: int = 0,
        batch_size: int = DEFAULT_API_BATCH_SIZE,
        timeout: float = DEFAULT_API_TIMEOUT,
    ) -> None:
        self._model_name = model_name
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.query_prefix = (
            _auto_query_prefix(model_name) if query_prefix is None
            else (query_prefix or "")
        )
        self._dimensions = int(dimensions) if dimensions and int(dimensions) > 0 else 0
        self._dim: int = self._dimensions
        self._batch_size = max(1, int(batch_size))
        self._timeout = float(timeout)
        # 按端点把批次夹进服务商硬上限（百炼 10 条/请求，超限直接 400）。
        # 夹的时候一定打日志：配置写了 32 却发 10，不能是静默行为。
        cap = _provider_batch_cap(self._api_base)
        if cap and self._batch_size > cap[0]:
            logger.warning(
                "embedding_api_batch_size=%d 超过 %s，自动降到 %d 条/请求"
                "（配置里的值不改，只影响实际请求）",
                self._batch_size, cap[1], cap[0])
            self._batch_size = cap[0]

    # ------------------------------------------------------------------
    # 身份 / 元信息
    # ------------------------------------------------------------------
    @property
    def identity(self) -> str:
        """真实编码配置的序列化。

        刻意不写"探测出来的维度"，而写配置里的 ``dimensions``（0 -> ``auto``）：
        身份串必须在**一次请求都不发**的前提下就确定，否则首次 Get 与后续 Get
        会拿到不同身份，把库判成两套配置。
        """
        dim = self._dimensions or "auto"
        return (f"api:{self._model_name}|base={self._api_base}|dim={dim}"
                f"|query_prefix={self.query_prefix}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def dimension(self) -> int:
        if not self._dim:
            # 探测一次：请求 1 条最短文本，取返回向量长度（随后缓存）
            probe = self._embed_batch(["dimension probe"])
            self._dim = len(probe[0])
        return self._dim

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """文档侧编码：不加查询前缀，归一化。返回 ``(n, dim)`` float32。"""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._embed_batch(list(texts))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """查询侧编码：加当前配置的查询前缀，归一化。返回 ``(n, dim)`` float32。"""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prefixed = [f"{self.query_prefix}{t}" for t in texts]
        return self._embed_batch(prefixed)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """按 `_batch_size` 分批请求，按输入顺序拼回 ``(n, dim)`` float32。

        单条文本先按 `_API_MAX_CHARS` 截断：DashScope text-embedding-v4 的单条
        输入上限是 16000（`Range of input length should be [1, 16000]`），BEAM
        这类数据集里存在无句边界的超长代码块（>16000 字符），此前会 400 ->
        `/add` 500 卡死整次灌库。本地 bge 走 512-token 截断，API 后端此前反而
        无截断；截断只影响"本来就发不出去"的超长文本，其它文本逐位不变。
        """
        capped = False
        safe: list[str] = []
        for t in texts:
            if len(t) > _API_MAX_CHARS:
                safe.append(t[:_API_MAX_CHARS])
                capped = True
            else:
                safe.append(t)
        if capped:
            logger.warning(
                "embedding API 输入里有超过 %d 字符的超长文本，已截断后编码"
                "（服务商单条上限，原文不动，只影响向量）", _API_MAX_CHARS)
        texts = safe
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start:start + self._batch_size]
            out.extend(self._request(chunk))
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def _request(self, batch: list[str], depth: int = 0) -> list[list[float]]:
        """POST 一批文本到 `/embeddings`；400/413 时二分递归缩小批次。

        重试策略：429 / 5xx / 网络异常 -> 指数退避重试；
        401/403（密钥错、地域不匹配）、404（模型名写错）、以及**无法再拆**的
        400/413（单条就超限）-> 立刻抛出，不浪费 4 次退避（此前一个错密钥会
        让每个批次白等 15 秒，批量 Add 直接卡死）。
        """
        import httpx

        url = f"{self._api_base}/embeddings"
        payload: dict = {"model": self._model_name, "input": batch}
        if self._dimensions:
            payload["dimensions"] = self._dimensions
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(API_RETRIES):
            try:
                with httpx.Client(timeout=httpx.Timeout(
                        self._timeout, read=self._timeout,
                        connect=self._timeout)) as client:
                    r = client.post(url, json=payload, headers=headers)
                if r.status_code in (401, 403):
                    raise _FatalAPIError(
                        f"Embedding API 鉴权失败 HTTP {r.status_code} "
                        f"(model={self._model_name!r}, base={self._api_base!r})："
                        f"{r.text[:500]}\n"
                        "检查密钥：百炼端点读 DASHSCOPE_API_KEY（key 与地域绑定，"
                        "北京/新加坡的 key 不能混用）。")
                if r.status_code == 404:
                    raise _FatalAPIError(
                        f"Embedding API 端点/模型不存在 HTTP 404 "
                        f"(model={self._model_name!r}, base={self._api_base!r})："
                        f"{r.text[:500]}")
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code in (400, 413) and len(batch) > 1 and depth < 6:
                    # 多半是"单批超 token / 超条数"：对半拆开重试，别让一条坏
                    # chunk 拖垮整次 Add。
                    mid = len(batch) // 2
                    return (self._request(batch[:mid], depth + 1)
                            + self._request(batch[mid:], depth + 1))
                if r.status_code >= 400:
                    raise _FatalAPIError(
                        f"Embedding API HTTP {r.status_code} "
                        f"(model={self._model_name!r}, base={self._api_base!r}, "
                        f"batch={len(batch)}): {r.text[:500]}")
                data = r.json()
                rows = data.get("data")
                if not isinstance(rows, list) or len(rows) != len(batch):
                    raise RuntimeError(
                        f"Embedding API returned unexpected shape: "
                        f"{str(data)[:300]}")
                # 有 index 就按 index 排（OpenAI 契约不保证顺序），没有就按原序
                if all(isinstance(x, dict) and "index" in x for x in rows):
                    rows = sorted(rows, key=lambda x: int(x["index"]))
                return [list(x["embedding"]) for x in rows]
            except _FatalAPIError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Embedding API failed after {API_RETRIES} retries: {last_err}")


@lru_cache(maxsize=8)
def get_embedder(
    model_name: str | None = None,
    query_prefix: str | None = None,
    device: str | None = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    backend: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    api_dimensions: int = 0,
    api_batch_size: int = DEFAULT_API_BATCH_SIZE,
) -> Embedder | APIEmbedder:
    """进程内单例工厂：避免重复加载模型（参考 baseline “单 worker 避免重复加载”）。

    取值优先级（**显式参数 > 环境变量 > 默认值**）：
      - 后端：``backend`` > ``EMBEDDING_BACKEND`` >（有 ``EMBEDDING_API_BASE`` 则 api，否则 local）
      - 模型：``model_name`` > ``EMBEDDING_MODEL`` > ``BAAI/bge-small-en-v1.5``
      - 端点：``api_base`` > ``EMBEDDING_API_BASE``
      - 密钥：``api_key`` > 按端点挑环境变量（见下）> 默认值
          百炼端点（``*aliyuncs.com``）：``DASHSCOPE_API_KEY`` > ``EMBEDDING_API_KEY`` > ``LLM_API_KEY``
          其它端点：``EMBEDDING_API_KEY`` > ``SILICONFLOW_API_KEY`` > ``LLM_API_KEY``
        之所以按端点分流：百炼的 key 与地域/服务商绑定，若沿用 `.env` 里给硅基流动
        填的 ``EMBEDDING_API_KEY`` 换到百炼端点上，只会拿到 401 而不是明确的提示。
      - 维度：``api_dimensions`` > ``EMBEDDING_DIMENSIONS``（0 = 交给端点默认，首次编码时探测）
      - 批次：``api_batch_size`` > ``EMBEDDING_API_BATCH_SIZE``（会被服务商硬上限夹住）

    ``query_prefix`` / ``backend`` 等都参与缓存键，故不同配置各自持有一份实例。
    """
    _load_env_files()
    resolved_model = model_name or _env("EMBEDDING_MODEL") or DEFAULT_MODEL
    resolved_backend = (backend or _env("EMBEDDING_BACKEND")).lower()
    resolved_base = api_base or _env("EMBEDDING_API_BASE")
    if not resolved_backend:
        resolved_backend = "api" if resolved_base else "local"
    if resolved_backend not in ("local", "api"):
        raise ValueError(
            f"embedding_backend={resolved_backend!r} 非法，只支持 'local' / 'api'")

    if resolved_backend == "api":
        if _is_dashscope(resolved_base):
            key = (api_key or _env("DASHSCOPE_API_KEY", "EMBEDDING_API_KEY",
                                   "LLM_API_KEY"))
        else:
            key = (api_key or _env("EMBEDDING_API_KEY", "SILICONFLOW_API_KEY",
                                   "LLM_API_KEY"))
        if not resolved_base:
            raise RuntimeError(
                "embedding_backend=api 但没配置端点：请在 local_eval/.env 里填 "
                "EMBEDDING_API_BASE（硅基流动 = https://api.siliconflow.cn/v1；"
                "阿里百炼 = https://dashscope.aliyuncs.com/compatible-mode/v1）")
        if not key:
            raise RuntimeError(
                "embedding_backend=api 但没配置密钥：请在 local_eval/.env 里填 "
                + ("DASHSCOPE_API_KEY（百炼控制台 -> API-KEY）"
                   if _is_dashscope(resolved_base) else
                   "EMBEDDING_API_KEY（硅基流动控制台 -> API 密钥）"))
        dims = int(api_dimensions or 0) or int(_env("EMBEDDING_DIMENSIONS") or 0)
        batch = int(api_batch_size or 0) or int(
            _env("EMBEDDING_API_BATCH_SIZE") or DEFAULT_API_BATCH_SIZE)
        return APIEmbedder(
            resolved_model, api_base=resolved_base, api_key=key,
            query_prefix=query_prefix, dimensions=dims, batch_size=batch,
        )

    return Embedder(resolved_model, query_prefix=query_prefix, device=device,
                    max_seq_length=max_seq_length)


if __name__ == "__main__":
    # 最小自测：local 首次运行会自动联网下载模型（或读取本地 EMBEDDING_MODEL 路径）；
    # api 后端则直接用 .env / 环境变量里的端点与密钥发一次真实请求。
    emb = get_embedder()
    print("backend:", emb.backend)
    print("model:", emb.model_name)
    print("identity:", emb.identity)
    print("dimension:", emb.dimension)
    docs = emb.encode_documents(
        ["Rob moved to Beijing last year.", "The store opens at 9am."]
    )
    q = emb.encode_queries(["Where does Rob live?"])
    sims = (docs @ q.T).ravel()
    print("query vs docs cosine:", [round(float(s), 4) for s in sims])
