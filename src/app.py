"""FastAPI adapter exposing AML-style Add / Search endpoints.

Contract-aligned with agentmemoryleaderboard.ai/api-guide (verified):
  Add  : POST {request_id, messages[], user_id, session_id}
         -> 200 {success:true, request_id, user_id, session_id} (echoed exactly)
  Search: POST {query, user_id, top_k, options?}
         -> 200 {data:[{id, content, score?, created_at?}]}
`user_id` is the sole isolation key; `data` must be a top-level array.
The platform retains the whole Search candidate (including `created_at`, if emitted)
and passes it intact into the fixed Answer workflow -- the Answer Context Budget keeps
the archived Search response as-is. `content` is the contractually required field for
answering; `created_at` is optional metadata that rides along. So whatever
`store.search` returns -- content AND any `created_at` we choose to emit -- is what
gets scored.

This module is a *thin* HTTP contract layer: it only maps requests/responses and
delegates all storage + retrieval to `store.MemoryStore`. Configuration is loaded
once via `config.load_config()` —— 不传参时读环境变量；设了 `MEMORY_CONFIG=<path>`
（等价 CLI 的 `--config`）时读该 YAML，且 YAML 优先级高于环境变量。

鉴权（对齐 vanilla-rag-memory api.py:114-129）：
  * 逐路由用 `Depends(require_api_key)` 声明，`/health` 没声明 -> **天然豁免**，
    不再靠 `EXEMPT_PATHS` 路径集合（`/health/` 带尾斜杠匹配不到的边界也一并消失）；
  * 比较用 `hmac.compare_digest`（恒定时间）；
  * **未配置 `AML_API_KEY` 时不再静默放行**，返回 503；本地/冒烟要放行必须显式
    `ALLOW_UNAUTHENTICATED=true`（`tools/smoke_test.py` / `tools/verify.py` 已自带）。
"""
from __future__ import annotations

import hmac
import logging
import math
import os

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import load_config
from schema import (AddMessage, AddRequest, SearchRequest, MemoryHit,
                    AddResponse, SearchResponse)
from store import MemoryStore, RequestConflictError

logger = logging.getLogger("aml_memory")

cfg = load_config()
store = MemoryStore(cfg)
app = FastAPI(title="AML Memory Adapter")

_TRUE = ("1", "true", "yes", "on")


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # Platform-preferred {"detail": {"reason": ...}} envelope, keeping the
    # structured field errors for troubleshooting.
    return JSONResponse(
        status_code=422,
        content={"detail": {"reason": "request validation failed", "errors": exc.errors()}},
    )


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception):
    """兜底：任何未捕获异常都返回契约形状的 JSON，而不是 FastAPI 默认 HTML/纯串。

    线上最怕的是"检索内部抛异常 -> 平台收到 500 + 非 JSON -> 整题判 0"，这里至少
    保证响应体可解析，并把异常类型落在日志里。
    """
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": {"reason": "internal server error",
                            "type": exc.__class__.__name__}},
    )


def _extract_api_key(request: Request) -> str | None:
    """Parse API key from Authorization: Bearer/Token <key> or X-Api-Key: <key>."""
    auth = request.headers.get("Authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() in ("bearer", "token"):
            return token or None
        return auth or None  # no scheme: treat whole value as the key
    return request.headers.get("X-Api-Key") or None


def require_api_key(request: Request) -> None:
    expected = os.getenv("AML_API_KEY", "")
    if not expected:
        if os.getenv("ALLOW_UNAUTHENTICATED", "false").strip().lower() in _TRUE:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"reason": "AML_API_KEY is not configured"},
        )
    provided = _extract_api_key(request)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "invalid or missing API key"},
        )


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/health/", methods=["GET", "HEAD"])
def health():
    # 契约只要求「无需鉴权的 GET 返回 2xx」：不声明 Depends(require_api_key) 即天然豁免。
    # 响应体刻意不带任何业务字段 —— 不做全表 COUNT（探针无需付出 DB 开销，
    # 也不对外暴露库内条目数）。
    #
    # 为什么用 api_route 且显式挂两条路径：
    #  1) FastAPI 的 APIRoute 不会像 Starlette 原生 Route 那样自动把 HEAD 并进 GET，
    #     只写 @app.get 时 HEAD /health 直接 405（Allow: GET），部分平台探活默认用 HEAD。
    #  2) 只挂 "/health" 时，"/health/" 会落到 redirect_slashes 的 307，而该 307 的
    #     Location 由 request.url 拼出；服务在 Caddy 后面、uvicorn 未认 X-Forwarded-Proto
    #     时会拼成 http://…（HTTPS 请求被降级到明文）。显式挂两条路径即无跳转。
    return {"status": "ok"}


@app.post("/add", response_model=AddResponse,
          dependencies=[Depends(require_api_key)])
def add(req: AddRequest):
    try:
        store.add_request(
            user_id=req.user_id,
            session_id=req.session_id,
            request_id=req.request_id,
            messages=req.messages,
        )
    except RequestConflictError as exc:
        # 同一 request_id 换了 payload / user_id / session_id：拒绝而不是覆盖。
        # 覆盖会 DELETE + 重插，SQLite 会复用 rowid，会话内全序（= rowid）就毁了。
        logger.error("add conflict: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": str(exc)},
        ) from exc
    return AddResponse(
        success=True,
        request_id=req.request_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )


@app.post("/search", dependencies=[Depends(require_api_key)])
def search(req: SearchRequest):
    # 说明：这里刻意不写 response_model=SearchResponse——因为要在 req.debug=true 时
    # 额外挂一个 `debug` 段，而声明了 response_model 的路由会把多余字段过滤掉。
    # 默认路径（debug=false）的响应体仍由 SearchResponse 序列化，与契约完全一致。
    if req.debug:
        hits, dbg = store.search(
            req.query, user_id=req.user_id, top_k=req.top_k,
            options=req.options, return_debug=True)
    else:
        hits = store.search(
            req.query, user_id=req.user_id, top_k=req.top_k,
            options=req.options)
        dbg = None
    data = [
        MemoryHit(
            id=e.id,
            content=e.content,
            # CrossEncoder 输出的是原始 logit，可能为负 / NaN，输出前兜成 0.0
            score=round(s, 4) if math.isfinite(s) else 0.0,
            # 该 message 没有可信时间戳（source_timestamp=0）时输出 null，
            # 绝不输出 1970-01-01 或入库时刻这类伪造值；created_at 是可选字段。
            created_at=e.created_at_iso() if e.created_at else None,
        )
        for s, e in hits
    ]
    payload = SearchResponse(data=data).model_dump()
    if dbg is not None:
        # `data` 之外只在 debug 请求下多一个键，线上默认路径不含它。
        payload["debug"] = dbg
    return payload
