"""HTTP contract models for the AML Add/Search adapter (extracted from app.py).

Field names / types mirror agentmemoryleaderboard.ai/api-guide exactly so the
service stays contract-compatible. 取值约束对齐 vanilla-rag-memory api.py:20-44：
关键字段不允许空串、`top_k` 必须为正且 ≤ 100 —— 让非法请求在 422 层就被拦住，
而不是带着空 user_id / top_k=0 走到检索内部才静默返回空结果。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AddMessage(BaseModel):
    role: str = Field(min_length=1)     # "user" | "assistant"
    content: str = Field(min_length=1)  # non-empty per contract
    timestamp: int | None = None        # Unix milliseconds；可为 None（无时间）
    speaker: str | None = None          # optional speaker label; if absent, `role`
                                        # is used as the prefix. Lets a loader (e.g.
                                        # LoCoMo) preserve "who said it" inside the
                                        # stored content. Backward-compatible with the
                                        # official contract, which never sends it.


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=100)        # 对齐 vanilla api.py:44 必填 + 上限 100
    options: list[str] | None = None
    debug: bool = False                     # 仅本地评测/排查用：true 时 /search 的响应
                                            # 额外带一个 `debug` 段（检索各阶段中间值）。
                                            # 默认 false => 响应体与官方契约逐字节一致。


class MemoryHit(BaseModel):
    id: str
    content: str
    score: float | None = None
    created_at: str | None = None   # ISO 8601


class AddResponse(BaseModel):
    success: bool
    request_id: str
    user_id: str
    session_id: str


class SearchResponse(BaseModel):
    data: list[MemoryHit]           # empty results -> []
