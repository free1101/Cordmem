"""Same-session adjacency expansion.

Given the fused + ranked results, take the top ``seed_k`` as seeds and, for each
seed, pull its immediate neighbours (``window`` before/after) from the *same*
``session_id``.

两条关键语义，均与 vanilla-rag-memory ``memory_store.py::_expand_result_positions``
对齐（此前我们两条都做反了，是"最差组合"）：

1. **邻居保留自己的分数**，不继承种子的分数。此前邻居直接拿 seed 的分数，
   于是与 query 毫无关系的块以高分插进结果，还把尾部真实命中挤出 top_k。
2. **邻居是"追加"而不是"挤占"**：放完 seed + 邻居后，**继续用原始排序结果把
   `cap` 补满**。此前只做 `[:cap]` 硬截，邻居越多、被挤掉的尾部命中越多。

代价（vanilla 同样接受）：开启邻接后返回序列**不再按分数严格单调**
（seed、邻居、seed、邻居…），换来的是尾部命中不被丢弃。
``session_rows`` 由调用方从 `search()` 的内存快照传入，本模块不查 DB。
"""
from __future__ import annotations

import json


def _safe_meta(meta) -> dict:
    try:
        return json.loads(meta or "{}")
    except (TypeError, ValueError):
        return {}


def expand(
    seeds: list,                       # [(score, MemoryEntry), ...] ordered by fused rank
    seed_k: int,
    window: int,
    session_rows: dict,               # session_id -> [{"id","content","ts","ingest_at","role","meta","rowid"}, ...] in session order
    cap: int | None = None,           # 结果条数上限（= top_k）
    user_id: str = "",
    render=None,                      # (role, speaker, content, ts_ms, seq=None) -> 展示用 content
    order_meta: bool = True,          # 无时间条目渲染 `order: #rowid`（兜底分支用；render 路径由 store 自控）
    mixed_timeline: bool = False,     # 该用户时间轴混有 ts=0 与 ts>0（兜底分支用）
) -> list:
    """Return expanded list of (score, MemoryEntry).

    Walk ``seeds`` in fused-rank order. For every seed we emit the seed itself, and
    for the first ``seed_k`` seeds we immediately emit (按距离由近到远) up to
    ``window`` neighbours on each side that have not been seen yet — **each with its
    own score**. Afterwards the remaining ranked results are appended until ``cap``
    is filled, so neighbours never displace real hits.
    """
    if seed_k <= 0 or window <= 0 or not session_rows:
        return list(seeds[:cap]) if cap else list(seeds)

    def _content(row) -> str:
        meta = row.get("meta") or "{}"
        speaker = _safe_meta(meta).get("speaker")
        role = row.get("role") or "user"
        # 不回落到 ingest_at（入库时刻）：与 store.search 的输出口径一致，
        # "没有时间"就该渲染成 "timestamp: unknown time"，不能伪装成现在。
        ts_ms = row.get("ts") or 0
        seq = row.get("rowid")
        if render is not None:
            # store 侧包装会按 cfg 开关与 mixed_timeline 决定是否渲染 order
            return render(role, speaker, row.get("content") or "", ts_ms,
                          seq=seq)
        # 与 store._result_content 同格式：`{speaker|role}: X; timestamp: ...; content: ...`
        from store import _iso_ms  # 延迟导入：store 与 adjacency 互相引用，顶层互导会炸
        tag = "speaker" if speaker else "role"
        value = speaker or role
        text = row.get("content") or ""
        if not ts_ms:
            if order_meta and seq is not None:
                return (f"{tag}: {value}; timestamp: unknown time; "
                        f"order: #{int(seq)}; content: {text}")
            return f"{tag}: {value}; timestamp: unknown time; content: {text}"
        if order_meta and mixed_timeline and seq is not None:
            return (f"{tag}: {value}; timestamp: {_iso_ms(ts_ms)}; "
                    f"order: #{int(seq)}; content: {text}")
        return f"{tag}: {value}; timestamp: {_iso_ms(ts_ms)}; content: {text}"

    def _entry(row, sid):
        # 延迟导入：store 与 adjacency 互相引用，顶层互导会炸
        from store import MemoryEntry
        return MemoryEntry(
            id=row["id"],
            user_id=user_id,
            session_id=sid,
            role=row.get("role") or "user",
            type="conversation",
            content=_content(row),
            created_at=float(row.get("ts") or 0),
            metadata=_safe_meta(row.get("meta")),
        )

    score_by_id = {e.id: float(s) for s, e in seeds}
    core = list(seeds[:cap]) if cap else list(seeds)
    expanded: list = []
    seen: set[str] = set()

    def add(entry, score) -> None:
        if entry.id in seen:
            return
        if cap is not None and len(expanded) >= cap:
            return
        seen.add(entry.id)
        expanded.append((float(score), entry))

    seed_count = min(int(seed_k), len(core))
    for idx, (seed_score, seed_entry) in enumerate(core):
        if idx >= seed_count:
            break
        add(seed_entry, seed_score)
        sid = getattr(seed_entry, "session_id", "") or ""
        rows = session_rows.get(sid)
        if not rows:
            continue
        pos = next((i for i, r in enumerate(rows) if r["id"] == seed_entry.id), None)
        if pos is None:
            continue
        for distance in range(1, int(window) + 1):
            for j in (pos - distance, pos + distance):
                if 0 <= j < len(rows):
                    row = rows[j]
                    # 邻居用自己的分数；不在候选池里（没被粗排召回）的邻居给 0.0，
                    # 绝不伪装成种子的高分。
                    add(_entry(row, sid), score_by_id.get(row["id"], 0.0))

    # 剩余预算用原始排序结果补满：邻居是追加，不再挤掉尾部真实命中
    for seed_score, seed_entry in core:
        add(seed_entry, seed_score)

    return expanded
