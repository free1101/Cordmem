"""检索返回层的两级「上下文归拢」重排（纯函数，无 IO）。

背景：top_k=100 的候选里约 27% 槽位被同一条消息的重复切片占据（BEAM 1M 实测），
相似度序把同一话题的切片散落在各处，LLM 看到的上下文混乱、容易被同主题但不对题
的片段误导。归拢把同源内容聚在一起，但**不改候选集合与条数**（只输出一个置换）：

  * L1 merge_same_message：同一消息（键 = request_id + msg_index）的切片聚成一组，
    组内按 chunk_index 升序（并列回落原 rank，保证确定性）；
  * L2 group_adjacent_messages：同一 session 内消息号相邻的消息吸成连续链，
    链内按会话全局序（msg_order）升序；dia_id/段号解析失败的组自成单链。

组 / 链之间仍按「组内最高原始相似度排名」（= 组内最小原下标）排序 —— 归拢不改变
谁排第一，只消除切片与相邻消息的交错。

调用方（MemoryStore.search）在 top_k 截断之后、返回之前套用置换：

    perm, stats = grouped_order(keys, msg_order, merge_same_message=l1,
                                group_adjacent_messages=l2)
    if perm is not None:
        out = [out[i] for i in perm]

`keys[i] = (request_id, msg_index, chunk_index, session_id)`，列表顺序即原始相似度
排名（rank = 下标）。`msg_order[(request_id, msg_index)]` 是该消息在其 session 内的
全局序号（0..M-1，由调用方基于快照全量消息计算）；相邻判定用 `序号差 == 1`，
这样"上段末尾 → 下段开头"的跨段相邻也能吸到一起。
"""
from __future__ import annotations

import re

_SEG_RE = re.compile(r":chunk-(\d+)$")


def parse_segment(request_id: str) -> int | None:
    """'locomo:beam-1:beam-1:s3:chunk-12' -> 12；解析失败返回 None。"""
    if not request_id:
        return None
    m = _SEG_RE.search(request_id)
    return int(m.group(1)) if m else None


def grouped_order(keys: list[tuple[str, int, int, str]],
                  msg_order: dict | None = None, *,
                  merge_same_message: bool = False,
                  group_adjacent_messages: bool = False
                  ) -> tuple[list[int] | None, dict]:
    """返回 (置换, 统计)；无需重排时置换为 None（调用方原样透传）。

    不变量：置换恰为 0..n-1 的一个排列（条数与 id 多重集都不变）。
    两开关均关闭、或候选不足 2 条时返回 (None, stats)。
    """
    stats: dict = {"merge_same_message": bool(merge_same_message),
                   "group_adjacent_messages": bool(group_adjacent_messages)}
    if not merge_same_message and not group_adjacent_messages:
        stats["noop"] = "disabled"
        return None, stats
    n = len(keys)
    if n <= 1:
        stats["noop"] = "too_few_candidates"
        return None, stats

    # ---- 消息分组：mkey = (request_id, msg_index) -> 条目下标列表 ----
    msgs: dict[tuple[str, int], list[int]] = {}
    for i, (rid, mi, _ci, _sid) in enumerate(keys):
        msgs.setdefault((rid, mi), []).append(i)

    if merge_same_message:
        # L1：组内按 chunk_index 升序；并列（理论上不会）回落原 rank 保稳定。
        for lst in msgs.values():
            lst.sort(key=lambda i: (keys[i][2], i))

    if not group_adjacent_messages:
        # 仅 L1：消息组之间按组内最小原 rank 排。
        mks = sorted(msgs, key=lambda mk: min(msgs[mk]))
        perm = [i for mk in mks for i in msgs[mk]]
        stats.update({"messages": len(mks), "chains": len(mks)})
        return perm, stats

    # ---- L2：session 内消息切成极大连续链，链内按全局序、链间按锚 ----
    sess_msgs: dict[str, list[tuple[str, int]]] = {}
    for mk, lst in msgs.items():
        sess_msgs.setdefault(keys[lst[0]][3], []).append(mk)

    chains: list[list[tuple[str, int]]] = []
    for _sid, mks in sess_msgs.items():
        if msg_order:
            # 全量消息算出的全局序：相邻 = 序号差 1（跨段也能接上）。
            # 未知键（不该发生）排到最后、不与任何键相邻。
            mks.sort(key=lambda mk: msg_order.get(mk, 1 << 60))
        else:
            # 兜底：仅按 (段号, msg_index) 排，排序相邻即视为链（可能把
            # 中间缺失的消息也连上）—— 仅在调用方没给 msg_order 时使用。
            mks.sort(key=lambda mk: (parse_segment(mk[0]) is None,
                                     parse_segment(mk[0]) or 0, mk[1]))
        cur = [mks[0]]
        for prev, mk in zip(mks, mks[1:]):
            a, b = msg_order.get(prev), msg_order.get(mk)
            if msg_order and a is not None and b is not None and b == a + 1:
                cur.append(mk)
            elif not msg_order:
                cur.append(mk)
            else:
                chains.append(cur)
                cur = [mk]
        chains.append(cur)

    chains.sort(key=lambda ch: min(min(msgs[mk]) for mk in ch))
    perm = [i for ch in chains for mk in ch for i in msgs[mk]]
    stats.update({"messages": len(msgs), "chains": len(chains)})
    return perm, stats
