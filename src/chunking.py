"""Text chunking for memory entries (extracted from store.py).

Pure, dependency-light, stateless — safe to unit test in isolation.

两套分词（对齐 vanilla-rag-memory chunking.py:8 的 `TOKEN_PATTERN`）：
  * `tokenize()`        —— 切块**计数**用：汉字单字、英数下划线成词、其余任何非单词
                           非空白字符（标点 / emoji / 俄文…）各算一个 token。
                           把标点算进去，chunk 的真实长度才接近 bge 的 subword 数。
  * `lexical_tokens()`  —— BM25 **词袋**用：在 `tokenize()` 基础上过滤掉"不含字母
                           数字下划线"的纯符号 token，避免标点污染词法打分。
"""
from __future__ import annotations

import re
from bisect import bisect_left

# 切分单位：与 vanilla-rag-memory 保持一致（含标点/其它文字，不再只认中英数）。
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)

# BM25 词干还原开关（feat/bm25-stemming）。索引侧语料与查询侧 query 都走
# `lexical_tokens()`（见 store.py:627 与 :737），改这里同时作用于两侧。
# 关闭它即可在 A/B 中对比"无词干还原"基线；切到不同取值后请在新进程里跑
# （进程内的检索快照缓存 key 不含此项，见 store.py 的 snapshot version）。
ENABLE_STEMMING = True

# 英文词干还原器：优先用 nltk Snowball（无需下载语料），缺失时回退到内置的
# 轻量规则 stemmer。两者都只作用于纯拉丁字母 token，中文单字/数字/符号原样保留。
try:
    from nltk.stem import SnowballStemmer
    _EN_STEMMER = SnowballStemmer("english")
except Exception:  # pragma: no cover - nltk 缺失时优雅降级
    _EN_STEMMER = None


def _rule_stem(token: str) -> str:
    """无 nltk 时的内置轻量词干还原：去掉常见英文后缀，保持索引/查询对称即可。

    不追求语言学正确（agencies->agenci），只要索引与查询走同一规则，形态变体
    两侧就会被归一成同一 token，从而打通 BM25 匹配。
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"          # babies -> baby
    if token.endswith("ing") and len(token) > 4:
        return token[:-3]                # researching -> research
    if token.endswith("ed") and len(token) > 3:
        return token[:-2]                # adopted -> adopt
    if token.endswith("es") and len(token) > 3:
        return token[:-2]                # agencies -> agenci
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]                # cats -> cat
    return token


def tokenize(text: str) -> list[str]:
    """切块计数用：保留标点等符号。"""
    return TOKEN_PATTERN.findall(text or "")


def lexical_tokens(text: str) -> list[str]:
    """BM25 词袋用：casefold + （可选）词干还原 + 丢弃纯符号 token。

    对齐 vanilla-rag-memory memory_store.py:41-46 的 `lexical_tokens()`；
    额外加了英文词干还原以打通 research/researching、adopt/adopted 等形态变体
    （正是此前零召回的一大类根因）。
    """
    out = []
    for token in TOKEN_PATTERN.findall(text or ""):
        if not any(character.isalnum() or character == "_" for character in token):
            continue
        t = token.casefold()
        if ENABLE_STEMMING and t.isascii() and t.isalpha():
            t = _EN_STEMMER.stem(t) if _EN_STEMMER is not None else _rule_stem(t)
        out.append(t)
    return out


def split_memory_text(text: str, chunk_tokens: int = 320,
                      overlap_tokens: int = 40,
                      drop_short_tail: bool = False) -> list[str]:
    """按 token 数滑窗切分；保留原文（用 token 字符偏移回切原始字符串片段）。

    参数先夹取到安全区间：chunk_tokens >= 1 且 0 <= overlap < chunk_tokens。
    否则滑窗不前进（overlap >= chunk_tokens 时 start 不增，chunk_tokens <= 0
    时 start 甚至回退），循环会无限往 chunks 里追加同一段文本、请求永不返回。
    合法参数（默认 320/40）行为不变。

    `drop_short_tail=True` 时丢弃"尾块碎片"：最后一块若不足 `chunk_tokens` 的一半，
    其内容已被前一块的 overlap 覆盖，留着只是给 top_k 塞一条近乎重复的候选。
    默认 False —— 开启会改变块数（进而改变 `chunk_index` 与 `mem_id`），必须同步
    `local_eval/run_eval.py` 里复刻的 provenance 反算公式，故作为显式开关而非默认行为。
    """
    chunk_tokens = max(1, int(chunk_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), chunk_tokens - 1))
    spans = list(TOKEN_PATTERN.finditer(text or ""))
    if not spans:
        return [text] if (text or "").strip() else []
    n = len(spans)
    if n <= chunk_tokens:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        chunk = text[spans[start].start():spans[end - 1].end()]
        chunks.append(chunk)
        if end == n:
            break
        start += chunk_tokens - overlap_tokens
    if drop_short_tail and len(chunks) > 1:
        if len(tokenize(chunks[-1])) < chunk_tokens // 2:
            chunks.pop()
    return chunks


# ---------------------------------------------------------------------------
# 相邻消息打包切块（2026-10-24）
#
# 替换「1 message = 独立切块」的默认行为：灌库时把**同一 session、同一 /add 批**
# 内的相邻消息贪心装进同一个块（<320 token 且 <4 条就继续装），再按 token 滑窗
# 切分。动机：locomo 等短消息数据集单条消息远短于 chunk_tokens（中位数 26、
# 100% ≤320，容量利用率仅 9%），逐条切块把语义相邻的问答对拆散，造成
# 「查询词落在 gold 的上一条消息」类零召回（conv-48#q0021 实例）。
# ---------------------------------------------------------------------------

# 单块消息数硬上限（不加配置开关，直接替换默认行为）。
PACK_MAX_MESSAGES = 4
# chunker 版本标记：进 store._check_identity 的身份串与 db_info 的
# config_fingerprint，保证旧库（逐条切块时代的 mem_id / 块边界）被拒绝复用。
CHUNKER_VERSION = "pack4-v1"


def pack_and_split(texts: list[str], chunk_tokens: int = 320,
                   overlap_tokens: int = 40,
                   max_messages: int = PACK_MAX_MESSAGES,
                   drop_short_tail: bool = False,
                   ) -> list[tuple[str, list[int]]]:
    """贪心打包相邻消息再按 token 滑窗切块（store 与评测侧共用的唯一实现）。

    调用方必须传入**同一 session、同一 /add 批**内的消息 —— store.add_request
    与 local_eval/run_eval.py 的 Adapter 都按 session 逐批调用，天然满足；
    跨 session 由调用方隔开，本函数不做也不需要 session 判别。

    打包规则（2026-10-24 确认口径）：
      * 开放缓冲区 token 数 < chunk_tokens 且块内消息数 < max_messages 时，
        继续装入下一条消息 —— 允许装入后超过 chunk_tokens（「照装，超了再切」）；
      * 装入后 token 数 >= chunk_tokens 或消息数达到 max_messages 即封口：
        超长组用与 split_memory_text 同款滑窗切分，除最后一个不足 chunk_tokens
        的残块外全部定稿为 chunk，**残块保持开放继续接收后续消息**；
      * 批末尾残块定稿；drop_short_tail 只在末批滑窗切分后生效（语义同
        split_memory_text：尾块不足 chunk_tokens 一半且不止一块才丢）。

    返回 [(chunk_text, mis), ...]：mis 是该块覆盖消息的**原始输入下标**（跳过
    空白消息，故可能不连续；跨滑窗边界的消息会同时出现在相邻两块里）。store 侧
    据此取首条消息的 role/speaker/ts、块内全部消息的 event_dates 并集；评测侧
    据此把一个 mem_id 标回多条 dia_id。纯函数、无状态、确定性（同输入同输出）。
    """
    chunk_tokens = max(1, int(chunk_tokens))
    overlap_tokens = max(0, min(int(overlap_tokens), chunk_tokens - 1))
    max_messages = max(1, int(max_messages))

    # 1) 拼接（"\n" 分隔，换行本身不产 token、不膨胀计数）并记录字符区间
    kept: list[int] = []                 # kept[k] = 保留消息的原始下标
    ranges: list[tuple[int, int]] = []   # 每条保留消息在 combined 的 [start, end)
    pieces: list[str] = []
    pos = 0
    for mi, t in enumerate(texts or []):
        t = t or ""
        if not t.strip():
            continue
        pieces.append(t)
        ranges.append((pos, pos + len(t)))
        kept.append(mi)
        pieces.append("\n")
        pos += len(t) + 1
    combined = "".join(pieces)
    spans = list(TOKEN_PATTERN.finditer(combined))
    if not spans:
        return []
    tok_start = [s.start() for s in spans]
    tok_end = [s.end() for s in spans]
    n_tok = len(spans)
    ends = [r[1] for r in ranges]

    def mis_in(a_char: int, b_char: int) -> list[int]:
        """与字符区间 [a, b) 相交的消息原始下标（ranges 不相交且有序 → 连续段）。"""
        i = bisect_left(ends, a_char + 1)     # 第一条 end > a_char
        out: list[int] = []
        while i < len(ranges) and ranges[i][0] < b_char:
            out.append(kept[i])
            i += 1
        return out

    results: list[tuple[str, list[int]]] = []
    buf_t0 = 0    # 开放缓冲区的起始 token 下标
    cur_t = 0     # 缓冲区当前末尾（exclusive）

    def emit(t0: int, t1: int) -> None:
        a, b = tok_start[t0], tok_end[t1 - 1]
        results.append((combined[a:b], mis_in(a, b)))

    def flush(final: bool) -> None:
        """对开放缓冲区 [buf_t0, cur_t) 封口。"""
        nonlocal buf_t0
        if cur_t <= buf_t0:
            return
        if cur_t - buf_t0 <= chunk_tokens:
            emit(buf_t0, cur_t)
            buf_t0 = cur_t
            return
        # 超长组：滑窗切分；最后一个不足 chunk_tokens 的残块保持开放
        start_t = buf_t0
        while cur_t - start_t > chunk_tokens:
            emit(start_t, start_t + chunk_tokens)
            start_t += chunk_tokens - overlap_tokens
        if not final:
            buf_t0 = start_t
            return
        emit(start_t, cur_t)
        buf_t0 = cur_t
        # drop_short_tail：仅对滑窗切出的尾块生效（其内容绝大部分已被前一块
        # 的 overlap 覆盖）；整组装下（未切分）的短块承载真实新内容，不丢。
        if (drop_short_tail and results
                and len(tokenize(results[-1][0])) < chunk_tokens // 2):
            results.pop()

    for k in range(len(ranges)):
        # 缓冲区末尾推进到第 k 条消息最后一个 token 之后
        cur_t = bisect_left(tok_start, ranges[k][1])
        first = bisect_left(ends, tok_start[buf_t0] + 1)
        if (cur_t - buf_t0 >= chunk_tokens
                or k - first + 1 >= max_messages):
            flush(final=False)
    flush(final=True)
    return results


# 切句分隔：句末标点（.!?，可连续如 "...!"）+ 空白，或任何换行。
# 沿用 TOKEN_PATTERN 的 ASCII 语义（英文语料）；句末标点保留在前一句里。
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, min_tokens: int = 4) -> list[str]:
    """把一段文本切成句子（句级多粒度索引的切句单元）。

    规则（纯函数、无状态，可独立单测）：
      * 按 `_SENT_SPLIT_RE` 切分：句末标点后的空白、以及换行都算句界；
        句末标点本身保留在前一句尾部，句子不再含首尾空白；
      * 短句合并：token 数（`tokenize()` 计数，与切块同口径）小于
        `min_tokens` 的句子并入其**前一句**（首个碎片无前句可并时留在原地，
        由后续句子并进来）——碎句单独成行只会浪费一条向量且引入噪声；
      * 空文本返回空列表。

    `min_tokens` 先夹取到 >= 1：0 或负数会让合并判断永远为假，等于不合并。
    """
    text = text or ""
    min_tokens = max(1, int(min_tokens))
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    if not parts:
        return []
    out: list[str] = [parts[0]]
    for part in parts[1:]:
        if len(tokenize(out[-1])) < min_tokens:
            out[-1] = out[-1] + " " + part
        else:
            out.append(part)
    return out
