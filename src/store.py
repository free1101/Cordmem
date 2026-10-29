"""Memory store + retriever for the AML Add/Search adapter.

Contract-aligned with agentmemoryleaderboard.ai/api-guide, implements the
升级设计 in `doc/我的思路.txt`. Internally organised into three clearly
separated segments (persistence / ingest / retriever); all tunable parameters
come from `config.ServerConfig` (no hard-coded magic numbers).

  Add (add_request): idempotent -> chunk -> embed -> SQLite.
  Search (search):   per-user fetch -> BM25 + cosine fusion -> (optional) rerank
                     -> (optional) same-session adjacency -> rank.

Persistence is SQLite (single Uvicorn worker), replacing the old JSONL.
get_embedder() is a process-wide singleton; constructing MemoryStore triggers
one model load and an embedding-identity check (switching models / context
encoding requires rebuilding the DB).

会话内顺序（#16 + #B1，对齐 vanilla-rag-memory 的 `ORDER BY rowid`）：
  chunk 行的会话内全序 = **插入顺序**，即 SQLite 的 rowid。它成立依赖两个不变式：
    ① 只用 `INSERT`，永不 `DELETE` / `INSERT OR REPLACE`（REPLACE = delete+insert，
       会重分配 rowid；DELETE 后再 INSERT 还会**复用**已释放的 rowid，
       使重发的消息从会话中段跳到末尾，且无任何报错）
    ② 灌库顺序 == 对话顺序
  破坏任一点都会让 `adjacency.expand()` 按 `pos ± window` 拉错邻居。
  因此 `add_request()` 遇到 request_id 冲突时改为抛 `RequestConflictError`（-> HTTP 409），
  不再"删旧块覆盖"；`msg_index` / `chunk_index` 保留但**只做诊断**，不参与排序。
  会话内全序 = `rowid`（插入序）：邻接扩展据此在 rowid 相邻处取同 session 邻居，
  与 vanilla-rag-memory 的 `ORDER BY rowid`（memory_store.py:459）一致；跨 request
  的乱序灌库不再用时间戳重排（与 vanilla 行为一致）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from calendar import monthrange
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from rank_bm25 import BM25Okapi

from chunking import (CHUNKER_VERSION, PACK_MAX_MESSAGES, TOKEN_PATTERN,
                      lexical_tokens, pack_and_split, split_sentences)
from config import ServerConfig, load_config
from context_grouping import grouped_order, parse_segment
from dateparse import parse_event_dates, parse_time_constraint
from embedder import get_embedder
from expander import bm25_query_text, dense_query_text, get_expander

# 小于该值视为秒时间戳，自动 ×1000 归一化为毫秒（兼容老数据）。
_SECONDS_THRESHOLD = 1_000_000_000_000

# source_timestamp 写入时用 0 表示"该 message 没有可信时间"：
# SQLite 无法把既有列从 NOT NULL 改成 nullable（需重建表），故用哨兵值代替 NULL。
# 0 必须一路传到输出层：content 前缀渲染成 "timestamp: unknown time"、
# created_at 输出 null —— **绝不回落到 ingest_at（入库时刻）**，那会把"没有时间"
# 伪装成"现在"，给答案模型喂错误证据，且每次重灌库都变、结果不可复现。
_UNKNOWN_TS = 0


def _to_ms_or_none(value) -> int | None:
    """把平台传入的 timestamp 归一化成毫秒；无时间戳时返回 None（不是"现在"）。

    返回 None 而非 `time.time()*1000`，是为了让"没有时间"这个事实可区分：
    固化成入库时刻会让同一批灌库每次得到不同的排序键，结果不可复现。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and value < _SECONDS_THRESHOLD:
        return int(float(value) * 1000.0)
    return int(float(value))


def _iso_ms(ts_ms: float | int) -> str:
    """毫秒 -> RFC3339 UTC（...Z）。"""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


# ---- forget tombstone（遗忘降权，见 search() 内 forget_penalty 块）----
# 用户说过 "Please forget ..." 的主题，其相关 chunk 在融合分上打折。检索期
# 即时构建墓碑（扫快照原文，缓存进 snap），完全不触碰生成 prompt —— 治
# "被遗忘的旧偏好与提问词面最相关而霸占 top_k，模型没条件区分新旧"。
_FORGET_RE = re.compile(r"\bforget\w*", re.I)
# 真删除请求（语料级闸门 forget_genuine_gate 的判据）：祈使式
# forget/erase/delete/remove/wipe + 明确宾语引导词，或 assistant 确认句
# "I'll forget that ..."。口语提及不算（"I'll never forget the day..." /
# "Don't forget to..." / "forget my troubles"）。语料级实证（2026-10-19）：
# PMv2 128k 真删除 114+ 条（模板化 "Please forget that/my ..."），locomo 与
# PMv1 1M 均为 0 —— tombstone 在没有真删除请求的语料上是纯噪声。
_FORGET_REQUEST_RE = re.compile(
    r"(?:^|[.!?]\s+|\n)\s*(?:please\s+|now\s+|just\s+|also\s+)*"
    r"(?:forget|erase|delete|remove|wipe)\s+"
    r"(?:that|this|about|my|the|everything|all|what|how|when|where|who|any|"
    r"each|i\s+said|we\s+(?:talked|discussed|spoke))\b"
    r"|\bi(?:'ll|’ll|\s+will)\s+(?:also\s+|just\s+)?forget\s+"
    r"(?:that|this|about|my|the|everything|what|how|when|where|you)\b",
    re.I)
# forget 句里的功能词不算主题词（主题词 = 句子的实义内容词）
_FORGET_STOP = frozenset({
    "forget", "forgetting", "please", "preference", "preferences",
    "information", "detail", "about", "that", "want", "would", "really",
    "stuff", "thing", "things", "favor",
})


# ---------------------------------------------------------------------------
# 输出期时间增强（对齐 vanilla-rag-memory memory_store.py:49-125 / 585-606）
# ---------------------------------------------------------------------------
def _shift_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    day = min(value.day, monthrange(year, zero_based_month + 1)[1])
    return value.replace(year=year, month=zero_based_month + 1, day=day)


def _shift_years(value: datetime, years: int) -> datetime:
    year = value.year + years
    day = min(value.day, monthrange(year, value.month)[1])
    return value.replace(year=year, day=day)


def _format_date(value: datetime) -> str:
    return f"{value:%B} {value.day}, {value.year}"


def _format_month(value: datetime) -> str:
    return f"{value:%B} {value.year}"


def _format_week(value: datetime) -> str:
    start = value - timedelta(days=value.weekday())
    end = start + timedelta(days=6)
    return f"{_format_date(start)} through {_format_date(end)}"


def _relative_time_annotations(content: str, message_time: datetime) -> list[str]:
    """把 "yesterday" / "last week" / "3 days ago" 解析成绝对日期。

    只在**输出期**追加到 content，不进索引（不污染 BM25 词袋 / 向量）。
    """
    rules = (
        (r"\blast night\b", _format_date(message_time - timedelta(days=1))),
        (r"\byesterday\b", _format_date(message_time - timedelta(days=1))),
        (r"\btomorrow\b", _format_date(message_time + timedelta(days=1))),
        (r"\b(?:today|tonight)\b", _format_date(message_time)),
        (r"\blast week\b", _format_week(message_time - timedelta(weeks=1))),
        (r"\bthis week\b", _format_week(message_time)),
        (r"\bnext week\b", _format_week(message_time + timedelta(weeks=1))),
        (r"\blast month\b", _format_month(_shift_months(message_time, -1))),
        (r"\bthis month\b", _format_month(message_time)),
        (r"\bnext month\b", _format_month(_shift_months(message_time, 1))),
        (r"\blast year\b", str(message_time.year - 1)),
        (r"\bthis year\b", str(message_time.year)),
        (r"\bnext year\b", str(message_time.year + 1)),
    )
    annotations: list[str] = []
    seen: set[str] = set()
    for pattern, resolved in rules:
        for match in re.finditer(pattern, content, flags=re.IGNORECASE):
            phrase = match.group(0).casefold()
            if phrase not in seen:
                annotations.append(f"{match.group(0)} = {resolved}")
                seen.add(phrase)

    quantities = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
    ago_pattern = re.compile(
        r"\b(?P<count>a|an|one|two|three|\d+)\s+"
        r"(?P<unit>day|week|month|year)s?\s+ago\b",
        flags=re.IGNORECASE,
    )
    for match in ago_pattern.finditer(content):
        phrase = match.group(0).casefold()
        if phrase in seen:
            continue
        raw_count = match.group("count").casefold()
        count = quantities.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
        unit = match.group("unit").casefold()
        if unit == "day":
            resolved_time = message_time - timedelta(days=count)
        elif unit == "week":
            resolved_time = message_time - timedelta(weeks=count)
        elif unit == "month":
            resolved_time = _shift_months(message_time, -count)
        else:
            resolved_time = _shift_years(message_time, -count)
        annotations.append(f"{match.group(0)} = {_format_date(resolved_time)}")
        seen.add(phrase)
    return annotations


class RequestConflictError(ValueError):
    """同一 request_id 被用于不同 payload / user_id / session_id。

    对齐 vanilla-rag-memory memory_store.py:37 + :362-368：绝不静默覆盖，
    由 app 层映射成 HTTP 409。
    """


def _content(m):
    return m.content if hasattr(m, "content") else m.get("content", "")


def _role(m):
    return m.role if hasattr(m, "role") else m.get("role", "user")


def _ts(m):
    return m.timestamp if hasattr(m, "timestamp") else m.get("timestamp")


def _speaker(m):
    return m.speaker if hasattr(m, "speaker") else m.get("speaker")


@dataclass
class MemoryEntry:
    id: str
    user_id: str = "global"
    session_id: str = ""
    role: str = "user"
    type: str = "conversation"
    content: str = ""
    created_at: float = field(default_factory=lambda: time.time() * 1000.0)
    metadata: dict = field(default_factory=dict)

    def created_at_iso(self) -> str:
        # 官方 timestamp 为毫秒；输出 RFC3339 UTC（...Z）。
        dt = datetime.fromtimestamp(self.created_at / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryStore:
    def __init__(self, cfg: "ServerConfig | str | None" = None,
                 db_path: str | None = None):
        # Accepts either:
        #   MemoryStore(cfg)            -> explicit ServerConfig
        #   MemoryStore(db_path="x")    -> legacy keyword
        #   MemoryStore("x")            -> legacy positional (treated as db_path)
        if isinstance(cfg, ServerConfig):
            config = cfg
        elif isinstance(cfg, str):
            db_path = cfg
            config = load_config()
        else:
            config = load_config()
        if db_path is not None:
            config.db_path = db_path
        self.cfg = config
        _db_dir = os.path.dirname(config.db_path)
        if _db_dir:
            os.makedirs(_db_dir, exist_ok=True)
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False,
                                     timeout=60)
        self._conn.isolation_level = None  # 手动管理事务（BEGIN IMMEDIATE）
        # 按名字取列，杜绝 `r[4]` / `r[5]` 这类位置错位（对齐 vanilla memory_store.py:247）
        self._conn.row_factory = sqlite3.Row
        # WAL + 放宽 busy_timeout：读写并发不再互相阻塞（#14）
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # 只保护**写事务**：检索的打分/推理全程不持这把锁（#C1 / #D6）。
        self._lock = threading.RLock()
        # 模型推理不是线程安全的，单独一把锁（对齐 vanilla memory_store.py:242）。
        self._embedding_lock = threading.RLock()
        # 检索快照缓存（user_id -> 快照），见 _load_snapshot 的注释。
        self._snapshot_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._init_db()
        self._check_identity()

    def _get_embedder(self):
        """按当前配置取进程内单例 embedder（查询前缀真正传进模型，#D13）。

        前缀三态直接透传给 Embedder，不再用 `or None` 塌缩：
          - cfg 为 None（默认/不设）= 自动前缀（BGE 英文模型加官方前缀）
          - cfg 为 ""（显式空串）= 零前缀（关掉）
          - cfg 为其它字符串 = 自定义前缀
        """
        return get_embedder(
            # 模型名显式取配置（YAML / override 优先于环境变量）：换 embedding 模型
            # 必须同时换库，否则 _check_identity() 的 identity 就与实际编码行为脱钩。
            model_name=self.cfg.embedding_model or None,
            query_prefix=self.cfg.embedding_query_prefix,
            max_seq_length=self.cfg.embedding_max_seq_length,
            backend=self.cfg.embedding_backend or None,
            api_base=self.cfg.embedding_api_base or None,
            api_dimensions=self.cfg.embedding_api_dimensions,
            api_batch_size=self.cfg.embedding_api_batch_size,
        )

    def _get_expander(self):
        """按配置取进程内单例扩展器（R3/R4 的查询侧）。

        **默认关闭**：`expand_enabled=False` 时本方法直接返回一个空扩展器，一次
        `expand()` 都不调、一次文件 IO 都不做 —— 关掉扩展的开销就是零，且检索结果
        与改版前逐位一致。

        设了开关但数据文件缺失（线上平台就是这样，`data/hyperE/` 不随 `src/` 提交）
        时，`expander` 会打 WARNING 并把自己变成"不可用"，这里照常返回它，
        `search()` 只要判 `available` 即可 —— **绝不因为缺文件而抛异常**。
        """
        if not self.cfg.expand_enabled:
            return get_expander(relations=())
        files = {}
        for item in self.cfg.expand_files or []:
            rel, _, path = str(item).partition("=")
            if rel.strip() and path.strip():
                files[rel.strip()] = path.strip()
        return get_expander(
            relations=tuple(self.cfg.expand_relations or ()),
            files=files,
            base_dir=self.cfg.expand_dir,
            per_token_n=int(self.cfg.expand_per_token_n),
            top_n=int(self.cfg.expand_top_n),
            max_graph_distance=float(self.cfg.expand_max_graph_distance),
            min_token_len=int(self.cfg.expand_min_token_len),
            multiword=self.cfg.expand_multiword,
            drop_stopwords=bool(self.cfg.expand_drop_stopwords),
            skip_capitalized=bool(self.cfg.expand_skip_capitalized),
            verify_pairs=bool(self.cfg.expand_verify_pairs),
        )

    # ==================================================================
    # 持久化段 (Persistence) — SQLite access only, no scoring logic
    # ==================================================================
    def _init_db(self):
        # db_meta 固化 embedding 身份（模型+查询前缀+上下文编码标记），旧库首次打开写入。
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS db_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            )""")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS add_requests (
                request_id   TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL,
                user_id      TEXT NOT NULL,
                session_id   TEXT NOT NULL,
                created_at   INTEGER NOT NULL
            )""")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id               TEXT PRIMARY KEY,
                request_id       TEXT NOT NULL,
                user_id          TEXT NOT NULL,
                session_id       TEXT NOT NULL,
                role             TEXT NOT NULL,
                type             TEXT NOT NULL,
                content          TEXT NOT NULL,
                source_timestamp INTEGER NOT NULL,   -- 0 = 该 message 无可信时间
                ingest_at        INTEGER NOT NULL DEFAULT 0,  -- 入库时刻，仅诊断（不参与排序/输出时间）
                embedding        BLOB NOT NULL,
                dimension        INTEGER NOT NULL,
                store_metadata   TEXT,
                msg_index        INTEGER NOT NULL DEFAULT 0,   -- 诊断用，不参与排序
                chunk_index      INTEGER NOT NULL DEFAULT 0,   -- 诊断用，不参与排序
                FOREIGN KEY (request_id) REFERENCES add_requests(request_id)
            )""")
        self._migrate_memories_position()
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
        # ---- 句级多粒度索引表（sent_index_enabled，默认关）----
        # 每个 chunk 切出的句子各存一条向量，外键挂到 memories.id。纯新增表：
        # 不动 memories 的 INSERT-only 语义（会话内全序 = rowid 不受影响），
        # 也不触发 _check_identity（chunk 向量的编码语义未变）。
        # 旧库没有此表时检索侧句级重排自动 no-op（_load_sentences 捕获
        # OperationalError 后按「无句级行」处理，只警告一次）。
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_sentences (
                sent_id     TEXT PRIMARY KEY,
                mem_id      TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                sent_index  INTEGER NOT NULL,
                content     TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                dimension   INTEGER NOT NULL,
                FOREIGN KEY (mem_id) REFERENCES memories(id)
            )""")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_sentences_user "
            "ON memory_sentences(user_id)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_sentences_mem "
            "ON memory_sentences(mem_id)")

    def _migrate_memories_position(self):
        """补齐 msg_index / chunk_index / ingest_at 三列（#16 + #D19）。

        `msg_index` / `chunk_index` 仅作**诊断**：哈希后的 id 不可逆，排查"某条证据
        为什么没召回"时需要直接看出它是第几条 message 的第几块。排序不再用它们
        （见模块 docstring：会话内全序 = rowid）。

        旧库迁移要点（CREATE TABLE IF NOT EXISTS 对已存在的表不加列，故此处 ALTER）：
          * 新加列一律 `NOT NULL DEFAULT 0`，保证迁移库与新建库 schema 完全一致
            （此前 ALTER 出来的是 nullable 无默认列，与新建表的 NOT NULL DEFAULT 0 分叉）；
          * 回填条件**逐列判断**（此前只看 `msg_index IS NULL`，将来只加一列就会漏）；
          * `chunk_index` / `ingest_at` 按 rowid / source_timestamp 回填。
        """
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
        if "msg_index" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN msg_index INTEGER NOT NULL DEFAULT 0")
        if "chunk_index" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN chunk_index INTEGER NOT NULL DEFAULT 0")
        if "ingest_at" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN ingest_at INTEGER NOT NULL DEFAULT 0")
        self._conn.execute(
            "UPDATE memories SET msg_index = 0 WHERE msg_index IS NULL")
        # chunk_index 是"message 内第几块"，旧库没有 msg_index、id 又是哈希，
        # 反推不出来 —— 统一填 0 表示"未知"。此前用全局 rowid 回填，会把块下标
        # 显示成 301、302…（必然错），诊断字段反而误导排查。
        self._conn.execute(
            "UPDATE memories SET chunk_index = 0 WHERE chunk_index IS NULL")
        self._conn.execute(
            "UPDATE memories SET ingest_at = source_timestamp "
            "WHERE ingest_at IS NULL OR ingest_at = 0")

    def _check_identity(self):
        """Embedding 身份校验（基线硬约束的升级版，#11 + #D11 + #D13）。

        固化的身份串直接取自 `Embedder.identity`（= 模型名 + 查询前缀的实际序列化），
        不再自己拼 JSON —— 保证"身份 == 真实编码行为"，杜绝改了配置身份没变、
        或身份变了行为没变的静默错配（对齐 vanilla-rag-memory embeddings.py:24）。

        上下文编码把**具体条数**编进身份（`|context=v1:{n}`，对齐 vanilla
        memory_store.py:209-213）：ctx=1 与 ctx=2 是两套语义，必须视为不同库；
        ctx=0 则不加后缀，与历史无 ctx 的库兼容。

        旧库（无 db_meta）首次打开时写入当前身份，等价于「登记」，后续变更才触发拒绝。
        """
        emb = self._get_embedder()
        expected = emb.identity
        if self.cfg.context_messages > 0:
            expected = f"{expected}|context=v1:{self.cfg.context_messages}"
        # 切块算法版本（2026-10-24 起相邻消息打包为默认行为）：块边界与 mem_id
        # 全变，旧库（逐条切块时代）必须拒绝打开 —— 无条件加后缀，保证旧库一律
        # 判 mismatch，杜绝新旧块混库。
        expected = f"{expected}|chunk={CHUNKER_VERSION}"
        try:
            cur = self._conn.execute(
                "SELECT value FROM db_meta WHERE key='embedding_identity'")
            row = cur.fetchone()
        except sqlite3.OperationalError:
            row = None
        if row is None:
            self._conn.execute(
                "INSERT OR REPLACE INTO db_meta(key, value) VALUES('embedding_identity', ?)",
                (expected,))
            return
        if row[0] != expected:
            self._conn.close()
            raise RuntimeError(
                "Embedding identity mismatch: stored identity differs from the "
                f"current config. Stored={row[0]} current={expected}. Changing the "
                "embedding model, query prefix or context encoding requires rebuilding "
                "the memory DB (move local_data/memories.db aside and re-run Add). "
                "See doc/PROJECT_LOG.md.")

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM memories")
        return int(cur.fetchone()[0])

    def close(self):
        self._conn.close()

    # ==================================================================
    # 写入段 (Ingest) — dedup-before-embed -> chunk -> embed -> store
    # ==================================================================
    def _payload_hash(self, user_id, session_id, messages) -> str:
        norm = [{"role": _role(m), "content": _content(m), "timestamp": _ts(m),
                 "speaker": _speaker(m)}
                for m in messages]
        blob = json.dumps({"u": user_id, "s": session_id, "m": norm},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _context_excerpt(text: str, token_limit: int, from_end: bool) -> str:
        """上下文编码用：把邻居文本截断到 `token_limit` 个 token。

        对齐 vanilla-rag-memory memory_store.py:344-353。不截断的话，长会话里
        每个 chunk 都要拼上完整邻居，emb_text 体积随 ctx_n × 邻居长度膨胀。
        """
        matches = list(TOKEN_PATTERN.finditer(text or ""))
        if not matches:
            return ""
        if len(matches) <= token_limit:
            return (text or "").strip()
        if from_end:
            return text[matches[-token_limit].start():matches[-1].end()].strip()
        return text[matches[0].start():matches[token_limit - 1].end()].strip()

    def add_request(self, user_id: str, session_id: str, request_id: str,
                    messages: list) -> None:
        """写入一个 Add 请求：幂等判重(向量化前) -> 切块 -> 向量化 -> SQLite 落库。

        `messages` 可为 pydantic 模型或 dict（兼容 app.py 与脚本传入）。
        与 vanilla-rag-memory 对齐：方法顶部先按 request_id 判重，命中相同
        payload 直接返回，避免在重复 /add 时白做最贵的切块 + 向量化。

        **冲突即拒绝**（#D7，对齐 vanilla memory_store.py:362-368）：同一 `request_id`
        若已用于不同的 `user_id` / `session_id` / payload，抛 `RequestConflictError`
        （app 层 -> HTTP 409），**绝不删除旧块覆盖**。覆盖会 DELETE 后重新 INSERT，
        SQLite 会复用已释放的 rowid，使重发的消息从会话中段跳到末尾，
        而会话内全序 = rowid（见模块 docstring），于是邻接扩展会静默拉错邻居。

        增强（默认关闭，由 config 开关启用）：
          - 说话人前缀（#9 明确保留）
          - 时间增强：**输出期**渲染日期 / 相对时间解析（#D10，不再烤进索引 content）
          - 上下文编码：索引时把前后文拼入「向量」（不改 content），改变向量语义，
            故必须配合 _check_identity 固化（#7 + #11 同批）
        """
        payload_hash = self._payload_hash(user_id, session_id, messages)

        # ---- 先判重 / 判冲突（在切块/向量化之前）----
        with self._lock:
            cur = self._conn.execute(
                "SELECT user_id, session_id, payload_hash FROM add_requests "
                "WHERE request_id=?", (request_id,))
            row = cur.fetchone()
            if row is not None:
                if (row["user_id"] != user_id or row["session_id"] != session_id
                        or row["payload_hash"] != payload_hash):
                    raise RequestConflictError(
                        f"request_id {request_id!r} was already used with a different "
                        "payload / user_id / session_id")
                return  # 完全相同 -> 幂等无操作（不切块、不向量化、不加载 embedder）

        emb = self._get_embedder()
        dim = emb.dimension
        chunk_tokens = self.cfg.chunk_tokens
        chunk_overlap = self.cfg.chunk_overlap
        ctx_n = self.cfg.context_messages

        # 预计算每个 message 的「上下文编码」前缀（仅依赖 message 下标 mi）
        n_msg = len(messages)
        ctx_prefix_by_mi = {}
        if ctx_n > 0:
            if n_msg == 1:
                # 上下文窗口只在**单次 add 的 messages 列表内**取邻居；平台常见的
                # 「一条 message 一个 request_id」用法下 n_msg==1 -> ctx 恒为空，
                # 而 identity 却会把 context=True 固化（关掉就拒绝开库）。必须让"开了没效果"可见。
                print(f"[warn] context_messages={ctx_n} but this /add carries a single "
                      f"message (request_id={request_id!r}); context encoding is a no-op "
                      "here. Batch multiple messages per request to make it effective.",
                      flush=True)
            # 单条上下文截断到 excerpt_tokens 个 token，避免长会话把 emb_text 撑爆
            # （对齐 vanilla memory_store.py:328 / :344-353）
            excerpt_tokens = max(16, min(96, chunk_tokens // 3))
            for mi in range(n_msg):
                lines = [f"{_role(messages[mi])}: {_content(messages[mi])}"]
                for distance in range(1, ctx_n + 1):
                    prev_idx = mi - distance
                    if prev_idx >= 0:
                        body = self._context_excerpt(
                            _content(messages[prev_idx]), excerpt_tokens, from_end=True)
                        if body:
                            lines.append(f"Previous {_role(messages[prev_idx])}: {body}")
                    next_idx = mi + distance
                    if next_idx < n_msg:
                        body = self._context_excerpt(
                            _content(messages[next_idx]), excerpt_tokens, from_end=False)
                        if body:
                            lines.append(f"Next {_role(messages[next_idx])}: {body}")
                ctx_prefix_by_mi[mi] = "\n".join(lines)

        # 逐消息预处理好消息级字段（2026-10-24 起灌库改为「相邻消息打包切块」，
        # 一个 chunk 可能覆盖多条消息，故先把消息级字段收好再按块归属）。
        msg_raw: list[str] = []
        msg_role: list[str] = []
        msg_speaker: list[str] = []
        msg_ts: list[int] = []
        msg_ev: dict[int, list[int]] = {}
        for mi, m in enumerate(messages):
            raw = _content(m)
            role = _role(m)
            speaker_label = _speaker(m) or role   # 无 speaker 时退化为 role
            ts_ms = _to_ms_or_none(_ts(m))
            # ---- 灌库期事件日期提取（temporal_event_dates，默认开）----
            # 把原文里的相对时间（last night / yesterday / 3 days ago ...）
            # 解析成绝对日期存进 metadata，供检索期时间窗比对 —— 修复
            # 事件日 ≠ 发送日的偏移（发送时间 source_timestamp 不动）。
            # payload_hash 不含 metadata：此字段不动 mem_id / 幂等语义；
            # 旧库没有该字段时检索期自动退化为只比发送时间。
            if self.cfg.temporal_event_dates:
                ev = parse_event_dates(raw, ts_ms)
                if ev:
                    msg_ev[mi] = ev
            msg_raw.append(raw)
            msg_role.append(role)
            msg_speaker.append(speaker_label)
            msg_ts.append(_UNKNOWN_TS if ts_ms is None else ts_ms)

        # 打包切块（pack_and_split：与 local_eval/run_eval.py 的 provenance 共享
        # 同一实现，块边界 / mem_id 公式天然一致）。返回 (chunk_text, mis)。
        packs = pack_and_split(msg_raw, chunk_tokens, chunk_overlap,
                               max_messages=PACK_MAX_MESSAGES,
                               drop_short_tail=self.cfg.drop_short_tail)

        # 先收集本批所有 (chunk, emb_text, mi, ci, role, ts, meta)，再整批 embed 一次。
        pending = []  # (content, emb_text, mi, ci, role, ts, meta)；mi = 块内首条消息
        now_ms = int(time.time() * 1000.0)
        for ci, (chunk, mis) in enumerate(packs):
            fi = mis[0]           # 块内首条消息：role/speaker/ts/ctx 前缀的归属者
            role = msg_role[fi]
            # 角色/说话人前缀不烤入索引/向量/BM25 语料，仅在 search 返回时拼回，
            # 对齐 vanilla（其只在输出层拼 [role | timestamp]）。
            # 时间增强 likewise 只在输出期（#D10）——索引期烤日期会把 "2024/01/01"
            # 这类噪声塞进 BM25 词袋。
            ts = msg_ts[fi]
            # 上下文编码：拼入向量文本（不改 content 本身）；打包块取首条消息的
            # 上下文前缀（ctx_n 默认 0，多数部署不经过此分支）。
            ctx_pre = ctx_prefix_by_mi.get(fi, "")
            # event_dates 取块内全部消息的并集去重排序（消息级粒度不丢）。
            ev_set = sorted({d for mi in mis for d in msg_ev.get(mi, [])})
            emb_text = (ctx_pre + "\n" + chunk) if ctx_pre else chunk
            meta_obj = {"role": role, "speaker": msg_speaker[fi]}
            if ev_set:
                meta_obj["event_dates"] = ev_set
            meta = json.dumps(meta_obj, ensure_ascii=False)
            pending.append((chunk, emb_text, fi, ci, role, ts, meta))

        rows = []
        sent_rows = []   # memory_sentences 行（仅 sent_index_enabled 时非空）
        if pending:
            # ---- 句级多粒度（sent_index_enabled，默认关）----
            # 每个 chunk 切句，句子文本并入**同一批** encode_documents ——
            # 只增大一次调用的批量，不增加编码请求数。句 id 由
            # (request_id, pack, ci, si) 派生，与 chunk 的 mem_id 同一确定性口径。
            sent_texts: list[str] = []
            sent_keys: list[tuple[int, int]] = []   # (pending 下标, 句序号 si)
            if self.cfg.sent_index_enabled:
                for pidx, p in enumerate(pending):
                    for si, sent in enumerate(
                            split_sentences(p[0], self.cfg.sent_min_tokens)):
                        sent_texts.append(sent)
                        sent_keys.append((pidx, si))
            with self._embedding_lock:
                # 整批一次编码：chunk emb_text + 句级文本
                vecs = emb.encode_documents([p[1] for p in pending] + sent_texts)
            chunk_vecs = vecs[:len(pending)]
            sent_vecs = vecs[len(pending):]
            cids: list[str] = []
            for (chunk, _emb_text, mi, ci, role, ts, meta), vec in zip(pending, chunk_vecs):
                # mem_id 公式（2026-10-24 打包切块版）：sha256(request_id:pack:ci)。
                # ci = 本 /add 批内 chunk 全局序号；"pack" 是版本标记，与逐条切块
                # 时代的旧 id 空间零碰撞。local_eval/run_eval.py 的 build_provenance
                # 调同一个 pack_and_split 复算，两侧不可能漂移。
                cid = "mem_" + hashlib.sha256(
                    f"{request_id}:pack:{ci}".encode("utf-8")).hexdigest()[:24]
                cids.append(cid)
                rows.append((
                    cid, request_id, user_id, session_id, role, "conversation",
                    chunk, ts, now_ms,
                    sqlite3.Binary(np.ascontiguousarray(vec, dtype=np.float32).tobytes()),
                    dim, meta, mi, ci,   # msg_index = 块内首条消息下标
                ))
            for (pidx, si), sent, svec in zip(sent_keys, sent_texts, sent_vecs):
                ci = pending[pidx][3]
                sent_id = "sent_" + hashlib.sha256(
                    f"{request_id}:pack:{ci}:{si}".encode("utf-8")).hexdigest()[:24]
                sent_rows.append((
                    sent_id, cids[pidx], user_id, session_id, si, sent,
                    sqlite3.Binary(np.ascontiguousarray(svec, dtype=np.float32).tobytes()),
                    dim,
                ))

        # BEGIN IMMEDIATE 事务：复查幂等 + 写入（防并发竞态；顶层 miss 时此处兜底）
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self._conn.execute(
                    "SELECT user_id, session_id, payload_hash FROM add_requests "
                    "WHERE request_id=?", (request_id,))
                row = cur.fetchone()
                now = int(time.time() * 1000.0)
                if row is not None:
                    if (row["user_id"] != user_id or row["session_id"] != session_id
                            or row["payload_hash"] != payload_hash):
                        raise RequestConflictError(
                            f"request_id {request_id!r} was already used with a "
                            "different payload / user_id / session_id")
                    self._conn.execute("COMMIT")
                    return  # 并发竞态下已被其他请求写入 -> 跳过
                self._conn.execute(
                    "INSERT INTO add_requests(request_id, payload_hash, user_id, "
                    "session_id, created_at) VALUES(?,?,?,?,?)",
                    (request_id, payload_hash, user_id, session_id, now))
                # 纯 INSERT：不能用 INSERT OR REPLACE / 不能先 DELETE 再插，
                # 否则 rowid 会被重分配或复用，会话内全序（= rowid）就毁了。
                self._conn.executemany(
                    """INSERT INTO memories
                       (id, request_id, user_id, session_id, role, type, content,
                        source_timestamp, ingest_at, embedding, dimension,
                        store_metadata, msg_index, chunk_index)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
                # 句级行与 chunk 同事务写入：外键（mem_id → memories.id）在
                # memories 行落库后立即满足，且失败一起 ROLLBACK，不留半批。
                if sent_rows:
                    self._conn.executemany(
                        """INSERT INTO memory_sentences
                           (sent_id, mem_id, user_id, session_id, sent_index,
                            content, embedding, dimension)
                           VALUES(?,?,?,?,?,?,?,?)""", sent_rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ==================================================================
    # 检索段 (Retriever) — per-user fetch -> fuse -> (rerank) -> (adjacency)
    # ==================================================================
    # 检索快照缓存上限（按 user_id 计）：评测 / 线上同时被检索的 user 通常很少，
    # 超出后按 LRU 淘汰，避免长期驻留把内存吃满。
    _SNAPSHOT_CACHE_SIZE = 8

    def _load_snapshot(self, user_id: str) -> dict:
        """取该 user 的全量候选快照（进程内缓存，key = user_id）。

        为什么缓存：单次 search() 的耗时几乎全在"反序列化全部向量 + 重新分词建
        BM25"上，而平台是边 Add 边 Search —— 不缓存时 /search 的延迟会随该 user
        的灌库量线性增长。

        新鲜度由版本键 `(行数, MAX(rowid))` 保证：本库只 INSERT、不 DELETE /
        REPLACE（见模块 docstring），故二者同时不变 == 数据未变，任一项变化即
        重建。判定与读行在**同一个 _lock 临界区**内完成，不会出现"版本旧、行新"。

        缓存只到"从 DB 读出的数据"及其**只读**派生结果（向量矩阵、BM25 语料与
        BM25 对象、session 分组）；打分 / 融合 / 重排仍每次现算，不受缓存影响。
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM memories WHERE user_id=?",
                (user_id,))
            row = cur.fetchone()
            version = (int(row[0]), int(row[1]))
            cached = self._snapshot_cache.get(user_id)
            if cached is not None and cached["version"] == version:
                self._snapshot_cache.move_to_end(user_id)
                return cached
            cur = self._conn.execute(
                "SELECT id, content, source_timestamp, ingest_at, session_id, "
                "embedding, dimension, role, store_metadata, rowid, "
                "request_id, msg_index, chunk_index "
                "FROM memories WHERE user_id=? "
                "ORDER BY rowid", (user_id,))
            fetched = cur.fetchall()
        # 反序列化 / 分词是 CPU 密集，放到锁外做。
        snap = self._build_snapshot(fetched, version, user_id)
        self._snapshot_cache[user_id] = snap
        self._snapshot_cache.move_to_end(user_id)
        while len(self._snapshot_cache) > self._SNAPSHOT_CACHE_SIZE:
            self._snapshot_cache.popitem(last=False)
        return snap

    def _build_snapshot(self, fetched, version, user_id: str) -> dict:
        """把 DB 行解析成检索用的内存快照（纯计算：不持锁、不查库）。"""
        snap = {"version": version, "ids": [], "contents": [], "timestamps": [],
                "ingest_ats": [], "sessions": [], "roles": [], "metas": [],
                "rowids": [],
                "request_ids": [], "msg_indexes": [], "chunk_indexes": [],
                "corpus": [], "bm25": None, "session_rows": {},
                "dimension": 0, "mat": np.zeros((0, 0), dtype=np.float32)}
        if not fetched:
            return snap
        # 维度校验（#D4）：逐个检查而不是只看第一行；读回时显式带上 dimension。
        dims = {int(r["dimension"]) for r in fetched}
        if len(dims) > 1:
            raise ValueError(
                f"memories of user_id={user_id!r} mix embedding dimensions "
                f"{sorted(dims)}; rebuild the memory DB (see doc/PROJECT_LOG.md)")
        dimension = dims.pop()
        snap["dimension"] = dimension
        embeddings = []
        for r in fetched:
            snap["ids"].append(r["id"])
            snap["contents"].append(r["content"] or "")
            snap["timestamps"].append(int(r["source_timestamp"] or 0))
            snap["ingest_ats"].append(int(r["ingest_at"] or 0))
            snap["sessions"].append(r["session_id"] or "")
            snap["roles"].append(r["role"] or "user")
            snap["metas"].append(r["store_metadata"] or "{}")
            # rowid：全局插入序（fetched 已 ORDER BY rowid，逐行取出即对齐）。
            # 仅用于无时间条目的 order 渲染，不参与任何排序/打分。
            snap["rowids"].append(int(r["rowid"]))
            # 消息身份三键：上下文归拢（src/context_grouping.py）用。
            # 迁移旧库 msg_index/chunk_index 恒 0（不可反推），归拢侧有全 0 防护。
            snap["request_ids"].append(r["request_id"] or "")
            snap["msg_indexes"].append(int(r["msg_index"] or 0))
            snap["chunk_indexes"].append(int(r["chunk_index"] or 0))
            embeddings.append(
                np.frombuffer(r["embedding"], dtype=np.float32, count=dimension))
        snap["mat"] = np.stack(embeddings)          # (n, dim)，已归一化
        snap["corpus"] = [lexical_tokens(c) for c in snap["contents"]]
        # 每个 session 的有序行：由这份快照派生（不回查 DB）。
        # fetched 按 `ORDER BY rowid`（插入序）排好，与 vanilla memory_store.py:459
        # 的 `ORDER BY rowid` 一致；邻接扩展即在 rowid 相邻处取同 session 邻居
        # （对齐 vanilla _expand_result_positions 的 `position ± distance` 语义）。
        for idx, sid in enumerate(snap["sessions"]):
            if not sid:
                continue
            snap["session_rows"].setdefault(sid, []).append({
                "id": snap["ids"][idx], "content": snap["contents"][idx],
                "ts": snap["timestamps"][idx], "ingest_at": snap["ingest_ats"][idx],
                "role": snap["roles"][idx], "meta": snap["metas"][idx],
                "rowid": snap["rowids"][idx],
            })
        # ---- 上下文归拢的派生结构（只读，随快照缓存）----
        # row_index：id -> 快照行下标（search 里把候选映射回消息三键）。
        snap["row_index"] = {mid: i for i, mid in enumerate(snap["ids"])}
        # msg_order：(request_id, msg_index) -> 该消息在 session 内的全局序号。
        # 消息全局序 = (段号 chunk-k, 段内 msg_index)。**必须用全量消息**算序号
        # —— 只用候选集会把"候选里恰好相邻"误判成"会话里相邻"。段号解析失败
        # 的消息不入表（grouped_order 里取不到序号 → 自成单链，不吸拢）。
        msg_order: dict = {}
        by_session: dict[str, set] = {}
        for i, rid in enumerate(snap["request_ids"]):
            if parse_segment(rid) is None:
                continue
            by_session.setdefault(snap["sessions"][i], set()).add(
                (rid, snap["msg_indexes"][i]))
        for sid, mks in by_session.items():
            ordered = sorted(mks, key=lambda t: (parse_segment(t[0]), t[1]))
            for pos, mk in enumerate(ordered):
                msg_order[mk] = pos
        snap["msg_order"] = msg_order
        return snap

    def _event_days_from_snapshot(self, snap: dict) -> list:
        """快照里每行的「事件日期」集合（灌库期写入 metadata.event_dates）。

        首次调用时从 metas 解析并缓存进 snap（快照按 user 缓存，成本摊销）；
        temporal_filter=off 时永远不会被调到。旧库没有该字段 -> 每行空集合，
        检索行为自动退化为只比发送时间。
        """
        cached = snap.get("event_dates")
        if cached is None:
            cached = []
            for meta_json in snap["metas"]:
                try:
                    ed = json.loads(meta_json).get("event_dates") or []
                except Exception:
                    ed = []
                cached.append(frozenset(ed))
            snap["event_dates"] = cached
        return cached

    def _load_sentences(self, user_id: str, snap: dict) -> dict:
        """快照 lazy 加载该 user 的句级行（仅 sent_rescore_enabled 时调用）。

        返回 {"parent_idx": int64 数组（句 → 快照行下标）, "mat": (n_s, dim) 矩阵}。
        无句级行（旧库没这张表 / 灌库时没开 sent_index_enabled）时 parent_idx
        为空数组，调用方据此 no-op；表本身不存在（sqlite3.OperationalError）
        与「表在但 0 行」同口径处理，只警告一次（实例级标志，避免每个 query
        刷屏）。句级行只随新 Add 增加，而新 Add 必然改变 memories 行数 →
        快照 version 变 → 缓存整体重建，无需单独的失效键。
        """
        cached = snap.get("sentences")
        if cached is not None:
            return cached
        dim = int(snap.get("dimension") or 0)
        parent_idx: list[int] = []
        embeddings: list[np.ndarray] = []
        rows = []
        try:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT mem_id, embedding, dimension FROM memory_sentences "
                    "WHERE user_id=? ORDER BY rowid", (user_id,))
                rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []
        if rows:
            row_index = snap.get("row_index") or {}
            for r in rows:
                i = row_index.get(r["mem_id"])
                if i is None:
                    continue
                d = int(r["dimension"] or 0)
                if d != dim or d <= 0:
                    continue   # 维度不符（理论不可能，防御性跳过）
                parent_idx.append(i)
                embeddings.append(
                    np.frombuffer(r["embedding"], dtype=np.float32, count=dim))
        out = {"parent_idx": np.asarray(parent_idx, dtype=np.int64),
               "mat": (np.stack(embeddings) if embeddings
                       else np.zeros((0, max(dim, 1)), dtype=np.float32))}
        if parent_idx:
            out["mat"] = out["mat"] / np.maximum(
                np.linalg.norm(out["mat"], axis=1, keepdims=True), 1e-9)
        elif not getattr(self, "_sent_noop_warned", False):
            print("[warn] sentence rescore: no sentence rows for this user "
                  "(old DB / sent_index_enabled was off at ingest) -> no-op",
                  flush=True)
            self._sent_noop_warned = True
        snap["sentences"] = out
        return out

    def search(self, query: str, user_id: str | None = None,
               top_k: int = 100, options: list[str] | None = None,
               return_debug: bool = False
               ) -> "list[tuple[float, MemoryEntry]] | tuple":
        """检索 + 可选返回全量计算中间结果（供本地评测 / 排查用）。

        `return_debug=False`（默认）行为完全不变：返回 `list[(score, MemoryEntry)]`，
        与 `app.py` / 评测脚本的现有调用保持兼容。
        `return_debug=True`：额外返回一个 `debug` 字典，含
          粗排(vec/bm25/fused 全量打分)、选项多视图(views)、精排(rerank recall /
          精排分)、同会话邻接扩展、最终返回序，便于「查找召回 / 排序原因」。
        该参数**只影响返回形态，不改变任何打分逻辑**——契约与线上结果完全一致。
        """
        if user_id is None:
            if return_debug:
                return [], {"error": "no user_id"}
            return []
        mode = (self.cfg.retrieval_mode or "hybrid").lower()
        strategy = (self.cfg.fuse_strategy or "weighted").lower()
        use_bm25 = mode in ("hybrid", "enhanced")

        # ---- 一次性快照：只有"读行"这一下持 DB 锁 ----
        # 对齐 vanilla-rag-memory：读完之后融合 / 重排 / 邻接全在内存里做，不再回查 DB。
        #   ① 不会出现"每个 session 各自持锁查询"的多快照不一致（#C1）；
        #   ② 模型推理与打分不再占着 DB 锁，并发检索不再被串行化（#D6）；
        #   ③ 快照按 user_id 缓存（_load_snapshot）：没有新写入时不再重复反序列
        #      化全部向量、不再重新分词建 BM25。
        snap = self._load_snapshot(user_id)
        ids = snap["ids"]
        if not ids:
            if return_debug:
                return [], {"candidate_pool_size": 0}
            return []
        contents = snap["contents"]
        timestamps = snap["timestamps"]
        ingest_ats = snap["ingest_ats"]
        sessions = snap["sessions"]
        roles = snap["roles"]
        metas = snap["metas"]
        mat = snap["mat"]
        dimension = snap["dimension"]
        session_rows = snap["session_rows"]
        n = len(ids)

        # ---- 查询侧时间约束（temporal_filter，默认 off；解析器见 dateparse.py）----
        # 解析源是**原始 query**，不是 retrieval_query：后者在 hybrid 下已拼进
        # options 文本（下方视图构造处），拿它解析会把候选答案里的日期也算进来
        # ——与下方"扩展源是原始 query"（expander 段）同理。
        # off 时连正则都不跑，行为与无此功能逐位一致（回归测试把关）。
        temporal_mode = (self.cfg.temporal_filter or "off").strip().lower()
        temporal_dbg: dict = {"enabled": temporal_mode != "off",
                              "mode": temporal_mode}
        t_start_ms = t_end_ms = None   # 时间窗（ms）；None = 本题没解析出时间约束
        ts_arr = None                  # lazy 构建：只有真解析出时间窗才转 numpy
        if temporal_mode != "off":
            # 相对时间基准：该 user 快照内最大的非 0 source_timestamp。
            # /search 契约（schema.SearchRequest）没有时间字段，基准只能在
            # store 内部自取 —— 与评测侧 conversation_anchor_iso()（取最新
            # session 日期）口径一致。
            reference_ms = max((t for t in timestamps if t), default=None)
            span = parse_time_constraint(query, reference_ms)
            if span is not None:
                start_iso, end_iso = span.iso()
                temporal_dbg.update(kind=span.kind, raw=span.raw,
                                    start=start_iso, end=end_iso)
                t_start_ms, t_end_ms = span.start_ms, span.end_ms
                ts_arr = np.asarray(timestamps, dtype=np.int64)
                # 事件日期（灌库期从原文相对时间提取）：软/硬两种模式都会
                # 把「事件日期落在窗内」的行视为窗内 —— 修复 last night
                # 这类 事件日 ≠ 发送日 的偏移。
                event_days = (self._event_days_from_snapshot(snap)
                              if self.cfg.temporal_event_dates else None)
            else:
                temporal_dbg["note"] = "no date expression in query"

        def _event_hit() -> np.ndarray:
            return np.fromiter(
                (any(t_start_ms <= d <= t_end_ms for d in days)
                 for days in event_days),
                dtype=bool, count=n)

        emb = self._get_embedder()
        option_view_dbg = None
        # 检索用 query：hybrid 下把 options 直接拼进查询文本（对齐 vanilla :472-473）；
        # enhanced 下才额外做"每个候选答案一个视图"的多路编码。
        retrieval_query = query
        if options and mode == "enhanced":
            views = [query] + [f"{query}\nCandidate answer: {opt}" for opt in options]
        elif options:
            retrieval_query = query + "\nOptions:\n" + "\n".join(options)
            views = [retrieval_query]
        else:
            views = [query]

        # ---- 查询扩展（R3 / R4 的查询侧）----
        # 扩展源是**原始 query**，不是 `retrieval_query`：hybrid 下后者已经拼进了
        # options 文本，把候选答案也拿去做扩展会让 R3/R4 的 A/B 说不清是谁的功劳。
        # 关闭时（默认）这里一次都不调 `expand()`，零开销。
        expander = self._get_expander()
        expansions = expander.expand(query) if expander.available else []
        use_exp = bool(expansions)
        expand_dbg = {
            "enabled": bool(expander.available),
            "relations": list(getattr(expander, "indices", {}) or {}),
            "n_terms": len(expansions),
            "terms": [e.as_dict() for e in expansions],
        }
        if use_exp:
            expand_dbg["bm25_query"] = bm25_query_text(retrieval_query, expansions)
            expand_dbg["dense_query"] = dense_query_text(
                views[0], expansions, self.cfg.expand_dense_query_mode)

        with self._embedding_lock:
            q_vecs = emb.encode_queries(views)
            # R4 的查询向量：与 R1 用**同一套视图结构**，只是文本多了扩展词 ——
            # 两路的差异因此只有"查询文本"这一项，融合时才能干净地比较权重。
            q_vecs_exp = (emb.encode_queries(
                [dense_query_text(v, expansions, self.cfg.expand_dense_query_mode)
                 for v in views]) if use_exp else None)
        if q_vecs.ndim != 2 or q_vecs.shape[1] != dimension:
            raise ValueError(
                f"Query embedding dimension {q_vecs.shape[1]} does not match stored "
                f"memories ({dimension})")

        def _aggregate(qv: np.ndarray):
            """dense 打分（R1 与 R4 共用同一套，保证两路差异只有查询文本）。

            enhanced 模式下叠加 options 非负增量（#3）：base + w * max(option_max - base, 0)。
            权重直接取自配置（默认 0.3，见 ServerConfig.option_view_weight），不再在这里
            兜一个硬编码 0.3 —— 显式设 0 就该真的不加增量，"设了 0 却被悄悄改成 0.3"
            属于配置静默失效。
            """
            if options and mode == "enhanced" and len(qv) > 1:
                base = mat @ qv[0]
                opt = (mat @ qv[1:].T).max(axis=1)
                w = float(self.cfg.option_view_weight)
                return base + w * np.maximum(opt - base, 0.0), base, opt
            return mat @ qv[0], None, None

        vec_scores, base_scores, opt_scores = _aggregate(q_vecs)

        # ---- PRF / Rocchio 查询扩展（prf_enabled，默认关）----
        # 用首轮 dense top-k 的向量质心扩展 query 向量：
        #   q' = normalize(q + β·mean(topk_vecs))
        # 再用 q' **替换**原向量重算 dense 分（替换语义，不加融合路，不扰动
        # RRF 权重标定；BM25 路与 R3/R4 扩展路不受影响）。纯 numpy，治
        # 「问题措辞 ≠ 判定依据措辞」的 BM25=0 / dense 偏低类 query。
        # 首轮排序用 base view（enhanced 下 base_scores 不含 option 增量）——
        # 质心应来自「query 本身」的近邻，而不是候选答案视图放大后的排序。
        # 失败回退：任何异常都保住首轮分数继续检索（对齐 _rerank 的 fallback）。
        prf_dbg: dict = {"enabled": bool(self.cfg.prf_enabled)}
        if self.cfg.prf_enabled:
            try:
                topk_n = max(1, int(self.cfg.prf_topk))
                beta = float(self.cfg.prf_beta)
                first_scores = base_scores if base_scores is not None else vec_scores
                topk_idx = np.argsort(-first_scores, kind="stable")[:topk_n]
                centroid = mat[topk_idx].mean(axis=0)
                cnorm = float(np.linalg.norm(centroid))
                if cnorm > 0.0 and beta > 0.0:
                    q_before = q_vecs.copy()
                    q_vecs = q_vecs + beta * (centroid / cnorm)[None, :]
                    q_vecs = q_vecs / np.maximum(
                        np.linalg.norm(q_vecs, axis=1, keepdims=True), 1e-9)
                    vec_scores, base_scores, opt_scores = _aggregate(q_vecs)
                    prf_dbg.update({
                        "applied": True, "beta": beta, "topk": topk_n,
                        "topk_ids": [ids[i] for i in topk_idx],
                        "topk_scores": [round(float(first_scores[i]), 6)
                                        for i in topk_idx],
                        "q_norm_before": round(float(np.linalg.norm(q_before)), 6),
                        "centroid_norm": round(cnorm, 6),
                    })
                else:
                    prf_dbg["applied"] = False
                    prf_dbg["noop"] = ("zero_beta" if beta <= 0.0
                                       else "zero_centroid")
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] PRF expansion failed, first-round dense scores "
                      f"kept: {ex!r}", flush=True)
                prf_dbg["error"] = repr(ex)

        if base_scores is not None:
            option_view_dbg = {
                "enabled": True, "weight": float(self.cfg.option_view_weight),
                "views": views,
                "base_scores": [round(float(x), 6) for x in base_scores],
                "option_max_scores": [round(float(x), 6) for x in opt_scores],
            }
        vec_scores_exp = (_aggregate(q_vecs_exp)[0] if use_exp
                          else np.zeros(n, dtype=np.float64))

        # BM25 词法打分：语料/query 都走 lexical_tokens（排除纯标点）。
        # 语料与 BM25 对象随快照缓存：它们只依赖文档、不依赖 query，而重建它们是
        # search() 里除模型推理外最贵的一步。BM25Okapi.get_scores() 只读自身统计
        # 量，多线程并发调用安全。
        # 全部文档都没有词元时 BM25Okapi 的 avgdl=0 会触发除零，这里显式兜底（#D16）。
        bm25_scores = np.zeros(n, dtype=np.float64)
        bm25_scores_exp = np.zeros(n, dtype=np.float64)
        if use_bm25:
            bm25 = snap.get("bm25")
            if bm25 is None and any(snap["corpus"]):
                try:
                    bm25 = BM25Okapi(snap["corpus"])
                    snap["bm25"] = bm25
                except ZeroDivisionError:
                    bm25 = None
            if bm25 is not None:
                bm25_scores = np.asarray(
                    bm25.get_scores(lexical_tokens(retrieval_query)),
                    dtype=np.float64)
                if use_exp:
                    # R3 = R2 的输入 + 扩展词，**只有这一项差异**。
                    # 必须过 `lexical_tokens()`：索引语料是先词干化再建的，不跟着
                    # 词干化的话 `research` 与索引里的 `research`（由 researching
                    # 归一而来）对不上，扩展等于白做。
                    bm25_scores_exp = np.asarray(
                        bm25.get_scores(lexical_tokens(
                            bm25_query_text(retrieval_query, expansions))),
                        dtype=np.float64)

        # ---- 融合（2 路 / 4 路）----
        # 对齐 vanilla memory_store.py:498-510：
        #   * dense 名次取**全量行**（不是 top-N），每条文档都有 dense RRF 名次；
        #   * lexical 只让 `bm25 > 0` 的文档进 RRF（零分不该拿到名次分）；
        #   * 最终排序用 np.lexsort((-vec_scores, -fused))：并列时按 dense 分定序，
        #     结果完全可复现（此前 np.argsort 默认快排不稳定，同分顺序随 numpy 版本变）。
        #
        # 开启扩展（`use_exp=True`）时是 4 路，新增两路**沿用各自同族路的规则**：
        #   R3（扩展 + BM25）与 R2 同规则：只让 `bm25_exp > 0` 的文档进 RRF、按 bm25_cap 截；
        #   R4（扩展 + dense）与 R1 同规则：dense 语义路给**全量**名次。
        # 关闭扩展时每个 `if use_exp` 分支都不进，融合结果与改版前**逐位一致**
        # （由 `local_test/tests/test_retrieval_regression.py` 把关）。
        k = int(self.cfg.rrf_k)
        lw = float(self.cfg.lexical_weight)
        exbw = float(self.cfg.expand_bm25_weight)
        exdw = float(self.cfg.expand_dense_weight)
        recall_n = int(self.cfg.retrieval_recall_n) or 300
        rerank_pool = int(self.cfg.rerank_recall_n) or 300
        # `retrieval_recall_n` 的语义是「给下游 rerank 备候选」，因此它对 BM25 路的
        # 截断**只在真会走重排时才有意义**；rerank 关闭时让 BM25 全量参与。
        # 对齐 vanilla 0.6.0：它的候选池截断同样只存在于
        # `if self.reranker is not None:` 分支内，默认无 reranker 时该分支不执行。
        # 门禁条件与下方 rerank 调用处严格一致（rerank_enabled 且给了模型）。
        use_rerank = bool(self.cfg.rerank_enabled and self.cfg.reranker_model)
        bm25_cap = recall_n if use_rerank else n
        dense_positions = np.argsort(-vec_scores, kind="stable")
        fused = np.zeros(n, dtype=np.float64)

        def _rrf_dense(acc, scores, weight):
            """dense 语义路（R1/R4）进 RRF：全量行都给名次，不截断。"""
            for rank, doc_idx in enumerate(np.argsort(-scores, kind="stable")):
                acc[int(doc_idx)] += weight / (k + rank + 1)

        def _rrf_lexical(acc, scores, weight):
            """词法路（R2/R3）进 RRF：只给 `score > 0` 的文档名次，并截到 bm25_cap。"""
            pos = np.flatnonzero(scores > 0)
            pos = pos[np.argsort(-scores[pos], kind="stable")][:bm25_cap]
            for rank, doc_idx in enumerate(pos):
                acc[int(doc_idx)] += weight / (k + rank + 1)

        if strategy == "rrf":
            _rrf_dense(fused, vec_scores, 1.0)
            if use_bm25:
                _rrf_lexical(fused, bm25_scores, lw)
            if use_exp:
                _rrf_dense(fused, vec_scores_exp, exdw)
                if use_bm25:
                    _rrf_lexical(fused, bm25_scores_exp, exbw)
        else:
            # 加权归一化融合：仅在并集内参与，bm25 归一化以并集内最大值为基准。
            vec_part = np.clip(vec_scores, 0.0, 1.0)
            vec_exp_part = (np.clip(vec_scores_exp, 0.0, 1.0) if use_exp
                            else np.zeros(n, dtype=np.float64))
            lexical_positions = np.argsort(-bm25_scores, kind="stable")[:bm25_cap]
            # dense 只有向量一路，无需按 recall_n 截断（与 rrf 分支"Dense 名次取
            # 全量"保持一致），否则 debug 的 all_scores 里 recall_n 名之后全是 0，
            # 排查"这条为什么没分"时会误读；hybrid 才需要并集截断。
            if use_bm25:
                union = set(int(i) for i in dense_positions[:recall_n])
                union |= set(int(i) for i in lexical_positions)
            else:
                union = set(int(i) for i in dense_positions)
            if use_exp:
                # R4 是 dense 族，按与 R1 相同的口径截到 recall_n 再并入并集；
                # 若并集里不含它，R4 在加权融合下就完全投不出票。
                union |= set(int(i) for i in
                             np.argsort(-vec_scores_exp, kind="stable")[:recall_n])
                if use_bm25:
                    union |= set(int(i) for i in
                                 np.argsort(-bm25_scores_exp,
                                            kind="stable")[:bm25_cap])
            union = sorted(union)
            if use_bm25:
                union_bm25_max = float(max(bm25_scores[i] for i in union)) if union else 0.0
                union_bm25_exp_max = (float(max(bm25_scores_exp[i] for i in union))
                                      if union else 0.0)
                for i in union:
                    bm25_norm = bm25_scores[i] / union_bm25_max if union_bm25_max > 0 else 0.0
                    fused[i] = (self.cfg.bm25_weight * bm25_norm
                                + self.cfg.vector_weight * vec_part[i])
                    if use_exp:
                        # 逐项各自按"并集内最大值"归一化（与 R2 同一口径），
                        # 再乘各自的权重 —— 不做这一层归一化的话，扩展路的原始
                        # BM25 量纲与 R2 不同，权重就没法横向比。
                        exp_bm25_norm = (bm25_scores_exp[i] / union_bm25_exp_max
                                         if union_bm25_exp_max > 0 else 0.0)
                        fused[i] += (exbw * exp_bm25_norm
                                     + exdw * vec_exp_part[i])
            else:
                # dense 下乘 vector_weight 不改变排序（单调缩放），但让
                # MEMORY_VECTOR_WEIGHT 真的生效，并与 hybrid 保持同一量纲 ——
                # 此前该配置在 dense 下被完全忽略（"配了但不生效"）。
                # 权重 <=0 时退化为 1.0：否则全部 fused 被打成 0，会被下面的
                # `fused[i] <= 0` 全过滤，直接返回空结果。
                vw = float(self.cfg.vector_weight)
                if not np.isfinite(vw) or vw <= 0.0:
                    vw = 1.0
                for i in union:
                    fused[i] = vw * vec_part[i]
                    if use_exp:
                        fused[i] += exdw * vec_exp_part[i]

        # ---- temporal_filter: soft —— 窗内融合分乘 (1+bonus)，只调排序 ----
        # 乘性而非加性：rrf 与 weighted 的分数量级不同，加常数在两种模式下
        # 相对强度不可比；fused<=0 的行乘完仍 <=0，不会凭空产生候选。
        if t_end_ms is not None and temporal_mode == "soft":
            in_win = (ts_arr >= t_start_ms) & (ts_arr <= t_end_ms)
            event_matched = 0
            if event_days is not None:
                ehit = _event_hit()
                event_matched = int(ehit.sum())
                in_win = in_win | ehit
            fused[in_win] *= (1.0 + float(self.cfg.temporal_soft_bonus))
            temporal_dbg["matched"] = int(in_win.sum())
            temporal_dbg["pool"] = int(n)
            temporal_dbg["bonus"] = float(self.cfg.temporal_soft_bonus)
            temporal_dbg["event_matched"] = event_matched
            # 逐行留痕（report.md 阶段轨迹账本用）：哪些行拿到了加成。
            # 值域 = 窗内行子集，通常远小于全库。
            temporal_dbg["bonus_ids"] = [ids[i] for i in np.nonzero(in_win)[0]]

        # ---- forget tombstone（遗忘降权，默认 forget_penalty=1.0 = 关）----
        # 时间限定版：forget 只删除"在它之前"说的内容 —— 快照 rowid 顺序即
        # 对话顺序，晚于 forget 声明的新证据不受影响。逐 forget 句独立匹配
        # 主题词（不并池），避免多条 forget 句的词池 union 造成误杀。
        # 降权掩码与 query 无关（只取决于快照内容），按 user 缓存进 snap；
        # forget 语句自身的 chunk 也一并降权（指令噪声，压掉腾位置）。
        forget_dbg: dict = {"enabled": False}
        fp = float(self.cfg.forget_penalty)
        # ---- forget 语料级闸门（forget_genuine_gate，默认关 = 逐位不变）----
        # tombstone 的设计前提是"用户真的下了删除指令"。快照里一条真删除请求
        # 都没有（< forget_genuine_min）→ 本快照整体跳过 tombstone，掩码都不
        # 构建。激活时（真删除 >= min）掩码构建逻辑与旧行为逐位一致。
        # 实证（2026-10-19）：PMv2 128k 真删除 114+ 条；locomo / PMv1 1M 为 0，
        # 在 locomo 上 tombstone 把 72% 候选池罩进降权掩码（judge 0.750→0.300）。
        gate_dbg: dict | None = None
        if fp < 1.0 and self.cfg.forget_genuine_gate:
            gn = snap.get("forget_genuine_n")
            ratio_min = float(self.cfg.forget_genuine_min_ratio)
            fn = snap.get("forget_n")
            if gn is None or (ratio_min > 0.0 and fn is None):
                gn = int(sum(1 for c in contents
                             if _FORGET_REQUEST_RE.search(c)))
                snap["forget_genuine_n"] = gn
                if ratio_min > 0.0:
                    fn = int(sum(1 for c in contents
                                 if _FORGET_RE.search(c)))
                    snap["forget_n"] = fn
            gate_dbg = {"genuine_n": gn,
                        "min": int(self.cfg.forget_genuine_min)}
            ok = gn >= int(self.cfg.forget_genuine_min)
            if ok and ratio_min > 0.0:
                # 密度判定：真删除句占全部 forget 句的比例。治"1 条真删除
                # 请求 + 几十句口语句"的用户——有真请求（闸门该开）但口语句
                # 建掩码照样误杀（LME 实测 B=0.8667 vs C=0.9667 的残差来源）。
                dens = (gn / fn) if fn else 0.0
                gate_dbg["forget_n"] = fn
                gate_dbg["density"] = round(dens, 4)
                gate_dbg["min_ratio"] = ratio_min
                ok = dens >= ratio_min
            if not ok:
                gate_dbg["tombstone_skipped"] = True
                forget_dbg = {"enabled": False, "genuine_gate": gate_dbg}
                fp = 1.0
        if fp < 1.0:
            fmode = (getattr(self.cfg, "forget_match_mode", "token") or "token").lower()
            # 掩码来源（forget_mask_source）：all = 全部含 forget 的句（旧行为，
            # 口语句也建掩码）；genuine = 仅真删除请求句（口语句退出黑名单，
            # 治"1 条真请求 + 几十句口语句"的毯式误杀）。PMv2 冠军掩码构成
            # 会变（17 句口语句退出），启用后须重跑 PMv2 70 题重验。
            _src_re = (_FORGET_REQUEST_RE
                       if (getattr(self.cfg, "forget_mask_source", "all")
                           or "all") == "genuine" else _FORGET_RE)
            fmask = snap.get("forget_mask")
            if fmask is None:
                fmask = np.zeros(n, dtype=bool)
                if fmode == "vector":
                    # 向量语义匹配：forget 句按 passage 侧编码（与 snap["mat"] 的
                    # chunk 向量同侧、均已归一化），余弦 >= forget_sim_threshold 才
                    # 算相关。多条 forget 句仍是逐句独立匹配后并集，但阈值语义化后
                    # 每句只抓自己的主题，并集不再罩住全库（离线探针实测 0.70 档
                    # 降权 6%~21%，token 路径同场景为 81%~98%）。掩码与 query 无关，
                    # 编码一次后随 fmask 一起按 user 缓存进 snap。
                    tau = float(self.cfg.forget_sim_threshold)
                    ftexts = [c for c in contents if _src_re.search(c)]
                    if ftexts:
                        fvecs = self._get_embedder().encode_documents(ftexts)
                        k = 0
                        for j, _c in enumerate(contents):
                            if not _src_re.search(contents[j]):
                                continue
                            sim = mat[:j] @ fvecs[k]    # 只打早于 forget 的行
                            fmask[:j] |= (sim >= tau)
                            fmask[j] = True             # forget 句自身也压
                            k += 1
                else:
                    min_shared = int(self.cfg.forget_min_shared)
                    if min_shared > 0:
                        for j, c in enumerate(contents):
                            if not _src_re.search(c):
                                continue
                            toks_j = set(snap["corpus"][j]) - _FORGET_STOP
                            if not toks_j:
                                continue
                            for i in range(j):          # 只打早于 forget 的行
                                if len(set(snap["corpus"][i]) & toks_j) >= min_shared:
                                    fmask[i] = True
                            fmask[j] = True
                snap["forget_mask"] = fmask
            demoted = int(fmask.sum())
            # ---- 覆盖率熔断（第三道闸，forget_max_coverage，默认 1.0 = 关）----
            # 掩码罩住比例超阈值 → 判定误伤，整体放弃降权（行为等同无
            # tombstone）。与前两道语料侧闸门互补：直接看「这次要打多少条」。
            # locomo 泛化防线：那次掩码罩了 72% 候选池（judge 0.750→0.300）。
            coverage = (demoted / n) if n else 0.0
            cov_max = float(getattr(self.cfg, "forget_max_coverage", 1.0))
            coverage_blown = bool(demoted and coverage > cov_max)
            if coverage_blown:
                demoted = 0
            if demoted:
                fused[fmask] *= fp
            forget_dbg = {"enabled": True, "penalty": fp, "demoted": demoted,
                          "pool": n, "time_scoped": True, "mode": fmode}
            if coverage_blown:
                forget_dbg["coverage_blown"] = round(coverage, 4)
                forget_dbg["max_coverage"] = cov_max
            # 逐行留痕（report.md 阶段轨迹账本用）：掩码罩住了哪些行。
            # coverage_blown=true 时这些行最终没被降权（熔断放弃）。
            forget_dbg["demoted_ids"] = [ids[i] for i in np.nonzero(fmask)[0]]
            if gate_dbg is not None:
                forget_dbg["genuine_gate"] = gate_dbg
            if fmode == "vector":
                forget_dbg["threshold"] = float(self.cfg.forget_sim_threshold)

        # ---- 句级 max-pool 加成（sent_rescore_enabled，默认关）----
        # 对每个候选 chunk，用其句级向量与 query 的相似度聚合出 pool 分
        # （max / mean / blend，见 sent_pool），对融合分做乘性加成
        #   fused *= (1 + sent_weight · pool)
        # 乘性而非加性：RRF 分数量级 ~0.0x，加性 bonus 会直接淹没融合分
        # （对齐 temporal_soft_bonus 的量纲无关先例）。query 向量用 q_vecs[0]
        # （base view，PRF 开启时即扩展后的向量）—— 句级加成与 PRF 天然串联。
        # 无句级行自动 no-op；任何异常保住 fused 继续检索。
        sent_dbg: dict = {"enabled": bool(self.cfg.sent_rescore_enabled),
                          "applied": False}
        if self.cfg.sent_rescore_enabled:
            try:
                sent = self._load_sentences(user_id, snap)
                parent_idx = sent["parent_idx"]
                if parent_idx.size:
                    sent_scores = np.clip(sent["mat"] @ q_vecs[0], 0.0, 1.0)
                    max_pool = np.zeros(n, dtype=np.float64)
                    sum_pool = np.zeros(n, dtype=np.float64)
                    cnt = np.zeros(n, dtype=np.int64)
                    np.maximum.at(max_pool, parent_idx, sent_scores)
                    np.add.at(sum_pool, parent_idx, sent_scores)
                    np.add.at(cnt, parent_idx, 1)
                    mean_pool = sum_pool / np.maximum(cnt, 1)
                    mode = (self.cfg.sent_pool or "max").strip().lower()
                    alpha = float(self.cfg.sent_alpha)
                    if mode == "mean":
                        pool = mean_pool
                    elif mode == "blend":
                        pool = alpha * max_pool + (1.0 - alpha) * mean_pool
                    else:
                        pool = max_pool
                    w = float(self.cfg.sent_weight)
                    if w > 0.0:
                        fused *= (1.0 + w * pool)
                    sent_dbg.update({
                        "applied": True, "pool": mode, "alpha": alpha,
                        "weight": w,
                        "n_sentences": int(parent_idx.size),
                        "chunks_with_sentences": int((cnt > 0).sum()),
                        "pool_max": round(float(pool.max()), 6),
                        # 逐行留痕（report.md 阶段轨迹账本用）：每行的句级 pool 值。
                        # 报告侧用它反推「句级加成前」的 fused 做逐阶段名次重建。
                        "pool_by_id": {ids[i]: round(float(pool[i]), 6)
                                       for i in np.nonzero(cnt > 0)[0]},
                    })
                else:
                    sent_dbg["noop"] = "no_sentence_rows"
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] sentence rescore failed, fused scores kept: {ex!r}",
                      flush=True)
                sent_dbg["error"] = repr(ex)

        order = np.lexsort((-vec_scores, -fused))

        # ---- temporal_filter: hard —— 窗外候选剔除（物化前，最省的一刀）----
        # ts=0（无可信时间，见 _UNKNOWN_TS）的行按 temporal_filter_keep_unknown
        # 决定去留；过滤后为空则整路回退未过滤结果 —— 绝不给空答案。
        if t_end_ms is not None and temporal_mode == "hard":
            in_win = (ts_arr >= t_start_ms) & (ts_arr <= t_end_ms)
            event_matched = 0
            if event_days is not None:
                ehit = _event_hit()
                event_matched = int(ehit.sum())
                in_win = in_win | ehit
            if self.cfg.temporal_filter_keep_unknown:
                in_win = in_win | (ts_arr == 0)
            temporal_dbg["matched"] = int(in_win.sum())
            temporal_dbg["pool"] = int(n)
            temporal_dbg["event_matched"] = event_matched
            # 逐行留痕：hard 模式被剔除的行（窗外且不保留 ts=0）。
            if in_win.any():
                removed = ~in_win[order]
                temporal_dbg["removed_ids"] = [ids[i] for i in order[removed]]
                order = order[in_win[order]]
            else:
                temporal_dbg["fallback"] = "empty_after_filter"

        cap = int(top_k) if top_k and top_k > 0 else 0

        # ---- MMR 去冗余 + session 软配额（mmr_enabled，默认关）----
        # 融合后对候选池做贪心 MMR：
        #   score(d) = λ·sim(q,d) − (1−λ)·max sim(d, 已选)
        # 治「同类闲聊互相投票霸占 top_k、gold 被挤出池」。相关性项用
        # vec_scores（dense 余弦），冗余项用快照 mat 行向量，纯 numpy。
        # 只重排 order：MMR 选中的 cap 条排前、其余候选按原融合序跟后
        # （rerank 仍能看到完整候选池），fused 分数与返回条数语义不变。
        # 软配额两阶段（session_quota_enabled）：先守「每 session 最多
        # session_quota_n 条」贪心选满 top_k；候选 session 数不足选不满时
        # 放开配额补选（防 top-k 填不满漏 gold）——两阶段共用同一套打分，
        # 冗余惩罚与已选屏蔽全程生效。放在 adjacency 之前：邻接扩展塞进的
        # 同 session 邻居不受配额约束。
        mmr_dbg: dict = {"enabled": bool(self.cfg.mmr_enabled)}
        mmr_active = bool(self.cfg.mmr_enabled) and len(order) > 1
        # ---- MMR 条件路由（mmr_route_enabled，默认关；2026-10-24 conv-48 复盘）----
        # 选人式 MMR 的冗余惩罚对「中等相似（0.5~0.75，跨 session 话题沾边）」
        # 也下重手，会把融合名次靠前的 gold 挤出 top_k（conv-48 实测 11 题
        # full->zero、-9.1pt；调 lambda 净效果为零，见 result/fullon_conv48_mock_*）。
        # 本闸把「要不要做多样性选人」交给本次查询自己的池子结构：池前 100 内
        # sim >= mmr_route_sim_threshold 的近重复对 >= mmr_route_min_dup_pairs
        # 才执行 MMR，否则整块跳过（逐位退化为融合序）。闲聊密集语料的近重复
        # 团会被检出（开 MMR），话题沾边但消息各异的语料检不出（不开）——
        # 路由不需要知道数据集是谁。
        if mmr_active and self.cfg.mmr_route_enabled:
            pool_n_route = (int(self.cfg.mmr_candidate_n)
                            if int(self.cfg.mmr_candidate_n) > 0 else rerank_pool)
            pool_route = order[:max(1, min(pool_n_route, len(order)))]
            m_route = min(100, len(pool_route))
            pm_route = mat[pool_route[:m_route]]
            pair_sims = np.clip(pm_route @ pm_route.T, 0.0, 1.0)
            np.fill_diagonal(pair_sims, 0.0)
            dup_pairs = int((pair_sims >=
                             float(self.cfg.mmr_route_sim_threshold)).sum()) // 2
            mmr_dbg["route"] = {"dup_pairs": dup_pairs,
                                "threshold": float(self.cfg.mmr_route_sim_threshold),
                                "min_dup_pairs": int(self.cfg.mmr_route_min_dup_pairs),
                                "probe_n": m_route}
            if dup_pairs < int(self.cfg.mmr_route_min_dup_pairs):
                mmr_dbg["applied"] = False
                mmr_dbg["skipped"] = "low_redundancy"
                mmr_active = False
        if mmr_active:
            try:
                pool_n = (int(self.cfg.mmr_candidate_n)
                          if int(self.cfg.mmr_candidate_n) > 0 else rerank_pool)
                pool = order[:max(1, min(pool_n, len(order)))]
                k_mmr = cap if cap > 0 else len(pool)
                lam = float(self.cfg.mmr_lambda)
                quota_n = max(1, int(self.cfg.session_quota_n))
                pool_mat = mat[pool]                          # (m, dim)，已归一化
                sims_q = np.clip(vec_scores[pool], 0.0, 1.0)  # λ·sim(q,d) 项
                m = len(pool)
                available = np.ones(m, dtype=bool)
                count: dict = {}                # session_id -> 已选条数
                selected: list[int] = []        # pool 内下标（按选中序）
                sel_rows: list[np.ndarray] = []
                phase_counts = {"quota": 0, "free": 0}

                def _pick(enforce_quota: bool) -> bool:
                    if selected:
                        red = (pool_mat @ np.stack(sel_rows).T).max(axis=1)
                        scores = lam * sims_q - (1.0 - lam) * red
                    else:
                        scores = sims_q.copy()
                    scores[~available] = -np.inf
                    if enforce_quota:
                        for i in range(m):
                            if available[i] and count.get(
                                    sessions[pool[i]], 0) >= quota_n:
                                scores[i] = -np.inf
                    best = int(scores.argmax())
                    if not np.isfinite(scores[best]):
                        return False    # 本阶段已无可选（配额或池耗尽）
                    selected.append(best)
                    sel_rows.append(pool_mat[best])
                    available[best] = False
                    count[sessions[pool[best]]] = count.get(
                        sessions[pool[best]], 0) + 1
                    return True

                use_quota = bool(self.cfg.session_quota_enabled)
                if use_quota:
                    while len(selected) < k_mmr and _pick(True):
                        phase_counts["quota"] += 1
                while len(selected) < k_mmr and _pick(False):
                    phase_counts["free"] += 1
                order = np.concatenate([
                    pool[np.asarray(selected, dtype=np.int64)],
                    pool[available],    # 未选中的按原融合序跟在后面
                ]).astype(order.dtype, copy=False)
                top_sessions: dict = {}
                for i in selected:
                    sid = sessions[pool[i]]
                    top_sessions[sid] = top_sessions.get(sid, 0) + 1
                mmr_dbg.update({
                    "applied": True, "lambda": lam,
                    "pool_n": m, "k": int(k_mmr),
                    "quota_enabled": use_quota,
                    "quota_n": quota_n if use_quota else None,
                    "selected_in_quota_phase": phase_counts["quota"],
                    "selected_in_free_phase": phase_counts["free"],
                    "sessions_in_topk": top_sessions,
                })
            except Exception as ex:  # noqa: BLE001
                print(f"[warn] MMR rerank failed, fused order kept: {ex!r}",
                      flush=True)
                mmr_dbg["error"] = repr(ex)

        # 融合后先截到 `rerank_pool`（送 rerank 的候选池），而非 top_k——
        # 这样 rerank 才有机会从 >100 名里救回相关片段（#修复此前 cap=top_k 导致 rerank 形同虚设）。
        # 截断**只在真会走 rerank 时执行**：rerank 关闭时这个池子没有下游消费者
        # （MMR 用自己的 mmr_candidate_n/rerank_pool 池、adjacency 只看 seed、
        # 最终名额由 top_k 决定），留着只会把融合名次 200 名外的候选白白挡在
        # top_k 之外（2026-10-23 conv-48 复盘：B 归因 44% 全是这条无消费者截断）。
        if use_rerank:
            order = order[:rerank_pool]
        # 逐行留痕（report.md 阶段轨迹账本用）：MMR/配额重排后的顺序。
        # 未启用 MMR 时不导出——报告侧回退用 fused 全序（两者相同）。
        if return_debug and mmr_dbg.get("applied"):
            mmr_dbg["order_after_ids"] = [ids[i] for i in order]
        # ---- 无时间条目的 order 渲染前置量（纯内存判定，不回查 DB）----
        # 快照含该用户全部行：has0/has1 直接在 timestamps 上算。
        #   纯时间轴（LoCoMo/LME）  -> mixed=False，且 ts=0 分支走不到 -> 无 order，输出逐字节不变；
        #   纯无时间轴（PersonaMem）-> mixed=False，仅 ts=0 条目渲染 order；
        #   混合时间轴              -> mixed=True，全部条目渲染 order（两把尺度可比）。
        order_on = bool(getattr(self.cfg, "untimestamped_order_meta", True))
        rowids = snap["rowids"]
        mixed_timeline = False
        if order_on:
            _ts = snap["timestamps"]
            mixed_timeline = any(t == 0 for t in _ts) and any(t > 0 for t in _ts)
        out = []
        for i in order:
            if fused[i] <= 0:
                continue
            meta = json.loads(metas[i])
            # 不回落到 ingest_at：见 _UNKNOWN_TS 处注释。0 -> "timestamp: unknown time"
            # + created_at=null，让"没有时间"这个事实可见、可复现。
            ts_ms = timestamps[i]
            out.append((float(fused[i]), MemoryEntry(
                id=ids[i], user_id=user_id, session_id=sessions[i],
                role=roles[i] or "user", type="conversation",
                content=self._result_content(roles[i], meta.get("speaker"),
                                             contents[i], ts_ms,
                                             seq=rowids[i] if order_on else None,
                                             mixed_timeline=mixed_timeline),
                created_at=float(ts_ms),
                metadata=meta)))
        out = out[:rerank_pool] if use_rerank else out

        # 粗排(融合)全量打分：用于本地排查「gold 证据排到哪了 / 为什么没进 top-k」。
        if return_debug:
            dbg = {
                "query": query,
                "user_id": user_id,
                "options": options,
                "config": {
                    "retrieval_mode": mode,
                    "fuse_strategy": strategy,
                    "rrf_k": k,
                    "lexical_weight": lw,
                    "bm25_weight": self.cfg.bm25_weight,
                    "vector_weight": self.cfg.vector_weight,
                    "option_view_weight": self.cfg.option_view_weight,
                    "rerank_enabled": self.cfg.rerank_enabled,
                    "reranker_model": self.cfg.reranker_model,
                    "retrieval_recall_n": self.cfg.retrieval_recall_n,
                    "rerank_recall_n": self.cfg.rerank_recall_n,
                    "rerank_top_k": self.cfg.rerank_top_k,
                    "adjacency_enabled": self.cfg.adjacency_enabled,
                    "adjacency_seed_k": self.cfg.adjacency_seed_k,
                    "adjacency_window": self.cfg.adjacency_window,
                    "chunk_tokens": self.cfg.chunk_tokens,
                    "chunk_overlap": self.cfg.chunk_overlap,
                    "embedding_model": self.cfg.embedding_model,
                    "expand_enabled": self.cfg.expand_enabled,
                    "expand_relations": list(self.cfg.expand_relations or []),
                    "expand_bm25_weight": exbw,
                    "expand_dense_weight": exdw,
                    "expand_dense_query_mode": self.cfg.expand_dense_query_mode,
                    "temporal_filter": self.cfg.temporal_filter,
                    "temporal_event_dates": self.cfg.temporal_event_dates,
                    "search_merge_same_message": self.cfg.search_merge_same_message,
                    "search_group_adjacent_messages": self.cfg.search_group_adjacent_messages,
                    "prf_enabled": self.cfg.prf_enabled,
                    "prf_beta": self.cfg.prf_beta,
                    "prf_topk": self.cfg.prf_topk,
                    "mmr_enabled": self.cfg.mmr_enabled,
                    "mmr_lambda": self.cfg.mmr_lambda,
                    "mmr_candidate_n": self.cfg.mmr_candidate_n,
                    "session_quota_enabled": self.cfg.session_quota_enabled,
                    "session_quota_n": self.cfg.session_quota_n,
                    "sent_index_enabled": self.cfg.sent_index_enabled,
                    "sent_rescore_enabled": self.cfg.sent_rescore_enabled,
                    "sent_pool": self.cfg.sent_pool,
                    "sent_alpha": self.cfg.sent_alpha,
                    "sent_weight": self.cfg.sent_weight,
                    "sent_min_tokens": self.cfg.sent_min_tokens,
                },
                "candidate_pool_size": n,
                # 全量候选的粗排四路打分（不截断），id 与 all_scores 一一对应。
                # 扩展关闭时 bm25_exp / vec_exp 恒为 0，仍保留字段 —— 下游
                # `local_eval/report.py` 因此不必判有没有这个键。
                "all_scores": [
                    {"id": ids[i],
                     "vec": round(float(vec_scores[i]), 6),
                     "bm25": round(float(bm25_scores[i]), 6),
                     "vec_exp": round(float(vec_scores_exp[i]), 6),
                     "bm25_exp": round(float(bm25_scores_exp[i]), 6),
                     "fused": round(float(fused[i]), 6)}
                    for i in range(n)
                ],
                "option_view": option_view_dbg,
                # 扩展开关 / 实际生效的关系 / 每个 query 词扩出了什么（含 d/scale）。
                # 没有这一段，新加的两路在 fullresult 里就不可复盘 —— 违反
                # `result/rule.md` §2.2「中间每一步可回溯」。
                "expander": expand_dbg,
                # 查询侧时间约束：解析出的时间窗 / 命中数 / 作用模式。
                # 没有这一段，temporal_filter 的效果在 fullresult 里无法归因
                # （`result/rule.md` §2.2「中间每一步可回溯」）。
                "temporal": temporal_dbg,
                # forget tombstone：降权惩罚 / 命中条数（关着 enabled=False）。
                "forget": forget_dbg,
                # PRF 查询扩展：β / topk / 质心来源 id（关着 enabled=False）。
                "prf": prf_dbg,
                # 句级 max-pool 加成：聚合方式 / 权重 / 命中句数（关着 enabled=False）。
                "sent_rescore": sent_dbg,
                # MMR 重排：λ / 配额 / 两阶段各选几条 / top-k 的 session 分布。
                "mmr": mmr_dbg,
                # 粗排(融合)排序后的 top（送 rerank 的候选池，rerank 之前的顺序）。
                # 调试视图固定按 rerank_pool 封顶：rerank 关闭时 order 不截断
                # （见上方门禁），不封顶会把全库灌进 fullresult。
                "coarse_topk": [
                    {"id": ids[i], "fused": round(float(fused[i]), 6)}
                    for i in order[:rerank_pool]
                ],
            }
        else:
            dbg = None

        # ---- 增强阶段：全程不持 DB 锁（快照已在内存里）----
        # CrossEncoder 重排（#4，默认关闭）
        if self.cfg.rerank_enabled and self.cfg.reranker_model:
            if return_debug:
                out, rer_dbg = self._rerank(query, out, cap, return_debug=True)
                dbg["rerank"] = rer_dbg
            else:
                out = self._rerank(query, out, cap)
        else:
            if return_debug:
                dbg["rerank"] = {"enabled": False}
        # 同会话邻接扩展（#5，默认关闭）
        if self.cfg.adjacency_enabled:
            if return_debug:
                out, adj_dbg = self._expand_adjacency(user_id, out, cap, session_rows,
                                                      return_debug=True)
                dbg["adjacency"] = adj_dbg
            else:
                out = self._expand_adjacency(user_id, out, cap, session_rows)
        else:
            if return_debug:
                dbg["adjacency"] = {"enabled": False}

        # 最终兜底：对外一律只返回 top_k（AML 规则 §5.4：结果 ≤ top_k）。
        # 邻接扩展已在内部按 cap 截断，这里再兜一次以防 rerank 候选池 > top_k。
        out = out[:cap]

        # ---- 上下文归拢（L1/L2，默认关闭；见 src/context_grouping.py）----
        # 只改返回顺序（候选集合与条数不变），放在 top_k 截断之后、最终返回之前，
        # 让 LLM 看到的上下文按「消息 → 会话段」连贯组织。
        l1_group = bool(self.cfg.search_merge_same_message)
        l2_group = bool(self.cfg.search_group_adjacent_messages)
        grouping_dbg: dict = {"merge_same_message": l1_group,
                              "group_adjacent_messages": l2_group}
        if l1_group or l2_group:
            out, grouping_dbg = self._group_context(out, snap, grouping_dbg)

        if return_debug:
            dbg["context_grouping"] = grouping_dbg
            dbg["final"] = [{"id": e.id, "score": round(s, 4)} for s, e in out]
            dbg["final_with_content"] = [
                {"id": e.id, "score": round(s, 4),
                 "content": e.content, "created_at": e.created_at_iso()}
                for s, e in out
            ]
            return out, dbg
        return out

    def _group_context(self, out: list, snap: dict, info: dict) -> "tuple[list, dict]":
        """对 top_k 截断后的候选做两级归拢（只重排，返回新顺序的同一个列表）。

        任何无法安全归拢的情形（迁移旧库 msg_index 恒 0 / id 不在快照里 /
        候选不足）都原样返回并写明 noop 原因 —— 绝不因归拢功能引入报错或丢结果。
        """
        l1 = bool(self.cfg.search_merge_same_message)
        l2 = bool(self.cfg.search_group_adjacent_messages)
        msg_indexes = snap.get("msg_indexes")
        if not msg_indexes or not any(msg_indexes):
            # 真·迁移旧库：msg_index 恒 0，同一段 /add 里的多条消息无法区分，
            # 归并会把它们误当成一条消息 —— 强制 no-op，让"开了没效果"可见。
            print("[warn] context grouping disabled: all msg_index == 0 "
                  "(migrated legacy DB cannot identify messages)", flush=True)
            info["noop"] = "all_msg_index_zero"
            return out, info
        row_index = snap.get("row_index") or {}
        keys: list[tuple[str, int, int, str]] = []
        for _score, e in out:
            i = row_index.get(e.id)
            if i is None:
                info["noop"] = "id_not_in_snapshot"
                return out, info
            keys.append((snap["request_ids"][i], msg_indexes[i],
                         snap["chunk_indexes"][i], snap["sessions"][i]))
        perm, stats = grouped_order(keys, snap.get("msg_order"),
                                    merge_same_message=l1,
                                    group_adjacent_messages=l2)
        if perm is None:
            info["noop"] = stats.get("noop", "identity")
            return out, info
        info.update(stats)
        # 逐行留痕（report.md 阶段轨迹账本用）：归拢前的顺序（≤top_k）。
        info["order_before_ids"] = [e.id for _, e in out]
        return [out[i] for i in perm], info

    def _rerank(self, query: str, out: list, cap: int,
                 return_debug: bool = False):
        """Recall top rerank_recall_n -> CrossEncoder 精排 -> 取 top_k。"""
        if not out:
            if return_debug:
                return out, {"enabled": True, "applied": False,
                             "reason": "empty input"}
            return out
        # 候选池已是融合后 top `rerank_recall_n`（调用方已截断到 rerank_pool），这里取全部送 CrossEncoder。
        candidates = out
        recall_n = len(candidates)
        contents = [e.content for _, e in candidates]
        debug = {
            "enabled": True,
            "recall_n": recall_n,
            "recall_ids": [e.id for _, e in candidates],
            "rerank_scores": [],
            "selected_ids": [],
            "applied": False,
            "status": "ok",
        }
        try:
            from rerank import get_reranker
            rk = get_reranker(self.cfg.reranker_model,
                              batch_size=self.cfg.reranker_batch_size,
                              max_length=self.cfg.reranker_max_length)
            scores = np.asarray(rk.score(query, contents), dtype=np.float64).reshape(-1)
            if scores.shape[0] != len(candidates) or not np.all(np.isfinite(scores)):
                raise ValueError("Reranker returned invalid scores")
            debug["rerank_scores"] = [round(float(x), 6) for x in scores]
            debug["applied"] = True
            # 保留条数用 max(cap, rerank_top_k)：rerank_top_k 配置成 0 时不能把结果清空（#D8）。
            keep = min(len(candidates),
                       max(int(cap) if cap and cap > 0 else 0,
                           int(self.cfg.rerank_top_k)))
            # 双键排序：并列时用原融合分定序，保证可复现（对齐 vanilla :526-528）。
            order = np.lexsort((-np.asarray([s for s, _ in candidates]), -scores))
            res = [(float(scores[i]), candidates[i][1]) for i in order[:keep]]
            debug["score_by_id"] = {candidates[i][1].id: round(float(scores[i]), 6)
                                    for i in range(len(candidates))}
            debug["selected_ids"] = [candidates[i][1].id for i in order[:keep]]
            return (res, debug) if return_debug else res
        except Exception as ex:  # noqa: BLE001
            # 模型加载/推理失败 -> 回退到融合排序，不阻断检索
            debug["status"] = f"fallback: {ex!r}"
            return (out, debug) if return_debug else out

    def _session_rows(self, user_id: str, session_id: str) -> list:
        """取单个 session 的 chunk 行，按会话内真实邻接顺序排列。

        会话内全序 = `rowid`（插入序），与 vanilla-rag-memory `ORDER BY rowid` 一致：
          * 邻接扩展据此在 rowid 相邻处取同 session 邻居（对齐 vanilla
            `_expand_result_positions` 的 `position ± distance` 语义）
          * 跨 request 的乱序灌库不再用时间戳纠正（与 vanilla 一致：vanilla
            同样按插入序，不按时间戳重排）

        **不要再往这里加 `request_id`**：`request_id` 形如 `...:chunk-{k}`，字符串
        比较下 `chunk-10 < chunk-2`，一旦某 session 超过 10 批就会把顺序打乱（#B2）。
        `msg_index` / `chunk_index` 同理不参与排序——它们是 per-request 局部下标。

        返回 [{"id", "content", "ts", "ingest_at", "role", "meta"}, ...]
        与 `search()` 内存快照派生的结构一致，可直接喂给 `adjacency.expand()`。
        仅供测试 / 排查；`search()` 本体不调用。
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, content, source_timestamp, ingest_at, role, "
                "store_metadata "
                "FROM memories WHERE user_id=? AND session_id=? "
                "ORDER BY source_timestamp, rowid",
                (user_id, session_id))
            return [{"id": r["id"], "content": r["content"] or "",
                     "ts": int(r["source_timestamp"] or 0),
                     "ingest_at": int(r["ingest_at"] or 0),
                     "role": r["role"] or "user",
                     "meta": r["store_metadata"] or "{}"}
                    for r in cur.fetchall()]

    def _result_content(self, role: str, speaker: str | None, content: str,
                        ts_ms: int, seq: int | None = None,
                        mixed_timeline: bool = False) -> str:
        """输出期渲染 content：索引里只存纯正文，前缀在这里才拼（#D10 / #D14）。

        格式（2026-10-01 拍板，替换旧 `[{label} | {ts}] {content}`）：
        无方括号、分号分隔、保留时间戳、不解析相对时间 ——
          * 有 speaker：`speaker: {speaker}; timestamp: {ISO8601}; content: {content}`
          * 无 speaker：`role: {role}; timestamp: {ISO8601}; content: {content}`
          * ts_ms=0（无可信时间）：timestamp 渲染为 "unknown time"
            （见 _UNKNOWN_TS 注释，绝不回落到 ingest_at）。

        order 渲染（untimestamped_order_meta 开关，默认开）：
          * ts_ms=0 且 seq 非空 -> 追加 `order: #{seq}`（rowid = Add 到达顺序，
            确定性可复现），让答案模型在无时间轴上判断先后；
          * ts_ms>0 且 mixed_timeline=True（该用户时间轴混有 ts=0 与 ts>0）
            -> timestamp 后同样追加 order，保证跨组可比；
          * 纯时间轴用户两条分支都走不到 -> 输出与改动前逐字节一致。

        seq / mixed_timeline 带默认值：旧调用点不传即完全不渲染（向后兼容）。
        """
        tag = "speaker" if speaker else ("role" if role else "user")
        value = speaker or role or "user"
        order_on = bool(getattr(self.cfg, "untimestamped_order_meta", True))
        if not ts_ms:
            if order_on and seq is not None:
                return (f"{tag}: {value}; timestamp: unknown time; "
                        f"order: #{int(seq)}; content: {content}")
            return f"{tag}: {value}; timestamp: unknown time; content: {content}"
        if order_on and mixed_timeline and seq is not None:
            return (f"{tag}: {value}; timestamp: {_iso_ms(ts_ms)}; "
                    f"order: #{int(seq)}; content: {content}")
        return f"{tag}: {value}; timestamp: {_iso_ms(ts_ms)}; content: {content}"

    def _expand_adjacency(self, user_id: str, out: list, cap: int,
                          session_rows: dict | None = None,
                          return_debug: bool = False):
        """同 session 邻接补全（#5，默认关闭）。

        `session_rows` 由 `search()` 的内存快照直接传入，**本方法不查 DB**
        （此前每个 seed 的 session 各查一次，既多快照不一致又多持锁 N 次，#C1）。
        """
        if not out:
            if return_debug:
                return out, {"enabled": True, "applied": False,
                             "reason": "empty input"}
            return out
        from adjacency import expand as _adj_expand
        seed_k = max(1, int(self.cfg.adjacency_seed_k))
        window = max(1, int(self.cfg.adjacency_window))
        seed_ids = {e.id for _, e in out[:seed_k]}
        # 邻居行的 order 渲染要与主路径逐字节一致：mixed_timeline 从 session_rows
        # 判定（快照全量行都按 session 分组在此），rowid 逐行传入 render。
        order_meta = bool(getattr(self.cfg, "untimestamped_order_meta", True))
        _rows = [r for rows in (session_rows or {}).values() for r in rows]
        mixed_timeline = bool(order_meta and _rows
                              and any((r.get("ts") or 0) == 0 for r in _rows)
                              and any((r.get("ts") or 0) > 0 for r in _rows))

        def _render(role, speaker, content, ts_ms, seq=None):
            return self._result_content(role, speaker, content, ts_ms,
                                        seq=seq, mixed_timeline=mixed_timeline)

        expanded = _adj_expand(out, seed_k, window, session_rows or {},
                               cap=cap, user_id=user_id,
                               render=_render, order_meta=order_meta,
                               mixed_timeline=mixed_timeline)
        original_ids = {e.id for _, e in out}
        expanded_ids = [e.id for _, e in expanded if e.id not in original_ids]
        debug = {
            "enabled": True,
            "seed_k": seed_k,
            "window": window,
            "seed_ids": list(seed_ids),
            "expanded_ids": expanded_ids,
            # 逐行留痕（report.md 阶段轨迹账本用）：邻接补全后的完整顺序
            # （≤cap，与 top_k 截断后一致）。
            "order_after_ids": [e.id for _, e in expanded],
            "applied": True,
        }
        return (expanded, debug) if return_debug else expanded
