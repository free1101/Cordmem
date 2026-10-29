"""Central configuration for the AML memory adapter.

All tunable parameters live here — no hard-coded magic numbers in `store`/`app`.

配置来源与优先级（高 -> 低）
----------------------------
1. 显式 override：`load_config(overrides={...})` / `set_override()` / CLI `--set k=v`
2. YAML 配置文件：由 `--config <path>` 或环境变量 `MEMORY_CONFIG=<path>` 指定
3. 环境变量：`MEMORY_*` / `EMBEDDING_*`（**只有 YAML 没声明该字段时才生效**）
4. `ServerConfig` 的 dataclass 默认值

**为什么 YAML 高于环境变量**：做对比实验时「一次实验 = 一个 YAML 文件」，
`configs/baseline.yaml` 复制一份只改一个变量就能 A/B。若环境变量优先级更高，
本机残留的 `MEMORY_*` 会静默盖掉 YAML 里改的那一项 —— 这正是本项目反复踩的
"配了但不生效"。所以 YAML 里声明过的字段，同名环境变量仍在时会被忽略，并打一条
WARNING 点名是哪个环境变量被忽略了（不留哑谜）。

**不传 `--config` / `MEMORY_CONFIG` 时，行为与改版前完全一致**（纯环境变量），
线上部署（docker compose 注入 env）不受影响。

校验（对齐 vanilla-rag-memory memory_store.py:216-231 的构造期校验）：
枚举走白名单、数值走区间、布尔走显式解析，非法值直接抛 `ConfigError`；
YAML / `--set` 里出现**未知字段同样抛 `ConfigError`**（字段名拼错不能静默忽略）。

**为什么必须校验**：此前 `MEMORY_RETRIEVAL_MODE` 被误赋给 `fuse_strategy`（真正的
`retrieval_mode` 才是这一层），于是 `MEMORY_RETRIEVAL_MODE=dense` 静默变成
`fuse_strategy="dense"` —— 既不是 "rrf" 也不是 "weighted"，走默认分支且 BM25
根本没关。这类"配了但不生效"没有任何报错，只能靠白名单拦住。
"""
from __future__ import annotations

import logging
import os
from dataclasses import MISSING, asdict, dataclass, field, fields

logger = logging.getLogger("aml_memory.config")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

#: 用环境变量指定 YAML 配置文件路径，等价于 CLI 的 `--config`
#: （服务在 import 期就读配置，子进程 / uvicorn 场景只能靠环境变量透传）
CONFIG_ENV_VAR = "MEMORY_CONFIG"

#: YAML 顶层允许出现的非 ServerConfig 字段（其余一律 ConfigError）
RUN_LEVEL_KEYS = ("exp_name", "output_root")

#: YAML 里 server: 段的键名
SERVER_SECTION = "server"

#: `expand_relations` 的合法取值（HyperE WordNet 关系的白名单）。
#: **这里是唯一真相来源**：`expander.ALL_RELATIONS` 直接引用它，避免两处清单漂移
#: —— 白名单漂移的后果是"配置里写对了却校验报错"，或"写错了却不报错"。
EXPAND_RELATIONS = ("hypernym_noun", "hypernym_verb", "domain_topic",
                    "member_holonym")

_MISSING = object()  # "该来源没有提供这个值" 的哨兵（区别于显式写了 null）


class ConfigError(ValueError):
    """配置取值非法。启动期抛出，避免带着错误配置静默运行。"""


@dataclass
class ServerConfig:
    # ----- persistence -----
    db_path: str = "local_data/memories.db"

    # ----- embedding -----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # 后端二选一：
    #   local = 进程内 sentence-transformers（默认，离线可用）；
    #   api   = OpenAI 兼容的 `POST {embedding_api_base}/embeddings`
    #           （硅基流动 / 阿里百炼 / OpenAI 都能用）。
    # 换后端 / 换模型 / 换端点 / 换维度 = 换了一套编码语义，`store._check_identity()`
    # 会拒绝打开旧库（不是静默错配），必须重灌。
    embedding_backend: str = "local"      # local | api
    # 例：硅基流动 = https://api.siliconflow.cn/v1
    #     阿里百炼 = https://dashscope.aliyuncs.com/compatible-mode/v1
    #     （新加坡 = https://dashscope-intl.aliyuncs.com/compatible-mode/v1）
    embedding_api_base: str = ""
    # 输出维度：0 = 不传 dimensions，用端点默认值（首次编码时探测实际长度）。
    # 例 text-embedding-v4 默认 1024，支持 2048/1536/1024/768/512/256/128/64。
    embedding_api_dimensions: int = 0
    # 一次 HTTP 请求送多少条文本。会被服务商硬上限夹住并打 WARNING：
    # 阿里百炼 text-embedding-v4 单请求最多 10 条（超限直接 400，不是降级）。
    embedding_api_batch_size: int = 32
    # 查询前缀三态语义（修复 #D13：此前设空串反而打开自动前缀、且无法关掉）：
    #   None（默认，或不设 EMBEDDING_QUERY_PREFIX）= 交给 Embedder 按模型名推断
    #        （BGE 英文模型自动加官方前缀，test1 等历史运行即此行为，保持不变）；
    #   ""（显式空串 / EMBEDDING_QUERY_PREFIX=""）= 不使用任何查询前缀（真正关掉）；
    #   非空串 = 使用自定义前缀。
    # 注意：想保留“自动前缀”请直接不设置该项，不要设成空串。
    embedding_query_prefix: str | None = None
    embedding_max_seq_length: int = 512   # 显式固定，避免超长 chunk 被静默截断

    # ----- chunking -----
    chunk_tokens: int = 320
    chunk_overlap: int = 40
    drop_short_tail: bool = False         # 丢弃尾块碎片（会改变块数 -> 改变 mem_id）

    # ----- fusion -----
    bm25_weight: float = 0.4              # 仅 fuse_strategy="weighted" 生效
    vector_weight: float = 0.6            # 仅 fuse_strategy="weighted" 生效
    fuse_strategy: str = "rrf"            # "weighted" | "rrf"
    rrf_k: int = 60
    lexical_weight: float = 0.5           # RRF 模式下词法路（BM25）的融合权重
    # BM25 路截断上限：语义是「给下游 rerank 备候选」，故**仅在真会走重排时生效**
    # （门禁 = rerank_enabled 且给了 reranker_model，见 store.search 的 use_rerank）。
    # rerank 关闭时 BM25 全量参与融合（对齐 vanilla 0.6.0：其候选池截断同样只在
    # `if self.reranker is not None` 分支内）。dense 路任何情况下都不截断。
    retrieval_recall_n: int = 300

    # ----- retrieval mode (dense / hybrid / enhanced) -----
    # 与 fuse_strategy（融合算法）是正交的两层：
    #   dense   = 纯向量，关 BM25
    #   hybrid  = 向量 + BM25（默认）
    #   enhanced= hybrid + options 非负增量（强制）
    retrieval_mode: str = "hybrid"
    # options 非负增量权重（仅 retrieval_mode="enhanced" 生效）：
    # base + w * max(option_max - base, 0)。默认 0.3 与 vanilla 一致；显式设 0
    # 即真的关闭增量，不再被 store 静默兜回 0.3（此前 0 会被改成 0.3）。
    option_view_weight: float = 0.3

    # ----- same-session adjacency (default off) -----
    adjacency_enabled: bool = False
    adjacency_window: int = 1             # 每个 seed 前后各补几条邻居
    adjacency_seed_k: int = 10            # 取前 N 条 top 结果作为 seed 做邻接扩展

    # ----- rerank (default off) -----
    rerank_enabled: bool = False
    reranker_model: str | None = None
    rerank_recall_n: int = 200            # 融合后送 rerank 的候选池大小；对齐 vanilla rerank_candidates 默认 200（rerank 关闭时无截断）
    rerank_top_k: int = 100               # 精排后最多保留条数
    reranker_batch_size: int = 32
    reranker_max_length: int = 512

    # ----- content augmentation (default off) -----
    context_messages: int = 0             # 索引时拼入前后文条数（>0 改变向量语义）
    temporal_enrichment: bool = False     # 输出期把日期 / 相对时间解析拼进 content

    # ----- search 上下文归拢（输出层重排，默认 off；见 src/context_grouping.py）-----
    # 只改返回顺序，不改候选集合与条数；键 = (request_id, msg_index)。
    # 旧迁移库 msg_index 恒 0 时自动 no-op（无法区分同段内的不同消息）。
    search_merge_same_message: bool = False       # L1：同消息切片聚成一组（组内 chunk_index 序）
    search_group_adjacent_messages: bool = False  # L2：同 session 相邻消息吸拢（链内会话全局序）

    # ----- temporal filter（查询侧时间约束，检索期；默认 off）-----
    # off  = 完全不解析（连正则都不跑），检索结果与无此功能逐位一致；
    # soft = 解析出时间窗后，窗内候选融合分乘 (1 + temporal_soft_bonus)，只调排序；
    # hard = 窗外候选直接过滤（source_timestamp=0 的行按
    #        temporal_filter_keep_unknown 决定去留；过滤后为空则回退未过滤结果）。
    # 解析器在 src/dateparse.py（纯标准库）；解析源是原始 query，不是
    # retrieval_query（后者在 hybrid 下已拼进 options 文本）。
    temporal_filter: str = "off"          # off | soft | hard
    temporal_soft_bonus: float = 0.15     # soft 模式窗内乘性加成幅度
    temporal_filter_keep_unknown: bool = True   # hard 模式是否保留无时间戳行
    # 灌库期把消息原文里的相对时间（last night / yesterday / 3 days ago ...）
    # 解析成「事件日期」存进 store_metadata（见 dateparse.parse_event_dates）；
    # 检索期 soft/hard 判断窗内时额外比对事件日期 —— 修复
    # 事件日 ≠ 发送日的偏移（如 last night 的晚饭发生在发送日前一天）。
    # 灌库解析微秒级/条；改此开关不影响已灌库（event_dates 不参与
    # payload_hash / mem_id），旧库没有该字段时自动退化为只比发送时间。
    temporal_event_dates: bool = True

    # ----- 无时间戳消息的顺序可见性（默认 on）-----
    # 该用户时间轴里存在 source_timestamp=0 的消息时（纯无时间轴，如 PersonaMem；
    # 或混合时间轴 = 同时存在 ts=0 与 ts>0），在检索结果的 content 前缀里渲染
    # `order: #<rowid>`（= Add 到达顺序，确定性可复现），让答案模型判断先后。
    #   * 纯无时间轴：仅 ts=0 条目渲染（`timestamp: unknown time; order: #N`）；
    #   * 混合时间轴：**全部**条目渲染（有时间条目在 timestamp 后追加 order），
    #     保证两把尺度可比；
    #   * 纯时间轴（LoCoMo/LME）：完全不渲染，输出逐字节不变。
    # 只影响输出渲染，不改 source_timestamp / created_at 语义，不合成时间。
    untimestamped_order_meta: bool = True

    # ----- query expansion（HyperE WordNet 双曲嵌入；默认关）-----
    # 开启后检索段从 2 路变 4 路：R1 dense / R2 bm25 不变，新增
    #   R3 = 扩展 + BM25（`bm25_scores_exp`）
    #   R4 = 扩展 + dense（`vec_scores_exp`）
    # 扩展只**多两路投票**，不替换原路；关掉时行为与改版前逐位一致。
    expand_enabled: bool = False
    # 只开 hypernym_noun：离线探针（`local_test/tools/expand_probe.py`）实测在
    # 89 条漏召回 gold 上，"扩展词落进 gold chunk" 的 9 条里 8 条来自名词 hypernym，
    # verb 1 条、domain_topic 1 条、member_holonym 0 条 —— 后三者只贡献噪声。
    expand_relations: list[str] = field(
        default_factory=lambda: ["hypernym_noun"])
    # "relation=path" 覆盖项（留空则用 expand_dir + 文件名模板拼）
    expand_files: list[str] = field(default_factory=list)
    expand_dir: str = "data/hyperE/hyperE.wn.10d"
    expand_per_token_n: int = 3           # 每个 query 词、每个关系取几个近邻
    expand_top_n: int = 20                # 每题扩展词总数上限
    # 双曲图距离上限。**单位是图距离（≈跳数）**，实测 1.0 = 一条 WordNet 边，
    # 所以 1.5 的含义是"只取直接相连的邻居，留 0.5 数值余量"（不是相似度阈值）。
    expand_max_graph_distance: float = 1.5
    expand_min_token_len: int = 3
    expand_multiword: str = "split"       # split | drop | keep（多词 lemma 处理）
    expand_drop_stopwords: bool = True    # 功能词不当扩展源（does -> premier 那类噪声）
    expand_skip_capitalized: bool = True  # 跳过人名等大写词（john -> public/lavatory）
    expand_verify_pairs: bool = True      # 启动期校验已知层级对 d/scale == 1.0
    expand_dense_query_mode: str = "related"   # related | terms_only
    expand_bm25_weight: float = 0.5       # R3 在融合里的权重
    expand_dense_weight: float = 0.5      # R4 在融合里的权重

    # ----- forget tombstone（遗忘降权，默认 1.0 = 关）-----
    # <1.0 时：用户说过 "Please forget ..." 的主题，其相关 chunk 的融合分
    # 乘 forget_penalty —— 检索期即时构建墓碑（扫快照原文，缓存），不改
    # 生成 prompt。治 "被遗忘旧偏好与提问词面最相关而霸占 top_k"。
    forget_penalty: float = 1.0
    # chunk 与 forget 主题词至少共享多少个内容词才算"相关"（防误伤）
    forget_min_shared: int = 2
    # forget 匹配方式：token = 与 forget 句共享 forget_min_shared 个内容词（旧路径，
    # 多条 forget 句掩码并集容易覆盖全库）；vector = forget 句与 chunk 的向量余弦
    # >= forget_sim_threshold 才算相关（语义级匹配，误伤小，推荐）。
    forget_match_mode: str = "token"
    # vector 模式的余弦阈值。离线探针实测（PMV1 4 用户，bge-small passage 侧）：
    # 0.65 时降权 16%~44%，0.70 时 6%~21%；先用 0.70 上 A/B，再按结果调。
    forget_sim_threshold: float = 0.70
    # ----- forget 语料级闸门（默认关）-----
    # 开启后，检索期先扫快照统计"真删除请求"句数（祈使式 forget/erase/delete
    # + 明确宾语，如 "Please forget that I ..."；口语提及 "I'll never forget
    # the day..." 不算）。少于 forget_genuine_min 条 → 本快照整体跳过
    # tombstone。动机（2026-10-19 locomo 实证）：tombstone 的设计前提是"用户
    # 真的下了删除指令"；在只有口语提及的语料上（locomo 真删除=0），4 条口语
    # forget 句通过 token 重叠把 72% 候选池罩进降权掩码，gold 被挤出 top-k
    # （judge 0.750→0.300）。PMv2 128k 真删除 114 条 → 闸门不触发，行为不变。
    forget_genuine_gate: bool = False
    # 激活 tombstone 所需的最少真删除请求数
    forget_genuine_min: int = 1
    # 密度闸门（改法①）：>0 时额外要求 真删除句数/全部forget句数 >= 该比例
    # 才激活。治"1 条真删除请求 + 几十句口语句"的用户（LME 实测残差
    # B=0.8667 vs C=0.9667）。实测分界：PMv2 128k 164/181=0.91 → 开；
    # LME-s 14/3697=0.004、locomo 与 PMv1 0/51、0/57 → 关。
    # 0.0 = 不做密度判定（只按 forget_genuine_min 计数）。
    forget_genuine_min_ratio: float = 0.0
    # ----- 覆盖率熔断（第三道闸，默认 1.0 = 关）-----
    # fmask 构建完成后统计掩码覆盖率（fmask.sum()/候选池 n），超过该阈值判定
    # 为误伤 → 本快照整体放弃降权（fp 强制回 1.0，行为等同无 tombstone）。
    # 与前两道闸互补：数量/密度闸只看语料侧统计，不看掩码实际罩住多少内容；
    # 本闸直接看「这次到底要打多少条」。动机 = locomo 灾难的泛化防线：那次
    # 掩码罩了 72% 候选池（judge 0.750→0.300）。默认 1.0 = 不启用（逐位不变）。
    forget_max_coverage: float = 1.0
    # 掩码来源（改法②）：all = 全部含 forget 的句都参与建掩码（旧行为，
    # 默认）；genuine = 仅真删除请求句建掩码，口语句退出黑名单。genuine
    # 会改变 PMv2 冠军的掩码构成（17 句口语句退出），启用后必须重跑
    # PMv2 70 题重验。
    forget_mask_source: str = "all"

    # ----- PRF / Rocchio 查询扩展（默认关）-----
    # 用首轮 dense top-k 的向量均值扩展 query 向量：q' = normalize(q + β·mean(topk_vecs))，
    # 然后用 q' 替换原 query 向量重算 dense 分。纯 numpy、无外部依赖，治
    # 「问题措辞 ≠ 判定依据措辞」（BM25=0 / dense 偏低）类 query。
    # 命门在「伪」：首轮被同类闲聊霸榜时质心会被带偏 —— 建议与 MMR 配套开。
    prf_enabled: bool = False
    # 挪多大一步：0 = 不动（等价关闭）；常用 0.3~0.8，需按标注集调。
    prf_beta: float = 0.5
    # 首轮 dense top-k 取几条做质心；越大越贴语料区域，越小越抗噪。
    prf_topk: int = 5

    # ----- MMR 去冗余 + session 软配额（默认关）-----
    # 融合后对候选池做贪心 MMR：score(d) = λ·sim(q,d) − (1−λ)·max sim(d,已选)。
    # 治「同类闲聊互相投票霸占 top_k、gold 被挤出池」。只重排 order（先选中的
    # 排前、其余按原融合序跟后），不改 fused 分数与返回条数；放在 rerank /
    # adjacency 之前，adjacency 塞进来的同 session 邻居不受配额约束。
    mmr_enabled: bool = False
    # λ=1 退化纯相关排序；0.5~0.7 常用，需按标注集调。
    mmr_lambda: float = 0.6
    # MMR 候选池大小；0 = 直接用 rerank_recall_n（送 rerank 的候选池）当池。
    mmr_candidate_n: int = 0
    # 每 session_id 最多 N 条（软配额两阶段）：先守配额贪心选满 top_k，
    # 候选 session 数不足选不满时放开配额补选（防 top-k 填不满漏 gold）。
    # 两阶段共用同一套 MMR 打分，冗余惩罚全程生效。
    session_quota_enabled: bool = False
    session_quota_n: int = 1
    # ----- MMR 条件路由（默认关；2026-10-24 locomo conv-48 复盘新增）-----
    # 选人式 MMR 的冗余惩罚对「中等相似（0.5~0.75，跨 session 话题沾边）」也
    # 下重手，会把融合名次靠前的 gold 挤出 top_k（conv-48 实测 11 题 full→zero、
    # -9.1pt；调 λ 与保护带均无法两全）。本闸把「要不要做多样性选人」交给
    # 本次查询自己的候选池结构：池前 100 内 sim ≥ mmr_route_sim_threshold 的
    # 近重复对 ≥ mmr_route_min_dup_pairs 才执行 MMR，否则整块跳过（逐位退化为
    # 融合序）。闲聊密集语料的近重复团会被检出（开 MMR），话题沾边但消息
    # 各异的语料检不出（关 MMR）——路由不需要知道数据集是谁。
    mmr_route_enabled: bool = False
    mmr_route_sim_threshold: float = 0.9
    mmr_route_min_dup_pairs: int = 5

    # ----- 多粒度索引：句级向量 + max-pool 聚合（默认关）-----
    # 灌库侧：切 chunk 的同时把每个 chunk 切句编码，写进新表 memory_sentences
    # （外键 → memories.id）。**不改 chunk 的 mem_id、不改 chunk 向量语义**
    # （不触发 _check_identity 换库），但开它必须重灌才有句级行 —— 该开关已
    # 进 db_info.py 的 REBUILD_ON_CHANGE。
    sent_index_enabled: bool = False
    # 检索侧：对候选 chunk 取其句级分数聚合（见 sent_pool），对融合分做
    # 乘性加成 fused *= (1 + sent_weight * pool)。库中没有句级行（旧库 /
    # 未开 sent_index_enabled）时自动 no-op 并打一次警告。
    sent_rescore_enabled: bool = False
    # 聚合方式：max = 对关键句命中最敏感（会放大小噪声）；
    # mean = 平稳但冲淡关键句；blend = α·max + (1−α)·mean。
    sent_pool: str = "max"
    # blend 模式的 α（max 的权重）。
    sent_alpha: float = 0.7
    # 乘性加成权重（pool ∈ [0,1]）：0 = 无加成；量纲无关（对齐
    # temporal_soft_bonus 的乘性先例），RRF 的 0.0x 分数量级下不能加性。
    sent_weight: float = 0.3
    # 切句时短于该 token 数的句子并入相邻句（碎句不成行，浪费一条向量）。
    sent_min_tokens: int = 4


# ---------------------------------------------------------------------------
# 字段规格表：YAML / --set / 环境变量三条来源共用同一套校验规则
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Spec:
    """单个字段的校验规则。

    kind: str | int | float | bool | choice | str_list。
    `str_list` 接受 YAML 的 list，也接受 `--set` / 环境变量传来的逗号分隔串；
    给了 `choices` 时逐元素校验（用于 `expand_relations` 这类白名单）。
    """
    env: str | None = None
    kind: str = "str"
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None
    nullable: bool = False       # 允许显式 None（embedding_query_prefix / reranker_model）
    blank_to_none: bool = False  # 空串按 None 处理（复刻 MEMORY_RERANKER_MODEL 旧行为）


_SPECS: dict[str, _Spec] = {
    "db_path":                  _Spec(env="MEMORY_DB_PATH", kind="str"),
    "embedding_model":          _Spec(env="EMBEDDING_MODEL", kind="str"),
    "embedding_backend":        _Spec(env="EMBEDDING_BACKEND", kind="choice",
                                      choices=("local", "api")),
    "embedding_api_base":       _Spec(env="EMBEDDING_API_BASE", kind="str"),
    "embedding_api_dimensions": _Spec(env="EMBEDDING_DIMENSIONS", kind="int", minimum=0),
    "embedding_api_batch_size": _Spec(env="EMBEDDING_API_BATCH_SIZE", kind="int",
                                      minimum=1),
    "embedding_query_prefix":   _Spec(env="EMBEDDING_QUERY_PREFIX", kind="str", nullable=True),
    "embedding_max_seq_length": _Spec(env="EMBEDDING_MAX_SEQ_LENGTH", kind="int", minimum=1),
    "chunk_tokens":             _Spec(env="MEMORY_CHUNK_TOKENS", kind="int", minimum=1),
    "chunk_overlap":            _Spec(env="MEMORY_CHUNK_OVERLAP", kind="int", minimum=0),
    "drop_short_tail":          _Spec(env="MEMORY_DROP_SHORT_TAIL", kind="bool"),
    "bm25_weight":              _Spec(env="MEMORY_BM25_WEIGHT", kind="float", minimum=0.0),
    "vector_weight":            _Spec(env="MEMORY_VECTOR_WEIGHT", kind="float", minimum=0.0),
    "fuse_strategy":            _Spec(env="MEMORY_FUSE_STRATEGY", kind="choice",
                                      choices=("weighted", "rrf")),
    "rrf_k":                    _Spec(env="MEMORY_RRF_K", kind="int", minimum=1),
    "lexical_weight":           _Spec(env="MEMORY_LEXICAL_WEIGHT", kind="float", minimum=0.0),
    "retrieval_recall_n":       _Spec(env="MEMORY_RETRIEVAL_RECALL_N", kind="int", minimum=1),
    "retrieval_mode":           _Spec(env="MEMORY_RETRIEVAL_MODE", kind="choice",
                                      choices=("dense", "hybrid", "enhanced")),
    "option_view_weight":       _Spec(env="MEMORY_OPTION_VIEW_WEIGHT", kind="float",
                                      minimum=0.0, maximum=1.0),
    "adjacency_enabled":        _Spec(env="MEMORY_ADJACENCY_ENABLED", kind="bool"),
    "adjacency_window":         _Spec(env="MEMORY_RESULT_WINDOW", kind="int", minimum=0),
    "adjacency_seed_k":         _Spec(env="MEMORY_SEED_K", kind="int", minimum=1),
    "rerank_enabled":           _Spec(env="MEMORY_RERANK_ENABLED", kind="bool"),
    "reranker_model":           _Spec(env="MEMORY_RERANKER_MODEL", kind="str",
                                      nullable=True, blank_to_none=True),
    "rerank_recall_n":          _Spec(env="MEMORY_RERANK_RECALL_N", kind="int", minimum=1),
    "rerank_top_k":             _Spec(env="MEMORY_RERANK_TOP_K", kind="int", minimum=1),
    "reranker_batch_size":      _Spec(env="MEMORY_RERANKER_BATCH_SIZE", kind="int", minimum=1),
    "reranker_max_length":      _Spec(env="MEMORY_RERANKER_MAX_LENGTH", kind="int", minimum=1),
    "context_messages":         _Spec(env="MEMORY_CONTEXT_MESSAGES", kind="int", minimum=0),
    "temporal_enrichment":      _Spec(env="MEMORY_TEMPORAL_ENRICHMENT", kind="bool"),
    # ----- search 上下文归拢（输出层重排）-----
    "search_merge_same_message": _Spec(env="MEMORY_SEARCH_MERGE_SAME_MESSAGE", kind="bool"),
    "search_group_adjacent_messages": _Spec(env="MEMORY_SEARCH_GROUP_ADJACENT_MESSAGES",
                                            kind="bool"),
    # ----- temporal filter（查询侧时间约束）-----
    "temporal_filter":          _Spec(env="MEMORY_TEMPORAL_FILTER", kind="choice",
                                      choices=("off", "soft", "hard")),
    "temporal_soft_bonus":      _Spec(env="MEMORY_TEMPORAL_SOFT_BONUS", kind="float",
                                      minimum=0.0),
    "temporal_filter_keep_unknown": _Spec(env="MEMORY_TEMPORAL_FILTER_KEEP_UNKNOWN",
                                          kind="bool"),
    "temporal_event_dates":     _Spec(env="MEMORY_TEMPORAL_EVENT_DATES", kind="bool"),
    "untimestamped_order_meta": _Spec(env="MEMORY_UNTIMESTAMPED_ORDER_META", kind="bool"),
    # ----- query expansion -----
    "expand_enabled":           _Spec(env="MEMORY_EXPAND_ENABLED", kind="bool"),
    "expand_relations":         _Spec(env="MEMORY_EXPAND_RELATIONS", kind="str_list",
                                      choices=EXPAND_RELATIONS),
    "expand_files":             _Spec(env="MEMORY_EXPAND_FILES", kind="str_list"),
    "expand_dir":               _Spec(env="MEMORY_EXPAND_DIR", kind="str"),
    "expand_per_token_n":       _Spec(env="MEMORY_EXPAND_PER_TOKEN_N", kind="int",
                                      minimum=1),
    "expand_top_n":             _Spec(env="MEMORY_EXPAND_TOP_N", kind="int", minimum=0),
    "expand_max_graph_distance": _Spec(env="MEMORY_EXPAND_MAX_GRAPH_DISTANCE",
                                       kind="float", minimum=0.0),
    "expand_min_token_len":     _Spec(env="MEMORY_EXPAND_MIN_TOKEN_LEN", kind="int",
                                      minimum=1),
    "expand_multiword":         _Spec(env="MEMORY_EXPAND_MULTIWORD", kind="choice",
                                      choices=("split", "drop", "keep")),
    "expand_drop_stopwords":    _Spec(env="MEMORY_EXPAND_DROP_STOPWORDS", kind="bool"),
    "expand_skip_capitalized":  _Spec(env="MEMORY_EXPAND_SKIP_CAPITALIZED", kind="bool"),
    "expand_verify_pairs":      _Spec(env="MEMORY_EXPAND_VERIFY_PAIRS", kind="bool"),
    "forget_penalty":            _Spec(env="MEMORY_FORGET_PENALTY", kind="float",
                                       minimum=0.0),
    "forget_min_shared":         _Spec(env="MEMORY_FORGET_MIN_SHARED", kind="int",
                                       minimum=0),
    "forget_match_mode":         _Spec(env="MEMORY_FORGET_MATCH_MODE", kind="choice",
                                       choices=("token", "vector")),
    "forget_sim_threshold":      _Spec(env="MEMORY_FORGET_SIM_THRESHOLD", kind="float",
                                       minimum=0.0, maximum=1.0),
    "forget_genuine_gate":       _Spec(env="MEMORY_FORGET_GENUINE_GATE", kind="bool"),
    "forget_genuine_min":        _Spec(env="MEMORY_FORGET_GENUINE_MIN", kind="int",
                                       minimum=0),
    "forget_genuine_min_ratio":  _Spec(env="MEMORY_FORGET_GENUINE_MIN_RATIO", kind="float",
                                       minimum=0.0, maximum=1.0),
    "forget_max_coverage":       _Spec(env="MEMORY_FORGET_MAX_COVERAGE", kind="float",
                                       minimum=0.0, maximum=1.0),
    "forget_mask_source":        _Spec(env="MEMORY_FORGET_MASK_SOURCE", kind="choice",
                                       choices=("all", "genuine")),
    "expand_dense_query_mode":  _Spec(env="MEMORY_EXPAND_DENSE_QUERY_MODE",
                                      kind="choice", choices=("related", "terms_only")),
    "expand_bm25_weight":       _Spec(env="MEMORY_EXPAND_BM25_WEIGHT", kind="float",
                                      minimum=0.0),
    "expand_dense_weight":      _Spec(env="MEMORY_EXPAND_DENSE_WEIGHT", kind="float",
                                      minimum=0.0),
    # ----- PRF / Rocchio 查询扩展 -----
    "prf_enabled":              _Spec(env="MEMORY_PRF_ENABLED", kind="bool"),
    "prf_beta":                 _Spec(env="MEMORY_PRF_BETA", kind="float",
                                      minimum=0.0, maximum=2.0),
    "prf_topk":                 _Spec(env="MEMORY_PRF_TOPK", kind="int", minimum=1),
    # ----- MMR 去冗余 + session 软配额 -----
    "mmr_enabled":              _Spec(env="MEMORY_MMR_ENABLED", kind="bool"),
    "mmr_lambda":               _Spec(env="MEMORY_MMR_LAMBDA", kind="float",
                                      minimum=0.0, maximum=1.0),
    "mmr_candidate_n":          _Spec(env="MEMORY_MMR_CANDIDATE_N", kind="int", minimum=0),
    "session_quota_enabled":    _Spec(env="MEMORY_SESSION_QUOTA_ENABLED", kind="bool"),
    "session_quota_n":          _Spec(env="MEMORY_SESSION_QUOTA_N", kind="int", minimum=1),
    # ----- MMR 条件路由 -----
    "mmr_route_enabled":        _Spec(env="MEMORY_MMR_ROUTE_ENABLED", kind="bool"),
    "mmr_route_sim_threshold":  _Spec(env="MEMORY_MMR_ROUTE_SIM_THRESHOLD",
                                      kind="float", minimum=0.0, maximum=1.0),
    "mmr_route_min_dup_pairs":  _Spec(env="MEMORY_MMR_ROUTE_MIN_DUP_PAIRS",
                                      kind="int", minimum=0),
    # ----- 多粒度索引：句级向量 -----
    "sent_index_enabled":       _Spec(env="MEMORY_SENT_INDEX_ENABLED", kind="bool"),
    "sent_rescore_enabled":     _Spec(env="MEMORY_SENT_RESCORE_ENABLED", kind="bool"),
    "sent_pool":                _Spec(env="MEMORY_SENT_POOL", kind="choice",
                                      choices=("max", "mean", "blend")),
    "sent_alpha":               _Spec(env="MEMORY_SENT_ALPHA", kind="float",
                                      minimum=0.0, maximum=1.0),
    "sent_weight":              _Spec(env="MEMORY_SENT_WEIGHT", kind="float",
                                      minimum=0.0, maximum=1.0),
    "sent_min_tokens":          _Spec(env="MEMORY_SENT_MIN_TOKENS", kind="int", minimum=1),
}


def _coerce(label: str, value, spec: _Spec):
    """按 spec 校验并转换单个值（YAML / --set / 环境变量共用）。"""
    if value is None:
        if spec.nullable:
            return None
        raise ConfigError(f"{label}=None is not allowed")

    if spec.kind == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{label}={value!r} must be a string")
        if value == "" and spec.blank_to_none:
            return None
        return value

    if spec.kind == "str_list":
        # YAML 里天然写列表；`--set` / 环境变量只能给字符串，所以统一支持
        # 逗号分隔。空串与空列表都合法（= 该项不生效），但不做 None 兜底 ——
        # `expand_relations:` 留空属于"写错"，按本模块约定直接抛错，不静默当 []。
        if isinstance(value, (list, tuple)):
            items = [str(v).strip() for v in value]
        else:
            items = [p.strip() for p in str(value).split(",")]
        items = [i for i in items if i]
        if spec.choices:
            bad = [i for i in items if i not in spec.choices]
            if bad:
                raise ConfigError(
                    f"{label}={bad} 不在允许取值 {list(spec.choices)} 内")
        return items

    if spec.kind == "bool":
        if isinstance(value, bool):
            return value
        raw = str(value).strip().lower()
        if raw in _TRUE:
            return True
        if raw in _FALSE:
            return False
        raise ConfigError(
            f"{label}={value!r} must be one of {list(_TRUE) + list(_FALSE)}")

    if spec.kind in ("int", "float"):
        if isinstance(value, bool):
            raise ConfigError(f"{label}={value!r} must be a number, not a boolean")
        try:
            out = int(value) if spec.kind == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{label}={value!r} is not a number") from exc
        if spec.minimum is not None and out < spec.minimum:
            raise ConfigError(f"{label}={out} must be >= {spec.minimum}")
        if spec.maximum is not None and out > spec.maximum:
            raise ConfigError(f"{label}={out} must be <= {spec.maximum}")
        return out

    if spec.kind == "choice":
        raw = str(value).strip().lower()
        if raw not in spec.choices:
            raise ConfigError(f"{label}={value!r} must be one of {list(spec.choices)}")
        return raw

    raise ConfigError(f"{label}: unknown spec kind {spec.kind!r}")  # pragma: no cover


def _from_env(spec: _Spec):
    """读一个环境变量；未设置（或数值/布尔类为空串）返回 _MISSING。"""
    if not spec.env:
        return _MISSING
    raw = os.getenv(spec.env)
    if raw is None:
        return _MISSING
    if raw == "" and spec.kind != "str":
        return _MISSING
    return _coerce(spec.env, raw, spec)


# ---- 兼容旧调用点的小工具（保留原签名与语义）----
def _env_str(name: str, default: str | None = None) -> str | None:
    value = _from_env(_Spec(env=name, kind="str", nullable=True))
    return default if value is _MISSING else value


def _env_int(name: str, default: int, minimum: int | None = None,
             maximum: int | None = None) -> int:
    value = _from_env(_Spec(env=name, kind="int", minimum=minimum, maximum=maximum))
    return default if value is _MISSING else value


def _env_float(name: str, default: float, minimum: float | None = None,
               maximum: float | None = None) -> float:
    value = _from_env(_Spec(env=name, kind="float", minimum=minimum, maximum=maximum))
    return default if value is _MISSING else value


def _env_bool(name: str, default: bool) -> bool:
    """布尔环境变量：只接受白名单字面量，其它一律 ConfigError。

    此前 `MEMORY_RERANK_ENABLED=maybe` 会静默变成 False —— 与本模块"非法值直接
    抛 ConfigError"的约定不符，属于典型的"配了但不生效"。
    """
    value = _from_env(_Spec(env=name, kind="bool"))
    return default if value is _MISSING else value


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = _from_env(_Spec(env=name, kind="choice", choices=choices))
    return default if value is _MISSING else value


# ---------------------------------------------------------------------------
# 各来源的应用
# ---------------------------------------------------------------------------
def _apply_env(cfg: ServerConfig) -> ServerConfig:
    """环境变量（含两个历史遗留行为）覆盖到 cfg 上。"""
    for f in fields(ServerConfig):
        spec = _SPECS.get(f.name)
        if spec is None:
            continue
        value = _from_env(spec)
        if value is not _MISSING:
            setattr(cfg, f.name, value)

    # 注意：retrieval_mode 由 MEMORY_RETRIEVAL_MODE 控制（此前被误赋给 fuse_strategy）。
    # MEMORY_RESULT_WINDOW 是历史遗留开关：设了就顺带打开邻接（保留旧行为）
    if os.getenv("MEMORY_RESULT_WINDOW"):
        cfg.adjacency_enabled = True

    # 是否重排由 MEMORY_RERANK_ENABLED 说了算：给了模型只是"装了模型"，
    # 显式 `MEMORY_RERANK_ENABLED=false` 时仍可关掉做 A/B（此前设了模型就强制
    # 开启，关不掉）。未显式表态时才沿用"给了模型就开"的旧行为。
    rerank_explicit = os.getenv("MEMORY_RERANK_ENABLED")
    if cfg.reranker_model and rerank_explicit is None:
        cfg.rerank_enabled = True
    return cfg


def _apply_mapping(cfg: ServerConfig, mapping: dict, source: str) -> ServerConfig:
    """把 YAML 的 server: 段 / override 字典覆盖到 cfg 上（优先级高于环境变量）。"""
    for key, value in (mapping or {}).items():
        spec = _SPECS.get(key)
        if spec is None:
            raise ConfigError(
                f"{source}: 未知配置字段 `{key}`；合法字段见 src/config.py 的 "
                f"ServerConfig（或 configs/baseline.yaml 的 server: 段）")
        setattr(cfg, key, _coerce(f"{source}:{key}", value, spec))
        if spec.env and os.getenv(spec.env) is not None:
            logger.warning(
                "%s 设置了 `%s=%s`，但环境变量 %s=%r 同时存在 —— 按优先级 YAML/override "
                "更高，该环境变量被忽略。若想让环境变量生效，请从 %s 里删掉 `%s`。",
                source, key, value, spec.env, os.getenv(spec.env), source, key)
    return cfg


#: RRF / weighted 两套融合**各用各的权重**，互不重叠（见 `store.search()`）：
#:   rrf      -> R1=1.0（写死的基准）+ lexical_weight + expand_bm25_weight
#:                 + expand_dense_weight，名次衰减 k=rrf_k
#:   weighted -> vector_weight * dense + bm25_weight * bm25（+ expand_*_weight）
#: 两组字段名字都很像"向量路 / BM25 路的权重"，但在对方的模式下完全读不到 ——
#: 改了没效果也不报错，是最容易白调的一处。所以两个方向都要点名提醒。
_RRF_ONLY_FIELDS = ("lexical_weight", "rrf_k")
_WEIGHTED_ONLY_FIELDS = ("vector_weight", "bm25_weight")


def _ignored_fusion_fields(cfg: ServerConfig, strategy: str) -> list[str]:
    """在给定融合策略下**不参与**的权重字段（值仍是默认值的不算，避免噪音）。"""
    defaults = {f.name: f.default for f in fields(ServerConfig)
                if f.default is not MISSING}
    names = (_WEIGHTED_ONLY_FIELDS if strategy == "rrf" else _RRF_ONLY_FIELDS)
    out = []
    for name in names:
        default = defaults.get(name)
        if default is None:
            continue
        if float(getattr(cfg, name)) != float(default):
            out.append(f"{name}={getattr(cfg, name)}")
    return out


def _validate(cfg: ServerConfig) -> ServerConfig:
    """跨字段校验（单一字段的约束已在 _coerce 里做完）。"""
    if cfg.chunk_overlap >= cfg.chunk_tokens:
        raise ConfigError(
            f"chunk_overlap={cfg.chunk_overlap} must be < "
            f"chunk_tokens={cfg.chunk_tokens}")

    strategy = (cfg.fuse_strategy or "rrf").lower()
    ignored = _ignored_fusion_fields(cfg, strategy)
    if ignored:
        # 不抛错：这些字段本身合法，只是在本策略下读不到。抛错会让"同一份配置
        # 换 fuse_strategy 做 A/B"变得不可能 —— 那正是我们要支持的用法。
        if strategy == "rrf":
            total = (1.0 + float(cfg.lexical_weight)
                     + float(cfg.expand_bm25_weight)
                     + float(cfg.expand_dense_weight))
            active = (
                "R1=1.0（写死的基准，见 store.py 的 "
                "`_rrf_dense(fused, vec_scores, 1.0)`）、"
                f"R2=lexical_weight={cfg.lexical_weight}、"
                f"R3=expand_bm25_weight={cfg.expand_bm25_weight}、"
                f"R4=expand_dense_weight={cfg.expand_dense_weight}，"
                f"名次衰减 k=rrf_k={cfg.rrf_k}；总票权 {total:g}"
                "（整体缩放不影响排序，故 R1 钉成 1.0 当基准）。"
                "调 dense 与 BM25 的话语权之比请改 lexical_weight，"
                "调扩展路请改 expand_bm25_weight / expand_dense_weight")
        else:
            active = (
                f"dense 用 vector_weight={cfg.vector_weight}、"
                f"BM25 用 bm25_weight={cfg.bm25_weight}"
                "（各自按并集内最大值归一化），再叠加 "
                f"expand_dense_weight={cfg.expand_dense_weight} 与 "
                f"expand_bm25_weight={cfg.expand_bm25_weight}")
        logger.warning("fuse_strategy=%r 下 %s **不参与融合** —— "
                       "本策略实际生效的是：%s。",
                       strategy, "、".join(ignored), active)
    return cfg


# ---------------------------------------------------------------------------
# YAML 读写
# ---------------------------------------------------------------------------
def load_yaml_file(path: str) -> dict:
    """读 YAML 配置文件；pyyaml 缺失或顶层不是 mapping 都抛 ConfigError。"""
    try:
        import yaml  # 延迟 import：不用 YAML 的环境不必装 pyyaml
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ConfigError("读取 YAML 配置需要 pyyaml：pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: 顶层必须是 mapping，实际是 {type(data).__name__}")
    return data


def split_yaml(data: dict, path: str = "<yaml>") -> tuple[dict, dict]:
    """拆分 YAML 顶层：run 级字段（exp_name/output_root）与 server 段。"""
    server = data.get(SERVER_SECTION) or {}
    if not isinstance(server, dict):
        raise ConfigError(
            f"{path}: `{SERVER_SECTION}` 段必须是 mapping，实际是 "
            f"{type(server).__name__}")
    run_level = {k: v for k, v in data.items() if k != SERVER_SECTION}
    unknown = [k for k in run_level
               if k not in RUN_LEVEL_KEYS and not str(k).startswith("_")]
    if unknown:
        raise ConfigError(
            f"{path}: 未知顶层字段 {unknown}；允许的顶层字段只有 "
            f"{list(RUN_LEVEL_KEYS)} 和 `{SERVER_SECTION}`"
            f"（以下划线开头的键会被忽略，可用来写备注）")
    return run_level, server


def dump_config(cfg: ServerConfig, path: str, run_level: dict | None = None) -> str:
    """把最终生效的配置写回 YAML（runs/<exp>/config.yaml 快照）。"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ConfigError("写 YAML 配置需要 pyyaml：pip install pyyaml") from exc
    payload: dict = {}
    if run_level:
        payload.update(run_level)
    payload[SERVER_SECTION] = asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def config_to_dict(cfg: ServerConfig) -> dict:
    return asdict(cfg)


# ---------------------------------------------------------------------------
# 进程级 override（CLI --set / 脚本里强制指定 db_path 等）
# ---------------------------------------------------------------------------
_OVERRIDES: dict[str, object] = {}


def set_overrides(mapping: dict | None) -> None:
    """注册全局 override（最高优先级）。传 None / {} 表示清空。"""
    _OVERRIDES.clear()
    if mapping:
        unknown = [k for k in mapping if k not in _SPECS]
        if unknown:
            raise ConfigError(f"override: 未知配置字段 {unknown}")
        _OVERRIDES.update(mapping)


def set_override(**kwargs) -> None:
    for key, value in kwargs.items():
        if key not in _SPECS:
            raise ConfigError(f"override: 未知配置字段 `{key}`")
        _OVERRIDES[key] = value


def clear_overrides() -> None:
    _OVERRIDES.clear()


def get_overrides() -> dict:
    return dict(_OVERRIDES)


def parse_overrides(items) -> dict:
    """把 CLI 的 ["retrieval_mode=dense", "rrf_k=10"] 解析成 dict（按字段类型校验）。"""
    out: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set 需要 key=value 形式，收到 {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        spec = _SPECS.get(key)
        if spec is None:
            raise ConfigError(f"--set: 未知配置字段 `{key}`")
        out[key] = _coerce(f"--set {key}", raw, spec)
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None,
                overrides: dict | None = None) -> ServerConfig:
    """Build a ServerConfig.

    优先级：overrides（含 `set_override()` 注册的全局 override）
          > YAML（`config_path` 或环境变量 `MEMORY_CONFIG`）
          > 环境变量（MEMORY_* / EMBEDDING_*）
          > ServerConfig 默认值

    不传任何参数时 = 改版前的纯环境变量行为，部署路径零影响。
    """
    path = config_path or os.getenv(CONFIG_ENV_VAR) or None

    cfg = ServerConfig()          # 4. 默认值
    cfg = _apply_env(cfg)         # 3. 环境变量
    if path:
        data = load_yaml_file(path)
        _, server = split_yaml(data, path)
        cfg = _apply_mapping(cfg, server, path)   # 2. YAML
    if overrides:
        cfg = _apply_mapping(cfg, overrides, "override")   # 1. 显式 override
    if _OVERRIDES:
        cfg = _apply_mapping(cfg, _OVERRIDES, "override")
    return _validate(cfg)


def load_run_config(config_path: str | None = None,
                    overrides: dict | None = None) -> tuple[ServerConfig, dict, str | None]:
    """同 `load_config()`，但一并返回 YAML 的 run 级字段与配置路径。

    供实验脚本用：`cfg, run_level, path = load_run_config("configs/baseline.yaml")`，
    `run_level` 里是 `exp_name` / `output_root`。
    """
    path = config_path or os.getenv(CONFIG_ENV_VAR) or None
    run_level: dict = {}
    if path:
        run_level, _ = split_yaml(load_yaml_file(path), path)
    cfg = load_config(config_path=path, overrides=overrides)
    return cfg, run_level, path
