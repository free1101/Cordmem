"""查询扩展：用 HyperE（Stanford DAWN / HazyResearch）WordNet 双曲嵌入把 query 扩成一批
语义相关的词，供检索段新增的两路使用：

    R3 = 扩展 + BM25        （`store.search` 里 `bm25_scores_exp`）
    R4 = 扩展 + dense       （`store.search` 里 `vec_scores_exp`）

R1（dense）/ R2（BM25）不变，扩展只是**多两路投票**，不会替换原路 —— 所以扩展词质量
差的时候，代价是"多一路噪声"，而不是"把原查询带偏"。

为什么是 HyperE 而不是 nltk + WordNet
-------------------------------------
HyperE 的 WordNet 嵌入满足：**两个实体的双曲距离 ÷ scale factor ≈ 输入图中的图距离**，
实测直接上位/下位词恰好 `d/scale = 1.0000`（`dog→canine`、`guitar→stringed_instrument`）。
于是只要一份嵌入文件，就能用"双曲最近邻"在**不装 nltk、不装 WordNet 数据库**的前提下
走层级：`d/scale ≈ 1` 是直接上位/下位词，`≈ 2` 隔一层。`src/requirements.txt` 因此
一行都不用加（只用 numpy）。

数据文件与**优雅降级**（最重要的一条）
--------------------------------------
嵌入文件放在仓库 `data/hyperE/hyperE.wn.10d/` 下，**不会跟着 `src/` 一起提交到平台**。
所以本模块的铁律是：**文件缺失 / 读失败 ⇒ 自动禁用扩展并打 WARNING，绝不抛异常**。
线上平台拿不到文件时，行为必须精确退回改版前的双路（R1+R2），而不是 500。

`_resolve()` 负责找文件：`run_eval.py` 是在 `local_eval/` 下启动的，所以配置里写的
`data/hyperE/...` 这种相对路径直接按 CWD 找必然失败 —— 必须再回退到"相对仓库根"
（本文件在 `src/` 下，`parents[1]` 即仓库根）。找不到就禁用。

词表口径（踩过的坑，别改回去）
------------------------------
* HyperE 文件里的 entity 是 WordNet **lemma**（`licence` / `commissioned_naval_officer`
  / `Holy_Order`），**不是 synset id**；
* 所以查表要用**未经词干还原**的词形 —— `src/chunking.py:lexical_tokens()` 会把
  `decision` 压成 `decis`，在 lemma 词表里必然查不到，覆盖率会假性归零；
* 但扩展出来的词**送回 BM25 时必须再过一遍 `lexical_tokens()`**（调用方做），
  否则索引侧的词干化 token 与扩展词对不上，扩展等于白做。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# `lexical_tokens()` 是本模块**过滤扩展词的最终依据**，见 `expand_with._accept_word()`
# 的说明：扩展词的合法形态由"调用方会再跑一次 lexical_tokens()"决定，所以判脏必须
# 走同一个函数。这不是"引入新依赖"—— `chunking` 与 `expander` 同在 `src/`，
# `chunking` 本身只依赖 `re`（+ 可选的 nltk），无循环导入。
from chunking import lexical_tokens                             # noqa: E402

logger_name = "aml_memory.expander"

#: 与 `chunking.TOKEN_PATTERN` 同一套切分（汉字单字 / 英数下划线成词 / 其余非空白单算）。
#: 这里复制字面量而不 import，是为了让**分词口径**这一件事可单独测试。
#: 但**词干还原不能复制**：见下面 `lexical_tokens` 的 import 说明 —— 扩展词的合法性
#: 必须按"送回 BM25 之后的最终形态"判，那一形态由 `chunking.lexical_tokens()` 决定。
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)

_SCALE_LINE = re.compile(r"scaling\s+factor", re.IGNORECASE)

#: `(1 - ‖v‖²)` 的下限。**必须极小**：实测本数据 `1-‖v‖²` 中位数只有 1.97e-05、
#: p25 是 3.1e-06、最小 6.8e-12，分母常态就在 1e-10 量级。写成 1e-5 那种"常规"值
#: 会直接篡改一半以上的点对距离（真值 1.0000 会被压成 0.038），整张距离矩阵失真。
_DENOM_FLOOR = 1e-30

#: 判定"同一 synset 的其它 lemma"（文件里独立成行、距离为 0 的孪生节点）的阈值。
#: 公开名：离线探针的统计口径必须与这里**同一个值**，否则两边算出来的"孪生率"
#: 对不上，排查时会以为算法不一致。
TWIN_EPS = 1e-6

#: 关系名 -> 文件名模板（实测文件名）。
RELATION_FILE = "hyperE.{relation}.10d.txt"
#: 优先加载的缓存后缀（`.gitattributes` 把 `*.pickle` 交给 Git LFS；`*.txt` 不在其中）。
PICKLE_SUFFIX = ".pickle"

#: 官方 10d 变体里的全部关系（hypernym 按词性拆两个文件）。
#: 白名单的**唯一真相来源在 config.EXPAND_RELATIONS**（配置校验要用它），这里只做
#: 转出，绝不另抄一份 —— 两处清单漂移会变成"配置写对了却校验报错"。
from config import EXPAND_RELATIONS as ALL_RELATIONS   # noqa: E402

#: 每个关系的**已知层级对**，用于启动期硬校验距离口径。必须按关系分开写：
#: 名词的 `dog-canine` 放进动词文件里当然找不到，混着写会假报失败。
KNOWN_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "hypernym_noun": (("dog", "canine"),
                      ("guitar", "stringed_instrument"),
                      ("guitar", "uke"),
                      ("entity", "physical_entity")),
    "hypernym_verb": (("catapult", "propel"), ("plead", "implore")),
    "domain_topic": (("fertilized_ovum", "genetic_science"),),
    "member_holonym": (("black_whale", "Globicephala"), ("genus_Sagitta", "sagitta")),
}

#: `max_graph_distance` 的默认值。**单位是图距离（≈跳数），不是相似度阈值**：
#: 实测直接上位/下位词恰好 `d/scale = 1.0`，所以 1.5 的含义是"只取直接相连的邻居，
#: 留 0.5 的数值余量"。不要照搬"相似度阈值 0.8"那类直觉 —— 这是本模块最容易
#: 被误改的地方（`configs/baseline.yaml` 里的同名默认值同样是 1.5）。
DEFAULT_MAX_GRAPH_DISTANCE = 1.5

#: 查询侧停用词。**必须要有**：不过滤时 `does` 会被扩成 `premier / premiere / scamp`
#: （`does` = 母鹿复数的同形词），白占预算还把 BM25 打分往无关方向拉。
#: 刻意只收功能词与助动词 —— `give -> provide/furnish/supply` 是有用的扩展，
#: 一刀切掉实义词会连收益一起砍掉。
_STOPWORDS = frozenset("""
a an the and or but if while of to in on at by for with about against between into
through during before after above below from up down out off over under again further
then once here there when where why how all any both each few more most other some
such no nor not only own same so than too very
i me my mine we us our ours you your yours he him his she her hers it its they them their
theirs this that these those who whom whose which what whichever whoever
is are was were be been being am do does did doing done
have has had having
can could may might must shall should will would
s t re ve ll d m
""".split())


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------
@dataclass
class HyperEIndex:
    """单个关系文件的词表 + 坐标（一个 lemma 一行）。"""
    relation: str
    path: str
    scale: float
    dim: int
    lemmas: list[str]
    lookup: dict[str, int]          # casefold(lemma) -> 行号
    mat: np.ndarray                 # (n, dim) float64
    sq_norm: np.ndarray             # (n,) ‖v‖²，距离式计算时复用

    @property
    def n_entities(self) -> int:
        return len(self.lemmas)

    def nearest(self, row: int, k: int,
                skip_twins: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """该行在本文档内的 k 个双曲最近邻：`(行号数组, d/scale 数组)`。

        `skip_twins=True` 丢弃距离 0 的孪生节点（同一 synset 的其它 lemma）——
        不排掉的话，每个词的"最近邻"永远是自己的同义写法，一条层级信息都取不到。
        """
        d = poincare_distance_matrix(
            self.mat[row:row + 1], self.sq_norm[row:row + 1],
            self.mat, self.sq_norm)[0]
        d[row] = np.inf
        room = min(int(k) + (8 if skip_twins else 0), d.shape[0])
        cand = (np.argpartition(d, room - 1)[:room]
                if room < d.shape[0] else np.arange(d.shape[0]))
        cand = cand[np.argsort(d[cand], kind="stable")]
        if skip_twins:
            cand = cand[d[cand] / self.scale > TWIN_EPS]
        cand = cand[:int(k)]
        return cand, d[cand] / self.scale


def poincare_distance_matrix(U: np.ndarray, sq_u: np.ndarray,
                             V: np.ndarray, sq_v: np.ndarray) -> np.ndarray:
    """Poincaré 球（曲率 −1）双曲距离，返回 `(len(U), len(V))`。

    用 `arccosh(1+y) = log1p(y + sqrt(y(y+2)))` 而不是 `np.arccosh(1+y)`：后者在
    `y` 很小时精度损失严重，而这里要判的正是 `d/scale ≈ 1` 这种小量级。
    """
    quad = sq_u[:, None] + sq_v[None, :] - 2.0 * (U @ V.T)
    np.maximum(quad, 0.0, out=quad)
    denom = (1.0 - sq_u)[:, None] * (1.0 - sq_v)[None, :]
    np.maximum(denom, _DENOM_FLOOR, out=denom)
    y = 2.0 * quad / denom
    return np.log1p(y + np.sqrt(y * (y + 2.0)))


def _build_index(relation: str, path: str, scale: float, dim: int,
                 lemmas: list[str], mat: np.ndarray) -> HyperEIndex:
    lookup: dict[str, int] = {}
    for i, lemma in enumerate(lemmas):
        lookup.setdefault(lemma.casefold(), i)   # 实测重复只是大小写碰撞，取首个
    return HyperEIndex(relation=relation, path=path, scale=float(scale),
                       dim=int(dim), lemmas=lemmas, lookup=lookup, mat=mat,
                       sq_norm=np.einsum("ij,ij->i", mat, mat))


def read_txt(path: Path, relation: str) -> HyperEIndex:
    """解析 HyperE GloVe 文本文件。

    首行是**标签 + 数字**：`Scaling factor 2.166084939249829` —— 不是裸数字，
    `float(line)` 会直接 ValueError；有的文件（`domain_topic`）首行前还有空行。
    """
    scale: float | None = None
    dim: int | None = None
    lemmas: list[str] = []
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            if scale is None:
                if _SCALE_LINE.search(line):
                    scale = float(line.split()[-1])
                    continue
                try:
                    scale = float(line.split()[0])
                    continue
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{lineno} 首行既不是 'Scaling factor <num>' "
                        f"也不是数字：{line[:80]!r}") from exc
            parts = line.rsplit()
            if dim is None:
                dim = len(parts) - 1
            elif len(parts) - 1 != dim:
                raise ValueError(
                    f"{path}:{lineno} 维度不一致：期望 {dim}，实际 {len(parts) - 1}")
            lemmas.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    if scale is None or dim is None or not rows:
        raise ValueError(f"{path}: 解析失败（找不到 Scaling factor 首行或没有数据行）")
    return _build_index(relation, str(path), scale, dim, lemmas,
                        np.asarray(rows, dtype=np.float64))


def read_pickle(path: Path, relation: str) -> HyperEIndex:
    import pickle
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    return _build_index(relation, str(path), blob["scale"], blob["dim"],
                        list(blob["lemmas"]),
                        np.asarray(blob["mat"], dtype=np.float64))


def load_index(path: Path, relation: str) -> HyperEIndex:
    """先找同名 `.pickle` 缓存，没有再解析 `.txt`。

    为什么优先 pickle：本仓库 `.gitattributes` 把 `*.pickle` 交给 Git LFS，但
    `*.txt` / `*.zip` **不在**规则里 —— 名词 hypernym 那个 17.5MB 的 txt 直接进
    git 对象库不合适。pickle 存 float64 原值，与 txt 逐位等价。
    """
    pkl = path.with_name(path.name[:-len(".txt")] + PICKLE_SUFFIX) \
        if path.name.endswith(".txt") else path
    if pkl.exists():
        return read_pickle(pkl, relation)
    return read_txt(path, relation)


# ---------------------------------------------------------------------------
# 词 -> 扩展词
# ---------------------------------------------------------------------------
def is_stopword(token: str) -> bool:
    return token.casefold() in _STOPWORDS


def lemma_candidates(token: str) -> list[str]:
    """把 query 里的词形还原成可能命中的 lemma（轻量复数回退）。

    刻意**不引入** Snowball/Porter：那类词干还原的目标是"压成词干"，与"还原成词典
    词形"方向相反，用在查表上会同时制造假阳（`analysis -> analys`）与假阴。
    这里只处理最规律的复数，命中不了就放弃 —— 宁可漏，不要错。
    """
    cands = [token]
    if len(token) > 3 and token.endswith("ies"):
        cands.append(token[:-3] + "y")        # agencies -> agency
    if len(token) > 3 and token.endswith("es"):
        cands.append(token[:-2])              # dishes -> dish
        cands.append(token[:-1])              # names  -> name
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        cands.append(token[:-1])              # dogs -> dog
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class Expansion:
    """一条扩展词及其来源（供 fullresult 溯源与人工复核）。"""
    term: str                  # 下划线已换成空格（与分词器口径一致）
    relation: str
    source: str                # 触发的 query 词
    graph_distance: float      # d / scale ≈ 图距离（1.0 = 一条边）

    def as_dict(self) -> dict:
        return {"term": self.term, "relation": self.relation,
                "source": self.source,
                "graph_distance": round(float(self.graph_distance), 4)}


def expand_with(text: str, indices: dict[str, HyperEIndex], *,
                per_token_n: int = 3, top_n: int = 20,
                max_graph_distance: float = DEFAULT_MAX_GRAPH_DISTANCE,
                min_token_len: int = 3,
                multiword: str = "split", drop_stopwords: bool = True,
                skip_capitalized: bool = True) -> list[Expansion]:
    """扩展算法的**唯一实现**（`Expander.expand` 与离线探针都调它）。

    做成自由函数的理由：离线探针（`local_test/tools/`)要在**不启动服务**的情况下
    复用同一套逻辑。算法里有两处极易写错的细节（分母 clamp 的量级、孪生节点必须
    跳过），一旦有两份副本就必然漂移 —— 探针量出来的数就不再代表线上行为。

    三个降噪开关都是实测踩出来的，不是预防性设计（细节见各常量处的注释）：
      * `drop_stopwords` —— 功能词不当扩展源（`does -> premier / premiere / scamp`）；
      * `skip_capitalized` —— 跳过首字母大写的词（`john -> public / lavatory`）；
      * `multiword` —— WordNet 多词 lemma 的 `split` / `drop` / `keep`。
    """
    if not indices:
        return []
    surface: list[str] = []
    seen_surface: set[str] = set()
    for pos, match in enumerate(_TOKEN_PATTERN.finditer(text or "")):
        raw = match.group(0)
        low = raw.casefold()
        if not any(c.isalnum() for c in low):
            continue
        if len(low) < int(min_token_len):
            continue
        if drop_stopwords and is_stopword(low):
            continue
        if skip_capitalized and pos > 0 and raw[:1].isupper():
            continue
        if low in seen_surface:
            continue
        seen_surface.add(low)
        surface.append(low)

    # 已经被 query 自己写出来的词，再扩进来只是重复计票（BM25 会把同一 token
    # 的 tf 累加），必须排掉。
    own: set[str] = set()
    for tok in surface:
        own.update(lemma_candidates(tok))

    found: list[Expansion] = []
    seen_terms: set[str] = set()

    def _accept(term: str, rel: str, source: str, g: float) -> None:
        key = term.casefold()
        if not term or key in seen_terms or key in own:
            return
        seen_terms.add(key)
        found.append(Expansion(term=term, relation=rel, source=source,
                               graph_distance=g))

    def _accept_word(word: str, rel: str, source: str, g: float) -> None:
        """**单个单词的唯一入口** —— 过滤只写一份，单词分支与多词拆分分支共用。

        踩过的 bug（实测发现，不是预防性设计），两层，都已修：

        1. 第一层：过滤原先只写在"多词拆分"那一支，**单词那一支直接
           `_accept(flat, ...)`**，于是 WordNet 里 1-2 字符的 lemma 与停用词
           原样进了词袋（实测 8058 个扩展词里 102 个，影响 93/861 题）。
        2. 第二层：只看**字面**还是不够。扩展词的最终形态由调用方再跑一次
           `lexical_tokens()` 决定，而这一步既会按标点再切、也会做词干还原：

               pilgrim's   -> [pilgrim, s]        （所有格，字面干净）
               lock-up     -> [lock, up]          （连字符，字面干净）
               beingness   -> [being]             （词干还原成停用词）
               mores       -> [more]

           修完第一层后仍有 89 个残留，**全部来自这一类**。所以判脏必须发生在
           `lexical_tokens()` 之后 —— 这正是本模块 import `chunking` 的唯一理由。

        判定口径与模块一贯的"宁可漏，不要错"一致：**只要这个 lemma 会给词袋带来
        任何一个脏 token，整条丢掉**（词袋是按 term 存的，只丢脏的那半做不到）。

        这条"丢整条"听起来很狠，实际代价极小，因为前面两步已经把能救的都救了：
          * `pilgrim's` 先剥所有格 -> `pilgrim` 留住；
          * `lock-up`   先按连字符拆 -> `lock` 留住，只有 `up` 丢；
        真被整条丢掉的只剩"词干还原后本身就是停用词"的那类（`beingness` -> `being`、
        `mores` -> `more`），它们本来就是垃圾词，丢了没有损失。

        两道过滤语义刻意不同：
          * **长度无开关** —— `min_token_len` 是词表硬约束，短 token 在任何配置下
            都是噪声（源词也这么滤，两侧口径必须一致）；
          * **停用词受 `drop_stopwords` 开关** —— 它是可调策略，不是硬约束。
        """
        toks = lexical_tokens(word)
        if not toks:
            return
        for tok in toks:
            if len(tok) < int(min_token_len):
                return
            if drop_stopwords and is_stopword(tok):
                return
        _accept(word, rel, source, g)

    for tok in surface:
        hits: list[tuple[str, int]] = []
        for rel, idx in indices.items():
            for cand in lemma_candidates(tok):
                row = idx.lookup.get(cand)
                if row is not None:
                    hits.append((rel, row))
        for rel, row in hits:
            idx = indices[rel]
            order, gd = idx.nearest(row, per_token_n)
            for o, g in zip(order, gd):
                if not np.isfinite(g) or g > max_graph_distance:
                    continue
                raw = idx.lemmas[int(o)].strip()
                # 所有格：WordNet 里存的是 `pilgrim's`，`lexical_tokens()` 会把它
                # 切成 [pilgrim, s]，那个 `s` 是纯噪声。先剥掉再往下走，
                # 保住 `pilgrim` 这个有用的词。
                if len(raw) > 2 and raw.endswith(("'s", "\u2019s")):
                    raw = raw[:-2]
                if not raw:
                    continue
                # 连字符也是 WordNet 的多词分隔符（`lock-up` / `ready-to-wear`），
                # 只换下划线会让它们被当成**单词**而绕过多词拆分那一支。
                flat = raw.replace("_", " ").replace("-", " ")
                if multiword == "keep" or " " not in flat:
                    # 距离 < 1 的邻居不是错误：那是"近孪生"节点（`genus_Orontium`
                    # / `Orontium`），对扩展来说等同同义词，不要按"必须≈1.0"滤掉。
                    _accept_word(flat, rel, tok, float(g))
                elif multiword == "split":
                    # 多词 lemma 拆成单词：`recording studio` 的 `recording` 有用，
                    # `theater of the absurd` 的 `of/the` 由统一入口挡掉。
                    for part in flat.split():
                        _accept_word(part, rel, tok, float(g))
                # multiword == "drop" 时整条丢弃
    # 先按"离 query 词多近"排（小的在前），同距离保持出现顺序 → 结果可复现
    found.sort(key=lambda e: e.graph_distance)
    return found[:int(top_n)]


# ---------------------------------------------------------------------------
# Expander
# ---------------------------------------------------------------------------
class Expander:
    """配置驱动的扩展器。**任何加载失败都只是"禁用"，不是异常。**"""

    def __init__(self, relations=("hypernym_noun",),
                 files: dict[str, str] | None = None,
                 base_dir: str = "data/hyperE/hyperE.wn.10d",
                 per_token_n: int = 3, top_n: int = 20,
                 max_graph_distance: float = DEFAULT_MAX_GRAPH_DISTANCE,
                min_token_len: int = 3,
                 multiword: str = "split", drop_stopwords: bool = True,
                 skip_capitalized: bool = True, verify_pairs: bool = True) -> None:
        self.relations = tuple(relations)
        self.per_token_n = int(per_token_n)
        self.top_n = int(top_n)
        self.max_graph_distance = float(max_graph_distance)
        self.min_token_len = int(min_token_len)
        self.multiword = multiword
        self.drop_stopwords = bool(drop_stopwords)
        self.skip_capitalized = bool(skip_capitalized)
        self.files = dict(files or {})
        self.base_dir = base_dir
        self.indices: dict[str, HyperEIndex] = {}
        self.load_errors: dict[str, str] = {}
        self.pair_check: dict[str, object] = {}
        self._load()
        if verify_pairs and self.indices:
            self._verify_pairs()

    # ---- 加载 ----
    def _resolve(self, path: str) -> Path | None:
        """按 CWD -> 仓库根 依次尝试（`run_eval.py` 的 CWD 是 `local_eval/`）。"""
        p = Path(path)
        cands = [p] if p.is_absolute() else [
            p, Path(__file__).resolve().parents[1] / p]
        for c in cands:
            if c.exists():
                return c
        return None

    def _load(self) -> None:
        for rel in self.relations:
            raw = self.files.get(rel) or (Path(self.base_dir) /
                                          RELATION_FILE.format(relation=rel))
            found = self._resolve(str(raw))
            if found is None:
                self.load_errors[rel] = f"文件不存在：{raw}"
                continue
            try:
                self.indices[rel] = load_index(found, rel)
            except Exception as exc:  # noqa: BLE001  坏文件只能降级，不能炸服务
                self.load_errors[rel] = f"{type(exc).__name__}: {exc}"
        if not self.relations:
            return          # 配置里没开扩展：静默，这不是异常（默认路径就是它）
        if not self.indices:
            print(f"[WARN] expander: 查询扩展已**禁用** —— 未能加载任何关系文件 "
                  f"(relations={list(self.relations)}, base_dir={self.base_dir!r})。"
                  f" 明细：{self.load_errors}。"
                  f" 线上平台不带 data/hyperE/ 数据文件时这是预期行为，检索会退回"
                  f" R1+R2 双路；若要本地生效请确认数据文件路径，然后重启进程"
                  f"（结果被 lru_cache 按配置签名缓存）。", flush=True)
        elif self.load_errors:
            print(f"[WARN] expander: 部分关系未加载：{self.load_errors}", flush=True)

    def _verify_pairs(self) -> None:
        """启动期硬校验距离口径：已知层级对必须落在 `1.0 ± 0.01`。

        这条不过就说明解析 / scale / 距离式之一坏了 —— 此时**继续用扩展词是有害的**
        （会把无关词塞进 BM25），所以直接把该关系从启用列表里摘掉。
        """
        for rel, idx in list(self.indices.items()):
            pairs = KNOWN_PAIRS.get(rel)
            if not pairs:
                continue
            vals, bad = [], []
            for a, b in pairs:
                ra, rb = idx.lookup.get(a.casefold()), idx.lookup.get(b.casefold())
                if ra is None or rb is None:
                    continue
                g = float(poincare_distance_matrix(
                    idx.mat[ra:ra + 1], idx.sq_norm[ra:ra + 1],
                    idx.mat[rb:rb + 1], idx.sq_norm[rb:rb + 1])[0, 0] / idx.scale)
                vals.append(round(g, 4))
                if abs(g - 1.0) > 0.01:
                    bad.append((a, b, round(g, 4)))
            self.pair_check[rel] = {"values": vals, "failed": bad}
            if bad:
                print(f"[WARN] expander: {rel} 的已知层级对不在 1.0±0.01（{bad}）—— "
                      f"距离口径可疑，该关系**已禁用**（继续用会把无关词塞进 BM25）。",
                      flush=True)
                self.indices.pop(rel, None)
                self.load_errors[rel] = f"口径校验未通过：{bad}"

    # ---- 状态 ----
    @property
    def available(self) -> bool:
        return bool(self.indices)

    @property
    def identity(self) -> str:
        """实际生效的扩展配置（用于日志 / 产物留痕）。"""
        return (f"relations={list(self.indices)}"
                f"|per_token_n={self.per_token_n}|top_n={self.top_n}"
                f"|max_gd={self.max_graph_distance}|multiword={self.multiword}")

    def describe(self) -> dict:
        return {
            "enabled": self.available,
            "relations": list(self.indices),
            "n_entities": {r: i.n_entities for r, i in self.indices.items()},
            "scale": {r: round(i.scale, 6) for r, i in self.indices.items()},
            "params": {"per_token_n": self.per_token_n, "top_n": self.top_n,
                       "max_graph_distance": self.max_graph_distance,
                       "min_token_len": self.min_token_len,
                       "multiword": self.multiword,
                       "drop_stopwords": self.drop_stopwords,
                       "skip_capitalized": self.skip_capitalized},
            "load_errors": self.load_errors or None,
            "pair_check": self.pair_check or None,
        }

    # ---- 扩展 ----
    def expand(self, text: str) -> list[Expansion]:
        """双曲最近邻扩展。`available=False` 时返回空列表（调用方不必判空）。"""
        if not self.indices:
            return []
        return expand_with(
            text, self.indices, per_token_n=self.per_token_n, top_n=self.top_n,
            max_graph_distance=self.max_graph_distance,
            min_token_len=self.min_token_len, multiword=self.multiword,
            drop_stopwords=self.drop_stopwords,
            skip_capitalized=self.skip_capitalized)


# ---------------------------------------------------------------------------
# 拼文本（供 BM25 与 dense 两路各自使用）
# ---------------------------------------------------------------------------
def bm25_query_text(text: str, expansions: list[Expansion]) -> str:
    """R3 的 BM25 输入 = **原查询文本 + 扩展词**。

    这样 R3 与 R2 的差异**只有扩展词**一项，A/B 才干净。
    调用方**必须**再对它跑一次 `chunking.lexical_tokens()` —— 索引语料是先过
    `lexical_tokens()`（含词干还原）才建的，扩展词不跟着做词干还原就跟索引对不上。
    """
    if not expansions:
        return text
    return f"{text} " + " ".join(e.term for e in expansions)


def dense_query_text(text: str, expansions: list[Expansion],
                     mode: str = "related") -> str:
    """R4 的 bge 输入。

    `mode="related"`（默认）拼成 `问题 + "Related terms: a, b, c"`：bge 这类检索
    模型对"问题后跟一串关键词"敏感，给个显式标签比裸拼稳。
    `mode="terms_only"` 则丢掉原问题、只用扩展词（更激进，留作对照）。
    """
    if not expansions:
        return text
    terms = ", ".join(e.term for e in expansions)
    if mode == "terms_only":
        return f"Related terms: {terms}"
    return f"{text}\nRelated terms: {terms}"


@lru_cache(maxsize=4)
def _cached(relations: tuple, files: tuple, base_dir: str, per_token_n: int,
            top_n: int, max_graph_distance: float, min_token_len: int,
            multiword: str, drop_stopwords: bool, skip_capitalized: bool,
            verify_pairs: bool) -> Expander:
    return Expander(relations=relations, files=dict(files), base_dir=base_dir,
                    per_token_n=per_token_n, top_n=top_n,
                    max_graph_distance=max_graph_distance,
                    min_token_len=min_token_len, multiword=multiword,
                    drop_stopwords=drop_stopwords,
                    skip_capitalized=skip_capitalized,
                    verify_pairs=verify_pairs)


def get_expander(*, relations=("hypernym_noun",), files=None,
                 base_dir: str = "data/hyperE/hyperE.wn.10d",
                 per_token_n: int = 3, top_n: int = 20,
                 max_graph_distance: float = DEFAULT_MAX_GRAPH_DISTANCE,
                min_token_len: int = 3,
                 multiword: str = "split", drop_stopwords: bool = True,
                 skip_capitalized: bool = True,
                 verify_pairs: bool = True) -> Expander:
    """进程内单例（同 `embedder.get_embedder()`）。

    缓存键是**配置签名**而不是文件 mtime：与 `_check_identity()` 的口径一致 ——
    改配置要重启进程。注意若首次调用时数据文件还不存在，空结果会被缓存住；
    把文件放好后重启即可（这条也写进了 WARNING 的排查提示）。
    """
    return _cached(tuple(relations), tuple(sorted((files or {}).items())), base_dir,
                   int(per_token_n), int(top_n), float(max_graph_distance),
                   int(min_token_len), multiword, bool(drop_stopwords),
                   bool(skip_capitalized), bool(verify_pairs))
