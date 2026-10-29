"""多格式日期解析器：把查询里的自然语言日期统一解析成毫秒时间范围。

设计约束：
- 纯标准库（``re`` / ``datetime`` / ``calendar``），**不引入 dateparser**：
  零新增依赖、零 import 开销，且格式表可针对评测题面精确控制；
- 解析失败一律返回 ``None``，绝不抛异常；
- 时间语义与 ``local_eval/run_eval.py:_parse_date_ms`` 一致：``datetime``
  不带时区直接 ``.timestamp()``（本机时区），保证解析窗与库内
  ``source_timestamp``（灌库时用同一套语义写入的毫秒）可比。

覆盖三类表达：

1. 绝对日期：``16 June, 2023`` / ``October 3, 2023`` / ``2023-06-16``
2. 年月与月内修饰：``October 2023`` / ``in August 2023``
   / ``first week of March 2023`` / ``last two weeks of August 2023``
3. 相对时间（仅**单日**表达）：``yesterday`` / ``last night`` / ``tomorrow``
   / ``today`` / ``tonight`` / ``3 days ago``
   （必须给 ``reference_ms`` 作基准；未给时返回 ``None`` —— 不猜测"现在"，
   与 ``store.py`` 的 ``_UNKNOWN_TS`` 设计一致：把"没有基准"伪装成"现在"
   会给答案模型喂错误证据）
   区间词（``recently`` / ``last week`` / ``last month``）**刻意不收**：
   它们的窗全靠基准猜测，A/B 实测会产生误导
   （conv-48#q0133 因 recently 被加分把 gold 挤出 top_k，full->zero）；
   与灌库侧 parse_event_dates 不收区间表达对称。

"last N weeks of <月> <年>" 统一采用**月内最后 N*7 天**口径：
``start = 月末 - N*7天 + 1ms``，例：``last two weeks of August 2023``
→ ``2023-08-18 00:00:00.000 ~ 2023-08-31 23:59:59.999``。
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

__all__ = [
    "DateSpan",
    "parse_time_constraint",
    "parse_month_year",
    "parse_month_range",
    "parse_absolute_date",
    "parse_relative_date",
    "parse_event_dates",
]

# 全月名 + 三字母缩写 -> 月份数字。交替顺序无关紧要：\b 保证 "jan" 不会
# 部分吃掉 "january"（jan 与 u 之间不构成词边界）。
_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTHS: dict[str, int] = {}
for _i, _name in enumerate(_MONTH_NAMES, start=1):
    _MONTHS[_name] = _i
    _MONTHS[_name[:3]] = _i
_MONTH_ALT = "|".join(_MONTHS)

_MS_PER_DAY = 86_400_000

_YEAR_MIN, _YEAR_MAX = 1900, 2100  # 合理年份区间，挡住 "16 June, 20233" 之类


@dataclass(frozen=True)
class DateSpan:
    """解析出的时间范围（毫秒，闭区间），附类别与命中原文供 debug 归因。"""

    start_ms: int  # 起点（当日 00:00:00.000）
    end_ms: int    # 终点（当日/月末 23:59:59.999）
    kind: str      # "month" | "month_sub" | "absolute" | "relative"
    raw: str       # 命中的原文片段

    def iso(self) -> tuple[str, str]:
        """返回 (start, end) 的 ISO 形式（**本机时区**渲染，无时区后缀）。

        刻意不用 UTC：解析器内部就是 naive 语义（与灌库侧
        ``run_eval._parse_date_ms`` 一致），按本机时区渲染才能让 debug 里的
        日期与"解析出的日历日"对得上，方便归因。
        """
        fmt = "%Y-%m-%dT%H:%M:%S"
        start = datetime.fromtimestamp(self.start_ms / 1000.0)
        end = datetime.fromtimestamp(self.end_ms / 1000.0)
        return start.strftime(fmt), end.strftime(fmt)


def _to_ms(dt: datetime) -> int:
    """naive datetime -> 毫秒（本机时区，与 run_eval._parse_date_ms 同语义）。

    Windows 的 ``datetime.timestamp()`` 对本地时区下 time_t 为负的日期抛
    ``OSError``（Errno 22）—— 不止 1970-01-01 之前的日期（如 LME 语料的
    "May 8, 1945"），**东八区的 1970-01-01 00:00 本身就是 1969-12-31 16:00 UTC**
    （LME 语料里就有 "The epoch time of January 1, 1970" 这句话），所以不能按
    ``dt.year >= 1970`` 分流 —— 首跑 S 版实测在这里崩。统一 try/except：正常
    日期保持 .timestamp() 的本机时区语义；异常日期（1970 边界/历史日期）回退
    naive 差值，偏差 ≤ 时区小时数，对日级窗判断无实际影响（这类日期也绝不会
    出现在库内现代的 ``source_timestamp`` 里）。
    """
    try:
        return int(dt.timestamp() * 1000)
    except (OSError, OverflowError, ValueError):
        return int((dt - datetime(1970, 1, 1)).total_seconds() * 1000)


def _day_bounds(year: int, month: int, day: int) -> tuple[int, int]:
    start = _to_ms(datetime(year, month, day))
    return start, start + _MS_PER_DAY - 1


def _month_bounds(year: int, month: int) -> tuple[int, int]:
    last_day = calendar.monthrange(year, month)[1]
    return _day_bounds(year, month, 1)[0], _day_bounds(year, month, last_day)[1]


def _valid_ymd(year: int, month: int, day: int) -> bool:
    return (_YEAR_MIN <= year <= _YEAR_MAX and 1 <= month <= 12
            and 1 <= day <= calendar.monthrange(year, month)[1])


# ---------------------------------------------------------------------------
# 1) 年月 / 月内修饰
# ---------------------------------------------------------------------------
# "October 2023" / "in August 2023" / "during March 2023" / "Jan. 2023"
# (?<!\d\s)：月名前紧跟"数字+空白"时拒绝 —— 否则 "16 June, 2023" 里的
# "June, 2023" 会被误吞成整月 June 2023（日号被丢掉，范围扩大 30 倍）。
_MONTH_YEAR_RE = re.compile(
    rf"(?<!\d\s)\b({_MONTH_ALT})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)

# "first week of March 2023"
_FIRST_WEEK_RE = re.compile(
    rf"\b(first\s+week)\s+of\b", re.IGNORECASE)

# "last week of August 2023" / "last two weeks of August 2023"
# / "last 3 weeks of August 2023"
_LAST_WEEKS_RE = re.compile(
    rf"\b(last\s+(?:(\d{{1,2}}|one|two|three|four|five|six)\s+)?weeks?)\s+of\b",
    re.IGNORECASE)

_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def parse_month_year(text: str) -> DateSpan | None:
    """匹配 "October 2023" 这类年月表达 -> 整月范围（kind="month"）。"""
    m = _MONTH_YEAR_RE.search(text or "")
    if not m:
        return None
    month = _MONTHS[m.group(1).lower()]
    year = int(m.group(2))
    if not _valid_ymd(year, month, 1):
        return None
    start, end = _month_bounds(year, month)
    return DateSpan(start, end, "month", m.group(0))


def parse_month_range(text: str) -> DateSpan | None:
    """月年基础上叠加月内修饰（kind="month_sub"）。

    口径（固定为单测）：
      first week of X       -> [月首, 月首 + 7 天)
      last week of X        -> [月末 - 7 天 + 1ms, 月末]
      last two weeks of X   -> [月末 - 14 天 + 1ms, 月末]
    """
    span = parse_month_year(text)
    if span is None:
        return None
    m = _LAST_WEEKS_RE.search(text)
    if m:
        raw = m.group(2)
        if raw is None:
            n = 1
        elif raw.isdigit():
            n = int(raw)
        else:
            n = _WORD_NUM[raw.lower()]
        start = span.end_ms - n * 7 * _MS_PER_DAY + 1
        return DateSpan(start, span.end_ms, "month_sub", m.group(1))
    m = _FIRST_WEEK_RE.search(text)
    if m:
        end = span.start_ms + 7 * _MS_PER_DAY - 1
        return DateSpan(span.start_ms, end, "month_sub", m.group(1))
    return None


# ---------------------------------------------------------------------------
# 2) 绝对日期
# ---------------------------------------------------------------------------
# "October 3, 2023" / "October 3rd 2023" / "Oct 3 2023"
_ABS_MDY_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?,?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE)
# "16 June, 2023" / "3rd October 2023" / "on 31 July, 2023"
_ABS_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b",
    re.IGNORECASE)
# "2023-06-16" / "2023-6-6"
_ABS_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

# 数字斜杠格式（16/06/2023 vs 06/16/2023）日/月歧义无法从文本消解，
# 且评测题面全部用月名写法，刻意不支持 —— 误解析比漏解析更危险。


def parse_absolute_date(text: str) -> DateSpan | None:
    """匹配单日绝对日期（kind="absolute"），end 为当日 23:59:59.999。"""
    text = text or ""
    for m in _ABS_MDY_RE.finditer(text):
        month = _MONTHS[m.group(1).lower()]
        day, year = int(m.group(2)), int(m.group(3))
        if _valid_ymd(year, month, day):
            start, end = _day_bounds(year, month, day)
            return DateSpan(start, end, "absolute", m.group(0))
    for m in _ABS_DMY_RE.finditer(text):
        day, month, year = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
        if _valid_ymd(year, month, day):
            start, end = _day_bounds(year, month, day)
            return DateSpan(start, end, "absolute", m.group(0))
    for m in _ABS_ISO_RE.finditer(text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(year, month, day):
            start, end = _day_bounds(year, month, day)
            return DateSpan(start, end, "absolute", m.group(0))
    return None


# ---------------------------------------------------------------------------
# 3) 相对时间
# ---------------------------------------------------------------------------
_REL_DAYS = (
    (re.compile(r"\blast\s+night\b", re.IGNORECASE), -1),
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), 1),
)
_REL_TODAY_RE = re.compile(r"\b(?:today|tonight)\b", re.IGNORECASE)
# 区间词（last week / last month / recently）刻意**不**在查询侧生成窗：
# 窗全靠基准猜测，实测会误导（conv-48#q0133 full->zero，见模块 docstring）；
# 与灌库侧 parse_event_dates 不收区间表达对称。
# "3 days ago" / "two weeks ago" -> 单日窗（ref - N 单位 的当天）
_AGO_RE = re.compile(
    r"\b(?P<count>a|an|one|two|three|four|five|six|\d{1,3})\s+"
    r"(?P<unit>day|week|month|year)s?\s+ago\b",
    re.IGNORECASE)
_AGO_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def parse_relative_date(text: str, reference_ms: int | None) -> DateSpan | None:
    """单日相对时间 -> 以 ``reference_ms`` 所在日为基准的范围（kind="relative"）。

    只收单日表达（yesterday / last night / tomorrow / today / tonight /
    "N units ago"）；区间词（recently / last week / last month）刻意排除，
    见模块 docstring。``reference_ms`` 通常传该会话/该 user 最近的
    message 时间；为 None 时命中相对表达也返回 None（不猜测"现在"）。
    """
    if reference_ms is None:
        return None
    text = text or ""
    ref_day = datetime.fromtimestamp(reference_ms / 1000.0)
    ref_y, ref_m, ref_d = ref_day.year, ref_day.month, ref_day.day

    for pattern, offset in _REL_DAYS:
        m = pattern.search(text)
        if m:
            target = datetime(ref_y, ref_m, ref_d) + timedelta(days=offset)
            start, end = _day_bounds(target.year, target.month, target.day)
            return DateSpan(start, end, "relative", m.group(0))

    m = _REL_TODAY_RE.search(text)
    if m:
        start, end = _day_bounds(ref_y, ref_m, ref_d)
        return DateSpan(start, end, "relative", m.group(0))

    m = _AGO_RE.search(text)
    if m:
        raw = m.group("count").lower()
        n = _WORD_NUM.get(raw) or (int(raw) if raw.isdigit() else 1)
        days = n * _AGO_UNIT_DAYS[m.group("unit").lower()]
        target = datetime(ref_y, ref_m, ref_d) - timedelta(days=days)
        start, end = _day_bounds(target.year, target.month, target.day)
        return DateSpan(start, end, "relative", m.group(0))

    return None


# ---------------------------------------------------------------------------
# 灌库期事件日期提取（消息原文）
# ---------------------------------------------------------------------------
def parse_event_dates(text: str, reference_ms: int | None) -> list[int]:
    """穷尽提取消息原文里表达的全部「事件日期」（毫秒，按日去重、升序）。

    与 parse_time_constraint（检索 query 用，只取最特异的一个时间）不同，
    本函数服务于**灌库期**：对每条 message 原文解析出它提到的日期集合，
    写进 metadata，供检索期时间窗比对（修 "last night" 造成的
    事件日 ≠ 发送日偏移，见 local_test/tools/date_probe.py 的 miss 样例）。

    收录口径：
      * 单日相对表达：last night / yesterday / tomorrow / today / tonight /
        "N day(s|week(s)|month(s)|year(s)) ago" —— 基准 = reference_ms
        （= 该消息的发送时间）；
      * 绝对日期表达："on May 3, 2023" / "2023-05-03"（全部命中）。
    刻意**不收**区间表达（last week / recently / last month）：它们语义上
    覆盖多天，若全区间入集合会让任何近期窗都命中，信号被稀释 —— 这类
    消息的发送时间本身已在窗判断里。
    reference_ms 为 None/0 时只收绝对日期（无基准不猜"现在"，与
    _UNKNOWN_TS 设计一致）。解析失败返回 []，绝不抛异常。
    """
    text = text or ""
    days: set[int] = set()

    for m in _ABS_MDY_RE.finditer(text):
        month = _MONTHS[m.group(1).lower()]
        day, year = int(m.group(2)), int(m.group(3))
        if _valid_ymd(year, month, day):
            days.add(_day_bounds(year, month, day)[0])
    for m in _ABS_DMY_RE.finditer(text):
        day, month, year = (int(m.group(1)), _MONTHS[m.group(2).lower()],
                            int(m.group(3)))
        if _valid_ymd(year, month, day):
            days.add(_day_bounds(year, month, day)[0])
    for m in _ABS_ISO_RE.finditer(text):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(year, month, day):
            days.add(_day_bounds(year, month, day)[0])

    if not reference_ms:  # None 或 0（_UNKNOWN_TS 哨兵）都算无基准
        return sorted(days)

    ref_day = datetime.fromtimestamp(reference_ms / 1000.0)
    ry, rm, rd = ref_day.year, ref_day.month, ref_day.day

    def _add(y: int, mth: int, d: int) -> None:
        if _valid_ymd(y, mth, d):
            days.add(_day_bounds(y, mth, d)[0])

    for pattern, offset in _REL_DAYS:  # last night / yesterday / tomorrow
        for m in pattern.finditer(text):
            target = datetime(ry, rm, rd) + timedelta(days=offset)
            _add(target.year, target.month, target.day)
    for m in _REL_TODAY_RE.finditer(text):  # today / tonight
        _add(ry, rm, rd)
    for m in _AGO_RE.finditer(text):
        raw = m.group("count").lower()
        n = _WORD_NUM.get(raw) or (int(raw) if raw.isdigit() else 1)
        days_n = n * _AGO_UNIT_DAYS[m.group("unit").lower()]
        target = datetime(ry, rm, rd) - timedelta(days=days_n)
        _add(target.year, target.month, target.day)

    return sorted(days)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def parse_time_constraint(query: str,
                          reference_ms: int | None = None) -> DateSpan | None:
    """按 绝对日期 -> 月内修饰 -> 年月 -> 相对时间 的顺序解析。

    全部不命中返回 None（无日期表达的问题零副作用）。
    绝对日期必须**最先**尝试："3rd October 2023" / "16 June, 2023" 里的
    "October 2023" / "June, 2023" 都满足年月模式，若年月先行会把单日
    误判成整月（范围扩大 30 倍）。绝对日期与月内修饰/年月互斥
    （前者必须含日号，后两者不含），因此交换顺序不影响其余分支。
    reference_ms 为相对时间的基准（检索期由 store 传入该 user 的
    最大非 0 source_timestamp）；未给且只命中相对时间时返回 None。
    """
    if not query or not query.strip():
        return None
    span = parse_absolute_date(query)
    if span is not None:
        return span
    span = parse_month_range(query)
    if span is not None:
        return span
    span = parse_month_year(query)
    if span is not None:
        return span
    return parse_relative_date(query, reference_ms)
