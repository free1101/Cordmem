#!/usr/bin/env python3
"""远端 Add/Search 契约自检（不消耗 AML 官方 smoke 配额）。

打一个**已经在跑**的 memory 服务（容器或域名皆可），逐项验证：
  1. GET /health（无 key）-> 2xx
  2. POST /add（无 key）-> 401/503（确认鉴权真的开了）
  3. POST /add（带 key）-> 200, success=true, 三 ID 原样回显
  4. POST /add 同 request_id 不同 body -> 409（冲突）
  5. POST /add 同 request_id 同 body   -> 200（幂等）
  6. POST /search（带 key）-> data 是 list，每条非空 id/content，条数 <= top_k
  7. POST /search 换一个 user_id       -> 空数组（隔离）
  8. top_k 边界：top_k=100 时返回条数不超过 100

只依赖标准库（ECS 默认有 python3），无需 pip。

用法：
  python3 deploy/check_remote.py --base http://127.0.0.1:8000 --key <AML_API_KEY>
  python3 deploy/check_remote.py --base https://aml.your-domain.com --key <AML_API_KEY>
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

REQ = json.dumps  # alias


def _req(method: str, base: str, path: str, payload: dict | None, key: str | None):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Api-Key"] = key
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            body = resp.read().decode()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def _check(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--key", required=True, help="AML_API_KEY（带正确 key 才能跑 add/search）")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    ok = True
    rid = "smoke:check_remote:conv-0:chunk-0"
    uid = "smoke:check_remote:conv-0"
    sid = "smoke:check_remote:sample:0"
    msg = [{"role": "user", "content": "I love hiking in the mountains on weekends."}]

    # 1. health 免鉴权
    st, _ = _req("GET", args.base, "/health", None, None)
    ok &= _check("GET /health 免鉴权 2xx", 200 <= st < 300, f"status={st}")

    # 2. 无 key 的 add 应被拒（401/503）
    st, _ = _req("POST", args.base, "/add", {"request_id": rid, "messages": msg,
                                             "user_id": uid, "session_id": sid}, None)
    ok &= _check("POST /add 无 key 被拒(401/503)", st in (401, 503), f"status={st}")

    # 3. 带 key 的 add 成功 + 三 ID 回显
    st, body = _req("POST", args.base, "/add",
                    {"request_id": rid, "messages": msg, "user_id": uid, "session_id": sid},
                    args.key)
    good = (st == 200 and isinstance(body, dict) and body.get("success") is True
            and body.get("request_id") == rid and body.get("user_id") == uid
            and body.get("session_id") == sid)
    ok &= _check("POST /add 带 key 成功+三ID回显", good, f"status={st} body={body}")

    # 4. 同 id 不同 body -> 409
    st, _ = _req("POST", args.base, "/add",
                 {"request_id": rid,
                  "messages": [{"role": "user", "content": "DIFFERENT body"}],
                  "user_id": uid, "session_id": sid}, args.key)
    ok &= _check("POST /add 同id不同body -> 409", st == 409, f"status={st}")

    # 5. 同 id 同 body -> 200 幂等
    st, _ = _req("POST", args.base, "/add",
                 {"request_id": rid, "messages": msg, "user_id": uid, "session_id": sid},
                 args.key)
    ok &= _check("POST /add 同id同body -> 200 幂等", st == 200, f"status={st}")

    # 6. search 返回合规 data
    st, body = _req("POST", args.base, "/search",
                    {"query": "What do I like to do on weekends?",
                     "user_id": uid, "top_k": args.top_k}, args.key)
    data = body.get("data") if isinstance(body, dict) else None
    valid = (st == 200 and isinstance(data, list)
             and all(isinstance(h, dict) and h.get("id") and h.get("content")
                     for h in data)
             and len(data) <= args.top_k)
    ok &= _check(f"POST /search 返回合规 data(<= {args.top_k})",
                 valid, f"status={st} n={len(data) if isinstance(data, list) else 'n/a'}")
    if isinstance(data, list) and data:
        print(f"        top-1: id={data[0].get('id')!r} "
              f"score={data[0].get('score')} content={data[0].get('content')[:60]!r}")

    # 7. 跨 user_id 隔离
    st, body = _req("POST", args.base, "/search",
                    {"query": "anything", "user_id": "smoke:check_remote:OTHER",
                     "top_k": args.top_k}, args.key)
    other = body.get("data") if isinstance(body, dict) else None
    ok &= _check("POST /search 跨 user_id 隔离(空)",
                 st == 200 and isinstance(other, list) and len(other) == 0,
                 f"status={st} n={len(other) if isinstance(other, list) else 'n/a'}")

    print("\n" + ("=== ALL PASS ===" if ok else "=== SOME FAILED ==="))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
