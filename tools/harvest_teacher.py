#!/usr/bin/env python3
"""Harvest verified Teacher Exemplars (reasoning + clean code) from local Qwen3.8-27B."""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
from eval_harness import extract_python_code, run_task_tests  # noqa: E402

TEACHER_TASKS = [
    {
        "id": "ex_parsing_composite_key",
        "categories": ["parsing"],
        "keywords": ["sort", "version", "semver", "parse", "prerelease", "format", "string", "split", "token"],
        "prompt": (
            "Write a compact Python function `sort_build_ids(ids: list[str]) -> list[str]` sorting strings of "
            "the form `<PREFIX>-<MAJOR>.<MINOR>[~<tag>]` (e.g. `'APP-2.9~alpha'`, `'APP-2.9'`, `'APP-10.2'`). "
            "Order by: alphabetic prefix, numeric major, numeric minor, and optional `~tag` ranks strictly BEFORE "
            "bare release (`~alpha` < `~beta` < no tag). Output 1 short rule line then one ```python ... ``` block."
        ),
        "tests": (
            "assert sort_build_ids(['APP-10.2', 'APP-2.9', 'APP-2.9~beta', 'APP-2.9~alpha', 'CORE-1.0']) "
            "== ['APP-2.9~alpha', 'APP-2.9~beta', 'APP-2.9', 'APP-10.2', 'CORE-1.0']\n"
            "assert sort_build_ids(['X-1.0', 'X-1.0~rc1']) == ['X-1.0~rc1', 'X-1.0']\n"
            "assert sort_build_ids([]) == []"
        ),
    },
    {
        "id": "ex_boundary_inplace_debug",
        "categories": ["debugging"],
        "keywords": ["in-place", "window", "limit", "remove", "bug", "perbaiki", "mutate", "default", "expired"],
        "prompt": (
            "Write a clean Python function `prune_stale(entries: list[tuple[str, float]], now: float, ttl: float, "
            "max_keep: int) -> int` that modifies `entries` in-place: remove all `(key, ts)` with `ts <= now - ttl`, "
            "then if `len(entries) > max_keep`, drop oldest items from the front in-place so at most `max_keep` remain; "
            "return `len(entries)`. Output a 2-bullet reasoning checklist followed by one ```python ... ``` block."
        ),
        "tests": (
            "e = [('a', 0.0), ('b', 1.0), ('c', 2.0)]\n"
            "assert prune_stale(e, 2.0, 2.0, 2) == 2 and e == [('b', 1.0), ('c', 2.0)]\n"
            "assert prune_stale(e, 2.5, 2.0, 1) == 1 and e == [('c', 2.0)]\n"
            "e2 = [('x', 1.0)]\n"
            "assert prune_stale(e2, 1.0, 1.0, 0) == 0 and e2 == []"
        ),
    },
    {
        "id": "ex_nested_structure_algo",
        "categories": ["algorithm", "data_structure", "graph", "dp"],
        "keywords": ["dict", "list", "nested", "recursive", "graph", "heap", "cache", "interval", "dp"],
        "prompt": (
            "Write a clean Python function `normalize_tree(node)` that recursively normalizes a nested structure of "
            "dicts and lists: strip whitespace from string leaves, keep ints/floats/bools/None unchanged, and "
            "preserve empty dicts `{}` and empty lists `[]` without dropping keys. Do not mutate the input. "
            "Output a 2-bullet reasoning checklist followed by one ```python ... ``` block."
        ),
        "tests": (
            "src = {'a': ' hi ', 'b': [' x ', {'c': {}}], 'd': []}\n"
            "out = normalize_tree(src)\n"
            "assert out == {'a': 'hi', 'b': ['x', {'c': {}}], 'd': []}\n"
            "assert src['a'] == ' hi '\n"
            "assert normalize_tree({}) == {}"
        ),
    },
]

SYS_PROMPT = (
    "You are an expert Python engineer (Qwen3.8 NEO-CODER). Be ultra-concise and exact. "
    "Write 1 short rule line naming the key parsing/boundary invariant, then one compact ```python ... ``` block."
)


def call_teacher(api_url: str, prompt: str, max_tokens: int) -> tuple[str, float, int, float]:
    body = {
        "model": "qwen38-27b",
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    chunks: list[str] = []
    n_tok = 0
    t_first = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            delta = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if t_first is None:
                    t_first = time.time()
                    print(f"  [first token at {t_first - t0:.1f}s]", flush=True)
                chunks.append(piece)
                n_tok += 1
                if n_tok % 15 == 0:
                    gen_dt = max(time.time() - t_first, 1e-6)
                    print(f"  [tok {n_tok:3d} | {n_tok / gen_dt:.3f} t/s | elapsed {time.time() - t0:.0f}s]", flush=True)
    dt = time.time() - t0
    gen_dt = max(dt - ((t_first - t0) if t_first else 0.0), 1e-6)
    tps = n_tok / gen_dt if n_tok else 0.0
    return "".join(chunks), dt, n_tok, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8082/v1/chat/completions")
    ap.add_argument("--only", help="Comma-separated teacher task IDs to harvest")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "teacher_exemplars.json")
    args = ap.parse_args()

    existing = {}
    if args.out.exists():
        for row in json.loads(args.out.read_text(encoding="utf-8")):
            existing[row["id"]] = row

    tasks = TEACHER_TASKS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        tasks = [t for t in tasks if t["id"] in wanted]

    for t in tasks:
        print(f"[HARVEST] {t['id']} ...", flush=True)
        reply, dt, n_tok, tps = call_teacher(args.api_url, t["prompt"], args.max_tokens)
        code = extract_python_code(reply)
        ok, err = run_task_tests(code, t["tests"])
        status = "VERIFIED" if ok else f"FAILED ({err.splitlines()[-1] if err else 'err'})"
        print(f"  -> {status} | {n_tok} tok ({tps:.3f} t/s) | {dt:.1f}s", flush=True)
        if ok:
            existing[t["id"]] = {
                "id": t["id"],
                "source_model": "Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ2_M(1).gguf",
                "categories": t["categories"],
                "keywords": t["keywords"],
                "prompt": t["prompt"],
                "teacher_response": reply.strip(),
                "verified": True,
                "stats": {"tokens": n_tok, "tps": round(tps, 4), "time_s": round(dt, 1)},
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(list(existing.values()), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
