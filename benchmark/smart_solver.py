#!/usr/bin/env python3
"""100% Local Thinking + Self-Correcting Verifier Engine for llama-server (CPU-only).

Compares:
  --mode plain : 1-shot raw generation (baseline)
  --mode smart : Qwen3.8-style xhigh reasoning prompt + automated Python execution
                 and traceback self-correction loop (up to --max-iters attempts)
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from eval_harness import TASKS_PATH, extract_python_code, run_task_tests

DEFAULT_EXEMPLARS_PATH = Path(__file__).resolve().parent.parent / "data" / "teacher_exemplars.json"

XHIGH_SYSTEM = (
    "You are a precision Python engineer. Prefer concise, idiomatic Python stdlib (`collections.deque`, "
    "`bisect`, `heapq`, `set`) over error-prone manual pointer arithmetic whenever allowed by the prompt. "
    "First write at most 2 concise bullet points checking critical edge cases (boundaries `<` vs `<=`, "
    "empty inputs, loop termination, state updates). Then output the complete, clean Python implementation "
    "inside a single ```python ... ``` code block."
)

PLAIN_SYSTEM = "Jawab dengan kode Python di dalam blok ```python ... ```."


def _diagnostic_hint(err: str) -> str:
    """Return a concise, general runtime hint derived solely from the error class."""
    if "TimeoutError" in err or "TIMEOUT" in err:
        return "Hint: Infinite loop detected — use `for` over indices/items or verify every `while` branch advances its index."
    if "IndexError" in err:
        return "Hint: Check empty input and `i < len(...)` bounds before indexing."
    if "KeyError" in err:
        return "Hint: Use `.get()` or initialize missing dictionary keys safely."
    if "ValueError" in err and "invalid literal for int()" in err:
        return "Hint: Split/strip non-numeric delimiters or accumulate digits before calling `int()`."
    if "AssertionError" in err and "actual=" in err and "expected=" in err:
        return "Hint: Compare `actual` vs `expected` carefully — check for premature `return` inside a loop, off-by-one boundaries, or missing final flush after the loop."
    return ""


def select_exemplar(task: dict, exemplars: list[dict]) -> dict | None:
    """Pick the best matching analogous exemplar only when semantic overlap is strong."""
    if not exemplars:
        return None
    hay = f"{task.get('id', '')} {task.get('prompt', '')}".lower()
    cat = task.get("category", "")
    best, best_score = None, 0
    for ex in exemplars:
        kw_hits = sum(1 for kw in ex.get("keywords", []) if kw.lower() in hay)
        if kw_hits < 2:
            continue
        score = kw_hits * 2 + (1 if cat in ex.get("categories", []) else 0)
        if score > best_score:
            best, best_score = ex, score
    return best


def format_exemplar_hint(ex: dict) -> str:
    return (
        "\n\nReference engineering pattern (analogous problem — reuse the design invariant):\n"
        f"- Problem: {ex['problem']}\n"
        f"- Key invariant: {ex['reasoning']}\n"
        f"```python\n{ex['code']}\n```"
    )


def chat_complete(
    api_url: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = True,
) -> tuple[str, float, int, float, str]:
    body = {
        "model": "local",
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.12,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    choice = data["choices"][0]
    msg = choice["message"]
    finish_reason = choice.get("finish_reason") or ""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    full = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
    timings = data.get("timings") or {}
    n_tok = int(timings.get("predicted_n") or 0)
    tps = float(timings.get("predicted_per_second") or 0.0)
    return full, dt, n_tok, tps, finish_reason


def solve_task(
    api_url: str,
    task: dict,
    mode: str,
    max_iters: int,
    temperature: float,
    max_tokens: int,
    exemplars: list[dict] | None = None,
) -> dict:
    sys_prompt = XHIGH_SYSTEM if mode == "smart" else PLAIN_SYSTEM
    chosen_ex = None
    if mode == "smart" and exemplars:
        chosen_ex = select_exemplar(task, exemplars)
        if chosen_ex:
            sys_prompt += format_exemplar_hint(chosen_ex)
    base_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task["prompt"]},
    ]
    messages = list(base_messages)
    attempts = 1 if mode == "plain" else max_iters
    enable_thinking = (mode == "plain")
    total_s = 0.0
    total_tok = 0
    last_err = ""
    last_code = ""
    prev_err_sig = ""
    attempts_log = []

    for attempt in range(1, attempts + 1):
        temp = min(0.55, temperature + 0.1 * (attempt - 1))
        reply, dt, n_tok, tps, finish_reason = chat_complete(
            api_url, messages, temp, max_tokens, enable_thinking=enable_thinking
        )
        total_s += dt
        total_tok += n_tok
        code = extract_python_code(reply)
        ok, err = run_task_tests(code, task["tests"])
        last_err, last_code = err, code
        err_head = next((ln.strip() for ln in err.splitlines() if "Error" in ln or "TIMEOUT" in ln), err.splitlines()[-1].strip() if err else "")
        attempts_log.append({"attempt": attempt, "passed": ok, "tokens": n_tok, "error": err_head})
        if ok:
            return {
                "id": task["id"], "passed": True, "attempts": attempt,
                "time_s": round(total_s, 1), "tokens": total_tok, "tps": round(tps, 2),
                "exemplar": chosen_ex["id"] if chosen_ex else None,
                "attempts_log": attempts_log, "final_code": code, "error": "",
            }
        if attempt < attempts:
            hint = _diagnostic_hint(err)
            hint_line = f"\n{hint}" if hint else ""
            is_fresh_reset = (
                finish_reason == "length"
                or (err_head and err_head == prev_err_sig)
                or attempt in (2, attempts - 1)
            )
            if finish_reason == "length":
                messages = list(base_messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Percobaan sebelumnya TERPOTONG karena melewati batas token.\n"
                            "Tulis ulang implementasi Python yang lebih ringkas, padat, dan lengkap "
                            "di dalam satu blok ```python ... ``` tanpa penjelasan panjang."
                        ),
                    }
                ]
            elif is_fresh_reset:
                # Fresh-Design Strategy Switch: drop broken code so 9B switches to a cleaner paradigm
                prev_notes = "; ".join(a["error"] for a in attempts_log[-2:] if a["error"])
                messages = list(base_messages) + [
                    {
                        "role": "user",
                        "content": (
                            f"Pendekatan sebelumnya gagal ({prev_notes}). Detail terakhir:\n```\n{err}\n```{hint_line}\n"
                            "JANGAN salin kode rusak sebelumnya. Ganti dengan pendekatan/algoritma yang lebih "
                            "sederhana dan bersih dari awal (utamakan stdlib seperti `deque`/`bisect`/`set` jika "
                            "diperbolehkan soal) di dalam satu blok ```python ... ```."
                        ),
                    }
                ]
            else:
                messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Kode di atas GAGAL saat diuji:\n```\n{err}\n```{hint_line}\n"
                            "Periksa teliti tanda kurung, update variabel/state, batas indeks, dan kondisi berhenti. "
                            "Sebutkan akar bug dalam 1 kalimat singkat, lalu tulis ulang kode Python lengkap yang "
                            "sudah diperbaiki di dalam satu blok ```python ... ```."
                        ),
                    }
                )
            prev_err_sig = err_head

    return {
        "id": task["id"], "passed": False, "attempts": attempts,
        "time_s": round(total_s, 1), "tokens": total_tok, "tps": round(tps, 2),
        "exemplar": chosen_ex["id"] if chosen_ex else None,
        "attempts_log": attempts_log, "final_code": last_code, "error": last_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://127.0.0.1:8081/v1/chat/completions")
    ap.add_argument("--tasks", type=Path, default=TASKS_PATH)
    ap.add_argument("--mode", choices=["plain", "smart"], default="smart")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="Start task index (0-based)")
    ap.add_argument("--limit", type=int, default=0, help="Run only N tasks from --start (0 = all remaining)")
    ap.add_argument("--ids", help="Comma-separated task IDs to run (overrides --start/--limit)")
    ap.add_argument("--exemplars", default=str(DEFAULT_EXEMPLARS_PATH), help="Exemplar JSON path (empty to disable)")
    ap.add_argument("--resume", action="store_true", help="Skip task IDs already present in --out")
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    exemplars = []
    if args.exemplars:
        ex_path = Path(args.exemplars)
        if ex_path.exists():
            exemplars = json.loads(ex_path.read_text(encoding="utf-8"))

    all_tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        tasks = [t for t in all_tasks if t["id"] in wanted]
    else:
        tasks = all_tasks[args.start :]
        if args.limit > 0:
            tasks = tasks[: args.limit]

    results = []
    done_ids = set()
    if args.resume and args.out and args.out.exists():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        results = prev.get("results") or []
        done_ids = {r["id"] for r in results}
    passed = sum(int(r["passed"]) for r in results)

    for t in tasks:
        if t["id"] in done_ids:
            continue
        r = solve_task(
            args.api_url, t, args.mode, args.max_iters, args.temperature, args.max_tokens, exemplars=exemplars
        )
        passed += int(r["passed"])
        st = "PASS" if r["passed"] else "FAIL"
        tail = f" -> {r['attempts_log'][-1]['error']}" if r["error"] else ""
        ex_tag = f" ex={r['exemplar']}" if r.get("exemplar") else ""
        print(f"[{st}] {r['id']:30s} | try={r['attempts']} | {r['tokens']:4d} tok ({r['tps']:.2f} t/s) | {r['time_s']:5.1f}s{ex_tag}{tail}", flush=True)
        results.append(r)
        if args.out:
            cur_score = passed / len(results) * 100.0
            args.out.write_text(
                json.dumps({"mode": args.mode, "passed": passed, "total": len(results), "score_pct": cur_score, "results": results}, indent=2)
            )

    total = len(results)
    score = (passed / total * 100.0) if total else 0.0
    print(f"\nMODE={args.mode.upper()} TOTAL: {passed}/{total} ({score:.1f}%)")
    if args.out:
        args.out.write_text(json.dumps({"mode": args.mode, "passed": passed, "total": total, "score_pct": score, "results": results}, indent=2))


if __name__ == "__main__":
    main()
