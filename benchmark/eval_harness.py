#!/usr/bin/env python3
"""Minimal coding benchmark harness (zero external dependencies).

Supports:
  1. Evaluating a JSONL file of model predictions:
     python3 eval_harness.py --predictions preds.jsonl
  2. Evaluating a live OpenAI-compatible endpoint (e.g. local llama-server):
     python3 eval_harness.py --api-url http://127.0.0.1:8080/v1/chat/completions --model local
  3. Self-check mode (verifies all tasks & test snippets parse cleanly):
     python3 eval_harness.py --self-check
"""
import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

TASKS_PATH = Path(__file__).parent / "tasks.json"


def extract_python_code(text: str) -> str:
    # Strip <think>...</think> blocks if present
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", cleaned)
    if blocks:
        return blocks[-1].strip()
    unclosed = re.search(r"```(?:python|py)?\s*\n([\s\S]*)$", cleaned)
    if unclosed:
        return unclosed.group(1).strip()
    return cleaned


def _instrument_asserts(tests: str) -> str:
    """Rewrite bare `assert a <op> b` statements to show actual vs expected + state on failure."""
    tree = ast.parse(tests)
    out_lines = [
        "def __dbg_state(ns):",
        "    return {k: v for k, v in ns.items() if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict, tuple, set))}",
    ]
    for stmt in tree.body:
        src = ast.get_source_segment(tests, stmt) or ast.unparse(stmt)
        if (
            isinstance(stmt, ast.Assert)
            and stmt.msg is None
            and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1
            and len(stmt.test.comparators) == 1
        ):
            lhs_src = ast.unparse(stmt.test.left)
            rhs_src = ast.unparse(stmt.test.comparators[0])
            op_node = ast.Compare(
                left=ast.Name(id="__lhs", ctx=ast.Load()),
                ops=stmt.test.ops,
                comparators=[ast.Name(id="__rhs", ctx=ast.Load())],
            )
            cond_src = ast.unparse(op_node)
            out_lines.append(f"__lhs = ({lhs_src})")
            out_lines.append(f"__rhs = ({rhs_src})")
            out_lines.append(
                f"assert {cond_src}, ({src!r} + f' | actual={{__lhs!r}}, expected={{__rhs!r}} | locals={{__dbg_state(locals())!r}}')"
            )
        else:
            out_lines.append(src)
    return "\n".join(out_lines)


def _annotate_code_lines(code: str, tmp_name: str, err: str) -> str:
    """Attach exact offending source lines from the submitted code to help 9B pinpoint bugs."""
    code_lines = code.splitlines()
    hits = sorted({
        int(m)
        for m in re.findall(re.escape(tmp_name) + r'", line (\d+)', err)
        if 1 <= int(m) <= len(code_lines)
    })
    if not hits:
        return err
    ctx_indices = sorted({i for h in hits for i in range(max(1, h - 1), min(len(code_lines), h + 1) + 1)})
    snippet = "\n".join(
        f"  L{i:02d}{' >>> ' if i in hits else '     '}{code_lines[i - 1]}"
        for i in ctx_indices
    )
    return f"{err}\nOffending code lines:\n{snippet}"


def run_task_tests(code: str, tests: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    instrumented = _instrument_asserts(tests)
    alarm_s = max(1, int(timeout_s) - 1)
    alarm_setup = (
        "import signal as __sig\n"
        "def __on_alarm(s, f):\n"
        f"    raise TimeoutError('Infinite loop (>{alarm_s}s): loop pointer/condition never terminates')\n"
        "__sig.signal(__sig.SIGALRM, __on_alarm)\n"
        f"__sig.alarm({alarm_s})\n"
    )
    script = f"{code}\n\n# --- UNIT TESTS ---\n{alarm_setup}{instrumented}\n__sig.alarm(0)\nprint('PASS_OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(script)
        tmp_path = Path(tf.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        ok = proc.returncode == 0 and "PASS_OK" in proc.stdout
        if ok:
            return True, ""
        err = (proc.stderr or proc.stdout).strip()[-900:]
        return False, _annotate_code_lines(code, str(tmp_path), err)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT ({timeout_s}s): process hung (likely infinite loop)"
    finally:
        tmp_path.unlink(missing_ok=True)


def query_openai_api(api_url: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Jawab dengan kode Python yang bersih di dalam blok ```python ... ```."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=TASKS_PATH)
    ap.add_argument("--predictions", type=Path, help="JSONL with {id, response}")
    ap.add_argument("--api-url", help="OpenAI-compatible /v1/chat/completions URL")
    ap.add_argument("--model", default="local")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, help="Optional JSON report path")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if args.self_check:
        for t in tasks:
            assert all(k in t for k in ("id", "category", "prompt", "tests")), f"Bad task: {t}"
            compile(t["tests"], t["id"], "exec")
            compile(_instrument_asserts(t["tests"]), f"{t['id']}_instrumented", "exec")
        print(f"SELF-CHECK OK: {len(tasks)} tasks validated.")
        return

    preds = {}
    if args.predictions:
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            preds[row["id"]] = row["response"]
    elif not args.api_url:
        ap.error("Provide --predictions, --api-url, or --self-check")

    results = []
    passed = 0
    for t in tasks:
        if args.api_url:
            raw = query_openai_api(args.api_url, args.model, t["prompt"], args.temperature, args.max_tokens)
        else:
            raw = preds.get(t["id"], "")
        code = extract_python_code(raw)
        ok, err = run_task_tests(code, t["tests"])
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {t['id']:30s} ({t['category']})" + (f" -> {err.splitlines()[-1]}" if err else ""))
        results.append({"id": t["id"], "category": t["category"], "passed": ok, "error": err})

    score = passed / len(tasks) * 100.0
    print(f"\nTOTAL: {passed}/{len(tasks)} ({score:.1f}% pass@1)")
    if args.out:
        args.out.write_text(json.dumps({"passed": passed, "total": len(tasks), "score_pct": score, "results": results}, indent=2))


if __name__ == "__main__":
    main()
