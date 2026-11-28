# `qwen-harness` Technical Reference

## 1. Architecture Overview

`qwen-harness` is a hybrid **Rust Core Orchestrator + Embedded Python AST Sandbox Worker**:

- **Rust binary (`harness-rs/target/release/qwen-harness`)**: CLI, OpenAI-compatible HTTP/HTTPS client (`ureq`), code extraction (`<think>` stripping + fenced block parsing), Extra Engine v3 self-repair loop, incremental `--resume` JSON reporting, and a hard OS watchdog (`child.kill()`).
- **Embedded Python worker (`src/py_worker.py`)**: Spawned via `python3 -I -u` per test run. Performs AST `assert` rewriting, `signal.SIGALRM` infinite-loop detection, and `<solution>` line annotation in memory (zero temp files).

Default local repo location:
- Source: `harness-rs/`
- Build release binary: `cd harness-rs && cargo build --release`
- Binary path: `harness-rs/target/release/qwen-harness`

---

## 2. `tasks.json` Schema

JSON array of task objects:

```json
[
  {
    "id": "unique_snake_case_id",
    "category": "parsing | algorithm | data_structure | debugging | graph | domain_custom",
    "prompt": "Clear specification of function/class signature, input/output types, constraints, and edge cases.",
    "tests": "assert fn(x) == expected\nassert fn([]) == []"
  }
]
```

### Crucial Rule for `tests` (AST Introspection)
`py_worker.py` automatically rewrites **bare comparison asserts without a custom message**:
- **DO write**: `assert solve(x) == expected` (the AST rewriter injects `actual=...`, `expected=...`, and `locals=...` automatically!).
- **AVOID**: `assert solve(x) == expected, "wrong"` (adding a custom message bypasses AST rewriting).
- **Use intermediate variables** for stateful tests (e.g. `ts = [0.0, 1.0]; assert allow(ts) is True; assert ts == [1.0]`) so `locals` automatically exposes mutated state on failure.

---

## 3. `exemplars.json` Schema (Optional)

Analogous design invariants injected in `--mode smart` when at least **2 keywords** match `task.id + task.prompt` (never leak the exact task solution):

```json
[
  {
    "id": "ex_unique_id",
    "categories": ["parsing"],
    "keywords": ["delimiter", "split", "version", "sort"],
    "problem": "Short analogous problem statement.",
    "reasoning": "Key engineering invariant to prevent common LLM bugs.",
    "code": "def analogous_example(s: str) -> list[str]:\n    ..."
  }
]
```

---

## 4. CLI Flags Reference

| Flag | Default | Purpose |
|---|---|---|
| `--tasks <PATH>` | `benchmark/tasks_30.json` | Tasks JSON file |
| `--exemplars <PATH>` | `data/teacher_exemplars.json` | Exemplars JSON (`""` or `none` to disable) |
| `--api-url <URL>` | `http://127.0.0.1:8081/v1/chat/completions` | OpenAI-compatible chat completions endpoint (set `OPENAI_API_KEY` env var if auth is required) |
| `--model <NAME>` | `Qwen3.5-9B` | Model ID sent in API payload |
| `--mode <smart\|plain>` | `smart` | `smart` = self-repair + exemplar + Fresh-Design Reset; `plain` = 1-shot baseline |
| `--max-iters <N>` | `3` | Max attempts per task in `smart` mode |
| `--timeout <SECS>` | `5.0` | Per-task sandbox timeout (`SIGALRM` + Rust OS `SIGKILL` watchdog) |
| `--temperature <FLOAT>` | `0.15` | Base temperature (escalates `+0.1` per retry up to `0.55`) |
| `--max-tokens <N>` | `768` | Max completion tokens |
| `--python <BIN>` | `python3` | Python executable for AST worker |
| `--start <N>` / `--limit <N>` | `0` / `0` | Slice tasks by index |
| `--ids <ID1,ID2>` | *(all)* | Run specific task IDs |
| `--predictions <JSONL>` | *(none)* | Offline eval of `{id, response}` JSONL |
| `--resume` | `false` | Skip task IDs already present in `--out` |
| `--out <PATH>` | *(none)* | Save incremental JSON report after each task |
| `--self-check` | `false` | Validate JSON schema + compile all `tests` and AST-rewritten tests |

---

## 5. Extending Beyond Python

To adapt `qwen-harness` to another target runtime (SQL, Node.js, Bash, Rust):
1. Replace `harness-rs/src/py_worker.py` with a worker that reads `{"op": "run"|"self_check", "code": ..., "tests": ..., "timeout_s": ...}` from `stdin` and prints `{"passed": bool, "error": str, "count": int}` to `stdout`.
2. Update the spawned binary/args in `harness-rs/src/sandbox.rs` (`invoke_worker`).
3. Customize prompts, code-fence extraction, and error hints in `harness-rs/src/solver.rs`.
4. Rebuild with `cd harness-rs && cargo build --release`.

---

## 6. OS Compatibility Notes

- **Linux / macOS / WSL**:
  - Binary: `./harness-rs/target/release/qwen-harness`
  - Python flag: `--python python3` (default)
  - Timeout: Dual-layer (`signal.SIGALRM` gives exact offending line numbers + Rust OS watchdog `child.kill()`).
  - Env var (if cloud API): `export OPENAI_API_KEY="..."`
- **Windows Native (PowerShell / CMD)**:
  - Binary: `.\harness-rs\target\release\qwen-harness.exe`
  - Python flag: pass `--python python` (or `py`) if `python3` alias is unavailable.
  - Timeout: `signal.SIGALRM` is skipped automatically on Windows; Rust OS watchdog (`child.kill()`) enforces hard process termination reliably.
  - Env var (PowerShell): `$env:OPENAI_API_KEY="..."`
