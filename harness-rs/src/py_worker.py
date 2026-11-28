"""Minimal Python AST sandbox worker invoked by the Rust harness orchestrator.

Protocol (JSON over stdin -> JSON over stdout):
  - {"op": "run", "code": str, "tests": str, "timeout_s": float}
      -> {"passed": bool, "error": str}
  - {"op": "self_check", "tasks": [{"id": str, "tests": str}, ...]}
      -> {"passed": bool, "error": str, "count": int}
"""
import ast
import contextlib
import io
import json
import re
import signal
import sys
import traceback


def instrument_asserts(tests: str) -> str:
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


def annotate_code_lines(code: str, err: str) -> str:
    """Attach exact offending source lines from `<solution>` to help pinpoint bugs."""
    code_lines = code.splitlines()
    hits = sorted({
        int(m)
        for m in re.findall(r'File "<solution>", line (\d+)', err)
        if 1 <= int(m) <= len(code_lines)
    })
    if not hits:
        return err
    ctx_indices = sorted({
        i for h in hits for i in range(max(1, h - 1), min(len(code_lines), h + 1) + 1)
    })
    snippet = "\n".join(
        f"  L{i:02d}{' >>> ' if i in hits else '     '}{code_lines[i - 1]}"
        for i in ctx_indices
    )
    return f"{err}\nOffending code lines:\n{snippet}"


def run_task(code: str, tests: str, timeout_s: float) -> dict:
    alarm_s = max(1, int(timeout_s) - 1)

    def on_alarm(_signum, _frame):
        raise TimeoutError(
            f"Infinite loop (>{alarm_s}s): loop pointer/condition never terminates"
        )

    instrumented = instrument_asserts(tests)
    ns = {"__name__": "__main__"}
    buf = io.StringIO()

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(alarm_s)

    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code_obj = compile(code, "<solution>", "exec")
            exec(code_obj, ns)
            test_obj = compile(instrumented, "<tests>", "exec")
            exec(test_obj, ns)
        return {"passed": True, "error": ""}
    except BaseException:
        tb = traceback.format_exc().strip()[-900:]
        return {"passed": False, "error": annotate_code_lines(code, tb)}
    finally:
        if has_alarm:
            signal.alarm(0)


def self_check(tasks: list) -> dict:
    for t in tasks:
        tid = t.get("id", "<unknown>")
        tests = t.get("tests", "")
        try:
            compile(tests, tid, "exec")
            compile(instrument_asserts(tests), f"{tid}_instrumented", "exec")
        except Exception as exc:
            return {"passed": False, "error": f"Task {tid}: {exc}", "count": len(tasks)}
    return {"passed": True, "error": "", "count": len(tasks)}


def main() -> None:
    real_out = sys.stdout
    req = json.loads(sys.stdin.read())
    op = req.get("op", "run")
    if op == "self_check":
        res = self_check(req.get("tasks", []))
    else:
        res = run_task(
            req.get("code", ""),
            req.get("tests", ""),
            float(req.get("timeout_s", 5.0)),
        )
    real_out.write(json.dumps(res))
    real_out.flush()


if __name__ == "__main__":
    main()
