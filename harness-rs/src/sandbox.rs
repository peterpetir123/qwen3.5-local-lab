use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const WORKER_SCRIPT: &str = include_str!("py_worker.py");

#[derive(Debug, Clone, Deserialize)]
pub struct EvalOutcome {
    pub passed: bool,
    pub error: String,
    #[serde(default)]
    pub count: usize,
}

#[derive(Serialize)]
struct RunRequest<'a> {
    op: &'a str,
    code: &'a str,
    tests: &'a str,
    timeout_s: f64,
}

#[derive(Serialize)]
struct CheckTaskRef<'a> {
    id: &'a str,
    tests: &'a str,
}

#[derive(Serialize)]
struct SelfCheckRequest<'a> {
    op: &'a str,
    tasks: Vec<CheckTaskRef<'a>>,
}

/// Spawn an isolated `python3 -I -u` worker, feed JSON over stdin, and guard it
/// with a hard OS watchdog (`child.kill()`) so hung/malicious code can never stall the runner.
fn invoke_worker(
    python_bin: &str,
    payload: Vec<u8>,
    hard_timeout: Duration,
    label_timeout_s: f64,
) -> EvalOutcome {
    let mut child = match Command::new(python_bin)
        .args(["-I", "-u", "-c", WORKER_SCRIPT])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return EvalOutcome {
                passed: false,
                error: format!("Failed to spawn Python worker (`{python_bin}`): {e}"),
                count: 0,
            };
        }
    };

    // Write stdin on a thread so large inputs can never deadlock against stdout/stderr pipes.
    let mut stdin = child.stdin.take().expect("piped stdin");
    let writer = thread::spawn(move || {
        let _ = stdin.write_all(&payload);
        // Drop stdin here to send EOF to python worker
    });

    let mut stdout = child.stdout.take().expect("piped stdout");
    let stdout_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });

    let mut stderr = child.stderr.take().expect("piped stderr");
    let stderr_reader = thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });

    // OS-level watchdog loop (polling try_wait avoids external crates while staying responsive).
    // ponytail: 15ms poll interval is fine for multi-second LLM test runs; upgrade to waitpid/pidfd if running >10k micro-tests/sec.
    let deadline = Instant::now() + hard_timeout;
    let mut killed_by_watchdog = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break Some(st),
            Ok(None) => {
                if Instant::now() >= deadline {
                    killed_by_watchdog = true;
                    let _ = child.kill();
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(15));
            }
            Err(_) => {
                let _ = child.kill();
                break child.wait().ok();
            }
        }
    };

    let _ = writer.join();
    let out_bytes = stdout_reader.join().unwrap_or_default();
    let err_bytes = stderr_reader.join().unwrap_or_default();

    if killed_by_watchdog {
        return EvalOutcome {
            passed: false,
            error: format!(
                "TIMEOUT ({label_timeout_s:.1}s): process hung and killed by Rust OS watchdog (likely infinite loop)"
            ),
            count: 0,
        };
    }

    match serde_json::from_slice::<EvalOutcome>(&out_bytes) {
        Ok(outcome) => outcome,
        Err(_) => {
            let stderr_txt = String::from_utf8_lossy(&err_bytes);
            let stdout_txt = String::from_utf8_lossy(&out_bytes);
            let detail = if !stderr_txt.trim().is_empty() {
                stderr_txt.trim().to_string()
            } else {
                stdout_txt.trim().to_string()
            };
            EvalOutcome {
                passed: false,
                error: format!(
                    "SANDBOX CRASH (exit={:?}): {}",
                    status.and_then(|s| s.code()),
                    detail
                ),
                count: 0,
            }
        }
    }
}

pub fn run_task_tests(python_bin: &str, code: &str, tests: &str, timeout_s: f64) -> EvalOutcome {
    let req = RunRequest {
        op: "run",
        code,
        tests,
        timeout_s,
    };
    let payload = serde_json::to_vec(&req).expect("serialize RunRequest");
    // Give Python's SIGALRM (timeout_s - 1s) room to emit a line-numbered TimeoutError
    // before the Rust hard watchdog SIGKILLs the process.
    let hard_timeout = Duration::from_secs_f64(timeout_s.max(1.5) + 0.5);
    invoke_worker(python_bin, payload, hard_timeout, timeout_s)
}

pub fn self_check_tasks(python_bin: &str, tasks: &[(String, String)]) -> Result<usize, String> {
    let refs: Vec<CheckTaskRef<'_>> = tasks
        .iter()
        .map(|(id, tests)| CheckTaskRef {
            id: id.as_str(),
            tests: tests.as_str(),
        })
        .collect();
    let req = SelfCheckRequest {
        op: "self_check",
        tasks: refs,
    };
    let payload = serde_json::to_vec(&req).expect("serialize SelfCheckRequest");
    let res = invoke_worker(python_bin, payload, Duration::from_secs(15), 15.0);
    if res.passed {
        Ok(res.count)
    } else {
        Err(res.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_valid_solution_even_with_print() {
        let code = "def add(a, b):\n    print('noisy debug')\n    return a + b\n";
        let tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n";
        let res = run_task_tests("python3", code, tests, 3.0);
        assert!(res.passed, "expected pass, got error: {}", res.error);
    }

    #[test]
    fn instruments_assert_with_actual_expected_and_locals() {
        let code = "def double(x):\n    return x + 1\n";
        let tests = "val = 4\nassert double(val) == 8\n";
        let res = run_task_tests("python3", code, tests, 3.0);
        assert!(!res.passed);
        assert!(res.error.contains("actual=5"), "err={}", res.error);
        assert!(res.error.contains("expected=8"), "err={}", res.error);
        assert!(res.error.contains("'val': 4"), "err={}", res.error);
    }

    #[test]
    fn annotates_offending_solution_lines() {
        let code = "def boom(arr):\n    return arr[99]\n";
        let tests = "assert boom([1, 2]) == 1\n";
        let res = run_task_tests("python3", code, tests, 3.0);
        assert!(!res.passed);
        assert!(res.error.contains("IndexError"), "err={}", res.error);
        assert!(res.error.contains("L02 >>>"), "err={}", res.error);
    }

    #[test]
    fn catches_infinite_loop_via_sigalrm_with_line_number() {
        let code = "def spin():\n    while True:\n        pass\n";
        let tests = "spin()\n";
        let res = run_task_tests("python3", code, tests, 2.0);
        assert!(!res.passed);
        assert!(res.error.contains("TimeoutError"), "err={}", res.error);
        assert!(res.error.contains("Offending code lines:"), "err={}", res.error);
    }

    #[test]
    fn rust_watchdog_kills_uninterruptible_loop() {
        // Code deliberately swallows TimeoutError/BaseException to prove Rust's OS SIGKILL still wins.
        let code = "def stubborn():\n    while True:\n        try:\n            while True:\n                pass\n        except BaseException:\n            pass\n";
        let tests = "stubborn()\n";
        let t0 = Instant::now();
        let res = run_task_tests("python3", code, tests, 1.5);
        assert!(!res.passed);
        assert!(res.error.contains("TIMEOUT"), "err={}", res.error);
        assert!(t0.elapsed() < Duration::from_secs(4), "took too long: {:?}", t0.elapsed());
    }
}
