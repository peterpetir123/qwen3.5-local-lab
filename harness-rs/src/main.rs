mod sandbox;
mod solver;

use sandbox::{run_task_tests, self_check_tasks};
use serde::Deserialize;
use solver::{
    AttemptLog, Exemplar, RunReport, SolverConfig, Task, TaskResult, build_http_agent,
    extract_python_code, solve_task,
};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Deserialize)]
struct PredictionRow {
    id: String,
    response: String,
}

struct CliArgs {
    api_url: String,
    model: String,
    tasks_path: PathBuf,
    exemplars_path: String,
    predictions_path: Option<PathBuf>,
    python_bin: String,
    mode: String,
    max_iters: usize,
    test_timeout_s: f64,
    start: usize,
    limit: usize,
    ids: Option<String>,
    resume: bool,
    self_check: bool,
    temperature: f64,
    max_tokens: u32,
    out_path: Option<PathBuf>,
}

fn print_usage() {
    eprintln!(
        "qwen-harness — Hybrid Rust Core Orchestrator + Python AST Sandbox Worker\n\
         \n\
         USAGE:\n\
           qwen-harness [OPTIONS]\n\
         \n\
         OPTIONS:\n\
           --tasks <PATH>          Tasks JSON file (default: benchmark/tasks_30.json)\n\
           --exemplars <PATH>      Exemplars JSON file, empty to disable (default: data/teacher_exemplars.json)\n\
           --api-url <URL>         OpenAI-compatible endpoint (default: http://127.0.0.1:8081/v1/chat/completions)\n\
           --model <NAME>          Model ID sent in API payload (default: Qwen3.5-9B)\n\
           --mode <smart|plain>    Solver mode (default: smart)\n\
           --max-iters <N>         Max self-repair attempts in smart mode (default: 3)\n\
           --timeout <SECS>        Per-task sandbox timeout in seconds (default: 5.0)\n\
           --temperature <FLOAT>   Base sampling temperature (default: 0.15)\n\
           --max-tokens <N>        Max completion tokens (default: 768)\n\
           --python <BIN>          Python executable for AST worker (default: python3)\n\
           --start <N>             Start task index, 0-based (default: 0)\n\
           --limit <N>             Run only N tasks from --start, 0 = all (default: 0)\n\
           --ids <ID1,ID2,...>     Comma-separated task IDs (overrides --start/--limit)\n\
           --predictions <JSONL>   Evaluate offline JSONL predictions ({{id, response}})\n\
           --resume                Skip task IDs already saved in --out\n\
           --out <PATH>            Save incremental JSON report to path\n\
           --self-check            Validate all tasks and AST assert instrumentation\n\
           -h, --help              Show this help message"
    );
}

fn parse_args() -> Result<CliArgs, String> {
    let mut cfg = CliArgs {
        api_url: "http://127.0.0.1:8081/v1/chat/completions".to_string(),
        model: "Qwen3.5-9B".to_string(),
        tasks_path: PathBuf::from("benchmark/tasks_30.json"),
        exemplars_path: "data/teacher_exemplars.json".to_string(),
        predictions_path: None,
        python_bin: "python3".to_string(),
        mode: "smart".to_string(),
        max_iters: 3,
        test_timeout_s: 5.0,
        start: 0,
        limit: 0,
        ids: None,
        resume: false,
        self_check: false,
        temperature: 0.15,
        max_tokens: 768,
        out_path: None,
    };

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        let next_val = |flag: &str, it: &mut std::iter::Skip<env::Args>| -> Result<String, String> {
            it.next().ok_or_else(|| format!("Missing value for {flag}"))
        };
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--api-url" => cfg.api_url = next_val(&arg, &mut it)?,
            "--model" => cfg.model = next_val(&arg, &mut it)?,
            "--tasks" => cfg.tasks_path = PathBuf::from(next_val(&arg, &mut it)?),
            "--exemplars" => cfg.exemplars_path = next_val(&arg, &mut it)?,
            "--predictions" => cfg.predictions_path = Some(PathBuf::from(next_val(&arg, &mut it)?)),
            "--python" => cfg.python_bin = next_val(&arg, &mut it)?,
            "--mode" => {
                let m = next_val(&arg, &mut it)?;
                if m != "smart" && m != "plain" {
                    return Err(format!("Invalid --mode `{m}` (expected `smart` or `plain`)"));
                }
                cfg.mode = m;
            }
            "--max-iters" => {
                cfg.max_iters = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid integer for --max-iters")?;
            }
            "--timeout" => {
                cfg.test_timeout_s = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid float for --timeout")?;
            }
            "--temperature" => {
                cfg.temperature = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid float for --temperature")?;
            }
            "--max-tokens" => {
                cfg.max_tokens = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid integer for --max-tokens")?;
            }
            "--start" => {
                cfg.start = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid integer for --start")?;
            }
            "--limit" => {
                cfg.limit = next_val(&arg, &mut it)?
                    .parse()
                    .map_err(|_| "Invalid integer for --limit")?;
            }
            "--ids" => cfg.ids = Some(next_val(&arg, &mut it)?),
            "--resume" => cfg.resume = true,
            "--self-check" => cfg.self_check = true,
            "--out" => cfg.out_path = Some(PathBuf::from(next_val(&arg, &mut it)?)),
            unknown => return Err(format!("Unknown argument: {unknown} (use --help)")),
        }
    }
    Ok(cfg)
}

fn save_report(path: &PathBuf, mode: &str, results: &[TaskResult]) -> Result<(), String> {
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let score_pct = if total > 0 {
        (passed as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    let report = RunReport {
        mode: mode.to_string(),
        passed,
        total,
        score_pct,
        results: results.to_vec(),
    };
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    fs::write(path, text).map_err(|e| format!("Failed writing {}: {e}", path.display()))
}

fn run() -> Result<(), String> {
    let cli = parse_args()?;

    let tasks_raw = fs::read_to_string(&cli.tasks_path)
        .map_err(|e| format!("Failed to read tasks file {}: {e}", cli.tasks_path.display()))?;
    let all_tasks: Vec<Task> = serde_json::from_str(&tasks_raw)
        .map_err(|e| format!("Invalid tasks JSON {}: {e}", cli.tasks_path.display()))?;

    if cli.self_check {
        let pairs: Vec<(String, String)> = all_tasks
            .iter()
            .map(|t| (t.id.clone(), t.tests.clone()))
            .collect();
        let count = self_check_tasks(&cli.python_bin, &pairs)?;
        println!("SELF-CHECK OK: {count} tasks validated via Rust -> Python AST worker.");
        return Ok(());
    }

    let tasks: Vec<Task> = if let Some(ids_csv) = &cli.ids {
        let wanted: HashSet<&str> = ids_csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        all_tasks
            .into_iter()
            .filter(|t| wanted.contains(t.id.as_str()))
            .collect()
    } else {
        let sliced = all_tasks.into_iter().skip(cli.start);
        if cli.limit > 0 {
            sliced.take(cli.limit).collect()
        } else {
            sliced.collect()
        }
    };

    // Offline predictions evaluation mode
    if let Some(pred_path) = &cli.predictions_path {
        let raw = fs::read_to_string(pred_path)
            .map_err(|e| format!("Failed to read predictions {}: {e}", pred_path.display()))?;
        let mut preds: HashMap<String, String> = HashMap::new();
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: PredictionRow = serde_json::from_str(trimmed)
                .map_err(|e| format!("Invalid JSONL line in {}: {e}", pred_path.display()))?;
            preds.insert(row.id, row.response);
        }

        let mut results = Vec::with_capacity(tasks.len());
        let mut passed = 0usize;
        for t in &tasks {
            let raw_resp = preds.get(&t.id).map(String::as_str).unwrap_or("");
            let code = extract_python_code(raw_resp);
            let ev = run_task_tests(&cli.python_bin, &code, &t.tests, cli.test_timeout_s);
            if ev.passed {
                passed += 1;
            }
            let st = if ev.passed { "PASS" } else { "FAIL" };
            let tail = ev
                .error
                .lines()
                .last()
                .map(|l| format!(" -> {l}"))
                .unwrap_or_default();
            println!("[{st}] {:30} ({}){tail}", t.id, t.category);
            results.push(TaskResult {
                id: t.id.clone(),
                passed: ev.passed,
                attempts: 1,
                time_s: 0.0,
                tokens: 0,
                tps: 0.0,
                exemplar: None,
                attempts_log: vec![AttemptLog {
                    attempt: 1,
                    passed: ev.passed,
                    tokens: 0,
                    error: ev.error.lines().last().unwrap_or("").to_string(),
                }],
                final_code: code,
                error: ev.error,
            });
        }
        let total = results.len();
        let score = if total > 0 {
            passed as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!("\nOFFLINE TOTAL: {passed}/{total} ({score:.1}% pass@1)");
        if let Some(out) = &cli.out_path {
            save_report(out, "offline", &results)?;
        }
        return Ok(());
    }

    let exemplars: Vec<Exemplar> =
        if !cli.exemplars_path.is_empty() && cli.exemplars_path != "none" {
            let p = PathBuf::from(&cli.exemplars_path);
            if p.exists() {
                let raw = fs::read_to_string(&p)
                    .map_err(|e| format!("Failed to read exemplars {}: {e}", p.display()))?;
                serde_json::from_str(&raw)
                    .map_err(|e| format!("Invalid exemplars JSON {}: {e}", p.display()))?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

    let mut results: Vec<TaskResult> = Vec::new();
    let mut done_ids: HashSet<String> = HashSet::new();
    if cli.resume {
        if let Some(out) = &cli.out_path {
            if out.exists() {
                let raw = fs::read_to_string(out)
                    .map_err(|e| format!("Failed reading resume file {}: {e}", out.display()))?;
                let prev: RunReport = serde_json::from_str(&raw)
                    .map_err(|e| format!("Invalid resume JSON {}: {e}", out.display()))?;
                for r in prev.results {
                    done_ids.insert(r.id.clone());
                    results.push(r);
                }
            }
        }
    }

    let solver_cfg = SolverConfig {
        api_url: cli.api_url,
        model: cli.model,
        api_key: env::var("OPENAI_API_KEY").ok(),
        python_bin: cli.python_bin,
        mode: cli.mode.clone(),
        max_iters: cli.max_iters,
        temperature: cli.temperature,
        max_tokens: cli.max_tokens,
        test_timeout_s: cli.test_timeout_s,
    };

    let agent = build_http_agent();
    let mut passed = results.iter().filter(|r| r.passed).count();

    for t in &tasks {
        if done_ids.contains(&t.id) {
            continue;
        }
        let r = solve_task(&agent, &solver_cfg, t, &exemplars);
        if r.passed {
            passed += 1;
        }
        let st = if r.passed { "PASS" } else { "FAIL" };
        let ex_tag = r
            .exemplar
            .as_ref()
            .map(|e| format!(" ex={e}"))
            .unwrap_or_default();
        let tail = if !r.error.is_empty() {
            r.attempts_log
                .last()
                .map(|a| format!(" -> {}", a.error))
                .unwrap_or_default()
        } else {
            String::new()
        };
        println!(
            "[{st}] {:30} | try={} | {:4} tok ({:.2} t/s) | {:5.1}s{ex_tag}{tail}",
            r.id, r.attempts, r.tokens, r.tps, r.time_s
        );
        results.push(r);
        if let Some(out) = &cli.out_path {
            save_report(out, &cli.mode, &results)?;
        }
    }

    let total = results.len();
    let score = if total > 0 {
        passed as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "\nMODE={} TOTAL: {passed}/{total} ({score:.1}%)",
        cli.mode.to_uppercase()
    );
    if let Some(out) = &cli.out_path {
        save_report(out, &cli.mode, &results)?;
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
