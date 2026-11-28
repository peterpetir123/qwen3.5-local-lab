use crate::sandbox::run_task_tests;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

pub const XHIGH_SYSTEM: &str = "You are a precision Python engineer. Prefer concise, idiomatic Python stdlib (`collections.deque`, \
`bisect`, `heapq`, `set`) over error-prone manual pointer arithmetic whenever allowed by the prompt. \
First write at most 2 concise bullet points checking critical edge cases (boundaries `<` vs `<=`, \
empty inputs, loop termination, state updates). Then output the complete, clean Python implementation \
inside a single ```python ... ``` code block.";

pub const PLAIN_SYSTEM: &str = "Jawab dengan kode Python di dalam blok ```python ... ```.";

#[derive(Debug, Clone, Deserialize)]
pub struct Task {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub tests: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Exemplar {
    pub id: String,
    #[serde(default)]
    pub categories: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    pub problem: String,
    pub reasoning: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptLog {
    pub attempt: usize,
    pub passed: bool,
    pub tokens: u64,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub id: String,
    pub passed: bool,
    pub attempts: usize,
    pub time_s: f64,
    pub tokens: u64,
    pub tps: f64,
    pub exemplar: Option<String>,
    pub attempts_log: Vec<AttemptLog>,
    pub final_code: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub mode: String,
    pub passed: usize,
    pub total: usize,
    pub score_pct: f64,
    pub results: Vec<TaskResult>,
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub api_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub python_bin: String,
    pub mode: String,
    pub max_iters: usize,
    pub temperature: f64,
    pub max_tokens: u32,
    pub test_timeout_s: f64,
}

/// Strip `<think>...</think>` blocks and extract the last fenced Python code block.
pub fn extract_python_code(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        cleaned.push_str(&rest[..start]);
        let after_open = &rest[start + "<think>".len()..];
        if let Some(end) = after_open.find("</think>") {
            rest = &after_open[end + "</think>".len()..];
        } else {
            cleaned.push_str(&rest[start..]);
            rest = "";
            break;
        }
    }
    cleaned.push_str(rest);
    let cleaned = cleaned.trim();

    let mut closed_blocks: Vec<&str> = Vec::new();
    let mut unclosed_block: Option<&str> = None;
    let mut cursor = 0;

    while let Some(rel) = cleaned[cursor..].find("```") {
        let fence_start = cursor + rel + 3;
        let after_ticks = &cleaned[fence_start..];
        let Some(nl_pos) = after_ticks.find('\n') else {
            break;
        };
        let lang = after_ticks[..nl_pos].trim();
        let body_start = fence_start + nl_pos + 1;
        let is_py = lang.is_empty() || lang.eq_ignore_ascii_case("python") || lang.eq_ignore_ascii_case("py");

        if let Some(close_rel) = cleaned[body_start..].find("```") {
            let body = &cleaned[body_start..body_start + close_rel];
            if is_py {
                closed_blocks.push(body.trim());
            }
            cursor = body_start + close_rel + 3;
        } else {
            if is_py {
                unclosed_block = Some(cleaned[body_start..].trim());
            }
            break;
        }
    }

    if let Some(last) = closed_blocks.last() {
        (*last).to_string()
    } else if let Some(unclosed) = unclosed_block {
        unclosed.to_string()
    } else {
        cleaned.to_string()
    }
}

pub fn diagnostic_hint(err: &str) -> &'static str {
    if err.contains("TimeoutError") || err.contains("TIMEOUT") {
        return "Hint: Infinite loop detected — use `for` over indices/items or verify every `while` branch advances its index.";
    }
    if err.contains("IndexError") {
        return "Hint: Check empty input and `i < len(...)` bounds before indexing.";
    }
    if err.contains("KeyError") {
        return "Hint: Use `.get()` or initialize missing dictionary keys safely.";
    }
    if err.contains("ValueError") && err.contains("invalid literal for int()") {
        return "Hint: Split/strip non-numeric delimiters or accumulate digits before calling `int()`.";
    }
    if err.contains("AssertionError") && err.contains("actual=") && err.contains("expected=") {
        return "Hint: Compare `actual` vs `expected` carefully — check for premature `return` inside a loop, off-by-one boundaries, or missing final flush after the loop.";
    }
    ""
}

pub fn select_exemplar<'a>(task: &Task, exemplars: &'a [Exemplar]) -> Option<&'a Exemplar> {
    if exemplars.is_empty() {
        return None;
    }
    let hay = format!("{} {}", task.id, task.prompt).to_lowercase();
    let mut best: Option<&'a Exemplar> = None;
    let mut best_score = 0usize;

    for ex in exemplars {
        let kw_hits = ex
            .keywords
            .iter()
            .filter(|kw| hay.contains(&kw.to_lowercase()))
            .count();
        if kw_hits < 2 {
            continue;
        }
        let cat_bonus = usize::from(ex.categories.iter().any(|c| c == &task.category));
        let score = kw_hits * 2 + cat_bonus;
        if score > best_score {
            best = Some(ex);
            best_score = score;
        }
    }
    best
}

fn format_exemplar_hint(ex: &Exemplar) -> String {
    format!(
        "\n\nReference engineering pattern (analogous problem — reuse the design invariant):\n\
         - Problem: {}\n\
         - Key invariant: {}\n\
         ```python\n{}\n```",
        ex.problem, ex.reasoning, ex.code
    )
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    timings: Option<Timings>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResp,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessageResp {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct Timings {
    #[serde(default)]
    predicted_n: Option<u64>,
    #[serde(default)]
    predicted_per_second: Option<f64>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    completion_tokens: Option<u64>,
}

struct CompletionOut {
    full_text: String,
    elapsed_s: f64,
    tokens: u64,
    tps: f64,
    finish_reason: String,
}

fn chat_complete(
    agent: &ureq::Agent,
    cfg: &SolverConfig,
    messages: &[Message],
    temperature: f64,
    enable_thinking: bool,
) -> Result<CompletionOut, String> {
    let mut body = serde_json::json!({
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.12,
        "max_tokens": cfg.max_tokens,
    });
    if !enable_thinking {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    }

    let mut req = agent
        .post(&cfg.api_url)
        .set("Content-Type", "application/json");
    if let Some(key) = &cfg.api_key {
        if !key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
    }

    let payload = serde_json::to_string(&body).map_err(|e| format!("JSON serialize error: {e}"))?;
    let t0 = Instant::now();
    let resp = req
        .send_string(&payload)
        .map_err(|e| format!("HTTP request to {} failed: {e}", cfg.api_url))?;
    let data: ChatResponse = serde_json::from_reader(resp.into_reader())
        .map_err(|e| format!("Invalid JSON response from LLM endpoint: {e}"))?;
    let dt = t0.elapsed().as_secs_f64();

    let choice = data
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| "LLM returned empty `choices` array".to_string())?;
    let content = choice.message.content.unwrap_or_default();
    let reasoning = choice.message.reasoning_content.unwrap_or_default();
    let full_text = if !reasoning.is_empty() {
        format!("<think>\n{reasoning}\n</think>\n\n{content}")
    } else {
        content
    };
    let finish_reason = choice.finish_reason.unwrap_or_default();

    let tokens = data
        .timings
        .as_ref()
        .and_then(|t| t.predicted_n)
        .or_else(|| data.usage.as_ref().and_then(|u| u.completion_tokens))
        .unwrap_or(0);
    let tps = data
        .timings
        .as_ref()
        .and_then(|t| t.predicted_per_second)
        .unwrap_or_else(|| if dt > 0.0 { tokens as f64 / dt } else { 0.0 });

    Ok(CompletionOut {
        full_text,
        elapsed_s: dt,
        tokens,
        tps,
        finish_reason,
    })
}

fn summarize_err_head(err: &str) -> String {
    if err.is_empty() {
        return String::new();
    }
    err.lines()
        .find(|ln| ln.contains("Error") || ln.contains("TIMEOUT") || ln.contains("SANDBOX CRASH"))
        .or_else(|| err.lines().last())
        .unwrap_or("")
        .trim()
        .to_string()
}

pub fn build_http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(900))
        .build()
}

pub fn solve_task(
    agent: &ureq::Agent,
    cfg: &SolverConfig,
    task: &Task,
    exemplars: &[Exemplar],
) -> TaskResult {
    let mut sys_prompt = if cfg.mode == "smart" {
        XHIGH_SYSTEM.to_string()
    } else {
        PLAIN_SYSTEM.to_string()
    };

    let chosen_ex = if cfg.mode == "smart" {
        select_exemplar(task, exemplars)
    } else {
        None
    };
    if let Some(ex) = chosen_ex {
        sys_prompt.push_str(&format_exemplar_hint(ex));
    }

    let base_messages = vec![
        Message {
            role: "system".to_string(),
            content: sys_prompt,
        },
        Message {
            role: "user".to_string(),
            content: task.prompt.clone(),
        },
    ];
    let mut messages = base_messages.clone();
    let attempts = if cfg.mode == "plain" {
        1
    } else {
        cfg.max_iters.max(1)
    };
    let enable_thinking = cfg.mode == "plain";

    let mut total_s = 0.0;
    let mut total_tok = 0u64;
    let mut last_tps = 0.0;
    let mut last_err = String::new();
    let mut last_code = String::new();
    let mut prev_err_sig = String::new();
    let mut attempts_log = Vec::with_capacity(attempts);

    for attempt in 1..=attempts {
        let temp = (cfg.temperature + 0.1 * (attempt - 1) as f64).min(0.55);
        let completion = match chat_complete(agent, cfg, &messages, temp, enable_thinking) {
            Ok(c) => c,
            Err(http_err) => {
                attempts_log.push(AttemptLog {
                    attempt,
                    passed: false,
                    tokens: 0,
                    error: http_err.clone(),
                });
                last_err = http_err;
                break;
            }
        };

        total_s += completion.elapsed_s;
        total_tok += completion.tokens;
        last_tps = completion.tps;

        let code = extract_python_code(&completion.full_text);
        let eval = run_task_tests(&cfg.python_bin, &code, &task.tests, cfg.test_timeout_s);
        let err_head = summarize_err_head(&eval.error);

        attempts_log.push(AttemptLog {
            attempt,
            passed: eval.passed,
            tokens: completion.tokens,
            error: err_head.clone(),
        });

        last_code = code.clone();
        last_err = eval.error.clone();

        if eval.passed {
            return TaskResult {
                id: task.id.clone(),
                passed: true,
                attempts: attempt,
                time_s: (total_s * 10.0).round() / 10.0,
                tokens: total_tok,
                tps: (last_tps * 100.0).round() / 100.0,
                exemplar: chosen_ex.map(|e| e.id.clone()),
                attempts_log,
                final_code: code,
                error: String::new(),
            };
        }

        if attempt < attempts {
            let hint = diagnostic_hint(&eval.error);
            let hint_line = if hint.is_empty() {
                String::new()
            } else {
                format!("\n{hint}")
            };

            let is_fresh_reset = completion.finish_reason == "length"
                || (!err_head.is_empty() && err_head == prev_err_sig)
                || attempt == 2
                || (attempts > 1 && attempt == attempts - 1);

            if completion.finish_reason == "length" {
                messages = base_messages.clone();
                messages.push(Message {
                    role: "user".to_string(),
                    content: "Percobaan sebelumnya TERPOTONG karena melewati batas token.\n\
                              Tulis ulang implementasi Python yang lebih ringkas, padat, dan lengkap \
                              di dalam satu blok ```python ... ``` tanpa penjelasan panjang."
                        .to_string(),
                });
            } else if is_fresh_reset {
                let prev_notes: Vec<&str> = attempts_log
                    .iter()
                    .rev()
                    .take(2)
                    .rev()
                    .map(|a| a.error.as_str())
                    .filter(|s| !s.is_empty())
                    .collect();
                let prev_notes = prev_notes.join("; ");
                messages = base_messages.clone();
                messages.push(Message {
                    role: "user".to_string(),
                    content: format!(
                        "Pendekatan sebelumnya gagal ({prev_notes}). Detail terakhir:\n```\n{}\n```{hint_line}\n\
                         JANGAN salin kode rusak sebelumnya. Ganti dengan pendekatan/algoritma yang lebih \
                         sederhana dan bersih dari awal (utamakan stdlib seperti `deque`/`bisect`/`set` jika \
                         diperbolehkan soal) di dalam satu blok ```python ... ```.",
                        eval.error
                    ),
                });
            } else {
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("```python\n{code}\n```"),
                });
                messages.push(Message {
                    role: "user".to_string(),
                    content: format!(
                        "Kode di atas GAGAL saat diuji:\n```\n{}\n```{hint_line}\n\
                         Periksa teliti tanda kurung, update variabel/state, batas indeks, dan kondisi berhenti. \
                         Sebutkan akar bug dalam 1 kalimat singkat, lalu tulis ulang kode Python lengkap yang \
                         sudah diperbaiki di dalam satu blok ```python ... ```.",
                        eval.error
                    ),
                });
            }
            prev_err_sig = err_head;
        }
    }

    TaskResult {
        id: task.id.clone(),
        passed: false,
        attempts: attempts_log.len().max(1),
        time_s: (total_s * 10.0).round() / 10.0,
        tokens: total_tok,
        tps: (last_tps * 100.0).round() / 100.0,
        exemplar: chosen_ex.map(|e| e.id.clone()),
        attempts_log,
        final_code: last_code,
        error: last_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_last_python_block_and_strips_think() {
        let raw = "<think>\n```python\ndef wrong(): pass\n```\n</think>\n\
                   Draft:\n```python\ndef v1(): return 1\n```\n\
                   Final:\n```py\ndef v2(): return 2\n```";
        assert_eq!(extract_python_code(raw), "def v2(): return 2");
    }

    #[test]
    fn extracts_unclosed_fence() {
        let raw = "Here:\n```python\ndef foo(x):\n    return x * 2";
        assert_eq!(extract_python_code(raw), "def foo(x):\n    return x * 2");
    }

    #[test]
    fn selects_matching_exemplar_only_when_two_keywords_hit() {
        let exs = vec![Exemplar {
            id: "ex_bfs".into(),
            categories: vec!["graph".into()],
            keywords: vec!["bfs".into(), "shortest".into(), "grid".into()],
            problem: "p".into(),
            reasoning: "r".into(),
            code: "c".into(),
        }];
        let t_match = Task {
            id: "t1".into(),
            category: "graph".into(),
            prompt: "Find shortest path on a grid".into(),
            tests: "".into(),
        };
        let t_weak = Task {
            id: "t2".into(),
            category: "graph".into(),
            prompt: "Draw a grid".into(),
            tests: "".into(),
        };
        assert_eq!(select_exemplar(&t_match, &exs).map(|e| e.id.as_str()), Some("ex_bfs"));
        assert!(select_exemplar(&t_weak, &exs).is_none());
    }
}
