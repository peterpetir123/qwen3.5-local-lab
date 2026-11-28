# Bot Agent Blueprint (`<agent-name> bot`)

Read this reference when the user requests **Bot Agent** functionality (controlling or chatting with their harness remotely via Telegram, Discord, Webhook HTTP, or running it as an autonomous background daemon).

---

## 1. Mandatory Security Rules (Never Skip)

A harness can execute code inside its sandbox and query local/cloud LLMs. Enforce these 4 boundaries on EVERY bot implementation:

1. **Secrets in Environment Variables Only**:
   - Read tokens exclusively from env vars (`TELEGRAM_BOT_TOKEN`, `DISCORD_WEBHOOK_URL`, `BOT_SHARED_SECRET`).
   - NEVER hardcode tokens in Rust/Python files or commit `.env` files (ensure `.env` is in `.gitignore`).
2. **Strict Allowlist (`ALLOWED_CHAT_IDS`)**:
   - Require an allowlist env var (e.g. `TELEGRAM_ALLOWED_CHATS="123456789,987654321"`).
   - Reject or silently ignore messages from any chat/user ID not in the allowlist! Fail closed (refuse to run or reject all chats) if allowlist is unset in untrusted environments.
3. **Always Route Code Through the Sandbox Watchdog**:
   - Never run raw shell commands from chat text unless explicitly requested AND gated by allowlist + timeout (`sandbox.rs` OS watchdog `child.kill()`).
4. **Prevent Concurrent CPU Thrashing (Local `llama-server`)**:
   - Local CPU `llama-server` typically runs `--parallel 1`. Process incoming bot requests **sequentially** (FIFO queue / single worker thread) and send an immediate `"⏳ Sedang diproses..."` reply if a task is long-running.

---

## 2. Architecture: Single Command Engine Shared by CLI & Bot

```text
  Terminal REPL (<agent-name>)  ──┐
                                  ├──▶ dispatch_slash() + solver.rs + sandbox.rs
  Bot Agent (<agent-name> bot)  ──┘
```

Because `dispatch_slash()` returns `String`, every `/command` built for the CLI automatically works inside the Bot Agent too (e.g. sending `/status`, `/eval`, `/check`, or a normal prompt from Telegram).

---

## 3. Bot Mode A — Telegram Long-Polling Bot (Recommended, Zero Extra Crates)

Uses the existing `ureq` + `serde_json` crates already in `harness-rs/Cargo.toml` — works behind NAT/laptop Wi-Fi with **zero public IP, zero ngrok, zero webhooks**.

### Core Polling Loop Pattern (Rust)

```rust
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Duration;

#[derive(Deserialize)]
struct TgResp<T> { ok: bool, result: T }

#[derive(Deserialize)]
struct TgUpdate {
    update_id: i64,
    #[serde(default)]
    message: Option<TgMessage>,
}

#[derive(Deserialize)]
struct TgMessage {
    chat: TgChat,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct TgChat { id: i64 }

fn tg_send(agent: &ureq::Agent, token: &str, chat_id: i64, text: &str) {
    // Telegram hard limit is 4096 chars per message; chunk cleanly.
    let url = format!("https://api.telegram.org/bot{token}/sendMessage");
    for chunk in chunk_utf8(text, 3900) {
        let body = serde_json::json!({ "chat_id": chat_id, "text": chunk });
        let _ = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_string(&body.to_string());
    }
}

pub fn run_telegram_bot(
    token: &str,
    allowed_chats: &HashSet<i64>,
    state: &mut ReplState,
    commands: &[SlashCommand],
) -> Result<(), String> {
    let poll_agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(45))
        .build();
    let llm_agent = build_http_agent();

    let mut offset: i64 = 0;
    println!("Telegram Bot Agent `{}` online (allowlist: {} chat(s))...", state.agent_name, allowed_chats.len());

    loop {
        let url = format!(
            "https://api.telegram.org/bot{token}/getUpdates?timeout=30&offset={offset}"
        );
        let resp = match poll_agent.get(&url).call() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Poll warning: {e}; retrying in 3s...");
                std::thread::sleep(Duration::from_secs(3));
                continue;
            }
        };
        let parsed: TgResp<Vec<TgUpdate>> = match serde_json::from_reader(resp.into_reader()) {
            Ok(p) if p.ok => p,
            _ => continue,
        };

        for upd in parsed.result {
            offset = upd.update_id + 1;
            let Some(msg) = upd.message else { continue };
            let Some(raw_text) = msg.text else { continue };
            let text = raw_text.trim();
            if text.is_empty() { continue; }

            if !allowed_chats.contains(&msg.chat.id) {
                eprintln!("Blocked unauthorized chat_id={}", msg.chat.id);
                continue;
            }

            // Normalize Telegram `@botname` suffix on slash commands (e.g. `/eval@MyBot` -> `/eval`)
            let normalized = strip_bot_mention(text);
            let reply = if normalized.starts_with('/') {
                let (_, out) = dispatch_slash(&normalized, state, &llm_agent, commands);
                out
            } else {
                handle_chat_turn_string(&normalized, state, &llm_agent)
            };
            tg_send(&poll_agent, token, msg.chat.id, &reply);
        }
    }
}
```

### Helper Edge Cases to Include & Unit-Test
1. **`strip_bot_mention("/eval@NovaBot tasks.json") -> "/eval tasks.json"`**: Telegram group chats append `@BotUsername` to slash commands; strip it on the first token only.
2. **`chunk_utf8(text, max_chars)`**: Split by character boundary (never slice mid-UTF-8 byte!) so multi-byte Indonesian/emoji outputs never panic.

---

## 4. Bot Mode B — Discord Webhook / Generic HTTP Bot

- **Outbound Notifications (Discord / Slack / Custom Webhook)**:
  POST JSON `{"content": "..."}` via `ureq` to `DISCORD_WEBHOOK_URL` whenever a scheduled `--eval` finishes or a task fails.
- **Local HTTP Trigger Server (Zero Extra Crates)**:
  Use `std::net::TcpListener::bind("127.0.0.1:8090")` with a shared `X-Bot-Secret` header to receive JSON `{"prompt": "/eval ..."}` from external automation (n8n, cron, local webhooks) and return JSON `{"reply": "..."}`.

---

## 5. Bot Mode C — Autonomous Watch / Scheduled Daemon Bot

When the user wants a bot that runs **without** a messaging platform:
- `<agent-name> bot --watch <tasks.json> --interval 300`
- Runs on a timer (`std::thread::sleep(Duration::from_secs(interval))`), checks file mtime or runs `solve_task` on changed/failed tasks, writes incremental JSON reports, and optionally pings a webhook on regressions.

---

## 6. Verification Checklist for Bot Agent

1. **Unit tests (`cargo test`)**:
   - Unauthorized `chat_id` is rejected by allowlist check.
   - `strip_bot_mention` normalizes `/status@MyBot arg` $\rightarrow$ `/status arg` without touching normal text or email addresses.
   - `chunk_utf8` splits >3900-char strings safely on UTF-8 char boundaries.
   - Missing `TELEGRAM_BOT_TOKEN` returns a clean error message (never panics).
2. **Optional live check** (only if user provides token in env):
   - `<agent-name> bot --check` calls `getMe` to verify token validity without entering the infinite loop.
