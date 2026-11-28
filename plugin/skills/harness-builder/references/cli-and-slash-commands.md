# Hermes-Style Named CLI & Slash Commands (`/`) Blueprint

Read this reference when the user wants their harness to have a **custom command name** (e.g. typing `nova` or `jarvis` in terminal/CMD launches an interactive Hermes-style shell) and **custom `/slash` commands**.

---

## 1. Custom CLI Command Name (`<agent-name>` in Terminal / CMD)

### Validate the Name First
- Must be lowercase kebab-case or alphanumeric (`[a-z][a-z0-9_-]*`), e.g. `nova`, `hermes-local`, `kuro`.
- Avoid colliding with core OS commands (`test`, `ls`, `cd`, `python`, `cargo`, `git`).

### Configure the Binary Name in `Cargo.toml`
Set `[[bin]]` so Cargo builds `<agent-name>` (or `<agent-name>.exe` on Windows) directly:

```toml
[[bin]]
name = "<agent-name>"
path = "src/main.rs"
```

### Install to `PATH` (Adapt to User's OS from Phase 2)
- **Linux / macOS / WSL**:
  ```bash
  cargo install --path harness-rs --force
  # Or symlink without cargo install:
  mkdir -p ~/.local/bin
  ln -sf "$(pwd)/harness-rs/target/release/<agent-name>" ~/.local/bin/<agent-name>
  # Ensure ~/.local/bin is in PATH (~/.bashrc or ~/.zshrc):
  # export PATH="$HOME/.local/bin:$PATH"
  ```
- **Windows Native (PowerShell)**:
  ```powershell
  cargo install --path .\harness-rs --force
  # Cargo installs to %USERPROFILE%\.cargo\bin\<agent-name>.exe (already in PATH after rustup).
  ```

After installation, running `<agent-name>` with **no arguments** opens the interactive Hermes-style REPL; running `<agent-name> --tasks ...` or `<agent-name> bot` runs non-interactive subcommands directly.

---

## 2. Hermes-Style Interactive Terminal UI (Zero Extra Crates)

Keep it fast, clean, and dependency-free using ANSI escape codes + `std::io::{stdin, stdout, Write}`:

```rust
pub struct ReplState {
    pub agent_name: String,
    pub model: String,
    pub api_url: String,
    pub mode: String, // "smart" | "plain"
    pub history: Vec<Message>,
}

fn print_banner(state: &ReplState) {
    println!("\x1b[1;36m╭──────────────────────────────────────────────╮\x1b[0m");
    println!("\x1b[1;36m│\x1b[0m  \x1b[1m{}\x1b[0m  •  Harness Agent CLI              \x1b[1;36m│\x1b[0m", state.agent_name);
    println!("\x1b[1;36m╰──────────────────────────────────────────────╯\x1b[0m");
    println!("  Model : {} ({})", state.model, state.api_url);
    println!("  Mode  : {}  |  Type \x1b[33m/help\x1b[0m for commands, \x1b[33m/exit\x1b[0m to quit\n", state.mode);
}

pub fn run_repl(state: &mut ReplState, agent: &ureq::Agent, registry: &SlashRegistry) {
    print_banner(state);
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("\x1b[1;32m{}>\x1b[0m ", state.agent_name);
        let _ = std::io::stdout().flush();
        line.clear();
        if stdin.read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF (Ctrl+D / Ctrl+Z)
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            if handle_slash_command(input, state, agent, registry) == LoopControl::Exit {
                break;
            }
        } else {
            // Regular prompt -> send to LLM (or solver/sandbox) with state.history
            handle_chat_turn(input, state, agent);
        }
    }
}
```

> **Windows ANSI tip**: Windows 10/11 Windows Terminal & modern PowerShell support ANSI escape codes natively; if the user uses legacy `cmd.exe`, provide a `--no-color` flag or detect `NO_COLOR` env var.

---

## 3. Modular Slash Command (`/`) Router

Every slash command requested by the client MUST be registered in ONE single dispatcher so **both** the interactive CLI REPL and the Bot Agent (e.g. Telegram `/eval`) share the exact same command logic!

### Recommended Core Built-In Slash Commands
Always include these baseline commands unless the client asks to strip them:
| Command | Purpose |
|---|---|
| `/help` | List all active `/commands` with 1-line descriptions |
| `/status` | Show active model, API health check (`/health`), OS, mode, history length |
| `/model <name>` | Switch target model ID at runtime |
| `/mode <smart\|plain>` | Toggle self-repair mode vs 1-shot baseline |
| `/clear` | Clear conversation history |
| `/exit` (or `/quit`) | Exit the interactive CLI |

### Common Client-Requested Harness Slash Commands
Implement whichever ones the client selects in Phase 2 (plus any custom `/<name>` they invent):
| Command | Implementation Hook |
|---|---|
| `/eval [tasks.json]` | Run `qwen-harness` evaluation loop on a tasks file and print live `[PASS]/[FAIL]` table |
| `/check [tasks.json]` | Run `self_check_tasks()` via the embedded Python AST worker |
| `/fix <file> <test_cmd>` | Read `<file>`, run tests in sandbox, feed error traceback to `solver.rs` self-repair loop |
| `/run <python_expr>` | Execute a quick snippet safely inside the OS-watchdog sandbox (`sandbox::run_task_tests`) |
| `/exemplar <list\|add>` | Inspect or add domain exemplars |
| `/bot <start\|status>` | Launch or inspect the configured Bot Agent bridge |
| `/export <path.json>` | Save current session history or latest eval report to JSON |
| `/<custom>` | Client-defined prompt template or tool workflow |

### Clean Dispatcher Pattern (Rust)

```rust
#[derive(PartialEq, Eq)]
pub enum LoopControl { Continue, Exit }

pub struct SlashCommand {
    pub name: &'static str,        // e.g. "/eval"
    pub usage: &'static str,       // e.g. "/eval [tasks.json] [--ids id1,id2]"
    pub description: &'static str,
    pub handler: fn(args: &str, state: &mut ReplState, agent: &ureq::Agent) -> Result<String, String>,
}

pub fn dispatch_slash(
    raw: &str,
    state: &mut ReplState,
    agent: &ureq::Agent,
    commands: &[SlashCommand],
) -> (LoopControl, String) {
    let mut parts = raw.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();

    if cmd == "/exit" || cmd == "/quit" {
        return (LoopControl::Exit, "Bye!".to_string());
    }
    if cmd == "/help" {
        let mut out = String::from("Available commands:\n");
        for c in commands {
            out.push_str(&format!("  {:24} {}\n", c.usage, c.description));
        }
        return (LoopControl::Continue, out);
    }
    if let Some(entry) = commands.iter().find(|c| c.name == cmd) {
        let msg = match (entry.handler)(args, state, agent) {
            Ok(s) => s,
            Err(e) => format!("Command error ({cmd}): {e}"),
        };
        return (LoopControl::Continue, msg);
    }
    (LoopControl::Continue, format!("Unknown command `{cmd}`. Type `/help` to list commands."))
}
```

Notice how `dispatch_slash` returns a `String` instead of printing directly — this lets **CLI REPL** print it to `stdout` AND lets **Bot Agent** send the exact same output back to Telegram/Discord!

---

## 4. Mandatory Verification for CLI & Slash Commands

Before handing back to the user in Phase 4:
1. `cargo test` covering `dispatch_slash` (known command, unknown command, `/help`, `/exit`, custom command parsing).
2. `cargo build --release` verifying binary name matches `<agent-name>`.
3. Non-interactive pipe smoke test:
   ```bash
   printf "/help\n/status\n/exit\n" | ./target/release/<agent-name>
   ```
   Confirm the banner renders, `/help` lists every client-requested `/command`, and `/exit` terminates with exit code `0`.
