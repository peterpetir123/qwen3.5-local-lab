# Harness Builder Plugin

Plugin & agent skill interaktif untuk membangun, memverifikasi, dan merevisi *LLM evaluation & self-repair harness* berbasis **`qwen-harness`** (Rust Core Orchestrator + Python AST Sandbox Worker).

## Apa yang Dilakukan Agent saat Plugin Ini Aktif

1. **Membaca & memahami file/codebase Anda** (fungsi, aturan bisnis, tipe data, edge case).
2. **Wawancara lengkap kebutuhan harness**:
   - Sistem operasi (Linux, macOS, Windows native, atau WSL).
   - **Nama command CLI ala Hermes** (`<nama-agent>` yang langsung bisa dipanggil di terminal/CMD via `PATH`).
   - **Fitur Slash Command (`/`)** yang diinginkan client (`/help`, `/eval`, `/check`, `/fix`, `/run`, `/bot`, dan custom `/<fitur>`).
   - **Fitur Bot Agent** (Telegram long-polling tanpa ngrok, Discord/HTTP Webhook, atau Watch Daemon lengkap dengan allowlist keamanan).
   - Model/endpoint target, edge case wajib, mode `smart`/`plain`, timeout, dan exemplar.
3. **Membangun & memverifikasi harness** menggunakan blueprint di `skills/harness-builder/references/` (`harness-spec.md`, `cli-and-slash-commands.md`, `bot-agent-blueprint.md`) lengkap dengan `cargo test`, `--self-check`, dan pipe smoke test CLI.
4. **Loop review iteratif**: menampilkan ringkasan & bukti verifikasi, lalu menanyakan bagian mana yang belum sesuai hingga harness benar-benar cocok dengan keinginan Anda.

## Instalasi Lokal (Antigravity)

Salin direktori `plugin/` ini ke `~/.gemini/config/plugins/harness-builder/`:

```bash
mkdir -p ~/.gemini/config/plugins
cp -r plugin ~/.gemini/config/plugins/harness-builder
```

Lalu restart Antigravity agar plugin baru terdeteksi oleh language server.

## Prasyarat Engine

Pastikan binary `qwen-harness` sudah dibangun di `../harness-rs/`:

```bash
cd ../harness-rs && cargo build --release
```
