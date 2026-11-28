# Qwen3.5-9B Local Lab — Selective GGUF Surgery & CPU-Only Reasoning Engine

**100% Local · Uncensored · 0 GPU · \$0 Cloud** — Engineered and benchmarked on a standard laptop (**Intel Core i5-1135G7 4C, 7.6 GiB RAM**).

- **Download Shrunk Model (`3.49 GiB` Uncensored GGUF)**: [huggingface.co/PeterPetir/Qwen3.5-9B-Shrunk-OutQ6K-GGUF](https://huggingface.co/PeterPetir/Qwen3.5-9B-Shrunk-OutQ6K-GGUF)
- **Portfolio**: [peterpetir123.github.io](https://peterpetir123.github.io)

---

## Highlights

1. **Selective GGUF Tensor Surgery (`4.60 GiB → 3.49 GiB`, -1.12 GiB)**
   - [`tools/shrink_output_head.py`](tools/shrink_output_head.py) calls `libggml-base.so` (`ggml_quantize_chunk`) via `ctypes` to quantize **only** `output.weight` from `BF16` (`1,940 MiB`, 41.3% of the original file) to `Q6_K` (`795.7 MiB`), while copying all remaining **426 tensors + 66 metadata fields** untouched bit-for-bit (zero double-quantization).
2. **Extra Engine v3 — Inference-Time Reasoning & Self-Repair (`0% → 90.0%` on 30 Python Tasks)**
   - [`benchmark/eval_harness.py`](benchmark/eval_harness.py): AST `assert` introspection (`actual vs expected + locals`), `signal.alarm` infinite-loop detection with exact offending line numbers.
   - [`benchmark/smart_solver.py`](benchmark/smart_solver.py): disables runaway `<think>` loops (`enable_thinking: false`), routes verified non-leaky engineering invariants ([`data/teacher_exemplars.json`](data/teacher_exemplars.json)), and performs adaptive retries + *Fresh-Design Reset* to eliminate anchoring bias.
3. **Hybrid Rust Orchestrator + Embedded Python AST Sandbox ([`harness-rs/`](harness-rs/))**
   - Compiles to a single **2.0 MB** stripped binary (`qwen-harness`, **~3.3 MB RSS** host footprint) with HTTP/HTTPS OpenAI-compatible client, incremental `--resume` reporting, and an unbypassable **Rust OS watchdog (`SIGKILL`)** wrapping an embedded `python3 -I -u` AST worker (zero temp files).
4. **Interactive Agent Plugin & Skill ([`plugin/`](plugin/) — `harness-builder`)**
   - Drop-in Antigravity/agent plugin bundle ([`plugin.json`](plugin/plugin.json) + [`SKILL.md`](plugin/skills/harness-builder/SKILL.md)) that guides an agent through 4 phases: inspecting a user's files, interviewing on harness requirements, generating & verifying custom `tasks.json` + `exemplars.json`, and running an iterative review loop until every part matches.
   - Includes a verified custom binary-domain example in [`benchmark/gguf_harness/`](benchmark/gguf_harness/).

---

## Verified 30-Task CPU-Only Benchmark Results

All raw JSON logs are stored in [`benchmark/`](benchmark/):

| Stage / Configuration | Pass Rate | Speed (CPU-Only) | Model Size |
|---|---:|---:|---:|
| Baseline 9B Plain (`<think>` unconstrained) | **0.0%** (`0/2`, stalled) | 10–13 min/task | `4.60 GiB` |
| Shrunk 9B + Smart Engine (initial 10 tasks) | **70.0%** (`7/10`, `try=1`) | `1.5–3 min/task` | **`3.49 GiB`** |
| Shrunk 9B + Stage 1–2 Verifier (initial 10 tasks) | **100.0%** (`10/10` combined, `try=1`) | `1.23–1.69 t/s` | **`3.49 GiB`** |
| Shrunk 9B + Extra Engine v2 (20 unseen tasks) | **65.0%** (`13/20`) | `0.90–2.04 t/s` | **`3.49 GiB`** |
| **Shrunk 9B + Extra Engine v3 (Full 30 tasks)** | **90.0% (`27/30` combined)** *(10/10 initial + 17/20 unseen)* | **`1.25–2.35 t/s`** | **`3.49 GiB`** |

### Head-to-Head vs `Qwen3.8-27B` (`IQ2_M`, Same Laptop: Intel i5 · 7.6 GiB RAM · 0 GPU)

| Metric | `Qwen3.8-27B` (`IQ2_M`) | `Qwen3.5-9B Shrunk + Extra Engine v3` | Ratio / Delta |
|---|---|---|---|
| **File Size** | `10.87 GiB` *(exceeds RAM, SSD `mmap`)* | **`3.49 GiB`** *(fits in RAM, ~3.7 GiB free)* | **3.1× smaller** |
| **Generation Speed (CPU)** | `0.081 t/s` *(160 tok = 34.2 min)* | **`1.25–2.35 t/s`** | **16×–29× faster** |
| **Direct Multi-Delimiter Probe (`try=1`)** | ❌ Failed (`sort_build_ids`, inverted `~tag` priority) | ✅ Passed (`py_10_semver_sort`, 269.9s) | Verifier + invariant wins |
| **Coding Accuracy** | *Repo GPU claim: ~73.5–88.2% (not run 30× on laptop due to ~20h swap runtime)* | **90.0% (`27/30` verified)** *(17/20 unseen)* | Practical local parity+ |

---

## Quick Start

### 1. Run the Shrunk Model Locally (`llama.cpp`)
Download `Qwen3.5-9B-shrunk-outQ6K.gguf` from [Hugging Face](https://huggingface.co/PeterPetir/Qwen3.5-9B-Shrunk-OutQ6K-GGUF), then start `llama-server`:

```bash
llama-server \
  -m Qwen3.5-9B-shrunk-outQ6K.gguf \
  --host 127.0.0.1 --port 8081 \
  -c 4096 -np 1 -t 4 --jinja --no-reasoning-preserve
```

### 2. Build & Run the Hybrid Rust Harness (`qwen-harness`, Recommended)
```bash
cd harness-rs && cargo build --release && cd ..

# Validate tasks & AST instrumentation
./harness-rs/target/release/qwen-harness --tasks benchmark/tasks_30.json --self-check

# Run smart self-repair mode against local llama-server (or any OpenAI-compatible API via OPENAI_API_KEY)
./harness-rs/target/release/qwen-harness \
  --tasks benchmark/tasks_30.json \
  --model Qwen3.5-9B \
  --mode smart --max-iters 5 --resume \
  --out benchmark/my_run.json
```

*(Legacy pure-Python runners remain available in `benchmark/eval_harness.py` and `benchmark/smart_solver.py`.)*

### 3. Install the `harness-builder` Agent Plugin
Give your coding agent the interactive 4-phase harness builder skill (inspect codebase → interview → build & `--self-check` → iterative review):

```bash
mkdir -p ~/.gemini/config/plugins
cp -r plugin ~/.gemini/config/plugins/harness-builder
```

Restart your agent/IDE so the plugin directory is loaded, then ask the agent to inspect your project and build a custom harness. See [`benchmark/gguf_harness/`](benchmark/gguf_harness/) for a verified custom domain example.

---

## License & Credits

- Code in this repository is released under **Apache-2.0**.
- Model credits: **Qwen Team** (`qwen35` architecture), **Nightmedia**, and **DavidAU** (upstream `Qwen3.5-9B` merge).
