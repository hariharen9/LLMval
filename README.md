# LLMVal 🚀

A clean, elegant dashboard for discovering the smartest, highest-value LLMs across OpenRouter, ranked against live [Artificial Analysis](https://artificialanalysis.ai) benchmark evaluations (General Intelligence, Coding, Agentic, and Speed).

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-standard--library-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## ✨ Features

- **General-Purpose Model Discovery**: Live intelligence benchmarks paired with real-time OpenRouter pricing.
- **Deep Benchmark Dimensions**:
  - **General Intelligence (AA)**: Aggregate multi-domain intelligence index.
  - **Coding Index (AA)** & **LiveCodeBench**: Real-world programming and contamination-free contest coding.
  - **Autonomous Agents & Tool-Use**: **TerminalBench** (CLI agents) and **Tau-bench** (multi-turn workflows).
  - **Math & Quantitative Reasoning**: Math 500 and AIME Olympiad logic.
  - **Frontier Reasoning**: GPQA (PhD-level hard science) and HLE (Humanity's Last Exam).
  - **Instruction Following**: IFBench strict formatting & JSON adherence.
- **Speed & Latency Telemetry**:
  - **Throughput (Tokens/Sec)**: Real generation speed.
  - **Time to First Token (TTFT)**: Snappiness for interactive chat and completion.
- **Creator & Lab Filtering**: Filter instantly by lab (e.g. *Anthropic*, *OpenAI*, *DeepSeek*, *Alibaba / Qwen*, *Google*, *Meta*, *Mistral*).
- **Curated Recommendations**:
  - 🏆 **Best Value**: Highest quality-per-dollar among paid models.
  - 🧠 **Smartest in Budget**: Highest scoring model under your price ceiling.
  - ⚡ **Fastest Generation**: Top throughput model passing your quality threshold.
  - 🎁 **Best Free Model**: Top-scoring 100% free model.
- **Batch API Toggle**: Optional toggle to include 50%-discounted asynchronous batch endpoints.
- **Quota-Protected Disk Caching**: OpenRouter prices cache for 30 minutes; Artificial Analysis benchmarks cache for 24 hours into `benchmarks.json`.
- **Zero External Dependencies**: Powered strictly by Python's standard library.

---

## 🚀 Installation & Usage

### Method 1: Install via pip (Recommended)

From the project directory:
```bash
pip install .
```
Or for local development (editable mode):
```bash
pip install -e .
```

Once installed, simply run anywhere from your terminal:
```bash
llmval
```

---

### Method 2: Run directly without installation

```bash
# Direct python run:
python main.py

# Or as a module:
python -m llmval
```
Your default browser will automatically open to `http://localhost:8000`.

---

## ⚙️ Options & Flags

```bash
# Custom port
llmval --port 9000

# Headless / server mode (do not open browser automatically)
llmval --no-browser

# Custom Artificial Analysis API Key
llmval --api-key your_api_key_here
```

Environment variables are also supported:
- `PORT=8000`
- `AA_API_KEY=your_key`
- `NO_BROWSER=1`

---

## 🛠️ Custom Model Matching & Overrides

### 1. Aliases (`aliases.json`)
If an OpenRouter model slug doesn't automatically match an Artificial Analysis entry, map it manually:
```json
{
  "anthropic/claude-3.7-sonnet": "claude-3-7-sonnet",
  "meta-llama/llama-3.3-70b-instruct": "llama-3-3-70b-instruct"
}
```

### 2. Manual Benchmarks (`benchmarks.json`)
You can supply your own score overrides for any OpenRouter slug:
```json
{
  "custom/my-fine-tuned-model": 48.5
}
```

---

## 📁 Project Structure

```
llmval/
├── llmval/
│   ├── __init__.py
│   ├── __main__.py
│   └── app.py            # Core engine, API fetchers & Web UI
├── main.py               # Top-level runner
├── pyproject.toml        # Package & metadata definition
├── requirements.txt      # (Standard library only)
├── .gitignore            # Ignores cache files & bytecode
├── .env                  # API keys and server configuration
├── aliases.example.json  # Example alias mapping
└── benchmarks.example.json # Example benchmark overrides
```

## 👨‍💻 Creator

Created by **[Hariharen](https://hariharen.site)**.

---

## 📄 License
MIT License.

