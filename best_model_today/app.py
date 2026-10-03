#!/usr/bin/env python3
"""Best Model Today v3 (Coding Edition): Comprehensive LLM intelligence, coding benchmarks & live pricing."""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def load_dotenv():
    """Loads environment variables from .env file with standard library only."""
    search_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
    ]
    for p in search_paths:
        if p.exists() and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass
            break


# Load .env file automatically
load_dotenv()

# --- Configuration & Paths ---
DEFAULT_PORT = int(os.environ.get("PORT", 8000))
AA_KEY = os.environ.get("AA_API_KEY", "")

DATA_DIR = Path(os.environ.get("BEST_MODEL_DATA_DIR", Path.cwd()))
CACHE_FILE = DATA_DIR / "cache.json"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"

# Cache TTLs in seconds: OpenRouter 30m, Artificial Analysis 24h (1 day limit)
OR_TTL = 1800
AA_TTL = 86400          # 1 day (24 hours)
AA_MIN_REFRESH = 86400  # Allow refresh from AA only once per 24 hours

LOCK = threading.Lock()

# Fallback seed snapshot if offline or without AA API access
SEED_DATE = "2026-10-03"
SEED = {
    "xiaomi/mimo-v2.6-pro": {"intel": 46.0, "coding": 48.0, "creator": "Xiaomi"},
    "z-ai/glm-5.3-flash": {"intel": 42.0, "coding": 44.5, "creator": "Z-AI"},
    "deepseek/deepseek-v4.1-flash": {"intel": 39.5, "coding": 43.0, "creator": "DeepSeek"},
    "xiaomi/mimo-v2.6-flash": {"intel": 37.9, "coding": 40.0, "creator": "Xiaomi"},
    "qwen/qwen3.8-27b": {"intel": 34.0, "coding": 36.0, "creator": "Alibaba"},
    "nvidia/nemotron-3-ultra-550b-a55b": {"intel": 23.0, "coding": 25.0, "creator": "NVIDIA"},
    "openai/gpt-6-luna": {"intel": 38.1, "coding": 42.0, "creator": "OpenAI"},
    "z-ai/glm-5.3": {"intel": 45.0, "coding": 49.0, "creator": "Z-AI"},
    "qwen/qwen3.8-max-0902": {"intel": 45.4, "coding": 51.0, "creator": "Alibaba"},
    "anthropic/claude-sonnet-5.5": {"intel": 56.0, "coding": 64.5, "creator": "Anthropic"},
    "anthropic/claude-opus-5.5": {"intel": 57.6, "coding": 66.0, "creator": "Anthropic"},
}

VARIANT_SUFFIXES = re.compile(
    r"(reasoning|nonreasoning|thinking|adaptive|max|xhigh|high|medium|low|minimal|fallback|default|effort|\d{4,})*"
)


def load_json(path: Path, default_val=None):
    """Safely loads JSON from a file, returning default_val on error."""
    if default_val is None:
        default_val = {}
    if not path.exists():
        return default_val
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_val


def save_json(path: Path, data):
    """Safely writes JSON to a file."""
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def http_get(url: str, headers: dict = None, timeout: int = 25):
    """Performs an HTTP GET request with standard headers and returns decoded JSON."""
    default_headers = {
        "User-Agent": "best-model-today-v3/3.0.0",
        "Accept": "application/json",
    }
    if headers:
        default_headers.update(headers)
    req = urllib.request.Request(url, headers=default_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_slug(text: str) -> str:
    """Standardizes slug or model name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def get_cached_or_fetch(cache_key: str, ttl: int, fetcher, force: bool, min_refresh: int, warns: list, label: str):
    """Retrieves cached item or fetches new data with disk persistence and quota protection."""
    cache = load_json(CACHE_FILE, {})
    entry = cache.get(cache_key)
    now = time.time()
    age = (now - entry["t"]) if (entry and "t" in entry) else 1e12

    # Return cached data if still within TTL and not forcing beyond min_refresh
    if entry and age < ttl and not (force and age >= min_refresh):
        return entry.get("d"), age

    # Prevent rapid spamming of quota-limited APIs (e.g. 1-day limit on AA)
    if entry and force and age < min_refresh:
        hours_ago = round(age / 3600, 1)
        next_in = round((min_refresh - age) / 3600, 1)
        warns.append(f"{label} refresh skipped: updated {hours_ago}h ago. Refreshes allowed once every 24h (next in {next_in}h).")
        return entry.get("d"), age

    # Attempt fetch
    try:
        data = fetcher()
        cache[cache_key] = {"t": now, "d": data}
        save_json(CACHE_FILE, cache)
        return data, 0
    except Exception as exc:
        if entry:
            warns.append(f"{label} fetch failed ({exc}); using cached data.")
            return entry.get("d"), age
        warns.append(f"{label} fetch failed ({exc}).")
        return None, None


def extract_creator_name(creator_val) -> str:
    """Extracts creator name whether it is a string or dict object."""
    if isinstance(creator_val, dict):
        return creator_val.get("name") or creator_val.get("slug") or ""
    elif isinstance(creator_val, str):
        return creator_val
    return ""


def normalize_benchmark_item(item: dict) -> dict:
    """Normalizes benchmark item whether it is in raw AA format or pre-parsed format."""
    if not isinstance(item, dict):
        return None

    if "coding" in item and "intel" in item and isinstance(item.get("creator"), str):
        return item

    evals = item.get("evaluations") or {}
    
    # Extract agentic benchmarks
    agentic_val = next(
        (v for k, v in evals.items() if "agentic" in k and v is not None),
        evals.get("tau2") or evals.get("tau_banking")
    )
    
    # Extract terminal / coding agent benchmark
    terminal_bench = next(
        (v for k, v in evals.items() if "terminalbench" in k and v is not None),
        None
    )

    creator_name = extract_creator_name(item.get("model_creator") or item.get("creator"))

    return {
        "name": item.get("name", ""),
        "slug": item.get("slug", ""),
        "creator": creator_name,
        "release_date": item.get("release_date") or "",
        
        # Primary Benchmarks
        "coding": evals.get("artificial_analysis_coding_index"),
        "intel": evals.get("artificial_analysis_intelligence_index"),
        "agentic": agentic_val,
        "math": evals.get("artificial_analysis_math_index") or evals.get("math_500"),
        
        # Granular Developer Benchmarks
        "livecodebench": evals.get("livecodebench"),
        "terminalbench": terminal_bench,
        "scicode": evals.get("scicode"),
        "gpqa": evals.get("gpqa"),
        "hle": evals.get("hle"),
        "ifbench": evals.get("ifbench"),
        "aime": evals.get("aime_25") or evals.get("aime"),
        
        # Latency & Throughput Telemetry
        "tps": item.get("median_output_tokens_per_second"),
        "ttft": item.get("median_time_to_first_token_seconds"),
        "ttfa": item.get("median_time_to_first_answer_token"),
    }


def parse_artificial_analysis_models(items: list) -> list:
    """Extracts benchmark metrics and speed telemetry defensively from AA API response."""
    results = []
    if not isinstance(items, list):
        return results

    for model in items:
        norm_item = normalize_benchmark_item(model)
        if norm_item:
            results.append(norm_item)
    return results


def find_matching_benchmark(base_slug: str, model_name: str, index: dict, aliases: dict) -> list:
    """Finds best matching benchmark item using exact, alias, or intelligent fuzzy heuristics."""
    clean_name = re.sub(r"\(.*?\)", "", model_name.split(":", 1)[-1])
    candidates = []

    if base_slug in aliases:
        candidates.append(normalize_slug(aliases[base_slug]))
    candidates.append(normalize_slug(base_slug.split("/")[-1]))
    candidates.append(normalize_slug(clean_name))

    # Exact match first
    for c in candidates:
        if c in index:
            return [index[c]]

    # Prefix match with allowed variation suffixes
    for c in candidates:
        if len(c) >= 4:
            hits = [
                entry for key, entry in index.items()
                if key.startswith(c) and VARIANT_SUFFIXES.fullmatch(key[len(c):])
            ]
            if hits:
                return hits
    return []


def get_benchmarks_data(force: bool, aa_key: str, warns: list) -> tuple:
    """Gets benchmarks from disk file or fetches from Artificial Analysis (1-day limit)."""
    bench_file_data = load_json(BENCHMARKS_FILE, None)
    now = time.time()
    
    file_age = 1e12
    if bench_file_data and isinstance(bench_file_data, dict) and "updated_at_ts" in bench_file_data:
        file_age = now - bench_file_data["updated_at_ts"]

    if aa_key:
        def fetch_aa():
            res = http_get(
                "https://artificialanalysis.ai/api/v2/data/llms/models",
                headers={"x-api-key": aa_key},
            )
            raw_models = res.get("data", res) if isinstance(res, dict) else res
            return parse_artificial_analysis_models(raw_models or [])

        aa_models, aa_age = get_cached_or_fetch(
            "artificial_analysis",
            AA_TTL,
            fetch_aa,
            force,
            AA_MIN_REFRESH,
            warns,
            "Artificial Analysis",
        )
        if aa_models:
            parsed = parse_artificial_analysis_models(aa_models)
            # Ensure benchmarks.json is synced on disk
            if not BENCHMARKS_FILE.exists() or (aa_age == 0):
                save_json(BENCHMARKS_FILE, {
                    "source": "Artificial Analysis API",
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - (aa_age or 0))),
                    "updated_at_ts": now - (aa_age or 0),
                    "models_count": len(parsed),
                    "models": parsed,
                })
            return parsed, aa_age

    # Fallback to local benchmarks.json if present
    if bench_file_data:
        if isinstance(bench_file_data, dict) and "models" in bench_file_data:
            return bench_file_data["models"], file_age
        elif isinstance(bench_file_data, dict):
            simple_models = []
            for k, v in bench_file_data.items():
                if k.startswith("_"):
                    continue
                score = float(v) if isinstance(v, (int, float)) else (v.get("coding") or v.get("intel") if isinstance(v, dict) else None)
                if score is not None:
                    simple_models.append({
                        "name": k,
                        "slug": k,
                        "creator": v.get("creator", "") if isinstance(v, dict) else "",
                        "coding": score,
                        "intel": score,
                        "agentic": None,
                        "math": None,
                        "livecodebench": None,
                        "terminalbench": None,
                        "tps": None,
                        "ttft": None,
                    })
            return simple_models, file_age
        elif isinstance(bench_file_data, list):
            return bench_file_data, file_age

    if not aa_key:
        warns.append(f"No AA_API_KEY in .env. Showing {SEED_DATE} snapshot. Set AA_API_KEY in .env for live 24h benchmark data.")

    return [], None


def build_dataset(force: bool = False, aa_key: str = None) -> dict:
    """Builds the comprehensive model table with matched benchmark scores and pricing."""
    if aa_key is None:
        aa_key = os.environ.get("AA_API_KEY", AA_KEY)

    warns = []

    # 1. Fetch OpenRouter Models (30m cache)
    or_data, or_age = get_cached_or_fetch(
        "openrouter",
        OR_TTL,
        lambda: http_get("https://openrouter.ai/api/v1/models").get("data", []),
        force,
        60,
        warns,
        "OpenRouter",
    )

    if not or_data:
        return {"error": "Could not connect to OpenRouter and no local cache was found. " + " ".join(warns)}

    # 2. Get Benchmarks (1-day limit per run)
    aa_raw_list, aa_age = get_benchmarks_data(force, aa_key, warns)
    aa_list = parse_artificial_analysis_models(aa_raw_list or [])

    # Build lookup index for AA
    aa_index = {}
    creators_set = set()
    for item in aa_list:
        slug_k = normalize_slug(item.get("slug"))
        name_k = normalize_slug(item.get("name"))
        if slug_k:
            aa_index[slug_k] = item
        if name_k:
            aa_index[name_k] = item
        c_name = extract_creator_name(item.get("creator"))
        if c_name:
            creators_set.add(c_name)

    # Load local overrides from working dir or package root
    aliases = load_json(DATA_DIR / "aliases.json", {})
    local_benchmarks = load_json(DATA_DIR / "benchmarks.json", {})

    rows = []
    for m in or_data:
        try:
            pricing = m.get("pricing") or {}
            pin = float(pricing.get("prompt", 0)) * 1e6
            pout = float(pricing.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            continue

        if pin < 0 or pout < 0:
            continue

        arch = m.get("architecture") or {}
        out_mods = arch.get("output_modalities") or ["text"]
        if "text" not in out_mods:
            continue

        mid = m.get("id", "")
        base, _, variant = mid.partition(":")
        params = m.get("supported_parameters") or []

        # Guess creator from slug namespace if not present
        creator_guess = base.split("/")[0].replace("-", " ").title() if "/" in base else ""

        row = {
            "id": mid,
            "name": m.get("name", mid),
            "creator": creator_guess,
            "pin": pin,
            "pout": pout,
            "blended": (pin + 3 * pout) / 4,
            "free": variant == "free" or (pin == 0 and pout == 0),
            "variant": variant,
            "ctx": m.get("context_length"),
            "created": m.get("created"),
            "tools": "tools" in params,
            "vision": "image" in (arch.get("input_modalities") or []),
            "reasoning": "reasoning" in params or "include_reasoning" in params,
            
            # Benchmarks
            "coding": None,
            "intel": None,
            "agentic": None,
            "math": None,
            "livecodebench": None,
            "terminalbench": None,
            "gpqa": None,
            "hle": None,
            "ifbench": None,
            "aime": None,
            
            # Latency & Speed
            "tps": None,
            "ttft": None,
            "ttfa": None,
            "src": "",
        }

        # Match benchmark score
        matched = find_matching_benchmark(base, row["name"], aa_index, aliases) if aa_index else []
        if matched:
            best_match = max(matched, key=lambda x: (x.get("coding") or x.get("intel") or 0))
            row.update({
                "creator": best_match.get("creator") or row["creator"],
                "coding": best_match.get("coding"),
                "intel": best_match.get("intel"),
                "agentic": best_match.get("agentic"),
                "math": best_match.get("math"),
                "livecodebench": best_match.get("livecodebench"),
                "terminalbench": best_match.get("terminalbench"),
                "gpqa": best_match.get("gpqa"),
                "hle": best_match.get("hle"),
                "ifbench": best_match.get("ifbench"),
                "aime": best_match.get("aime"),
                "tps": best_match.get("tps"),
                "ttft": best_match.get("ttft"),
                "ttfa": best_match.get("ttfa"),
                "src": "Artificial Analysis",
            })
            c_matched = extract_creator_name(best_match.get("creator") or row["creator"])
            row["creator"] = c_matched
            if c_matched:
                creators_set.add(c_matched)
        elif base in local_benchmarks and isinstance(local_benchmarks[base], (int, float)):
            val = float(local_benchmarks[base])
            row["coding"] = val
            row["intel"] = val
            row["src"] = "benchmarks.json"
        elif base in SEED:
            seed_data = SEED[base]
            if isinstance(seed_data, dict):
                row["coding"] = seed_data.get("coding")
                row["intel"] = seed_data.get("intel")
                row["creator"] = seed_data.get("creator") or row["creator"]
            else:
                row["coding"] = float(seed_data)
                row["intel"] = float(seed_data)
            row["src"] = f"seed {SEED_DATE}"

        rows.append(row)

    scored_count = sum(1 for r in rows if r["coding"] is not None or r["intel"] is not None)

    return {
        "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "scored": scored_count,
        "or_age": or_age,
        "aa_age": aa_age,
        "live_bench": bool(aa_list),
        "creators": sorted(list(creators_set)),
        "warnings": warns,
        "rows": rows,
    }


def get_data(force: bool = False, aa_key: str = None) -> dict:
    """Thread-safe dataset getter."""
    with LOCK:
        return build_dataset(force, aa_key)


# --- HTML Dashboard (Focused on Coding & Developer Workflows) ---
HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Best Model Today · Coding & Intelligence Leaderboard</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --mute: #64748b;
      --line: #e2e8f0;
      --acc: #0284c7;
      --acc-subtle: #f0f9ff;
      --acc-hover: #0369a1;
      --code-acc: #0d9488;
      --tag-bg: #f1f5f9;
      --warn: #d97706;
      --badge-free: #10b981;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0b0f17;
        --card: #151d28;
        --ink: #f1f5f9;
        --mute: #94a3b8;
        --line: #222f3e;
        --acc: #38bdf8;
        --acc-subtle: #0c2333;
        --acc-hover: #0ea5e9;
        --code-acc: #14b8a6;
        --tag-bg: #1e293b;
        --warn: #f59e0b;
        --badge-free: #34d399;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      font-size: 14px;
      line-height: 1.5;
    }
    main {
      max-width: 1240px;
      margin: 0 auto;
      padding: 30px 20px 64px;
    }
    header {
      margin-bottom: 22px;
    }
    .header-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 12px;
    }
    h1 {
      font-family: Georgia, serif;
      font-size: 32px;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 4px 0;
      color: var(--ink);
    }
    .badge-focus {
      background: var(--acc-subtle);
      color: var(--acc);
      border: 1px solid var(--acc);
      padding: 2px 9px;
      border-radius: 14px;
      font-size: 12px;
      font-weight: 600;
      display: inline-block;
      vertical-align: middle;
      margin-left: 8px;
    }
    .meta-line {
      font-size: 13px;
      color: var(--mute);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 4px;
    }
    .live-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      background: #10b981;
      border-radius: 50%;
      margin-right: 4px;
    }
    .warn-box {
      background: var(--card);
      border-left: 4px solid var(--warn);
      padding: 8px 14px;
      margin: 12px 0;
      border-radius: 6px;
      font-size: 13px;
      color: var(--ink);
      box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Filter Panel */
    .filter-panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px 18px;
      margin: 18px 0 22px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: flex-end;
      box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .checkbox-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--ink);
      cursor: pointer;
      user-select: none;
      padding-bottom: 7px;
    }
    .checkbox-label input[type="checkbox"] {
      cursor: pointer;
      width: 16px;
      height: 16px;
      accent-color: var(--acc);
      margin: 0;
    }
    .filter-group label {
      font-size: 11.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--mute);
    }
    input, select, button {
      font: inherit;
      font-size: 13.5px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      color: var(--ink);
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    input:focus, select:focus {
      border-color: var(--acc);
      box-shadow: 0 0 0 2px var(--acc-subtle);
    }
    .btn-primary {
      background: var(--acc);
      color: #ffffff;
      border: 0;
      font-weight: 500;
      cursor: pointer;
    }
    .btn-primary:hover {
      background: var(--acc-hover);
    }
    .btn-secondary {
      background: var(--card);
      color: var(--ink);
      border: 1px solid var(--line);
      cursor: pointer;
    }
    .btn-secondary:hover {
      background: var(--tag-bg);
    }

    /* Recommendation Cards (4-Grid) */
    .picks-grid {
      display: grid;
      grid-template-columns: 1.3fr 1fr 1fr 1fr;
      gap: 12px;
      margin-bottom: 28px;
    }
    @media (max-width: 980px) {
      .picks-grid {
        grid-template-columns: 1fr 1fr;
      }
    }
    @media (max-width: 600px) {
      .picks-grid {
        grid-template-columns: 1fr;
      }
    }
    .pick-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      position: relative;
    }
    .pick-card.featured {
      border: 2px solid var(--acc);
      background: linear-gradient(to bottom right, var(--card), var(--acc-subtle));
    }
    .pick-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--acc);
      margin-bottom: 4px;
    }
    .pick-title {
      font-family: Georgia, serif;
      font-size: 18px;
      font-weight: 600;
      margin: 0 0 6px 0;
      line-height: 1.25;
    }
    .pick-card.featured .pick-title {
      font-size: 22px;
    }
    .pick-metric {
      font-size: 13.5px;
      color: var(--ink);
      margin-bottom: 4px;
      font-weight: 500;
    }
    .pick-desc {
      font-size: 12px;
      color: var(--mute);
      margin-top: 4px;
    }
    .pick-id {
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 11px;
      color: var(--mute);
      margin-top: 6px;
      word-break: break-all;
    }

    /* Section Headings */
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin: 26px 0 10px;
      flex-wrap: wrap;
      gap: 8px;
    }
    h2 {
      font-family: Georgia, serif;
      font-size: 20px;
      font-weight: 600;
      margin: 0;
    }
    .subtext {
      font-size: 12.5px;
      color: var(--mute);
    }

    /* Table */
    .table-container {
      overflow-x: auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      min-width: 980px;
      font-size: 13.5px;
    }
    th, td {
      padding: 9px 12px;
      text-align: right;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th:first-child, td:first-child {
      text-align: left;
      white-space: normal;
      min-width: 220px;
    }
    th {
      font-weight: 600;
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--mute);
      background: var(--card);
      position: sticky;
      top: 0;
      cursor: pointer;
      user-select: none;
    }
    th:hover {
      color: var(--ink);
    }
    tr:last-child td {
      border-bottom: 0;
    }
    tr:hover td {
      background: rgba(0, 0, 0, 0.015);
    }

    /* Badges & Value Bars */
    .tag {
      font-size: 11px;
      font-weight: 500;
      padding: 1px 6px;
      border-radius: 10px;
      background: var(--tag-bg);
      color: var(--mute);
      margin-left: 4px;
      display: inline-block;
      vertical-align: middle;
    }
    .tag.free {
      background: rgba(16, 185, 129, 0.12);
      color: var(--badge-free);
      font-weight: 600;
    }
    .tag.creator {
      background: rgba(2, 132, 199, 0.1);
      color: var(--acc);
      font-size: 11px;
    }
    .tag.batch {
      background: rgba(245, 158, 11, 0.12);
      color: var(--warn);
      font-weight: 600;
    }
    .val-cell {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 6px;
    }
    .val-bar {
      height: 6px;
      background: var(--code-acc);
      border-radius: 3px;
      min-width: 3px;
    }
    .score-highlight {
      font-weight: 600;
      color: var(--ink);
    }

    footer {
      margin-top: 48px;
      padding-top: 20px;
      border-top: 1px solid var(--line);
      font-size: 13px;
      color: var(--mute);
      line-height: 1.6;
    }
    footer a {
      color: var(--acc);
      text-decoration: none;
    }
    footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="header-top">
        <div>
          <h1>Best model today</h1>
          <div class="meta-line" id="meta">
            <span class="live-dot"></span> Loading live pricing and benchmark evaluations…
          </div>
        </div>
      </div>
      <div id="warnings"></div>
    </header>

    <div class="filter-panel">
      <div class="filter-group">
        <label>Benchmark Metric</label>
        <select id="metric">
          <option value="intel" selected>General Intelligence (AA)</option>
          <option value="coding">Coding Index (AA)</option>
          <option value="livecodebench">LiveCodeBench (Programming)</option>
          <option value="agentic">Agentic (Tau / Tool-Use)</option>
          <option value="terminalbench">TerminalBench (Dev Agent)</option>
          <option value="math">Math & Logic Index</option>
          <option value="gpqa">GPQA (PhD Reasoning)</option>
          <option value="ifbench">Instruction Following (IFBench)</option>
          <option value="hle">Humanity's Last Exam (HLE)</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Lab / Creator</label>
        <select id="creatorFilter">
          <option value="">All Creators & Labs</option>
        </select>
      </div>

      <div class="filter-group">
        <label>Min Score</label>
        <input id="minScore" type="number" value="30" step="1" min="0" style="width: 75px;">
      </div>

      <div class="filter-group">
        <label>Max $/1M (Blended)</label>
        <input id="maxPrice" type="number" value="1.50" step="0.1" min="0" style="width: 90px;">
      </div>

      <div class="filter-group">
        <label>Max Age</label>
        <select id="age">
          <option value="0">Any age</option>
          <option value="90">Last 90 days</option>
          <option value="180" selected>Last 180 days</option>
          <option value="365">Last 1 year</option>
        </select>
      </div>

      <div class="filter-group" style="flex: 1; min-width: 140px;">
        <label>Search Models</label>
        <input id="search" type="text" placeholder="Search by name, slug, or lab…">
      </div>

      <div class="filter-group">
        <label class="checkbox-label">
          <input type="checkbox" id="includeBatch"> Include Batch (async-only)
        </label>
      </div>

      <div class="filter-group" style="flex-direction: row; gap: 8px;">
        <button class="btn-primary" id="btnRefresh">Refresh Data</button>
        <button class="btn-secondary" id="btnExport">Export CSV</button>
      </div>
    </div>

    <div class="picks-grid" id="picksGrid"></div>

    <div class="section-header">
      <h2>Ranked Candidates</h2>
      <div class="subtext">Value = score² ÷ blended price · Blended = (In + 3×Out) ÷ 4 · Click headers to sort</div>
    </div>

    <div class="table-container">
      <table>
        <thead><tr id="tableHead"></tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>

    <footer>
      <p>
        <strong>How it works:</strong> Live prices are queried from OpenRouter. Benchmark scores (Coding Index, LiveCodeBench, TerminalBench, General Intelligence, Throughput & Latency) are populated from <a href="https://artificialanalysis.ai" target="_blank" rel="noopener">Artificial Analysis</a> and cached locally in <code>benchmarks.json</code> for 24 hours. Value ranking favors high output capability per dollar ($1\text{in}:3\text{out}$ blended).
      </p>
    </footer>
  </main>

  <script>
    let DATA = null;
    let currentRows = [];
    let sortKey = "value";
    let sortDesc = true;

    const $ = id => document.getElementById(id);

    const COLUMNS = [
      { key: "name", label: "Model" },
      { key: "score", label: "Score" },
      { key: "coding", label: "Coding (AA)" },
      { key: "livecodebench", label: "LiveCode" },
      { key: "terminalbench", label: "Terminal" },
      { key: "pin", label: "In $/1M" },
      { key: "pout", label: "Out $/1M" },
      { key: "blended", label: "Blended" },
      { key: "value", label: "Value" },
      { key: "tps", label: "Speed (tok/s)" },
      { key: "ttft", label: "TTFT (s)" },
      { key: "ctx", label: "Context" },
      { key: "src", label: "Source" }
    ];

    function formatMoney(val) {
      if (val === 0) return "Free";
      if (val < 0.1) return "$" + val.toFixed(3);
      if (val < 10) return "$" + val.toFixed(2);
      return "$" + val.toFixed(1);
    }

    function createElement(tag, className, text) {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.textContent = text;
      return el;
    }

    function populateCreators(creators) {
      const select = $("creatorFilter");
      const current = select.value;
      select.replaceChildren(createElement("option", "", "All Creators & Labs"));
      select.options[0].value = "";
      
      (creators || []).forEach(c => {
        if (!c) return;
        const opt = createElement("option", "", c);
        opt.value = c;
        if (c === current) opt.selected = true;
        select.append(opt);
      });
    }

    function filterRows() {
      if (!DATA || !DATA.rows) return [];
      const metric = $("metric").value;
      const creator = $("creatorFilter").value.toLowerCase();
      const includeBatch = $("includeBatch").checked;
      const minScore = +$("minScore").value || 0;
      const maxPrice = +$("maxPrice").value || 999999;
      const ageDays = +$("age").value;
      const query = $("search").value.toLowerCase().trim();
      const now = Date.now() / 1000;

      return DATA.rows
        .filter(r => {
          const s = r[metric];
          if (s == null || s < minScore) return false;
          if (r.blended > maxPrice) return false;
          if (!includeBatch && (r.variant === "batch" || r.id.endsWith(":batch") || r.name.toLowerCase().includes("(batch)"))) return false;
          if (creator && !(r.creator || "").toLowerCase().includes(creator)) return false;
          if (ageDays && r.created && (now - r.created > ageDays * 86400)) return false;
          if (query && !(r.name + " " + r.id + " " + (r.creator || "")).toLowerCase().includes(query)) return false;
          return true;
        })
        .map(r => {
          const s = r[metric];
          const val = r.free ? null : (s ** 2) / Math.max(r.blended, 0.02);
          return { ...r, score: s, value: val };
        });
    }

    function createPickCard(badgeTitle, row, reason, isFeatured = false) {
      const card = createElement("div", "pick-card" + (isFeatured ? " featured" : ""));
      const topDiv = createElement("div");
      topDiv.append(createElement("div", "pick-label", badgeTitle));

      if (!row) {
        topDiv.append(
          createElement("h3", "pick-title", "No match found"),
          createElement("div", "pick-desc", "Try relaxing the price or score filters.")
        );
        card.append(topDiv);
        return card;
      }

      topDiv.append(
        createElement("h3", "pick-title", row.name),
        createElement("div", "pick-metric", `Score ${row.score.toFixed(1)} · ${formatMoney(row.pin)} in / ${formatMoney(row.pout)} out` + (row.tps ? ` · ${Math.round(row.tps)} tok/s` : ""))
      );

      const botDiv = createElement("div");
      botDiv.append(
        createElement("div", "pick-desc", reason),
        createElement("div", "pick-id", row.id)
      );

      card.append(topDiv, botDiv);
      return card;
    }

    function renderTable(rows) {
      const thead = $("tableHead");
      thead.replaceChildren();

      COLUMNS.forEach(col => {
        const th = createElement("th", "", col.label + (col.key === sortKey ? (sortDesc ? " ▾" : " ▴") : ""));
        th.onclick = () => {
          if (sortKey === col.key) {
            sortDesc = !sortDesc;
          } else {
            sortKey = col.key;
            sortDesc = true;
          }
          render();
        };
        thead.append(th);
      });

      const maxVal = Math.max(...rows.map(r => r.value || 0), 1);

      rows.sort((a, b) => {
        let va = a[sortKey];
        let vb = b[sortKey];
        if (typeof va === "string") {
          return (sortDesc ? -1 : 1) * va.localeCompare(vb);
        }
        va = va ?? -999999;
        vb = vb ?? -999999;
        return (sortDesc ? -1 : 1) * (va - vb);
      });

      const tbody = $("tableBody");
      tbody.replaceChildren();

      rows.forEach(r => {
        const tr = createElement("tr");

        // Name cell with creator, batch, and capability badges
        const tdName = createElement("td", "");
        tdName.append(document.createTextNode(r.name + " "));
        if (r.creator) tdName.append(createElement("span", "tag creator", r.creator));
        if (r.free) tdName.append(createElement("span", "tag free", "FREE"));
        if (r.variant === "batch" || r.id.endsWith(":batch") || r.name.toLowerCase().includes("(batch)")) {
          tdName.append(createElement("span", "tag batch", "BATCH"));
        }
        if (r.reasoning) tdName.append(createElement("span", "tag", "Reasoning"));
        if (r.tools) tdName.append(createElement("span", "tag", "Tools"));
        if (r.vision) tdName.append(createElement("span", "tag", "Vision"));

        // Value cell with visual bar
        const tdVal = createElement("td", "");
        if (r.value != null) {
          const valWrapper = createElement("div", "val-cell");
          valWrapper.append(document.createTextNode(r.value.toFixed(0)));
          const bar = createElement("span", "val-bar");
          bar.style.width = Math.max(3, Math.min(60, 60 * (r.value / maxVal))) + "px";
          valWrapper.append(bar);
          tdVal.append(valWrapper);
        } else {
          tdVal.textContent = "–";
        }

        tr.append(
          tdName,
          createElement("td", "score-highlight", r.score != null ? r.score.toFixed(1) : "–"),
          createElement("td", "", r.coding != null ? r.coding.toFixed(1) : "–"),
          createElement("td", "", r.livecodebench != null ? r.livecodebench.toFixed(1) : "–"),
          createElement("td", "", r.terminalbench != null ? r.terminalbench.toFixed(1) : "–"),
          createElement("td", "", formatMoney(r.pin)),
          createElement("td", "", formatMoney(r.pout)),
          createElement("td", "", formatMoney(r.blended)),
          tdVal,
          createElement("td", "", r.tps ? Math.round(r.tps) : "–"),
          createElement("td", "", r.ttft ? r.ttft.toFixed(2) + "s" : "–"),
          createElement("td", "", r.ctx ? Math.round(r.ctx / 1000) + "K" : "–"),
          createElement("td", "", r.src || "–")
        );

        tbody.append(tr);
      });
    }

    function render() {
      if (!DATA) return;
      const rows = filterRows();
      currentRows = rows;

      const paidRows = rows.filter(r => !r.free);
      const freeRows = rows.filter(r => r.free);

      const bestValue = [...paidRows].sort((a, b) => (b.value || 0) - (a.value || 0))[0];
      const smartestInBudget = [...rows].sort((a, b) => (b.score || 0) - (a.score || 0))[0];
      const fastestCoder = [...paidRows.filter(r => r.tps)].sort((a, b) => (b.tps || 0) - (a.tps || 0))[0];
      const bestFree = [...freeRows].sort((a, b) => (b.score || 0) - (a.score || 0))[0];

      const grid = $("picksGrid");
      grid.replaceChildren(
        createPickCard("Best Value", bestValue, "Highest score-per-dollar ratio within your filters.", true),
        createPickCard("Smartest in Budget", smartestInBudget, "Highest overall score under your price ceiling."),
        createPickCard("Fastest Generation", fastestCoder, "Fastest generation speed (tokens/sec) passing your filters."),
        createPickCard("Best Free Model", bestFree, "Highest scoring completely free model.")
      );

      renderTable(rows);
    }

    async function loadData(force = false) {
      $("meta").innerHTML = '<span class="live-dot" style="background:#f59e0b"></span> Fetching live updates…';
      try {
        const resp = await fetch(force ? "/api/refresh" : "/api/data");
        const json = await resp.json();
        if (json.error) {
          $("meta").textContent = "Error: " + json.error;
          return;
        }
        DATA = json;

        const ageText = (sec) => {
          if (sec == null) return "live";
          if (sec < 90) return "just now";
          if (sec < 3600) return Math.round(sec / 60) + "m ago";
          return (sec / 3600).toFixed(1) + "h ago";
        };

        const orStatus = ageText(json.or_age);
        const aaStatus = json.live_bench ? `AA Live (${ageText(json.aa_age)})` : "Seed Snapshot";

        $("meta").innerHTML = `<span class="live-dot"></span> Fetched ${json.fetched} · ${json.total} OpenRouter models · ${json.scored} scored · Prices: ${orStatus} · Benchmarks: ${aaStatus}`;
        
        $("warnings").replaceChildren(
          ...json.warnings.map(w => createElement("div", "warn-box", w))
        );

        populateCreators(json.creators);
        render();
      } catch (err) {
        $("meta").textContent = "Failed to load data: " + err;
      }
    }

    function exportCSV() {
      if (!currentRows.length) return;
      const headers = ["id", "name", "creator", "metric_score", "livecodebench", "terminalbench", "pin_usd_per_1m", "pout_usd_per_1m", "blended_usd_per_1m", "value", "speed_tps", "ttft_sec", "context_length", "source"];
      const lines = [headers.join(",")];
      
      currentRows.forEach(r => {
        const row = [
          r.id,
          JSON.stringify(r.name),
          JSON.stringify(r.creator || ""),
          r.score ?? "",
          r.livecodebench ?? "",
          r.terminalbench ?? "",
          r.pin.toFixed(4),
          r.pout.toFixed(4),
          r.blended.toFixed(4),
          r.value ? r.value.toFixed(1) : "",
          r.tps ? Math.round(r.tps) : "",
          r.ttft ? r.ttft.toFixed(2) : "",
          r.ctx ?? "",
          JSON.stringify(r.src || "")
        ];
        lines.push(row.join(","));
      });

      const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `best_models_${new Date().toISOString().slice(0,10)}.csv`;
      link.click();
    }

    // Event listeners
    ["metric", "creatorFilter", "minScore", "maxPrice", "age", "search", "includeBatch"].forEach(id => {
      $(id).addEventListener("input", render);
      $(id).addEventListener("change", render);
    });

    $("btnRefresh").onclick = () => loadData(true);
    $("btnExport").onclick = exportCSV;

    loadData(false);
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP server handler for the dashboard and API endpoints."""

    def do_GET(self):
        if self.path.startswith("/api/refresh"):
            payload = json.dumps(get_data(force=True))
            self.respond_with(payload, "application/json; charset=utf-8")
        elif self.path.startswith("/api/data"):
            payload = json.dumps(get_data(force=False))
            self.respond_with(payload, "application/json; charset=utf-8")
        else:
            self.respond_with(HTML_PAGE, "text/html; charset=utf-8")

    def respond_with(self, content: str, content_type: str):
        raw_bytes = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.end_headers()
        self.wfile.write(raw_bytes)

    def log_message(self, format, *args):
        # Suppress routine GET logging to keep console clean
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Best Model Today - LLM Coding Intelligence & Price Dashboard")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port to bind to (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    parser.add_argument("--api-key", type=str, default=None, help="Artificial Analysis API key override")
    return parser.parse_args()


def main():
    args = parse_args()
    port = args.port

    if args.api_key:
        os.environ["AA_API_KEY"] = args.api_key

    print("=" * 60)
    print(f" Best Model Today (Coding Edition) running at http://localhost:{port}")
    print("=" * 60)
    print(" Press Ctrl+C in terminal to stop.")

    if not args.no_browser and not os.environ.get("NO_BROWSER", "").lower() in ("1", "true"):
        threading.Timer(0.8, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully.")
        server.server_close()


if __name__ == "__main__":
    main()
