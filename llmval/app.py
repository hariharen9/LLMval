#!/usr/bin/env python3
"""LLMVal: Comprehensive LLM intelligence, coding benchmarks & live pricing dashboard backend."""

import argparse
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Enable UTF-8 encoding on Windows terminal consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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

DATA_DIR = Path(os.environ.get("LLMVAL_DATA_DIR", Path.cwd()))
CACHE_FILE = DATA_DIR / "cache.json"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"
PKG_BENCHMARKS_FILE = Path(__file__).parent / "data" / "benchmarks.json"

WEB_DIR = Path(__file__).parent / "web"
INDEX_HTML = WEB_DIR / "index.html"

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


def get_ssl_context(fallback_unverified: bool = False) -> ssl.SSLContext:
    """Creates a robust SSL context, with certifi fallback or unverified fallback for macOS/proxies."""
    if fallback_unverified:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # Try certifi if installed in the environment
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass

    # Standard default SSL context
    try:
        return ssl.create_default_context()
    except Exception:
        ctx = ssl._create_unverified_context() if hasattr(ssl, "_create_unverified_context") else ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def http_get(url: str, headers: dict = None, timeout: int = 25):
    """Performs an HTTP GET request with standard headers and returns decoded JSON, with SSL verification fallback."""
    default_headers = {
        "User-Agent": "llmval/1.0.2",
        "Accept": "application/json",
    }
    if headers:
        default_headers.update(headers)
    req = urllib.request.Request(url, headers=default_headers)
    
    ctx = get_ssl_context(fallback_unverified=False)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        err_msg = str(exc).lower()
        if "certificate" in err_msg or "ssl" in err_msg or "verify failed" in err_msg:
            fallback_ctx = get_ssl_context(fallback_unverified=True)
            with urllib.request.urlopen(req, timeout=timeout, context=fallback_ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        raise


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

    # If it's already a pre-parsed or cached item
    if "evaluations" not in item:
        creator_name = extract_creator_name(item.get("model_creator") or item.get("creator"))
        return {
            "name": item.get("name", ""),
            "slug": item.get("slug", ""),
            "creator": creator_name,
            "release_date": item.get("release_date") or "",
            "coding": item.get("coding"),
            "intel": item.get("intel"),
            "agentic": item.get("agentic"),
            "math": item.get("math"),
            "livecodebench": item.get("livecodebench"),
            "terminalbench": item.get("terminalbench"),
            "scicode": item.get("scicode"),
            "gpqa": item.get("gpqa"),
            "hle": item.get("hle"),
            "ifbench": item.get("ifbench"),
            "aime": item.get("aime"),
            "tps": item.get("tps"),
            "ttft": item.get("ttft"),
            "ttfa": item.get("ttfa"),
        }

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
    if not bench_file_data and PKG_BENCHMARKS_FILE.exists():
        bench_file_data = load_json(PKG_BENCHMARKS_FILE, None)
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


def get_web_page() -> str:
    """Loads the Web UI HTML from the separated static assets directory."""
    candidate_paths = [
        INDEX_HTML,
        Path(__file__).parent.parent / "index.html",
        Path.cwd() / "llmval" / "web" / "index.html",
        Path.cwd() / "index.html",
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    return "<!DOCTYPE html><html><body><h1>LLMVal: UI asset not found</h1></body></html>"


class RequestHandler(BaseHTTPRequestHandler):
    """HTTP server handler for the dashboard and API endpoints."""

    def do_GET(self):
        url_path = self.path.split("?")[0]
        if url_path == "/api/refresh":
            payload = json.dumps(get_data(force=True))
            self.respond_with(payload, "application/json; charset=utf-8")
        elif url_path == "/api/data":
            payload = json.dumps(get_data(force=False))
            self.respond_with(payload, "application/json; charset=utf-8")
        else:
            self.respond_with(get_web_page(), "text/html; charset=utf-8")

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


def print_terminal_dashboard(data: dict, port: int, max_age_days: int = 90):
    """Renders a sleek, professional terminal CLI summary of today's best models."""
    now_ts = time.time()
    rows = data.get("rows", [])

    # Filter: scored models released in last 90 days, excluding batch variants by default
    filtered = []
    for r in rows:
        score = r.get("coding") or r.get("intel")
        if score is None:
            continue
        created = r.get("created")
        age_days = (now_ts - created) / 86400 if created else 9999
        if age_days > max_age_days:
            continue
        if r.get("variant") == "batch" or r.get("id", "").endswith(":batch"):
            continue
        blended = r.get("blended", 0)
        value = (score * score) / blended if (blended > 0 and not r.get("free")) else (score * score * 1000 if r.get("free") else 0)
        filtered.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "creator": r.get("creator") or "Unknown",
            "score": score,
            "coding": r.get("coding"),
            "intel": r.get("intel"),
            "pin": r.get("pin", 0),
            "pout": r.get("pout", 0),
            "blended": blended,
            "free": r.get("free", False),
            "value": value,
            "tps": r.get("tps"),
            "ttft": r.get("ttft"),
            "ctx": r.get("ctx"),
        })

    # Find Top Picks
    paid = [r for r in filtered if not r["free"]]
    best_value = max(paid, key=lambda x: x["value"]) if paid else None
    
    budget_models = [r for r in paid if r["blended"] <= 2.5]
    smartest_budget = max(budget_models, key=lambda x: x["score"]) if budget_models else (max(paid, key=lambda x: x["score"]) if paid else None)
    
    speed_models = [r for r in filtered if r["tps"]]
    fastest = max(speed_models, key=lambda x: x["tps"]) if speed_models else None
    
    free_models = [r for r in filtered if r["free"]]
    best_free = max(free_models, key=lambda x: x["score"]) if free_models else None

    # Top 5 ranked
    top_5 = sorted(paid, key=lambda x: x["value"], reverse=True)[:5] if paid else []

    # ANSI Colors
    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    MAGENTA = "\033[1;35m"
    WHITE = "\033[1;37m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    box_w = 76
    t1_plain = "LLMVAL · Developer & LLM Intelligence / Pricing Index"
    p1 = max(0, box_w - 2 - len(t1_plain))
    t2_plain = "Created by Hariharen"
    p2 = max(0, box_w - 2 - len(t2_plain))

    print("\n" + f"{CYAN}╔{'═' * box_w}╗{RESET}")
    print(f"{CYAN}║{RESET}  {BOLD}{WHITE}LLMVAL{RESET} · {YELLOW}Developer & LLM Intelligence / Pricing Index{RESET}{' ' * p1}{CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  {GRAY}Created by Hariharen{RESET}{' ' * p2}{CYAN}║{RESET}")
    print(f"{CYAN}╚{'═' * box_w}╝{RESET}")
    
    or_age = data.get("or_age")
    or_str = "live" if or_age is None else (f"{int(or_age/60)}m ago" if or_age < 3600 else f"{round(or_age/3600,1)}h ago")
    aa_state = "AA live" if data.get("live_bench") else "seed snapshot"
    
    print(f" {GRAY}Feeds:{RESET} OpenRouter ({or_str}) · {aa_state} · {GRAY}Window:{RESET} Last {max_age_days} days · {GRAY}Tracked:{RESET} {data.get('total', 0)} models ({data.get('scored', 0)} scored)")
    print(f" {GRAY}Web UI:{RESET} {GREEN}http://localhost:{port}{RESET}  {GRAY}(Interactive Leaderboard, Compare & Calculator){RESET}\n")

    print(f" {BOLD}{WHITE}TODAY'S RECOMMENDED PICKS (LAST {max_age_days} DAYS):{RESET}")
    print(f" ─" * 40)
    
    if best_value:
        b_price = f"${best_value['blended']:.2f}/1M" if best_value['blended'] >= 0.05 else f"${best_value['blended']:.3f}/1M"
        print(f"  {YELLOW}🏆 BEST VALUE PICK{RESET}       : {BOLD}{WHITE}{best_value['name']}{RESET} ({best_value['creator']})")
        print(f"     {GRAY}Score:{RESET} {GREEN}{best_value['score']:.1f}{RESET}  {GRAY}Blended:{RESET} {b_price}  {GRAY}Value Index:{RESET} {BOLD}{int(best_value['value']):,}{RESET}")

    if smartest_budget:
        s_price = f"${smartest_budget['blended']:.2f}/1M" if smartest_budget['blended'] >= 0.05 else f"${smartest_budget['blended']:.3f}/1M"
        print(f"  {MAGENTA}🧠 SMARTEST IN BUDGET (<$2.50){RESET}: {BOLD}{WHITE}{smartest_budget['name']}{RESET} ({smartest_budget['creator']})")
        print(f"     {GRAY}Score:{RESET} {GREEN}{smartest_budget['score']:.1f}{RESET}  {GRAY}Blended:{RESET} {s_price}")

    if fastest:
        ttft_str = f"{fastest['ttft']:.2f}s" if fastest.get("ttft") else "—"
        print(f"  {CYAN}⚡ FASTEST GENERATION{RESET}    : {BOLD}{WHITE}{fastest['name']}{RESET}")
        print(f"     {GRAY}Throughput:{RESET} {GREEN}{int(fastest['tps'])} tok/s{RESET}  {GRAY}TTFT:{RESET} {ttft_str}")

    if best_free:
        print(f"  {GREEN}🎁 BEST FREE MODEL{RESET}       : {BOLD}{WHITE}{best_free['name']}{RESET}")
        print(f"     {GRAY}Score:{RESET} {GREEN}{best_free['score']:.1f}{RESET}  {GRAY}Price:{RESET} Free ($0.00)")

    print(f" ─" * 40)
    
    if top_5:
        print(f"\n {BOLD}{WHITE}TOP 5 VALUE LEADERBOARD:{RESET}")
        header = f"  {GRAY}#   {'Model':<34} {'Lab':<14} {'Score':<7} {'Blended':<10} {'Value':<8}{RESET}"
        print(header)
        print(f"  {GRAY}────────────────────────────────────────────────────────────────────────────{RESET}")
        for idx, m in enumerate(top_5, 1):
            name_disp = m['name'][:32] + ".." if len(m['name']) > 34 else m['name']
            lab_disp = m['creator'][:12] + ".." if len(m['creator']) > 14 else m['creator']
            price_disp = f"${m['blended']:.2f}" if m['blended'] >= 0.05 else f"${m['blended']:.3f}"
            val_disp = f"{int(m['value']):,}"
            print(f"  {YELLOW if idx==1 else WHITE}{idx:<3}{RESET} {name_disp:<34} {GRAY}{lab_disp:<14}{RESET} {GREEN}{m['score']:<7.1f}{RESET} {WHITE}{price_disp:<10}{RESET} {BOLD}{val_disp:<8}{RESET}")

    print(f"\n {GRAY}Press {WHITE}Ctrl+C{GRAY} in terminal to shutdown.{RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="LLMVal - LLM Intelligence, Coding & Price Dashboard")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help=f"Port to bind to (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    parser.add_argument("--api-key", type=str, default=None, help="Artificial Analysis API key override")
    return parser.parse_args()


def main():
    args = parse_args()
    port = args.port

    if args.api_key:
        os.environ["AA_API_KEY"] = args.api_key

    # Pre-fetch dataset and print rich CLI dashboard
    data = get_data(force=False)
    print_terminal_dashboard(data, port, max_age_days=90)

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

