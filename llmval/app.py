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


def fetch_openrouter_discover_lanes() -> dict:
    """Extracts live discover lanes directly from openrouter.ai/discover ItemList."""
    try:
        req = urllib.request.Request("https://openrouter.ai/discover", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        ctx = get_ssl_context(fallback_unverified=True)
        with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
            html = resp.read().decode("utf-8")
        m = re.search(r'\{"@context":"https://schema\.org","@type":"ItemList","@id":"https://openrouter\.ai/discover#models".*?\]\}', html)
        if m:
            items = json.loads(m.group(0)).get("itemListElement", [])
            return {
                "value_leaders": items[0:5],
                "fastest_models": items[5:10],
                "latest_aliases": items[10:15],
                "free_models": items[15:20],
            }
    except Exception:
        pass
    return {}


def build_dataset(force: bool = False, aa_key: str = None) -> dict:
    """Builds the comprehensive model table with matched benchmark scores, OpenRouter percentiles and pricing."""
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
        "OpenRouter Models",
    )

    if not or_data:
        return {"error": "Could not connect to OpenRouter and no local cache was found. " + " ".join(warns)}

    # 1b. Fetch OpenRouter Authoritative Benchmarks (AA percentiles & weighted prices)
    or_bench_data, _ = get_cached_or_fetch(
        "openrouter_benchmarks",
        AA_TTL,
        lambda: http_get("https://openrouter.ai/api/frontend/v1/rankings/benchmarks").get("data", {}),
        force,
        3600,
        warns,
        "OpenRouter Benchmarks",
    )

    # 1c. Fetch OpenRouter Multi-Provider Performance (real p50 throughput & latency)
    or_perf_data, _ = get_cached_or_fetch(
        "openrouter_performance",
        OR_TTL,
        lambda: http_get("https://openrouter.ai/api/frontend/v1/rankings/performance").get("data", []),
        force,
        60,
        warns,
        "OpenRouter Performance",
    )

    # 1d. Fetch OpenRouter Discovery (weekly token volume totals)
    or_disc_data, _ = get_cached_or_fetch(
        "openrouter_discovery",
        3600,
        lambda: http_get("https://openrouter.ai/api/frontend/v1/rankings/discovery").get("data", {}),
        force,
        600,
        warns,
        "OpenRouter Discovery",
    )

    # 1e. Fetch OpenRouter Discover Lanes (exact live discover ordering)
    or_lanes_raw, _ = get_cached_or_fetch(
        "openrouter_discover_lanes",
        3600,
        fetch_openrouter_discover_lanes,
        force,
        600,
        warns,
        "OpenRouter Discover Lanes",
    )

    # 2. Get Raw Artificial Analysis Benchmarks (1-day limit per run)
    aa_raw_list, aa_age = get_benchmarks_data(force, aa_key, warns)
    aa_list = parse_artificial_analysis_models(aa_raw_list or [])

    # Build lookup index for AA raw data
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

    # Extract OpenRouter benchmark percentiles
    or_bench = or_bench_data or {}
    aa_data_block = or_bench.get("aaData") or {}
    pcts_by_slug = aa_data_block.get("percentilesBySlug") or {}

    pct_lookup = {}
    for k, v in pcts_by_slug.items():
        pct_lookup[normalize_slug(k)] = v
        no_date = re.sub(r"-\d{8}$", "", k)
        if no_date != k:
            pct_lookup[normalize_slug(no_date)] = v

    # Performance lookup by slug and id
    perf_map = {}
    if isinstance(or_perf_data, list):
        for p in or_perf_data:
            if isinstance(p, dict):
                if p.get("slug"):
                    perf_map[p["slug"]] = p
                    perf_map[normalize_slug(p["slug"])] = p
                if p.get("id"):
                    perf_map[p["id"]] = p
                    perf_map[normalize_slug(p["id"])] = p

    # Discovery totals lookup (weekly tokens)
    disc_totals = {}
    if isinstance(or_disc_data, dict):
        disc_totals = or_disc_data.get("totals") or {}

    def get_weekly_tokens(mid: str, base: str) -> int:
        for k in [mid, base]:
            if k in disc_totals:
                return disc_totals[k].get("t", 0)
        norm_b = normalize_slug(base)
        for tk, tv in disc_totals.items():
            if normalize_slug(tk).startswith(norm_b) or norm_b.startswith(normalize_slug(tk)):
                return tv.get("t", 0)
        return 0

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

        # OpenRouter standard: combined rate weighted toward input to match typical usage (75% in, 25% out)
        blended = (3 * pin + pout) / 4.0

        # Percentile matching from OpenRouter aaData
        norm_base = normalize_slug(base)
        norm_mid = normalize_slug(mid)
        p_info = pct_lookup.get(norm_mid) or pct_lookup.get(norm_base)
        if not p_info:
            for pk, pv in pct_lookup.items():
                if len(pk) >= 5 and (pk == norm_base or norm_base.startswith(pk) or pk.startswith(norm_base)):
                    p_info = pv
                    break

        intel_pct = p_info.get("intelligence") if p_info else None
        coding_pct = p_info.get("coding") if p_info else None
        agentic_pct = p_info.get("agentic") if p_info else None
        valid_pcts = [p for p in [intel_pct, coding_pct, agentic_pct] if p is not None and isinstance(p, (int, float))]
        avg_pct = (sum(valid_pcts) / len(valid_pcts)) if valid_pcts else None

        # Performance telemetry matching
        perf_item = perf_map.get(mid) or perf_map.get(base) or perf_map.get(norm_mid) or perf_map.get(norm_base)
        tps = perf_item.get("p50_throughput") if perf_item else None
        ttft = (perf_item.get("p50_latency") / 1000.0) if (perf_item and perf_item.get("p50_latency")) else None
        providers = perf_item.get("provider_count") if perf_item else None

        # Weekly tokens
        weekly_tokens = get_weekly_tokens(mid, base)

        row = {
            "id": mid,
            "name": m.get("name", mid),
            "creator": creator_guess,
            "pin": pin,
            "pout": pout,
            "blended": blended,
            "free": variant == "free" or (pin == 0 and pout == 0),
            "variant": variant,
            "ctx": m.get("context_length"),
            "created": m.get("created"),
            "tools": "tools" in params,
            "vision": "image" in (arch.get("input_modalities") or []),
            "reasoning": "reasoning" in params or "include_reasoning" in params,

            # OpenRouter Percentiles
            "intel_pct": intel_pct,
            "coding_pct": coding_pct,
            "agentic_pct": agentic_pct,
            "avg_pct": avg_pct,
            "weekly_tokens": weekly_tokens,
            "providers": providers,

            # Benchmarks
            "coding": coding_pct,
            "intel": intel_pct,
            "agentic": agentic_pct,
            "math": None,
            "livecodebench": None,
            "terminalbench": None,
            "gpqa": None,
            "hle": None,
            "ifbench": None,
            "aime": None,

            # Latency & Speed
            "tps": tps,
            "ttft": ttft,
            "ttfa": None,
            "src": "OpenRouter / AA",
        }

        # Match extra granular benchmark scores from AA raw or seed
        matched = find_matching_benchmark(base, row["name"], aa_index, aliases) if aa_index else []
        if matched:
            best_match = max(matched, key=lambda x: (x.get("coding") or x.get("intel") or 0))
            row.update({
                "creator": best_match.get("creator") or row["creator"],
                "coding": row["coding"] or best_match.get("coding"),
                "intel": row["intel"] or best_match.get("intel"),
                "agentic": row["agentic"] or best_match.get("agentic"),
                "math": best_match.get("math"),
                "livecodebench": best_match.get("livecodebench"),
                "terminalbench": best_match.get("terminalbench"),
                "gpqa": best_match.get("gpqa"),
                "hle": best_match.get("hle"),
                "ifbench": best_match.get("ifbench"),
                "aime": best_match.get("aime"),
                "tps": row["tps"] or best_match.get("tps"),
                "ttft": row["ttft"] or best_match.get("ttft"),
                "ttfa": best_match.get("ttfa"),
            })
            c_matched = extract_creator_name(best_match.get("creator") or row["creator"])
            row["creator"] = c_matched
            if c_matched:
                creators_set.add(c_matched)
        elif base in local_benchmarks and isinstance(local_benchmarks[base], (int, float)):
            val_override = float(local_benchmarks[base])
            row["coding"] = row["coding"] or val_override
            row["intel"] = row["intel"] or val_override
            row["src"] = "benchmarks.json"
        elif base in SEED:
            seed_data = SEED[base]
            if isinstance(seed_data, dict):
                row["coding"] = row["coding"] or seed_data.get("coding")
                row["intel"] = row["intel"] or seed_data.get("intel")
                row["creator"] = seed_data.get("creator") or row["creator"]
            else:
                row["coding"] = row["coding"] or float(seed_data)
                row["intel"] = row["intel"] or float(seed_data)

        # Capability per dollar value formula
        score_val = row["avg_pct"] if row["avg_pct"] is not None else (row["coding"] or row["intel"])
        row["value"] = (score_val / blended) if (blended > 0 and score_val and score_val > 0) else 0

        rows.append(row)

    scored_count = sum(1 for r in rows if r["coding"] is not None or r["intel"] is not None or r["avg_pct"] is not None)

    # Build OpenRouter Discover Lanes
    lane_data = {}
    if or_lanes_raw and isinstance(or_lanes_raw, dict):
        lane_data = or_lanes_raw

    # Build dynamic OpenRouter Value Leaders (top half qualification: avg_pct >= 50, capability per dollar)
    qualifying_value = [
        r for r in rows
        if (r.get("avg_pct") or 0) >= 50
        and r.get("blended", 0) > 0
        and not r.get("free")
        and "contributor" not in r["id"].lower()
        and not r["id"].endswith("-latest")
    ]
    qualifying_value.sort(key=lambda x: x.get("value", 0), reverse=True)
    seen_bases = set()
    computed_value_leaders = []
    for q in qualifying_value:
        b_key = q["id"] if ":batch" in q["id"] else q["id"].split(":")[0]
        if b_key not in seen_bases:
            seen_bases.add(b_key)
            computed_value_leaders.append(q)

    # Build dynamic OpenRouter Fastest Models (tps >= 40, providers >= 2)
    fast_qualifying = [r for r in rows if r.get("tps") and (r.get("tps") >= 40) and ((r.get("providers") or 0) >= 2) and r.get("ttft") and (r.get("ttft") > 0)]
    fast_qualifying.sort(key=lambda x: x.get("tps", 0), reverse=True)
    seen_fast = set()
    computed_fastest = []
    for f in fast_qualifying:
        b_key = re.sub(r"-\d{8}$", "", f["id"].split(":")[0])
        if b_key not in seen_fast and ":batch" not in f["id"]:
            seen_fast.add(b_key)
            computed_fastest.append(f)

    return {
        "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(rows),
        "scored": scored_count,
        "or_age": or_age,
        "aa_age": aa_age,
        "live_bench": bool(aa_list) or bool(pcts_by_slug),
        "creators": sorted(list(creators_set)),
        "warnings": warns,
        "rows": rows,
        "openrouter_lanes": {
            "discover_items": lane_data,
            "value_leaders": computed_value_leaders[:10],
            "fastest_models": computed_fastest[:10],
        },
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


def format_tokens(n: int) -> str:
    """Formats large token counts into human readable notation (e.g. 14.1T, 46.6B)."""
    if not n or n <= 0:
        return "—"
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{int(n):,}"


def print_terminal_dashboard(data: dict, port: int, max_age_days: int = 90):
    """Renders a sleek, professional terminal CLI summary matching OpenRouter Discover rankings."""
    now_ts = time.time()
    rows = data.get("rows", [])
    lanes = data.get("openrouter_lanes", {})

    # OpenRouter Value Leaders & Fastest Models
    or_value_leaders = lanes.get("value_leaders", [])
    or_fastest = lanes.get("fastest_models", [])

    # Filtered models for general picks
    filtered = []
    for r in rows:
        score = r.get("avg_pct") or r.get("coding") or r.get("intel")
        if score is None:
            continue
        created = r.get("created")
        age_days = (now_ts - created) / 86400 if created else 9999
        if age_days > max_age_days:
            continue
        filtered.append(r)

    paid = [r for r in filtered if not r.get("free") and r.get("blended", 0) > 0]
    free_models = [r for r in filtered if r.get("free")]

    best_free = max(free_models, key=lambda x: (x.get("avg_pct") or x.get("coding") or x.get("intel") or 0)) if free_models else None
    budget_models = [r for r in paid if r.get("blended", 0) <= 2.5]
    smartest_budget = max(budget_models, key=lambda x: (x.get("avg_pct") or x.get("coding") or x.get("intel") or 0)) if budget_models else None

    # ANSI Colors
    CYAN = "\033[1;36m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    MAGENTA = "\033[1;35m"
    WHITE = "\033[1;37m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    box_w = 78
    t1_plain = "LLMVAL · OpenRouter Discover & Artificial Analysis Rankings"
    p1 = max(0, box_w - 2 - len(t1_plain))
    t2_plain = "Synchronized live with openrouter.ai/discover"
    p2 = max(0, box_w - 2 - len(t2_plain))

    print("\n" + f"{CYAN}╔{'═' * box_w}╗{RESET}")
    print(f"{CYAN}║{RESET}  {BOLD}{WHITE}LLMVAL{RESET} · {YELLOW}OpenRouter Discover & Artificial Analysis Rankings{RESET}{' ' * p1}{CYAN}║{RESET}")
    print(f"{CYAN}║{RESET}  {GRAY}Synchronized live with openrouter.ai/discover{RESET}{' ' * p2}{CYAN}║{RESET}")
    print(f"{CYAN}╚{'═' * box_w}╝{RESET}")

    or_age = data.get("or_age")
    or_str = "live" if or_age is None else (f"{int(or_age/60)}m ago" if or_age < 3600 else f"{round(or_age/3600,1)}h ago")

    print(f" {GRAY}Feeds:{RESET} OpenRouter Discover ({or_str}) · {GRAY}Price Weight:{RESET} 75% In / 25% Out · {GRAY}Tracked:{RESET} {data.get('total', 0)} models")
    print(f" {GRAY}Web UI:{RESET} {GREEN}http://localhost:{port}{RESET}  {GRAY}(Interactive Leaderboard, Discover Lanes & Calculator){RESET}\n")

    # 1. VALUE LEADERS LANE (EXACT OPENROUTER DISCOVER)
    print(f" {BOLD}{WHITE}VALUE LEADERS{RESET}  {GRAY}(The most capability per dollar of list price · Top half qualifying){RESET}")
    print(f"  {GRAY}#   {'Model':<30} {'Coding':<8} {'Agentic':<9} {'Tokens':<9} {'Blended $/1M':<14} {'Value'}{RESET}")
    print(f"  {GRAY}──────────────────────────────────────────────────────────────────────────────{RESET}")
    
    # Check discover lane items first or computed
    disc_items = lanes.get("discover_items", {}).get("value_leaders", [])
    display_value_leaders = []
    
    if disc_items:
        for it in disc_items:
            slug = it.get("item", "").replace("https://openrouter.ai/", "")
            row_match = next((r for r in rows if r["id"] == slug or r["id"].split(":")[0] == slug), None)
            if row_match:
                display_value_leaders.append(row_match)
            else:
                display_value_leaders.append({
                    "name": it.get("name"),
                    "coding_pct": None,
                    "agentic_pct": None,
                    "weekly_tokens": 0,
                    "blended": 0,
                    "value": 0,
                })
    if not display_value_leaders:
        display_value_leaders = or_value_leaders[:5]

    for idx, m in enumerate(display_value_leaders[:5], 1):
        name_disp = m["name"][:28] + ".." if len(m["name"]) > 30 else m["name"]
        c_pct = f"{m['coding_pct']}%" if m.get("coding_pct") is not None else "—"
        a_pct = f"{m['agentic_pct']}%" if m.get("agentic_pct") is not None else "—"
        tok_str = format_tokens(m.get("weekly_tokens", 0))
        price_str = f"${m['blended']:.3f}" if m.get("blended") else "—"
        val_str = f"{m['value']:.1f}" if m.get("value") else "—"
        print(f"  {YELLOW if idx==1 else WHITE}{idx:<3}{RESET} {BOLD}{name_disp:<30}{RESET} {CYAN}{c_pct:<8}{RESET} {MAGENTA}{a_pct:<9}{RESET} {GRAY}{tok_str:<9}{RESET} {WHITE}{price_str:<14}{RESET} {GREEN}{val_str}{RESET}")

    print(f"\n {BOLD}{WHITE}FASTEST MODELS{RESET}  {GRAY}(Highest median output speed across multiple providers){RESET}")
    print(f"  {GRAY}#   {'Model':<30} {'Throughput':<14} {'TTFT p50':<12} {'Providers'}{RESET}")
    print(f"  {GRAY}──────────────────────────────────────────────────────────────────────────────{RESET}")
    
    disc_fast_items = lanes.get("discover_items", {}).get("fastest_models", [])
    display_fastest = []
    if disc_fast_items:
        for it in disc_fast_items:
            slug = it.get("item", "").replace("https://openrouter.ai/", "")
            row_match = next((r for r in rows if r["id"] == slug or r["id"].split(":")[0] == slug), None)
            if row_match and row_match.get("tps"):
                display_fastest.append(row_match)
            else:
                display_fastest.append({"name": it.get("name"), "tps": 0, "ttft": 0, "providers": 0})
    if not display_fastest:
        display_fastest = or_fastest[:5]

    for idx, m in enumerate(display_fastest[:5], 1):
        name_disp = m["name"][:28] + ".." if len(m["name"]) > 30 else m["name"]
        tps_str = f"{int(m['tps'])} tok/s" if m.get("tps") else "—"
        ttft_str = f"{int(m['ttft'] * 1000)}ms" if m.get("ttft") else "—"
        prov_str = f"{m['providers']} providers" if m.get("providers") else "—"
        print(f"  {CYAN if idx==1 else WHITE}{idx:<3}{RESET} {BOLD}{name_disp:<30}{RESET} {GREEN}{tps_str:<14}{RESET} {GRAY}{ttft_str:<12}{RESET} {WHITE}{prov_str}{RESET}")

    print(f"\n {BOLD}{WHITE}CURATED HIGHLIGHTS:{RESET}")
    print(f" ─" * 39)
    if smartest_budget:
        s_price = f"${smartest_budget['blended']:.3f}/1M" if smartest_budget['blended'] < 0.1 else f"${smartest_budget['blended']:.2f}/1M"
        sc_val = smartest_budget.get("avg_pct") or smartest_budget.get("coding") or smartest_budget.get("intel")
        print(f"  {MAGENTA}🧠 SMARTEST IN BUDGET (<$2.50){RESET}: {BOLD}{WHITE}{smartest_budget['name']}{RESET} ({smartest_budget['creator']})")
        print(f"     {GRAY}Score/Percentile:{RESET} {GREEN}{sc_val:.1f}{RESET}  {GRAY}Blended:{RESET} {s_price}")
    if best_free:
        f_val = best_free.get("avg_pct") or best_free.get("coding") or best_free.get("intel") or 0
        print(f"  {GREEN}🎁 BEST FREE MODEL{RESET}       : {BOLD}{WHITE}{best_free['name']}{RESET}")
        print(f"     {GRAY}Score/Percentile:{RESET} {GREEN}{f_val:.1f}{RESET}  {GRAY}Price:{RESET} Free ($0.00)")
    print(f" ─" * 39)

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

