#!/usr/bin/env python3
"""
Crawl Regression Test Matrix.

Evaluates a crawl output against a baseline and prints + saves a report
covering five checks:

    1. browser_session_success     pass/fail   site opened and login worked
    2. main_structure_coverage     pass/fail   all must-have journeys present
    3. site_hierarchy_correctness  score-based found / total >= threshold
    4. navigation_flow_coverage    pass/fail   all required nav flows present
    5. latency                     pass/fail   crawl time under threshold

Usage:
    python run_test_matrix.py \
        --crawl    crawls/<site>.json \
        --baseline baselines/baseline_<site>.json \
        --run      crawls/<site>_run.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# Defaults used when the baseline omits the field.
DEFAULT_HIERARCHY_THRESHOLD = 0.6
DEFAULT_LATENCY_THRESHOLD_MS = 60_000


# ---------- I/O ----------

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Crawl-output extraction ----------

def collect_site_structure_names(site_structure: dict) -> set:
    """Walk site_structure.root recursively and return the set of all node names."""
    names: set = set()

    def walk(node):
        if not isinstance(node, dict):
            return
        name = node.get("name")
        if name:
            names.add(name)
        for child in node.get("children", []) or []:
            walk(child)

    walk(site_structure.get("root", {}))
    return names


def collect_navigation_flow_names(navigation_flows: list) -> set:
    return {f.get("name") for f in (navigation_flows or []) if f.get("name")}


def name_in_set(name: str, names: set) -> bool:
    """Case-insensitive membership check."""
    if not name:
        return False
    if name in names:
        return True
    lowered = {n.lower() for n in names}
    return name.lower() in lowered


# ---------- The five checks ----------

def check_browser_session(crawl: dict, run: dict) -> dict:
    """
    Pass if the site was opened and login worked (if required).

    Signals:
      - run.crawl_run.status == "completed"
      - run.crawl_run.site_structure_all_pages_authenticated truthy
      - crawl.metadata.page_count > 0
    """
    metadata = crawl.get("metadata", {}) or {}
    page_count = metadata.get("page_count", 0) or 0
    site_opened = page_count > 0

    crawl_run = (run or {}).get("crawl_run", {}) or {}
    status_completed = crawl_run.get("status") == "completed"
    auth_ok = crawl_run.get("site_structure_all_pages_authenticated")
    # If the run doesn't tell us about auth, fall back to crawl metadata.
    if auth_ok is None:
        auth_ok = metadata.get("authenticated", True)

    passed = site_opened and status_completed and bool(auth_ok)

    reasons = []
    if not site_opened:
        reasons.append("no pages captured")
    if not status_completed:
        reasons.append(f"crawl status={crawl_run.get('status', 'unknown')!r}")
    if not auth_ok:
        reasons.append("not all pages authenticated")

    return {
        "status": "passed" if passed else "failed",
        "site_opened": site_opened,
        "status_completed": status_completed,
        "all_pages_authenticated": bool(auth_ok),
        "reason": "; ".join(reasons) if reasons else "ok",
    }


def check_main_structure_coverage(must_have_journeys: list, crawl_names_union: set) -> dict:
    """Pass if every must-have journey appears anywhere in the crawl output."""
    found, missing = [], []
    for j in must_have_journeys:
        (found if name_in_set(j, crawl_names_union) else missing).append(j)

    total = len(must_have_journeys)
    score = round(len(found) / total, 4) if total > 0 else 1.0
    return {
        "status": "passed" if not missing else "failed",
        "score": score,
        "found_count": len(found),
        "total": total,
        "found": found,
        "missing": missing,
    }


def check_site_hierarchy_correctness(baseline_journeys: list, crawl_names_union: set,
                                     threshold: float) -> dict:
    """
    Score = found_nodes / total_baseline_nodes.
    Pass if score >= threshold. Threshold is configurable per baseline.
    """
    total = len(baseline_journeys)
    if total == 0:
        # Nothing to score against — treat as a pass with a perfect score.
        return {
            "status": "passed",
            "score": 1.0,
            "threshold": threshold,
            "found_count": 0,
            "total": 0,
            "found": [],
            "missing": [],
        }

    found, missing = [], []
    for journey in baseline_journeys:
        name = journey.get("name") if isinstance(journey, dict) else journey
        if not name:
            continue
        (found if name_in_set(name, crawl_names_union) else missing).append(name)

    score = round(len(found) / total, 4)
    return {
        "status": "passed" if score >= threshold else "failed",
        "score": score,
        "threshold": threshold,
        "found_count": len(found),
        "total": total,
        "found": found,
        "missing": missing,
    }


def check_navigation_flow_coverage(must_have_journeys: list, flow_names: set) -> dict:
    """Pass if every required journey appears as a name in crawl navigation_flows."""
    found, missing = [], []
    for j in must_have_journeys:
        (found if name_in_set(j, flow_names) else missing).append(j)

    total = len(must_have_journeys)
    score = round(len(found) / total, 4) if total > 0 else 1.0
    return {
        "status": "passed" if not missing else "failed",
        "score": score,
        "found_count": len(found),
        "total": total,
        "found": found,
        "missing": missing,
    }


def compute_latency_ms(run: dict) -> int | None:
    """
    Return crawl latency in milliseconds, or None if it can't be determined.

    Tries (in order):
      1. run.latency / run.crawl_run.latency (float = seconds, int = ms)
      2. updated_at - created_at on crawl_run
    """
    if not run:
        return None

    crawl_run = run.get("crawl_run", {}) or {}

    latency = run.get("latency")
    if latency is None:
        latency = crawl_run.get("latency")
    if latency is not None:
        return int(latency * 1000) if isinstance(latency, float) else int(latency)

    created_at = crawl_run.get("created_at")
    updated_at = crawl_run.get("updated_at")
    if not (created_at and updated_at):
        return None

    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    try:
        t1 = datetime.strptime(created_at, fmt).replace(tzinfo=timezone.utc)
        t2 = datetime.strptime(updated_at, fmt).replace(tzinfo=timezone.utc)
        return int((t2 - t1).total_seconds() * 1000)
    except ValueError:
        return None


def check_latency(latency_ms: int | None, threshold_ms: int) -> dict:
    """
    Score is latency_ms / threshold_ms.
    Lower is better; pass when score <= 1.0 (i.e. latency under threshold).
    """
    if latency_ms is None:
        return {
            "status": "failed",
            "score": None,
            "latency_ms": None,
            "threshold_ms": threshold_ms,
            "reason": "could not determine crawl latency from run data",
        }
    score = round(latency_ms / threshold_ms, 4) if threshold_ms > 0 else None
    return {
        "status": "passed" if latency_ms <= threshold_ms else "failed",
        "score": score,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
    }


# ---------- Driver ----------

def run_test_matrix(crawl: dict, baseline: dict, run: dict) -> dict:
    # Baseline fields with sensible defaults for missing keys.
    url = baseline.get("url", "unknown")
    must_have_journeys = baseline.get("must_have_journeys", []) or []
    baseline_journeys = baseline.get("baseline_journeys", []) or []
    hierarchy_threshold = float(baseline.get("hierarchy_threshold", DEFAULT_HIERARCHY_THRESHOLD))
    latency_threshold_ms = int(baseline.get("latency_threshold_ms", DEFAULT_LATENCY_THRESHOLD_MS))

    # Crawl-output name sets used by multiple checks.
    site_names = collect_site_structure_names(crawl.get("site_structure", {}) or {})
    flow_names = collect_navigation_flow_names(crawl.get("navigation_flows", []) or [])
    crawl_names_union = site_names | flow_names

    checks = {
        "browser_session_success":    check_browser_session(crawl, run),
        "main_structure_coverage":    check_main_structure_coverage(must_have_journeys, crawl_names_union),
        "site_hierarchy_correctness": check_site_hierarchy_correctness(baseline_journeys, crawl_names_union, hierarchy_threshold),
        "navigation_flow_coverage":   check_navigation_flow_coverage(must_have_journeys, flow_names),
        "latency":                    check_latency(compute_latency_ms(run), latency_threshold_ms),
    }

    # Overall fails if ANY individual check failed.
    overall = "failed" if any(c["status"] == "failed" for c in checks.values()) else "passed"

    return {
        "url": url,
        "status": overall,
        "summary": {name: c["status"] for name, c in checks.items()},
        "checks": checks,
    }


# ---------- Reporting ----------

def print_report(report: dict) -> None:
    label = {"passed": "PASS", "failed": "FAIL"}
    bar = "=" * 60

    print(bar)
    print(f"TEST REPORT  -  {report['url']}")
    print(bar)
    print(f"Overall: [{label.get(report['status'], '????')}] {report['status'].upper()}")
    print()

    print("Matrix Results:")
    for name, status in report["summary"].items():
        print(f"  [{label.get(status, '????')}] {name}: {status}")
    print()

    checks = report["checks"]

    # Hierarchy detail (always shown — it's the only score-based check).
    h = checks["site_hierarchy_correctness"]
    print(f"  Hierarchy score: {h['score_pct']}% (threshold {h['threshold_pct']}%)"
          f"  [{h['found_count']}/{h['total']}]")

    # Latency detail.
    lat = checks["latency"]
    if lat["latency_ms"] is not None:
        print(f"  Latency: {lat['latency_ms']} ms (threshold {lat['threshold_ms']} ms)")
    else:
        print(f"  Latency: unknown  ({lat.get('reason', '')})")

    # Browser session detail when something is off.
    bs = checks["browser_session_success"]
    if bs["status"] != "passed":
        print(f"  Browser session: {bs['reason']}")

    # Missing items per coverage check.
    for key in ("main_structure_coverage", "navigation_flow_coverage",
                "site_hierarchy_correctness"):
        missing = checks[key].get("missing") or []
        if missing:
            print(f"  Missing ({key}): {missing}")


def slugify_url(url: str) -> str:
    """Turn a URL into a filename-safe slug, e.g. https://foo.bar/baz -> foo.bar_baz."""
    s = url.replace("https://", "").replace("http://", "").rstrip("/")
    return s.replace("/", "_") or "site"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run crawl regression test matrix (browser session, structure, "
                    "hierarchy, nav flows, latency).",
    )
    parser.add_argument("--crawl",    required=True, help="Path to the crawl JSON file")
    parser.add_argument("--baseline", required=True, help="Path to the baseline JSON file")
    parser.add_argument("--run",      required=True, help="Path to the run JSON file")
    args = parser.parse_args()

    # Validate input files up front.
    for label, path in [("crawl", args.crawl), ("baseline", args.baseline), ("run", args.run)]:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(2)

    crawl    = load_json(args.crawl)
    baseline = load_json(args.baseline)
    run      = load_json(args.run)

    report = run_test_matrix(crawl, baseline, run)
    print_report(report)

    # Save the report under reports/, named after the baseline URL.
    Path("reports").mkdir(exist_ok=True)
    output_path = f"reports/{slugify_url(baseline.get('url', 'site'))}_report.json"
    save_json(report, output_path)
    print(f"\nReport saved to: {output_path}")

    # Non-zero exit code on failure so this can be used in CI.
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
