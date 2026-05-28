#!/usr/bin/env python3
"""
Crawl Regression Test Matrix — see README for setup and usage.
"""

import json
import time
import os
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_baseline_role(baseline: dict, role: str = "customer") -> dict:
    if role in baseline:
        return baseline[role]
    return baseline


def score_to_label(score: float, threshold: float = 0.8) -> str:
    return "passed" if score >= threshold else "failed"


def exact_match_flows(must_have_flows: list, navigation_flows: list) -> tuple:
    """
    Try exact string matching first.
    Returns (found, unmatched_baseline, unmatched_crawl_names)
    """
    crawl_flow_names = [f.get("name", "") for f in navigation_flows]
    crawl_flow_names_lower = [n.lower() for n in crawl_flow_names]

    found_exact   = []
    need_semantic = []

    for flow in must_have_flows:
        if flow in crawl_flow_names:
            found_exact.append(flow)
        elif flow.lower() in crawl_flow_names_lower:
            found_exact.append(flow)
        else:
            need_semantic.append(flow)

    return found_exact, need_semantic, crawl_flow_names


def exact_match_pages(must_have_pages: list, site_structure: dict) -> tuple:
    """
    Extract all page/feature names from site_structure and exact match.
    Returns (found, unmatched)
    """
    def extract_names(node, names=None):
        if names is None:
            names = set()
        if isinstance(node, dict):
            name = node.get("name", "")
            if name:
                names.add(name)
                names.add(name.lower())
            for child in node.get("children", []):
                extract_names(child, names)
        return names

    all_names = extract_names(site_structure.get("root", {}))

    found_exact   = []
    need_semantic = []

    for page in must_have_pages:
        if page in all_names or page.lower() in all_names:
            found_exact.append(page)
        else:
            need_semantic.append(page)

    return found_exact, need_semantic


def run_test_matrix(crawl_output: dict, baseline: dict, run_output: dict = None, role: str = "customer") -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set.\n"
            "Add it to your .env file: OPENAI_API_KEY=your_key_here"
        )

    client = OpenAI(api_key=api_key)

    # Extract crawl data
    metadata         = crawl_output.get("metadata", {})
    navigation_flows = crawl_output.get("navigation_flows", [])
    site_structure   = crawl_output.get("site_structure", {})
    coverage_summary = crawl_output.get("coverage_summary", {})

    # Extract auth and latency from run output
    crawl_run                       = run_output.get("crawl_run", run_output) if run_output else {}
    created_at                      = crawl_run.get("created_at")
    updated_at                      = crawl_run.get("updated_at")
    latency_from_run                = crawl_run.get("latency")
    all_pages_authenticated_from_run = crawl_run.get("site_structure_all_pages_authenticated")

    # Extract baseline config
    role_baseline            = get_baseline_role(baseline, role)
    url                      = baseline.get("url", "unknown")
    must_have_pages          = role_baseline.get("must_have_pages", [])
    expected_page_count      = role_baseline.get("expected_page_count", len(must_have_pages))
    expected_hierarchy       = role_baseline.get("expected_hierarchy", {})
    expected_hierarchy_nodes = role_baseline.get("expected_hierarchy_node_count", 0)
    expected_depth           = role_baseline.get("expected_depth", 2)
    must_have_flows          = role_baseline.get("must_have_flows", [])
    expected_flow_count      = role_baseline.get("expected_flow_count", len(must_have_flows))
    latency_threshold        = role_baseline.get("latency_threshold_ms", 480000)
    latency_warning_ms       = 900000  # 15 minutes

    # Flatten expected hierarchy nodes
    all_expected_hierarchy_nodes = []
    for parent, children in expected_hierarchy.items():
        for child in children:
            all_expected_hierarchy_nodes.append(f"{parent} → {child}")

    # ── Step 1: Python exact matching ──
    exact_found_flows, semantic_flows, crawl_flow_names = exact_match_flows(must_have_flows, navigation_flows)
    exact_found_pages, semantic_pages                   = exact_match_pages(must_have_pages, site_structure)

    # ── Step 2: GPT handles only unmatched items + hierarchy + browser session ──
    prompt = f"""
You are a regression test evaluator for a website crawl system.

Python has already done exact string matching. Your job is ONLY to:
1. Check browser session success
2. Semantically match the REMAINING unmatched items
3. Check hierarchy nodes
4. Check authentication

Return ONLY a valid JSON object — no markdown, no preamble, no explanation.

---

TARGET WEBSITE: {url}

CRAWL METADATA:
{json.dumps(metadata, indent=2)}

SITE STRUCTURE:
{json.dumps(site_structure, indent=2)}

NAVIGATION FLOWS ({len(navigation_flows)} total):
{json.dumps(navigation_flows, indent=2)}

COVERAGE SUMMARY:
{json.dumps(coverage_summary, indent=2)}

---

ALREADY MATCHED BY PYTHON (do not re-check these):
- Flows already found: {exact_found_flows}
- Pages already found: {exact_found_pages}

YOUR TASKS:

1. browser_session_success:
   - "passed" if metadata shows authenticated=true, page_count > 0, or summary indicates completion
   - "failed" if there are clear error indicators

2. all_pages_authenticated:
   - true if all pages were accessed authenticated
   - false if any page shows login prompts

3. Semantically match ONLY these remaining unmatched flows (empty list = nothing to check):
   Unmatched flows: {semantic_flows}
   Crawl flow names available: {crawl_flow_names}
   - semantic_found_flows: which of the unmatched flows ARE semantically present
   - semantic_missing_flows: which are NOT present

4. Semantically match ONLY these remaining unmatched pages:
   Unmatched pages: {semantic_pages}
   - semantic_found_pages: which ARE present in site_structure
   - semantic_missing_pages: which are NOT present

5. Hierarchy nodes — check ALL of these:
   {all_expected_hierarchy_nodes}
   - found_hierarchy_nodes: list of "Parent → Child" entries that ARE correctly placed
   - missing_hierarchy_nodes: list that are NOT found or misplaced

6. actual_depth: deepest nesting level in site_structure as a number

7. unauthenticated_pages: list any pages that appear unauthenticated

Return this exact JSON:
{{
  "browser_session_success": "passed|failed",
  "all_pages_authenticated": true,
  "unauthenticated_pages": [],
  "semantic_found_flows": [],
  "semantic_missing_flows": [],
  "semantic_found_pages": [],
  "semantic_missing_pages": [],
  "found_hierarchy_nodes": [],
  "missing_hierarchy_nodes": [],
  "actual_depth": 0
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    raw  = response.choices[0].message.content
    data = json.loads(raw)

    # ── Step 3: Combine Python exact + GPT semantic results ──

    # Flows
    semantic_found_flows   = data.get("semantic_found_flows", [])
    semantic_missing_flows = data.get("semantic_missing_flows", semantic_flows)
    found_flows            = exact_found_flows + semantic_found_flows
    missing_flows          = semantic_missing_flows
    flows_found            = len(found_flows)

    # Pages
    semantic_found_pages   = data.get("semantic_found_pages", [])
    semantic_missing_pages = data.get("semantic_missing_pages", semantic_pages)
    found_pages            = exact_found_pages + semantic_found_pages
    missing_pages          = semantic_missing_pages
    pages_found            = len(found_pages)

    # Hierarchy (GPT only)
    found_hierarchy_nodes   = data.get("found_hierarchy_nodes", [])
    missing_hierarchy_nodes = data.get("missing_hierarchy_nodes", [])
    hierarchy_correct       = len(found_hierarchy_nodes)

    actual_depth = data.get("actual_depth", 0)

    # Calculate scores
    page_score      = round(pages_found / expected_page_count,            2) if expected_page_count      > 0 else 1.0
    hierarchy_score = round(hierarchy_correct / expected_hierarchy_nodes,  2) if expected_hierarchy_nodes > 0 else 1.0
    flow_score      = round(flows_found / expected_flow_count,             2) if expected_flow_count      > 0 else 1.0

    # Apply 80% threshold
    page_label      = score_to_label(page_score)
    hierarchy_label = score_to_label(hierarchy_score)
    flow_label      = score_to_label(flow_score)

    # Browser session
    if all_pages_authenticated_from_run is not None:
        all_pages_authenticated = all_pages_authenticated_from_run
    else:
        all_pages_authenticated = data.get("all_pages_authenticated", True)

    browser_result = data.get("browser_session_success", "failed")
    if not all_pages_authenticated:
        browser_result = "failed"

    # Latency
    if latency_from_run is not None:
        latency_ms = int(latency_from_run * 1000) if isinstance(latency_from_run, float) else latency_from_run
    elif created_at and updated_at:
        from datetime import datetime, timezone
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        try:
            t1 = datetime.strptime(created_at, fmt).replace(tzinfo=timezone.utc)
            t2 = datetime.strptime(updated_at, fmt).replace(tzinfo=timezone.utc)
            latency_ms = int((t2 - t1).total_seconds() * 1000)
        except Exception:
            latency_ms = None
    else:
        latency_ms = None

    if latency_ms is not None:
        if latency_ms <= latency_threshold:
            latency_label = "passed"
        elif latency_ms <= latency_warning_ms:
            latency_label = "warning"
        else:
            latency_label = "failed"
    else:
        latency_label = "skipped"

    # Overall status
    all_results = [browser_result, page_label, hierarchy_label, flow_label, latency_label]
    if "failed" in all_results:
        overall = "failed"
    elif "warning" in all_results:
        overall = "warning"
    else:
        overall = "passed"

    # Build report
    report = {
        "test_id": url.replace("https://", "").replace("/", "_").rstrip("_") + "_regression",
        "url":     url,
        "role":    role,
        "status":  overall,
        "summary": {
            "browser_session_success":    browser_result,
            "main_structure_coverage":    page_label,
            "site_hierarchy_correctness": hierarchy_label,
            "navigation_flow_coverage":   flow_label,
            "latency":                    latency_label
        },
        "scores": {
            "main_structure_coverage": {
                "actual":   pages_found,
                "expected": expected_page_count,
                "score":    page_score,
                "result":   f"{pages_found}/{expected_page_count}"
            },
            "site_hierarchy_correctness": {
                "actual":   hierarchy_correct,
                "expected": expected_hierarchy_nodes,
                "score":    hierarchy_score,
                "result":   f"{hierarchy_correct}/{expected_hierarchy_nodes}"
            },
            "navigation_flow_coverage": {
                "actual":   flows_found,
                "expected": expected_flow_count,
                "score":    flow_score,
                "result":   f"{flows_found}/{expected_flow_count}"
            },
            "latency": {
                "actual_ms":    latency_ms,
                "threshold_ms": latency_threshold,
                "warning_ms":   latency_warning_ms,
                "result":       f"{latency_ms}ms" if latency_ms else "skipped"
            }
        },
        "details": {
            "all_pages_authenticated":  all_pages_authenticated,
            "unauthenticated_pages":    data.get("unauthenticated_pages", []),
            "found_pages":              found_pages,
            "missing_pages":            missing_pages,
            "found_flows":              found_flows,
            "missing_flows":            missing_flows,
            "found_hierarchy_nodes":    found_hierarchy_nodes,
            "missing_hierarchy_nodes":  missing_hierarchy_nodes,
            "expected_depth":           expected_depth,
            "actual_depth":             actual_depth,
            "latency_ms":               latency_ms
        }
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run crawl regression test matrix using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json --run crawls/<site>_run.json
        """
    )
    parser.add_argument("--crawl",    required=True,      help="Path to navigation-laptop.json from Supabase")
    parser.add_argument("--baseline", required=True,      help="Path to baseline JSON")
    parser.add_argument("--run",      default=None,       help="Path to crawl run API response JSON")
    parser.add_argument("--output",   default=None,       help="Path to save report (default: auto-named)")
    parser.add_argument("--role",     default="customer", help="Role to evaluate: customer or admin (default: customer)")
    args = parser.parse_args()

    for label, path in [("Crawl", args.crawl), ("Baseline", args.baseline)]:
        if not Path(path).exists():
            print(f"Error: {label} file not found: {path}")
            exit(1)

    if args.run and not Path(args.run).exists():
        print(f"Error: Run file not found: {args.run}")
        exit(1)

    print(f"Loading crawl:    {args.crawl}")
    print(f"Loading baseline: {args.baseline}")
    if args.run:
        print(f"Loading run:      {args.run}")
    else:
        print(f"Loading run:      not provided (latency and auth will use crawl metadata)")
    print(f"Role:             {args.role}")

    crawl_output = load_json(args.crawl)
    baseline     = load_json(args.baseline)
    run_output   = load_json(args.run) if args.run else None

    print("\nRunning test matrix evaluation...")
    start   = time.time()
    report  = run_test_matrix(crawl_output, baseline, run_output=run_output, role=args.role)
    elapsed = round((time.time() - start) * 1000)
    report["details"]["evaluation_time_ms"] = elapsed

    if args.output:
        output_path = args.output
    else:
        url_slug    = baseline.get("url", "site").replace("https://", "").replace("http://", "").replace("/", "_").rstrip("_")
        output_path = f"reports/{url_slug}_{args.role}_report.json"
        Path("reports").mkdir(exist_ok=True)

    save_json(report, output_path)

    status_icon = {"passed": "✅", "warning": "⚠️ ", "failed": "❌"}.get(report["status"], "?")
    icons       = {"passed": "✅", "warning": "⚠️ ", "failed": "❌", "skipped": "⏭️ "}

    print(f"\n{'='*55}")
    print(f"TEST REPORT — {report['url']} ({args.role})")
    print(f"{'='*55}")
    print(f"Overall Status: {status_icon} {report['status'].upper()}")

    print(f"\nMatrix Results:")
    scores = report.get("scores", {})
    for check, result in report["summary"].items():
        icon       = icons.get(result, "?")
        score_info = scores.get(check, {})
        score_str  = f"  [{score_info.get('result', '')}]" if score_info.get("result") else ""
        print(f"  {icon} {check}: {result}{score_str}")

    details = report["details"]
    print(f"\nAll pages authenticated: {details['all_pages_authenticated']}")
    if details.get("unauthenticated_pages"):
        print(f"Unauthenticated pages:   {details['unauthenticated_pages']}")
    if details.get("missing_pages"):
        print(f"Missing pages:           {details['missing_pages']}")
    if details.get("missing_flows"):
        print(f"Missing flows:           {details['missing_flows']}")
    if details.get("missing_hierarchy_nodes"):
        print(f"Missing hierarchy nodes: {details['missing_hierarchy_nodes']}")
    print(f"Depth:                   expected={details['expected_depth']}, actual={details['actual_depth']}")
    if details.get("latency_ms"):
        print(f"Latency:                 {details['latency_ms']}ms (threshold: 480000ms, warning: 900000ms)")

    print(f"\nEvaluation time: {elapsed}ms")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()