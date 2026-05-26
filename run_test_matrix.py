#!/usr/bin/env python3
"""
Crawl Regression Test Matrix
Uses OpenAI GPT API to evaluate crawl output against a baseline.

Setup:
  1. Get an API key at https://platform.openai.com
  2. pip install openai python-dotenv
  3. Create a .env file with: OPENAI_API_KEY=your_key_here

Usage:
  python run_test_matrix.py --crawl <crawl.json> --baseline <baseline.json> [--run <run.json>] [--output <report.json>] [--role customer|admin]

Examples:
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json --run crawls/<site>_run.json
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json --run crawls/<site>_run.json --role admin
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


def score_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "passed"
    elif score >= threshold * 0.6:
        return "warning"
    else:
        return "failed"


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

    # Extract timestamps from run output for latency
    crawl_run  = run_output.get("crawl_run", run_output) if run_output else {}
    created_at = crawl_run.get("created_at")
    updated_at = crawl_run.get("updated_at")

    # Extract baseline config
    role_baseline              = get_baseline_role(baseline, role)
    url                        = baseline.get("url", "unknown")
    pass_threshold             = baseline.get("pass_threshold", 0.8)
    must_have_pages            = role_baseline.get("must_have_pages", [])
    expected_page_count        = role_baseline.get("expected_page_count", len(must_have_pages))
    expected_hierarchy         = role_baseline.get("expected_hierarchy", {})
    expected_hierarchy_nodes   = role_baseline.get("expected_hierarchy_node_count", 0)
    expected_depth             = role_baseline.get("expected_depth", 2)
    must_have_flows            = role_baseline.get("must_have_flows", [])
    expected_flow_count        = role_baseline.get("expected_flow_count", len(must_have_flows))
    latency_threshold          = role_baseline.get("latency_threshold_ms", 480000)

    prompt = f"""
You are a regression test evaluator for a website crawl system.

Compare the crawl output against the baseline and return numeric scores
alongside pass/fail results.

Return ONLY a valid JSON object — no markdown, no preamble, no explanation.

---

TARGET WEBSITE: {url}
PASS THRESHOLD: {pass_threshold} (80%)

BASELINE CONFIG:
{json.dumps(role_baseline, indent=2)}

CRAWL METADATA:
{json.dumps(metadata, indent=2)}

SITE STRUCTURE:
{json.dumps(site_structure, indent=2)}

NAVIGATION FLOWS FOUND ({len(navigation_flows)} total):
{json.dumps(navigation_flows, indent=2)}

COVERAGE SUMMARY:
{json.dumps(coverage_summary, indent=2)}

CRAWL TIMESTAMPS:
created_at: {created_at}
updated_at: {updated_at}

---

Return this exact JSON structure:

{{
  "test_id": "<url_slug>_site_structure_regression",
  "url": "{url}",
  "role": "{role}",
  "status": "passed|failed|warning",
  "summary": {{
    "browser_session_success":    "passed|failed",
    "main_structure_coverage":   "passed|failed|warning",
    "site_hierarchy_correctness": "passed|failed|warning",
    "navigation_flow_coverage":  "passed|failed|warning",
    "latency":                   "passed|failed|skipped"
  }},
  "scores": {{
    "main_structure_coverage":   {{ "actual": 0, "expected": {expected_page_count}, "score": 0.0, "result": "X/Y" }},
    "site_hierarchy_correctness": {{ "actual": 0, "expected": {expected_hierarchy_nodes}, "score": 0.0, "result": "X/Y" }},
    "navigation_flow_coverage":  {{ "actual": 0, "expected": {expected_flow_count}, "score": 0.0, "result": "X/Y" }},
    "latency":                   {{ "actual_ms": null, "threshold_ms": {latency_threshold}, "result": "Xms / Yms" }}
  }},
  "details": {{
    "missing_pages":     [],
    "missing_flow":      [],
    "hierarchy_issue":   [],
    "expected_depth":    {expected_depth},
    "actual_depth":      0,
    "latency_ms":        null
  }}
}}

Evaluation rules:

1. browser_session_success:
   - PASS if metadata shows crawl completed (authenticated=true, page_count > 0, or completion summary)
   - FAIL if there are error indicators

2. main_structure_coverage (NUMERIC):
   - Must-have pages: {must_have_pages}
   - Count how many of these pages are found in the site_structure (use semantic matching)
   - actual = number found, expected = {expected_page_count}
   - score = actual / expected
   - PASS if score >= {pass_threshold}, WARNING if >= {pass_threshold * 0.6:.2f}, FAIL otherwise
   - List missing pages in missing_pages

3. site_hierarchy_correctness (NUMERIC):
   - Expected hierarchy: {json.dumps(expected_hierarchy)}
   - Count how many expected hierarchy nodes are correctly placed
   - actual = correctly placed nodes, expected = {expected_hierarchy_nodes}
   - score = actual / expected
   - PASS if score >= {pass_threshold}, WARNING if >= {pass_threshold * 0.6:.2f}, FAIL otherwise
   - List issues in hierarchy_issue

4. navigation_flow_coverage (NUMERIC):
   - Must-have flows: {must_have_flows}
   - Count how many are found using semantic matching
   - actual = flows found, expected = {expected_flow_count}
   - score = actual / expected
   - PASS if score >= {pass_threshold}, WARNING if >= {pass_threshold * 0.6:.2f}, FAIL otherwise
   - List missing flows in missing_flow

5. latency:
   - Threshold: {latency_threshold}ms
   - Calculate from created_at and updated_at timestamps if available
   - PASS if within threshold, FAIL if over, SKIP if timestamps not available
   - Set actual_depth from the deepest level found in site_structure

Overall status:
- "failed"  if ANY item is "failed"
- "warning" if ANY item is "warning" and none are failed
- "passed"  if ALL items are "passed" or "skipped"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Run crawl regression test matrix using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json --run crawls/<site>_run.json
  python run_test_matrix.py --crawl crawls/<site>.json --baseline baselines/baseline_<site>.json --run crawls/<site>_run.json --role admin
        """
    )
    parser.add_argument("--crawl",    required=True,      help="Path to navigation-laptop.json from Supabase")
    parser.add_argument("--baseline", required=True,      help="Path to baseline JSON")
    parser.add_argument("--run",      default=None,       help="Path to crawl run API response JSON (for latency timestamps)")
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
        print(f"Loading run:      not provided (latency will be skipped)")
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

    # Print summary
    status_icon = {"passed": "✅", "warning": "⚠️ ", "failed": "❌"}.get(report["status"], "?")
    icons       = {"passed": "✅", "warning": "⚠️ ", "failed": "❌", "skipped": "⏭️ "}

    print(f"\n{'='*55}")
    print(f"TEST REPORT — {report['url']} ({args.role})")
    print(f"{'='*55}")
    print(f"Overall Status: {status_icon} {report['status'].upper()}")

    print(f"\nMatrix Results:")
    scores = report.get("scores", {})
    for check, result in report["summary"].items():
        icon = icons.get(result, "?")
        score_info = scores.get(check, {})
        score_str  = f"  [{score_info.get('result', '')}]" if score_info.get('result') else ""
        print(f"  {icon} {check}: {result}{score_str}")

    details = report["details"]
    if details.get("missing_pages"):
        print(f"\nMissing pages:    {details['missing_pages']}")
    if details.get("missing_flow"):
        print(f"Missing flows:    {details['missing_flow']}")
    if details.get("hierarchy_issue"):
        print(f"Hierarchy issues: {details['hierarchy_issue']}")
    print(f"Depth:            expected={details['expected_depth']}, actual={details['actual_depth']}")
    if details.get("latency_ms"):
        print(f"Latency:          {details['latency_ms']}ms")

    print(f"\nEvaluation time: {elapsed}ms")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()