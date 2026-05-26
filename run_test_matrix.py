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


def run_test_matrix(crawl_output: dict, baseline: dict, run_output: dict = None, role: str = "customer") -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set.\n"
            "Add it to your .env file: OPENAI_API_KEY=your_key_here"
        )

    client = OpenAI(api_key=api_key)

    # Extract crawl data from navigation-laptop.json
    metadata         = crawl_output.get("metadata", {})
    navigation_flows = crawl_output.get("navigation_flows", [])
    coverage_summary = crawl_output.get("coverage_summary", {})

    # Extract screenshots from crawl run API response if provided
    # Falls back to navigation-laptop.json if not provided
    if run_output:
        crawl_run        = run_output.get("crawl_run", run_output)
        screenshots      = crawl_run.get("site_structure_llm_screenshot_urls", [])
    else:
        screenshots      = crawl_output.get("site_structure_llm_screenshot_urls", [])

    role_baseline      = get_baseline_role(baseline, role)
    url                = baseline.get("url", "unknown")
    min_journeys       = role_baseline.get("journey_count_range", {}).get("min", 3)
    max_journeys       = role_baseline.get("journey_count_range", {}).get("max", 10)
    must_haves         = role_baseline.get("must_have_journeys", [])
    latency_threshold  = role_baseline.get("latency_threshold_ms", 480000)
    screenshot_enabled = role_baseline.get("screenshot_check", {}).get("enabled", True)
    screenshot_min     = role_baseline.get("screenshot_check", {}).get("min_count", 1)

    prompt = f"""
You are a regression test evaluator for a website crawl system.

Compare the crawl output against the baseline and determine if the website
is still behaving as expected.

Return ONLY a valid JSON object — no markdown, no preamble, no explanation.

---

TARGET WEBSITE: {url}

BASELINE CONFIG:
{json.dumps(role_baseline, indent=2)}

CRAWL METADATA:
{json.dumps(metadata, indent=2)}

NAVIGATION FLOWS FOUND ({len(navigation_flows)} total):
{json.dumps(navigation_flows, indent=2)}

COVERAGE SUMMARY:
{json.dumps(coverage_summary, indent=2)}

SCREENSHOTS FOUND ({len(screenshots)} total):
{json.dumps(screenshots, indent=2)}

---

Return this exact JSON structure:

{{
  "test_id": "<url_slug>_regression",
  "url": "{url}",
  "role": "{role}",
  "status": "passed|failed|warning",
  "summary": {{
    "browser_session_success":    "passed|failed",
    "journey_count_range":        "passed|failed",
    "must_have_journey_coverage": "passed|failed|warning",
    "navigation_flows":           "passed|failed|warning",
    "latency":                    "passed|failed|skipped",
    "screenshots":                "passed|failed|skipped"
  }},
  "details": {{
    "missing_journeys":  [],
    "extra_journeys":    [],
    "new_journey_count": 0,
    "latency_ms":        null,
    "screenshot_count":  0,
    "flow_issues":       [],
    "notes":             ""
  }}
}}

Evaluation rules:

1. browser_session_success:
   - PASS if metadata shows crawl completed (authenticated=true, page_count > 0, or completion summary)
   - FAIL if there are error indicators or crawl clearly did not complete

2. journey_count_range:
   - Count total navigation_flows in the crawl output
   - PASS if count is between {min_journeys} and {max_journeys}
   - FAIL if below {min_journeys} (too few journeys — possible crawl failure)
   - FAIL if above {max_journeys} (too many — possible scope explosion)
   - Set new_journey_count to the actual count

3. must_have_journey_coverage:
   - Must-have journeys to check: {must_haves}
   - Use SEMANTIC matching — "Add item to cart" matches "Add Popcorn Chicken to cart"
   - PASS if all must-haves are clearly represented in the navigation flows
   - WARN if some are similar but not clearly matching
   - FAIL if any must-have is clearly absent
   - List missing ones in missing_journeys

4. navigation_flows:
   - Check that each flow's steps are in a logical order for that website
   - PASS if all flows have sensible step sequences
   - WARN if minor ordering issues but core flows intact
   - FAIL if steps from different flows are clearly mixed up
   - List specific issues in flow_issues

5. latency:
   - Threshold: {latency_threshold}ms
   - Check metadata for any timing information
   - PASS if within threshold, FAIL if over, SKIP if no timing data available

6. screenshots:
   - screenshot_check enabled: {screenshot_enabled}
   - SKIP if screenshot_check.enabled is false
   - PASS if screenshots array has {screenshot_min} or more entries
   - FAIL if the array is empty or missing
   - Set screenshot_count to the actual number of screenshots found

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
    parser.add_argument("--run",      default=None,       help="Path to crawl run API response JSON (for screenshots)")
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
        print(f"Loading run:      not provided (screenshots will be skipped or read from crawl)")
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
    for check, result in report["summary"].items():
        print(f"  {icons.get(result, '?')} {check}: {result}")

    details = report["details"]
    print(f"\nJourneys found:    {details['new_journey_count']}")
    print(f"Screenshots found: {details.get('screenshot_count', 0)}")
    if details.get("missing_journeys"):
        print(f"Missing:           {details['missing_journeys']}")
    if details.get("flow_issues"):
        print(f"Flow issues:       {details['flow_issues']}")
    if details.get("notes"):
        print(f"Notes:             {details['notes']}")

    print(f"\nEvaluation time: {elapsed}ms")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()