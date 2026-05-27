# Crawl Regression Test

Regression test system for crawl-related features. Compares fresh crawl output against a known-good baseline to detect regressions after code, prompt, or model changes.

## Setup

```bash
pip install openai python-dotenv
```

Create a `.env` file with:

```
OPENAI_API_KEY=your_key_here
```

## Input Files

| File                             | What it is             | How to get it                                                         |
| -------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| `crawls/<site>.json`             | Site structure JSON    | Download `navigation-laptop.json` from Supabase after crawl completes |
| `crawls/<site>_run.json`         | Crawl run API response | Save the GET crawl poll response with `> crawls/<site>_run.json`      |
| `baselines/baseline_<site>.json` | Known-good baseline    | Copy `baseline_template.json` and fill in manually                    |

## Usage

```bash
python run_test_matrix.py \
  --crawl crawls/<site>.json \
  --baseline baselines/baseline_<site>.json \
  --run crawls/<site>_run.json
```

## Adding a New Website

1. Run a Doable crawl and download the JSON to `crawls/`
2. Save the crawl run API response to `crawls/<site>_run.json`
3. Copy `baselines/baseline_template.json`, rename it `baseline_<site>.json`, and fill in must-have pages, hierarchy, and flows
4. Run the script

Note: site-specific baselines are gitignored — only `baseline_template.json` is committed.

## Example: After Adding a Site

```
├── baselines/
│   └── baseline_<site>.json         ← gitignored, local only
├── crawls/
│   ├── <site>.json                  ← gitignored, local only
│   └── <site>_run.json              ← gitignored, local only
└── reports/
    └── <site>_customer_report.json  ← gitignored, local only
```
