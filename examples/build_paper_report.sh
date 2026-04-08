
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tables}"
SUMMARY_MD="${SUMMARY_MD:-docs/paper_results.md}"
MANIFEST_PATH="${MANIFEST_PATH:-outputs/tables/paper_report_manifest.json}"

PYTHONPATH=src "$PYTHON_BIN" scripts/analysis/build_paper_report.py   --output-dir "$OUTPUT_DIR"   --summary-md "$SUMMARY_MD"   --manifest-path "$MANIFEST_PATH"
