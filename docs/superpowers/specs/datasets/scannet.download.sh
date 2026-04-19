#!/usr/bin/env bash
# Thin wrapper over the upstream ScanNet download-scannet.py script.
# 1. Obtain a ScanNet access token by emailing scannet@googlegroups.com.
# 2. Drop the provided download-scannet.py into this folder (gitignored).
# 3. Run:
#      bash docs/superpowers/specs/datasets/scannet.download.sh <TOKEN> data/external/scannet
set -euo pipefail

TOKEN="${1:-}"
TARGET="${2:-data/external/scannet}"

if [[ -z "${TOKEN}" ]]; then
    echo "usage: $0 <scannet-token> [target-dir]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER="${SCRIPT_DIR}/download-scannet.py"

if [[ ! -f "${DOWNLOADER}" ]]; then
    echo "missing ${DOWNLOADER}"
    echo "request the ScanNet downloader script from scannet@googlegroups.com,"
    echo "then drop download-scannet.py into ${SCRIPT_DIR}/"
    exit 1
fi

mkdir -p "${TARGET}/scans"
python "${DOWNLOADER}" -o "${TARGET}" --type .sens --type _vh_clean_2.ply --type .aggregation.json --type _vh_clean_2.0.010000.segs.json
echo "done. extract with:"
echo "  python docs/superpowers/specs/datasets/scannet.extract.py --raw-root ${TARGET}/scans --output-dir ${TARGET}/rendered"
