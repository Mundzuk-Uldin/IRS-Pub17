#!/usr/bin/env bash
# Fetch the tax year 2025 corpus (~2,000 pages) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for f in p17 p501 p502 i1040gi f1040s1 f1040s1a f1040s2 f1040s3; do
    if [ -s "data/raw/$f.pdf" ]; then
        echo "  have  $f.pdf"
    else
        echo "  fetch $f.pdf"
        curl -fsSL -o "data/raw/$f.pdf" "https://www.irs.gov/pub/irs-pdf/$f.pdf"
    fi
done
