#!/usr/bin/env bash
set -euo pipefail

# Keep original corpora outside Git tracking and pin both sources to audited commits.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$ROOT/raw"
mkdir -p "$RAW"

fetch_sparse() {
  local name="$1" url="$2" revision="$3"
  shift 3
  local checkout="$RAW/$name"
  if [[ ! -d "$checkout/.git" ]]; then
    git clone --filter=blob:none --sparse --depth 1 "$url" "$checkout"
  fi
  git -C "$checkout" fetch --depth 1 origin "$revision"
  git -C "$checkout" checkout --detach "$revision"
  git -C "$checkout" sparse-checkout set "$@"
  [[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]]
  printf '%s %s\n' "$name" "$revision"
}

fetch_sparse crs \
  https://github.com/coreruleset/coreruleset.git \
  af76ccfcd1ad947e38ef9998bdd464f1a39fef86 \
  tests/regression/tests/REQUEST-930-APPLICATION-ATTACK-LFI \
  tests/regression/tests/REQUEST-941-APPLICATION-ATTACK-XSS \
  tests/regression/tests/REQUEST-942-APPLICATION-ATTACK-SQLI \
  tests/regression/tests/REQUEST-932-APPLICATION-ATTACK-RCE

fetch_sparse http-params \
  https://github.com/Morzeux/HttpParamsDataset.git \
  926670a710283f87c05b554680facf3f9530548c \
  payload_full.csv LICENSE README.md

# This public archive is versioned by its SHA-256 in generated/v2/report.json.
if [[ ! -f "$RAW/wcp-legitimate.zip" ]]; then
  curl --fail --location --retry 3 \
    https://downloads.openappsec.io/waf-comparison-project/legitimate.zip \
    --output "$RAW/wcp-legitimate.zip"
fi
