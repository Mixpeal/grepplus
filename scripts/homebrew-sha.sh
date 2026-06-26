#!/usr/bin/env bash
# Print the sha256 Homebrew expects for a tagged source tarball.
# Usage: ./scripts/homebrew-sha.sh v0.1.0
set -euo pipefail

tag="${1:?usage: $0 <tag>  e.g. v0.1.0}"
version="${tag#v}"
url="https://github.com/Mixpeal/grepplus/archive/refs/tags/${tag}.tar.gz"

if [[ "${USE_GIT_ARCHIVE:-}" == 1 ]] || ! command -v curl >/dev/null; then
  git archive --format=tar.gz --prefix="grepplus-${version}/" "$tag" | shasum -a 256
else
  curl -fsSL "$url" | shasum -a 256
fi
