#!/usr/bin/env bash
# Download the fastText language-id model lid.176.ftz and check its sha256.
# The Containerfile adds the same file to the image; the evaluation needs it on the host.
#
# Usage: scripts/fetch-lid-model.sh [destination]   (default: .cache/lid.176.ftz)
set -euo pipefail

url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
sha256="8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
dest="${1:-.cache/lid.176.ftz}"

if [[ -f "${dest}" ]] && echo "${sha256}  ${dest}" | sha256sum -c --quiet 2>/dev/null; then
  echo "${dest}: already present"
  exit 0
fi
mkdir -p "$(dirname "${dest}")"
curl -fsSL -o "${dest}.tmp" "${url}"
echo "${sha256}  ${dest}.tmp" | sha256sum -c --quiet
mv "${dest}.tmp" "${dest}"
echo "${dest}: downloaded and checked"
