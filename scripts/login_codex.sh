#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found in PATH." >&2
  exit 1
fi

CODEX_HOME_DIR="${WORKSPACE_ROOT}/.mini_loop/codex_home"
mkdir -p "$CODEX_HOME_DIR"

echo "Using CODEX_HOME=${CODEX_HOME_DIR}"
CODEX_HOME="$CODEX_HOME_DIR" codex login --device-auth
CODEX_HOME="$CODEX_HOME_DIR" codex login status
