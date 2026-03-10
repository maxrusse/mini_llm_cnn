#!/usr/bin/env bash
set -euo pipefail

TIER="medium"
HOURS="8"
SLEEP_SECONDS="5"
SEARCH_SPACE="open"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)
      TIER="$2"
      shift 2
      ;;
    --hours)
      HOURS="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --search-space)
      SEARCH_SPACE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$SEARCH_SPACE" != "open" && "$SEARCH_SPACE" != "limited" ]]; then
  echo "search space must be 'open' or 'limited'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/config.json"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config.json at $CONFIG_PATH" >&2
  exit 1
fi

PYTHON_FOR_CONFIG="${PYTHON_FOR_CONFIG:-python3}"
if ! command -v "$PYTHON_FOR_CONFIG" >/dev/null 2>&1; then
  PYTHON_FOR_CONFIG="python"
fi

BENCH_PY="$("$PYTHON_FOR_CONFIG" - <<'PY' "$CONFIG_PATH" "$REPO_ROOT"
import json, pathlib, sys
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
repo = pathlib.Path(sys.argv[2]).resolve()
raw = pathlib.Path(str(cfg["benchmark_python_exe"]))
path = raw if raw.is_absolute() else (repo / raw).resolve()
print(path)
PY
)"

if [[ ! -f "$BENCH_PY" ]]; then
  echo "Configured benchmark python does not exist: $BENCH_PY" >&2
  exit 1
fi

if [[ "$SEARCH_SPACE" == "limited" ]]; then
  SEARCH_DOC="${REPO_ROOT}/search_space_limited.md"
else
  SEARCH_DOC="${REPO_ROOT}/search_space_open.md"
fi

LOG_DIR="${REPO_ROOT}/logs"
META_DIR="${REPO_ROOT}/.mini_loop"
mkdir -p "$LOG_DIR" "$META_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_LOG="${LOG_DIR}/night_run_launcher_${STAMP}.log"
SESSION_META="${META_DIR}/launcher_session.json"

echo "Repo: ${REPO_ROOT}"
echo "Python: ${BENCH_PY}"
echo "Search space: ${SEARCH_SPACE} (${SEARCH_DOC})"
echo "Log: ${SESSION_LOG}"

cat >"$SESSION_META" <<EOF
{
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tier": "${TIER}",
  "hours": "${HOURS}",
  "sleep_seconds": "${SLEEP_SECONDS}",
  "search_space": "${SEARCH_SPACE}",
  "search_space_doc": "${SEARCH_DOC}",
  "launcher_log": "${SESSION_LOG}",
  "dry_run": $([[ "$DRY_RUN" == "1" ]] && echo "true" || echo "false")
}
EOF

CMD=("$BENCH_PY" "${REPO_ROOT}/run_loop.py" "night-run" "--tier" "$TIER" "--hours" "$HOURS" "--sleep-seconds" "$SLEEP_SECONDS")
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=("--dry-run")
fi

printf '[%s] LAUNCH %q' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CMD[0]}" >"$SESSION_LOG"
for arg in "${CMD[@]:1}"; do
  printf ' %q' "$arg" >>"$SESSION_LOG"
done
printf '\n' >>"$SESSION_LOG"

"${CMD[@]}" 2>&1 | tee -a "$SESSION_LOG"
