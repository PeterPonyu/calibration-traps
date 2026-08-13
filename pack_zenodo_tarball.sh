#!/usr/bin/env bash
# pack_zenodo_tarball.sh — Zenodo-safe source tarball for calibration-traps.
# git archive honors .gitattributes export-ignore (portal/, _site/, .github/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="calibration-traps"
REF=""
OUT=""
FROM_WORKDIR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-workdir) FROM_WORKDIR=1; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      if [[ -z "$REF" && "$1" != --* ]]; then REF="$1"; shift
      else echo "Unknown arg: $1" >&2; exit 2; fi
      ;;
  esac
done

REF="${REF:-HEAD}"
OUT="${OUT:-$REPO_ROOT/${REPO_NAME}-${REF}.tar.gz}"
if [[ "$OUT" == *"-HEAD.tar.gz" ]]; then
  OUT="${OUT%-HEAD.tar.gz}-dryrun.tar.gz"
fi

HARD_EXCLUDES=(
  .git .omc .omx .cursor .claude .codex .github portal _site
  .venv .tox .mypy_cache .pytest_cache .ruff_cache .ipynb_checkpoints
  .idea .vscode .DS_Store __pycache__ .env hud-stdin-cache.json
)

ALLOWLIST_DOT_REGEX='^('"${REPO_NAME}"'/\.gitignore|'"${REPO_NAME}"'/\.gitattributes|'"${REPO_NAME}"'/\.zenodo\.json|'"${REPO_NAME}"'/(.*/)?\.claims(/|$))'

verify_tarball() {
  local tarball="$1"
  local members process_hits
  members="$(tar -tzf "$tarball")"

  process_hits="$(printf '%s\n' "$members" | awk -v RS='\n' '
    {
      n = split($0, a, "/")
      for (i = 1; i <= n; i++) {
        if (a[i] == ".git" || a[i] == ".omc" || a[i] == ".omx" ||
            a[i] == ".cursor" || a[i] == ".claude" || a[i] == ".codex" ||
            a[i] == ".github" || a[i] == "portal" || a[i] == "_site" ||
            a[i] == ".venv" || a[i] == ".tox" ||
            a[i] == ".mypy_cache" || a[i] == ".pytest_cache" ||
            a[i] == ".ruff_cache" || a[i] == ".ipynb_checkpoints" ||
            a[i] == ".idea" || a[i] == ".vscode" || a[i] == ".DS_Store" ||
            a[i] == "__pycache__" || a[i] == ".env" ||
            a[i] == "hud-stdin-cache.json" || a[i] ~ /\.pyc$/) {
          print $0
          next
        }
      }
    }')"

  if [[ -n "$process_hits" ]]; then
    echo "FAIL: process/hidden paths present in tarball:" >&2
    printf '%s\n' "$process_hits" | head -80 >&2
    exit 1
  fi

  local bad
  bad="$(printf '%s\n' "$members" | awk -F/ '
    {
      for (i = 1; i <= NF; i++) if ($i ~ /^\./) { print $0; next }
    }' | grep -Ev "$ALLOWLIST_DOT_REGEX" || true)"
  if [[ -n "$bad" ]]; then
    echo "FAIL: unexpected dot paths:" >&2
    printf '%s\n' "$bad" | head -80 >&2
    exit 1
  fi

  echo "PASS: tarball verified — portal/_site/.github omitted"
}

pack_git_archive() {
  cd "$REPO_ROOT"
  git rev-parse --verify "$REF^{commit}" >/dev/null
  git archive --format=tar.gz --prefix="${REPO_NAME}/" -o "$OUT" "$REF"
  echo "Packed via git archive from $REF → $OUT"
}

pack_workdir() {
  local staging
  staging="$(mktemp -d)"
  rsync -a \
    --exclude='.git/' \
    --exclude='.omc/' \
    --exclude='.omx/' \
    --exclude='.cursor/' \
    --exclude='.claude/' \
    --exclude='.codex/' \
    --exclude='.github/' \
    --exclude='portal/' \
    --exclude='_site/' \
    --exclude='.venv/' \
    --exclude='.tox/' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='.DS_Store' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='*.tar.gz' \
    --exclude='hud-stdin-cache.json' \
    "$REPO_ROOT/" "$staging/${REPO_NAME}/"
  tar -czf "$OUT" -C "$staging" "$REPO_NAME"
  rm -rf "$staging"
  echo "Packed via workdir rsync+tar → $OUT"
}

mkdir -p "$(dirname "$OUT")"
if [[ "$FROM_WORKDIR" -eq 1 ]]; then
  pack_workdir
else
  pack_git_archive
fi

ls -lh "$OUT"
sha256sum "$OUT"
verify_tarball "$OUT"
