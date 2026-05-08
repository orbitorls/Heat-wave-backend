#!/usr/bin/env bash
# HeatShield AI — Colab bootstrap (เซลล์แรกของทุก notebook)
# First-cell bootstrap for every Colab notebook.
#
# Usage (inside Colab cell):
#   !bash /content/Heat-wave-backend/scripts/colab_bootstrap.sh
# หรือ before the repo is cloned:
#   from google.colab import userdata
#   import os, urllib.request
#   os.environ["GH_PAT"] = userdata.get("GH_PAT")
#   urllib.request.urlretrieve(
#       "https://raw.githubusercontent.com/orbitorls/HeatShield/main/scripts/colab_bootstrap.sh",
#       "/tmp/bootstrap.sh")
#   !bash /tmp/bootstrap.sh
#
# Idempotent: ปลอดภัยถ้ารันซ้ำ — Drive mount, repo clone, symlink และ pip install
# ทุกขั้นจะ skip ถ้าทำไปแล้ว.
# Idempotent: safe to re-run — Drive mount, clone, symlink, and pip install
# all skip if already done.

set -euo pipefail

# --- config (override via env if needed) ------------------------------------
GH_OWNER="${GH_OWNER:-orbitorls}"
GH_REPO="${GH_REPO:-HeatShield}"
GH_BRANCH="${GH_BRANCH:-main}"
REPO_DIR="${REPO_DIR:-/content/Heat-wave-backend}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/heatshield}"

log() { printf '[bootstrap] %s\n' "$*"; }

# --- 1. Check Drive is mounted (must be done in Colab cell, not subprocess) --
if [ ! -d /content/drive/MyDrive ]; then
  log "ERROR: Google Drive not mounted."
  log "Run this in a Python cell first:"
  log "  from google.colab import drive; drive.mount('/content/drive')"
  exit 1
else
  log "Drive mounted OK."
fi

# --- 2. Ensure Drive folder layout -----------------------------------------
mkdir -p "${DRIVE_ROOT}/data/raw"
mkdir -p "${DRIVE_ROOT}/data/cache"
mkdir -p "${DRIVE_ROOT}/models/forecast_v3"
mkdir -p "${DRIVE_ROOT}/logs"
log "Drive layout ready under ${DRIVE_ROOT}"

# --- 3. Clone or update the repo -------------------------------------------
if [ ! -d "${REPO_DIR}/.git" ]; then
  if [ -z "${GH_PAT:-}" ]; then
    log "ERROR: GH_PAT is not set. Add it via Colab Secrets and re-run."
    log "  from google.colab import userdata"
    log "  import os; os.environ['GH_PAT'] = userdata.get('GH_PAT')"
    exit 1
  fi
  log "Cloning ${GH_OWNER}/${GH_REPO} (${GH_BRANCH}) ..."
  git clone --depth 1 --branch "${GH_BRANCH}" \
    "https://${GH_PAT}@github.com/${GH_OWNER}/${GH_REPO}.git" "${REPO_DIR}"
else
  log "Repo present; pulling latest ${GH_BRANCH} ..."
  git -C "${REPO_DIR}" fetch --depth 1 origin "${GH_BRANCH}" || true
  git -C "${REPO_DIR}" checkout "${GH_BRANCH}" || true
  git -C "${REPO_DIR}" pull --ff-only || log "(pull skipped — local changes?)"
fi

# --- 4. Symlink data/ and app/models/ to Drive (persistent across sessions) -
link_dir() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [ -L "${dst}" ]; then
    log "symlink already in place: ${dst} -> $(readlink "${dst}")"
    return
  fi
  if [ -e "${dst}" ]; then
    log "moving existing ${dst} aside (-> ${dst}.bak.$(date +%s))"
    mv "${dst}" "${dst}.bak.$(date +%s)"
  fi
  ln -s "${src}" "${dst}"
  log "symlinked ${dst} -> ${src}"
}

link_dir "${DRIVE_ROOT}/data"             "${REPO_DIR}/data"
link_dir "${DRIVE_ROOT}/models/forecast_v3" "${REPO_DIR}/app/models/forecast_v3"

# --- 5. Install training requirements --------------------------------------
REQ_FILE="${REPO_DIR}/requirements-train.txt"
STAMP_FILE="/content/.heatshield_pip_stamp"
if [ -f "${REQ_FILE}" ]; then
  REQ_HASH="$(sha1sum "${REQ_FILE}" | awk '{print $1}')"
  if [ -f "${STAMP_FILE}" ] && [ "$(cat "${STAMP_FILE}")" = "${REQ_HASH}" ]; then
    log "pip deps already installed for current requirements-train.txt — skipping."
  else
    log "Installing requirements-train.txt ..."
    pip install --quiet -r "${REQ_FILE}"
    echo "${REQ_HASH}" > "${STAMP_FILE}"
  fi
else
  log "WARN: ${REQ_FILE} not found — skipping pip install."
fi

# --- 6. Make the repo importable in subsequent cells ------------------------
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
log "PYTHONPATH=${PYTHONPATH}"

log "Bootstrap complete. cd ${REPO_DIR}"
