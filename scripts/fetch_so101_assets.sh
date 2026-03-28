#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

clone_sparse() {
  local repo_url="$1"
  local destination="$2"
  local sparse_path="$3"

  if [ -e "${destination}/.git/index.lock" ]; then
    echo "Stale git lock detected at ${destination}/.git/index.lock" >&2
    echo "Remove the partial clone or the lock file, then rerun this script." >&2
    exit 1
  fi

  if [ ! -d "${destination}/.git" ]; then
    git clone --filter=blob:none --sparse "${repo_url}" "${destination}"
  fi

  git -C "${destination}" sparse-checkout set "${sparse_path}"
  git -C "${destination}" checkout "$(git -C "${destination}" branch --show-current)"
}

mkdir -p "${EXTERNAL_DIR}"

clone_sparse \
  "https://github.com/google-deepmind/mujoco_menagerie.git" \
  "${EXTERNAL_DIR}/mujoco_menagerie" \
  "robotstudio_so101"

clone_sparse \
  "https://github.com/TheRobotStudio/SO-ARM100.git" \
  "${EXTERNAL_DIR}/SO-ARM100" \
  "Simulation/SO101"

cat <<EOF
Fetched SO-101 assets into:
  ${EXTERNAL_DIR}/mujoco_menagerie/robotstudio_so101
  ${EXTERNAL_DIR}/SO-ARM100/Simulation/SO101

Recommended environment variables:
  export SO101_MENAGERIE_DIR="${EXTERNAL_DIR}/mujoco_menagerie/robotstudio_so101"
  export SO101_TRS_DIR="${EXTERNAL_DIR}/SO-ARM100/Simulation/SO101"
EOF
