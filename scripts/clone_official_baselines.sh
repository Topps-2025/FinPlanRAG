#!/usr/bin/env bash
# Fetch the six official baseline repositories at the commits used in the paper.
# Usage: scripts/clone_official_baselines.sh [target-dir]
set -euo pipefail

ROOT="${1:-vendor/official_baselines}"
mkdir -p "$ROOT"

clone_at() {
  local name="$1" url="$2" commit="$3"
  local dir="$ROOT/$name"
  if [ -d "$dir/.git" ]; then
    echo "skip $name (already present)"
    return
  fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
  echo "$name $(git -C "$dir" rev-parse HEAD)"
}

clone_at findver    https://github.com/yilunzhao/FinDVer.git       e8bb237def4ce555a606a45edba22666e31df248
clone_at finfact    https://github.com/IIT-DM/Fin-Fact.git         b30387813734d4b461e6fcdd576feb819cc0bcbd
clone_at selfrag    https://github.com/AkariAsai/self-rag.git      1fcdc420e48f50a7d7ab1ece5494221b93252e99
clone_at crag       https://github.com/HuskyInSalt/CRAG.git        de7c2961ae624a1483a138c5798e1f6d0c4fb0e0
clone_at flare      https://github.com/jzbjyb/FLARE.git            ec4b06b502b5ab54f3f9236b0112a5d28482e7bb
clone_at adaptiverag https://github.com/starsuzi/Adaptive-RAG.git  0c88670af8707667eb5c1163151bb5ce61b14acb

echo "official baselines are under $ROOT"
