#!/usr/bin/env bash
# Install the version-controlled git hooks (run once per clone).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[ -d .githooks ] || { echo "install-hooks: .githooks/ not found" >&2; exit 1; }
chmod +x .githooks/*
git config core.hooksPath .githooks
echo "core.hooksPath -> .githooks"
echo "Bypass the score once:  SKIP_QUALITY_GATE=1 git commit ...   (hard checks still run)"
echo "Bypass everything:      git commit --no-verify ..."
echo "Uninstall:              git config --unset core.hooksPath"
