#!/usr/bin/env bash
# Self-test for .githooks/pre-commit. Builds a throwaway git repo, stages bad and good files,
# and checks the hook blocks or passes as expected. Touches nothing in the real repo.
# Usage:  bash tests/test_precommit.sh        (exit 0 if every case behaves)
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
cd "$T" && git init -q . && git config user.email t@t && git config user.name t
mkdir -p .githooks scripts && cp "$SRC/.githooks/pre-commit" .githooks/ && cp "$SRC/scripts/quality_score.py" scripts/
printf '"""header"""\nx = 1\n' > ok.py; git add ok.py; git commit -qm base --no-verify
PASS=0; FAILN=0
expect() {  # expect <blocked|passes> <label> ; runs hook on current index
  bash .githooks/pre-commit >/dev/null 2>&1; rc=$?
  if { [ "$1" = blocked ] && [ $rc -ne 0 ]; } || { [ "$1" = passes ] && [ $rc -eq 0 ]; }; then PASS=$((PASS+1)); echo "ok   $2"
  else FAILN=$((FAILN+1)); echo "FAIL $2 (rc=$rc, expected $1)"; fi
  git reset -q; rm -f bad* good* Data -r 2>/dev/null
}
printf 'def f(:\n' > bad_syntax.py;                         git add bad_syntax.py;   expect blocked "syntax error"
printf '"""h"""\napi_%s = "abcdefghijklmnop1234"\n' key > bad_key.py; git add bad_key.py; expect blocked "API key literal"
printf 'a,b\n1,2\n' > bad_data.csv;                         git add bad_data.csv;    expect blocked "csv staged"
mkdir -p Data; echo x > Data/f.txt;                         git add Data/f.txt;      expect blocked "Data/ staged"
printf '"""h"""\np = "/Users/someone/Dropbox/x"\n' > bad_path.py; git add bad_path.py; expect blocked "hardcoded user path"
printf '"""h"""\nimport os\nroot = os.environ.get("X", "")\n' > good.py;  git add good.py; expect passes "clean file"
printf 'import random\nimport matplotlib.pyplot as plt\nv = random.random()\nfig = plt.figure()\nfig.savefig("a.png")\nfig.savefig("b.png")\nfig.savefig("c.png")\n' > bad_score.py; git add bad_score.py
bash .githooks/pre-commit >/dev/null 2>&1 && { FAILN=$((FAILN+1)); echo "FAIL low score should block"; } || { PASS=$((PASS+1)); echo "ok   score below 80 blocks"; }
SKIP_QUALITY_GATE=1 bash .githooks/pre-commit >/dev/null 2>&1 && { PASS=$((PASS+1)); echo "ok   SKIP_QUALITY_GATE skips the score only"; } || { FAILN=$((FAILN+1)); echo "FAIL SKIP_QUALITY_GATE"; }
echo "$PASS passed, $FAILN failed"; [ $FAILN -eq 0 ]
