#!/usr/bin/env python3
"""Quality score (0-100) for the project's Python files.

Usage:  python3 scripts/quality_score.py FILE.py [FILE.py ...]
Exit 0 if every .py file scores >= 80 (other file types are skipped), exit 1 otherwise.
The deductions mirror the 'script' rows of .claude/rules/quality-gates.md; reviewer-only rows
(domain bugs, estimators re-implemented, merge checks) are not detectable here.
"""
import ast
import re
import sys
from pathlib import Path

THRESHOLD = 80
SECRET = re.compile(r"""(api[_-]?key|token|secret|passwd|password)["']?\s*[:=]\s*["'][A-Za-z0-9_\-]{16,}["']""", re.I)
USER_PATH = re.compile(r"(/Users/[A-Za-z0-9_]+|~/" r"Dropbox|/home/[A-Za-z0-9_]+)")  # split so this file does not trip its own check
RANDOM_USE = re.compile(r"\b(random\.|np\.random\.|numpy\.random\.|default_rng|KFold\(|train_test_split\(|\.sample\(|bootstrap\w*\()", re.I)
SEEDED = re.compile(r"(default_rng\(\s*[^)\s]|\.seed\(|random_state\s*=|\bseed\s*=)")


def score_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    issues = []  # (deduction, message)
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return 0, [(100, f"syntax error line {e.lineno}: {e.msg}")]

    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if SECRET.search(line):
            return 0, [(100, f"secret/API-key literal at line {i}")]

    if path.name != "packages.py":
        hits = [i for i, l in enumerate(lines, 1) if USER_PATH.search(l)]
        if hits:
            issues.append((20, f"hardcoded user path (lines {hits[:5]}); use packages.py"))

    if RANDOM_USE.search(text) and not SEEDED.search(text):
        issues.append((10, "stochastic call with no seed (use default_rng(seed) once at the top)"))

    no_dpi = no_date = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "savefig":
                if not any(k.arg == "dpi" for k in node.keywords):
                    no_dpi += 1
            if name in ("savefig", "to_csv") and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and not re.search(r"\d{4}-\d{2}-\d{2}", a.value):
                    no_date += 1
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append((2, f"bare except at line {node.lineno}"))
        if isinstance(node, ast.ImportFrom) and node.names and node.names[0].name == "*" and node.module != "packages":
            issues.append((2, f"'from {node.module} import *' at line {node.lineno}"))
    if no_dpi:
        issues.append((min(5 * no_dpi, 15), f"{no_dpi} savefig call(s) without dpi="))
    if no_date:
        issues.append((min(3 * no_date, 9), f"{no_date} output path(s) written without a date stamp"))

    if not ast.get_docstring(tree) and not (lines and lines[0].startswith(("#!", "#"))):
        issues.append((3, "missing short standard header (module docstring or leading comment)"))

    long_lines = sum(1 for l in lines if len(l) > 120)
    if long_lines:
        issues.append((min(long_lines, 5), f"{long_lines} line(s) over 120 characters"))

    return max(0, 100 - sum(d for d, _ in issues)), issues


def main(argv):
    files = [Path(a) for a in argv[1:]]
    if not files:
        print(__doc__)
        return 0
    failed = False
    for p in files:
        if p.suffix != ".py":
            print(f"{p}: skipped (not a .py file)")
            continue
        if not p.exists():
            print(f"{p}: not found")
            failed = True
            continue
        s, issues = score_file(p)
        print(f"{p}: {s}/100" + ("" if s >= THRESHOLD else "  BELOW 80"))
        for d, msg in issues:
            print(f"    -{d}  {msg}")
        failed |= s < THRESHOLD
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
