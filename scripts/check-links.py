#!/usr/bin/env python3
"""List relative markdown links that point to files that do not exist.

Usage:  python3 scripts/check-links.py [paths...]     (default: CLAUDE.md MEMORY.md .claude templates)
Exit 1 if any dangling link is found. Links with a URL scheme or a bare #anchor are ignored.
"""
import re
import sys
from pathlib import Path

LINK = re.compile(r"\]\(([^)\s]+)\)")
ROOT = Path(__file__).resolve().parent.parent


def files(targets):
    for t in targets:
        p = ROOT / t
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.md"))


def main(argv):
    targets = argv[1:] or ["CLAUDE.md", "MEMORY.md", ".claude", "templates"]
    bad = 0
    for f in files(targets):
        in_fence = False
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for m in LINK.finditer(line):
                href = m.group(1).split("#")[0]
                if not href or re.match(r"^[a-z]+:", href) or href.startswith("mailto:"):
                    continue
                if not (f.parent / href).resolve().exists():
                    print(f"{f.relative_to(ROOT)}:{n}: dangling -> {href}")
                    bad += 1
    print(f"{bad} dangling link(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
