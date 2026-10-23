#!/usr/bin/env python3
"""Measure exhibit files against the Nature/Science formatting standard.

Usage:  python3 scripts/figure_check.py FILE_OR_DIR [...]
Reports, per PNG: pixel size, dpi metadata, printed width in mm and the nearest standard column width;
per PDF: whether `pdffonts` shows only embedded fonts (if pdffonts is installed);
and whether each PNG has a matching vector PDF. Exit 1 if any measured item fails.
Visual items (labels, units, CI note, palette) are checked by looking, see .claude/skills/figure-check.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

COLUMN_MM = {"single": 89.0, "1.5-column": 120.0, "double": 183.0}
TOL_MM = 3.0
MIN_DPI = 300


def collect(args):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.suffix.lower() in {".png", ".pdf"})
        elif p.exists():
            yield p
        else:
            print(f"{a}: not found")


def check_png(p: Path):
    fails = []
    with Image.open(p) as im:
        w, h = im.size
        dpi = im.info.get("dpi", (None, None))[0]
    line = f"{p.name}: {w}x{h}px"
    if not dpi:
        fails.append("no dpi metadata")
    else:
        mm = w / dpi * 25.4
        name, ref = min(COLUMN_MM.items(), key=lambda kv: abs(kv[1] - mm))
        line += f", {dpi:.0f} dpi, {mm:.0f} mm wide (nearest: {name} {ref:.0f} mm)"
        if dpi < MIN_DPI:
            fails.append(f"dpi {dpi:.0f} < {MIN_DPI}")
        if abs(mm - ref) > TOL_MM and mm > COLUMN_MM["double"] + TOL_MM:
            fails.append(f"wider than double column ({mm:.0f} mm)")
        elif abs(mm - ref) > TOL_MM:
            fails.append(f"width {mm:.0f} mm is not a standard column width")
    if not p.with_suffix(".pdf").exists():
        fails.append("no vector PDF alongside")
    return line, fails


def check_pdf(p: Path):
    if not shutil.which("pdffonts"):
        return f"{p.name}: font embedding not checked (pdffonts not installed)", []
    out = subprocess.run(["pdffonts", str(p)], capture_output=True, text=True).stdout.splitlines()[2:]
    not_emb = [l.split()[0] for l in out if len(l.split()) > 4 and l.split()[-5] == "no"]
    return f"{p.name}: {len(out)} font(s)", ([f"fonts not embedded: {', '.join(not_emb)}"] if not_emb else [])


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 0
    bad = 0
    for p in collect(argv[1:]):
        line, fails = check_png(p) if p.suffix.lower() == ".png" else check_pdf(p)
        print(line)
        for f in fails:
            print(f"    FIX  {f}")
        bad += bool(fails)
    print(f"{bad} exhibit(s) with measured problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
