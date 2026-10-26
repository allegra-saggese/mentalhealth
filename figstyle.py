"""Shared figure style for all exhibits (Nature / Science family).

Single source of truth for size, fonts, colours and saving; the standards are in
.claude/rules/figure-and-table-standards.md. Change a value here, not in an exhibit script.

Typical use in an exhibit script:
    import figstyle
    figstyle.apply_style()
    fig, ax = figstyle.new_figure("single", height_mm=60)
    ax.plot(...)
    figstyle.panel_label(ax, "a")
    figstyle.save_figure(fig, figs_dir, tagged("4a-designs", "twfe_first_expansion"))   # tagged() from packages.py
"""
from datetime import date
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4
WIDTH_MM = {"single": 89.0, "one-and-half": 120.0, "double": 183.0,
            "slide": 145.0}            # usable width of a 16:9 Beamer frame (cm 14.5); a slide figure is placed at 1:1 scale
SLIDE_HEIGHT_MM = 70.0                 # usable height under the frame title
MAX_HEIGHT_MM = 247.0
PNG_DPI = 600                      # line art; use 300 for photographic images and maps

# Okabe-Ito colour-blind-safe categorical palette (fixed order: same variable, same colour everywhere)
PALETTE = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
CYCLE = [PALETTE[k] for k in ("blue", "vermillion", "green", "orange", "sky", "purple", "black")]
SEQUENTIAL_CMAP = "viridis"        # continuous, perceptually uniform, greyscale-safe
DIVERGING_CMAP = "RdBu_r"          # signed quantities; centre on zero with a TwoSlopeNorm


def apply_style(profile="print"):
    """Set matplotlib rcParams once, at the top of an exhibit script. profile="slide": 8 pt text for 1:1 placement on a slide."""
    if profile not in ("print", "slide"):
        raise ValueError("profile must be 'print' or 'slide'")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "lines.linewidth": 0.8, "lines.markersize": 3,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": PNG_DPI, "savefig.bbox": "standard",
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts in vector output
    })
    if profile == "slide":                              # larger text and marks; the canvas is already slide size
        mpl.rcParams.update({
            "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
            "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 4.2,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        })


def new_slide(nrows=1, ncols=1, height_mm=SLIDE_HEIGHT_MM, **subplot_kw):
    """Slide-size multi-panel figure (145 mm wide); returns (fig, axes) with axes a 2-D array."""
    return new_grid(nrows, ncols, width="slide", height_mm=height_mm, **subplot_kw)


def new_figure(width="single", height_mm=60.0, **subplot_kw):
    """Figure at final print size. `width` is a key of WIDTH_MM; returns (fig, ax)."""
    if width not in WIDTH_MM:
        raise ValueError(f"width must be one of {sorted(WIDTH_MM)}")
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds {MAX_HEIGHT_MM} mm")
    fig, ax = plt.subplots(figsize=(WIDTH_MM[width] / MM_PER_INCH, height_mm / MM_PER_INCH),
                           layout="constrained", **subplot_kw)
    return fig, ax


def new_grid(nrows, ncols, width="double", height_mm=120.0, **subplot_kw):
    """Multi-panel figure at final print size; returns (fig, axes) with axes a 2-D array (nrows x ncols)."""
    if width not in WIDTH_MM:
        raise ValueError(f"width must be one of {sorted(WIDTH_MM)}")
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds {MAX_HEIGHT_MM} mm")
    return plt.subplots(nrows, ncols, figsize=(WIDTH_MM[width] / MM_PER_INCH, height_mm / MM_PER_INCH),
                        layout="constrained", squeeze=False, **subplot_kw)


def panel_label(ax, letter):
    """Bold lower-case panel label (a, b, c) at the top left, outside the axes."""
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def dated(stem):
    """Date-stamped file stem, YYYY-MM-DD_stem, as used for all outputs."""
    return f"{date.today().isoformat()}_{stem}"


def save_figure(fig, directory, stem, png_dpi=PNG_DPI):
    """Save a vector PDF and a PNG side by side; returns the two paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / dated(stem)
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=png_dpi)
    return pdf, png


def save_text_table(df, directory, stem, title, note="", float_format="{:.3f}"):
    """Write a table as an aligned plain-text file, YYYY-MM-DD_stem.txt, with a title above and a note below; returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = df.to_string(index=False, float_format=lambda v: float_format.format(v), na_rep="")
    text = f"{title}\n{'=' * len(title)}\n\n{body}\n" + (f"\n{note.strip()}\n" if note.strip() else "")
    path = directory / f"{dated(stem)}.txt"
    path.write_text(text, encoding="utf-8")
    return path
