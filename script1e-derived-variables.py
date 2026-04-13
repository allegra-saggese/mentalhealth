#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script1e-derived-variables.py

Computes derived variables for County Health Rankings (CHR) data in the
full merged panel, then re-exports a panel with these columns appended.

Two passes:
  Pass 1 — Rate-only CHR variables (no numerator/denominator):
    Impute a count using: count_imputed = raw_value / 100 * census_population
    These are labelled *_count_imputed to distinguish from directly measured counts.
    Applies to 6 variables where CHR reports only raw_value (a percent or proportion).

  Pass 2 — CHR variables with both numerator and denominator:
    Verify that numerator / denominator approximates raw_value (after unit scaling).
    Flags rows where the implied ratio deviates by > ABS_TOL or > REL_TOL
    as a data-quality check.  No values are modified — flags only.

Output:
  Reads: latest *_full_merged.csv from merged_dir
  Writes: same-date *_full_merged_derived.csv to merged_dir
    (separate file; the original merged panel is unchanged)
"""

from packages import *
from functions import *
import re

# ── Configuration ────────────────────────────────────────────────────────────

merged_dir = os.path.join(db_data, "merged")
SUFFIX = "_mentalhealthrank_full"
POP_COL = "population_population_full"   # US Census pop — 100% coverage

# Tolerance for Pass 2 verification.
# The CHR may use different scale (e.g. per 100 vs per 100,000) depending on
# the variable, so we check on both a percentage scale and rate-per-100k scale.
ABS_TOL = 0.05     # absolute difference in proportion space (5 pp)
REL_TOL = 0.10     # relative difference in proportion space (10%)

# Pass 1: variables with raw_value only (no numerator column).
# raw_value is a proportion 0–1 for population-share variables, or a percent
# 0–100 for health behaviour variables.  We detect the scale automatically.
RATE_ONLY_VARS = [
    "frequent_mental_distress",
    "frequent_physical_distress",
    "insufficient_sleep",
    "access_to_exercise_opportunities",
    "%_american_indian_and_alaskan_native",
    "%_non-hispanic_african_american",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_name(var: str) -> str:
    """Slugify a variable stem for use in column names."""
    return re.sub(r"[^a-z0-9]+", "_", var.lower()).strip("_")


def _raw_col(var: str) -> str:
    return f"{var}_raw_value{SUFFIX}"


def _num_col(var: str) -> str:
    return f"{var}_numerator{SUFFIX}"


def _den_col(var: str) -> str:
    return f"{var}_denominator{SUFFIX}"


def _detect_scale(series: pd.Series) -> float:
    """
    Return 100 if the raw_value series appears to be a proportion (0–1),
    else return 1 (already a percentage 0–100).
    Detection: if median > 1 treat as percentage, else as proportion.
    """
    med = series.dropna().median()
    return 100.0 if med <= 1.0 else 1.0


def _find_latest_merged(merged_dir: str) -> str:
    pattern = os.path.join(merged_dir, "*_full_merged.csv")
    candidates = sorted(glob.glob(pattern))
    # Exclude derived outputs from previous runs
    candidates = [p for p in candidates if "derived" not in os.path.basename(p)]
    if not candidates:
        raise FileNotFoundError(f"No *_full_merged.csv found in {merged_dir}")
    return candidates[-1]


# ── Load panel ───────────────────────────────────────────────────────────────

merged_path = _find_latest_merged(merged_dir)
print(f"Loading: {os.path.basename(merged_path)}")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_panel_key(df, dropna=True)
print(f"Panel shape: {df.shape}")

if POP_COL not in df.columns:
    raise KeyError(f"Population column '{POP_COL}' not found in merged panel.")

pop = pd.to_numeric(df[POP_COL], errors="coerce")


# ── Pass 1: impute counts for rate-only variables ────────────────────────────
print("\n── Pass 1: Rate-only variables → impute counts ─────────────────────")

pass1_results = []
for var in RATE_ONLY_VARS:
    raw_col = _raw_col(var)
    num_col = _num_col(var)

    if raw_col not in df.columns:
        print(f"  SKIP {var}: column '{raw_col}' not in panel")
        continue

    if num_col in df.columns:
        # Has numerator — skip (handle in Pass 2)
        print(f"  SKIP {var}: numerator column exists ('{num_col}'), defer to Pass 2")
        continue

    raw = pd.to_numeric(df[raw_col], errors="coerce")
    scale = _detect_scale(raw)  # 100 if proportion (0-1), 1 if percent (0-100)
    pct = raw * scale           # now in percent-space (0-100)

    out_col = f"{_safe_name(var)}_count_imputed"
    df[out_col] = (pct / 100.0 * pop).round(0).astype("Int64")

    n_valid = df[out_col].notna().sum()
    n_total = len(df)
    print(
        f"  {var}:\n"
        f"    raw scale detected: {'proportion (×100)' if scale == 100 else 'percentage (×1)'}\n"
        f"    output column: {out_col}\n"
        f"    non-null: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)"
    )
    pass1_results.append(out_col)

print(f"\nPass 1 complete. {len(pass1_results)} count_imputed columns added.")


# ── Pass 2: verify numerator / denominator ≈ raw_value ──────────────────────
print("\n── Pass 2: Numerator/denominator verification ───────────────────────")

# Auto-detect all variables that have BOTH numerator and denominator columns.
all_cols = set(df.columns)
num_vars = {}
for c in all_cols:
    if c.endswith(f"_numerator{SUFFIX}"):
        stem = c[: -(len("_numerator") + len(SUFFIX))]
        den_col = _den_col(stem)
        raw_col = _raw_col(stem)
        if den_col in all_cols and raw_col in all_cols:
            num_vars[stem] = {"num": c, "den": den_col, "raw": raw_col}

print(f"Variables with num + den + raw: {len(num_vars)}")

flag_summary = []
for stem, cols in sorted(num_vars.items()):
    num = pd.to_numeric(df[cols["num"]], errors="coerce")
    den = pd.to_numeric(df[cols["den"]], errors="coerce")
    raw = pd.to_numeric(df[cols["raw"]], errors="coerce")

    # Compute implied proportion from numerator/denominator
    implied = (num / den).where(den > 0)

    scale = _detect_scale(raw)
    raw_prop = raw / scale  # convert raw_value to proportion space

    # Absolute and relative difference in proportion space
    abs_diff = (implied - raw_prop).abs()
    rel_diff = (abs_diff / raw_prop.abs()).where(raw_prop.abs() > 1e-9)

    # Flag rows where either threshold is exceeded
    flag_mask = (abs_diff > ABS_TOL) | (rel_diff > REL_TOL)
    n_checkable = implied.notna() & raw_prop.notna()
    n_flagged = (flag_mask & n_checkable).sum()
    n_checkable_count = n_checkable.sum()

    flag_col = f"{_safe_name(stem)}_ratio_flag"
    df[flag_col] = pd.NA
    df.loc[n_checkable, flag_col] = flag_mask[n_checkable].astype("Int64")

    pct_flagged = (100 * n_flagged / n_checkable_count) if n_checkable_count else 0
    flag_summary.append({
        "variable": stem,
        "n_checkable": n_checkable_count,
        "n_flagged": n_flagged,
        "pct_flagged": round(pct_flagged, 1),
        "flag_col": flag_col,
    })

flag_df = pd.DataFrame(flag_summary).sort_values("pct_flagged", ascending=False)
print("\nVerification results (sorted by % flagged):")
print(flag_df.to_string(index=False))

high_flag = flag_df[flag_df["pct_flagged"] > 5.0]
if not high_flag.empty:
    print(
        f"\nWARNING: {len(high_flag)} variables have >5% rows where "
        "num/den deviates from raw_value — inspect these in QA."
    )
    print(high_flag[["variable", "n_flagged", "pct_flagged"]].to_string(index=False))
else:
    print("\nAll variables pass: num/den ≈ raw_value within tolerance for ≥95% of rows.")


# ── Export ───────────────────────────────────────────────────────────────────
today_str = date.today().strftime("%Y-%m-%d")
out_name = f"{today_str}_full_merged_derived.csv"
out_path = os.path.join(merged_dir, out_name)

df = df.sort_values(["fips", "year"]).reset_index(drop=True)
df.to_csv(out_path, index=False)
print(f"\nSaved derived panel: {out_path}")
print(f"Shape: {df.shape}")
print(
    f"New columns added: "
    f"{len(pass1_results)} count_imputed + {len(num_vars)} ratio_flag = "
    f"{len(pass1_results) + len(num_vars)} total"
)
