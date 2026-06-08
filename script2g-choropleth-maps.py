#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2h-choropleth-maps.py

Three sections of county-level choropleth maps, all using Plotly. The panel and
GeoJSON are loaded once and shared across sections.

  Section 1 — FSIS establishment counts (2017–2023)
    Faceted 2×4 choropleths for total, slaughterhouse, meat, and poultry
    establishments per 10k population. Four figures × (HTML + PNG).
    Outputs → figs/fsis-choropleth/

  Section 2 — Mental health outcome values
    1×5 faceted choropleths showing actual outcome values (not coverage flags)
    across 5 representative years. Consistent color scale within each outcome.
    Five figures × (HTML + PNG).
    Outputs → figs/mental-outcome-coverage-maps/

  Section 3 — Agricultural first-difference maps
    Faceted choropleths of Δ(CAFO total ops) and Δ(FSIS slaughterhouse
    establishments) at the county-year level, to diagnose identifying variation
    for panel econometrics. Includes a year-level summary CSV.
    Outputs → figs/ag-first-diff-maps/

Dependencies: packages.py, functions.py (log_per10k, load_county_geojson,
              COUNTY_GEOJSON_URL, latest_file_glob, normalize_panel_key)
"""

from packages import *
from functions import *
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise SystemExit("plotly not available — run: pip install plotly kaleido")

try:
    import plotly.express as px
except Exception:
    px = None

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir        = os.path.join(db_data, "merged")
fsis_choro_dir    = os.path.join(merged_dir, "figs", "fsis-choropleth")
mental_maps_dir   = os.path.join(merged_dir, "figs", "mental-outcome-coverage-maps")
first_diff_dir    = os.path.join(merged_dir, "figs", "ag-first-diff-maps")
for d in (fsis_choro_dir, mental_maps_dir, first_diff_dir):
    os.makedirs(d, exist_ok=True)

POP_COL = "population"

# ── Load merged panel ─────────────────────────────────────────────────────────
merged_path = latest_file_glob(merged_dir, "*_full_merged.csv")
df_raw = pd.read_csv(merged_path, low_memory=False)
df_raw = normalize_panel_key(df_raw)
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
df_raw = df_raw.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df_raw["fips_str"] = df_raw["fips"].astype("Int64").astype("string").str.zfill(5)
df     = df_raw[df_raw["rural"] == 1].copy()
panel_years = sorted(df["year"].dropna().astype(int).unique().tolist())
print(f"Rural panel: {len(df):,} rows | {df['fips'].nunique():,} counties | "
      f"{min(panel_years)}–{max(panel_years)}")

# ── Load county GeoJSON once ──────────────────────────────────────────────────
print("Loading county GeoJSON …")
county_geojson = load_county_geojson(merged_dir)
print("GeoJSON loaded.")


# =============================================================================
# SECTION 1 — FSIS establishment choropleths (2017–2023)
# =============================================================================
print("\n--- Section 1: FSIS establishment choropleths ---")

FSIS_YEARS = list(range(2017, 2024))

FSIS_TREATMENTS = {
    "Total FSIS Establishments":        "n_unique_establishments_fsis",
    "Slaughterhouse Establishments":    "n_slaughterhouse_present_establishments_fsis",
    "Meat Slaughter Establishments":    "n_meat_slaughter_establishments_fsis",
    "Poultry Slaughter Establishments": "n_poultry_slaughter_establishments_fsis",
}
FSIS_STUBS = ["Z1_fsis_total", "Z2_fsis_slaughter", "Z3_fsis_meat", "Z4_fsis_poultry"]

df_fsis = df[df["year"].between(2017, 2023)].copy()
for col in FSIS_TREATMENTS.values():
    if col in df_fsis.columns:
        df_fsis[f"{col}_log10k"] = log_per10k(df_fsis[col], df_fsis[POP_COL])


def _make_fsis_choropleth(log_col, title, out_stub, colorscale="YlOrRd"):
    n_rows, n_cols = 2, 4
    specs = [[{"type": "choropleth"}] * n_cols for _ in range(n_rows)]
    year_positions = [(r, c) for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    subplot_titles = [str(y) for y in FSIS_YEARS] + [""]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles, specs=specs,
        vertical_spacing=0.05, horizontal_spacing=0.01,
    )

    _all_vals = df_fsis[log_col].dropna()
    z_min, z_max = float(_all_vals.quantile(0.01)), float(_all_vals.quantile(0.99))

    for idx, yr in enumerate(FSIS_YEARS):
        row, col = year_positions[idx]
        df_yr = df_fsis[df_fsis["year"] == yr][["fips_str", log_col]].dropna()
        geo_key = "geo" if idx == 0 else f"geo{idx + 1}"

        fig.add_trace(go.Choropleth(
            geojson=county_geojson, locations=df_yr["fips_str"], z=df_yr[log_col],
            colorscale=colorscale, zmin=z_min, zmax=z_max, marker_line_width=0,
            showscale=(idx == 0),
            colorbar=dict(title="log(n/10k+1)", x=1.01, thickness=12, len=0.8),
            name=str(yr),
        ), row=row, col=col)
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(f"{title}<br><sup>Rural US counties | log(establishments per 10,000 residents + 1)"
                  " | Consistent color scale across years</sup>"),
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=600, width=1400,
        margin=dict(l=10, r=60, t=80, b=10),
        paper_bgcolor="white",
    )

    html_path = os.path.join(fsis_choro_dir, f"{today_str}_{out_stub}_by_year.html")
    png_path  = os.path.join(fsis_choro_dir, f"{today_str}_{out_stub}_by_year.png")
    fig.write_html(html_path)
    print("  Saved HTML:", html_path)
    try:
        fig.write_image(png_path, width=1400, height=600, scale=2)
        print("  Saved PNG:", png_path)
    except Exception as e:
        print(f"  PNG export failed (kaleido): {e}")


for (treat_label, raw_col), out_stub in zip(FSIS_TREATMENTS.items(), FSIS_STUBS):
    log_col = f"{raw_col}_log10k"
    if log_col not in df_fsis.columns:
        print(f"  Skipping {treat_label}: column '{raw_col}' not in panel")
        continue
    _make_fsis_choropleth(
        log_col=log_col,
        title=f"FSIS {treat_label} — Rural US Counties, 2017–2023",
        out_stub=out_stub,
    )


# =============================================================================
# SECTION 2 — Mental health outcome value maps
# =============================================================================
print("\n--- Section 2: Mental health outcome value maps ---")

MENTAL_OUTCOMES = {
    "Poor Mental Health Days (avg days/month, CHR)":  "poor_mental_health_days",
    "Excessive Drinking (%, CHR)":                    "excessive_drinking_per100k",
    "Frequent Mental Distress (%, CHR)":              "frequent_mental_distress_per100k",
    "Deaths of Despair — Crude Rate (per 100k, own)": "crude_rate_from_census_pop",
    "Mental Health Providers (per 100k, CHR)":        "mental_health_providers_per100k",
}
FACET_YEARS = {
    "poor_mental_health_days":          [2012, 2014, 2016, 2018, 2020],
    "excessive_drinking_per100k":       [2012, 2014, 2016, 2018, 2020],
    "frequent_mental_distress_per100k": [2017, 2018, 2019, 2020, 2021],
    "crude_rate_from_census_pop":       [2010, 2013, 2016, 2018, 2020],
    "mental_health_providers_per100k":  [2013, 2015, 2017, 2019, 2021],
}
COLORSCALES = {
    "poor_mental_health_days":          "YlOrRd",
    "excessive_drinking_per100k":       "Blues",
    "frequent_mental_distress_per100k": "YlOrRd",
    "crude_rate_from_census_pop":       "Reds",
    "mental_health_providers_per100k":  "YlGnBu",
}


def _make_mental_facet(df_panel, col, label, years, colorscale="YlOrRd"):
    """1×N faceted choropleth with consistent color scale across all years."""
    all_vals = pd.concat([
        to_numeric_series(df_panel.loc[df_panel["year"].astype("Int64") == yr, col])
        for yr in years
    ]).dropna()
    z_min = float(all_vals.quantile(0.02)) if len(all_vals) else 0
    z_max = float(all_vals.quantile(0.98)) if len(all_vals) else 1

    panel_titles = [
        f"{yr}  (n={to_numeric_series(df_panel.loc[df_panel['year'].astype('Int64')==yr, col]).notna().sum():,})"
        for yr in years
    ]
    specs = [[{"type": "choropleth"}] * len(years)]
    fig   = make_subplots(rows=1, cols=len(years), specs=specs,
                          subplot_titles=panel_titles, horizontal_spacing=0.01)

    for i, yr in enumerate(years):
        d = df_panel[df_panel["year"].astype("Int64") == yr][["fips_str", col]].copy()
        d["val"] = to_numeric_series(d[col])
        d = d.dropna(subset=["val"])
        geo_key = "geo" if i == 0 else f"geo{i + 1}"

        fig.add_trace(go.Choropleth(
            geojson=county_geojson, locations=d["fips_str"], z=d["val"],
            zmin=z_min, zmax=z_max, colorscale=colorscale,
            marker_line_width=0.0, showscale=(i == len(years) - 1),
            colorbar=dict(title=col, x=1.01, thickness=12, len=0.7),
            name=str(yr),
            hovertemplate=f"FIPS: %{{location}}<br>{label}: %{{z:.2f}}<extra></extra>",
        ), row=1, col=i + 1)
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(f"{label}<br><sup>Rural US counties | Consistent color scale across years"
                  f" | Range: {z_min:.1f}–{z_max:.1f}</sup>"),
            x=0.5, xanchor="center", font_size=14,
        ),
        height=320, width=320 * len(years),
        margin={"r": 60, "t": 80, "l": 10, "b": 10},
        paper_bgcolor="white",
    )
    return fig


for label, col in MENTAL_OUTCOMES.items():
    if col not in df.columns:
        print(f"  Skipping {label}: column '{col}' not in panel")
        continue
    years = FACET_YEARS.get(col, [2012, 2014, 2016, 2018, 2020])
    years = [yr for yr in years
             if to_numeric_series(df.loc[df["year"].astype("Int64") == yr, col]).notna().sum() > 0]
    if not years:
        print(f"  Skipping {label}: no data in requested facet years")
        continue

    fig  = _make_mental_facet(df, col, label, years, colorscale=COLORSCALES.get(col, "YlOrRd"))
    stub = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")

    html_path = os.path.join(mental_maps_dir, f"{today_str}_{stub}_values_facet_5yr.html")
    png_path  = os.path.join(mental_maps_dir, f"{today_str}_{stub}_values_facet_5yr.png")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("  Saved HTML:", html_path)
    try:
        fig.write_image(png_path, width=320 * len(years), height=320, scale=2)
        print("  Saved PNG:", png_path)
    except Exception as e:
        print(f"  PNG export failed ({e})")


# =============================================================================
# SECTION 3 — Agricultural first-difference maps
# =============================================================================
print("\n--- Section 3: Agricultural first-difference maps ---")

MAP_SPECS = [
    {
        "label": "CAFO Total Operations",
        "col":   "cafo_total_ops_all_animals",
        "stub":  "cafo_total_ops",
        "years": [2007, 2012, 2017, 2022],
        "colorscale": "RdBu",
    },
    {
        "label": "FSIS Slaughterhouse-Present Establishments",
        "col":   "n_slaughterhouse_present_establishments_fsis",
        "stub":  "fsis_slaughter",
        "years": [2018, 2019, 2020, 2021, 2022, 2023],
        "colorscale": "RdBu",
    },
]


def _make_first_diff(df_panel, col):
    d = df_panel[["fips", "year", "fips_str", col]].copy()
    d[col] = to_numeric_series(d[col])
    d = d.sort_values(["fips", "year"]).copy()
    d["first_diff"] = d.groupby("fips")[col].diff()
    return d


def _pick_available_years(df_diff, years_requested):
    available = sorted(
        df_diff.loc[df_diff["first_diff"].notna(), "year"].dropna().astype(int).unique())
    return [yr for yr in years_requested if yr in available]


def _year_summary(df_diff, col, label):
    rows = []
    for yr, g in df_diff.groupby("year"):
        x = to_numeric_series(g["first_diff"])
        valid = x.notna().sum()
        if valid == 0:
            continue
        nz = ((x != 0) & x.notna()).sum()
        rows.append({
            "variable": col, "label": label, "year": int(yr),
            "n_counties_with_diff": int(valid),
            "n_nonzero_diff": int(nz),
            "share_nonzero_diff_pct": float(nz / valid * 100),
            "mean_diff": float(x.mean()), "std_diff": float(x.std()),
            "p05_diff": float(x.quantile(0.05)),
            "p50_diff": float(x.quantile(0.50)),
            "p95_diff": float(x.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def _make_first_diff_figure(df_diff, label, years, colorscale="RdBu"):
    all_vals = pd.concat([
        to_numeric_series(df_diff.loc[df_diff["year"].astype("Int64") == yr, "first_diff"])
        for yr in years
    ]).dropna()
    max_abs = float(all_vals.abs().quantile(0.98)) if len(all_vals) else 1.0
    max_abs = max(max_abs, 1e-6)
    z_min, z_max = -max_abs, max_abs

    panel_titles = []
    for yr in years:
        x = to_numeric_series(df_diff.loc[df_diff["year"].astype("Int64") == yr, "first_diff"])
        n_obs = x.notna().sum()
        n_nz  = (x.fillna(0) != 0).sum()
        panel_titles.append(f"{yr}  (n={n_obs:,}, non-zero={n_nz/n_obs*100:.1f}%)" if n_obs else str(yr))

    specs = [[{"type": "choropleth"}] * len(years)]
    fig   = make_subplots(rows=1, cols=len(years), specs=specs,
                          subplot_titles=panel_titles, horizontal_spacing=0.01)

    for i, yr in enumerate(years):
        d = df_diff[df_diff["year"].astype("Int64") == yr][["fips_str", "first_diff"]].dropna(subset=["first_diff"])
        geo_key = "geo" if i == 0 else f"geo{i + 1}"
        fig.add_trace(go.Choropleth(
            geojson=county_geojson, locations=d["fips_str"], z=d["first_diff"],
            zmin=z_min, zmax=z_max, colorscale=colorscale, marker_line_width=0.0,
            showscale=(i == len(years) - 1),
            colorbar=dict(title="Δ count", x=1.01, thickness=12, len=0.70),
            name=str(yr),
            hovertemplate="FIPS: %{location}<br>First difference: %{z:.2f}<extra></extra>",
        ), row=1, col=i + 1)
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(f"{label} — County-Year First Differences<br>"
                  f"<sup>Rural US counties | Δ = value(t) - value(t-1)"
                  f" | Symmetric scale: {z_min:.1f} to {z_max:.1f}</sup>"),
            x=0.5, xanchor="center", font_size=14,
        ),
        height=340, width=330 * len(years),
        margin={"r": 60, "t": 84, "l": 10, "b": 10},
        paper_bgcolor="white",
    )
    return fig


summary_frames = []
for spec in MAP_SPECS:
    col   = spec["col"]
    label = spec["label"]
    if col not in df.columns:
        print(f"  Skipping {label}: column '{col}' not in panel")
        continue

    df_diff = _make_first_diff(df, col)
    years   = _pick_available_years(df_diff, spec["years"])
    if not years:
        print(f"  Skipping {label}: no years with non-missing first differences")
        continue

    summary_frames.append(_year_summary(df_diff, col, label))

    fig      = _make_first_diff_figure(df_diff, label, years, colorscale=spec.get("colorscale", "RdBu"))
    html_path = os.path.join(first_diff_dir, f"{today_str}_{spec['stub']}_first_diff_facet.html")
    png_path  = os.path.join(first_diff_dir, f"{today_str}_{spec['stub']}_first_diff_facet.png")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("  Saved HTML:", html_path)
    try:
        fig.write_image(png_path, width=330 * len(years), height=340, scale=2)
        print("  Saved PNG:", png_path)
    except Exception as e:
        print(f"  PNG export failed ({e})")

if summary_frames:
    summary_path = os.path.join(first_diff_dir, f"{today_str}_ag_first_diff_year_summary.csv")
    pd.concat(summary_frames, ignore_index=True).to_csv(summary_path, index=False)
    print("  Saved first-difference year summary:", summary_path)

print("\nChoropleth map pipeline complete.")
