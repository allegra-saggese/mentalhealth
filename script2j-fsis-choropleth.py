#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2j-fsis-choropleth.py

County-level choropleth maps of FSIS establishment counts, 2017–2023.
Structure mirrors the CAFO choropleth (script2f Figures G/H): faceted by year,
using log(establishments per 10k population + 1) to normalize for county size.

Four figures produced (each saved as HTML + PNG):
  Z1_fsis_total_by_year.png       — Total FSIS establishments per 10k, 2017–2023
  Z2_fsis_slaughter_by_year.png   — Slaughterhouse-present establishments per 10k
  Z3_fsis_meat_by_year.png        — Meat slaughter establishments per 10k
  Z4_fsis_poultry_by_year.png     — Poultry slaughter establishments per 10k

Layout: 2 rows × 4 columns (7 years: 2017–2023; 8th panel left empty).
Outputs saved to: Dropbox/Mental/Data/merged/figs/fsis-choropleth/
"""

from packages import *
from functions import *

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly
    print(f"plotly {plotly.__version__}")
except ImportError:
    raise SystemExit("plotly not available — run: pip install plotly kaleido")

from urllib.request import urlopen
import json

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir = os.path.join(db_data, "merged")
out_dir    = os.path.join(db_data, "merged", "figs", "fsis-choropleth")
os.makedirs(out_dir, exist_ok=True)
today_str  = date.today().strftime("%Y-%m-%d")

# ── Load panel ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(latest_file_glob(merged_dir, "*_full_merged.csv"), low_memory=False)
df_raw = df_raw[df_raw["rural"] == 1].copy()
print(f"Rural panel: {len(df_raw):,} rows | {df_raw['fips'].nunique():,} counties")

POP_COL   = "population"
FSIS_YEARS = list(range(2017, 2024))   # 2017–2023

def log_per10k(series, pop):
    x   = pd.to_numeric(series, errors="coerce")
    pop = pd.to_numeric(pop,    errors="coerce").replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)

# Derive log per-10k for all FSIS treatments
FSIS_TREATMENTS = {
    "Total FSIS Establishments":         "n_unique_establishments_fsis",
    "Slaughterhouse Establishments":     "n_slaughterhouse_present_establishments_fsis",
    "Meat Slaughter Establishments":     "n_meat_slaughter_establishments_fsis",
    "Poultry Slaughter Establishments":  "n_poultry_slaughter_establishments_fsis",
}
FSIS_STUBS = ["Z1_fsis_total", "Z2_fsis_slaughter", "Z3_fsis_meat", "Z4_fsis_poultry"]

df_fsis = df_raw[df_raw["year"].between(2017, 2023)].copy()
df_fsis["fips_str"] = df_fsis["fips"].astype(str).str.zfill(5)

for col in FSIS_TREATMENTS.values():
    df_fsis[f"{col}_log10k"] = log_per10k(df_fsis[col], df_fsis[POP_COL])

# ── Load county GeoJSON ───────────────────────────────────────────────────────
print("Loading county GeoJSON …")
_geojson_url = (
    "https://raw.githubusercontent.com/plotly/datasets"
    "/master/geojson-counties-fips.json"
)
with urlopen(_geojson_url) as r:
    counties_geojson = json.load(r)
print("GeoJSON loaded.")

# ── Helper: build 2×4 choropleth facet grid ───────────────────────────────────
def _make_fsis_choropleth(log_col, title, out_stub, colorscale="YlOrRd"):
    """
    Build a 2×4 facet grid (7 years 2017–2023 + 1 empty) of county choropleth maps.
    log_col: column in df_fsis containing log(n per 10k + 1) values.
    """
    n_cols = 4
    n_rows = 2
    specs  = [[{"type": "choropleth"}] * n_cols for _ in range(n_rows)]

    year_positions = [(r, c) for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    subplot_titles = [str(y) for y in FSIS_YEARS] + [""]   # 8th panel empty

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.05, horizontal_spacing=0.01,
    )

    # Global color range across all years for a consistent scale
    _all_vals = df_fsis[log_col].dropna()
    z_min, z_max = float(_all_vals.quantile(0.01)), float(_all_vals.quantile(0.99))

    for idx, yr in enumerate(FSIS_YEARS):
        row, col = year_positions[idx]
        df_yr = df_fsis[df_fsis["year"] == yr][["fips_str", log_col]].dropna()

        geo_key = "geo" if idx == 0 else f"geo{idx + 1}"
        fig.add_trace(
            go.Choropleth(
                geojson=counties_geojson,
                locations=df_yr["fips_str"],
                z=df_yr[log_col],
                colorscale=colorscale,
                zmin=z_min, zmax=z_max,
                marker_line_width=0,
                showscale=(idx == 0),
                colorbar=dict(title="log(n/10k+1)", x=1.01, thickness=12, len=0.8),
                name=str(yr),
            ),
            row=row, col=col,
        )
        fig.update_traces(
            selector={"name": str(yr)},
            geojson=counties_geojson,
        )
        # Scope to continental US
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(f"{title}<br>"
                  "<sup>Rural US counties | log(establishments per 10,000 residents + 1)"
                  " | Consistent color scale across years</sup>"),
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        height=600, width=1400,
        margin=dict(l=10, r=60, t=80, b=10),
        paper_bgcolor="white",
    )

    # Save HTML
    html_path = os.path.join(out_dir, f"{today_str}_{out_stub}_by_year.html")
    fig.write_html(html_path)
    print("Saved HTML:", html_path)

    # Save PNG via kaleido
    png_path = os.path.join(out_dir, f"{today_str}_{out_stub}_by_year.png")
    try:
        fig.write_image(png_path, width=1400, height=600, scale=2)
        print("Saved PNG:", png_path)
    except Exception as e:
        print(f"  PNG export failed (kaleido issue): {e}")

    return png_path


# ── Generate all 4 FSIS figures ───────────────────────────────────────────────
for (treat_label, raw_col), out_stub in zip(FSIS_TREATMENTS.items(), FSIS_STUBS):
    log_col = f"{raw_col}_log10k"
    _make_fsis_choropleth(
        log_col=log_col,
        title=f"FSIS {treat_label} — Rural US Counties, 2017–2023",
        out_stub=out_stub,
        colorscale="YlOrRd",
    )

print(f"\nAll FSIS choropleth figures saved to: {out_dir}")
