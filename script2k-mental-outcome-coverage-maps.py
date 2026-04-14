#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2k: Mental-outcome value maps (county-level).

Purpose:
- For each mental health outcome, produce a 1×5 faceted choropleth showing
  the ACTUAL outcome value (e.g. poor mental health days = 10) per county,
  across 5 representative years.
- Mirrors the CAFO choropleth structure (script2f Figures G/H): consistent
  color scale across years within each outcome so trends are visually comparable.

Outputs:
- Dropbox/Mental/Data/merged/figs/mental-outcome-coverage-maps/
  {date}_{outcome}_values_facet_5yr.html
  {date}_{outcome}_values_facet_5yr.png
"""

from packages import *
from functions import *
import json

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None
    make_subplots = None


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(merged_dir, "figs", "mental-outcome-coverage-maps")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
load_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key

COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

MENTAL_OUTCOMES = {
    "Poor Mental Health Days (avg days/month, CHR)":    "poor_mental_health_days",
    "Excessive Drinking (%, CHR)":                      "excessive_drinking_per100k",
    "Frequent Mental Distress (%, CHR)":                "frequent_mental_distress_per100k",
    "Deaths of Despair — Crude Rate (per 100k, own)":   "crude_rate_from_census_pop",
    "Mental Health Providers (per 100k, CHR)":          "mental_health_providers_per100k",
}

# 5 representative years per outcome (chosen by coverage)
FACET_YEARS = {
    "poor_mental_health_days":      [2012, 2014, 2016, 2018, 2020],
    "excessive_drinking_per100k":   [2012, 2014, 2016, 2018, 2020],
    "frequent_mental_distress_per100k": [2017, 2018, 2019, 2020, 2021],
    "crude_rate_from_census_pop":   [2010, 2013, 2016, 2018, 2020],
    "mental_health_providers_per100k": [2013, 2015, 2017, 2019, 2021],
}

# Colorscale per outcome (diverging/sequential as appropriate)
COLORSCALES = {
    "poor_mental_health_days":          "YlOrRd",
    "excessive_drinking_per100k":       "Blues",
    "frequent_mental_distress_per100k": "YlOrRd",
    "crude_rate_from_census_pop":       "Reds",
    "mental_health_providers_per100k":  "YlGnBu",
}


# ---------------------------------------------------------------------
# GeoJSON + facet helpers
# ---------------------------------------------------------------------
def load_county_geojson():
    """
    Load county GeoJSON.
    Priority:
      1) Direct download from plotly dataset URL
      2) Fallback: parse embedded FeatureCollection from existing local HTML map
    """
    try:
        from urllib.request import urlopen
        with urlopen(COUNTY_GEOJSON_URL, timeout=20) as r:
            return json.load(r)
    except Exception as e:
        print(f"GeoJSON URL fetch failed ({e}); trying local HTML fallback.")

    html_candidates = (
        sorted(glob.glob(os.path.join(merged_dir, "figs", "core-visuals", "*_F2_poor_mental_health_map_2012.html")), reverse=True)
        + sorted(glob.glob(os.path.join(merged_dir, "figs", "core-visuals", "*_F1_cafo_intensity_map_2012.html")), reverse=True)
        + sorted(glob.glob(os.path.join(merged_dir, "figs", "fsis-choropleth", "*.html")), reverse=True)
    )

    decoder = json.JSONDecoder()
    for html_path in html_candidates:
        try:
            txt = open(html_path, "r", encoding="utf-8", errors="ignore").read()
            start = txt.find('{"type":"FeatureCollection"')
            if start < 0:
                idx = txt.find('"type":"FeatureCollection"')
                if idx < 0:
                    continue
                start = txt.rfind("{", 0, idx)
                if start < 0:
                    continue
            obj, _ = decoder.raw_decode(txt[start:])
            if isinstance(obj, dict) and obj.get("type") == "FeatureCollection" and "features" in obj:
                print(f"Loaded county GeoJSON from local fallback: {os.path.basename(html_path)}")
                return obj
        except Exception:
            continue

    raise RuntimeError("Could not load county GeoJSON from URL or local HTML fallback.")


def make_faceted_value_figure(df_panel, col, label, years, county_geojson,
                              colorscale="YlOrRd"):
    """
    Create a 1×5 faceted choropleth showing ACTUAL outcome values per county.
    Color scale is consistent across all 5 panels (global p2–p98 range).
    Missing counties appear light grey.
    """
    n_years = len(years)
    n_cols  = n_years
    n_rows  = 1

    # Global colour range across all selected years (p2–p98 to avoid outlier distortion)
    all_vals = pd.concat([
        to_num(df_panel.loc[df_panel["year"].astype("Int64") == yr, col])
        for yr in years
    ]).dropna()
    z_min = float(all_vals.quantile(0.02)) if len(all_vals) else 0
    z_max = float(all_vals.quantile(0.98)) if len(all_vals) else 1

    # Panel titles: year + n counties with data
    panel_titles = []
    for yr in years:
        n_obs = to_num(df_panel.loc[df_panel["year"].astype("Int64") == yr, col]).notna().sum()
        panel_titles.append(f"{yr}  (n={n_obs:,})")

    specs = [[{"type": "choropleth"}] * n_cols]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=panel_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.08,
    )

    for i, yr in enumerate(years):
        col_idx = i + 1
        d = df_panel[df_panel["year"].astype("Int64") == yr][["fips_str", col]].copy()
        d["val"] = to_num(d[col])
        d = d.dropna(subset=["val"])
        geo_key = "geo" if i == 0 else f"geo{i + 1}"

        fig.add_trace(
            go.Choropleth(
                geojson=county_geojson,
                locations=d["fips_str"],
                z=d["val"],
                zmin=z_min,
                zmax=z_max,
                colorscale=colorscale,
                marker_line_width=0.0,
                showscale=(i == n_years - 1),   # show colorbar on rightmost panel only
                colorbar=dict(
                    title=col,
                    x=1.01,
                    thickness=12,
                    len=0.7,
                ),
                name=str(yr),
                hovertemplate=(
                    "FIPS: %{location}<br>"
                    f"{label}: " + "%{z:.2f}<extra></extra>"
                ),
            ),
            row=1, col=col_idx,
        )
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(f"{label}<br>"
                  "<sup>Rural US counties | Consistent color scale across years"
                  f" | Range: {z_min:.1f}–{z_max:.1f}</sup>"),
            x=0.5, xanchor="center", font_size=14,
        ),
        height=320,
        width=320 * n_cols,
        margin={"r": 60, "t": 80, "l": 10, "b": 10},
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------
# Load latest merged panel
# ---------------------------------------------------------------------
merged_path = load_file(merged_dir, "*_full_merged.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df["fips_str"] = df["fips"].astype("Int64").astype("string").str.zfill(5)
# Rural filter — same as all analysis scripts
df = df[df["rural"] == 1].copy()

panel_years = sorted(df["year"].dropna().astype(int).unique().tolist())
print(f"Using merged file: {os.path.basename(merged_path)}")
print(f"Rural counties: {df['fips'].nunique():,} | years {min(panel_years)}–{max(panel_years)}")

if go is None or make_subplots is None:
    raise SystemExit("plotly.graph_objects unavailable — cannot build faceted maps.")

county_geojson = load_county_geojson()

# ---------------------------------------------------------------------
# Value maps: 1 figure per outcome, 1×5 facet of actual values
# ---------------------------------------------------------------------
for label, col in MENTAL_OUTCOMES.items():
    if col not in df.columns:
        print(f"Skipping {label}: column '{col}' not in panel")
        continue

    years = FACET_YEARS.get(col, [2012, 2014, 2016, 2018, 2020])
    # Drop requested years where the column has no data at all
    years = [yr for yr in years
             if to_num(df.loc[df["year"].astype("Int64") == yr, col]).notna().sum() > 0]
    if not years:
        print(f"Skipping {label}: no data in requested facet years")
        continue

    cscale = COLORSCALES.get(col, "YlOrRd")
    fig = make_faceted_value_figure(df, col, label, years, county_geojson, colorscale=cscale)

    stub = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
    html_path = os.path.join(out_dir, f"{today_str}_{stub}_values_facet_5yr.html")
    png_path  = os.path.join(out_dir, f"{today_str}_{stub}_values_facet_5yr.png")

    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Saved HTML:", html_path)

    try:
        fig.write_image(png_path, width=320 * len(years), height=320, scale=2)
        print("Saved PNG: ", png_path)
    except Exception as e:
        print(f"PNG export failed ({e})")

print(f"\nAll value maps saved to: {out_dir}")
