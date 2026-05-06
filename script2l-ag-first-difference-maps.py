#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2l-ag-first-difference-maps.py

Purpose:
- Diagnose identifying variation for panel econometrics by mapping first differences
  at the county-year level for key treatment variables:
    1) CAFO total operations
    2) FSIS slaughterhouse-present establishments

Outputs:
- Dropbox/Mental/Data/merged/figs/ag-first-diff-maps/
  {date}_cafo_total_ops_first_diff_facet.html
  {date}_cafo_total_ops_first_diff_facet.png
  {date}_fsis_slaughter_first_diff_facet.html
  {date}_fsis_slaughter_first_diff_facet.png
  {date}_ag_first_diff_year_summary.csv
"""

from packages import *
from functions import *
import json

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
out_dir = os.path.join(merged_dir, "figs", "ag-first-diff-maps")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
load_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key

COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

MAP_SPECS = [
    {
        "label": "CAFO Total Operations",
        "col": "cafo_total_ops_all_animals",
        "stub": "cafo_total_ops",
        "years": [2007, 2012, 2017],
        "colorscale": "RdBu",
    },
    {
        "label": "FSIS Slaughterhouse-Present Establishments",
        "col": "n_slaughterhouse_present_establishments_fsis",
        "stub": "fsis_slaughter",
        "years": [2018, 2019, 2020, 2021, 2022, 2023],
        "colorscale": "RdBu",
    },
]


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


def make_first_diff(df_panel, col):
    d = df_panel[["fips", "year", "fips_str", col]].copy()
    d[col] = to_num(d[col])
    d = d.sort_values(["fips", "year"]).copy()
    d["first_diff"] = d.groupby("fips")[col].diff()
    return d


def pick_available_years(df_diff, years_requested):
    available = sorted(df_diff.loc[df_diff["first_diff"].notna(), "year"].dropna().astype(int).unique().tolist())
    return [yr for yr in years_requested if yr in available]


def year_summary(df_diff, col, label):
    rows = []
    for yr, g in df_diff.groupby("year"):
        x = to_num(g["first_diff"])
        valid = x.notna().sum()
        if valid == 0:
            continue
        nz = (x != 0).sum()
        rows.append(
            {
                "variable": col,
                "label": label,
                "year": int(yr),
                "n_counties_with_diff": int(valid),
                "n_nonzero_diff": int(nz),
                "share_nonzero_diff_pct": float(nz / valid * 100),
                "mean_diff": float(x.mean()),
                "std_diff": float(x.std()),
                "p05_diff": float(x.quantile(0.05)),
                "p50_diff": float(x.quantile(0.50)),
                "p95_diff": float(x.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def make_faceted_first_diff_figure(df_diff, label, years, county_geojson, colorscale="RdBu"):
    n_years = len(years)

    all_vals = pd.concat([
        to_num(df_diff.loc[df_diff["year"].astype("Int64") == yr, "first_diff"])
        for yr in years
    ]).dropna()

    max_abs = float(all_vals.abs().quantile(0.98)) if len(all_vals) else 1.0
    if max_abs <= 0:
        max_abs = 1.0

    z_min, z_max = -max_abs, max_abs

    panel_titles = []
    for yr in years:
        x = to_num(df_diff.loc[df_diff["year"].astype("Int64") == yr, "first_diff"])
        n_obs = x.notna().sum()
        n_nz = (x.fillna(0) != 0).sum()
        share_nz = (n_nz / n_obs * 100) if n_obs else 0
        panel_titles.append(f"{yr}  (n={n_obs:,}, non-zero={share_nz:.1f}%)")

    specs = [[{"type": "choropleth"}] * n_years]
    fig = make_subplots(
        rows=1,
        cols=n_years,
        specs=specs,
        subplot_titles=panel_titles,
        horizontal_spacing=0.01,
    )

    for i, yr in enumerate(years):
        d = df_diff[df_diff["year"].astype("Int64") == yr][["fips_str", "first_diff"]].copy()
        d = d.dropna(subset=["first_diff"])
        geo_key = "geo" if i == 0 else f"geo{i + 1}"

        fig.add_trace(
            go.Choropleth(
                geojson=county_geojson,
                locations=d["fips_str"],
                z=d["first_diff"],
                zmin=z_min,
                zmax=z_max,
                colorscale=colorscale,
                marker_line_width=0.0,
                showscale=(i == n_years - 1),
                colorbar=dict(
                    title="Δ count",
                    x=1.01,
                    thickness=12,
                    len=0.70,
                ),
                name=str(yr),
                hovertemplate=(
                    "FIPS: %{location}<br>"
                    "First difference: %{z:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(**{
            f"{geo_key}_scope": "usa",
            f"{geo_key}_showlakes": False,
            f"{geo_key}_bgcolor": "white",
        })

    fig.update_layout(
        title=dict(
            text=(
                f"{label} — County-Year First Differences<br>"
                "<sup>Rural US counties | Δ = value(t) - value(t-1)"
                f" | Symmetric scale: {z_min:.1f} to {z_max:.1f}</sup>"
            ),
            x=0.5,
            xanchor="center",
            font_size=14,
        ),
        height=340,
        width=330 * n_years,
        margin={"r": 60, "t": 84, "l": 10, "b": 10},
        paper_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------
# Load merged panel
# ---------------------------------------------------------------------
merged_path = load_file(merged_dir, "*_full_merged.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df["fips_str"] = df["fips"].astype("Int64").astype("string").str.zfill(5)

# Rural filter — same as analysis scripts
if "rural" in df.columns:
    df = df[df["rural"] == 1].copy()

panel_years = sorted(df["year"].dropna().astype(int).unique().tolist())
print(f"Using merged file: {os.path.basename(merged_path)}")
print(f"Rural counties: {df['fips'].nunique():,} | years {min(panel_years)}–{max(panel_years)}")

if go is None or make_subplots is None:
    raise SystemExit("plotly.graph_objects unavailable — cannot build first-difference maps.")

county_geojson = load_county_geojson()

# ---------------------------------------------------------------------
# Build maps + year summary
# ---------------------------------------------------------------------
summary_frames = []

for spec in MAP_SPECS:
    col = spec["col"]
    label = spec["label"]

    if col not in df.columns:
        print(f"Skipping {label}: column '{col}' not in merged panel")
        continue

    df_diff = make_first_diff(df, col)
    years = pick_available_years(df_diff, spec["years"])

    if not years:
        print(f"Skipping {label}: no available years with non-missing first differences")
        continue

    summary_frames.append(year_summary(df_diff, col, label))

    fig = make_faceted_first_diff_figure(
        df_diff=df_diff,
        label=label,
        years=years,
        county_geojson=county_geojson,
        colorscale=spec.get("colorscale", "RdBu"),
    )

    html_path = os.path.join(out_dir, f"{today_str}_{spec['stub']}_first_diff_facet.html")
    png_path = os.path.join(out_dir, f"{today_str}_{spec['stub']}_first_diff_facet.png")

    fig.write_html(html_path, include_plotlyjs="cdn")
    print("Saved HTML:", html_path)

    try:
        fig.write_image(png_path, width=330 * len(years), height=340, scale=2)
        print("Saved PNG: ", png_path)
    except Exception as e:
        print(f"PNG export failed ({e})")

if summary_frames:
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = os.path.join(out_dir, f"{today_str}_ag_first_diff_year_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("Saved first-difference year summary:", summary_path)
else:
    print("No first-difference summaries written (no eligible variables/years).")
