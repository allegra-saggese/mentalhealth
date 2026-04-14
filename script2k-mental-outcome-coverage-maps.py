#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2k: Mental-outcome coverage maps (county-level).

Purpose:
- Build a map series for mental-health outcome coverage using the latest merged panel.
- For each outcome variable, produce:
  1) % of active-window years observed in each county
  2) Binary observed/missing map for that variable's latest data year

Outputs:
- Dropbox/Mental/Data/merged/figs/mental-outcome-coverage-maps/
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
    "Poor Mental Health Days (CHR)": "poor_mental_health_days",
    "Frequent Mental Distress (CHR, per 100k)": "frequent_mental_distress_per100k",
    "Excessive Drinking (CHR, per 100k)": "excessive_drinking_per100k",
    "Mental Health Providers (CHR, per 100k)": "mental_health_providers_per100k",
    "Deaths of Despair Crude Rate (CDC)": "crude_rate_despair",
}
MAX_YEARS_PER_FACET_PNG = 10


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


def make_faceted_availability_figure(df_panel, col, label, years, county_geojson):
    """
    Create a year-faceted county map figure (Observed vs Missing) for one outcome.
    """
    n_years = len(years)
    n_cols = min(4, max(1, n_years))
    n_rows = int(np.ceil(n_years / n_cols))
    panel_titles = []
    for yr in years:
        d_yr = df_panel[df_panel["year"].astype("Int64") == yr][[col]].copy()
        cov_pct = float(to_num(d_yr[col]).notna().mean() * 100)
        panel_titles.append(f"{yr}<br>{cov_pct:.1f}% observed")

    specs = [[{"type": "choropleth"}] * n_cols for _ in range(n_rows)]
    year_positions = [(r, c) for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=panel_titles + [""] * (n_rows * n_cols - n_years),
        horizontal_spacing=0.01,
        vertical_spacing=0.08,
    )

    colorscale = [[0.0, "#d9d9d9"], [1.0, "#2a9d8f"]]

    for i, yr in enumerate(years):
        row, col_idx = year_positions[i]
        d = df_panel[df_panel["year"].astype("Int64") == yr][["fips_str", col]].copy()
        d["observed"] = to_num(d[col]).notna().astype(int)
        geo_key = "geo" if i == 0 else f"geo{i + 1}"

        fig.add_trace(
            go.Choropleth(
                geojson=county_geojson,
                locations=d["fips_str"],
                z=d["observed"],
                zmin=0,
                zmax=1,
                colorscale=colorscale,
                marker_line_width=0.0,
                showscale=(i == 0),
                colorbar=dict(
                    title="Data",
                    tickvals=[0, 1],
                    ticktext=["Missing", "Observed"],
                    len=0.35,
                ),
                name=str(yr),
                hovertemplate="FIPS: %{location}<br>Observed: %{z}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )
        fig.update_traces(selector={"name": str(yr)}, geojson=county_geojson)
        fig.update_layout(
            **{
                f"{geo_key}_scope": "usa",
                f"{geo_key}_showlakes": False,
                f"{geo_key}_bgcolor": "white",
            }
        )

    min_year = min(years)
    max_year = max(years)
    fig.update_layout(
        title=f"{label}: County Availability by Year ({min_year}-{max_year})",
        title_font_size=16,
        margin={"r": 10, "t": 75, "l": 10, "b": 10},
        paper_bgcolor="white",
    )
    return fig, n_rows, n_cols


# ---------------------------------------------------------------------
# Load latest merged panel
# ---------------------------------------------------------------------
merged_path = load_file(merged_dir, "*_full_merged.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df["fips_str"] = df["fips"].astype("Int64").astype("string").str.zfill(5)

panel_years = sorted(df["year"].dropna().astype(int).unique().tolist())
n_panel_years = len(panel_years)

print(f"Using merged file: {os.path.basename(merged_path)}")
print(f"Panel years: {min(panel_years)}-{max(panel_years)} (n={n_panel_years})")


# ---------------------------------------------------------------------
# Coverage summary table (long format)
# ---------------------------------------------------------------------
summary_rows = []

for label, col in MENTAL_OUTCOMES.items():
    if col not in df.columns:
        print(f"Skipping {label}: missing column '{col}'")
        continue

    s = to_num(df[col])
    years_with_any = sorted(df.loc[s.notna(), "year"].dropna().astype(int).unique().tolist())
    if not years_with_any:
        print(f"Skipping {label}: no non-missing observations")
        continue

    active_min, active_max = min(years_with_any), max(years_with_any)
    n_active_years = len(years_with_any)

    tmp = df[["fips", "fips_str", "year"]].copy()
    tmp["has_data"] = s.notna().astype(int)

    county_cov = (
        tmp.groupby(["fips", "fips_str"], as_index=False)
        .agg(
            n_years_observed=("has_data", "sum"),
            first_year_observed=("year", lambda x: int(x[tmp.loc[x.index, "has_data"] == 1].min()) if (tmp.loc[x.index, "has_data"] == 1).any() else np.nan),
            last_year_observed=("year", lambda x: int(x[tmp.loc[x.index, "has_data"] == 1].max()) if (tmp.loc[x.index, "has_data"] == 1).any() else np.nan),
        )
    )

    county_cov["coverage_pct_panel_window"] = county_cov["n_years_observed"] / n_panel_years * 100.0
    county_cov["coverage_pct_active_window"] = county_cov["n_years_observed"] / n_active_years * 100.0
    county_cov["variable"] = col
    county_cov["label"] = label
    county_cov["active_year_min"] = active_min
    county_cov["active_year_max"] = active_max
    county_cov["n_active_years"] = n_active_years

    summary_rows.append(county_cov)

if not summary_rows:
    raise SystemExit("No mental outcome columns with data found; no maps were created.")

coverage_long = pd.concat(summary_rows, ignore_index=True)
coverage_csv = os.path.join(out_dir, f"{today_str}_mental_outcome_county_coverage_long.csv")
coverage_long.to_csv(coverage_csv, index=False)
print("Saved:", coverage_csv)


# ---------------------------------------------------------------------
# Map series
# ---------------------------------------------------------------------
if px is None:
    raise SystemExit("plotly is not available; install plotly to render maps.")

county_geojson = load_county_geojson()

for label, col in MENTAL_OUTCOMES.items():
    d = coverage_long[coverage_long["variable"] == col].copy()
    if d.empty:
        continue

    active_min = int(d["active_year_min"].iloc[0])
    active_max = int(d["active_year_max"].iloc[0])
    latest_year = active_max

    # 1) Active-window % coverage by county
    fig_cov = px.choropleth(
        d,
        geojson=county_geojson,
        locations="fips_str",
        color="coverage_pct_active_window",
        color_continuous_scale="YlGnBu",
        range_color=(0, 100),
        scope="usa",
        hover_data={
            "fips_str": True,
            "n_years_observed": True,
            "n_active_years": True,
            "coverage_pct_active_window": ":.1f",
        },
        title=(
            f"{label}: County Coverage (% of Active Years with Data)<br>"
            f"Active window {active_min}-{active_max}"
        ),
        labels={"coverage_pct_active_window": "% Years Observed"},
    )
    fig_cov.update_geos(fitbounds="locations", visible=False)
    fig_cov.update_layout(margin={"r": 0, "t": 70, "l": 0, "b": 0})

    stub = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
    cov_html = os.path.join(out_dir, f"{today_str}_{stub}_coverage_active_window.html")
    fig_cov.write_html(cov_html, include_plotlyjs="cdn")
    print("Saved:", cov_html)

    # 2) Latest-year availability (observed vs missing)
    latest_df = df[df["year"].astype("Int64") == latest_year][["fips_str", col]].copy()
    latest_df["observed"] = np.where(to_num(latest_df[col]).notna(), "Observed", "Missing")
    latest_df = latest_df.drop(columns=[col])

    fig_latest = px.choropleth(
        latest_df,
        geojson=county_geojson,
        locations="fips_str",
        color="observed",
        color_discrete_map={"Observed": "#2a9d8f", "Missing": "#d9d9d9"},
        scope="usa",
        hover_data={"fips_str": True, "observed": True},
        title=f"{label}: Availability in Latest Data Year ({latest_year})",
    )
    fig_latest.update_geos(fitbounds="locations", visible=False)
    fig_latest.update_layout(margin={"r": 0, "t": 60, "l": 0, "b": 0})

    latest_html = os.path.join(out_dir, f"{today_str}_{stub}_availability_latest_year_{latest_year}.html")
    fig_latest.write_html(latest_html, include_plotlyjs="cdn")
    print("Saved:", latest_html)

    # 3) Faceted by-year availability maps (PNG + HTML)
    if go is None or make_subplots is None:
        print("Skipping faceted map export: plotly.graph_objects unavailable.")
        continue

    years_all = sorted(df.loc[to_num(df[col]).notna(), "year"].dropna().astype(int).unique().tolist())
    if not years_all:
        continue

    year_chunks = [years_all[i:i + MAX_YEARS_PER_FACET_PNG] for i in range(0, len(years_all), MAX_YEARS_PER_FACET_PNG)]

    for part_i, years in enumerate(year_chunks, start=1):
        suffix = f"_part{part_i}" if len(year_chunks) > 1 else ""
        fig_facet, n_rows, n_cols = make_faceted_availability_figure(df, col, label, years, county_geojson)
        facet_html = os.path.join(out_dir, f"{today_str}_{stub}_availability_facet_by_year{suffix}.html")
        facet_png = os.path.join(out_dir, f"{today_str}_{stub}_availability_facet_by_year{suffix}.png")
        fig_facet.write_html(facet_html, include_plotlyjs="cdn")
        print("Saved:", facet_html)

        try:
            width = 430 * n_cols
            height = 290 * n_rows + 120
            fig_facet.write_image(facet_png, width=width, height=height, scale=2)
            print("Saved:", facet_png)
        except Exception as e:
            print(f"PNG export skipped for {col}{suffix}: {e}")

print(f"\nMental-outcome coverage maps saved to: {out_dir}")
