#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script1d-maps.py -- every county map, one section per treatment family (stage 1, descriptive; nothing is estimated).

Reads:   newest Data/merged/*_panel.csv (via designs.load_designs: panel + first-expansion / first-entry designs),
         newest Data/merged/*_panel_fsis.csv (FSIS columns live only there), newest Data/clean/*rural-key.csv
Writes:  Data/output/maps/YYYY-MM-DD_1d_{section}_{map}.html and .png (plus one first-difference summary CSV)
Run:     PYTHONHASHSEED=0 python3 script1d-maps.py [--only SECTION [SECTION ...]] [--list] [--no-png]
Moved from script2f-maps.py (fsis_density, outcomes, first_diff) and script2b-final-visuals.py (cafo_census = Figures G
and H; snapshot_2012 = Figures F1 and F2). New: entry, expansion, large_ops, species, species_count, estimation_sample,
coverage, fsis_entry, wave_change. Treatments are never redefined here: they come from
script4_treatment.py / designs.py. Missing cafo_* counts are 0 from 2002 (D-34).
GeoJSON: functions.load_county_geojson downloads the plotly county file on every run (no cache, 20 s timeout) and only
falls back to a map embedded in old HTML files; coordinates are rounded once here to keep each HTML file small.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import figstyle
from designs import load_designs, species_treatments, usable_sample
from functions import latest_file_glob, load_county_geojson, log_per10k, normalize_panel_key, to_numeric_series
from packages import db_data, maps_dir
from script4_treatment import CENSUS_YEARS, COVARIATE_ORDER, OUTCOMES

SCRIPT = "script1d-maps.py"
MERGED_DIR = str(Path(db_data) / "merged")
CLEAN_DIR = str(Path(db_data) / "clean")
TAG = "1d"                      # output-name tag (packages.py): <date>_1d_<stem>
MAPS_DIR = Path(maps_dir)

LATEST_WAVE = max(CENSUS_YEARS)
GEO_PRECISION = 2  # decimal degrees kept in the county outlines (about 1 km); geometry only
FSIS_YEARS = list(range(2017, 2024))
FSIS_ABSENT_AS_ZERO = True  # a county-year with no FSIS record has no listed establishment (see load_fsis)
PNG_SCALE = 2  # 1400 px wide x 2 = 2800 px, about 390 dpi at 183 mm

SPECIES = {"Cattle": "cattle", "Beef": "beef", "Dairy": "dairy", "Hogs": "hogs", "Chickens": "chickens"}
COUNT_SPECIES = {k: v for k, v in SPECIES.items() if v != "cattle"}  # D-36: cattle contains dairy and beef
SIZE_COLS = {
    "Small": "small_cafo",
    "Medium": "medium_cafo",
    "Large": "large_cafo",
}  # panel columns: cattle + hogs + chickens
ESTIMATION_OUTCOME = OUTCOMES["Poor MH Days"]  # the outcome whose estimation sample map 6 shows
ESTIMATION_CONTROLS = COVARIATE_ORDER[:8]  # the 8 pre-committed full-coverage controls (D-16, D-31): "the control set"

# Colour (one place). Sequential: Viridis (perceptually uniform); signed: ColorBrewer RdBu 7-class;
# categories: Okabe-Ito.
SEQ = "Viridis"
LAND = "#f7f7f7"  # county outside the sample / no data
ZERO = "#d9d9d9"  # in the sample, count is zero
NEUTRAL = "#b3b3b3"  # in the sample, none of the category of interest
VERMILLION = figstyle.PALETTE["vermillion"]
BLUE = figstyle.PALETTE["blue"]
DIVERGING_7 = ["#2166ac", "#67a9cf", "#d1e5f0", ZERO, "#fddbc7", "#ef8a62", "#b2182b"]  # decrease ... increase
COUNT_BINS = ([0, 1, 4, 9, 24], ["0", "1", "2-4", "5-9", "10-24", "25+"])  # upper edges (inclusive), labels
DELTA_BINS = ([-10, -5, -1, 0, 4, 9], ["<= -10", "-9 to -5", "-4 to -1", "0", "1 to 4", "5 to 9", ">= 10"])
COUNT_COLOURS = [ZERO] + sample_colorscale(SEQ, [0.12, 0.35, 0.58, 0.8, 0.97])

# National outline drawn over every map by outline_country (Plotly's built-in country boundaries, no county data used).
OUTLINE_COLOUR = "#404040"  # dark grey: readable over the county fill and the light-grey land
OUTLINE_WIDTH = 1  # px
OUTLINE_RESOLUTION = 50  # Natural Earth scale of the boundary: 1:50 million (Plotly's alternative is 110)


# =============================================================================
# Data
# =============================================================================
def apply_zero_rule(df):
    """D-34: a missing cafo_* count (and the three *_cafo size totals) is 0 from 2002 on; 2000-01 stay missing."""
    cols = [
        c for c in df.columns if (c.startswith("cafo_") or c.endswith("_cafo")) and pd.api.types.is_numeric_dtype(df[c])
    ]
    late = df["year"] >= min(CENSUS_YEARS)
    n_filled = int(df.loc[late, cols].isna().to_numpy().sum())
    df.loc[late, cols] = df.loc[late, cols].fillna(0)
    print(
        f"Zero rule (D-34): {n_filled:,} missing cafo values set to 0 in {len(cols)} columns "
        f"(year >= {min(CENSUS_YEARS)})"
    )
    return df


def load_fsis():
    """Rural county-years 2017-2023 from the FSIS panel. A row with every *_fsis column blank means the county had no
    FSIS-listed establishment that year (the panel is a left join onto the rural frame); with FSIS_ABSENT_AS_ZERO those
    blanks become 0 so that absence is drawn as zero, not as missing. This is a labelled assumption,
    not a D-number rule.
    """
    f = normalize_panel_key(pd.read_csv(latest_file_glob(MERGED_DIR, "*_panel_fsis.csv"), low_memory=False))
    f = f.loc[:, ~f.columns.duplicated()].drop_duplicates(["fips", "year"]).copy()
    f["fips"] = f["fips"].astype("Int64").astype("string").str.zfill(5)
    cols = [c for c in f.columns if c.endswith("_fsis")]
    absent = f[cols].isna().all(axis=1)
    print(
        f"FSIS panel: {len(f):,} county-years, {f['fips'].nunique():,} counties; "
        f"{int(absent.sum()):,} county-years with no FSIS record" + (" -> set to 0" if FSIS_ABSENT_AS_ZERO else "")
    )
    if FSIS_ABSENT_AS_ZERO:
        f.loc[absent, cols] = 0
    return f.loc[f["rural"] == 1].reset_index(drop=True)


def rounded_geojson(geojson, nd=GEO_PRECISION):
    """Round every coordinate to nd decimals and drop repeated vertices (outline geometry only; no data touched)."""

    def ring(coords):
        out = []
        for x, y in coords:
            p = [round(x, nd), round(y, nd)]
            if not out or out[-1] != p:
                out.append(p)
        return out if len(out) >= 4 else None

    feats = []
    for ft in geojson["features"]:
        g = ft["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        new = []
        for poly in polys:
            rings = [r for r in (ring(r) for r in poly) if r]
            if rings:
                new.append(rings)
        if new:
            geom = (
                {"type": "Polygon", "coordinates": new[0]}
                if len(new) == 1
                else {"type": "MultiPolygon", "coordinates": new}
            )
            feats.append({"type": "Feature", "id": ft["id"], "properties": {}, "geometry": geom})
    return {"type": "FeatureCollection", "features": feats}


class Data:
    """Everything the sections share, loaded once."""

    def __init__(self):
        self.df, self.designs, _, self.counts = load_designs()
        self.df.attrs = {}  # load_panel stores a DataFrame in attrs, which breaks pd.concat's attrs comparison
        for design in self.designs.values():
            design["data"].attrs = {}
        self.panel_path = latest_file_glob(MERGED_DIR, "*_panel.csv")
        assert not self.df.duplicated(["fips", "year"]).any(), "panel has duplicate county-years"
        self.df = apply_zero_rule(self.df)
        self.df["fips_str"] = self.df["fips"]
        self.fsis = load_fsis()
        self.fsis_path = latest_file_glob(MERGED_DIR, "*_panel_fsis.csv")
        self.key_path = latest_file_glob(CLEAN_DIR, "*rural-key.csv")
        self.key = pd.read_csv(self.key_path, dtype={"fips_generated": str}, low_memory=False)
        self.key["fips"] = self.key["fips_generated"].str.zfill(5)
        names = self.key.drop_duplicates("fips").set_index("fips")
        self.names = (names["county_name"].fillna("") + ", " + names["state_abbrev"].fillna("")).to_dict()
        print("Loading county GeoJSON ...")
        self.geojson = rounded_geojson(load_county_geojson(MERGED_DIR))
        print(
            f"GeoJSON: {len(self.geojson['features']):,} county outlines "
            f"(coordinates rounded to {GEO_PRECISION} decimals)"
        )
        self.fsis_sample = (
            f"rural (non-large-metro) counties in the FSIS panel: {self.fsis['fips'].nunique():,} counties, "
            f"{FSIS_YEARS[0]}-{FSIS_YEARS[-1]}; counties with no FSIS record drawn as 0"
        )
        n_c, n_y = self.df["fips"].nunique(), self.df["year"].nunique()
        self.sample = (
            f"Rural (non-large-metro) US counties: {n_c:,} counties, {n_y} years "
            f"({self.df['year'].min()}-{self.df['year'].max()})"
        )

    def at(self, year, col, df=None):
        """Series of `col` for one year, indexed by 5-digit FIPS."""
        d = self.df if df is None else df
        return d.loc[d["year"] == year].set_index("fips")[col]

    def note(self, source, sample=None, land=True):
        grey = " Very light grey: outside the sample." if land else ""
        return f"{source}. Sample: {sample or self.sample}.{grey} Panel {Path(self.panel_path).name}; {SCRIPT}."


# =============================================================================
# Rendering: one generic faceted-choropleth writer, wrappers for continuous and categorical scales
# =============================================================================
def facet_map(
    data,
    panels,
    stem,
    title,
    note,
    colorscale,
    zmin,
    zmax,
    colorbar,
    ncols,
    hover_label=None,
    text=None,
    legend=None,
    legend_title="",
    panel_w=1400,
    no_png=False,
):
    """panels: list of (subtitle, Series of z indexed by FIPS). Shared colour axis.
    Continuous maps get a horizontal colour
    bar, categorical maps (legend = [(label, colour)]) get a legend, both under the maps. Writes HTML + PNG."""
    n = len(panels)
    panel_w = {1: 1100, 2: 1400}.get(ncols, max(panel_w, 300 * ncols + 130))
    nrows = int(np.ceil(n / ncols))
    specs = [[{"type": "choropleth"} if r * ncols + c < n else None for c in range(ncols)] for r in range(nrows)]
    titles = [p[0] for p in panels] + [""] * (nrows * ncols - n)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.07 if nrows > 1 else 0.0,
        horizontal_spacing=0.01,
    )
    for i, (_, z) in enumerate(panels):
        z = z.dropna()
        hover = [f"{data.names.get(k, k)} ({k})" for k in z.index]
        if text is not None:
            hover = [f"{h}<br>{text(v)}" for h, v in zip(hover, z.to_numpy())]
        elif hover_label:
            hover = [f"{h}<br>{hover_label}: {v:.2f}" for h, v in zip(hover, z.to_numpy())]
        fig.add_trace(
            go.Choropleth(
                geojson=data.geojson,
                locations=z.index,
                z=z.to_numpy(),
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                marker_line_width=0,
                showscale=(i == 0 and legend is None),
                colorbar=colorbar,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
        fig.update_layout(
            **{
                ("geo" if i == 0 else f"geo{i + 1}"): dict(
                    scope="usa",
                    projection_type="albers usa",
                    showlakes=False,
                    bgcolor="white",
                    landcolor=LAND,
                    showsubunits=True,
                    subunitcolor="#8c8c8c",
                    subunitwidth=0.3,
                )
            }
        )
    for label, colour in legend or []:  # legend entries: empty markers carrying the category colour
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="markers",
                name=label,
                showlegend=True,
                marker=dict(symbol="square", size=12, color=colour, line=dict(width=0.5, color="#666666")),
            )
        )
    top, bottom = 80, 170
    plot_h = nrows * (panel_w - 20) / ncols * 0.62
    height = int(plot_h + top + bottom)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        height=height,
        width=panel_w,
        margin=dict(l=10, r=10, t=top, b=bottom),
        paper_bgcolor="white",
        font=dict(size=12),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.01,
            yanchor="top",
            title=dict(text=legend_title),
            font=dict(size=12),
            itemsizing="constant",
        ),
    )
    fig.update_annotations(font_size=13)
    wrapped = "<br>".join(textwrap.wrap(note, int(panel_w / 6.2)))
    fig.add_annotation(
        text=wrapped,
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        yshift=-80,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=10, color="#444444"),
        align="left",
    )
    write_map(fig, stem, panel_w, height, no_png)


def outline_country(fig):
    """Draw the US national border (country outline only; state borders are drawn by the geo settings above) on every
    geo panel of fig. Plotly fetches its built-in boundary topology when the figure is drawn, so this adds no data
    to the figure."""
    fig.update_geos(
        showcountries=True,
        countrycolor=OUTLINE_COLOUR,
        countrywidth=OUTLINE_WIDTH,
        resolution=OUTLINE_RESOLUTION,
    )


def write_map(fig, stem, width, height, no_png):
    """Single exit for every map: adds the national outline, then writes HTML and PNG."""
    outline_country(fig)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    base = MAPS_DIR / figstyle.dated(f"{TAG}_{stem}")
    fig.write_html(str(base.with_suffix(".html")), include_plotlyjs="cdn")
    print("  HTML:", base.with_suffix(".html").name)
    if no_png:
        return
    try:
        fig.write_image(str(base.with_suffix(".png")), width=width, height=height, scale=PNG_SCALE)
        print("  PNG: ", base.with_suffix(".png").name)
    except Exception as exc:  # kaleido missing or no browser: the HTML is still written
        print(f"  PNG failed ({type(exc).__name__}: {exc})")


def continuous_map(
    data,
    panels,
    stem,
    title,
    note,
    unit,
    ncols,
    zmin=None,
    zmax=None,
    cscale=SEQ,
    hover_label=None,
    no_png=False,
    quantiles=(0.01, 0.99),
):
    """Shared continuous scale. Default range = the given quantiles over every panel together
    (consistent within a map)."""
    vals = pd.concat([z for _, z in panels]).dropna()
    zmin = float(vals.quantile(quantiles[0])) if zmin is None else zmin
    zmax = float(vals.quantile(quantiles[1])) if zmax is None else zmax
    cb = dict(
        title=dict(text=unit, side="top"),
        orientation="h",
        thickness=12,
        len=0.35,
        x=0.5,
        xanchor="center",
        y=-0.01,
        yanchor="top",
        outlinewidth=0,
    )
    facet_map(
        data,
        panels,
        stem,
        title,
        note + f" Colour range {zmin:.2f} to {zmax:.2f}, shared across panels; values beyond are clipped.",
        cscale,
        zmin,
        zmax,
        cb,
        ncols,
        hover_label=hover_label or unit,
        no_png=no_png,
    )


def categorical_map(data, panels, cats, stem, title, note, ncols, legend_title, no_png=False):
    """panels hold integer category codes 0..k-1 (NaN = outside the sample); cats = [(label, colour)]."""
    k = len(cats)
    scale = []
    for j, (_, col) in enumerate(cats):
        scale += [[j / k, col], [(j + 1) / k, col]]
    facet_map(
        data,
        panels,
        stem,
        title,
        note,
        scale,
        -0.5,
        k - 0.5,
        None,
        ncols,
        no_png=no_png,
        legend=cats,
        legend_title=legend_title + ":  ",
        text=lambda v: cats[int(v)][0],
    )


def bin_codes(s, edges):
    """Integer codes: 0 = value <= edges[0], 1 = (edges[0], edges[1]], ..., len(edges) = above the last edge."""
    return pd.cut(s, bins=[-np.inf] + list(edges) + [np.inf], labels=False).astype("float")


def cohort_panel(cohort, in_sample):
    """Codes for a cohort-year map: 0..3 = cohort in census waves after the first, 4 = never treated, NaN outside."""
    waves = [w for w in CENSUS_YEARS if w > min(CENSUS_YEARS)]
    code = cohort.map({w: j for j, w in enumerate(waves)}).where(cohort.notna(), len(waves))
    return code.where(in_sample), waves


def wave_colours(waves):
    return sample_colorscale(SEQ, list(np.linspace(0.05, 0.9, len(waves))))


# =============================================================================
# Sections moved from script2f-maps.py
# =============================================================================
def section_fsis_density(data, no_png):
    """FSIS establishments per 10,000 residents, 2017-2023 (script2f section 1; four figures)."""
    f = data.fsis[data.fsis["year"].isin(FSIS_YEARS)].copy()
    treatments = {
        "Total FSIS establishments": ("n_unique_establishments_fsis", "Z1_fsis_total"),
        "Slaughterhouse establishments": ("n_slaughterhouse_present_establishments_fsis", "Z2_fsis_slaughter"),
        "Meat slaughter establishments": ("n_meat_slaughter_establishments_fsis", "Z3_fsis_meat"),
        "Poultry slaughter establishments": ("n_poultry_slaughter_establishments_fsis", "Z4_fsis_poultry"),
    }
    for label, (col, stub) in treatments.items():
        f["_v"] = log_per10k(f[col], f["population"])
        panels = [(str(y), f.loc[f["year"] == y].set_index("fips")["_v"]) for y in FSIS_YEARS]
        continuous_map(
            data,
            panels,
            f"fsis_density_{stub}_by_year",
            f"{label}: log(establishments per 10,000 residents + 1), 2017-2023",
            data.note(
                "Source: FSIS establishment records aggregated to county (no plant-level data shown)", data.fsis_sample
            ),
            "log(n per 10k + 1)",
            ncols=4,
            hover_label="log(n/10k+1)",
            no_png=no_png,
        )


MENTAL_OUTCOMES = {  # label, column, facet years, unit (script2f section 2)
    "poor_mental_health_days": ("Poor mental health days", [2012, 2014, 2016, 2018, 2020], "days per month (CHR)"),
    "excessive_drinking_per100k": ("Excessive drinking", [2012, 2014, 2016, 2018, 2020], "per 100,000 (CHR)"),
    "frequent_mental_distress_per100k": (
        "Frequent mental distress",
        [2017, 2018, 2019, 2020, 2021],
        "per 100,000 (CHR)",
    ),
    "crude_rate_from_census_pop": (
        "Deaths of despair, crude rate",
        [2010, 2013, 2016, 2018, 2020],
        "per 100,000 (own, census pop.)",
    ),
    "mental_health_providers_per100k": ("Mental health providers", [2013, 2015, 2017, 2019, 2021], "per 100,000 (CHR)"),
}


def section_outcomes(data, no_png):
    """Mental-health outcome values in five representative years (script2f section 2)."""
    for col, (label, years, unit) in MENTAL_OUTCOMES.items():
        if col not in data.df.columns:
            print(f"  skipped {col}: not in panel")
            continue
        d = data.df.assign(_v=to_numeric_series(data.df[col]))
        years = [y for y in years if d.loc[d["year"] == y, "_v"].notna().any()]
        panels = [(f"{y} (n={int(d.loc[d['year'] == y, '_v'].notna().sum()):,})", data.at(y, "_v", d)) for y in years]
        continuous_map(
            data,
            panels,
            f"outcomes_{col}_values_facet_5yr",
            f"{label}, {unit}",
            data.note("Source: County Health Rankings / CDC WONDER / Census (see variable register)"),
            unit,
            ncols=3,
            quantiles=(0.02, 0.98),
            no_png=no_png,
        )


def first_difference_panels(d, col, years):
    d = d.assign(_v=to_numeric_series(d[col])).sort_values(["fips", "year"])
    d["_d"] = d.groupby("fips")["_v"].diff()
    return d, [y for y in years if d.loc[d["year"] == y, "_d"].notna().any()]


def section_first_diff(data, no_png):
    """Year-to-year change in CAFO operations and FSIS slaughterhouses (script2f section 3), with its summary CSV."""
    specs = [
        (
            "CAFO total operations",
            data.df,
            "cafo_total_ops_all_animals",
            "cafo_total_ops",
            [2007, 2012, 2017, 2022],
            "Source: USDA Census of Agriculture (census years; inter-census years carried forward in the panel)",
            None,
        ),
        (
            "FSIS slaughterhouse-present establishments",
            data.fsis,
            "n_slaughterhouse_present_establishments_fsis",
            "fsis_slaughter",
            [2018, 2019, 2020, 2021, 2022, 2023],
            "Source: FSIS establishment records aggregated to county",
            data.fsis_sample,
        ),
    ]
    summaries = []
    for label, src, col, stub, years, source, sample in specs:
        d, years = first_difference_panels(src, col, years)
        titles, panels = [], []
        for y in years:
            z = d.loc[d["year"] == y].set_index("fips")["_d"]
            n, nz = int(z.notna().sum()), int((z.dropna() != 0).sum())
            panels.append((f"{y} (n={n:,}, non-zero={nz / n * 100:.1f}%)", z))
            x = z.dropna()
            summaries.append(
                {
                    "variable": col,
                    "label": label,
                    "year": y,
                    "n_counties_with_diff": n,
                    "n_nonzero_diff": nz,
                    "share_nonzero_diff_pct": nz / n * 100,
                    "mean_diff": x.mean(),
                    "std_diff": x.std(),
                    "p05_diff": x.quantile(0.05),
                    "p50_diff": x.quantile(0.5),
                    "p95_diff": x.quantile(0.95),
                }
            )
        m = pd.concat([z for _, z in panels]).abs().quantile(0.98)
        m = max(float(m), 1e-6)
        continuous_map(
            data,
            panels,
            f"first_diff_{stub}_first_diff_facet",
            f"{label}: change from the previous year",
            data.note(source, sample),
            "change in count",
            ncols=3 if len(years) > 4 else 2,
            zmin=-m,
            zmax=m,
            cscale=DIVERGING_7,
            hover_label="change",
            no_png=no_png,
        )
    path = MAPS_DIR / figstyle.dated(f"{TAG}_first_diff_year_summary.csv")
    pd.DataFrame(summaries).to_csv(path, index=False)
    print("  CSV: ", path.name)


# =============================================================================
# Sections moved from script2b-final-visuals.py
# =============================================================================
def section_cafo_census(data, no_png):
    """CAFO operations per 10,000 residents by animal type (Figure G) and size class (Figure H), census years."""
    df = data.df
    families = {}
    for animal in ("Cattle", "Hogs", "Chickens"):
        tot = species_treatments(df, SPECIES[animal], ["small", "medium", "large"])["count"]
        families[f"G_{animal.lower()}_cafo_census_years"] = (
            f"{animal}",
            df.assign(_v=log_per10k(tot, df["population"])),
            "G",
        )
    for size, col in SIZE_COLS.items():
        families[f"H_{size.lower()}_cafo_census_years"] = (
            f"{size} operations (cattle + hogs + chickens)",
            df.assign(_v=log_per10k(df[col], df["population"])),
            "H",
        )
    for fam in ("G", "H"):  # one colour range per family so animals (sizes) are comparable
        members = {k: v for k, v in families.items() if v[2] == fam}
        zmax = float(
            pd.concat([v[1].loc[v[1]["year"].isin(CENSUS_YEARS), "_v"] for v in members.values()])
            .dropna()
            .quantile(0.99)
        )
        for stem, (label, d, _) in members.items():
            panels = [(str(y), data.at(y, "_v", d)) for y in CENSUS_YEARS]
            continuous_map(
                data,
                panels,
                f"cafo_census_{stem}",
                f"CAFO {label.lower()}: log(operations per 10,000 residents + 1)",
                data.note(
                    "Source: USDA Census of Agriculture, operations by size bin; "
                    "size panels sum cattle, hogs and chickens (panel small/medium/large_cafo; dairy and beef sit "
                    "inside cattle)"
                    if fam == "H"
                    else "Source: USDA Census of Agriculture, operations by size bin"
                ),
                "log(n per 10k + 1)",
                ncols=3,
                zmin=0,
                zmax=zmax,
                hover_label="log(n/10k+1)",
                no_png=no_png,
            )


def section_snapshot_2012(data, no_png):
    """Figures F1 and F2 of script2b: 2012 CAFO intensity and poor mental health days."""
    df = data.df
    d = df.assign(
        _cafo=log_per10k(df["cafo_total_ops_all_animals"], df["population"]),
        _mh=to_numeric_series(df["poor_mental_health_days"]),
    )
    for col, stem, title, unit in [
        ("_cafo", "F1_cafo_intensity_map_2012", "Total CAFO operations, 2012", "log(n per 10k + 1)"),
        ("_mh", "F2_poor_mental_health_map_2012", "Poor mental health days, 2012", "days per month"),
    ]:
        continuous_map(
            data,
            [("", data.at(2012, col, d))],
            f"snapshot_2012_{stem}",
            title,
            data.note("Source: USDA Census of Agriculture" if col == "_cafo" else "Source: County Health Rankings"),
            unit,
            ncols=1,
            quantiles=(0.01, 0.99),
            hover_label=unit,
            no_png=no_png,
        )


# =============================================================================
# New sections, one per treatment family
# =============================================================================
def entry_panels(data):
    """Never-large comparison group and first-entry cohort year, from the first-entry design (designs.load_designs)."""
    des = data.designs["first entry (clean controls)"]
    d = des["data"]
    cohort = d.groupby("fips")[des["cohort"]].first()
    in_sample = pd.Series(True, index=cohort.index)
    return cohort, in_sample


def section_entry(data, no_png):
    """Design 'first entry': counties whose large-dairy count goes from 0 to 1 or more, by census wave (map 1)."""
    cohort, in_sample = entry_panels(data)
    waves = [w for w in CENSUS_YEARS if w > min(CENSUS_YEARS)]
    panels = [
        (f"{w}: {int((cohort == w).sum())} counties enter", (cohort == w).astype(float).where(in_sample)) for w in waves
    ]
    source = "Source: USDA Census of Agriculture, large dairy = 500+ milk cows"
    sample = (
        f"first-entry sample = {len(cohort):,} counties (never large dairy {int(cohort.isna().sum()):,} + "
        f"entering {int(cohort.notna().sum()):,})"
    )
    categorical_map(
        data,
        panels,
        [("In sample, not entering this wave", NEUTRAL), ("Enters this wave", VERMILLION)],
        "entry_counties_entering_by_wave",
        "Counties entering large dairy (0 to 1+ operations) in each census wave",
        data.note(source, sample),
        2,
        "Large dairy",
        no_png=no_png,
    )
    code, waves = cohort_panel(cohort, in_sample)
    cats = [(f"Enters {w}", c) for w, c in zip(waves, wave_colours(waves))] + [
        ("Never large dairy (comparison)", NEUTRAL)
    ]
    categorical_map(
        data,
        [("", code)],
        cats,
        "entry_cohort_year",
        "First-entry cohort year (first census wave with a large dairy)",
        data.note(source, sample + "; counties with large dairy from the first wave are outside the sample"),
        1,
        "Cohort",
        no_png=no_png,
    )


def section_expansion(data, no_png):
    """Design 'first expansion': first census wave at which the large-dairy count rises, all rural counties (map 2)."""
    des = data.designs["first expansion"]
    cohort = des["data"].groupby("fips")[des["cohort"]].first()
    code, waves = cohort_panel(cohort, pd.Series(True, index=cohort.index))
    cats = [(f"Expands {w}", c) for w, c in zip(waves, wave_colours(waves))] + [("Never expands", NEUTRAL)]
    counts = ", ".join(f"{w}: {int((cohort == w).sum())}" for w in waves)
    categorical_map(
        data,
        [("", code)],
        cats,
        "expansion_cohort_year",
        "First-expansion cohort year (first census wave at which large-dairy operations rise)",
        data.note(
            "Source: USDA Census of Agriculture, large dairy = 500+ milk cows",
            f"{len(cohort):,} rural counties; counties per cohort {counts}; never {int(cohort.isna().sum()):,}",
        ),
        1,
        "Cohort",
        no_png=no_png,
    )


def count_map(data, panels, stem, title, source, ncols, no_png, sample=None):
    cats = list(zip(COUNT_BINS[1], COUNT_COLOURS))
    categorical_map(
        data,
        [(t, bin_codes(z, COUNT_BINS[0])) for t, z in panels],
        cats,
        stem,
        title,
        data.note(source, sample),
        ncols,
        "Large operations",
        no_png=no_png,
    )


def section_large_ops(data, no_png):
    """Large operations, total and per 10,000 residents, latest census year (map 3)."""
    y = LATEST_WAVE
    source = (
        f"Source: USDA Census of Agriculture {y}; largest size bin, cattle (incl. dairy and beef) + hogs + chickens"
    )
    total = data.at(y, "large_cafo")
    count_map(data, [("", total)], "large_ops_total", f"Large CAFO operations, {y}", source, 1, no_png)
    per10k = total / data.at(y, "population").replace(0, np.nan) * 10_000
    continuous_map(
        data,
        [("", per10k)],
        "large_ops_per10k",
        f"Large CAFO operations per 10,000 residents, {y}",
        data.note(source),
        "per 10,000 residents",
        ncols=1,
        zmin=0,
        hover_label="per 10k",
        no_png=no_png,
        quantiles=(0.0, 0.99),
    )


def section_species(data, no_png):
    """One panel per species, large operations in the latest census year (map 4)."""
    df, y = data.df, LATEST_WAVE
    panels = []
    for label, sp in SPECIES.items():
        c = species_treatments(df, sp, ["large"])["count"]
        s = data.at(y, "_c", df.assign(_c=c))
        panels.append((f"{label} (counties with any: {int((s > 0).sum()):,})", s))
    count_map(
        data,
        panels,
        "species_large_ops_by_species",
        f"Large operations by species, {y}",
        f"Source: USDA Census of Agriculture {y}. Cattle contains the dairy and beef herds (D-36)",
        3,
        no_png,
    )


def section_species_count(data, no_png):
    """Number of species with at least one large operation, by census wave (map 5). Species per D-36 (no cattle)."""
    df = data.df
    large = pd.concat({sp: species_treatments(df, sp, ["large"])["count"] > 0 for sp in COUNT_SPECIES.values()}, axis=1)
    df = df.assign(_n=large.sum(axis=1).astype(float))
    k = len(COUNT_SPECIES)
    cats = list(zip([str(i) for i in range(k + 1)], [ZERO] + sample_colorscale(SEQ, list(np.linspace(0.15, 0.95, k)))))
    panels = [
        (f"{w}: {int((data.at(w, '_n', df) >= 2).sum()):,} counties with 2+", data.at(w, "_n", df))
        for w in CENSUS_YEARS
    ]
    categorical_map(
        data,
        panels,
        cats,
        "species_count_species_with_large_ops",
        "Number of species with large operations (dairy, beef, hogs, chickens), by census wave",
        data.note(
            "Source: USDA Census of Agriculture. Species set per D-36; "
            "cattle excluded because it contains dairy and beef"
        ),
        3,
        "Species",
        no_png=no_png,
    )


def section_estimation_sample(data, no_png):
    """Poor mental health days estimation sample, both designs: treated by cohort, comparison, not in sample (map 6)."""
    rural = pd.Index(sorted(data.df["fips"].unique()))
    samples = {}
    for name, label in [("first expansion", "First expansion"), ("first entry (clean controls)", "First entry")]:
        des = data.designs[name]
        d = usable_sample(des["data"], ESTIMATION_OUTCOME, des["treat"], des["cohort"], ESTIMATION_CONTROLS)
        d["fips"] = d["fips"].astype(str).str.zfill(5)
        samples[label] = d
    # a wave gets a legend entry only if a treated county of a design is in the sample (needs a pre-period)
    waves = [
        w
        for w in CENSUS_YEARS
        if w > min(CENSUS_YEARS) and any((d.drop_duplicates("fips")["cohort"] == w).any() for d in samples.values())
    ]
    dropped = [w for w in CENSUS_YEARS if w > min(CENSUS_YEARS) and w not in waves]
    # codes 0..n_wave-1 = treated by wave, n_wave = comparison, n_wave + 1 = rural but not in the sample
    n_wave = len(waves)
    cats = [(f"Treated, cohort {w}", c) for w, c in zip(waves, wave_colours(waves))] + [
        ("Comparison (in sample, not treated)", NEUTRAL),
        ("Rural county, not in the sample", "#ffffff"),
    ]
    panels, years = [], set()
    for label, d in samples.items():
        cohort = d.groupby("fips")["cohort"].first()
        code = (
            cohort.map({w: j for j, w in enumerate(waves)})
            .where(cohort.notna(), n_wave)
            .reindex(rural)
            .fillna(n_wave + 1)
        )
        n_treated, n_comp = int((code < n_wave).sum()), int((code == n_wave).sum())
        per_wave = ", ".join(f"{w}: {int((code == j).sum())}" for j, w in enumerate(waves))
        print(
            f"  {label}: {len(cohort):,} counties in sample ({n_treated:,} treated [{per_wave}], {n_comp:,} "
            "comparison), "
            f"{int((code == n_wave + 1).sum()):,} rural counties not in sample; {len(d):,} county-years, "
            f"{d['year'].min()}-{d['year'].max()}"
        )
        panels.append((f"{label} (in sample: {n_treated:,} treated, {n_comp:,} comparison)", code))
        years |= {int(d["year"].min()), int(d["year"].max())}
    categorical_map(
        data,
        panels,
        cats,
        "estimation_sample_by_design",
        "Poor mental health days estimation sample, large dairy",
        data.note(
            "Source: USDA Census of Agriculture (large dairy = 500+ milk cows), County Health Rankings, "
            "annual controls file. "
            f"In sample = county-years {min(years)}-{max(years)} with the outcome, the treatment dummy and "
            "the 8 pre-committed "
            "controls non-missing and, for a treated county, an observed year before its cohort wave "
            "(designs.usable_sample, "
            "the call of script4d-triangulation.py, section estimators). Cohort = first census wave of expansion (first expansion) or entry (first entry; "
            "comparison = never large dairy)"
            + (
                f"; no county of cohort {', '.join(map(str, dropped))} is in the sample "
                f"(the outcome starts in {min(years)}, so none has a pre-period)"
                if dropped
                else ""
            )
            + ". A missing cafo_* count is 0 from 2002 (D-34), so a county without a record is a comparison "
            "county, not missing. "
            "FSIS is not used here; elsewhere in this script a county-year with no FSIS record is drawn as 0",
            f"{len(rural):,} rural counties; the first-entry design holds entering plus never-large "
            "counties, so other rural "
            "counties are drawn as not in the sample",
            land=False,
        )
        + " Very light grey: not a rural county of the panel.",
        2,
        "Status",
        no_png=no_png,
    )


def section_coverage(data, no_png):
    """Sample coverage: counties dropped by the metro filter (map 7). Rural = NCHS non-large-metro, by county-year."""
    g = data.key.groupby("fips")["non_large_metro"].agg(["sum", "count"])
    status = pd.Series(np.select([g["sum"] == g["count"], g["sum"] > 0], [0.0, 1.0], 2.0), index=g.index)
    n = status.value_counts()
    cats = [
        (f"Rural in every year (n={n.get(0.0, 0):,})", BLUE),
        (f"Rural in some years only (n={n.get(1.0, 0):,})", figstyle.PALETTE["orange"]),
        (f"Large metro in every year, dropped (n={n.get(2.0, 0):,})", NEUTRAL),
    ]
    years = f"{data.key['year'].min()}-{data.key['year'].max()}"
    categorical_map(
        data,
        [("", status)],
        cats,
        "coverage_metro_filter",
        "Sample coverage: counties kept and dropped by the metro filter",
        data.note(
            "Source: NCHS urban-rural classification (rural-key file); rural = not in a large metro area",
            f"{len(g):,} counties in the key, {years}; panel keeps a county-year only when rural",
            land=False,
        ),
        1,
        "Sample",
        no_png=no_png,
    )


def section_fsis_entry(data, no_png):
    """FSIS slaughterhouse presence and entry, county level, 2017-2023 (map 8). Entry is only observable from 2018."""
    f = data.fsis[data.fsis["year"].isin(FSIS_YEARS)]
    pres = (f["n_slaughterhouse_present_establishments_fsis"] > 0).groupby(f["fips"])
    n_years = pres.sum().astype(float)
    first = f[f["n_slaughterhouse_present_establishments_fsis"] > 0].groupby("fips")["year"].min()
    first = first.reindex(n_years.index)
    source = "Source: FSIS establishment records aggregated to county (no plant-level data shown)"
    sample = data.fsis_sample
    bins = ([0, 2, 4, 6], ["0 years", "1-2 years", "3-4 years", "5-6 years", "All 7 years"])
    colours = [ZERO] + sample_colorscale(SEQ, [0.2, 0.45, 0.7, 0.95])
    categorical_map(
        data,
        [("", bin_codes(n_years, bins[0]))],
        list(zip(bins[1], colours)),
        "fsis_entry_years_present",
        "Years with a slaughterhouse-present FSIS establishment, 2017-2023",
        data.note(source, sample),
        1,
        "Years present",
        no_png=no_png,
    )
    ent = FSIS_YEARS[1:]
    code = first.map({y: j + 1 for j, y in enumerate(ent)}).where(first.notna(), np.nan)
    code[first == FSIS_YEARS[0]] = 0
    code[first.isna()] = len(ent) + 1
    cats = (
        [(f"Present in {FSIS_YEARS[0]} (entry date not observed)", "#4d4d4d")]
        + [(f"Enters {y}", c) for y, c in zip(ent, sample_colorscale(SEQ, list(np.linspace(0.05, 0.9, len(ent)))))]
        + [("No slaughterhouse 2017-2023", ZERO)]
    )
    categorical_map(
        data,
        [("", code)],
        cats,
        "fsis_entry_first_year_present",
        "First year with a slaughterhouse-present FSIS establishment",
        data.note(source, sample),
        1,
        "First year",
        no_png=no_png,
    )


def section_wave_change(data, no_png):
    """Change in large operations between consecutive census waves, cattle + hogs + chickens (map 9)."""
    df = data.df
    panels = []
    for a, b in zip(CENSUS_YEARS[:-1], CENSUS_YEARS[1:]):
        d = data.at(b, "large_cafo") - data.at(a, "large_cafo")
        panels.append((f"{a} to {b}: {int((d > 0).sum()):,} up, {int((d < 0).sum()):,} down", d))
    cats = list(zip(DELTA_BINS[1], DIVERGING_7))
    categorical_map(
        data,
        [(t, bin_codes(z, DELTA_BINS[0])) for t, z in panels],
        cats,
        "wave_change_large_ops",
        "Change in the number of large CAFO operations between census waves",
        data.note(
            "Source: USDA Census of Agriculture; largest size bin, cattle (incl. dairy and beef) + hogs + chickens"
        ),
        2,
        "Change in count",
        no_png=no_png,
    )


SECTIONS = {  # run order: moved sections first, then one per treatment family
    "fsis_density": section_fsis_density,
    "outcomes": section_outcomes,
    "first_diff": section_first_diff,
    "cafo_census": section_cafo_census,
    "snapshot_2012": section_snapshot_2012,
    "entry": section_entry,
    "expansion": section_expansion,
    "large_ops": section_large_ops,
    "species": section_species,
    "species_count": section_species_count,
    "estimation_sample": section_estimation_sample,
    "coverage": section_coverage,
    "fsis_entry": section_fsis_entry,
    "wave_change": section_wave_change,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), help="run only these sections")
    ap.add_argument("--list", action="store_true", help="list sections and exit")
    ap.add_argument("--no-png", action="store_true", help="write HTML only (skips the kaleido export)")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:20s} {v.__doc__.splitlines()[0]}" for k, v in SECTIONS.items()))
        return
    data = Data()
    print(
        f"Panel: {Path(data.panel_path).name} | FSIS: {Path(data.fsis_path).name} "
        f"| rural key: {Path(data.key_path).name}"
    )
    for name in args.only or SECTIONS:
        print(f"\n--- {name} ---")
        SECTIONS[name](data, args.no_png)
    print(f"\nMaps written to {MAPS_DIR}")


if __name__ == "__main__":
    main()
