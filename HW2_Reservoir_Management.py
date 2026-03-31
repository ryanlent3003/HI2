"""
HW2 Reservoir Management Report
================================
McPhee Reservoir — Dolores River, Colorado

Runs Parts 1-5 end-to-end without any user input.
All required output folders (files/, figures/) are created automatically.
Figures are saved to figures/ as PNG files.
Script is fully reproducible; re-running overwrites existing outputs.

Usage
-----
    python HW2_Reservoir_Management.py
"""


# ── HW2: Reservoir Management Report ----------------------------
"""
# HW2: Reservoir Management Report
## McPhee Reservoir — Dolores River, Colorado

**Assignment:** Provide reservoir management recommendations to balance water storage and mitigate flooding potential for Water Year 2024 (previous water year), evaluated as of April 1, 2025.

**Study System:**
- **Reservoir:** McPhee Reservoir (Montezuma County, Colorado)
- **River:** Dolores River (Upper Colorado River Basin)
- **Key Decision Date:** April 1, 2025

---

### Report Sections
1. **Area of Interest** — Watershed map, station locations, and basin description *(this notebook)*
2. Snow Water Equivalent (SWE) Analysis — SNOTEL data, seasonal SWE curves
3. Streamflow Analysis — USGS inflow discharge, hydrograph interpretation
4. Reservoir Storage — Storage volume, percent capacity through the water year
5. Management Recommendations — Decision synthesis for April 1, 2025

> **Environment note:** Run from the `hyriver` conda environment (`conda activate hyriver`)
"""

# ── --- ---------------------------------------------------------
"""
---
# Part 1: Area of Interest

This section produces:
1. An **interactive map** showing the watershed boundary, USGS streamflow gauge, and SNOTEL station(s)
2. A **static map** suitable for the written report
3. Computed **watershed characteristics** (area, elevation range)
4. A written **watershed description**
"""

# ── 1.1 Import Libraries and Configure Environment --------------
"""
## 1.1 Import Libraries and Configure Environment
"""

import os
import io
import warnings
import datetime
import requests

# Resolve all paths relative to this script's directory so the script is
# portable and can be run from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import contextily as cx

import folium
from folium.plugins import MousePosition
from pynhd import NLDI

try:
    from IPython.display import display
except Exception:
    # Fallback: plain print() when running outside a Jupyter kernel.
    def display(obj):
        print(obj)

warnings.filterwarnings('ignore')
print('Libraries loaded successfully.')

# ── 1.2 Study System Parameters ---------------------------------
"""
## 1.2 Study System Parameters

Adjust the parameters below if you change reservoir/gauge selection.
"""

# ============================================================
# STUDY SYSTEM CONFIGURATION
# ============================================================

# USGS gauge upstream of McPhee Reservoir (natural, unregulated inflow proxy)
# 09165000 = Dolores River at Dolores, CO  |  drainage area: ~1,596 mi²
USGS_GAGE_ID   = '09165000'

RESERVOIR_NAME = 'McPhee Reservoir'
RIVER_NAME     = 'Dolores River'
STATE          = 'Colorado'
STATE_ABB      = 'CO'

# Approximate reservoir center (Folium marker only)
RESERVOIR_LAT  = 37.593
RESERVOIR_LON  = -108.540

# Water Year under analysis (complete water year BEFORE the decision date)
WY            = 2024   # WY2024 = Oct 1 2023 – Sep 30 2024
DECISION_DATE = '2025-04-01'

# Output directories — created automatically
FILES_DIR   = os.path.join(BASE_DIR, 'files')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FILES_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f'Reservoir  : {RESERVOIR_NAME}')
print(f'River      : {RIVER_NAME}, {STATE}')
print(f'USGS Gauge : {USGS_GAGE_ID}')
print(f'Water Year : WY{WY}  (Oct 1 {WY-1} – Sep 30 {WY})')
print(f'Decision   : {DECISION_DATE}')

# ── 1.3 Watershed Delineation -----------------------------------
"""
## 1.3 Watershed Delineation

The [USGS Hydro Network Linked Data Index (NLDI)](https://labs.waterdata.usgs.gov/about-nldi/index.html) service delineates the contributing drainage area upstream of the specified gauge using the National Hydrography Dataset (NHDPlus). The resulting polygon defines exactly which land area drains to McPhee Reservoir.
"""

nldi = NLDI()

print('Delineating watershed from NLDI...', end=' ')
basin = nldi.get_basins(USGS_GAGE_ID)
print('done')

# Save for future use
basin.to_file(f'{FILES_DIR}/{RESERVOIR_NAME.replace(" ", "_")}_basin.shp')

# Get the USGS gauge point feature
site_feature = nldi.getfeature_byid('nwissite', f'USGS-{USGS_GAGE_ID}')

# Get upstream main-stem flowlines
print('Fetching upstream flowlines...', end=' ')
upstream_network = nldi.navigate_byid(
    'nwissite', f'USGS-{USGS_GAGE_ID}',
    'upstreamMain', 'flowlines', distance=9999
)
print('done')

print(f'\nBasin CRS            : {basin.crs}')
print(f'Basin geometry type  : {basin.geometry.geom_type.values[0]}')

# ── 1.4 Identify SNOTEL Stations Within the Watershed -----------
"""
## 1.4 Identify SNOTEL Stations Within the Watershed

SNOTEL (SNOwpack TELemetry) stations are operated by the USDA Natural Resources Conservation Service (NRCS) and measure daily snow water equivalent (SWE), precipitation, temperature, and other variables. Here, all active SNOTEL stations whose location falls inside the delineated watershed are identified.
"""

# Load the full SNOTEL/CCSS station catalogue (maintained by E. Gagli, UW)
SNOTEL_URL = 'https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson'

print('Loading SNOTEL station catalogue...', end=' ')
all_stations_gdf = gpd.read_file(SNOTEL_URL).set_index('code')
all_stations_gdf = all_stations_gdf[all_stations_gdf['csvData'] == True]
print(f'done ({len(all_stations_gdf)} total active stations)')

# Spatial join: keep only stations within the basin polygon
basin_poly = basin.geometry.iloc[0]
gdf_in_basin = all_stations_gdf[all_stations_gdf.geometry.within(basin_poly)].copy()
gdf_in_basin.reset_index(drop=False, inplace=True)

# Standardize date columns to strings
for col in ['beginDate', 'endDate']:
    if col in gdf_in_basin.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf_in_basin[col]):
            gdf_in_basin[col] = gdf_in_basin[col].dt.strftime('%Y-%m-%d')
        else:
            gdf_in_basin[col] = pd.to_datetime(gdf_in_basin[col], errors='coerce').dt.strftime('%Y-%m-%d')

print(f'\nSNOTEL stations inside {RESERVOIR_NAME} watershed: {len(gdf_in_basin)}')
display_cols = [c for c in ['code', 'name', 'elevation', 'beginDate', 'endDate'] if c in gdf_in_basin.columns]
print(gdf_in_basin[display_cols].to_string(index=False))

# ── 1.5 Interactive Map -----------------------------------------
"""
## 1.5 Interactive Map

The interactive map below shows:
- **Blue shaded polygon** — Delineated watershed boundary
- **Blue water droplet icon** — USGS streamflow gauge (inlet)
- **Orange snowflake icon(s)** — SNOTEL station(s)
- **Red tint icon** — McPhee Reservoir (approximate center)

> *Hover over a marker for its identifier. Toggle layers with the control in the upper-right corner.*
"""

# Determine map center
minx, miny, maxx, maxy = basin.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

# Base map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles='http://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Sources: National Geographic | USGS | NRCS',
    control_scale=True
)
MousePosition().add_to(m)

# Watershed boundary
folium.GeoJson(
    basin.to_crs(epsg=4326).to_json(),
    name='Watershed Boundary',
    style_function=lambda x: {
        'color': 'darkblue', 'weight': 2.5,
        'fillColor': 'steelblue', 'fillOpacity': 0.25
    }
).add_to(m)

# Main-stem flowlines
folium.GeoJson(
    upstream_network.to_crs(epsg=4326).to_json(),
    name='Main-stem Flowline',
    style_function=lambda x: {'color': 'royalblue', 'weight': 2}
).add_to(m)

# USGS gauge
folium.GeoJson(
    site_feature.to_crs(epsg=4326).to_json(),
    name=f'USGS Gauge {USGS_GAGE_ID}',
    tooltip=folium.GeoJsonTooltip(['identifier']),
    marker=folium.Marker(icon=folium.Icon(color='blue', icon='water', prefix='fa'))
).add_to(m)

# SNOTEL stations
if len(gdf_in_basin) > 0:
    snotel_cols = [c for c in ['code', 'name', 'elevation'] if c in gdf_in_basin.columns]
    folium.GeoJson(
        gdf_in_basin.to_json(),
        name='SNOTEL Station(s)',
        tooltip=folium.GeoJsonTooltip(snotel_cols),
        marker=folium.Marker(icon=folium.Icon(color='orange', icon='snowflake', prefix='fa'))
    ).add_to(m)

# Reservoir approximate center
folium.Marker(
    [RESERVOIR_LAT, RESERVOIR_LON],
    popup=folium.Popup(f'<b>{RESERVOIR_NAME}</b><br>Dolores River, CO<br>Capacity: ~381,000 ac-ft', max_width=200),
    tooltip=RESERVOIR_NAME,
    icon=folium.Icon(color='red', icon='tint', prefix='fa')
).add_to(m)

folium.LayerControl().add_to(m)
m.fit_bounds(m.get_bounds())
m

# ── 1.6 Static Map (for Report) ---------------------------------
"""
## 1.6 Static Map (for Report)

A two-panel publication-quality map:
- **Left panel** — Regional context (watershed in the landscape)
- **Right panel** — Watershed detail with gauge and SNOTEL markers
"""

# Project to Web Mercator for contextily basemaps
basin_wm     = basin.to_crs(epsg=3857)
site_wm      = site_feature.to_crs(epsg=3857)
network_wm   = upstream_network.to_crs(epsg=3857)
if len(gdf_in_basin) > 0:
    snotel_wm = gdf_in_basin.to_crs(epsg=3857)

# Basemap source (ESRI World Topo)
BASEMAP = cx.providers.Esri.WorldTopoMap

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('white')

# ── Left panel: CONUS context with watershed location ──────────────────────
ax1 = axes[0]
try:
    states_url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json'
    states = gpd.read_file(states_url)
    state_code_col = 'id' if 'id' in states.columns else None
    if state_code_col is None:
        raise ValueError('Unable to identify state abbreviation column in CONUS source.')

    conus_states = states.loc[~states[state_code_col].isin(['AK', 'HI', 'PR'])].to_crs(epsg=3857)
    conus_states.plot(ax=ax1, color='whitesmoke', edgecolor='lightgray', linewidth=0.5, zorder=1)
    conus_states.boundary.plot(ax=ax1, color='gray', linewidth=0.5, zorder=2)
except Exception:
    # Fallback if CONUS state layer is unavailable.
    basin_wm.plot(ax=ax1, facecolor='white', edgecolor='white', alpha=0)

# Highlight watershed location inside CONUS
basin_wm.plot(ax=ax1, facecolor='crimson', edgecolor='darkred', linewidth=1.2, alpha=0.9, zorder=4,
              label='McPhee Watershed')
basin_centroid = basin_wm.geometry.centroid
gpd.GeoDataFrame(geometry=basin_centroid, crs=basin_wm.crs).plot(
    ax=ax1, color='gold', edgecolor='black', markersize=90, marker='*', zorder=5
)

ax1.set_title(f'{RESERVOIR_NAME}\nCONUS Context', fontsize=11, fontweight='bold', pad=8)
ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
ax1.set_axis_off()

# ── Right panel: watershed detail ──────────────────────────────────────────
ax2 = axes[1]
basin_wm.plot(ax=ax2, facecolor='steelblue', edgecolor='darkblue',
              linewidth=2, alpha=0.35, zorder=3)
network_wm.plot(ax=ax2, color='royalblue', linewidth=1.5, zorder=4)
cx.add_basemap(ax2, source=BASEMAP, zoom=10, zorder=1)

site_wm.plot(ax=ax2, color='royalblue', markersize=150, zorder=7,
             marker='^', label=f'USGS {USGS_GAGE_ID}')
if len(gdf_in_basin) > 0:
    snotel_wm.plot(ax=ax2, color='darkorange', markersize=150, zorder=7,
                   marker='*', label='SNOTEL Station(s)')
    for _, row in snotel_wm.iterrows():
        station_name = row.get('name', row.get('code', ''))
        ax2.annotate(
            station_name,
            (row.geometry.x, row.geometry.y),
            textcoords='offset points', xytext=(6, 6),
            fontsize=7.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')
        )
gdf_resv = gpd.GeoDataFrame(
    {'label': [RESERVOIR_NAME]},
    geometry=gpd.points_from_xy([RESERVOIR_LON], [RESERVOIR_LAT]),
    crs='EPSG:4326'
).to_crs(epsg=3857)
gdf_resv.plot(ax=ax2, color='crimson', markersize=150, zorder=7,
              marker='D', label=RESERVOIR_NAME)

ax2.set_title(f'{RESERVOIR_NAME} Watershed\n{RIVER_NAME} at USGS {USGS_GAGE_ID} — Detail',
              fontsize=11, fontweight='bold', pad=8)
ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax2.set_axis_off()

plt.suptitle(
    f'Area of Interest: {RESERVOIR_NAME}, {STATE}\nWY{WY} Reservoir Management Analysis',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()

out_path = f'{FIGURES_DIR}/01_watershed_map.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Map saved -> {out_path}')

# ── 1.7 Watershed Characteristics -------------------------------
"""
## 1.7 Watershed Characteristics

Quantitative basin attributes are computed below using:
- **Area** — reprojected to UTM Zone 13N (EPSG:32613) for accurate area in km²
- **Elevation** — 3DEP 30-m DEM retrieved via `py3dep`
"""

# ── Watershed area ─────────────────────────────────────────────────────────
basin_utm  = basin.to_crs(epsg=32613)  # UTM Zone 13N, WGS 84 (covers Colorado)
area_m2    = basin_utm.geometry.area.sum()
area_km2   = area_m2 / 1e6
area_mi2   = area_km2 * 0.3861
area_acres = area_km2 * 247.105

# ── Compute watershed centroid (geographic) ────────────────────────────────
centroid = basin.to_crs(epsg=4326).geometry.centroid.iloc[0]

print('=' * 55)
print(f'  WATERSHED CHARACTERISTICS: {RESERVOIR_NAME}')
print('=' * 55)
print(f'  Area               : {area_km2:>8.0f} km²')
print(f'                     : {area_mi2:>8.0f} mi²')
print(f'  Centroid           : {centroid.y:.3f}°N, {centroid.x:.3f}°W')
print(f'  USGS Gauge         : {USGS_GAGE_ID}')
print(f'  SNOTEL stations    : {len(gdf_in_basin)}')

# ── Elevation statistics from 3DEP ────────────────────────────────────────
try:
    import py3dep
    print('\nFetching 30-m DEM from 3DEP (this may take ~1 min)...', end=' ')
    # py3dep ≥0.16 uses get_dem(); earlier versions use get_map()
    try:
        dem = py3dep.get_dem(basin.geometry.iloc[0], resolution=30, crs=basin.crs)
    except AttributeError:
        dem = py3dep.get_map('DEM', basin.geometry.iloc[0], resolution=30, crs=basin.crs)
    print('done')

    elev = dem.values.flatten()
    elev = elev[~np.isnan(elev)]

    print(f'\n  Elevation (3DEP 30-m DEM):')
    print(f'    Min  : {elev.min():>6.0f} m  ({elev.min()*3.2808:>6.0f} ft)')
    print(f'    Max  : {elev.max():>6.0f} m  ({elev.max()*3.2808:>6.0f} ft)')
    print(f'    Mean : {elev.mean():>6.0f} m  ({elev.mean()*3.2808:>6.0f} ft)')
    print(f'    Range: {elev.max()-elev.min():>6.0f} m')

    # Save elevation stats for later reference
    elev_stats = {
        'min_m': float(elev.min()), 'max_m': float(elev.max()),
        'mean_m': float(elev.mean()), 'range_m': float(elev.max()-elev.min())
    }

except Exception as e:
    print(f'\n  py3dep elevation retrieval failed: {e}')
    print('  Using literature values for the Dolores River headwaters:')
    print(f'    Min  :  ~2,580 m  (~8,460 ft)  — dam site near Dolores, CO')
    print(f'    Max  :  ~4,290 m  (~14,070 ft) — San Juan Mountain headwaters')
    print(f'    Range:  ~1,710 m')
    elev_stats = {'min_m': 2580, 'max_m': 4290, 'mean_m': 3100, 'range_m': 1710}

# SNOTEL station summary
if len(gdf_in_basin) > 0:
    print('\n  SNOTEL Stations inside watershed:')
    for _, row in gdf_in_basin.iterrows():
        elev_col  = row.get('elevation', 'N/A')
        name_col  = row.get('name', row.get('code', 'N/A'))
        code_col  = row.get('code', 'N/A')
        state_col = row.get('state', STATE_ABB)
        print(f'    {name_col}  (ID: {code_col}_{state_col}, Elev: {elev_col} m)')

print('=' * 55)

# ── 1.8 Hypsometric Curve (Optional) ----------------------------
"""
## 1.8 Hypsometric Curve (Optional)

A *hypsometric curve* shows what fraction of the watershed lies above each elevation — a useful descriptor of watershed shape and snowpack potential.
"""

try:
    # Uses 'elev' array from the 3DEP cell above
    sorted_elev = np.sort(elev)[::-1]
    fraction    = np.linspace(0, 1, len(sorted_elev))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fraction * 100, sorted_elev, color='steelblue', linewidth=2)
    ax.set_xlabel('Percent area above elevation (%)', fontsize=11)
    ax.set_ylabel('Elevation (m)', fontsize=11)
    ax.set_title(f'Hypsometric Curve\n{RESERVOIR_NAME} Watershed', fontsize=11, fontweight='bold')
    ax.axhline(elev_stats['mean_m'], color='darkorange', linestyle='--', linewidth=1.5,
               label=f"Mean elevation: {elev_stats['mean_m']:.0f} m")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    hyps_path = f'{FIGURES_DIR}/02_hypsometric_curve.png'
    plt.savefig(hyps_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Hypsometric curve saved → {hyps_path}')

except NameError:
    print('Elevation data not available — skipping hypsometric curve.')
    print('Run the DEM cell (1.7) first to populate the `elev` array.')

# ── --- ---------------------------------------------------------
"""
---
## 1.9 Watershed Description

### McPhee Reservoir and the Upper Dolores River Watershed

**Location and Reservoir Overview**

McPhee Reservoir is located in Montezuma County, southwestern Colorado (approximately 37.5°N, 108.5°W), on the Dolores River within the Upper Colorado River Basin (HUC-8: 14080104). The reservoir was impounded by McPhee Dam, which was completed in 1985, and has a total storage capacity of approximately **381,000 acre-feet (470 Mm³)**. It is the second-largest reservoir in Colorado and the primary surface water supply for agricultural users in the Dolores Water Conservancy District and for municipal/industrial users including the Ute Mountain Ute Tribe.

**Watershed Area and Topography**

The contributing drainage area upstream of USGS gauge 09165000 (inlet proxy) encompasses approximately **1,590–1,700 km² (~620–660 mi²)**. The basin exhibits extreme relief, extending from the dam site near the town of Dolores (~2,580 m / 8,460 ft) to the high peaks of the **San Juan Mountains** to the east and northeast (~4,290 m / 14,070 ft). The main-stem Dolores River, along with its primary tributaries (West Fork Dolores River, Bear Creek, Fish Creek), drains this rugged alpine and subalpine terrain carved by glacial and fluvial processes.

**Climate and Hydrology**

The catchment is characterized by a **semi-arid continental climate** with strong elevation-dependent precipitation gradients. Mean annual precipitation ranges from ~35–45 cm at lower elevations to >100 cm at high-elevation sites. The region receives orographically enhanced snowfall from November through April, with the seasonal snowpack serving as the dominant water-storage mechanism. **Snowmelt runoff (April–June) generates 70–85% of annual streamflow**, making April 1 snow water equivalent (SWE) a critical predictor of seasonal inflow to McPhee Reservoir. A secondary precipitation pulse from the North American Monsoon (July–September) produces convective storms that can generate localized flash flooding.

**Elevation Zones and Vegetation**

| Elevation Zone | Approximate Range | Dominant Vegetation |
|---|---|---|
| Alpine / Tundra | > 3,500 m (11,500 ft) | Rocky fellfields, cushion plants, alpine grasses |
| Subalpine | 3,000–3,500 m (9,800–11,500 ft) | Engelmann spruce, subalpine fir, Krummholz |
| Montane | 2,600–3,000 m (8,500–9,800 ft) | Quaking aspen, ponderosa pine, Douglas-fir |
| Transition (near dam) | ~2,580 m | Piñon-juniper woodland, sagebrush shrubland |

**Hydrologic Monitoring Network**

- **USGS 09165000** — *Dolores River at Dolores, CO*: Period of record 1895–present; measures natural (unregulated) daily streamflow just above McPhee Reservoir. Drainage area: ~1,596 mi².
- **SNOTEL Network** (see stations identified in section 1.4): High-elevation stations in the San Juan Mountains track SWE accumulation and melt throughout the season. These data drive seasonal inflow forecasts.

**Relevance to Reservoir Management (April 1)**

April 1 is historically the date of **peak or near-peak snowpack** across the Dolores River watershed. SWE measurements on this date are used by the Natural Resources Conservation Service (NRCS) in official *water supply forecasts*, which directly inform McPhee's operational releases and storage targets. To avoid downstream flooding while maximizing end-of-season storage, operators must balance: (1) current reservoir storage level, (2) observed April 1 SWE relative to the median, and (3) expected snowmelt timing derived from temperature outlooks.
"""

# ── --- ---------------------------------------------------------
"""
---
## Summary — Part 1 Complete

| Item | Value |
|---|---|
| Reservoir | McPhee Reservoir, Dolores River, CO |
| USGS Inlet Gauge | 09165000 — Dolores River at Dolores, CO |
| SNOTEL Stations | See section 1.4 output |
| Basin Area | Computed in section 1.7 (≈ 1,590–1,700 km²) |
| Elevation Range | ≈ 2,580–4,290 m (computed/3DEP in section 1.7) |
| Water Year Analyzed | WY2024 (Oct 1, 2023 – Sep 30, 2024) |
| Decision Date | April 1, 2025 |

**Next Part: Proceed to Part 2 — SWE Analysis** to retrieve SNOTEL time series and compare WY2024 SWE to the period-of-record median.
"""

# ── --- ---------------------------------------------------------
"""
---

# Part 2: SWE Analysis for SNOTEL Site(s)

This section retrieves daily SNOTEL snow water equivalent (SWE) observations for all stations inside the McPhee Reservoir watershed, summarizes station characteristics, and compares April 1, 2025 conditions to the historical distribution.

Outputs in this section include:

- A table describing the SNOTEL site(s)
- A multi-panel climatology figure showing the historical SWE envelope and WY2025 observations
- A basin-mean SWE index for April 1, 2025 relative to the historical mean and median
"""

from dataretrieval import nwis

# Analysis window and unit-conversion constants used across SWE/flow sections.
FORECAST_DATE = pd.Timestamp(DECISION_DATE)
FORECAST_WY = FORECAST_DATE.year + 1 if FORECAST_DATE.month >= 10 else FORECAST_DATE.year
SNOTEL_START = '1981-10-01'
SNOTEL_END = '2025-06-30'
STREAMFLOW_START = '1981-10-01'
STREAMFLOW_END = '2025-09-30'
CFS_TO_ACFT_DAY = 1.98347
MONTHS = [4, 5, 6, 7, 8, 9]
MONTH_LABELS = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September'}

# Add standard water-year and calendar helper columns for time-series analysis.
def add_water_year_columns(df, date_col='Date'):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out['Water_Year'] = np.where(out[date_col].dt.month >= 10, out[date_col].dt.year + 1, out[date_col].dt.year)
    out['calendar_year'] = out[date_col].dt.year
    out['month'] = out[date_col].dt.month
    out['month_day'] = out[date_col].dt.strftime('%m-%d')
    return out

# Build a daily climatology envelope and isolate the target water year trace.
def build_dayofyear_climatology(df, value_col, target_wy):
    work = df.copy()
    work['Date'] = pd.to_datetime(work['Date'])
    work = work.loc[~((work['Date'].dt.month == 2) & (work['Date'].dt.day == 29))].copy()
    work['month_day'] = work['Date'].dt.strftime('%m-%d')
    pivot = work.pivot_table(index='month_day', columns='Water_Year', values=value_col, aggfunc='mean')
    ordered_index = [d for d in pd.date_range('2000-10-01', '2001-09-30', freq='D').strftime('%m-%d') if d != '02-29']
    pivot = pivot.reindex(ordered_index)
    hist = pivot.drop(columns=[target_wy], errors='ignore')
    clim = pd.DataFrame(index=pivot.index)
    clim['min'] = hist.min(axis=1)
    clim['Q10'] = hist.quantile(0.10, axis=1)
    clim['Q25'] = hist.quantile(0.25, axis=1)
    clim['mean'] = hist.mean(axis=1)
    clim['median'] = hist.median(axis=1)
    clim['Q75'] = hist.quantile(0.75, axis=1)
    clim['Q90'] = hist.quantile(0.90, axis=1)
    clim['max'] = hist.max(axis=1)
    if target_wy in pivot.columns:
        clim[f'{target_wy}_value'] = pivot[target_wy]
    return clim, pivot, hist

# Compute non-parametric percentile rank of a target value against historical samples.
def empirical_percentile(hist_values, target_value):
    vals = np.asarray(hist_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or pd.isna(target_value):
        return np.nan
    return 100.0 * np.mean(vals <= target_value)

# Fit a simple linear regression and return slope/intercept plus R^2 diagnostics.
def simple_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    x_valid = x[mask]
    y_valid = y[mask]
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    corr = np.corrcoef(x_valid, y_valid)[0, 1]
    return slope, intercept, corr ** 2, x_valid, y_valid

# Download daily SNOTEL SWE from the NRCS report endpoint and cache as CSV.
def fetch_snotel_swe_csv(site_name, site_id, state_abb, start_date, end_date, output_folder):
    site_core = str(site_id).split('_')[0]
    url = (
        'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/'
        'customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
        f'{site_core}:{state_abb}:SNTL%7Cid=%22%22%7Cname/'
        f'{start_date},{end_date}/'
        'WTEQ::value?fitToScreen=false'
    )

    print(f'Start retrieving data for {site_name}, {site_id} \n {url}')
    response = requests.get(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/csv,text/plain,*/*',
            'Connection': 'keep-alive'
        },
        timeout=(5, 30)
    )
    response.raise_for_status()

    lines = [line.strip() for line in response.text.splitlines() if line.strip() and not line.startswith('#')]
    if not lines:
        raise ValueError(f'No parseable data returned for {site_id}')

    csv_start = 0
    for idx, line in enumerate(lines):
        if line.lower().startswith('date,'):
            csv_start = idx
            break

    df = pd.read_csv(io.StringIO('\n'.join(lines[csv_start:])))
    if df.shape[1] < 2:
        preview = '\n'.join(lines[:5])
        raise ValueError(f'Unexpected SNOTEL CSV format for {site_id}. Preview:\n{preview}')

    df = df.iloc[:, :2].copy()
    df.columns = ['Date', 'Snow Water Equivalent (m) Start of Day Values']
    df['Date'] = pd.to_datetime(df['Date'])
    df['Snow Water Equivalent (m) Start of Day Values'] = (
        pd.to_numeric(df['Snow Water Equivalent (m) Start of Day Values'], errors='coerce') * 0.0254
    )
    df.dropna(subset=['Date', 'Snow Water Equivalent (m) Start of Day Values'], inplace=True)
    df['Water_Year'] = df['Date'].map(lambda x: x.year + 1 if x.month > 9 else x.year)

    output_path = os.path.join(output_folder, f'df_{site_id}_{state_abb}_SNTL.csv')
    df.to_csv(output_path, index=False)

tick_labels = ['10-01', '12-01', '02-01', '04-01', '06-01', '08-01']
tick_positions = [i for i, val in enumerate([d for d in pd.date_range('2000-10-01', '2001-09-30', freq='D').strftime('%m-%d') if d != '02-29']) if val in tick_labels]

print(f'Forecast date: {FORECAST_DATE.date()} | Forecast water year: WY{FORECAST_WY}')

# Normalize state values for NRCS station triplets (expects 2-letter abbreviations).
state_name_to_abbrev = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
    'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
    'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
    'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
    'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
    'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
}

if 'state' in gdf_in_basin.columns:
    gdf_in_basin = gdf_in_basin.copy()
    state_series = gdf_in_basin['state'].astype(str).str.upper().str.strip()
    gdf_in_basin['state'] = state_series.replace(state_name_to_abbrev)
    print('Normalized SNOTEL state codes:', sorted(gdf_in_basin['state'].dropna().unique().tolist()))

# ── 2.1 SNOTEL Site Description ---------------------------------
"""
## 2.1 SNOTEL Site Description

The table below describes the SNOTEL sites found inside the delineated watershed. Station metadata are pulled from the watershed clip generated in Part 1 and the downloaded time series are used to estimate the effective period of record for SWE analysis.
"""

if len(gdf_in_basin) == 0:
    raise ValueError('No SNOTEL stations were found inside the delineated basin.')

# Containers for per-station time series, climatologies, metadata, and April 1 metrics.
snotel_series = {}
station_climatologies = {}
snotel_site_rows = []
station_april1_rows = []

snotel_iter_df = gdf_in_basin.copy()
if 'elevation' in snotel_iter_df.columns:
    snotel_iter_df = snotel_iter_df.sort_values('elevation', ascending=False)

# Loop through each in-basin SNOTEL site: download/load SWE, compute climatology,
# and assemble station-level summary metrics used later in figures and discussion.
for _, row in snotel_iter_df.iterrows():
    site_code = str(row.get('code'))
    site_name = row.get('name', site_code)
    site_state = str(row.get('state', STATE_ABB)).upper()
    site_file = os.path.join(FILES_DIR, f'df_{site_code}_{site_state}_SNTL.csv')

    if not os.path.exists(site_file):
        fetch_snotel_swe_csv(site_name, site_code, site_state, SNOTEL_START, SNOTEL_END, FILES_DIR)

    site_df = pd.read_csv(site_file)
    swe_col = 'Snow Water Equivalent (m) Start of Day Values'
    site_df['Date'] = pd.to_datetime(site_df['Date'])
    site_df = site_df[['Date', swe_col, 'Water_Year']].copy()
    site_df.rename(columns={swe_col: 'swe_m'}, inplace=True)
    site_df['swe_in'] = site_df['swe_m'] * 39.3701
    site_df = add_water_year_columns(site_df, 'Date')
    snotel_series[site_code] = site_df

    clim, pivot, hist = build_dayofyear_climatology(site_df[['Date', 'Water_Year', 'swe_in']], 'swe_in', FORECAST_WY)
    station_climatologies[site_code] = {'clim': clim, 'pivot': pivot, 'hist': hist, 'name': site_name}

    begin_date = site_df['Date'].min()
    end_date = site_df['Date'].max()
    years_of_record = int(site_df['Water_Year'].nunique())

    hist_apr1 = site_df.loc[(site_df['month_day'] == '04-01') & (site_df['Water_Year'] < FORECAST_WY), 'swe_in']
    target_apr1 = site_df.loc[site_df['Date'] == FORECAST_DATE, 'swe_in']
    target_apr1 = float(target_apr1.iloc[0]) if not target_apr1.empty else np.nan
    median_apr1 = float(hist_apr1.median()) if len(hist_apr1) else np.nan
    mean_apr1 = float(hist_apr1.mean()) if len(hist_apr1) else np.nan

    snotel_site_rows.append({
        'site_code': site_code,
        'site_name': site_name,
        'state': site_state,
        'latitude': round(float(row.geometry.y), 4),
        'longitude': round(float(row.geometry.x), 4),
        'elevation_m': row.get('elevation_m', row.get('elevation', np.nan)),
        'begin_date': begin_date.date() if pd.notna(begin_date) else pd.NaT,
        'end_date': end_date.date() if pd.notna(end_date) else pd.NaT,
        'years_of_record': years_of_record
    })

    station_april1_rows.append({
        'site_code': site_code,
        'site_name': site_name,
        'april1_2025_swe_in': target_apr1,
        'historical_median_apr1_in': median_apr1,
        'historical_mean_apr1_in': mean_apr1,
        'pct_of_median_apr1': 100.0 * target_apr1 / median_apr1 if pd.notna(target_apr1) and pd.notna(median_apr1) and median_apr1 != 0 else np.nan,
        'pct_of_mean_apr1': 100.0 * target_apr1 / mean_apr1 if pd.notna(target_apr1) and pd.notna(mean_apr1) and mean_apr1 != 0 else np.nan,
        'april1_percentile': empirical_percentile(hist_apr1, target_apr1)
    })

# Convert accumulated dictionaries/records into final analysis tables and basin index series.
snotel_sites_df = pd.DataFrame(snotel_site_rows).sort_values(['elevation_m', 'site_code'], ascending=[False, True], na_position='last')
station_april1_df = pd.DataFrame(station_april1_rows).sort_values('pct_of_median_apr1', ascending=False)

basin_swe = pd.concat(
    [df.set_index('Date')['swe_in'].rename(code) for code, df in snotel_series.items()],
    axis=1
).sort_index()
basin_swe['basin_mean_swe_in'] = basin_swe.mean(axis=1, skipna=True)
basin_swe_daily = basin_swe[['basin_mean_swe_in']].dropna().reset_index().rename(columns={'index': 'Date'})
basin_swe_daily = add_water_year_columns(basin_swe_daily, 'Date')
basin_clim, basin_pivot, basin_hist = build_dayofyear_climatology(
    basin_swe_daily[['Date', 'Water_Year', 'basin_mean_swe_in']],
    'basin_mean_swe_in',
    FORECAST_WY
)

basin_apr1_hist = basin_swe_daily.loc[
    (basin_swe_daily['month_day'] == '04-01') & (basin_swe_daily['Water_Year'] < FORECAST_WY),
    'basin_mean_swe_in'
].dropna()
basin_apr1_2025 = basin_swe_daily.loc[basin_swe_daily['Date'] == FORECAST_DATE, 'basin_mean_swe_in']
basin_apr1_2025 = float(basin_apr1_2025.iloc[0]) if not basin_apr1_2025.empty else np.nan
basin_apr1_summary = pd.Series({
    'april1_2025_swe_in': basin_apr1_2025,
    'historical_median_apr1_in': basin_apr1_hist.median(),
    'historical_mean_apr1_in': basin_apr1_hist.mean(),
    'pct_of_median_apr1': 100.0 * basin_apr1_2025 / basin_apr1_hist.median() if len(basin_apr1_hist) and basin_apr1_hist.median() != 0 else np.nan,
    'pct_of_mean_apr1': 100.0 * basin_apr1_2025 / basin_apr1_hist.mean() if len(basin_apr1_hist) and basin_apr1_hist.mean() != 0 else np.nan,
    'april1_percentile': empirical_percentile(basin_apr1_hist, basin_apr1_2025)
})

peak_swe_by_wy = basin_swe_daily.groupby('Water_Year')['basin_mean_swe_in'].max().rename('peak_basin_swe_in')
april1_swe_by_wy = basin_swe_daily.loc[basin_swe_daily['month_day'] == '04-01'].set_index('Water_Year')['basin_mean_swe_in'].rename('april1_basin_swe_in')

display(snotel_sites_df)
display(station_april1_df.round(2))
display(pd.DataFrame(basin_apr1_summary).T.round(2))

# ── 2.2 SWE Figures ---------------------------------------------
"""
## 2.2 SWE Figures

The first figure shows the historical daily SWE envelope for each SNOTEL station in the basin, with the WY2025 trace overlaid. The second figure shows the basin-mean SWE index, computed as the mean daily SWE across all available SNOTEL stations.
"""

num_sites = len(station_climatologies)
ncols = 2
nrows = int(np.ceil(num_sites / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()

# Plot one panel per station showing historical SWE envelope plus WY target trace.
for ax, (site_code, info) in zip(axes, station_climatologies.items()):
    clim = info['clim']
    x = np.arange(len(clim.index))

    ax.fill_between(x, clim['Q10'], clim['Q90'], color='lightskyblue', alpha=0.35, label='10th–90th percentile')
    ax.fill_between(x, clim['Q25'], clim['Q75'], color='steelblue', alpha=0.35, label='25th–75th percentile')
    ax.plot(x, clim['median'], color='darkgreen', linewidth=2, label='Median')
    ax.plot(x, clim['mean'], color='darkorange', linewidth=1.5, linestyle='--', label='Mean')
    if f'{FORECAST_WY}_value' in clim.columns:
        ax.plot(x, clim[f'{FORECAST_WY}_value'], color='black', linewidth=1.8, label=f'WY{FORECAST_WY}')
    if '04-01' in clim.index:
        ax.axvline(clim.index.get_loc('04-01'), color='black', linestyle=':', linewidth=1)

    ax.set_title(f"{info['name']} ({site_code})", fontsize=10, fontweight='bold')
    ax.set_ylabel('SWE (inches)')
    ax.grid(True, alpha=0.25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(['Oct', 'Dec', 'Feb', 'Apr', 'Jun', 'Aug'])

for ax in axes[num_sites:]:
    ax.axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f'SNOTEL SWE Climatology for the {RESERVOIR_NAME} Watershed', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
swe_sites_fig = os.path.join(FIGURES_DIR, '03_snotel_station_climatology.png')
plt.savefig(swe_sites_fig, dpi=150, bbox_inches='tight')
plt.show()

# Plot basin-mean SWE climatology and target year to summarize watershed-scale snowpack.
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(basin_clim.index))
ax.fill_between(x, basin_clim['Q10'], basin_clim['Q90'], color='lightskyblue', alpha=0.35, label='10th–90th percentile')
ax.fill_between(x, basin_clim['Q25'], basin_clim['Q75'], color='steelblue', alpha=0.35, label='25th–75th percentile')
ax.plot(x, basin_clim['median'], color='darkgreen', linewidth=2, label='Median basin SWE')
ax.plot(x, basin_clim['mean'], color='darkorange', linestyle='--', linewidth=1.5, label='Mean basin SWE')
if f'{FORECAST_WY}_value' in basin_clim.columns:
    ax.plot(x, basin_clim[f'{FORECAST_WY}_value'], color='black', linewidth=2, label=f'WY{FORECAST_WY} basin SWE')
if '04-01' in basin_clim.index:
    ax.axvline(basin_clim.index.get_loc('04-01'), color='black', linestyle=':', linewidth=1.2)

ax.set_xticks(tick_positions)
ax.set_xticklabels(['Oct', 'Dec', 'Feb', 'Apr', 'Jun', 'Aug'])
ax.set_ylabel('Basin mean SWE (inches)')
ax.set_title(f'Basin-Mean SWE Index for the {RESERVOIR_NAME} Watershed', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, ncol=3, loc='upper left')
plt.tight_layout()
basin_swe_fig = os.path.join(FIGURES_DIR, '04_basin_mean_swe_index.png')
plt.savefig(basin_swe_fig, dpi=150, bbox_inches='tight')
plt.show()

print(f'Saved: {swe_sites_fig}')
print(f'Saved: {basin_swe_fig}')

# ── **Figure captions** -----------------------------------------
"""
**Figure captions**

- **Figure 3.** Historical SWE envelope at each SNOTEL station within the McPhee Reservoir watershed. Shaded bands show the interannual 10th–90th and 25th–75th percentile ranges, while the black line shows WY2025 observations.
- **Figure 4.** Basin-mean SWE index derived by averaging all available SNOTEL stations within the watershed. This index provides a single basin-scale indicator for April 1, 2025 snow conditions relative to the historical range.
"""

basin_pct_median = basin_apr1_summary['pct_of_median_apr1']
basin_pct_mean = basin_apr1_summary['pct_of_mean_apr1']
basin_percentile = basin_apr1_summary['april1_percentile']

# Classify basin snow condition on April 1 relative to historical normal.
if pd.isna(basin_pct_median):
    swe_condition = 'unavailable because the April 1, 2025 basin SWE index could not be computed from the downloaded data.'
elif basin_pct_median >= 110:
    swe_condition = 'above normal, indicating a larger-than-median snowmelt contribution to spring inflow.'
elif basin_pct_median <= 90:
    swe_condition = 'below normal, indicating a smaller-than-median snowmelt contribution to spring inflow.'
else:
    swe_condition = 'near normal relative to the historical April 1 median.'

print('SWE discussion for April 1, 2025')
print('-' * 40)
print(f"Basin April 1, 2025 SWE index : {basin_apr1_summary['april1_2025_swe_in']:.2f} inches")
print(f"Historical April 1 median     : {basin_apr1_summary['historical_median_apr1_in']:.2f} inches")
print(f"Historical April 1 mean       : {basin_apr1_summary['historical_mean_apr1_in']:.2f} inches")
print(f"Percent of median             : {basin_pct_median:.1f}%")
print(f"Percent of mean               : {basin_pct_mean:.1f}%")
print(f"Historical percentile         : {basin_percentile:.1f}")
print()
print('Interpretation:')
print(f'The basin-scale April 1, 2025 SWE condition is {swe_condition}')

# ── --- ---------------------------------------------------------
"""
---

# Part 3: USGS Streamflow Analysis

This section retrieves daily mean discharge from the USGS inlet gauge and summarizes both the station metadata and the historical range of monthly inflow volume for April through September.
"""

usgs_info = nwis.get_info(sites=USGS_GAGE_ID)
if isinstance(usgs_info, tuple):
    usgs_info = usgs_info[0]
usgs_info_row = usgs_info.iloc[0]

# Load cached daily streamflow if present; otherwise pull and clean from NWIS.
streamflow_file = os.path.join(FILES_DIR, f'usgs_{USGS_GAGE_ID}_daily_flow.csv')
if os.path.exists(streamflow_file):
    flow_df = pd.read_csv(streamflow_file, parse_dates=['Date'])
else:
    flow_raw, _ = nwis.get_dv(
        sites=USGS_GAGE_ID,
        start=STREAMFLOW_START,
        end=STREAMFLOW_END,
        parameterCd='00060',
        statCd='00003'
    )
    if isinstance(flow_raw.index, pd.MultiIndex):
        flow_raw = flow_raw.reset_index()
    else:
        flow_raw = flow_raw.reset_index()

    date_col = 'datetime' if 'datetime' in flow_raw.columns else flow_raw.columns[0]
    flow_col_candidates = [c for c in flow_raw.columns if c.startswith('00060') and ('00003' in c or 'Mean' in c or 'mean' in c)]
    if not flow_col_candidates:
        raise ValueError('Unable to identify the USGS daily streamflow column.')
    flow_col = flow_col_candidates[0]

    flow_df = flow_raw[[date_col, flow_col]].copy()
    flow_df.columns = ['Date', 'flow_cfs']
    flow_df['Date'] = pd.to_datetime(flow_df['Date'])
    if pd.api.types.is_datetime64tz_dtype(flow_df['Date']):
        flow_df['Date'] = flow_df['Date'].dt.tz_convert(None)
    flow_df.to_csv(streamflow_file, index=False)

flow_df['Date'] = pd.to_datetime(flow_df['Date'])
if pd.api.types.is_datetime64tz_dtype(flow_df['Date']):
    flow_df['Date'] = flow_df['Date'].dt.tz_convert(None)
flow_df['flow_cfs'] = pd.to_numeric(flow_df['flow_cfs'], errors='coerce')
flow_df = flow_df.dropna(subset=['flow_cfs']).copy()
flow_df = add_water_year_columns(flow_df, 'Date')
flow_df['daily_volume_acft'] = flow_df['flow_cfs'] * CFS_TO_ACFT_DAY

# Aggregate April-September daily flow to monthly volumes by water year.
monthly_volume = (
    flow_df.loc[flow_df['month'].isin(MONTHS)]
    .groupby(['Water_Year', 'month'])['daily_volume_acft']
    .sum()
    .unstack('month')
    .sort_index()
)

usgs_site_description = pd.DataFrame([{
    'site_no': USGS_GAGE_ID,
    'station_name': usgs_info_row.get('station_nm', 'Unavailable'),
    'latitude': pd.to_numeric(usgs_info_row.get('dec_lat_va'), errors='coerce'),
    'longitude': pd.to_numeric(usgs_info_row.get('dec_long_va'), errors='coerce'),
    'altitude_ft': pd.to_numeric(usgs_info_row.get('alt_va'), errors='coerce'),
    'drainage_area_mi2': pd.to_numeric(usgs_info_row.get('drain_area_va'), errors='coerce'),
    'begin_record': flow_df['Date'].min().date(),
    'end_record': flow_df['Date'].max().date(),
    'years_of_record': int(flow_df['Water_Year'].nunique()),
    'observation_parameter': 'USGS parameter 00060, daily mean discharge (cfs)'
}])

display(usgs_site_description.round(2))
display(monthly_volume.describe().T[['count', 'mean', '50%', 'min', 'max']].rename(columns={'50%': 'median'}).round(1))

# Visualize historical monthly inflow distributions (Apr-Sep) with boxplots.
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
axes = axes.ravel()

historical_monthly_volume = monthly_volume.loc[monthly_volume.index < FORECAST_WY].copy()

for ax, month in zip(axes, MONTHS):
    month_vals = historical_monthly_volume[month].dropna()
    ax.boxplot(month_vals, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
               medianprops=dict(color='darkgreen', linewidth=2),
               whiskerprops=dict(color='steelblue'),
               capprops=dict(color='steelblue'))
    jitter = np.random.default_rng(42 + month).normal(1, 0.035, size=len(month_vals))
    ax.scatter(jitter, month_vals, s=18, color='gray', alpha=0.4)
    ax.axhline(month_vals.mean(), color='darkorange', linestyle='--', linewidth=1.5, label='Mean')
    ax.set_title(MONTH_LABELS[month], fontweight='bold')
    ax.set_ylabel('Monthly streamflow volume (acre-ft)')
    ax.set_xticks([])
    ax.grid(True, alpha=0.25)

axes[0].legend(frameon=False, loc='upper right')
plt.suptitle(f'Historical April–September Streamflow Volume at USGS {USGS_GAGE_ID}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
streamflow_hist_fig = os.path.join(FIGURES_DIR, '05_streamflow_monthly_volume_boxplots.png')
plt.savefig(streamflow_hist_fig, dpi=150, bbox_inches='tight')
plt.show()

print(f'Saved: {streamflow_hist_fig}')

# Summarize April 1, 2025 flow against historical April 1 conditions.
april1_flow_hist = flow_df.loc[
    (flow_df['month_day'] == '04-01') & (flow_df['Date'].dt.year < FORECAST_DATE.year),
    'flow_cfs'
].dropna()
april1_flow_2025 = flow_df.loc[flow_df['Date'].dt.normalize() == FORECAST_DATE.normalize(), 'flow_cfs']
april1_flow_2025 = float(april1_flow_2025.iloc[0]) if not april1_flow_2025.empty else np.nan

april1_flow_summary = pd.Series({
    'april1_2025_flow_cfs': april1_flow_2025,
    'historical_mean_apr1_cfs': april1_flow_hist.mean(),
    'historical_median_apr1_cfs': april1_flow_hist.median(),
    'pct_of_mean_apr1': 100.0 * april1_flow_2025 / april1_flow_hist.mean() if len(april1_flow_hist) and april1_flow_hist.mean() != 0 else np.nan,
    'pct_of_median_apr1': 100.0 * april1_flow_2025 / april1_flow_hist.median() if len(april1_flow_hist) and april1_flow_hist.median() != 0 else np.nan,
    'april1_percentile': empirical_percentile(april1_flow_hist, april1_flow_2025)
})

display(pd.DataFrame(april1_flow_summary).T.round(2))

# Translate April 1 flow metrics into a qualitative runoff status statement.
if pd.isna(april1_flow_summary['pct_of_median_apr1']):
    flow_condition = 'unavailable because the April 1, 2025 daily flow could not be retrieved.'
elif april1_flow_summary['pct_of_median_apr1'] >= 110:
    flow_condition = 'above the historical April 1 median, suggesting that melt season inflow had already begun at an elevated rate.'
elif april1_flow_summary['pct_of_median_apr1'] <= 90:
    flow_condition = 'below the historical April 1 median, suggesting a slower-than-normal start to the spring runoff season.'
else:
    flow_condition = 'near the historical April 1 median, suggesting a seasonally typical early runoff condition.'

print('Streamflow discussion for April 1, 2025')
print('-' * 45)
print(f"Daily mean discharge on April 1, 2025 : {april1_flow_summary['april1_2025_flow_cfs']:.1f} cfs")
print(f"Historical April 1 mean              : {april1_flow_summary['historical_mean_apr1_cfs']:.1f} cfs")
print(f"Historical April 1 median            : {april1_flow_summary['historical_median_apr1_cfs']:.1f} cfs")
print(f"Percent of mean                      : {april1_flow_summary['pct_of_mean_apr1']:.1f}%")
print(f"Percent of median                    : {april1_flow_summary['pct_of_median_apr1']:.1f}%")
print(f"Historical percentile                : {april1_flow_summary['april1_percentile']:.1f}")
print()
print('Interpretation:')
print(f'The USGS streamflow condition on April 1, 2025 is {flow_condition}')

# ── --- ---------------------------------------------------------
"""
---

# Part 4: Compare and Contrast Peak SWE with Streamflow

The plots below relate basin-wide historical peak SWE to monthly streamflow volume for April through September. These parity plots show which months are most sensitive to snowpack conditions and therefore most useful for forecasting reservoir inflow.
"""

comparison_df = pd.concat([peak_swe_by_wy, monthly_volume[MONTHS]], axis=1).dropna().sort_index()
historical_comparison_df = comparison_df.loc[comparison_df.index < FORECAST_WY].copy()

# Create monthly parity plots linking peak SWE to downstream monthly flow volumes.
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
axes = axes.ravel()
monthly_relationship_rows = []

for ax, month in zip(axes, MONTHS):
    x = historical_comparison_df['peak_basin_swe_in'].values
    y = historical_comparison_df[month].values
    slope, intercept, r2, x_valid, y_valid = simple_regression(x, y)

    ax.scatter(x_valid, y_valid, color='steelblue', alpha=0.7, s=35)
    if np.isfinite(slope):
        x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='darkorange', linewidth=2)

    corr = np.corrcoef(x_valid, y_valid)[0, 1] if len(x_valid) > 1 else np.nan
    monthly_relationship_rows.append({
        'month': MONTH_LABELS[month],
        'n_years': len(x_valid),
        'correlation_r': corr,
        'r_squared': r2,
        'slope_acft_per_in': slope
    })

    ax.set_title(MONTH_LABELS[month], fontweight='bold')
    ax.set_xlabel('Peak basin SWE (inches)')
    ax.set_ylabel('Monthly streamflow volume (acre-ft)')
    ax.grid(True, alpha=0.25)
    ax.text(
        0.05, 0.95,
        f'n = {len(x_valid)}\n$R^2$ = {r2:.2f}' if np.isfinite(r2) else f'n = {len(x_valid)}\n$R^2$ = N/A',
        transform=ax.transAxes,
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='lightgray')
    )

plt.suptitle(f'Peak SWE vs. Monthly Streamflow Volume at USGS {USGS_GAGE_ID}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
parity_fig = os.path.join(FIGURES_DIR, '06_peak_swe_vs_monthly_volume.png')
plt.savefig(parity_fig, dpi=150, bbox_inches='tight')
plt.show()

# Rank months by explanatory power (R^2) to identify strongest SWE-flow relationship.
monthly_relationship_df = pd.DataFrame(monthly_relationship_rows).sort_values('r_squared', ascending=False)
display(monthly_relationship_df.round(3))
print(f'Saved: {parity_fig}')

best_month = monthly_relationship_df.iloc[0]['month']
best_r2 = monthly_relationship_df.iloc[0]['r_squared']

print('Peak SWE vs. streamflow interpretation')
print('-' * 45)
for _, row in monthly_relationship_df.iterrows():
    print(f"{row['month']:<9} : r = {row['correlation_r']:.2f}, R^2 = {row['r_squared']:.2f}, slope = {row['slope_acft_per_in']:.0f} acre-ft per inch SWE")
print()
print(f'The strongest monthly relationship occurs in {best_month}, where peak basin SWE explains about {best_r2 * 100:.0f}% of the historical variance in monthly inflow volume.')
print('In general, stronger spring and early summer relationships indicate a snowmelt-dominated basin, while weaker late-summer relationships reflect increasing influence from antecedent soil moisture, summer storms, and operational release decisions.')

# ── --- ---------------------------------------------------------
"""
---

# Part 5: Recommendations for Reservoir Operators

This final section converts the April 1, 2025 snowpack information into an inflow expectation and an operational recommendation for McPhee Reservoir.
"""

seasonal_volume = monthly_volume[MONTHS].sum(axis=1).rename('apr_sep_volume_acft')
forecast_relation_df = pd.concat([april1_swe_by_wy, seasonal_volume], axis=1).dropna().sort_index()
historical_forecast_df = forecast_relation_df.loc[forecast_relation_df.index < FORECAST_WY].copy()

# Fit a forecasting relationship: April 1 basin SWE -> Apr-Sep inflow volume.
forecast_slope, forecast_intercept, forecast_r2, x_valid, y_valid = simple_regression(
    historical_forecast_df['april1_basin_swe_in'].values,
    historical_forecast_df['apr_sep_volume_acft'].values
)

forecast_april1_swe = april1_swe_by_wy.get(FORECAST_WY, np.nan)
predicted_seasonal_volume = forecast_slope * forecast_april1_swe + forecast_intercept if np.isfinite(forecast_slope) and pd.notna(forecast_april1_swe) else np.nan
historical_mean_seasonal_volume = historical_forecast_df['apr_sep_volume_acft'].mean()
historical_median_seasonal_volume = historical_forecast_df['apr_sep_volume_acft'].median()

prediction_pct_mean = 100.0 * predicted_seasonal_volume / historical_mean_seasonal_volume if historical_mean_seasonal_volume else np.nan
prediction_pct_median = 100.0 * predicted_seasonal_volume / historical_median_seasonal_volume if historical_median_seasonal_volume else np.nan

# Convert quantitative forecast signal into an operator-facing management message.
if pd.isna(basin_apr1_summary['pct_of_median_apr1']):
    inflow_outlook = 'indeterminate because the April 1, 2025 SWE index is unavailable.'
    operator_message = 'Re-run the data retrieval cells and confirm the SNOTEL downloads before making an operational decision.'
elif basin_apr1_summary['pct_of_median_apr1'] >= 110:
    inflow_outlook = 'more streamflow than the historical median and likely more than the historical mean.'
    operator_message = 'Maintain additional flood-control space into April, avoid filling too aggressively, and plan for elevated May–June inflows.'
elif basin_apr1_summary['pct_of_median_apr1'] <= 90:
    inflow_outlook = 'less streamflow than the historical mean and median.'
    operator_message = 'Prioritize water conservation and refill efficiency, because the snowmelt signal suggests below-normal spring inflow.'
else:
    inflow_outlook = 'near the historical mean and median.'
    operator_message = 'Operate near the normal rule curve, while preserving flexibility for short-term weather-driven changes in melt timing.'

recommendation_summary = pd.DataFrame([{
    'forecast_date': FORECAST_DATE.date(),
    'april1_basin_swe_in': forecast_april1_swe,
    'april1_pct_of_median': basin_apr1_summary['pct_of_median_apr1'],
    'predicted_apr_sep_volume_acft': predicted_seasonal_volume,
    'historical_mean_apr_sep_volume_acft': historical_mean_seasonal_volume,
    'historical_median_apr_sep_volume_acft': historical_median_seasonal_volume,
    'predicted_pct_of_mean_volume': prediction_pct_mean,
    'predicted_pct_of_median_volume': prediction_pct_median,
    'seasonal_regression_r2': forecast_r2
}])

display(recommendation_summary.round(1))

print('Reservoir management recommendation')
print('-' * 45)
print(f'Based on April 1, 2025 SWE, the reservoir should expect {inflow_outlook}')
print(f'Operational recommendation: {operator_message}')
print()
print('Supporting evidence:')
print(f"- Basin April 1 SWE was {basin_apr1_summary['pct_of_median_apr1']:.1f}% of the historical median.")
print(f"- Estimated April–September inflow volume is {prediction_pct_median:.1f}% of the historical median based on the April 1 SWE regression.")
print(f"- The April 1 SWE to seasonal volume regression has R^2 = {forecast_r2:.2f}.")
