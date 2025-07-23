---
contributors:
  - Damilola Aladesuru
  - Josef Baumert

---

## Overview

This repository contains the details for replicating the paper "Spatio-temporal Patterns of Field Structural Change and Related Farm Developments in Lower Saxony" by Damilola Aladesuru, Josef Baumert, Christine Wieck, Till Kuhn, Hugo Storm.

### Memory and Runtime Requirements
All processes in this repository have been run using a CPU. 

### Description of programs/code

Organization of the code largely follows the Cookiecutter Data Science template (https://drivendata.github.io/cookiecutter-data-science/)

The script-containing parts are:
- Programs in `src/data` are used to prepare and process data sets. 
- Programs in `src/analysis` are used to analysed processed data.
- Programs in `src/visualization` are used to visualize and to prepare the results. 

### License for Code

The code is licensed under a MIT license. See [LICENSE](LICENSE) for details.

## Replication Workflow

### Project Directory Setup

All scripts expect to run from the project root or a directory where relative paths resolve correctly. In my case this looks something like `~/Documents/DataAnalysis/Lab/Niedersachsen/final`.  
Set your working directory accordingly.


### Requirements

- **Python:** 3.x
- **Packages:** Full list of packages to install and their versions are provided in the `reuirements.txt` file. Packages used within scripts are listed per step below.

### Data
| Data.Name  | Needed.Data.Files | Storage.Location | Publicly.Accessible | Link|
| -- | -- | -- | -- |-- |
| "State boundaries" | `NDS_Landesflaeche.shp`; `NDS_Landkreise.shp` | `.../verwaltungseinheiten/` | TRUE | [OpenGeoData](https://ni-lgln-opengeodata.hub.arcgis.com/documents/e9631a1bccac4823912963aa8f0bc0b0/about)|
| "10km² tessellated grid" |  `eea....shp` |  `data/raw/` | TRUE | [EEA Datahub](https://www.eea.europa.eu/en/datahub/datahubitem-view/3c362237-daa4-45e2-8c16-aaadfb1a003b) |
| "Climate" |  `Klimaregionen.shp` |  `data/raw/` | TRUE | [Landesamt für Bergbau, Energie und Geologie (LBEG)](https://www.lbeg.niedersachsen.de/kartenserver/nibis/niedersaechsisches-bodeninformationssystem-nibis-841.html) |
| "Elevation" |  `lsax_elevation.tif` |  `data/raw/` | TRUE | [Google Earth Engine](https://code.earthengine.google.com/5dae420b96373698a054feda3e64d6d5) |
| "Farm" |  `farm.zip` |  `data/raw/` | TRUE | [Regional database](https://www1.nls.niedersachsen.de/statistik/html/default.asp) |
| "Animal" |  `animal.zip` |  `data/raw/` | TRUE | [Regional database](https://www1.nls.niedersachsen.de/statistik/html/default.asp) |
---

- Download Lower Saxony boundary files from OpenGeoData. From the files contained in the downloaded zipped file, you need `NDS_Landesflaeche.shp` for the state’s boundary and `NDS_Landkreise.shp` for the NUTS3 regions of Lower Saxony.
- Download the EEA 10km² tessellated reference grid from the EEA Datahub.
- The farm and animal datasets can be found in the fourth folder of the regional databases:

**Farm data access guide**:
1. Click on database link
2. Click `WEITER` -> `4 - Wirtschaftsbereiche, Verkehr` -> `41 - Land- und Forstwirtschaft, Fischerei` -> `41121 Agrarstrukturerhebung` -> `Landwirtschaftliche Betriebe mit LF nach Größenklasse der LF (2010,2016,2020)`
3. Click `Kreisfr. Stadt, Landkreis` -> `alle auswählen` -> OK
4. Unselect 0, 1, 2, 3, 4
5. Click `Table erstellen` -> `gezippte Excel-Datei herunterladen`
6. Rename downloaded zipped file and move to `data/raw`

**Animal data access guide**:
1. Click on database link
2. Click `WEITER` -> `4 - Wirtschaftsbereiche, Verkehr` -> `41 - Land- und Forstwirtschaft, Fischerei` -> `41121 Agrarstrukturerhebung` -> `Landwirtschaftliche Betriebe mit Viehhaltung und Viehbestand (2010,2016,2020)`
3. Click `Kreisfr. Stadt, Landkreis` -> `alle auswählen` -> OK
4. Proceed similarly to farm data guide above

---

---

### Step 1 — Join NUTS3 Regions with 10km Grid

**Script:** `src/data/gridregionjoin.py`

**Workflow:**
1. Updata base_dir and grid path
2. Load Landkreise and grid data for spatial joining.
3. Use `Landesflaeche` to verify spatial cextent of Lower Saxony.
4. Ensure EEA grids are filtered only to within Lower Saxony’s boundary.
5. When joining, remove duplicate grid cells by keeping the grid cell with the largest overlap per region i.e., Landkreis.
6. Save output to `data/interim/grid_landkreise.pkl`.

**Requirements:**
- **Packages:** `geopandas`, `pandas`, `os`, `pickle`
- **Input files:**
  - `NDS_Landkreise.shp`
  - `NDS_Landesflaeche.shp`
  - EEA grid shapefile
- **Required paths:**  
- Path for admin boundaries e.g., `data/Niedersachsen/`  
- Path for grid shapefile boundaries e.g., `data/raw/eea_10_km_eea-ref-grid-de_p_2013_v02_r00/`  
Update as needed.
- **Output directory** `data/interim` (must exist or be creatable)

---

### Step 2 – Load and Process Yearly Field Data

**Scripts:** `src/data/dataload.py`, `src/data/firstoverview.py`

**Workflow:**
1. Load yearly field shapefiles (2012–2023).
2. Harmonize columns:
   - Standardize `year`, `Kulturcode`, and area columns.
   - Drop `shape_leng`, `schlagnr`, `schlagbez`.
3. Spatial join field data sets with `NDS_Landesflaeche.shp` to keep only fields inside Lower Saxony.
4. Check for duplicates.
5. Combine all years into a single GeoDataFrame.
6. Handle missing values (drop missing values if <1% of rows contains missing values; otherwise, fill with mean).
7. Add area (m², ha), perimeter (m), and shape metrics.
8. Spatial join with `grid_landkreise.pkl` to add region and grid info.
9. Resolve duplicates by retaining the largest overlap.
10. Save output to `data/interim/gld_base.pkl`.

**Requirements:**
- **Packages:** `os`, `pickle`, `zipfile`, `geopandas`, `pandas`, `datetime`, `math`, `logging`
- **Dependencies:** `src.data.gridregionjoin`
- **Input files:**
  - Zipped yearly field shapefiles (2012–2023)
  - `NDS_Landesflaeche.shp`
  - `NDS_Landkreise.shp`
  - EEA grid shapefile
- **Output:** Pickle file in `data/interim`
- **Notes:** Full processing takes about 13 hours. Logging shows progress and warnings.

#### Optional – First Overview (`firstoverview.py`)

Run `firstoverview.py` for a quick data preview before proceeding to the full data loading workflow, which takes 13 hours. The functions of the script are similar to that of `dataload.py`; just broken down in chunks for easy following. Package and input requirements are also similar to that of `dataload.py`

**Note:** Even within the first overview script, spatial join for filtering fields to within Niedersachsen boundaries takes ~3.5 hours.

- **Packages:** `os`, `zipfile`, `geopandas`, `pandas`, `logging`, `matplotlib`  
- **Input:** Yearly shapefiles and `NDS_Landesflaeche.shp`

**Required paths:**  
- Path for field shapefiles e.g., `data/Niedersachsen/Needed/`  
- Path for admin boundaries e.g., `data/Niedersachsen/verwaltungseinheiten/`  
Update as needed.

---

### Step 3 — Process Crop Code and Code Description, and Explore the Data

---

#### Step 3A — Process Crop Code and Description

**Script:** `src/data/eca_new.py`

**Workflow:**
1. Load dataset using `dl.load_data`.

**Requirements:**  
- **Packages:** `pandas`, `seaborn`, `matplotlib`  
- **Dependencies:** `src.data.dataload`

---

#### Step 3B — Explore the Data

**Script:** `src/analysis/feel_the_data.py`

**Workflow:**
1. Load dataset using `dl.load_data`.
2. Create a working copy with `kulturcode`, `CELLCODE`, `LANDKREIS` as categoricals.
3. Plot numeric column distributions (2012 & 2023) and save to `reports/`.
4. Remove fields with area < 100 m².
5. Visualize field size distribution before/after outlier removal.
6. Print min/max and unique counts for numeric and categorical columns.
7. Export yearly stats (min, max, mean, median, std) to Excel.
8. Compute and plots correlation matrix (heatmap).
9. Aggregate yearly stats and plot total area trends.

**Notes:**  
- Plots & summaries are saved in `reports/`
- Script uses modular functions for easy extension.

**Requirements:**  
- **Packages:** `pandas`, `seaborn`, `matplotlib`  
- **Dependencies:** `src.data.dataload`

---

#### Step 3C — Examine Field Shape Values

**Script:** `src/analysis/examining_shape.py`

**Purpose:** Analyze and visualize fields with shape index ≈ 1.00 (near circular).

**Workflow:**
1. Load processed field-level and grid-level data with `src.analysis.desc.gridgdf_desc`.
2. Filter rows where shape ≈ 1.00.
3. Check for duplicate IDs.
4. Compute custom perimeter-area ratio (e.g., `cpar_25` i.e., cpar using 0.25 as constant).
5. Normalize & plot geometries for a selected year (e.g., 2021). Save to `reports/figures/examiningshape.svg`.
6. Print summary stats for shape index in both the field-level and grid-level datasets.
This gives you idea of minimum field shape or minimum average field shape in a grid cell when grid data is used.

**Note:**
Although this step is 3C, it works better if run after step 4 because it depends on the `gridgdf_desc` script created in step 4 to load field-level and grid-level data. It is possibe to only load field data with `dl.load_data` if interest lies only in examining shape at field level.

**Requirements:**
- **Packages:** `os`, `numpy`, `pandas`, `matplotlib`, `shapely`
- **Dependencies:** `src.analysis.desc.gridgdf_desc`
- **Input:** Processed field/grid data
- **Output:** SVG plot in `reports/figures/`

---
### Step 4 — Compute Grid Aggregates and Temporal Change in Field Structure Metrics

---

#### 4A) Field-Level Descriptives

**Script:** `src/analysis/desc/gld_desc_raw`

**Purpose:**  
Adjusts field data `gld` by adding `kulturcode` description i.e., `kulturart` and removing fields < 100 m².
Optional: Compute change metrics without grid aggregation.  

**Use Note:**
- Main function is `adjust_gld()`
- The main function is called in `src.analysis.desc.gridgdf_desc`
- Input: processed field-level data and Kulturcode master spreadsheet
- Output: cleaned `gld` GeoDataFrame

- **Packages:** `pandas`, `os`
- **Dependencies:** `src.data.dataload`, `src.data.eca_new` 

---

#### 4B) Grid-Aggregated Descriptives

**Script:** `src/analysis/desc/gridgdf_desc`  
**Main Functions:**  
- `create_gridgdf()`
- `desc_grid()`

**Workflow:**  
- Aggregate fields to 10 km² grid cells.
- Compute yearly averages and change values.
- Merge grid statistics with spatial geometry from prejoined grid-region.
- Remove outliers.
- Save grid-level statistics (mean, median, std, etc.) for all years and by year.

**Use Note:**
1. Import into any script
2. Call `create_gridgdf()` → returns:
   - Cleaned `gld`
   - Grid-level GeoDataFrame `gridgdf` with change metrics (e.g., those used in Figures 4 & 5) 
3. Call `desc_grid()` → returns:
   - Overall descriptive stats, `grid_allyears_stats`  
   - Yearly average stats for plot (Figure 3)


**Requirements:**  
- **Packages:** `pandas`, `geopandas`, `numpy`, `os`, `pickle`, `logging`, `contextlib`, `io`
- **Data:**  
  - `grid_landkreise.pkl` (grid + region join)
  - Cleaned field-level data (`src.analysis.desc.gld_desc_raw.adjust_gld()`)
- **Output paths**: `gridgdf/` and `statistics/` dirs (created if not present)

---

### Step 5 — Plot the Temporal Change Figures Shown in the Paper

---

#### Line & Map Plots

**Scripts:**  
- `src/visualization/lineplots.py` → Figures 4 & 5 (line plots)  
- `src/visualization/mapplots.py` → Figure 6 (maps)

**Functions:**  
- Visualizes temporal trends (line plots)
- Maps grid metric values (choropleth)

**Use Note:**
- `lineplots.py` needs: `src/visualization/plotting_module.py`
- `mapplots.py` requires reprojecting to EPSG:4326 before plotting.
- Run scripts → outputs saved to `reports/figures/`

**Packages:**  
- `pandas`, `geopandas`, `numpy`, `matplotlib`, `seaborn`
- **Dependencies:** `src.visulaization.plotting_module`
---

### Step 6 — Crop & Topo-Climatic Disaggregation

---

#### 6A) Add Elevation, Climate & Main Cultivated Crop Information to Grid Data

**Script:** `src/data/maincrop_klima_elev.py`

**Functions:**  
Merges grid-level data with:
- Climate zones (shapefile)
- Information on crop occupying the largest area in each region
- Elevation (raster)

**Output:** 
Grid geodataframe with added columns: `data/interim/gridgdf/gridgdf_klima_crop_elev.pkl`

**Packages:**  
`os`, `numpy`, `pandas`, `geopandas`, `matplotlib`, `seaborn`, `shapely`, `rasterio`, `rasterstats`

---

#### 6B) Crop Disaggregated Analysis

**Script:** `src/analysis/hetero/crop_disaggregation.py`

This script is used to analyse field structure metrics changes by crop group (2012–2023) for appendix A and B as well as Table 2.


**Workflow:**  
1. Load `gridgdf_klima_crop_elev.pkl` + field data.
2. For target year (e.g. 2023), subset field data by main crop in the LANDKREIS.
3. Compute aggregate area, median size, shape, field counts.
4. Save tables → `report/kchange_csvs/`

**Packages:** `os`, `pandas`  
**Dependencies:** `src.analysis.desc.gridgdf_desc`

---

#### 6C) Climate & Elevation Relationship

**Script:** `src/analysis/hetero/crop_disaggregation.py`

This script is used to analyse how field size & field count changes relate to climate & elevation. Prepares Figure 6 plots

**Workflow:**  
1. Load grid data.
2. Visualize mean elevation, spatial patterns.
3. Crosstabs & heatmaps for combinations.
4. Scatterplots + regressions colored by climate zone.
5. Save to `results/`.

**Packages:** `os`, `pandas`, `numpy`, `seaborn`, `matplotlib`  
**Input:** `gridgdf_klima_crop_elev.pkl`
**Dirs:** `interim/`, `raw/`, `results/`

---

### Step 7 — Relate Field Structure Change to Farm Statistics

---

#### 7A) Process Farm & Animal Data

**Script:** `src/data/gridded_farmanimchange_data.py`

**Workflow:**  
Load, clean, and merge farm census and animal census data (XML inside ZIPs) with gridded data for Niedersachsen.  
Calculate yearly & relative changes for farm variables.  
Output another grid-level GeoDataFrame for further analysis.

1. Make sure:
   - `animaldata.zip` and `farmdata.zip` are in `data/raw/`.
   - `gridgdf_klima_crop_elev.pkl` is ready.
2. Run the script.
3. Output: `grid_fanim.pkl`

**Requirements:**
- **Python:** 3.x
- **Packages:** `pandas`, `numpy`, `geopandas`, `matplotlib`, `xml`, `os`, `zipfile`
- **Data Files:**
  - `data/raw/animaldata.zip` (XML)
  - `data/raw/farmdata.zip` (XML)
  - `gridgdf_klima_crop_elev.pkl`
- **Dirs:** `data/raw/`, `data/interim/gridgdf/` (read/write)

---

#### 7B) Examine Fram Structural Change Relationship to Field Structure

**Script:** `src/analysis/hetero/farmfield_relationship.py`

**Purpose:**  
Analyze & visualize how **farm counts**, **field structure**, and **livestock density** interact at grid level. Prepares Figure 7 plots.

**How to Use:**
1. Make sure `grid_fanim.pkl` is ready.
2. Run the script.
3. Outputs include:  
   - Bar plots  
   - Histograms  
   - Choropleth maps  
   - Grouped regressions  
   Saved to: `reports/figures/farm_field/`

**Requirements:**
- **Python:** 3.x
- **Packages:** `pandas`, `numpy`, `matplotlib`, `os`
- **Data Files:** `grid_fanim.pkl`
- **Dirs:** `data/interim/gridgdf/`, `reports/figures/farm_field/` (read/write)

---

---


## Acknowledgements

This readme was created following the guidelines from [Hindawi](https://social-science-data-editors.github.io/template_README/).