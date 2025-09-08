# %%
from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity

from src.data.processing_fielddata_utils import field_dataset as fd

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()


# %% load data
gld_no_geom = fd.get_gld_nogeoms()

# %%
# --- Print total number of observations in each year
for year in sorted(gld_no_geom):
    print(f"{year}: Total observations = {len(gld_no_geom[year])}")
    
#---- Check total area of fields ----#
for year in sorted(gld_no_geom):
    if 'area' in gld_no_geom[year].columns:
        print(f"{year}: Total area = {gld_no_geom[year]['area_ha'].sum():,.2f}")
  
# %%
'''
Plot how the area changes over the years

1. For each DataFrame (data[year]), sum the area column.
2. Collect these sums in a list or Series.
3. Plot sum of area vs. year as a line plot.'''
def plot_total_area(gld_no_geom):
    area_sums = {}
    for year in sorted(gld_no_geom):
        df = gld_no_geom[year]
        if 'area_ha' in df.columns:
            area_sums[year] = df['area_ha'].sum()
        else:
            print(f"{year}: No 'area' column found.")

    plt.figure(figsize=(10, 6))
    plt.plot(area_sums.keys(), area_sums.values(), marker='o')
    plt.title("Total Area by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Area")
    plt.grid(True)
    
    #plt.savefig("reports/figures/total_area_per_year.png", dpi=300)

    plt.show()
plot_total_area(gld_no_geom)

# %%
'''
Plot how total observation changes over the years '''
# Collect counts
year_counts = {year: len(df) for year, df in gld_no_geom.items()}

# Sort by year
years = sorted(year_counts.keys())
counts = [year_counts[y] for y in years]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(years, counts, marker='o', linestyle='-', linewidth=2)
plt.title("Total Observations per Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Observations", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

# Add labels above points
for x, y in zip(years, counts):
    plt.text(x, y + max(counts)*0.01, str(y), ha="center", fontsize=9)

plt.tight_layout()
#plt.savefig("reports/figures/total_observations_per_year.png", dpi=300)
plt.show()


# %% Examine SHAPE OF fields
# 2. Define target values and extract fitting rows
target_shape_values = [1.00]
filtered_gld_dict = {}

for year, df in gld_no_geom.items():
    # Filter rows where 'shape' is approximately equal to any target value
    filtered = df[df["shape"].apply(lambda x: np.any(np.isclose(x, target_shape_values, atol=0.01)))]
    filtered_gld_dict[year] = filtered

# Concatenate all years into one DataFrame with a 'year' column
filtered_gld = pd.concat(
    [df.assign(year=year) for year, df in filtered_gld_dict.items()],
    ignore_index=True
)

# Display results
print(filtered_gld.head())
print(f"Final shape: {filtered_gld.shape}")

target_shape_values = [1.24]
filtered_gld_dict = {}

for year, df in gld_no_geom.items():
    # Filter rows where 'shape' is approximately equal to any target value
    filtered = df[df["shape"].apply(lambda x: np.any(np.isclose(x, target_shape_values, atol=0.001)))]
    filtered_gld_dict[year] = filtered

# Concatenate all years into one DataFrame with a 'year' column
filtered_gld_124 = pd.concat(
    [df.assign(year=year) for year, df in filtered_gld_dict.items()],
    ignore_index=True
)

# Display results
#print(filtered_gld_124.head())
print(f"Final shape: {filtered_gld_124.shape}")

target_shape_values = [2.00]
filtered_gld_dict = {}

for year, df in gld_no_geom.items():
    # Filter rows where 'shape' is approximately equal to any target value
    filtered = df[df["shape"].apply(lambda x: np.any(np.isclose(x, target_shape_values, atol=0.01)))]
    filtered_gld_dict[year] = filtered

# Concatenate all years into one DataFrame with a 'year' column
filtered_gld_2 = pd.concat(
    [df.assign(year=year) for year, df in filtered_gld_dict.items()],
    ignore_index=True
)

# Display results
#print(filtered_gld_2.head())
print(f"Final shape: {filtered_gld_2.shape}")

del filtered_gld_dict

# %%
# -----------------------------
# STEP 1: Select rows
# -----------------------------

# From filtered_gld_124 keep only row where row_id = A238
row1 = filtered_gld_124[filtered_gld_124["row_id"] == "A238"]

# From filtered_gld_2 keep only row where row_id = A587
row2 = filtered_gld_2[filtered_gld_2["row_id"] == "A587"]

# Put them together → sample_2012
sample_2012 = pd.concat([row1, row2], ignore_index=True)

# From filtered_gld keep only row where row_id = J838091
sample_2021 = filtered_gld[filtered_gld["row_id"] == "J838091"]

# -----------------------------
# STEP 2: Load geometry files and merge
# -----------------------------
# %%
# 2012 geometries
geom2012 = fd.load_gldgeom_for_year(2012)
sample_2012 = sample_2012.merge(
    geom2012[["row_id", "geometry"]],
    on="row_id",
    how="left"
)

# 2021 geometries
geom2021 = fd.load_gldgeom_for_year(2021)
sample_2021 = sample_2021.merge(
    geom2021[["row_id", "geometry"]],
    on="row_id",
    how="left"
)
# %%
# -----------------------------
# STEP 3: Combine all rows
# -----------------------------
selected = pd.concat([sample_2012, sample_2021], ignore_index=True)
selected = gpd.GeoDataFrame(selected, geometry="geometry", crs="EPSG:25832")

# -----------------------------
# STEP 4: Plot
# -----------------------------
fig, ax = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))
if len(selected) == 1:
    ax = [ax]  # make iterable if only one axis


fig.suptitle("Sample Fields with their SHAPE values", fontsize=14, y=0.9)

fixed_xlim = (-1.2, 1.2)
fixed_ylim = (-1.2, 1.2)

for i, (idx, row) in enumerate(selected.iterrows()):
    geom = row.geometry
    if geom is None:
        continue

    # Handle Polygon & MultiPolygon
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        continue

    for poly in polygons:
        # Normalize: center + scale
        centroid = poly.centroid
        centered = affinity.translate(poly, xoff=-centroid.x, yoff=-centroid.y)

        bounds = centered.bounds  # (minx, miny, maxx, maxy)
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if max_dim == 0:
            continue
        scaled = affinity.scale(centered, xfact=2/max_dim, yfact=2/max_dim, origin=(0, 0))

        x, y = scaled.exterior.xy
        ax[i].plot(x, y, 'b-', linewidth=1)
        ax[i].fill(x, y, 'lightgray', alpha=0.5)

    ax[i].set_xlim(fixed_xlim)
    ax[i].set_ylim(fixed_ylim)
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    # Add text with area + shape
    ax[i].text(
        0, fixed_ylim[0] - 0.1,
        f"Area: {row['area_ha']:.2f} ha\nSHAPE: {row['shape']:.2f}",
        fontsize=8, ha='center', va='top'
    )

plt.subplots_adjust(wspace=0.5)
os.makedirs("reports/figures/", exist_ok=True)
plt.savefig("reports/figures/examiningshape.svg", format="svg", bbox_inches='tight')
plt.show()

# %%
