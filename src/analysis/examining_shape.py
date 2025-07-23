'''In gld, find rows where shape is 1.00'''
# %%
import os
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen/final")

from src.analysis.desc import gridgdf_desc as gd


gld, gridgdf = gd.silence_prints(gd.create_gridgdf)
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)

# %%
# Define target values
target_par_values = [1.00]

# Filter rows where 'par' is approximately equal to any target value
filtered_gld = gld[gld["shape"].apply(lambda x: np.any(np.isclose(x, target_par_values, atol=0.01)))]

# Display results
print(filtered_gld)

# %%
filtered_gld['cpar_25'] = ((0.25 * filtered_gld['peri_m']) / (filtered_gld['area_m2']**0.5))

# %%
# Check for duplicate 'flik' values in filtered_gld
duplicates = filtered_gld[filtered_gld.duplicated(subset=['flik'], keep=False)]
if not duplicates.empty:
    print("Duplicate 'flik' values found:")
    print(duplicates)
else:
    print("No duplicate 'flik' values found.")

# %%
selected = filtered_gld[filtered_gld['year'] == 2021].copy()

# %%
print("gld min SHAPE:", gld['shape'].min())
print("gld max SHAPE:", gld['shape'].max())
print("gld median SHAPE:", gld['shape'].median())
print("gld mean SHAPE:", gld['shape'].mean())
print("gridgdf_cl min SHAPE:", gridgdf_cl['medshape'].min())
print("gridgdf_cl max SHAPE:", gridgdf_cl['medshape'].max())

# %%
from shapely import affinity
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Add a suptitle here
fig.suptitle("Fields with SHAPE = 1", fontsize=14, y=0.9)

# Fixed box size for display
fixed_xlim = (-1.2, 1.2)
fixed_ylim = (-1.2, 1.2)

for i, (idx, row) in enumerate(selected.iterrows()):
    geom = row.geometry

    # Handle both Polygon and MultiPolygon
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        continue

    for poly in polygons:
        # Normalize the polygon: center and scale
        centroid = poly.centroid
        centered = affinity.translate(poly, xoff=-centroid.x, yoff=-centroid.y)

        bounds = centered.bounds  # (minx, miny, maxx, maxy)
        max_dim = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if max_dim == 0:
            continue  # skip degenerate geometry

        scaled = affinity.scale(centered, xfact=2/max_dim, yfact=2/max_dim, origin=(0, 0))

        x, y = scaled.exterior.xy
        ax[i].plot(x, y, 'b-', linewidth=1)
        ax[i].fill(x, y, 'lightgray', alpha=0.5)

    ax[i].set_xlim(fixed_xlim)
    ax[i].set_ylim(fixed_ylim)
    ax[i].set_aspect('equal')
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].text(
        0, fixed_ylim[0] - 0.1,
        f"Area: {row['area_ha']:.2f} ha\nSHAPE: {row['shape']:.2f}\ncpar_25: {row['cpar_25']:.2f}",
        fontsize=8, ha='center', va='top'
    )

plt.subplots_adjust(wspace=0.5)
plt.savefig("reports/figures/examiningshape.svg", format="svg", bbox_inches='tight')