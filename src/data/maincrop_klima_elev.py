# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling

# %%
# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()


from src.analysis.desc import gridgdf_desc as gd

###########################################
# %% load typically needed existing data
gld, gridgdf = gd.silence_prints(gd.create_gridgdf)
# I always want to load gridgdf and process clean gridgdf separately so I can have uncleeaned data for comparison or sensitivity analysis
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)

###################################################################
# process climatic zone
###################################################################
# %%
def assign_attribute_by_largest_overlap(grid_gdf, region_shp_path, region_attr, grid_id='CELLCODE'):
    """
    Assigns a region attribute (e.g. KLIMAREGIO) to grid cells 
    based on the region with the largest area of overlap.
    
    Parameters:
        grid_gdf (GeoDataFrame): Grid data with unique geometries and ID
        region_shp_path (str): Path to shapefile with region geometries
        region_attr (str): Name of the attribute to assign (e.g. 'KLIMAREGIO')
        grid_id (str): Unique ID column of the grid (default: 'CELLCODE')
    
    Returns:
        GeoDataFrame: Updated grid_gdf with region_attr column added
    """
    # Read and reproject region data
    region_gdf = gpd.read_file(region_shp_path)
    region_gdf = region_gdf.to_crs(grid_gdf.crs)

    # Intersect geometries
    overlay = gpd.overlay(grid_gdf, region_gdf[[region_attr, 'geometry']], how='intersection')
    overlay['area'] = overlay.geometry.area

    # Keep the largest overlap per grid cell
    overlay_max = (
        overlay.sort_values('area', ascending=False)
               .drop_duplicates(subset=grid_id)
    )

    # Merge back to original grid
    merged = grid_gdf.merge(
        overlay_max[[grid_id, region_attr]],
        on=grid_id,
        how='left'
    )
    return merged

# %%
def plot_grid_with_attribute(grid_gdf, attr_col, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    grid_gdf.plot(column=attr_col, ax=ax, cmap='tab20', legend=True, edgecolor='gray', linewidth=0.3)
    ax.set_title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# %%
# Strip down grid to unique geometries
gridgdf = gridgdf_cl[['CELLCODE', 'geometry', 'LANDKREIS']].drop_duplicates('CELLCODE')

# Assign KLIMAREGIO
grid_with_klima = assign_attribute_by_largest_overlap(
    gridgdf, data_main_path+"/raw/Klimaregionen/Klimaregionen.shp", "KLIMAREGIO"
)
plot_grid_with_attribute(grid_with_klima, "KLIMAREGIO", "10km Grid with KLIMAREGIO")

# %% check that klima has all the LANDKREISs as gridgdf_cl
all_kreis_contained = set(gridgdf_cl['LANDKREIS']).issubset(set(grid_with_klima['LANDKREIS']))
if all_kreis_contained:
    print("All LANDKREIS in gridgdf_cl are contained in klima.")
else:
    print("Not all LANDKREIS in gridgdf_cl are contained in klima.")


all_kreis_contained = set(gridgdf_cl['CELLCODE']).issubset(set(grid_with_klima['CELLCODE']))
if all_kreis_contained:
    print("All CELLCODEs in gridgdf_cl are contained in klima.")
else:
    print("Not all CELLCODEs in gridgdf_cl are contained in klima.")

###################################################################
# Also obtain main cultivated crop or kulturart FOR EACH landkreis
###################################################################
# %% Sum area_ha across years by LANDKREIS and kulturart
area_sum = (
    gld.groupby(['LANDKREIS', 'kulturart', 'Gruppe'], as_index=False)['area_ha']
       .sum()
       .rename(columns={'area_ha': 'total_area_ha'})
)

# For each LANDKREIS, find the kulturart with the maximum total_area_ha
idx = area_sum.groupby('LANDKREIS')['total_area_ha'].idxmax()

dominant_kulturart = area_sum.loc[idx].reset_index(drop=True)
# dominant_kulturart has columns: LANDKREIS, kulturart, Gruppe, total_area_ha

grid_klima_crop = grid_with_klima.merge(
    dominant_kulturart[['LANDKREIS', 'kulturart', 'Gruppe', 'total_area_ha']],
    on='LANDKREIS',
    how='left'
)
grid_klima_crop.info()

#%% examine data in maps
plot_grid_with_attribute(grid_klima_crop, "KLIMAREGIO", "10km Grid - Klimaregion")
plot_grid_with_attribute(grid_klima_crop, "Gruppe", "LANDKREIS - Dominant CropGroup")

###################################################################
# Merge back to yearly data
###################################################################
# %%
grid_klima_crop_df = grid_klima_crop.drop(columns='geometry')
# merge gridgdf_cl with naturraum_klima_east on CELLCODE and LANDKREIS
gridgdf_cluster = gridgdf_cl.merge(grid_klima_crop_df, on=['CELLCODE', 'LANDKREIS'],
                                   how='left')
gridgdf_cluster.info()

# %%
# column renames
gridgdf_cluster = gridgdf_cluster.rename(columns={"KLIMAREGIO": "main_Klima",
                                                        "kulturart": "main_crop",
                                                        "Gruppe": "main_crop_group",
                                                        "total_area_ha": "main_crop_totarea"})

# %%
# gdf contains unique cellcode, their landkreis, predominant climate region,
# predominant crop type and associated group

# Check for missing values in the merged DataFrame
missing_values = gridgdf_cluster.isnull().sum()
print("Missing values in the merged DataFrame:")
print(missing_values[missing_values > 0])

# %%
# check the first and last columns of gridgdf_cluster
print("First columns:")
print(gridgdf_cluster.columns[:10])
print("Last columns:")
print(gridgdf_cluster.columns[-10:])
       
# %% Map climate and main topographic element to english translations
# Define translation mapping without "Region"
climate_translation = {
    "Maritim-Subkontinentale Region": "Maritime–Subcontinental",
    "Maritime Region": "Maritime",
    "Subkontinentale Region": "Subcontinental",
    "Submontane Region": "Submontane"
}

# Create new English column with translated climate zones
gridgdf_cluster.loc[:, "main_Klima_EN"] = gridgdf_cluster["main_Klima"].map(climate_translation)

# Optional check for unmapped values
unmapped = gridgdf_cluster[gridgdf_cluster["main_Klima_EN"].isna()]["main_Klima"].unique()
if len(unmapped) > 0:
    print("⚠️ Unmapped values found:", unmapped)

# create categorical bins for change columns
# this is useful for heatmapping mean climate information and range of change
bins = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labels = ["<-6%", "-4 to -6%", "-2 to -4%", "-1 to -2%", "0 to -1%", "0 to 1%", "1 to 2%", "2 to 4%", "4 to 6%", ">6%"]
columns = ["medfs_ha_percdiff_to_y1", "medperi_percdiff_to_y1", "fields_ha_percdiff_to_y1", "medshape_percdiff_to_y1"]

for col in columns:
    binned_col = f"{col}_bins"
    gridgdf_cluster[binned_col] = pd.cut(gridgdf_cluster[col], bins=bins, labels=labels, right=False)


###########################################
# add mean elevation data
###########################################
# %% 1. checkout raster data
raster_file_path = data_main_path+"/raw/lsax_elevation.tif"
with rasterio.open(raster_file_path) as src:
    print(f"Raster: {raster_file_path}")
    print(f"- CRS: {src.crs}")
    print(f"- Dimensions: {src.width} x {src.height}")
    print(f"- Bands: {src.count}")
    print(f"- NoData: {src.nodata}")

    for i in range(1, src.count + 1):
        band = src.read(i)
        print(f"\nBand {i} stats:")

        # Apply NoData mask
        if src.nodata is not None:
            band = np.ma.masked_equal(band, src.nodata)
        else:
            band = np.ma.masked_invalid(band)  # in case of NaN values

        # Check if band is not completely masked
        if band.count() > 0:
            print(f"  Min:  {band.min()}")
            print(f"  Max:  {band.max()}")
            print(f"  Mean: {band.mean():.2f}")
            print("Shape of band:", band.shape)
            print("First few values:", band[:5, :5])
        else:
            print("  All values are NoData or masked.")

with rasterio.open(raster_file_path) as src:
    elevation_data = src.read(1)
    raster_crs = src.crs
    print(raster_crs)

# Plot the raster data using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(elevation_data, cmap='viridis')  # You can change the colormap (cmap)
plt.colorbar(label="Elevation (m)")  # Add a color bar to show elevation scale
plt.title("Raster Elevation Data")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()

# %% 2. create gridgdf by removing duplicates so as to keep only one year's unique gridcell but all cells
# and reproject raster data
gridgdf = gridgdf_cluster[['CELLCODE', 'geometry']].drop_duplicates('CELLCODE')

target_crs = gridgdf.crs

# Input and output paths
src_path = data_main_path+"/raw/lsax_elevation.tif"
dst_path = data_main_path+"/raw/lsax_elevation_reprojected.tif"

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds)
    
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': target_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open(dst_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear)

# %% 3. check that all of gridgdf geometries is covered by elevation bounds
# Load the reprojected raster to get its bounds
with rasterio.open(dst_path) as src:
    raster_bounds = box(*src.bounds)
    raster_crs = src.crs

# Create a GeoDataFrame for the raster extent
raster_extent_gdf = gpd.GeoDataFrame({'geometry': [raster_bounds]}, crs=raster_crs)

# Reproject grid if needed
if gridgdf.crs != raster_crs:
    gridgdf = gridgdf.to_crs(raster_crs)

# Check which grid cells intersect the raster
intersections = gpd.sjoin(gridgdf, raster_extent_gdf, predicate='intersects', how='left')
covered_cells = intersections['index_right'].notna().sum()
print(f"Grid cells overlapping raster: {covered_cells} / {len(gridgdf)}")

# Optional: remove non-intersecting geometries
gridgdf = intersections[intersections['index_right'].notna()].drop(columns='index_right')


# %% 4. compute mean elevation for every geometry row in gridgdf
# Compute mean elevation per grid cell
elevation_stats = zonal_stats(
    gridgdf,
    dst_path,
    stats=["mean"],
    nodata=-32768,  # <- replace with actual nodata if different
    geojson_out=True
)

# Attach results to GeoDataFrame
from shapely.geometry import shape

# Build GeoDataFrame from results
gridgdf['mean_elevation'] = [feat['properties']['mean'] for feat in elevation_stats]
gridgdf.info()


# %% 5. there appers to be numm elevation values. Let's check why.
missing = gridgdf[gridgdf['mean_elevation'].isna()]
print(f"⚠️ Rows with missing elevation: {len(missing)}")

missing_cellcodes = gridgdf[gridgdf['mean_elevation'].isna()]['CELLCODE'].unique()
print("CELLCODEs with missing elev:")
print(missing_cellcodes)

# %% check min, max and all elevation values here
with rasterio.open(dst_path) as src:
    masked_data, masked_transform = mask(
        src,
        missing.geometry,
        crop=True,
        nodata=src.nodata,
        all_touched=True  # use False if you want stricter coverage
    )

    # Extract valid values
    values = masked_data[0]  # assuming 1 band
    valid_values = values[values != src.nodata]  # or use `np.ma.masked_equal(values, src.nodata).compressed()`

    """
    COMMENTJB: the values below are not truly "valid_values", because all of them are nan.
     my alternative suggestion therefore:
    write:
    if int(np.invert(np.isnan(masked_data[0])).sum())==0:
        print("No valid elevation data within this cell")
    """
    if valid_values.size > 0:
        print(f"Min elevation: {valid_values.min()}")
        print(f"Max elevation: {valid_values.max()}")
        print(f"All values (flattened):\n{valid_values}")
    else:
        print("⚠️ No valid elevation data within this cell.")


# %% examine the problem cell over elevation plot
# Filter only missing cells
missing = gridgdf[gridgdf['mean_elevation'].isna()].copy()

# Load raster and plot
raster_path = data_main_path+"/raw/lsax_elevation_reprojected.tif"
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    fig, ax = plt.subplots(figsize=(10, 10))

    show(src, ax=ax, cmap='viridis')
    # Plot all grids boundaries
    gridgdf.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Plot boundaries of missing cells
    missing.boundary.plot(ax=ax, edgecolor='red', linewidth=1.2)

    # Add CELLCODE labels
    for idx, row in missing.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(row['CELLCODE']), fontsize=4.3, color='red', ha='center')

    ax.set_title("Missing Elevation Cells over Raster")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# %% 6. fill CELLCODE '10kmE438N335' na with VALUE FROM '10kmE437N335' 
# because we have only few fields in the '10kmE438N335' wih no elev value
# and it is more likely that the fields have elevation of the neighboring area in niedersachsen

# 1. Get the elevation value from CELLCODE '10kmE437N335'
reference_cellcode = '10kmE437N335'
ref_value = gridgdf.loc[gridgdf['CELLCODE'] == reference_cellcode, 'mean_elevation'].values

# 2. Check if we got a valid result
if ref_value.size == 1:
    fill_value = ref_value[0]
    print(f"Filling missing values with elevation: {fill_value:.2f} from '{reference_cellcode}'")

    # 3. Fill elevation for '10kmE438N335'
    target_cell = '10kmE438N335'
    condition = (gridgdf['CELLCODE'] == target_cell) & (gridgdf['mean_elevation'].isna())
    gridgdf.loc[condition, 'mean_elevation'] = fill_value

    print(f"Filled {condition.sum()} row(s) in '{target_cell}'")
    
else:
    print(f"⚠️ Reference CELLCODE '{reference_cellcode}' not found or has multiple values.")

# %%
gridgdf.info()

# %% 7. merge back to yearly data
gridgdf_cluster_new = gridgdf_cluster.merge(gridgdf[['CELLCODE', 'mean_elevation']],
                    on='CELLCODE',
                    how='left')
gridgdf_cluster_new.info()

# %% save
gridgdf_cluster_new.to_pickle(data_main_path+"/interim/gridgdf/gridgdf_klima_crop_elev.pkl")


# %%
