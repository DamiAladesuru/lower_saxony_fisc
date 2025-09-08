# %%
import pandas as pd
import geopandas as gpd
import os, gc
from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc": # lower_saxony_fisc or workspace
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()


from src.data.processing_fielddata_utils import dataload as dl
from src.data.processing_fielddata_utils import eca_new as eca

''' This script contains functions for:
    - modifying gld to contain kulturcode descriptions.
    - extracting and saving field geometries separately
    - saving and loading gld without geometries
    - loading field data deometries for given years
'''

def load_geodata():
    gld_paths = dl.load_data(loadExistingData=True)
    kulturcode_mastermap = eca.process_kulturcode(data_main_path, load_existing=True)
    # ensure no multiple matches per old_kulturcode
    kulturcode_mastermap_unique = kulturcode_mastermap.drop_duplicates(subset=['old_kulturcode'])
            
    years = range(2012, 2025)
    gld_allyears = {}  # Will store year → GeodataFrame of data

    for year in years:
        # Load the gld data for the current year
        gld_path = gld_paths[year]
        gld = gpd.read_parquet(gld_path)
        print(f"{year}: CRS of data: EPSG:{gld.crs.to_epsg()}")
        
        # merge gld on 'kulturcode' with kulturcode_mastermap on 'old_kulturcode'
        gld = gld.rename(columns={'kulturcode': 'old_kulturcode'})
        gld = gld.merge(kulturcode_mastermap_unique, on='old_kulturcode', how='left')
        # drop 'int_kulturcode' and rename 'old_kulturcode' to 'kulturcode'
        gld = gld.drop(columns=['old_kulturcode','int_kulturcode'])
        gld = gld.rename(columns={'new_kulturcode': 'kulturcode'})
        
        # Apply threshold of minimum 100m2 fields
        gld = gld[~(gld['area_m2'] < 100)]
        print(f"updated {year} data")
        
        # reset index
        gld = gld.reset_index()
    
    # save the gld data to a dictionary
        gld_allyears[year] = gld
        
    # Clean up memory
    del gld, kulturcode_mastermap
    gc.collect()

    return gld_allyears

def get_gld_nogeoms():
    """
    Return dict of dataframes without geometry, one per year.
    - Saves geometry parquet files (if missing) in gldgeoms_dir.
    - Saves no-geometry parquet files (if missing) in gld_no_geoms_dir.
    - If all no-geometry parquet files already exist, skip load_geodata().
    """
    gldgeoms_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    gld_no_geoms_dir = os.path.join(data_main_path, "interim", "gld_no_geoms")
    os.makedirs(gldgeoms_dir, exist_ok=True)
    os.makedirs(gld_no_geoms_dir, exist_ok=True)

    gld_no_geom = {}

    # check which years are already stored as no-geom parquet
    existing_files = {
        f.split("_")[-1].replace(".parquet", "")
        for f in os.listdir(gld_no_geoms_dir)
        if f.startswith("gld_no_geom_") and f.endswith(".parquet")
    }

    if existing_files:
        # If cached no-geom files exist, load only those
        print(f"Found cached gld_no_geom files for years: {sorted(existing_files)}")
        for fname in os.listdir(gld_no_geoms_dir):
            if fname.startswith("gld_no_geom_") and fname.endswith(".parquet"):
                year = fname.split("_")[-1].replace(".parquet", "")
                path = os.path.join(gld_no_geoms_dir, fname)
                gld_no_geom[year] = pd.read_parquet(path)
        return gld_no_geom

    else:
        # Otherwise, load full geodata and build fresh
        gld_allyears = load_geodata()
        
        for year, gdf in gld_allyears.items():
            geom_path = os.path.join(gldgeoms_dir, f"geometry_{year}.parquet")
            nogeom_path = os.path.join(gld_no_geoms_dir, f"gld_no_geom_{year}.parquet")

            # Save geometry if missing
            if not os.path.exists(geom_path):
                geom_df = gdf[['row_id', 'geometry']].copy()
                geom_df.to_parquet(geom_path, index=False)
                print(f"Saved geometry for {year} → {geom_path}")
            else:
                print(f"Geometry for {year} already exists, skipping → {geom_path}")

            # Save no-geometry file
            df_no_geom = gdf.drop(columns=['geometry']).copy()
            df_no_geom.to_parquet(nogeom_path, index=False)
            print(f"Saved no-geometry data for {year} → {nogeom_path}")

            gld_no_geom[year] = df_no_geom

        del gld_allyears
        gc.collect()
    
    return gld_no_geom


# Helper to load geometry parquet for a year
def load_gldgeom_for_year(year: int):

    geom_dir = os.path.join(data_main_path, "interim", "gld_geoms")
    geom_path = os.path.join(geom_dir, f"geometry_{year}.parquet")
    return gpd.read_parquet(geom_path)
