#%%
import os
import requests
import zipfile
import geopandas as gpd
import pandas as pd
import math as m
import tempfile
import logging
import gc
from pathlib import Path
from typing import List, Union
from src.data import gridregionjoin


# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

print("Current working dir:", project_root)


#this script loads the 2012 - 2023 niedersacsen data, filters with landesflaeche shapefile to remove areas outside of niedersacsen boundaries,
#spatially joins all years with regional information (kreise) and eea reference 10km grid, and prepares it for analysis.
#the original data extracted from original zip and some renamed for looping can be found in N:\ds\data\Niedersachsen\Niedersachsen\Needed
#the land and kreise data in N:\ds\data\Niedersachsen\verwaltungseinheiten
#eea reference data in /data/raw of the current project directory

'''Simply run this script and the processed data will be saved as a pickle file – data/interim/gld_base_ori.pkl.'''

# Initialize logging
# #note that time is displayed in utc by devcontainer default
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.info("Logging works!")

#######################Utility functions#########################
# functions for geometric measures
def paratio(p, a):
    return p/a

def shapeindex(p, a):
    return (0.282*p)/(m.sqrt(a))

# Functions to download zip files from given URLs and load data
def download_zip_files(urls, schlaege_dir):
    """
    Downloads each zip file from the provided URLs into schlaege_dir,
    if schlaege_dir exists. Skips already downloaded files.

    Parameters
    ----------
    urls : list of str
        URLs to the zip files.
    schlaege_dir : str
        Path to directory where zips should be saved.
    """
    if not os.path.isdir(schlaege_dir):
        print(f"Directory does not exist: {schlaege_dir}")
        return

    for url in urls:
        filename = os.path.basename(url)
        dest_path = os.path.join(schlaege_dir, filename)

        if os.path.exists(dest_path):
            print(f"Skipping (already exists): {filename}")
            continue

        try:
            print(f"Downloading: {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            print(f"Saved to: {dest_path}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


def normalize_data_info(data_info: List[Union[list, dict]]):
    """
    Normalize data_info into a list of dicts with keys: year, zip, shp, kultur.
    """
    normalized = []
    if not data_info:
        return normalized

    first = data_info[0]
    # Skip header if present
    if isinstance(first, list) and first and str(first[0]).lower() == "year":
        iterable = data_info[1:]
    else:
        iterable = data_info

    for row in iterable:
        if isinstance(row, dict):
            normalized.append({
                "year": row.get("year"),
                "zip": row.get("zip") or row.get("zipped_folder"),
                "shp": row.get("shp") or row.get("schlaege_shp")
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            normalized.append({
                "year": row[0],
                "zip": row[1],
                "shp": row[2],
            })
        else:
            logger.warning("Skipping malformed row: %r", row)
    return normalized


def load_geodata(base_dir: str, data_info: list) -> dict:
    """
    Loads geodata for all shapefiles listed in data_info from zipped archives.

    Returns a dict of year -> GeoDataFrame for successfully loaded years.
    """
    records = normalize_data_info(data_info)
    geodata = {}
    base_dir = Path(base_dir)

    for rec in records:
        year, zip_name, shp_name = rec["year"], rec["zip"], rec["shp"]
        if not all([year, zip_name, shp_name]):
            logger.warning("Skipping record with missing info: %r", rec)
            continue

        zip_path = base_dir / zip_name
        if not zip_path.exists():
            logger.warning("[%s] Zip not found: %s", year, zip_path)
            continue

        try:
            with zipfile.ZipFile(zip_path) as z:
                # Try direct access using fiona's /vsizip/ path if possible
                vsizip_path = f"/vsizip/{zip_path}/{shp_name}"
                try:
                    geodf = gpd.read_file(vsizip_path)
                    geodata[year] = geodf
                    logger.info("[%s] Loaded via /vsizip/ (%d features)", year, len(geodf))
                    continue
                except Exception as e:
                    logger.info("[%s] /vsizip/ failed, extracting temporarily: %s", year, e)

                # Extract relevant files to temp directory for fallback load
                with tempfile.TemporaryDirectory() as temp_dir:
                    for name in z.namelist():
                        if Path(name).stem == Path(shp_name).stem and Path(name).suffix in {".shp", ".dbf", ".shx", ".prj", ".cpg"}:
                            z.extract(name, temp_dir)
                    extracted_shp = Path(temp_dir) / shp_name
                    # Find .shp in temp_dir if path is nested
                    if not extracted_shp.exists():
                        candidates = list(Path(temp_dir).rglob(Path(shp_name).name))
                        if candidates:
                            extracted_shp = candidates[0]
                        else:
                            logger.warning("[%s] Could not find extracted .shp: %s", year, shp_name)
                            continue

                    try:
                        geodf = gpd.read_file(str(extracted_shp))
                        geodata[year] = geodf
                        logger.info("[%s] Loaded via extraction (%d features)", year, len(geodf))
                    except Exception as e:
                        logger.error("[%s] Failed to load shapefile after extraction: %s", year, e)
        except Exception as ex:
            logger.error("[%s] Error processing: %s", year, ex)
    return geodata

# functions to process the data
def harmonize_columns(data):
    # Rename year columns
    old_year_names = ['jahr', 'antjahr', 'antragsjah']
    new_year_name = 'year'
    old_kulturcode_names = ['kc_gem', 'kulturartf', 'kc_festg', 'kc', 'nc_festg', 'kulturcode']
    new_kulturcode_name = 'kulturcode'
    old_area_names = ['gemeldetef', 'flaeche_ge', 'akt_fl', 'aktuellefl']
    new_area_name = 'area'
    columns_to_delete = ['schlag_nr', 'teilschlag', 'schlagnr', 'schlagbeze']

    for year in data:
        df = data[year]

        # Normalize column names to lowercase for easier matching
        df.columns = df.columns.str.lower()

        # Rename year
        for old_name in old_year_names:
            if old_name in df.columns:
                df.rename(columns={old_name: new_year_name}, inplace=True)

        # Rename kulturcode
        for old_name in old_kulturcode_names:
            if old_name in df.columns:
                df.rename(columns={old_name: new_kulturcode_name}, inplace=True)

        # Rename area
        for old_name in old_area_names:
            if old_name in df.columns:
                df.rename(columns={old_name: new_area_name}, inplace=True)

        print(f"{year} Columns after rename: {df.columns}")

        # Convert year to integer
        if 'year' in df.columns:
            df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year.astype(int)

        # Check kulturcode
        if 'kulturcode' in df.columns:
            unique_kulturcodes = df['kulturcode'].unique()
            non_numeric = [code for code in unique_kulturcodes if not str(code).replace('.', '', 1).isdigit()]
            if non_numeric:
                logging.warning(f"{year}: Non-numeric kulturcode values found: {non_numeric}")
            else:
                df['kulturcode'] = df['kulturcode'].astype(int)
                logging.info(f"{year}: All kulturcode values are numeric. Converted to int.")

        # Drop unnecessary columns
        data[year].drop(columns=[col for col in columns_to_delete if col in data[year].columns], inplace=True)

    return data

def first_index_reset(data, years):
    for year in years:
        data[year] = data[year].reset_index().rename(columns={'index': 'id'})
        logging.info(f"{year}: Index has been reset.")
    return data

def calculate_geometric_measures(all_years):
    all_years['area_m2'] = all_years.area
    all_years['area_ha'] = all_years['area_m2'] * (1/10000)
    all_years['peri_m'] = all_years.length
    all_years['par'] = all_years.apply(lambda row: paratio(row['peri_m'], row['area_m2']), axis=1)
    all_years['shape'] = all_years.apply(lambda row: shapeindex(row['peri_m'], row['area_m2']), axis=1)
    return all_years

def process_and_save_years(data, years, land, out_dir, use_wkb_fallback=False, compression="zstd"):
    """
    Process each year: spatial join with land, check duplicates, drop temp cols, save parquet.
    Skips years if the output file already exists.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved_paths = {}

    for year in years:
        out_path = os.path.join(out_dir, f"gld_{year}.parquet")

        # Skip if already saved
        if os.path.exists(out_path):
            logging.info(f"{year}: File already exists at {out_path}, skipping.")
            saved_paths[year] = out_path
            continue

        logging.info(f"{year}: Starting processing...")

        # 1. Spatial join
        gdf = gpd.sjoin(data[year], land, how='inner', predicate='intersects')
        logging.info(f"{year}: Spatially joined with land boundary.")

        # 2. Duplicate check
        dup_count = gdf[["year", "id"]].duplicated().sum() if "id" in gdf.columns else 0
        logging.info(f"{year}: {dup_count} duplicates found after join.")

        # 3. Drop temp columns if they exist
        drop_cols = [col for col in ['id', 'index_right', 'LAND'] if col in gdf.columns]
        if drop_cols:
            gdf.drop(columns=drop_cols, inplace=True)
            
        # 4. Calculate geometric measures
        gdf = calculate_geometric_measures(gdf)
        logging.info(f"{year}: Calculated geometric measures.")

        # 5. Save to parquet
        if not use_wkb_fallback:
            try:
                gdf.to_parquet(out_path, index=False, compression=compression)
                logging.info(f"{year}: Saved GeoParquet to {out_path}")
                saved_paths[year] = out_path
                del gdf
                gc.collect()
                continue
            except Exception as e:
                logging.warning(f"{year}: GeoParquet write failed, falling back to WKB. Error: {e}")

        # Fallback: store geometry as WKB
        gdf["geom_wkb"] = gdf.geometry.apply(lambda g: None if g is None else g.wkb)
        gdf_no_geom = gdf.drop(columns=[gdf.geometry.name])

        gdf_no_geom.to_parquet(out_path, index=False, compression=compression)
        logging.info(f"{year}: Saved parquet with WKB geometry to {out_path}")
        saved_paths[year] = out_path


    return saved_paths


def spatial_join_with_gridregion(gdf, grid_landkreise):
    gdf_landkreise = gpd.sjoin(gdf, grid_landkreise, how='left', predicate="intersects")
    return gdf_landkreise

def handle_grid_duplicates(gdf_landkreise, grid_landkreise):

    # Step 1: Identify duplicate entries based on the 'id' column
    duplicates = gdf_landkreise.duplicated('id')
    print(f"Number of duplicate entries found: {duplicates.sum()}")  # Display the number of duplicates

    if duplicates.any():
        # Step 2: Create a DataFrame containing only the double-assigned polygons
        double = gdf_landkreise[gdf_landkreise.index.isin(
            gdf_landkreise[gdf_landkreise.index.duplicated()].index
        )]

        # Step 3: Remove these double-assigned polygons from the original DataFrame
        gdf_landkreise = gdf_landkreise[~gdf_landkreise.index.isin(
            gdf_landkreise[gdf_landkreise.index.duplicated()].index
        )]

        # Step 4: Calculate the intersection area for each polygon in 'double'
        doublecopy = double.copy()
        doublecopy['intersection'] = [
            a.intersection(grid_landkreise[grid_landkreise.index == b].geometry.values[0]).area / 10000
            for a, b in zip(doublecopy.geometry.values, doublecopy.index_right)
        ]

        # Step 5: Sort by intersection area and keep the row with the largest intersection for each 'id'
        doublesorted = (doublecopy.sort_values(by='intersection').groupby('id', group_keys=False)
                        .apply(lambda g: g.tail(1)).reset_index(drop=True))

        # Step 6: Merge the cleaned double-assigned polygons back into the main DataFrame
        gdf_regions = pd.concat([gdf_landkreise, doublesorted])
        
        # Step 7: recheck for duplicates
        duplicates_after = gdf_regions.duplicated('id')
        print(f"Number of duplicate entries after handling: {duplicates_after.sum()}")  #

        return gdf_regions
    else:
        print("No duplicates found. Returning the original DataFrame.")
        return gdf_landkreise

# final function to load and process data
def load_data(loadExistingData=False):
    years = range(2012, 2025)

    # url list for downloading Niedersachsen Schlaege data
    urls = [
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_skizzen_2012.zip"
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_skizzen_2013.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_skizzen_2014.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_skizzen_2015.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2016.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2017.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2018.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2019.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2020.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2021.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2022.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2023.zip",
        "https://sla.niedersachsen.de/mapbender_sla/download/schlaege_hauptzahlung_2024.zip",

    ]

    # data information for each year
    data_info = [
        ['year', 'zipped_folder', 'schlaege_shp'],
        [2012, 'schlaege_skizzen_2012.zip', 'ud_12_s.shp'],
        [2013, 'schlaege_skizzen_2013.zip', 'ud_13_s.shp'],
        [2014, 'schlaege_skizzen_2014.zip', 'ud_14_s.shp'],
        [2015, 'schlaege_skizzen_2015.zip', 'ud_15_s.shp'],
        [2016, 'schlaege_hauptzahlung_2016.zip', 'ud_16_ts_anh2.shp'],
        [2017, 'schlaege_hauptzahlung_2017.zip', 'ud_17_ts_anh.shp'],
        [2018, 'schlaege_hauptzahlung_2018.zip', 'ud_18_ts_bewi.shp'],
        [2019, 'schlaege_hauptzahlung_2019.zip', 'ud_19_ts_bewi.shp'],
        [2020, 'schlaege_hauptzahlung_2020.zip', 'ud_20_ts_bewi.shp'],
        [2021, 'schlaege_hauptzahlung_2021.zip', 'ud_21_ts_bewi.shp'],
        [2022, 'schlaege_hauptzahlung_2022.zip', 'ud_22_ts_bewi.shp'],
        [2023, 'schlaege_hauptzahlung_2023.zip', 'ud_23_ts_akt_bewi.shp'],
        [2024, 'schlaege_hauptzahlung_2024.zip', 'ud_24_ts_akt_bewi.shp']
    ]

    # Paths
    base_dir = data_main_path+"/raw"
    schlaege_dir = os.path.join(base_dir, "nieder_origin/schlaege")
    gld_dir = data_main_path+"/interim/gld_per_year"
    admin_files_dir = os.path.join(base_dir, "verwaltungseinheiten")

    # Check paths exist for decision logic
    final_paths = {year: f"{gld_dir}/gld_{year}_gridjoined.parquet" for year in years}
    init_files  = {year: f"{gld_dir}/gld_{year}.parquet" for year in years}

    all_field_files_exist = all(os.path.exists(path) for path in final_paths.values())
    all_init_files_exist  = all(os.path.exists(path) for path in init_files.values())

    # -------------------------------------------------------
    # CASE 1: Just load final processed files if they exist
    # -------------------------------------------------------
    if loadExistingData and all_field_files_exist:
        logging.info("All grid-joined files already exist. Loaded.")
        return final_paths

    # -------------------------------------------------------
    # CASE 2: Start from Step 2 if intermediate parquet files exist
    # -------------------------------------------------------
    elif loadExistingData and all_init_files_exist:
        logging.info("Initial files exist. Skipping Step 1, starting from Step 2.")

        # Load grid region layer and set CRS
        grid_landkreise = gridregionjoin.join_gridregion(loadExistingData=True)

        for year in years:
            in_parquet  = f"{gld_dir}/gld_{year}.parquet"
            out_parquet = f"{gld_dir}/gld_{year}_gridjoined.parquet"

            if os.path.exists(out_parquet):
                logging.info(f"{year}: Already processed. Skipping.")
                continue

            gdf_year = gpd.read_parquet(in_parquet)
            gdf_year = gdf_year.reset_index().rename(columns={'index': 'id'})
            
            # --- CRS Check and Align ---
            logging.info(f"{year}: CRS of input data: EPSG:{gdf_year.crs.to_epsg()}")
            if grid_landkreise.crs != gdf_year.crs:
                logging.warning(f"{year}: CRS mismatch detected ({grid_landkreise.crs} vs {gdf_year.crs}). Reprojecting.")
                grid_landkreise.crs = grid_landkreise.to_crs(gdf_year)

            # Spatial join
            gdf_landkreise = spatial_join_with_gridregion(gdf_year, grid_landkreise)
            gdf_regions = handle_grid_duplicates(gdf_landkreise, grid_landkreise)
            gdf_regions = gdf_regions.drop(columns=['id', 'index_right', 'intersection'])
            gdf_regions = gdf_regions.reset_index().rename(columns={'index': 'id'})

            logging.info(f"{year}: Spatial join complete, {len(gdf_regions)} rows.")

            # Save
            gdf_regions.to_parquet(out_parquet)
            logging.info(f"{year}: Saved {out_parquet}")

            del gdf_year, gdf_landkreise, gdf_regions
            gc.collect()

        return final_paths

    # -------------------------------------------------------
    # CASE 3: Start from raw data (full pipeline)
    # -------------------------------------------------------
    else:
        logging.info("Starting full processing pipeline from raw data.")

        #if schlaege_dir or gld_dir does not yet exist, create it
        os.makedirs(schlaege_dir, exist_ok=True)
        os.makedirs(gld_dir, exist_ok=True)
        
        # Step 1: Download and load
        download_zip_files(urls, schlaege_dir)
        data = load_geodata(schlaege_dir, data_info)
        data = harmonize_columns(data)
        data = first_index_reset(data, years)

        # Load admin area shape and set CRS
        land = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landesflaeche.shp"))
        land = land.to_crs(epsg=25832)

        # Save initial per-year parquet files
        init_files = process_and_save_years(data, years, land, gld_dir, use_wkb_fallback=False)

        # Step 2: Spatial join with grid_region
        grid_landkreise = gridregionjoin.join_gridregion(loadExistingData=True)

        for year in years:
            in_parquet  = f"{gld_dir}/gld_{year}.parquet"
            out_parquet = f"{gld_dir}/gld_{year}_gridjoined.parquet"

            if os.path.exists(out_parquet):
                logging.info(f"{year}: Already processed. Skipping.")
                continue

            gdf_year = gpd.read_parquet(in_parquet)
            gdf_year = gdf_year.reset_index().rename(columns={'index': 'id'})
            
            # --- CRS Check and Align ---
            logging.info(f"{year}: CRS of input data: EPSG:{gdf_year.crs.to_epsg()}")
            if grid_landkreise.crs != gdf_year.crs:
                logging.warning(f"{year}: CRS mismatch detected ({grid_landkreise.crs} vs {gdf_year.crs}). Reprojecting.")
                grid_landkreise.crs = grid_landkreise.to_crs(gdf_year)

            # Spatial join
            gdf_landkreise = spatial_join_with_gridregion(gdf_year, grid_landkreise)
            gdf_regions = handle_grid_duplicates(gdf_landkreise, grid_landkreise)
            gdf_regions = gdf_regions.drop(columns=['id', 'index_right', 'intersection'])
            gdf_regions = gdf_regions.reset_index().rename(columns={'index': 'id'})

            logging.info(f"{year}: Spatial join complete, {len(gdf_regions)} rows.")

            # Save
            gdf_regions.to_parquet(out_parquet)
            logging.info(f"{year}: Saved {out_parquet}")

            del gdf_year, gdf_landkreise, gdf_regions
            gc.collect()
            logging.info("Done!")

        return final_paths

    
# %%
if __name__ == '__main__':
    loadExistingData = True
    gld_paths = load_data(loadExistingData)


