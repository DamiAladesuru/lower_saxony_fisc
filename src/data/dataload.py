#%%
import os
import pickle
import zipfile
import geopandas as gpd
import pandas as pd
from datetime import datetime as dt
import math as m
import logging
from pathlib import Path
import sys

# Set up the project root directory
# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()
os.makedirs(data_main_path+"/interim", exist_ok=True)
sys.path.append(project_root)

from src.data import gridregionjoin



#%%
#this script loads the 2012 - 2023 niedersacsen data, filters with landesflaeche shapefile to remove areas outside of niedersacsen boundaries,
#spatially joins all years with regional information (kreise) and eea reference 10km grid, and prepares it for analysis.
#the original data extracted from original zip and some renamed for looping can be found in N:\ds\data\Niedersachsen\Niedersachsen\Needed
#the land and kreise data in N:\ds\data\Niedersachsen\verwaltungseinheiten
#eea reference data in /data/raw of the current project directory

'''Simply run this script and the processed data will be saved as a pickle file – data/interim/gld_base.pkl.'''


# Initialize logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Logging works!")
os.makedirs(data_main_path+"/interim", exist_ok=True)

# Utility functions for geometric measures
def paratio(p, a):
    return p/a

def shapeindex(p, a):
    return (0.282*p)/(m.sqrt(a))

def load_geodata(base_dir, years, specific_file_names):
    data = {}
    for year in years:
        zip_file_path = os.path.join(base_dir, "Needed", f"schlaege_{year}.zip")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            for specific_file_name in specific_file_names:
                if specific_file_name in file_names:
                    data[year] = gpd.read_file(f"/vsizip/{zip_file_path}/{specific_file_name}")
                else:
                    logging.warning(f"File {specific_file_name} does not exist in {zip_file_path}.")
    return data

def harmonize_columns(data):
    # Rename year columns
    old_year_names = ['jahr', 'antjahr', 'antragsjah']
    new_year_name = 'year'
    old_kulturcode_names = ['kc_gem', 'kc_festg', 'kc', 'nc_festg', 'kulturcode']
    new_kulturcode_name = 'kulturcode'
    old_area_names = ['shape_area', 'flaeche', 'akt_fl', 'aktuellefl']
    new_area_name = 'area'
    columns_to_delete = ['shape_leng', 'schlagnr', 'schlagbez']

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

def spatial_join_with_land(data, years, land):
    for year in years:
        data[year] = gpd.sjoin(data[year], land, how='inner', predicate='intersects')
        logging.info(f"{year}: Spatially joined with land boundary.")
    return data

def first_duplicates_check(data, years):
    for year in years:
        logging.info(f"{year}: Checked for duplicates after joining land. {data[year][['year', 'id']].duplicated().sum()} duplicates found.")
        data[year].drop(columns=['id', 'index_right', 'LAND'], inplace=True)
    return data    

def calculate_geometric_measures(all_years):
    all_years['area_m2'] = all_years.area
    all_years['area_ha'] = all_years['area_m2'] * (1/10000)
    all_years['peri_m'] = all_years.length
    all_years['par'] = all_years.apply(lambda row: paratio(row['peri_m'], row['area_m2']), axis=1)
    all_years['shape'] = all_years.apply(lambda row: shapeindex(row['peri_m'], row['area_m2']), axis=1)
    return all_years

def spatial_join_with_gridregion(all_years, grid_landkreise,):
    allyears_landkreise = gpd.sjoin(all_years, grid_landkreise, how='left', predicate="intersects")
    return allyears_landkreise

def handle_grid_duplicates(allyears_landkreise, grid_landkreise):

    # Step 1: Identify duplicate entries based on the 'id' column
    duplicates = allyears_landkreise.duplicated('id')
    print(f"Number of duplicate entries found: {duplicates.sum()}")  # Display the number of duplicates

    if duplicates.any():
        # Step 2: Create a DataFrame containing only the double-assigned polygons
        double = allyears_landkreise[allyears_landkreise.index.isin(
            allyears_landkreise[allyears_landkreise.index.duplicated()].index
        )]

        # Step 3: Remove these double-assigned polygons from the original DataFrame
        allyears_landkreise = allyears_landkreise[~allyears_landkreise.index.isin(
            allyears_landkreise[allyears_landkreise.index.duplicated()].index
        )]

        # Step 4: Calculate the intersection area for each polygon in 'double'
        double['intersection'] = [
            a.intersection(grid_landkreise[grid_landkreise.index == b].geometry.values[0]).area / 10000
            for a, b in zip(double.geometry.values, double.index_right)
        ]

        # Step 5: Sort by intersection area and keep the row with the largest intersection for each 'id'
        doublesorted = double.sort_values(by='intersection').groupby('id').last().reset_index()

        # Step 6: Merge the cleaned double-assigned polygons back into the main DataFrame
        allyears_regions = pd.concat([allyears_landkreise, doublesorted])

        return allyears_regions
    else:
        print("No duplicates found. Returning the original DataFrame.")
        return allyears_landkreise


def load_data(loadExistingData=False):
    base_dir = data_main_path+"/Niedersachsen"
    years = range(2012, 2024)
    specific_file_names = [
        "Schlaege_mitNutzung_2012.shp", "Schlaege_mitNutzung_2013.shp",
        "Schlaege_mitNutzung_2014.shp", "Schlaege_mitNutzung_2015.shp",
        "schlaege_2016.shp", "schlaege_2017.shp", "schlaege_2018.shp",
        "schlaege_2019.shp", "schlaege_2020.shp", "ud_21_s.shp",
        "Schlaege_2022_ende_ant.shp", "UD_23_S_AKT_ANT.shp"
    ]
    intpath = data_main_path+'/interim'

    #if the directory does not yet exist, create it
    os.makedirs(intpath, exist_ok=True)
    
    existing_file_path = os.path.join(intpath, 'gld_base.pkl')
    
    if loadExistingData and os.path.isfile(existing_file_path):
        gld = pickle.load(open(existing_file_path, 'rb'))
        logging.info(f"Loaded existing data from {existing_file_path}")
        return gld
    else:
        logging.info(f"Proceeding with loading and processing new data.")
        # Load and preprocess data
        data = load_geodata(base_dir, years, specific_file_names)
        data = harmonize_columns(data)
        data = first_index_reset(data, years)

        # Load land boundary and perform spatial join
        admin_files_dir = os.path.join(base_dir, "verwaltungseinheiten")
        land = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landesflaeche.shp"))
        land = land.to_crs(epsg=25832)
        data = spatial_join_with_land(data, years, land)
        data = first_duplicates_check(data, years) # also drops index_right, id and LAND from land join
        
        # Combine all years into a single dataframe
        all_years = pd.concat(list(data.values()), ignore_index=True)
        logging.info(f"Combined data from {years[0]} to {years[-1]}.")
        all_years = all_years.info()
        
        # Handle missing values
        missing_values_count = all_years.isnull().any(axis=1).sum()
        if missing_values_count > 0:
            logging.info(f"Found {missing_values_count} missing values.")
            if missing_values_count < 0.01 * len(all_years):
                all_years = all_years.dropna()
            else:
                all_years = all_years.fillna(all_years.mean())
    
        # Calculate geometric measures
        all_years = calculate_geometric_measures(all_years)
        logging.info("Calculated geometric measures.")
                
        
        # Load gridregion and perform spatial join
        all_years = all_years.reset_index().rename(columns={'index': 'id'})
        grid_landkreise = gridregionjoin.join_gridregion(loadExistingData = True)
        allyears_landkreise = spatial_join_with_gridregion(all_years, grid_landkreise)
        
        # Handle duplicates caused by sjoin with grid_landkreise
        allyears_regions = handle_grid_duplicates(allyears_landkreise, grid_landkreise)

        # Only keep needed columns
        gld = allyears_regions.drop(columns=['id', 'index_right', 'intersection'])
        gld.reset_index(drop=True, inplace=True)
        gld.info()        
        
        
        # Save the combined data to pickle
        gld_pickle_path = os.path.join(intpath, f"gld_base.pkl")
        gld.to_pickle(gld_pickle_path)
        logging.info(f"Saved processed data to {gld_pickle_path}")
        
        return gld

    

#
if __name__ == '__main__':
    
    loadExistingData = True
    gld = load_data(loadExistingData)


# %%
