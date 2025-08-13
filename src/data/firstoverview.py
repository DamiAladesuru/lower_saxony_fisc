#%%
import os
import zipfile
import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path

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


# Initialize logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Logging works!")


'''Run the codes in this script to get a first overview of the data before 
    running the full dataload workflow which takes about 13 hrs.'''

# %%
base_dir = data_main_path+"/Niedersachsen"
years = range(2012, 2024)
specific_file_names = [
    "Schlaege_mitNutzung_2012.shp", "Schlaege_mitNutzung_2013.shp",
    "Schlaege_mitNutzung_2014.shp", "Schlaege_mitNutzung_2015.shp",
    "schlaege_2016.shp", "schlaege_2017.shp", "schlaege_2018.shp",
    "schlaege_2019.shp", "schlaege_2020.shp", "ud_21_s.shp",
    "Schlaege_2022_ende_ant.shp", "UD_23_S_AKT_ANT.shp"
]

def load_geodata(base_dir, years, specific_file_names):
    data = {}
    for year in years:
        zip_file_path = os.path.join(base_dir, "Needed", f"schlaege_{year}.zip")
        print("Trying to open:", zip_file_path)
        print("Exists?", os.path.exists(zip_file_path))
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            for specific_file_name in specific_file_names:
                if specific_file_name in file_names:
                    data[year] = gpd.read_file(f"/vsizip/{zip_file_path}/{specific_file_name}")
                else:
                    logging.warning(f"File {specific_file_name} does not exist in {zip_file_path}.")
    return data
data = load_geodata(base_dir, years, specific_file_names)
#%%
data
#---- Basic checks ----#
# %%
# Check the CRS of each GeoDataFrame in the data dictionary
for key, df in data.items():
    print(f"--- {key} ---")
    print(df.crs)
    
# %%
# See the columns and dtypes of dfs in the data dictionary
for key, df in data.items():
    print(f"--- {key} ---")
    print(df.info())

# %%    
# Get 3-row head for each key in a data dictionary
for key, df in data.items():
    print(f"--- {key} ---")
    print(df.head(3))
        
#----------------------#
    
# %% harmonize data ----
'''rename columns and align data types'''
def harmonize_columns(data):
    old_year_names = ['jahr', 'antjahr', 'antragsjah']
    new_year_name = 'year'
    old_kulturcode_names = ['kc_gem', 'kc_festg', 'kc', 'nc_festg', 'kulturcode']
    new_kulturcode_name = 'kulturcode'
    old_area_names = ['shape_area', 'flaeche', 'akt_fl', 'aktuellefl']
    new_area_name = 'area'

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

    return data

data2 = harmonize_columns(data)
#%%
data2
#---- Check total area of fields ----#
# %%
for year in sorted(data2):
    if 'area' in data2[year].columns:
        print(f"{year}: Total area (ha) = {data2[year]['area'].sum():,.2f}")
        
''' 2015 has no values in area column, so we need to recalculate it from geometry
    and seems like area was computed in m² for some years but ha for other,
    so we also need to harmonize the units        
'''
# %%
def calc_area(data):

    for year in data:
        df = data[year]
        print(f"Year: {year}, number of rows: {len(df)}")
        if df['area'].sum() == 0:
            # If area column sums to zero, recalculate from geometry
            df['area'] = df.geometry.area
            logging.info(f"{year}: Calculated area from geometry because area column was missing or zero sum.")
calc_area(data2)

def harmonize_area_units(data): 
    m2_years = [2012, 2013, 2014, 2015]

    for year in data:
        df = data[year]

        if year in m2_years:
            df['area'] = df['area'] / 10_000
            print(f"{year}: Converted area from m² to ha, first few values:\n{df['area'].head()}")
        else:
            print(f"{year}: Area already in ha, no conversion needed.")

    return data

data3 = harmonize_area_units(data2)

'''
we can plot these values to see how the area changes over the years

1. For each DataFrame (data[year]), sum the area column.
2. Collect these sums in a list or Series.
3. Plot sum of area vs. year as a line plot.'''
# %%
import matplotlib.pyplot as plt

def plot_total_area(data):
    area_sums = {}
    for year in sorted(data):
        df = data[year]
        if 'area' in df.columns:
            area_sums[year] = df['area'].sum()
        else:
            logging.warning(f"{year}: No 'area' column found.")

    plt.figure(figsize=(10, 6))
    plt.plot(area_sums.keys(), area_sums.values(), marker='o')
    plt.title("Total Area by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Area")
    plt.grid(True)
    plt.show()
plot_total_area(data3)

'''' we can also simply compute area for all years freshly and plot it'''
# %%
def compute_area_and_plot(data):
    area_sums = {}

    for year in sorted(data):
        df = data[year]
        # Compute area from geometry and store in new column
        df['area_comp'] = df.area
        
        # Convert from m² to ha (divide by 10,000)
        df['area_comp'] = df['area_comp'] / 10_000
        
        # Sum and store total area_comp per year
        total_area = df['area_comp'].sum()
        area_sums[year] = total_area
        
        print(f"{year}: Computed and converted area_comp, total = {total_area:,.2f} ha")

    # Plot total area_comp by year
    plt.figure(figsize=(10, 6))
    plt.plot(list(area_sums.keys()), list(area_sums.values()), marker='o')
    plt.title("Total Computed Area (ha) by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Area (ha)")
    plt.grid(True)
    plt.show()

    return data  # returning updated data with 'area_comp' column

data_with_area_comp = compute_area_and_plot(data2)
#----------------------#

''' Okay, this is what the total area of data looks like. You can run more checks
    such as filtering data for fields within the Niedersachsen boundary using the
    next functions or yyou can directly proceed with spatial joins and
    other processing by running the entire dataload script.
    
    PS: The filtering spatial join below takes about 3:30hrs to run
'''

#---- Use landesflaeche to filter data and then check total area ----#
# %%
def first_index_reset(data, years):
    for year in years:
        data[year] = data[year].reset_index().rename(columns={'index': 'id'})
        logging.info(f"{year}: Index has been reset.")
    return data
data3 = first_index_reset(data3, years)

# %%
def spatial_join_with_land(data, years, land):
    for year in years:
        data[year] = gpd.sjoin(data[year], land, how='inner', predicate='intersects')
        logging.info(f"{year}: Spatially joined with land boundary.")
    return data
# Load land boundary and perform spatial join
admin_files_dir = os.path.join(base_dir, "verwaltungseinheiten")
land = gpd.read_file(os.path.join(admin_files_dir, "NDS_Landesflaeche.shp"))
land = land.to_crs(epsg=25832)
data4 = spatial_join_with_land(data3, years, land)

# %%     
def first_duplicates_check(data, years):
    for year in years:
        logging.info(f"{year}: Checked for duplicates after joining land. {data[year][['year', 'id']].duplicated().sum()} duplicates found.")
        data[year].drop(columns=['id', 'index_right', 'LAND'], inplace=True)
    return data    
data4 = first_duplicates_check(data4, years) # also drops index_right, id and LAND from land join

# %%
plot_total_area(data4)

#----------------------------------------------------------------------#

