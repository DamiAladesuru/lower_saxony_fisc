# %%
from pathlib import Path
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

from src.data import dataload as dl

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root = os.getcwd()
data_main_path = open(project_root + "/datapath.txt").read()


# ================================
# Load the yearly gld data
def load_geodata():
    gld_paths = dl.load_data(loadExistingData=True)
    
    years = range(2012, 2025)
    data = {}  # Will store year → GeodataFrame of data

    for year in years:
        # Load the gld data for the current year
        gld_path = gld_paths[year]
        gld = gpd.read_parquet(gld_path)
        print(f"{year}: CRS of data: EPSG:{gld.crs.to_epsg()}")
    
    # save the gld data to a dictionary
        data[year] = gld

    return data
 
data = load_geodata()

#%%
# See the columns and dtypes of dfs in the data dictionary
for key, df in data.items():
    print(f"--- {key} ---")
    print(df.info())

# %%
#---- Check total area of fields ----#
for year in sorted(data):
    if 'area' in data[year].columns:
        print(f"{year}: Total area = {data[year]['area'].sum():,.2f}")

# --- Print total number of observations in each year
for year in sorted(data):
    print(f"{year}: Total observations = {len(data[year])}")
    
    
# %%
'''
we can plot these values to see how the area changes over the years

1. For each DataFrame (data[year]), sum the area column.
2. Collect these sums in a list or Series.
3. Plot sum of area vs. year as a line plot.'''
def plot_total_area(data):
    area_sums = {}
    for year in sorted(data):
        df = data[year]
        if 'area_ha' in df.columns:
            area_sums[year] = df['area_ha'].sum()
        else:
            print(f"{year}: No 'area' column found.")

    plt.figure(figsize=(10, 6))
    plt.plot(area_sums.keys(), area_sums.values(), marker='o')
    plt.title("Total Area (Computed) by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Area")
    plt.grid(True)
    plt.show()
plot_total_area(data)

# %%
