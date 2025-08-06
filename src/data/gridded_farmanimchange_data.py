# %% Import necessary libraries
import zipfile
import pandas as pd
import os
import xml.etree.ElementTree as ET
import geopandas as gpd
import numpy as np
from pathlib import Path

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

###########################################
# %% load needed existing data
# gridgdf_cl with naturraum, klima and eastwest columns
gridgdf_cluster = pd.read_pickle(data_main_path+'/interim/gridgdf/gridgdf_naturraum_klima_east_elev.pkl')

#############################################
# %%
# Step 1: Load animal data xml
# Unzip the folder containing animal data
zip_path = data_main_path+'/raw/animaldata.zip' 
extract_path = data_main_path+'/raw/animaldata'

# Create the directory if it doesn't exist
if not os.path.exists(extract_path):
    os.makedirs(extract_path)   

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Find the XML file (assuming it's .xml)
for filename in os.listdir(extract_path):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(extract_path, filename)
        break

# Load the XML
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Register the namespace used in the file (required for correct tag searches)
namespaces = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

# Find the table rows
rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', namespaces)

data = []

# Iterate through each row and extract cell data
for row in rows:
    row_data = []
    for cell in row.findall('ss:Cell', namespaces):
        data_element = cell.find('ss:Data', namespaces)
        if data_element is not None:
            row_data.append(data_element.text)
        else:
            row_data.append('')
    data.append(row_data)

# Remove any irrelevant header/footer rows based on content
# This will depend on your data; below filters out top metadata and keeps actual data
# Adjust the slicing index as needed
data = [r for r in data if any(r)]  # Remove completely empty rows
data = data[5:]  # Skip metadata/header rows if needed

# Assume first row after slicing is header
header = data[18]
rows = data[0:]

# Create DataFrame
df = pd.DataFrame(rows, columns=header)

print(df.head())

# %% last column is missing column name because of shifted rows
# let's fix that by moving current row idx 3 and 4 one column to the right
for idx in [3, 4]:
    df.iloc[idx, 1:] = df.iloc[idx, :-1].values

'''
based on data structure on data web page
we know that the first 8 columns are for 'viehbetriebe'
and the rest are 'viehanzahl'
given column name duplication which can affect data processing,
let's split the data for easier handling 
'''
viehbetriebe = df.iloc[:, :-8] # All columns except the last 8
viehanzahl = df.iloc[:, [0] + list(range(-8, 0))] # First column and last 8 columns

# let fourth row conteht be used as header and 
# then drop first 5 rows for viehbetriebe and viehanzahl

viehbetriebe.columns = viehbetriebe.iloc[3]
viehbetriebe = viehbetriebe.iloc[5:, :]
# reset index
viehbetriebe.reset_index(drop=True, inplace=True)

viehanzahl.columns = viehanzahl.iloc[3]
viehanzahl = viehanzahl.iloc[5:, :]
# reset index
viehanzahl.reset_index(drop=True, inplace=True)

# we need both betrieb and anzahl to have the column containing kreis and year
viehanzahl['kreis'] = viehanzahl.iloc[:, 0]  # copy first column of viehanzahl and name it kreis
viehanzahl = viehanzahl.iloc[:, 1:]  # Drop the first column from viehanzahl

viehbetriebe['kreis'] = viehbetriebe.iloc[:, 0]  # Copy the first column from viehbetriebe
viehbetriebe = viehbetriebe.iloc[:, 1:]  # Drop the first column from viehbetriebe

# Add suffix 'betrieb' to columns in viehbetriebe
viehbetriebe.columns = viehbetriebe.columns.map(lambda col: f"{col}_betrieb")
# Add suffix 'anzahl' to columns in viehanzahl
viehanzahl.columns = viehanzahl.columns.map(lambda col: f"{col}_anzahl")

# Verify the changes
print("Viehbetriebe columns:", viehbetriebe.columns)
print("Viehanzahl columns:", viehanzahl.columns)

def propagate_and_reorder(df):
    """
    Propagate 'year' and 'LANDKREIS' columns from kreis column 
    and move these two columns to be the first two columns.
    """
    # Create new columns with default NaN
    df['year'] = pd.NA
    df['LANDKREIS'] = pd.NA

    # Propagate 'year' and 'LANDKREIS' values
    for i in range(len(df) - 3):
        val = df.iloc[i, 8]
        try:
            float(val)  # Check if it's numeric
        except (ValueError, TypeError):
            # If not numeric, set values in year and LANDKREIS for the next 3 rows
            for j in range(1, 4):
                df.loc[i + j, 'year'] = df.iloc[i + j, 8]
                df.loc[i + j, 'LANDKREIS'] = val  # Propagate the non-numeric label

    # Move 'year' and 'LANDKREIS' to be the first two columns
    columns_order = ['year', 'LANDKREIS'] + [col for col in df.columns if col not in ['year', 'LANDKREIS']]
    df = df[columns_order]

    return df

# Apply the function to both viehbetriebe and viehanzahl
viehbetriebe = propagate_and_reorder(viehbetriebe)
viehanzahl = propagate_and_reorder(viehanzahl)

# Verify the changes
print("Viehbetriebe columns:", viehbetriebe.columns)
print("Viehanzahl columns:", viehanzahl.columns)

# drop kreis column
viehanzahl = viehanzahl.iloc[:, :-1]
viehbetriebe = viehbetriebe.iloc[:, :-1]

# drop rows with missing values in 'year' and 'LANDKREIS'
viehbetriebe.dropna(subset=['year', 'LANDKREIS'], inplace=True)
viehanzahl.dropna(subset=['year', 'LANDKREIS'], inplace=True)

# merge the two DataFrames on 'year' and 'LANDKREIS'
viehbestand = pd.merge(viehbetriebe, viehanzahl, on=['year', 'LANDKREIS'], how='outer')

### correct landkreis issues
# drop row where 'LANDKREIS' contains 241001
viehbestand = viehbestand[~viehbestand['LANDKREIS'].str.contains('241001', na=False)]

# standardize LANDKREIS names
def standardize_landkreis(value):
    try:
        if pd.isna(value):
            return None  # or 'UNKNOWN'
        parts = value.strip().split()
        if len(parts) < 2:
            return None  # flag malformed entries
        name = ' '.join(parts[1:])
        if ',Stadt' in name:
            name = name.replace(',Stadt', '')
            return f'Stadt {name} (kreisfrei)'
        else:
            return name.strip()
    except Exception:
        return None  # fallback in case of unexpected error

# Apply function
viehbestand['LANDKREIS_STANDARDIZED'] = viehbestand['LANDKREIS'].apply(standardize_landkreis)

# Flag problematic entries
viehbestand['FLAG_FOR_REVIEW'] = viehbestand['LANDKREIS_STANDARDIZED'].isna()

# correct region Hannover and drop oldenburg
viehbestand['LANDKREIS_STANDARDIZED'] = viehbestand['LANDKREIS_STANDARDIZED'].replace("Hannover,Region", "Region Hannover")
# Drop rows where 'LANDKREIS_STANDARDIZED' is 'Stadt Oldenburg(Oldb) (kreisfrei)'
viehbestand = viehbestand[~viehbestand['LANDKREIS_STANDARDIZED'].str.contains('Stadt Oldenburg\(Oldb\) \(kreisfrei\)', na=False)]

# Get unique LANDKREIS values from both DataFrames
vieh_landkreise = set(viehbestand['LANDKREIS_STANDARDIZED'].unique())
grid_landkreise = set(gridgdf_cluster["LANDKREIS"].unique())

# Check if they are the same
if vieh_landkreise == grid_landkreise:
    print("✅ LANDKREIS values in viehdf and griddf match exactly!")
else:
    print("❌ There are differences in LANDKREIS values between the two DataFrames.")
    
    # Find mismatched values
    only_in_vieh = vieh_landkreise - grid_landkreise
    only_in_grid = grid_landkreise - vieh_landkreise
    
    print(f"⚠️ Present in vieh but missing in grid: {only_in_vieh}")
    print(f"⚠️ Present in grid but missing in vieh: {only_in_grid}")

# drop column 'FLAG_FOR_REVIEW'
viehbestand.drop(columns=['FLAG_FOR_REVIEW', 'LANDKREIS'], inplace=True)
# rename 'LANDKREIS_STANDARDIZED' to 'LANDKREIS'
viehbestand.rename(columns={'LANDKREIS_STANDARDIZED': 'LANDKREIS'}, inplace=True)

print(viehbestand['LANDKREIS'].nunique())

####################################################
viehbestand.isnull().sum()
# Renaming columns
viehbestand.rename(columns={
    'Insgesamt\nGVE_betrieb': 'InsgesamtGVE_betrieb',
    'Insgesamt\nGVE_anzahl': 'InsgesamtGVE_anzahl',
    'sonst.\nGeflügel_betrieb': 'sonstGefluegel_betrieb',
    'sonst.\nGeflügel_anzahl': 'sonstGefluegel_anzahl',
}, inplace=True)

# reshape viehbestand DataFrame using pd.melt and some string manipulation 
'''columns: 
LANDKREIS
year
Animal
Animal_count
Farm_count

where column Animal contains all animals for each year and LANDKREIS i.e., Rinder, Schweine, Schafe, Ziegen, Einhufer, Hühner, sonstGefluegel and Ingesamt
column Animal_count takes value for each animal count i.e., from column with suffix anzahl for each year and LANDKREIS
column Farm_count takes value for each farm count i.e., from column with suffix betrieb for each year and LANDKREIS
'''
# Assuming viehbestand has a multi-index or 'year' as a column
if 'year' not in viehbestand.columns:
    viehbestand = viehbestand.reset_index()  # ensure year is a column if it's an index

# Melt the dataframe to long format
melted = viehbestand.melt(id_vars=['year', 'LANDKREIS'], var_name='Measure', value_name='Value')

# Extract the animal name and the type of measure (count or farm)
melted['Animal'] = melted['Measure'].str.replace('_anzahl', '', regex=False)
melted['Animal'] = melted['Animal'].str.replace('_betrieb', '', regex=False)
melted['MeasureType'] = melted['Measure'].apply(lambda x: 'Animal_count' if 'anzahl' in x else 'Farm_count')

# Pivot so each animal/year/landkreis has both count and farm count
animaldat = melted.pivot_table(
    index=['year', 'LANDKREIS', 'Animal'],
    columns='MeasureType',
    values='Value',
    aggfunc='first'  # In case of duplicates
).reset_index()

# reorder columns
animaldat = animaldat[['LANDKREIS', 'year', 'Animal', 'Animal_count', 'Farm_count']]

# fill each missing Animal_count cell with the value from the previous year
# for the same LANDKREIS and Animal.
# 1. Replace '-' with NaN in both columns
animaldat['Animal_count'] = animaldat['Animal_count'].replace('-', np.nan)
animaldat['Farm_count'] = animaldat['Farm_count'].replace('-', np.nan)

# 2. Convert both to numeric
animaldat['Animal_count'] = pd.to_numeric(animaldat['Animal_count'], errors='coerce')
animaldat['Farm_count'] = pd.to_numeric(animaldat['Farm_count'], errors='coerce')

# 3. Sort values for proper ffill and then fill missing values (optional: with 0 first, then ffill)
animaldat = animaldat.sort_values(['LANDKREIS', 'Animal', 'year'])
 
animaldat['Animal_count'] = animaldat.groupby(['LANDKREIS', 'Animal'])['Animal_count'].ffill().fillna(0)
animaldat['Farm_count'] = animaldat.groupby(['LANDKREIS', 'Animal'])['Farm_count'].ffill().fillna(0)

animaldat.info()

# from animaldat_ydiff_all, keep only rows with InsgesamtGVE
animaldatGVE = animaldat[animaldat['Animal'] == 'InsgesamtGVE']
# drop the 'Animal' column
animaldatGVE = animaldatGVE.drop(columns=['Animal'])
# rename Farm_count to animalfarms
animaldatGVE = animaldatGVE.rename(columns={'Farm_count': 'animalfarms'})

#############################################################
# import similarly structured farm data
#############################################################
# %%
# Unzip the folder containing animal data
farmzip_path = data_main_path+'/raw/farmdata.zip' 
farmextract_path = data_main_path+'data/raw/farmdata'

# Create the directory if it doesn't exist
if not os.path.exists(farmextract_path):
    os.makedirs(farmextract_path)   


with zipfile.ZipFile(farmzip_path, 'r') as zip_ref:
    zip_ref.extractall(farmextract_path)

# Find the XML file (assuming it's .xml)
for filename in os.listdir(farmextract_path):
    if filename.endswith('.xml'):
        xml_file_path = os.path.join(farmextract_path, filename)
        break

# Load the XML
tree = ET.parse(xml_file_path)
root = tree.getroot()

# Register the namespace used in the file (required for correct tag searches)
namespaces = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}

# Find the table rows
rows = root.findall('.//ss:Worksheet/ss:Table/ss:Row', namespaces)

data = []

# Iterate through each row and extract cell data
for row in rows:
    row_data = []
    for cell in row.findall('ss:Cell', namespaces):
        data_element = cell.find('ss:Data', namespaces)
        if data_element is not None:
            row_data.append(data_element.text)
        else:
            row_data.append('')
    data.append(row_data)

# Remove any irrelevant header/footer rows based on content
# This will depend on your data; below filters out top metadata and keeps actual data
# Adjust the slicing index as needed
data = [r for r in data if any(r)]  # Remove completely empty rows
data = data[8:]  # Skip metadata/header rows if needed

# Assume first row after slicing is header
header = data[18]
rows = data[0:]

# Create DataFrame
df = pd.DataFrame(rows, columns=header)

print(df.head())

# %% last column is missing column name because of shifted rows
# let's fix that by moving current idx 1 and 2 one column to the right
for idx in [1, 2]:
    df.iloc[idx, 1:] = df.iloc[idx, :-1].values

# %%
# Set values in row index 0
df.iloc[0, [1, 2]] = 2010
df.iloc[0, [3, 4]] = 2016
df.iloc[0, [5, 6]] = 2020

# %%
df = df.drop(index=2)

# %%
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.dropna(how='all')

# %%
df = df.reset_index(drop=True)

# %%
#copy df to a new DataFrame and drop last row containinng footer information
df_copy = df.copy()
df_copy = df_copy.iloc[:-1, :] 

# project LANDKREIS value to new column for each row
# Starting from row index 2
start_idx = 2
block_size = 20  # 1 header + 19 classes

LANDKREIS_list = []

# For rows before start_idx (0 and 1), fill None:
LANDKREIS_list.extend([None] * start_idx)

i = start_idx
while i < len(df_copy):
    LANDKREIS = df_copy.iloc[i, 0]

    rows_left = len(df_copy) - i
    num_class_rows = min(rows_left - 1, 19)  # always max 19 class rows after header

    # For the header row, add None:
    LANDKREIS_list.append(None)

    # For the 19 class rows, add LANDKREIS name:
    LANDKREIS_list.extend([LANDKREIS] * num_class_rows)

    i += 1 + num_class_rows

# Safety pad if needed:
while len(LANDKREIS_list) < len(df_copy):
    LANDKREIS_list.append(None)

df_copy['LANDKREIS'] = LANDKREIS_list

# Keep the first two rows and only rows where LANDKREIS is not None
first_two = df_copy.iloc[:2]

rest = df_copy.iloc[2:]
rest_filtered = rest[rest['LANDKREIS'].notna()]

# Concatenate back
df_copy_filtered = pd.concat([first_two, rest_filtered]).reset_index(drop=True)

df_copy_filtered.rename(columns={df_copy_filtered.columns[0]: 'Klasse'}, inplace=True)

# %% split the DataFrame into three separate DataFrames for each year
df_2010 = df_copy_filtered.iloc[:, [0, 1, 2, 7]].copy()
df_2016 = df_copy_filtered.iloc[:, [0, 3, 4, 7]].copy()
df_2020 = df_copy_filtered.iloc[:, [0, 5, 6, 7]].copy()

# rename Anzahl and LF columns
df_2010.rename(columns={df_2010.columns[1]: 'Anzahl', df_2010.columns[2]: 'LF_ha'}, inplace=True)
df_2016.rename(columns={df_2016.columns[1]: 'Anzahl', df_2016.columns[2]: 'LF_ha'}, inplace=True)
df_2020.rename(columns={df_2020.columns[1]: 'Anzahl', df_2020.columns[2]: 'LF_ha'}, inplace=True)

# create year column and insert it at index 1
df_2010.insert(1, 'year', 2010)
df_2016.insert(1, 'year', 2016)
df_2020.insert(1, 'year', 2020)

# Reorder columns: Klasse, LANDKREIS, rest
cols = ['LANDKREIS', 'Klasse'] + [col for col in df_2010.columns if col not in ['LANDKREIS', 'Klasse']]
df_2010 = df_2010[cols]
df_2016 = df_2016[cols]
df_2020 = df_2020[cols]

# keeps all rows from index 2 onward (i.e., drops rows 0 and 1)
df_2010 = df_2010.iloc[2:].reset_index(drop=True)
df_2016 = df_2016.iloc[2:].reset_index(drop=True)
df_2020 = df_2020.iloc[2:].reset_index(drop=True)

# %% Combine the three DataFrames into one
df_farm = pd.concat([df_2010, df_2016, df_2020], ignore_index=True)

# delete helper dataframes
del df_2010, df_2016, df_2020, df_copy_filtered, df_copy

# %% keep only rows where Klasse is Ingesamt
df_farmI = df_farm[df_farm['Klasse'] == 'Insgesamt'].reset_index(drop=True)

# %% check for Anzahl and LF_ha column cells that are not numeric
# if all rows in a column are numeric, we can convert them to float
# if not, we will create a new column with boolean values indicating numeric status
def is_numeric_column(column):
    """Check if all values in the column are numeric."""
    return pd.to_numeric(column, errors='coerce').notna().all()
def convert_to_numeric(column):
    """Convert column to numeric, coercing errors to NaN."""
    return pd.to_numeric(column, errors='coerce')
def check_and_convert_numeric(df, column_name):
    """Check if a column is numeric and convert it if so."""
    if is_numeric_column(df[column_name]):
        df[column_name] = convert_to_numeric(df[column_name])
    else:
        df[f"{column_name}_is_numeric"] = df[column_name].apply(lambda x: isinstance(x, (int, float)))
    return df
df_farmI = check_and_convert_numeric(df_farmI, 'Anzahl')
df_farmI = check_and_convert_numeric(df_farmI, 'LF_ha')

df_farmI.info()

#compute LF_mean
df_farmI['LF_mean'] = df_farmI['LF_ha'] / df_farmI['Anzahl']

# %% sort by LANDKREIS and year
df_farmI.sort_values(by=['LANDKREIS', 'year'], inplace=True)

# %% drop row where 'LANDKREIS' contains 241001
df_farmI = df_farmI[~df_farmI['LANDKREIS'].str.contains('241001', na=False)]

# %%
df_farmI['LANDKREIS'] = df_farmI['LANDKREIS'].apply(standardize_landkreis)

# %% correct region Hannover and drop oldenburg
df_farmI['LANDKREIS'] = df_farmI['LANDKREIS'].replace("Hannover,Region", "Region Hannover")
# Drop rows where 'LANDKREIS' is 'Stadt Oldenburg(Oldb) (kreisfrei)'
df_farmI = df_farmI[~df_farmI['LANDKREIS'].str.contains('Stadt Oldenburg\(Oldb\) \(kreisfrei\)', na=False)]

# %% drop Klasse column
farmdata_ge = df_farmI.drop(columns=['Klasse'])
 

##################################################################
# Work with farmdata_ge and animaldatGVE
# ################################################################
# %%
# Get unique LANDKREIS values from both DataFrames
farm_landkreise = set(farmdata_ge["LANDKREIS"].unique())
animal_landkreise = set(animaldatGVE["LANDKREIS"].unique())

# Check if they are the same
if farm_landkreise == animal_landkreise:
    print("✅ LANDKREIS values in farmdata_ge and animaldatGVE match exactly!")
else:
    print("❌ There are differences in LANDKREIS values between the two DataFrames.")
    
    # Find mismatched values
    only_in_farm = farm_landkreise - animal_landkreise
    only_in_animal = animal_landkreise - farm_landkreise
    
    print(f"⚠️ Present in farmdata_ge but missing in animaldatGVE: {only_in_farm}")
    print(f"⚠️ Present in animaldatGVE but missing in farmdata_ge: {only_in_animal}")

# %%
print(animaldatGVE['year'].unique())
print(farmdata_ge['year'].unique())

# %% ensure that year columns in both DataFrames are of the same type
animaldatGVE['year'] = animaldatGVE['year'].astype(int)
farmdata_ge['year'] = farmdata_ge['year'].astype(int)

# %%
print(animaldatGVE.info())
print(farmdata_ge.info())

# %%
farm_anim = farmdata_ge.merge(animaldatGVE, on=["LANDKREIS", "year"])
farm_anim.info()

# %% rename columns in farm_anim
farm_anim = farm_anim.rename(columns={
    'LF_ha': 'total_farm_area',
    'LF_mean': 'mean_farmsize',
    'Anzahl': 'farm_count',
    'Animal_count': 'animal_count'})

# %% create stocking density column
farm_anim['stocking_density'] = farm_anim['animal_count'] / farm_anim['total_farm_area']
print(farm_anim.info())
# save farm_anim to csv
farm_anim.to_csv(data_main_path+'/interim/farm_anim.csv', index=False)

#####################################################################
# Compute differences
#####################################################################
#%% Load the cleaned DataFrame from CSV
farm_anim = pd.read_csv(data_main_path+'/interim/farm_anim.csv')

# %%
farm_anim['farmc_ha'] = farm_anim['farm_count'] / farm_anim['total_farm_area']

# %%
farm_anim_ext = farm_anim.copy()

# reset index to avoid issues with groupby
farm_anim_ext.reset_index(drop=True, inplace=True)

#resort
farm_anim_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
numeric_columns = farm_anim_ext.select_dtypes(include='number').columns

# Create a dictionary to store the new columns
new_columns = {}


# Loop through each numeric column
for col in numeric_columns:
    # Calculate yearly absolute difference (from previous year)
    new_columns[f'{col}_yearly_diff'] = farm_anim_ext.groupby('LANDKREIS')[col].diff().fillna(0)
    
    # Calculate yearly percentage difference (from previous year)
    new_columns[f'{col}_yearly_percdiff'] = (
        farm_anim_ext.groupby('LANDKREIS')[col].diff() / farm_anim_ext.groupby('LANDKREIS')[col].shift(1)
    ).fillna(0) * 100

    # Difference from first year value within each group
    new_columns[f'{col}_diff_to_y1'] = (
        farm_anim_ext[col] - farm_anim_ext.groupby('LANDKREIS')[col].transform('first')
    )

    # Percentage difference from first year value within each group
    new_columns[f'{col}_percdiff_to_y1'] = (
        (farm_anim_ext[col] - farm_anim_ext.groupby('LANDKREIS')[col].transform('first')) / 
        farm_anim_ext.groupby('LANDKREIS')[col].transform('first')
    ).fillna(0) * 100


# Concatenate the new columns to the original DataFrame all at once
new_columns_df = pd.DataFrame(new_columns)
farm_anim_ydiff = pd.concat([farm_anim_ext, new_columns_df], axis=1)
farm_anim_ydiff.head()


# %%
'''Since field data is from 2012 to 2023, for missing yearly data between 2010 and 2023,
let 2010 data be used for 2011 and 2012, 2016 data for 2013 to 2015, and 2020 data for 2017 to 2023'''
# Create a list to store all DataFrames
all_dfs = [farm_anim_ydiff]

# Define the years to copy data for
years_to_copy = {2010: range(2011, 2013), 2016: range(2013, 2016), 2020: list(range(2017, 2020)) + list(range(2021, 2024))}

# Loop through the years and create copies
for base_year, copy_years in years_to_copy.items():
    base_data = farm_anim_ydiff[farm_anim_ydiff['year'] == base_year].copy()
    for year in copy_years:
        df_copy = base_data.copy()
        df_copy['year'] = year
        all_dfs.append(df_copy)

# Concatenate all DataFrames
farm_anim_ydiff_all = pd.concat(all_dfs, ignore_index=True)

# Sort the DataFrame by year
farm_anim_ydiff_all = farm_anim_ydiff_all.sort_values(by='year').reset_index(drop=True)

# print unique years in the dataset
print(farm_anim_ydiff_all['year'].unique())
# %%
# drop rows for years 2010 and 2011 in df_farm since we now have 2012 which is a coherent baseyear with field data
farm_anim_ydiff_all = farm_anim_ydiff_all[farm_anim_ydiff_all['year'] > 2011]
print(farm_anim_ydiff_all['year'].unique())


# %%
columns_to_keep = [
    'LANDKREIS', 'CELLCODE', 'year', 'medfs_ha', 'medfs_ha_yearly_percdiff',
    'fields', 'fsha_sum', 'fields_ha', 'fields_ha_yearly_percdiff',
    'medfs_ha_percdiff_to_y1', 'fields_ha_percdiff_to_y1',
    'geometry', 'eastwest', 'main_Klima_EN',
    'Naturraum', 'main_crop', 'main_crop_group'
]

# Drop all other columns
df_grids = gridgdf_cluster[columns_to_keep]

# %%
grid_fanim = farm_anim_ydiff_all.merge(df_grids, on=["LANDKREIS", "year"])

grid_fanim.info()

# %% make grid_fanim geodataframes
grid_fanim = gpd.GeoDataFrame(grid_fanim, geometry='geometry')
grid_fanim.to_pickle(data_main_path+"/interim/gridgdf/grid_fanim.pkl")

# %%
