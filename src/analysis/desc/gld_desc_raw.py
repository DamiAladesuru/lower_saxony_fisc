# %%
import pandas as pd
import os

# Set up the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # or two levels up if needed
print(project_root)

os.chdir(project_root)
print("Current working dir:", os.getcwd())


from src.data import dataload as dl
from src.data import eca_new as eca

''' this script allows to load the raw field data created with dataload.py,
to join kulturart information and to calculate yearly averages and differences
directly without grids. It gives you a first overview of what the trend of 
FiSC looks like over years.'''

# %% load data, add kulturcode information, drop area_ <100m2
def adjust_gld(file='data/interim/gld_wtkc.pkl'):
    # Check if the file already exists
    if os.path.exists(file):
        # Load data from the file if it exists
        gld = pd.read_pickle(file)
        print("Loaded gld data from existing file.")
    else:
        # Load base data
        gld = dl.load_data(loadExistingData=True)
        # Add additional columns to the data
        kulturcode_mastermap = eca.process_kulturcode()
        gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
        # Drop unnecessary columns
        gld = gld.drop(columns='sourceyear')
        # Save the processed data to a file
        gld.to_pickle(file)
        print("Processed and saved new data.")
    
    # Apply threshold of minimum 100m2 fields
    gld = gld[~(gld['area_m2'] < 100)]
    print(f"removed rows with area_m2 less than 100")
    
    return gld


# %% calculate yearly averages
def compute_year_average(gld):
        # 2. Group by 'year' and calculate descriptives
        gld_yearly_desc = gld.groupby('year').agg(
            fields_total=('geometry', 'count'),
                              
            kc_unique=('kulturcode', 'nunique'),                              
            group_unique=('Gruppe', 'nunique'),

            area_sum=('area_ha', 'sum'),
            area_mean=('area_ha', 'mean'),
            area_median = ('area_ha', 'median'),
            area_sd = ('area_ha', 'std'),
            
            peri_sum=('peri_m', 'sum'),
            peri_mean=('peri_m', 'mean'),
            peri_median = ('peri_m', 'median'),
            peri_sd = ('peri_m', 'std'),

            meanPAR=('shape', 'mean'),
            medianPAR=('shape', 'median'),
            par_sd=('shape', 'std') 

        ).reset_index()
            
        return gld_yearly_desc

# %% calculate yearly differences
#yearly gridcell differences and differences from first year
def calculate_yearlydiff(gld_yearly_desc): #yearly differences
    # Create a copy of the original dictionary to avoid altering the original data
    cop = gld_yearly_desc.copy()
    
    # Ensure the data is sorted by 'year'
    cop.sort_values(by='year', inplace=True)
    numeric_columns = cop.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_diff'] = cop[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_percdiff'] = (cop[col].diff() / cop[col].shift(1)).fillna(0) * 100
    
    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    cop = pd.concat([cop, new_columns_df], axis=1)

    return cop    


# %% calculate differences from first year
def calculate_diff_fromy1(gld_yearly_desc): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    cop = gld_yearly_desc.copy()

    # Ensure the data is sorted by 'year'
    cop.sort_values(by='year', inplace=True)
    
    # Get a list of all numeric columns except 'year'
    numeric_columns = [col for col in cop.columns if col != 'year']
    
    # Create a dictionary to store the new columns
    new_columns = {}

    # Get the first year's row in the dataFrame
    y1_df = cop[cop['year'] == cop['year'].min()] 
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Perform a cross join by adding a temporary key to both dataframes
    cop['key'] = 1
    y1_df['key'] = 1

    # Merge on the temporary key to create the cross join
    cop = cop.merge(y1_df.drop(columns=['year']), on='key', how='left').drop(columns='key')
    
    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = cop[col] - cop[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((cop[col] - cop[f'{col}_y1']) / cop[f'{col}_y1'])*100

    # Drop the temporary first year columns
    cop.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    cop_y1 = pd.concat([cop, new_columns_df], axis=1)

    return cop_y1


# %% combine the differences dataframes
def combine_diffgriddfs(cop, cop_y1):
    # Ensure the merge is based on 'year'
    # Select columns from cop_y1 that are not in cop (excluding 'year')
    columns_to_add = [col for col in cop_y1.columns if col not in cop.columns or col in 'year']

    # Merge the DataFrames on 'CELLCODE' and 'year', keeping the existing columns in cop
    combined_griddf = pd.merge(cop, cop_y1[columns_to_add], on='year', how='left')
    
    return combined_griddf


# %% process descriptives over time
def gld_overyears():
    gld = adjust_gld()

    gydesc = compute_year_average(gld)
    gydesc['fields_ha'] = gydesc['fields_total'] / gydesc['area_sum']
    cop = calculate_yearlydiff(gydesc)
    cop_y1 = calculate_diff_fromy1(gydesc)
    gydesc_new = combine_diffgriddfs(cop, cop_y1)
    
    
    return gld, gydesc_new


# %% process descriptives over time with filter
def gld_overyears_filt(column, xgroups):
    gld = adjust_gld()
    
    # Filter rows based on the specified column
    gld = gld[~gld[column].isin(xgroups)]
    
    gydesc = compute_year_average(gld)
    gydesc['fields_ha'] = gydesc['fields_total'] / gydesc['area_sum']
    cop = calculate_yearlydiff(gydesc)
    cop_y1 = calculate_diff_fromy1(gydesc)
    gydesc_filt = combine_diffgriddfs(cop, cop_y1)
    
    return gld, gydesc_filt

# %%
