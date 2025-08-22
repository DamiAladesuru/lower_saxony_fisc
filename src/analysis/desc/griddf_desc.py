# %%
import pandas as pd
import geopandas as gpd
import os, gc
import logging
import numpy as np
import pickle
# Silence the print statements in a function call
import contextlib
import io
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


from src.data import dataload as dl
from src.data import eca_new as eca

''' This script contains functions for:
    - modifying gld to include columns for basic additional metrics and kulturcode descriptions.
    - creating griddf (without removing any assumed outlier and for subsamples).
    - computing descriptive statistics for griddf.
The functions are called in the trend_of_fisc script
'''


# %% A.
def load_geodata():
    gld_paths = dl.load_data(loadExistingData=True)
    kulturcode_mastermap = eca.process_kulturcode(data_main_path, load_existing=True)
            
    years = range(2012, 2025)
    gld_allyears = {}  # Will store year → GeodataFrame of data

    for year in years:
        # Load the gld data for the current year
        gld_path = gld_paths[year]
        gld = gpd.read_parquet(gld_path)
        print(f"{year}: CRS of data: EPSG:{gld.crs.to_epsg()}")
            
        # merge gld on 'kulturcode' with kulturcode_mastermap on 'old_kulturcode'
        gld = gld.rename(columns={'kulturcode': 'old_kulturcode'})
        gld = gld.merge(kulturcode_mastermap, on='old_kulturcode', how='left')
        # drop 'int_kulturcode' and rename 'old_kulturcode' to 'kulturcode'
        gld = gld.drop(columns=['old_kulturcode','int_kulturcode'])
        gld = gld.rename(columns={'new_kulturcode': 'kulturcode'})
        
        # Apply threshold of minimum 100m2 fields
        gld = gld[~(gld['area_m2'] < 100)]
        print(f"updated {year} data")
    
    # save the gld data to a dictionary
        gld_allyears[year] = gld
        
    # Clean up memory
    del gld, kulturcode_mastermap
    gc.collect()

    return gld_allyears


def create_griddf_base(gld):
    """
    Create a griddf GeodataFrame with aggregated statistics that summarize field data.

    Parameters:
    gld (geodataFrame): Input geodataFrame with columns ['CELLCODE', 'year', 'LANDKREIS'] 
                        and additional numeric columns for aggregation.

    Returns:
    A  geoDataFrame with unique CELLCODE, year, and LANDKREIS rows,
                  enriched with aggregated fields.
    """
    
    required_columns = ['CELLCODE', 'year', 'LANDKREIS', 'geometry',\
        'Gruppe', 'area_m2', 'area_ha', 'peri_m', 'shape']
    missing = [col for col in required_columns if col not in gld.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    
    # Step 1: Create base griddf
    columns = ['CELLCODE', 'year', 'LANDKREIS']
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")

    # Helper function for aggregation
    def add_aggregated_column(griddf, gld, column, aggfunc, new_col):
        logging.info(f"Adding column '{new_col}' using '{aggfunc}' on '{column}'.")
        temp = gld.groupby(['CELLCODE', 'year'])[column].agg(aggfunc).reset_index()
        temp.columns = ['CELLCODE', 'year', new_col]
        return pd.merge(griddf, temp, on=['CELLCODE', 'year'], how='left')

    # Define aggregations
    aggregations = [
        {'column': 'geometry', 'aggfunc': 'count', 'new_col': 'fields'},
        {'column': 'Gruppe', 'aggfunc': 'nunique', 'new_col': 'group_count'},
        {'column': 'area_m2', 'aggfunc': 'sum', 'new_col': 'fsm2_sum'},
        {'column': 'area_ha', 'aggfunc': 'sum', 'new_col': 'fsha_sum'},
        {'column': 'peri_m', 'aggfunc': 'sum', 'new_col': 'peri_sum'},
        {'column': 'shape', 'aggfunc': 'sum', 'new_col': 'shape_sum'},
        {'column': 'area_ha', 'aggfunc': 'mean', 'new_col': 'mfs_ha'},
        {'column': 'peri_m', 'aggfunc': 'mean', 'new_col': 'mperi'},
        {'column': 'shape', 'aggfunc': 'mean', 'new_col': 'mshape'},
        {'column': 'area_ha', 'aggfunc': 'median', 'new_col': 'medfs_ha'},
        {'column': 'peri_m', 'aggfunc': 'median', 'new_col': 'medperi'},
        {'column': 'shape', 'aggfunc': 'median', 'new_col': 'medshape'},
    ]

    # Apply each aggregation
    for agg in aggregations:
        griddf = add_aggregated_column(griddf, gld, agg['column'], agg['aggfunc'], agg['new_col'])

    # Rate of fields per hectare of land per grid
    griddf['fields_ha'] = (griddf['fields'] / griddf['fsha_sum'])

    return griddf


# check for duplicates in the griddf
def check_duplicates(griddf):
    duplicates = griddf[griddf.duplicated(subset=['CELLCODE', 'year'], keep=False)]
    print(f"Number of duplicates in griddf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")
            
#yearly gridcell differences and differences from first year
def calculate_yearlydiff(griddf): #yearly gridcell differences
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()
    
    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = griddf_ext.groupby('CELLCODE')[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_yearly_percdiff'] = (griddf_ext.groupby('CELLCODE')[col].diff() / griddf_ext.groupby('CELLCODE')[col].shift(1)).fillna(0) * 100
    
    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_ext = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_ext    


# %%
def calculate_diff_fromy1(griddf): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()

    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Get the first occurrence of each unique CELLCODE in the griddf_ext DataFrame. 
    y1_df = griddf_ext.groupby('CELLCODE').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['CELLCODE'] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Merge the first year values back into the original DataFrame
    griddf_ext = pd.merge(griddf_ext, y1_df, on='CELLCODE', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = griddf_ext[col] - griddf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((griddf_ext[col] - griddf_ext[f'{col}_y1']) / griddf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    griddf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_exty1 = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_exty1


# %% 
def combine_griddfs(griddf_ext, griddf_exty1):
    # Ensure the merge is based on 'CELLCODE' and 'year'
    # Select columns from griddf_exty1 that are not in griddf_ext (excluding 'CELLCODE' and 'year')
    columns_to_add = [col for col in griddf_exty1.columns if col not in griddf_ext.columns or col in ['CELLCODE', 'year']]

    # Merge the DataFrames on 'CELLCODE' and 'year', keeping the existing columns in griddf_ext
    combined_griddf = pd.merge(griddf_ext, griddf_exty1[columns_to_add], on=['CELLCODE', 'year'], how='left')
    
    return combined_griddf


# %%
def to_gdf(griddf_ext):
    # Load Germany grid_landkreise to obtain the geometry
    with open(data_main_path+'/interim/grid_landkreise.pkl', 'rb') as f:
        geoms = pickle.load(f)
    geoms.info()
    
    gridgdf = griddf_ext.merge(geoms, on='CELLCODE')
    # Convert the DataFrame to a GeoDataFrame
    gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')
    # Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
    gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
    gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

    
    return geoms, gridgdf

# %%
def create_fullgriddf():
    output_dir = data_main_path+'/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dynamically refer to filename based on parameters
    griddf_filename = os.path.join(output_dir, 'griddf.parquet')

    # load the geodata for all years
    gld_allyears = load_geodata()

    if os.path.exists(griddf_filename):
        griddf = gpd.read_parquet(griddf_filename)
        print(f"Loaded griddf from {griddf_filename}")
    else:
        # create a griddf_base for each year
        griddf_allyears = {}
        for key, gld in gld_allyears.items():
            griddf_yearly = create_griddf_base(gld)
            check_duplicates(griddf_yearly)
            griddf_allyears[key] = griddf_yearly
            
        # put all years' griddfs into one dataframe
        griddf_base = pd.concat(griddf_allyears.values(), ignore_index=True)

        # calculate differences
        griddf_ydiff = calculate_yearlydiff(griddf_base)
        griddf_exty1 = calculate_diff_fromy1(griddf_base)
        griddf_ext = combine_griddfs(griddf_ydiff, griddf_exty1)
        
        # Check for infinite values in all columns
        for column in griddf_ext.columns:
            infinite_values = griddf_ext[column].isin([np.inf, -np.inf])
            print(f"Infinite values present in {column}:", infinite_values.any())

            # Optionally, print the rows with infinite values
            if infinite_values.any():
                print(f"Rows with infinite values in {column}:")
                print(griddf_ext[infinite_values])

            # Handle infinite values by replacing them with NaN
            griddf_ext[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # rename griddf_ext to griddf
        griddf = griddf_ext
        griddf.to_parquet(griddf_filename)
        print(f"Saved griddf to {griddf_filename}")
    
    # clean up memory
    del griddf_allyears, griddf_base, griddf_ydiff, griddf_exty1, griddf_ext
    gc.collect()

    return gld_allyears, griddf

# %% drop griddf outliers i.e., grids which have  fields < 300
# but only if all fields < 300 for all years in which the grid is in the dataset
def clean_griddf(griddf):
    # Remove grids with fields < 300
    griddf_clean = griddf[~(griddf['fields'] < 300)]
    outliers = griddf[griddf['fields'] < 300]
    # Log the count of unique values of 'CELLCODE' in the outliers
    logging.info(f"Unique CELLCODES with fields < 300: {outliers['CELLCODE'].nunique()}")

    # Step 2: Create a DataFrame for unique CELLCODES
    # and total count of years for unique CELLOCODES in griddf and outliers
    griddf_unique = griddf.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()
    outliers_unique = outliers.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()

    # Step 3: Add a column to check if occurrences match
    merged_outliers = outliers_unique.merge(
        griddf_unique, 
        on='CELLCODE', 
        suffixes=('_outliers', '_griddf'),
        how='left'
    )
    merged_outliers['all_years_in_data'] = merged_outliers['total_occurrence_outliers'] == merged_outliers['total_occurrence_griddf']

    # Step 4: Filter out rows where 'all_years_in_data' is no
    unmatched_outliers = merged_outliers[merged_outliers['all_years_in_data'] == False]

    # Step 5: Filter original outliers DataFrame for unmatched CELLCODES
    unmatched_outlier_codes = unmatched_outliers['CELLCODE']
    unmatched_outliers_df = outliers[outliers['CELLCODE'].isin(unmatched_outlier_codes)]

    # Step 6: Join these rows to griddf_clean
    final_cleaned_griddf = pd.concat([griddf_clean, unmatched_outliers_df], ignore_index=True)
    
    # Step 7: Create a final outliers DataFrame without the unmatched CELLCODES
    final_outliers = outliers[~(outliers['CELLCODE'].isin(unmatched_outlier_codes))]
    
    # Step 8: Drop rows with specific CELLCODEs and LANDKREIS
    # drop griddf_cl rows where ['CELLCODE'].isin(['10kmE431N333', '10kmE431N334'])
    griddf_cl = final_cleaned_griddf.copy()
    griddf_cl = griddf_cl[~griddf_cl['CELLCODE'].isin(['10kmE431N333', '10kmE431N334'])]
    
    # drop griddf_cl rows where ['LANDKREIS'].isin(['Küstenmeer Region Lüneburg', 'Küstenmeer Region Weser-Ems'])
    griddf_cl = griddf_cl[~griddf_cl['LANDKREIS'].isin(["Küstenmeer Region Lüneburg", "Küstenmeer Region Weser-Ems"])]
    
    logging.info(f"Final cleaned griddf shape: {griddf_cl.shape}")
    
    # clean up memory
    del griddf, griddf_clean, outliers, griddf_unique, outliers_unique, merged_outliers, unmatched_outliers, unmatched_outlier_codes, unmatched_outliers_df, final_cleaned_griddf, final_outliers
    gc.collect()

    return griddf_cl, final_outliers

# %% B.
#########################################################################
# compute mean and median for columns in griddf. save the results to a csv file
def desc_grid(griddf):
    def compute_grid_allyear_stats(griddf):
        # 1. Compute general all year data descriptive statistics
        grid_allyears_stats = griddf.select_dtypes(include='number').describe()
        # Add a column to indicate the type of statistic
        grid_allyears_stats['statistic'] = grid_allyears_stats.index
        # Reorder columns to place 'statistic' at the front
        grid_allyears_stats = grid_allyears_stats[['statistic'] + list(grid_allyears_stats.columns[:-1])]
        
        # Save the descriptive statistics to a CSV file
        output_dir = 'reports/statistics'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'grid_allyears_stats.csv')
        if not os.path.exists(filename):
            grid_allyears_stats.to_csv(filename, index=False)
            print(f"Saved gen_stats to {filename}")
        
        return grid_allyears_stats
    grid_allyears_stats = compute_grid_allyear_stats(griddf)
    
    def compute_grid_year_average(griddf):
        # 2. Group by 'year' and calculate useful stats across grids
        grid_yearly_stats = griddf.groupby('year').agg(
            fields_sum=('fields', 'sum'),
            fields_mean=('fields', 'mean'),
            fields_std = ('fields', 'std'),
            fields_av_yearlydiff=('fields_yearly_diff', 'mean'),
            fields_adiffy1=('fields_diff_from_y1', 'mean'),
            fields_apercdiffy1=('fields_percdiff_to_y1', 'mean'),
                    
            group_count_mean=('group_count', 'mean'),
            group_count_av_yearlydiff=('group_count_yearly_diff', 'mean'),
            group_count_adiffy1=('group_count_diff_from_y1', 'mean'),
            group_count_apercdiffy1=('group_count_percdiff_to_y1', 'mean'),

            fsha_sum_sum=('fsha_sum', 'sum'),
            fsha_sum_mean=('fsha_sum', 'mean'),
            fsha_sum_std = ('fsha_sum', 'std'),
            fsha_sum_av_yearlydiff=('fsha_sum_yearly_diff', 'mean'),
            fsha_sum_adiffy1=('fsha_sum_diff_from_y1', 'mean'),
            fsha_sum_apercdiffy1=('fsha_sum_percdiff_to_y1', 'mean'),

            mfs_ha_mean=('mfs_ha', 'mean'),
            mfs_ha_std=('mfs_ha', 'std'),
            mfs_ha_av_yearlydiff=('mfs_ha_yearly_diff', 'mean'),
            mfs_ha_adiffy1=('mfs_ha_diff_from_y1', 'mean'),
            mfs_ha_apercdiffy1=('mfs_ha_percdiff_to_y1', 'mean'),

            med_fsha_mean=('medfs_ha', 'mean'),
            med_fsha_med=('medfs_ha', 'median'),
            med_fsha_std=('medfs_ha', 'std'),
            med_fsha_av_yearlydiff=('medfs_ha_yearly_diff', 'mean'),
            med_fsha_adiffy1=('medfs_ha_diff_from_y1', 'mean'),
            med_fsha_apercdiffy1=('medfs_ha_percdiff_to_y1', 'mean'),

            med_fsha_yearlydiff_med=('medfs_ha_yearly_diff', 'median'),
            med_fsha_diffy1_med=('medfs_ha_diff_from_y1', 'median'),
            med_fsha_percdiffy1_med=('medfs_ha_percdiff_to_y1', 'median'),            

            mperi_mean=('mperi', 'mean'), #averge mean perimeter
            mperi_std = ('mperi', 'std'),
            mperi_av_yearlydiff=('mperi_yearly_diff', 'mean'),
            mperi_adiffy1=('mperi_diff_from_y1', 'mean'),
            mperi_apercdiffy1=('mperi_percdiff_to_y1', 'mean'),
            
            medperi_med=('medperi', 'median'),
            medperi_yearlydiff_med=('medperi_yearly_diff', 'median'),
            medperi_diffy1_med=('medperi_diff_from_y1', 'median'),
            medperi_percdiffy1_med=('medperi_percdiff_to_y1', 'median'),

            mshape_mean=('mshape', 'mean'),
            mshape_std=('mshape', 'std'),
            mshape_av_yearlydiff=('mshape_yearly_diff', 'mean'),
            mshape_adiffy1=('mshape_diff_from_y1', 'mean'),
            mshape_apercdiffy1=('mshape_percdiff_to_y1', 'mean'),
            
            medshape_mean=('medshape', 'mean'),
            medshape_std=('medshape', 'std'),
            medshape_av_yearlydiff=('medshape_yearly_diff', 'mean'),
            medshape_adiffy1=('medshape_diff_from_y1', 'mean'),
            medshape_apercdiffy1=('medshape_percdiff_to_y1', 'mean'),

            medshape_med=('medshape', 'median'),
            medshape_yearlydiff_med=('medshape_yearly_diff', 'median'),
            medshape_diffy1_med=('medshape_diff_from_y1', 'median'),
            medshape_percdiffy1_med=('medshape_percdiff_to_y1', 'median'),
            
            fields_ha_mean=('fields_ha', 'mean'),
            fields_ha_med=('fields_ha', 'median'),
            fields_ha_std=('fields_ha', 'std'),
            fields_ha_av_yearlydiff=('fields_ha_yearly_diff', 'mean'),
            fields_ha_adiffy1=('fields_ha_diff_from_y1', 'mean'),
            fields_ha_diffy1_med=('fields_ha_diff_from_y1', 'median'),
            fields_ha_apercdiffy1=('fields_ha_percdiff_to_y1', 'mean'),
            fields_ha_percdiffy1_med=('fields_ha_percdiff_to_y1', 'median')

        ).reset_index()
            
        return grid_yearly_stats
    grid_yearly_stats = compute_grid_year_average(griddf)

    return grid_allyears_stats, grid_yearly_stats

# Silence the print statements in a function call
def silence_prints(func, *args, **kwargs):
    # Create a string IO stream to catch any print outputs
    with io.StringIO() as f, contextlib.redirect_stdout(f):
        return func(*args, **kwargs)  # Call the function without print outputs
######################################################################################

# %%

