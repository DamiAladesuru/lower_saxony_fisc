# %%
import os
import pandas as pd

# Set up the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # or two levels up if needed
print(project_root)

os.chdir(project_root)
print("Current working dir:", os.getcwd())

from src.analysis.desc import gridgdf_desc as gd

#%% load extended gridgdf i.e., grid level data with climate, main crop, elevation and other attributes
gld, _ = gd.silence_prints(gd.create_gridgdf)
gridgdf_cluster = pd.read_pickle('data/interim/gridgdf/gridgdf_naturraum_klima_east_elev.pkl')

# %%
def create_filtered_dfs(gridgdf, year, category_col, category_mapping):
    """
    Create DataFrames for each category value in a specific column for a given year.

    Parameters:
    - gridgdf: GeoDataFrame containing the data
    - year: int, the year to filter on
    - category_col: str, column name to use for category filtering
    - category_mapping: dict, mapping of labels to column values 
                        (e.g., {'west': 1, 'east': 0} or {'cereals': 'Cereals'})

    Returns:
    - dict: keys are labels from category_mapping, values are filtered DataFrames
    """
    filtered_dfs = {}
    for label, value in category_mapping.items():
        filtered_df = gridgdf[
            (gridgdf['year'] == year) &
            (gridgdf[category_col] == value)
        ]
        filtered_dfs[label] = filtered_df

    return filtered_dfs

'''
# For 'eastwest':
category_mapping = {'west': 1, 'east': 0}
dfs = create_filtered_dfs(gridgdf_cluster, 2020, 'eastwest', category_mapping)
'''
# %%
# For 'main_crop_group':
category_mapping = {
    'forage': 'ackerfutter',
    'grassland': 'dauergrünland',
    'cereal': 'getreide'
}
cat_dfs = create_filtered_dfs(gridgdf_cluster, 2023, 'main_crop_group', category_mapping)

# %%
def create_gld_subsets(dfs, gld):
    """
    Extract rows from gld DataFrame based on CELLCODEs in thresh_dfs.
    """
    subsets = {}
    for key, grids_df in dfs.items():
        subset_name = f"gld_{key}"
        subsets[subset_name] = gld[gld['CELLCODE'].isin(grids_df['CELLCODE'])]
    return subsets

gld_subsets = create_gld_subsets(cat_dfs, gld)

# %%
##################### Appendix A ########################
# Set year variables
year1 = 2012
year2 = 2023
top_n = 10  # number of top kulturart to keep

# --- Step 1: Total area across all years ---
total_area_all_years = (
    gld.groupby('kulturart')['area_ha']
    .sum()
    .rename('total_area_all_years')
)

# Compute percentage share of each kulturart
total_area_all_years_df = total_area_all_years.to_frame()
total_area_all_years_df['area_percent_all_years'] = (
    100 * total_area_all_years_df['total_area_all_years'] / total_area_all_years_df['total_area_all_years'].sum()
)

# --- Step 2: Aggregate for 2012 ---
aggregated_2012 = gld[gld['year'] == year1].groupby('kulturart').agg(
    area_ha_2012=('area_ha', 'sum'),
    area_med_2012=('area_ha', 'median'),
    shape_med_2012=('shape', 'median'),
    row_count_2012=('kulturart', 'size')
).join(total_area_all_years_df)


# --- Step 3: Aggregate for 2023 ---
aggregated_2023 = gld[gld['year'] == year2].groupby('kulturart').agg(
    area_ha_2023=('area_ha', 'sum'),
    area_med_2023=('area_ha', 'median'),
    shape_med_2023=('shape', 'median'),
    row_count_2023=('kulturart', 'size')
).join(total_area_all_years_df)


# --- Step 4: Merge both years ---
df_group = pd.merge(
    aggregated_2012,
    aggregated_2023,
    how='outer',
    left_index=True,
    right_index=True,
    suffixes=('_2012', '_2023')
)
# Calculate how many fields each hectare of each kulturart has
df_group['fields_ha_2012'] = (
    df_group['row_count_2012'] / df_group['area_ha_2012']
)
df_group['fields_ha_2012'] = df_group['fields_ha_2012'].round(2)
df_group['fields_ha_2023'] = (
    df_group['row_count_2023'] / df_group['area_ha_2023']
)
df_group['fields_ha_2023'] = df_group['fields_ha_2023'].round(2)

# Ensure we carry total_area_all_years and area_percent_all_years only once
df_group['total_area_all_years'] = df_group['total_area_all_years_2012'].fillna(df_group['total_area_all_years_2023'])
df_group['area_percent_all_years'] = df_group['area_percent_all_years_2012'].fillna(df_group['area_percent_all_years_2023'])

# Drop duplicate columns
df_group.drop(columns=[
    'total_area_all_years_2012', 'total_area_all_years_2023',
    'area_percent_all_years_2012', 'area_percent_all_years_2023'
], inplace=True)


# Fill any remaining NaNs with 0
df_group.fillna(0, inplace=True)

# --- Step 5: Calculate Differences ---
df_group['areadiff_y2_y1'] = df_group['area_ha_2023'] - df_group['area_ha_2012']
df_group['medfsdiff_y2_y1'] = df_group['area_med_2023'] - df_group['area_med_2012']
df_group['shapediff_y2_y1'] = df_group['shape_med_2023'] - df_group['shape_med_2012']
df_group['fieldsdiff_y2_y1'] = df_group['row_count_2023'] - df_group['row_count_2012']
df_group['fields_ha_diff_y2_y1'] = df_group['fields_ha_2023'] - df_group['fields_ha_2012']

# Ensure all numeric columns are of type float
numeric_cols = [
    'area_ha_2012', 'area_ha_2023', 'area_med_2012', 'area_med_2023',
    'shape_med_2012', 'shape_med_2023', 'row_count_2012', 'row_count_2023',
    'total_area_all_years', 'area_percent_all_years',
    'areadiff_y2_y1', 'medfsdiff_y2_y1', 'shapediff_y2_y1',
    'fieldsdiff_y2_y1', 'fields_ha_diff_y2_y1'
]
df_group[numeric_cols] = df_group[numeric_cols].astype(float)

# --- Step 6: Filter to top N kulturarts by total area ---
df_top_kultur = df_group.sort_values(by='total_area_all_years', ascending=False).head(top_n)

# --- Step 7: Save to CSV ---
df_top_kultur.to_csv("reports/kchange_csvs/top_kulturart.csv", encoding='utf-8-sig')

##################### Appendix B ########################
# %% obtain summary statistics for each category DataFrame
# This will calculate the number of unique CELLCODEs and 
# the counts of positive, negative, and zero values for the specified columns.
# It will then create a summary DataFrame and save it to a CSV file.
# This statistics in used in Appendix B descriptive text.
summary_stats = []

columns_to_analyze = ['fields_ha_percdiff_to_y1', 'medfs_ha_percdiff_to_y1']

for key, df in cat_dfs.items():
    total = df['CELLCODE'].nunique()

    row = {
        'category': key,
        'total_cellcodes': total
    }

    for col in columns_to_analyze:
        positive = (df[col] > 0).sum()
        negative = (df[col] < 0).sum()
        zero = (df[col] == 0).sum()

        pos_prop = (positive / total)*100
        
        row[f'{col}_pos'] = positive
        row[f'{col}_neg'] = negative
        row[f'{col}_zero'] = zero
        row[f'{col}_positive_proportion'] = pos_prop

    summary_stats.append(row)

# Create the summary DataFrame
summary_df = pd.DataFrame(summary_stats)

# Save to CSV
summary_df.to_csv("reports/kchange_csvs/maincrop_dfs_summary.csv", index=False, encoding='utf-8-sig')

# %%
def calculate_change_for_groups(groups_dict, year1, year2):
    """
    Calculate the row count change for the groups (subsets in gld_subsets) in the given year.
    """
    results = {}
    
    for group_name, group_data in groups_dict.items():
        # Group by 'kulturart' and aggregate 'area_ha' and 'row_count' for each group and year
        aggregated_2012 = group_data[group_data['year'] == year1].groupby('kulturart').agg(
            area_ha_2012=('area_ha', 'sum'),
            area_med_2012=('area_ha', 'median'),
            row_count_2012=('kulturart', 'size')
        )

        aggregated_2023 = group_data[group_data['year'] == year2].groupby('kulturart').agg(
            area_ha_2023=('area_ha', 'sum'),
            area_med_2023=('area_ha', 'median'),
            row_count_2023=('kulturart', 'size')
        )

        # Merge the 2012 and 2023 aggregated data for comparison
        df_group = pd.merge(aggregated_2012[[f"area_ha_{year1}", f"area_med_{year1}", f"row_count_{year1}"]],
                             aggregated_2023[[f"area_ha_{year2}", f"area_med_{year2}", f"row_count_{year2}"]],
                             how='outer',
                             left_index=True, right_index=True)

        # Fill any missing values with 0
        df_group.fillna(0, inplace=True)

        # Calculate how many fields each hectare of each kulturart has
        df_group['fields_ha_2012'] = (df_group['row_count_2012'] / df_group['area_ha_2012'])
        df_group['fields_ha_2012'] = df_group['fields_ha_2012'].round(2)
        df_group['fields_ha_2023'] = (df_group['row_count_2023'] / df_group['area_ha_2023'])
        df_group['fields_ha_2023'] = df_group['fields_ha_2023'].round(2)

        # Calculate the difference in rows between 2023 and 2012 for each kulturart
        df_group['areadiffy2_y1'] = df_group[f"area_ha_{year2}"] - df_group[f"area_ha_{year1}"]
        df_group['medfsdiffy2_y1'] = df_group[f"area_med_{year2}"] - df_group[f"area_med_{year1}"]
        df_group['fieldsdiffy2_y1'] = df_group[f"row_count_{year2}"] - df_group[f"row_count_{year1}"]
        df_group['fields_hadiffy2_y1'] = df_group['fields_ha_2023'] - df_group['fields_ha_2012']
        
        # Identify the direction of change per crop
        df_group['crop_area_direction_of_change'] = df_group['areadiffy2_y1'].apply(
            lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
        )
        df_group['crop_medfs_direction_of_change'] = df_group['medfsdiffy2_y1'].apply(
            lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
        )
        df_group['crop_fields_direction_of_change'] = df_group['fieldsdiffy2_y1'].apply(
            lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
        )        
        df_group['crop_fields_ha_direction_of_change'] = df_group['fields_hadiffy2_y1'].apply(
            lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change')
        )

        # compute total change
        df_group['total_fieldsdiffy2_y1'] = df_group['fieldsdiffy2_y1'].sum()
        df_group['total_areadiffy2_y1'] = df_group['areadiffy2_y1'].sum()
        
        # Calculate how much the change of each kulturart contributes to the total change
        df_group['contribution_to_total_fieldschange'] = (df_group['fieldsdiffy2_y1'] / df_group['total_fieldsdiffy2_y1']) * 100
        df_group['contribution_to_total_areachange'] = (df_group['areadiffy2_y1'] / df_group['total_areadiffy2_y1']) * 100
 
        total_rows_2012 = group_data[group_data['year'] == year1].shape[0]
        total_area_2012 = group_data[group_data['year'] == year1]['area_ha'].sum()
        total_rows_2023 = group_data[group_data['year'] == year2].shape[0]
        total_area_2023 = group_data[group_data['year'] == year2]['area_ha'].sum()

        total_fields_ha_2012 = total_rows_2012 / total_area_2012 if total_area_2012 != 0 else 0
        total_fields_ha_2023 = total_rows_2023 / total_area_2023 if total_area_2023 != 0 else 0

        aggregate_fields_hadiff = total_fields_ha_2023 - total_fields_ha_2012

        df_group['total_fields_ha_2012'] = total_fields_ha_2012
        df_group['total_fields_ha_2023'] = total_fields_ha_2023
        df_group['aggregate_fields_hadiff'] = aggregate_fields_hadiff
        
        
        # Store results for the group
        results[group_name] = df_group

    return results

change = calculate_change_for_groups(gld_subsets, 2012, 2023)

# %%
def get_top_changes(dfs: dict, column: str, top_n: int = 10, save_to_csv: bool = False, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    results = {}

    for key, df in dfs.items():
        top_increase = df.sort_values(column, ascending=False).head(top_n)
        top_decrease = df.sort_values(column, ascending=True).head(top_n)
        combined = pd.concat([top_increase, top_decrease])
        
        results[f'{key}_increase'] = top_increase
        results[f'{key}_decrease'] = top_decrease
        results[f'{key}_combined'] = combined

        if save_to_csv:
            filename = f"{key}_top_{column}_combined.csv"
            filepath = os.path.join(output_dir, filename)
            combined.to_csv(filepath, encoding='utf-8-sig')

    return results


top_fields = get_top_changes(change, column='contribution_to_total_fieldschange', save_to_csv=True, output_dir='reports/kchange_csvs')
top_area = get_top_changes(change, column='contribution_to_total_areachange', save_to_csv=True, output_dir='reports/kchange_csvs')

# %%
def filter_by_top_kultur(main_df: pd.DataFrame, group_dfs: dict, top_n: int = 10, save_to_csv: bool = False, output_dir: str = "."):

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    grouped = main_df.groupby('kulturart')['area_ha'].sum().reset_index()
    top_kultur = grouped.sort_values(by='area_ha', ascending=False).head(top_n)

    filtered_dfs = {}
    for key, df in group_dfs.items():
        filtered = df[df.index.isin(top_kultur['kulturart'])].copy()
        filtered_dfs[key] = filtered

        if save_to_csv:
            filename = f"{key}_top_kulturart_filtered.csv"
            filepath = os.path.join(output_dir, filename)
            filtered.to_csv(filepath, encoding='utf-8-sig', index=True)

    return filtered_dfs

filtered_top_kultur = filter_by_top_kultur(
    main_df=gld,
    group_dfs=change,
    top_n=10,
    save_to_csv=True,
    output_dir="reports/kchange_csvs"
)

# %%
