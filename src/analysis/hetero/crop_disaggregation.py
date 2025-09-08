# %%
from pathlib import Path
import os
import pandas as pd
from pathlib import Path

from src.data.processing_griddata_utils import griddf_desc as gd

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):
    if parent.name == "lower_saxony_fisc": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root = os.getcwd()
data_main_path = open(project_root + "/datapath.txt").read()


os.makedirs("reports/kchange_csvs", exist_ok=True)

# %%
# Load data and create no-geometry combined DataFrame
gld_allyears = gd.load_geodata()
# %%
gld_no_geom = {}
for year, gdf in gld_allyears.items():
    df_no_geom = gdf.drop(columns=['geometry']).copy()
    gld_no_geom[year] = df_no_geom

# %%
# Select the years you want
years_to_combine = [2012, 2024]

# Concatenate into a single DataFrame, adding a year column
gld = pd.concat(
    [gld_no_geom[year].assign(year=year) for year in years_to_combine],
    ignore_index=True
)

print(gld.head())
print(f"Shape of combined DataFrame: {gld.shape}")


#%% load extended griddf i.e., grid level data with climate, main crop, elevation and other attributes
griddf_cluster = pd.read_parquet('data/interim/gridgdf/griddf_klima_crop_elev.parquet')

# %%
def create_filtered_dfs(griddf, year, category_col, category_mapping):
    """
    Create DataFrames for each category value in a specific column for a given year.

    Parameters:
    - griddf: DataFrame containing the data
    - year: int, the year to filter on
    - category_col: str, column name to use for category filtering
    - category_mapping: dict, mapping of labels to column values 
                        (e.g., {'west': 1, 'east': 0} or {'cereals': 'Cereals'})

    Returns:
    - dict: keys are labels from category_mapping, values are filtered DataFrames
    """
    filtered_dfs = {}
    for label, value in category_mapping.items():
        filtered_df = griddf[
            (griddf['year'] == year) &
            (griddf[category_col] == value)
        ]
        filtered_dfs[label] = filtered_df

    return filtered_dfs

'''
# For 'eastwest':
category_mapping = {'west': 1, 'east': 0}
dfs = create_filtered_dfs(griddf_cluster, 2020, 'eastwest', category_mapping)
'''
# %%
# For 'main_crop_group':
category_mapping = {
    'forage': 'Ackerfutter',
    'grassland': 'Dauergrünland',
    'cereal': 'Getreide',
    'roots': 'Hackfrüchte'
}
cat_dfs = create_filtered_dfs(griddf_cluster, 2024, 'main_crop_group', category_mapping)

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
year2 = 2024
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


# --- Step 3: Aggregate for 2024 ---
aggregated_2024 = gld[gld['year'] == year2].groupby('kulturart').agg(
    area_ha_2024=('area_ha', 'sum'),
    area_med_2024=('area_ha', 'median'),
    shape_med_2024=('shape', 'median'),
    row_count_2024=('kulturart', 'size')
).join(total_area_all_years_df)


# --- Step 4: Merge both years ---
df_group = pd.merge(
    aggregated_2012,
    aggregated_2024,
    how='outer',
    left_index=True,
    right_index=True,
    suffixes=('_2012', '_2024')
)
# Calculate how many fields each hectare of each kulturart has
df_group['fields_ha_2012'] = (
    df_group['row_count_2012'] / df_group['area_ha_2012']
)
df_group['fields_ha_2012'] = df_group['fields_ha_2012'].round(2)
df_group['fields_ha_2024'] = (
    df_group['row_count_2024'] / df_group['area_ha_2024']
)
df_group['fields_ha_2024'] = df_group['fields_ha_2024'].round(2)

# Ensure we carry total_area_all_years and area_percent_all_years only once
df_group['total_area_all_years'] = df_group['total_area_all_years_2012'].fillna(df_group['total_area_all_years_2024'])
df_group['area_percent_all_years'] = df_group['area_percent_all_years_2012'].fillna(df_group['area_percent_all_years_2024'])

# Drop duplicate columns
df_group.drop(columns=[
    'total_area_all_years_2012', 'total_area_all_years_2024',
    'area_percent_all_years_2012', 'area_percent_all_years_2024'
], inplace=True)


# Fill any remaining NaNs with 0
df_group.fillna(0, inplace=True)

# --- Step 5: Calculate Differences ---
df_group['areadiff_y2_y1'] = df_group['area_ha_2024'] - df_group['area_ha_2012']
df_group['medfsdiff_y2_y1'] = df_group['area_med_2024'] - df_group['area_med_2012']
df_group['shapediff_y2_y1'] = df_group['shape_med_2024'] - df_group['shape_med_2012']
df_group['fieldsdiff_y2_y1'] = df_group['row_count_2024'] - df_group['row_count_2012']
df_group['fields_ha_diff_y2_y1'] = df_group['fields_ha_2024'] - df_group['fields_ha_2012']

# Ensure all numeric columns are of type float
numeric_cols = [
    'area_ha_2012', 'area_ha_2024', 'area_med_2012', 'area_med_2024',
    'shape_med_2012', 'shape_med_2024', 'row_count_2012', 'row_count_2024',
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

        pos_prop = (positive / total) * 100
        
        # counts & proportions
        row[f'{col}_pos'] = positive
        row[f'{col}_neg'] = negative
        row[f'{col}_zero'] = zero
        row[f'{col}_positive_proportion'] = pos_prop

        # averages across CELLCODEs
        row[f'{col}_mean'] = df[col].mean(skipna=True)

    summary_stats.append(row)

# Create the summary DataFrame
summary_df = pd.DataFrame(summary_stats)

# Save to CSV
summary_df.to_csv("reports/kchange_csvs/maincrop_dfs_summary.csv", index=False, encoding='utf-8-sig')

print(summary_df)

# %%
summary_stats = []

columns_to_analyze = ['fields_ha_percdiff_to_y1', 'medfs_ha_percdiff_to_y1']

for key, df in cat_dfs.items():
    # group by LANDKREIS instead of CELLCODE
    grouped = df.groupby("LANDKREIS")

    for landkreis, g in grouped:
        total = g['CELLCODE'].nunique()

        row = {
            'category': key,
            'LANDKREIS': landkreis,
            'total_cellcodes': total
        }

        for col in columns_to_analyze:
            positive = (g[col] > 0).sum()
            negative = (g[col] < 0).sum()
            zero = (g[col] == 0).sum()

            pos_prop = (positive / total) * 100 if total > 0 else 0

            # counts & proportions
            row[f'{col}_pos'] = positive
            row[f'{col}_neg'] = negative
            row[f'{col}_zero'] = zero
            row[f'{col}_positive_proportion'] = pos_prop

            # average across CELLCODEs in this LANDKREIS
            row[f'{col}_mean'] = g[col].mean(skipna=True)

        summary_stats.append(row)

# Create summary DataFrame
summary_df_landkreis = pd.DataFrame(summary_stats)

# Save to CSV
summary_df_landkreis.to_csv("reports/kchange_csvs/maincrop_dfs_summary_landkreis.csv",
                            index=False, encoding='utf-8-sig')

print(summary_df_landkreis.head())


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

        aggregated_2024 = group_data[group_data['year'] == year2].groupby('kulturart').agg(
            area_ha_2024=('area_ha', 'sum'),
            area_med_2024=('area_ha', 'median'),
            row_count_2024=('kulturart', 'size')
        )

        # Merge the 2012 and 2024 aggregated data for comparison
        df_group = pd.merge(aggregated_2012[[f"area_ha_{year1}", f"area_med_{year1}", f"row_count_{year1}"]],
                             aggregated_2024[[f"area_ha_{year2}", f"area_med_{year2}", f"row_count_{year2}"]],
                             how='outer',
                             left_index=True, right_index=True)

        # Fill any missing values with 0
        df_group.fillna(0, inplace=True)

        # Calculate how many fields each hectare of each kulturart has
        df_group['fields_ha_2012'] = (df_group['row_count_2012'] / df_group['area_ha_2012'])
        df_group['fields_ha_2012'] = df_group['fields_ha_2012'].round(2)
        df_group['fields_ha_2024'] = (df_group['row_count_2024'] / df_group['area_ha_2024'])
        df_group['fields_ha_2024'] = df_group['fields_ha_2024'].round(2)

        # Calculate the difference in rows between 2024 and 2012 for each kulturart
        df_group['areadiffy2_y1'] = df_group[f"area_ha_{year2}"] - df_group[f"area_ha_{year1}"]
        df_group['medfsdiffy2_y1'] = df_group[f"area_med_{year2}"] - df_group[f"area_med_{year1}"]
        df_group['fieldsdiffy2_y1'] = df_group[f"row_count_{year2}"] - df_group[f"row_count_{year1}"]
        df_group['fields_hadiffy2_y1'] = df_group['fields_ha_2024'] - df_group['fields_ha_2012']
        
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
        total_rows_2024 = group_data[group_data['year'] == year2].shape[0]
        total_area_2024 = group_data[group_data['year'] == year2]['area_ha'].sum()

        total_fields_ha_2012 = total_rows_2012 / total_area_2012 if total_area_2012 != 0 else 0
        total_fields_ha_2024 = total_rows_2024 / total_area_2024 if total_area_2024 != 0 else 0

        aggregate_fields_hadiff = total_fields_ha_2024 - total_fields_ha_2012

        df_group['total_fields_ha_2012'] = total_fields_ha_2012
        df_group['total_fields_ha_2024'] = total_fields_ha_2024
        df_group['aggregate_fields_hadiff'] = aggregate_fields_hadiff
        
        
        # Store results for the group
        results[group_name] = df_group

    return results

change = calculate_change_for_groups(gld_subsets, 2012, 2024)

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
summary_stats = []

for subset_name, df in gld_subsets.items():
    # filter by years 2012 and 2024
    df_2012 = df[df["year"] == 2012]
    df_2024 = df[df["year"] == 2024]

    # number of rows
    count_2012 = len(df_2012)
    count_2024 = len(df_2024)
    count_diff = count_2024 - count_2012

    # total area (ha)
    area_2012 = df_2012["area_ha"].sum()
    area_2024 = df_2024["area_ha"].sum()
    area_diff = area_2024 - area_2012

    # collect results
    summary_stats.append({
        "subset": subset_name,
        "rows_2012": count_2012,
        "rows_2024": count_2024,
        "row_diff": count_diff,
        "area_ha_2012": area_2012,
        "area_ha_2024": area_2024,
        "area_ha_diff": area_diff
    })

# Create DataFrame with results
summary_df = pd.DataFrame(summary_stats)

# Save to CSV if needed
summary_df.to_csv("reports/kchange_csvs/gld_subsets_2012_2024_summary.csv", index=False, encoding="utf-8-sig")

print(summary_df)

# %%
