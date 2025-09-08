# %% Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc": # or workspace if not lower_saxony_fisc
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

os.makedirs("reports/figures/farm_field", exist_ok=True)
from src.data.processing_griddata_utils import griddf_desc as gd

# %%
grid_fanim = pd.read_parquet("data/interim/gridgdf/grid_fanim.parquet")
'''grid_fanim is prepared in gridded_farmanimchange_data.py'''
_, grid_fanim = gd.to_gdf(grid_fanim)

# %%
# Calculate the total farm count per year
tot_farmc = grid_fanim.groupby('year', as_index=False)['farm_count'].sum()
tot_farmc.rename(columns={'farm_count': 'tot_farm_count'}, inplace=True)

# Bar plot: year vs tot_farm_count
plt.figure(figsize=(8, 5))
plt.bar(tot_farmc['year'], tot_farmc['tot_farm_count'], color='mediumseagreen', edgecolor='black')
plt.xlabel('Year')
plt.ylabel('Total Farm Count')
plt.title('Total Farms in the Study Area per Year')
plt.tight_layout()
plt.show()

# %% examine variables' histogram plots for one year
# so as to see distribution of values and decide on bins
gdf_oneyear = grid_fanim[grid_fanim['year'] == 2023]

# Plot histogram
ax = gdf_oneyear["farm_count_percdiff_to_y1"].hist(bins=20)

# Set labels and title
plt.xlabel("Farm Count Yearly % Change")
plt.ylabel("Number of Grid Cells")
plt.title("Distribution of Yearly % Change in Number of Farms")

# Enable minor ticks
plt.minorticks_on()

# Customize tick appearance (optional)
#ax.tick_params(axis='both', which='both', direction='in', length=6)  # both major and minor
ax.tick_params(axis='both', which='minor', length=3, color='gray')   # only minor

# Show the plot
plt.show()

##############################################################
# NOT NECCEASARY but useful:
# run these to check how many cellcodes are in each landkreis
# to verify why we have single cellcodes with distinct change value
#############################################################
# %%
# 1. Count the number of CELLCODEs for each LANDKREIS
landkreis_counts = grid_fanim.groupby('LANDKREIS')['CELLCODE'].nunique().reset_index()
landkreis_counts.columns = ['LANDKREIS', 'Unique_CELLCODE_Count']

# save to csv
landkreis_counts.to_csv('data/interim/landkreis_counts.csv', index=False)

# 2. Check if each CELLCODE uniquely belongs to a LANDKREIS
cellcode_landkreis_counts = grid_fanim.groupby('CELLCODE')['LANDKREIS'].nunique().reset_index()
cellcode_landkreis_counts.columns = ['CELLCODE', 'Unique_LANDKREIS_Count']

# Identify CELLCODEs that belong to more than one Landkreis
non_unique_cellcodes = cellcode_landkreis_counts[cellcode_landkreis_counts['Unique_LANDKREIS_Count'] > 1]

print(non_unique_cellcodes)

# Check if all CELLCODEs are unique to a single Landkreis
if non_unique_cellcodes.empty:
    print("Each CELLCODE uniquely belongs to a single Landkreis.")
else:
    print(f"There are {len(non_unique_cellcodes)} CELLCODEs that belong to multiple Landkreise.")

#############################################################################
# Create the map plots of change in variables of interest
#############################################################################
# %%
def plot_choropleth(gdf, column, title, bins, cmap="coolwarm", save_path=None, colorbar_label=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import colors, cm

    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])
    labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
    gdf["binned"] = pd.cut(gdf[column], bins=bins, labels=labels, right=False)

    norm = colors.BoundaryNorm(boundaries=bins, ncolors=256)
    scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.1,
        edgecolor="white",
        norm=norm,
        legend=False,
        ax=ax
    )

    cbar = plt.colorbar(scalar_mappable, ax=ax, orientation="vertical", pad=0.02, aspect=30, shrink=0.7)
    cbar.set_label(colorbar_label if colorbar_label else title)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    

# we work with 2020 because this is the most recent year in the farmanim data
grid_fanimy = grid_fanim[grid_fanim['year'] == 2020]

# ---- PLOTS ----
plot_choropleth(
    grid_fanimy,
    "farm_count_yearly_percdiff",
    title="Change in Total Number of Farms (2016 vs 2010)\n"
    "- Proxy for 2015",
    bins=[-15, -10, -9, -8, -7, -6, -5, 0, 5, 6, 7, 8, 9, 10, 15],
    colorbar_label="Percentage Change (%)",
    save_path="reports/figures/farm_field/farmcount_change_2015.svg"
)

#%%
plot_choropleth(
    grid_fanimy,
    "farm_count_percdiff_to_y1",
    title="Change in Total Number of Farms (2020 vs 2010/12)",
    bins=[-22, -20, -18, -16, -14, -12, -10, 0, 10, 12, 14, 16, 18, 20, 22],
    colorbar_label="Percentage Change (%)",
    save_path="reports/figures/farm_field/farmcount_toy1_2020.svg"
)

# %%
plot_choropleth(
    grid_fanimy,
    "stocking_density_yearly_percdiff",
    title="Change in Stocking Density (2016 vs 2010)\n"
    "- Proxy for 2015",
    bins=[-80, -20, -15, -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 20, 80],
    colorbar_label="Percentage Change (%)",
    save_path="reports/figures/farm_field/stockingdensity_change_2015.svg"
)

#bins=[-80, -20, -15, -10, -7.5, 0, 2.5, 5, 7.5, 10, 15, 20, 80]
#%%
plot_choropleth(
    grid_fanimy,
    "animalfarms_yearly_percdiff",
    title="Change in Number of Farms with Livestock (2016 vs 2010)\n"
    "- Proxy for 2015",
    bins=[-20, -15, -10, -9, -8, -7, -6, -5, 0, 5, 6, 7, 8, 9, 10, 15, 20],
    colorbar_label="Percentage Change (%)",
    save_path="reports/figures/farm_field/animalfarmcount_change_2015.svg"
)



#################################################
# examine relationships
#################################################
# %%
def plot_grouped_regressions(df, x, y, xlabel, ylabel, suptitle, save_path=None):
    df_copy = df.copy()
    df_copy["main_crop_group"] = df_copy["main_crop_group"].replace({
        "Ackerfutter": "Forages",
        "Dauergrünland": "Grasslands",
        "Getreide": "Cereals"
    })
    
    # Add 'All' group
    df_copy_all = df_copy.copy()
    df_copy_all["main_crop_group"] = "All"
    df_copy = pd.concat([df_copy, df_copy_all], ignore_index=True)

    unique_groups = ["All", "Cereals", "Forages", "Grasslands"]
    n_cols = len(unique_groups)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True)
    plt.style.use('seaborn-v0_8-paper')

    for i, (group, ax) in enumerate(zip(unique_groups, axes)):
        group_df = df_copy[df_copy["main_crop_group"] == group]
        
        if group_df.empty:
            ax.set_title(group + " (no data)")
            continue
        
        # Scatter plot
        ax.scatter(group_df[x], group_df[y], alpha=0.6, label='Data')
        
        # Regression line
        if len(group_df) > 1:
            slope, intercept = np.polyfit(group_df[x], group_df[y], 1)
            x_vals = np.linspace(group_df[x].min(), group_df[x].max(), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=2,
                    label=f'Fit (slope={slope:.2f})')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(group)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    plt.suptitle(suptitle, fontsize=16, y=1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#####################################################################
# %% # Examine relationships for percentage differences to year1
# stocking density vs farm count
x = 'stocking_density_percdiff_to_y1'
y = 'farm_count_percdiff_to_y1'
plot_grouped_regressions(
    df=grid_fanimy,
    x=x,
    y=y,
    xlabel="Change in Stocking Density",
    ylabel="Change in Number of Farms",
    suptitle="(To year1) Change in Stocking Density vs Number of Farms by Crop Group (2020)",
    save_path="reports//figures/farm_field/stocking_vs_farms_by_crop_y1.svg"
)

# stocking density vs fields/ha 
x = 'stocking_density_percdiff_to_y1'
y = 'fields_ha_percdiff_to_y1'
plot_grouped_regressions(
    df=grid_fanimy,
    x=x,
    y=y,
    xlabel="Change in Stocking Density",
    ylabel="Change in Number of Fields",
    suptitle="(To year1) Change in Stocking Density vs Number of Fields by Crop Group (2020)",
    save_path="reports//figures/farm_field/stocking_vs_fields_by_crop_y1.svg"
)

# number of farms vs. fields/ha
x = 'farm_count_percdiff_to_y1'
y = 'fields_ha_percdiff_to_y1'
plot_grouped_regressions(
    df=grid_fanimy,
    x=x,
    y=y,
    xlabel="Change in Number of Farms",
    ylabel="Change in Number of Fields",
    suptitle="(To year1) Change in Number of Farms vs Number of Fields by Crop Group (2020)",
    save_path="reports//figures/farm_field/farms_vs_fields_by_crop_y1.svg"
)

# mean farm size vs. median field size
x = 'mean_farmsize_percdiff_to_y1'
y = 'medfs_ha_percdiff_to_y1'
plot_grouped_regressions(
    df=grid_fanimy,
    x=x,
    y=y,
    xlabel="Change in Average Farm Size",
    ylabel="Change in Average Field Size",
    suptitle="(To year1) Change in Average Farm Size vs Average Field Size by Crop Group (2020)",
    save_path="reports//figures/farm_field/avfarm_vs_avfields_size_by_crop_y1.svg"
)

##############################################################
# %% Examine relationships for yearly percentage differences
# For yearly percentage diffs, it's better to work with filters to drop one or two outlier points.
'''
Data range is as follows:
fields_ha_yearly_percdiff: -5 to 5
farm_count_yearly_percdiff: -15 to 0
stocking_density_yearly_percdiff: -20 to 10

So, for example, for when x is fields column, drop rows where x > 6
when y is farm column, drop rows where y > 0
'''
# stocking density vs farm count 
x = 'stocking_density_yearly_percdiff'
y = 'farm_count_yearly_percdiff'
filtered_df = grid_fanimy[(grid_fanimy[x] <= 10) & (grid_fanimy[y] <= 5)]
plot_grouped_regressions(
    df=filtered_df,
    x=x,
    y=y,
    xlabel="Change in Stocking Density",
    ylabel="Change in Number of Farms",
    suptitle="Change in Stocking Density vs Number of Farms by Crop Group (2020 to 2016)",
    save_path="reports//figures/farm_field/stocking_vs_farms_by_crop_yearly.svg"
)

# stocking density vs fields/ha 
x = 'stocking_density_yearly_percdiff'
y = 'fields_ha_yearly_percdiff'
filtered_df = grid_fanimy[(grid_fanimy[x] <= 10) & (grid_fanimy[y] <= 5)]
plot_grouped_regressions(
    df=filtered_df,
    x=x,
    y=y,
    xlabel="Change in Stocking Density",
    ylabel="Change in Number of Fields",
    suptitle="Change in Stocking Density vs Number of Fields by Crop Group (2020 to 2016)",
    save_path="reports//figures/farm_field/stocking_vs_fields_by_crop_yearly.svg"
)

# number of farms vs. fields/ha
x = 'farm_count_yearly_percdiff'
y = 'fields_ha_yearly_percdiff'
filtered_df = grid_fanimy[(grid_fanimy[x] <= 5) & (grid_fanimy[y] <= 5)]
plot_grouped_regressions(
    df=filtered_df,
    x=x,
    y=y,
    xlabel="Change in Number of Farms",
    ylabel="Change in Number of Fields",
    suptitle="Change in Number of Farms vs Number of Fields by Crop Group (2020 to 2016)",
    save_path="reports//figures/farm_field/farms_vs_fields_by_crop_yearly.svg"
)

# %%
