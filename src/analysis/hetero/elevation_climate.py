# %%
import os
import pandas as pd
import seaborn as sns
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

from src.data.processing_griddata_utils import griddf_desc as gd
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/figures/results", exist_ok=True)

# %% Load extended grid-level data (climate, main crop, elevation, etc.)
griddf_cluster = pd.read_parquet('data/interim/gridgdf/griddf_klima_crop_elev.parquet')
_, gridgdf_cluster = gd.to_gdf(griddf_cluster)

############################################
# 2024-specific inspection & plotting
############################################
# %% Filter for year 2024 only
gridgdf_cluster24 = gridgdf_cluster[gridgdf_cluster['year'] == 2024]

# %% Plot mean elevation for 2024 agricultural polygons
fig, ax = plt.subplots(figsize=(10, 10))
gridgdf_cluster24.plot(
    column='mean_elevation', ax=ax, legend=True,
    legend_kwds={'label': "Mean Elevation (m) (2024 only)", 'orientation': "horizontal"},
    cmap='viridis', edgecolor='black', linewidth=0.5
)
ax.set_axis_off()
ax.set_title("Mean Elevation (2024)", fontsize=16)
plt.show()

# %% Stripplot of elevation points per climate category (2024 only)
plt.figure(figsize=(12, 6))
sns.stripplot(data=gridgdf_cluster24, x='main_Klima_EN', y='mean_elevation', jitter=True, alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.title('Elevation Points per Climate Category (2024 only)')
plt.tight_layout()
plt.show()


############################################
# Aggregated analysis over all years
############################################
# %% Aggregate numeric columns by CELLCODE (mean over all years)
numeric_cols = gridgdf_cluster.select_dtypes(include='number')
cols_to_agg = [col for col in numeric_cols if col != ['LK']]
cellcode_means = gridgdf_cluster.groupby('CELLCODE')[cols_to_agg].mean().reset_index()
# Note: mean here is aggregated over **all years**

# %% Get most frequent main_Klima_EN per CELLCODE and merge
main_klima = gridgdf_cluster.groupby('CELLCODE')['main_Klima_EN']\
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
cellcode_means = cellcode_means.merge(main_klima, on='CELLCODE', how='left')

############################################
# Regression & scatter plots (aggregated over all years)
############################################
# %% Median field size change vs mean elevation (aggregated)
plt.figure(figsize=(8, 6))
sns.regplot(
    x='medfs_ha_percdiff_to_y1', y='mean_elevation',
    data=cellcode_means, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}
)
plt.xlabel("Average Change (%) in Median Field Size", fontsize=12)
plt.ylabel("Mean Elevation (m) (aggregated over all years)", fontsize=12)
plt.title("Average Change in Field Size vs Elevation", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("reports//figures/results/changemedfs_v_meanelev_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %% Number of fields change vs mean elevation (aggregated)
plt.figure(figsize=(8, 6))
sns.regplot(
    x='fields_ha_percdiff_to_y1', y='mean_elevation',
    data=cellcode_means, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}
)
plt.xlabel("Average Change (%) in Number of Fields", fontsize=12)
plt.ylabel("Mean Elevation (m) (aggregated over all years)", fontsize=12)
plt.title("Average Change in Fields vs Elevation", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("reports//figures/results/changefields_v_meanelev_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %% Scatter: median field size change vs elevation by climate (aggregated)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='medfs_ha_percdiff_to_y1', y='mean_elevation',
    hue='main_Klima_EN', data=cellcode_means, palette='Set1', alpha=0.6
)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Average Change (%) in Median Field Size", fontsize=12)
plt.ylabel("Mean Elevation (m) (aggregated over all years)", fontsize=12)
plt.title("Average Change in Field Size vs Elevation and Climate", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Climatic Zone", loc='upper left', frameon=True, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("reports//figures/results/changemedfs_v_meanelev_v_klima_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %% Scatter: number of fields change vs elevation by climate (aggregated)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='fields_ha_percdiff_to_y1', y='mean_elevation',
    hue='main_Klima_EN', data=cellcode_means, palette='Set1', alpha=0.6
)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Average Change (%) in Number of Fields", fontsize=12)
plt.ylabel("Mean Elevation (m) (aggregated over all years)", fontsize=12)
plt.title("Average Change in Number of Fields vs Elevation and Climate", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title="Climatic Zone", loc='upper left', frameon=True, fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig("reports//figures/results/changefields_v_meanelev_klima_agg.svg", dpi=300, bbox_inches='tight')
plt.show()
