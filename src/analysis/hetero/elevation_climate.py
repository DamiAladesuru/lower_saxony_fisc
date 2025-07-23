# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # or two levels up if needed
print(project_root)

os.chdir(project_root)
print("Current working dir:", os.getcwd())


#%% load extended gridgdf i.e., grid level data with climate, main crop, elevation and other attributes
gridgdf_cluster = pd.read_pickle('data/interim/gridgdf/gridgdf_klima_crop_elev.pkl')

# %% visually inspect
gridgdf_cluster23 = gridgdf_cluster[gridgdf_cluster['year'] == 2023]

# Plot the agricultural land polygons colored by mean elevation
fig, ax = plt.subplots(figsize=(10, 10))

# Use the 'mean_elevation' column to color the polygons
gridgdf_cluster23.plot(column='mean_elevation', ax=ax, legend=True,
               legend_kwds={'label': "Mean Elevation (m)",
                            'orientation': "horizontal"},
               cmap='viridis', edgecolor='black', linewidth=0.5)

# Remove map box/border
ax.set_axis_off()  # Removes axis lines, ticks, and box

# Optional: Add a title
ax.set_title("Mean Elevation", fontsize=16)

# Display the plot
plt.show()

############################################
#plots
############################################
# %%
numeric_cols = gridgdf_cluster.select_dtypes(include='number')
cols_to_agg = [col for col in numeric_cols if col != ['LK']]


# Group by CELLCODE and calculate mean for each numeric column except 'LK'
cellcode_means = gridgdf_cluster.groupby('CELLCODE')[cols_to_agg].mean().reset_index()

# %%
# For each CELLCODE, get the most frequent (mode) value of 'main_Klima_EN'
main_klima = gridgdf_cluster.groupby('CELLCODE')['main_Klima_EN'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()

# Merge with cellcode_means on 'CELLCODE'
cellcode_means = cellcode_means.merge(main_klima, on='CELLCODE', how='left')


# %%
plt.figure(figsize=(8, 6))
sns.regplot(x='medfs_ha_percdiff_to_y1', y='mean_elevation',
            data=cellcode_means, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})

plt.xlabel("Average Change (%) in Median Field Size", fontsize=12)
plt.ylabel("Mean Elevation (m)", fontsize=12)
plt.title("Average Change in Field Size vs Elevation", fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("reports//figures/results/changemedfs_v_meanelev_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.regplot(x='fields_ha_percdiff_to_y1', y='mean_elevation',
            data=cellcode_means, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})

plt.xlabel("Average Change (%) in Number of Fields", fontsize=12)
plt.ylabel("Mean Elevation (m)", fontsize=12)
plt.title("Average Change in Fields vs Elevation", fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("reports//figures/results/changefields_v_meanelev_agg.svg", dpi=300, bbox_inches='tight')
plt.show()


# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='medfs_ha_percdiff_to_y1',
    y='mean_elevation',
    hue='main_Klima_EN',
    data=cellcode_means,
    palette='Set1',
    alpha=0.6
)

plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

plt.xlabel("Average Change (%) in Median Field Size", fontsize=12)
plt.ylabel("Mean Elevation (m)", fontsize=12)
plt.title("Average Change in Field Size based on Elevation and Climatic Region", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Force legend inside lower-left corner
plt.legend(
    title="Climatic Zone",
    loc='upper left',       # Upper right
    #bbox_to_anchor=(0.02, 0.02),  # bbox_to_anchor=(0.98, 0.98))
    frameon=True,
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()
plt.savefig("reports//figures/results/changemedfs_v_meanelev_v_klima_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='fields_ha_percdiff_to_y1',
    y='mean_elevation',
    hue='main_Klima_EN',
    data=cellcode_means,
    palette='Set1',
    alpha=0.6
)

plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)

plt.xlabel("Average Change (%) in Number of Fields", fontsize=12)
plt.ylabel("Mean Elevation (m)", fontsize=12)
plt.title("Average Change in Number of Fields based on Elevation and Climatic Region", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Force legend inside lower-left corner
plt.legend(
    title="Climatic Zone",
    loc='upper left',       # Upper right
    #bbox_to_anchor=(0.02, 0.02),  # bbox_to_anchor=(0.98, 0.98))
    frameon=True,
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()
plt.savefig("reports//figures/results/changefields_v_meanelev_klima_agg.svg", dpi=300, bbox_inches='tight')
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.stripplot(data=gridgdf_cluster23, x='main_Klima_EN', y='mean_elevation', jitter=True, alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.title('Elevation Points per Climate Category')
plt.tight_layout()
plt.show()


# %%