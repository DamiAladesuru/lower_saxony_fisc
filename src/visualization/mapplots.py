# %%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from matplotlib import font_manager
print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))

# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()

from src.analysis.desc import gridgdf_desc as gd
os.makedirs("reports/figures", exist_ok=True)
# %% load data
gld, gridgdf = gd.silence_prints(gd.create_gridgdf)
# I always want to load gridgdf and process clean gridgdf separately so I can have uncleeaned data for comparison or sensitivity analysis
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)

# %% re-project data
geoData = gridgdf_cl.to_crs(epsg=4326)

'''Create the 2012 and difference plots together'''
# %% combined baseline and difference plots
def plot_choropleths_maps(gdf, year_list, columns, col_titles, bins_labels_dict, basecmap="Paired", diffcmap="coolwarm", suptitle=""):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    nrows = 4
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))  # Added extra height for colorbar

    # First column: Baseline values for 2012
    gdf_2012 = gdf[gdf['year'] == 2012]
    axes[0, 0].set_title("2012", fontsize=12)

    for row, column in enumerate(columns[:4]):
        bins, labels = bins_labels_dict[column]
        
        binned_column = f"{column}_bins"
        gdf_2012[binned_column] = pd.cut(gdf_2012[column], bins=bins, labels=labels, right=False)
        
        gdf_2012.plot(
            column=binned_column,
            cmap=basecmap,
            legend=True,
            legend_kwds={
                "loc": "lower left",
                "fontsize": "x-small",
                "title": None,
                "title_fontsize": "x-small",
                "frameon": False,
                "labelspacing": 0.2,
                "alignment": "center",
                "borderpad": 0.5,
                "bbox_to_anchor": (0, -0.1),
            },
            edgecolor="white",
            linewidth=0.1,
            ax=axes[row, 0],
        )
        
        axes[row, 0].set_axis_off()
        axes[row, 0].text(-0.05, 0.5, col_titles[row], rotation=90, verticalalignment='center', 
                          horizontalalignment='center', transform=axes[row, 0].transAxes, fontsize=12)

    # Remaining columns: Percentage differences
    bins, labels = bins_labels_dict[columns[4]]  # Use the same bins and labels for all percentage difference columns
    norm = colors.BoundaryNorm(bins, plt.get_cmap(diffcmap).N)

    for col, year in enumerate(year_list, start=1):
        gdf_year = gdf[gdf['year'] == year]
        
        for row, column in enumerate(columns[4:]):
            binned_column = f"{column}_bins"
            gdf_year[binned_column] = pd.cut(gdf_year[column], bins=bins, labels=labels, right=False)
            
            gdf_year.plot(
                column=column,  # Use the actual values, not the binned ones
                cmap=diffcmap,
                norm=norm,
                legend=False,  # We'll add a shared legend later
                edgecolor="white",
                linewidth=0.1,
                ax=axes[row, col],
            )
            
            if row == 0:
                axes[row, col].set_title(f"{year}", fontsize=12)
            
            axes[row, col].set_axis_off()

    # Add a shared colorbar legend below the subplots
    scalar_mappable = plt.cm.ScalarMappable(cmap=diffcmap, norm=norm)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.1,
        aspect=40,
        shrink=0.8,
    )

    # Compute bin midpoints for tick placement
    bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    # Set tick positions and labels
    cbar.set_ticks(bin_midpoints)
    cbar.set_ticklabels(labels)
    cbar.set_label("Percentage Change from Base Year (2012)")

    # Adjust layout using subplots_adjust
    fig.subplots_adjust(
        left=0.15,  # Adjust left margin
        right=0.85,  # Adjust right margin
        top=0.9,  # Adjust top margin
        bottom=0.15,  # Adjust bottom margin for the legend
        wspace=0,  # Adjust width between subplots
        hspace=0,  # Adjust height between subplots
    )
    
    # Add a super title
    fig.suptitle(suptitle, fontsize=16, y=0.925) # reduce y to move title nearer to plots
    
    #save plot as svg
    plt.savefig("reports/figures/FiSC_choropleth_mapBerlin_.svg", format="svg", bbox_inches='tight')
    plt.savefig("reports/figures/FiSC_choropleth_mapBerlinPN_.png", format="png", bbox_inches='tight')
    
    plt.show()
# %%
# Usage
columns = ["medfs_ha", "medperi", "fields_ha", "medshape", 
           "medfs_ha_percdiff_to_y1", "medperi_percdiff_to_y1", "fields_ha_percdiff_to_y1", "medshape_percdiff_to_y1"]
year_list = [2015, 2019, 2023]
col_titles = ["Median Field Size (ha)", "Median Perimeter (m)", "Number of Fields/TotalArea", "Median Shape Complexity"]

binsA = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labelsA = ["<-6%", "-5%", "-3%", "-1.5%", "-0.5%", "0.5%", "1.5%", "3%", "5%", ">6%"]

mfsbins = [0.5, 1.5, 2.0, 2.5, 3.5, 5.0]
mfslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(mfsbins[:-1], mfsbins[1:])]

peribins = [440, 620, 670, 730, 780, 1300]
perilabels = ["440 - 620", "620 - 670", "670 - 730", "730 - 780", "780 - 1300"]

fieldsbins = [0, 0.29, 0.33, 0.37, 0.43, 1.0]
fieldslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(fieldsbins[:-1], fieldsbins[1:])]

shapebins = [1, 1.3, 1.35, 1.4, 1.5, 2.0]
shapelabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(shapebins[:-1], shapebins[1:])]

bins_labels_dict = {
    "medfs_ha": (mfsbins, mfslabels),
    "medperi": (peribins, perilabels),
    "fields_ha": (fieldsbins, fieldslabels),
    "medshape": (shapebins, shapelabels),
    "medfs_ha_percdiff_to_y1": (binsA, labelsA),
    "medperi_percdiff_to_y1": (binsA, labelsA),
    "fields_ha_percdiff_to_y1": (binsA, labelsA),
    "medshape_percdiff_to_y1": (binsA, labelsA)
}
# %%
plot_choropleths_maps(geoData, year_list, columns, col_titles, bins_labels_dict, basecmap="berlin", diffcmap="coolwarm", suptitle="Field Structural Change Over Time")


# %%
