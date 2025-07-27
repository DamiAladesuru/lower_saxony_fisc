# %%
import geopandas as gpd
import pandas as pd
import os
import pickle
from pathlib import Path

#COMMENTJB: I removed the absolute path names and streamlined the path name variables a bit
# Set up the project root directory
# Set up the project root directory
current_path = Path(__file__).resolve().parent
for parent in [current_path] + list(current_path.parents):

    if parent.name == "lower_saxony_fisc":
        os.chdir(parent)
        print(f"Changed working directory to: {parent}")
        break
project_root=os.getcwd()
data_main_path=open(project_root+"/datapath.txt").read()
#%%

# this script joins the eea grid with the landkreise boundaries of Niedersachsen
# and saves the result as a pickle file in the interim data folder.
# the result is a GeoDataFrame with the grid cells that intersect with the landkreise boundaries.
# the geodataframe is called and used in the dataload script to ssign landkreis and grid cell
# to the field data points.

'''This script should automatically run when called from the dataload script.'''

def join_gridregion(loadExistingData=False):

    # Define default values
    base_dir = data_main_path+"/verwaltungseinheiten"
    
    # define path to data interim
    intpath = data_main_path+'/interim'
        
    # load data if it exists already
    if loadExistingData and os.path.isfile(os.path.join(intpath, 'grid_landkreise.pkl')):
        with open(os.path.join(intpath, 'grid_landkreise.pkl'), 'rb') as f:
            grid_landkreise = pickle.load(f)
    else:
        # Load Landkreis file for regional boundaries
        landkreise = gpd.read_file(os.path.join(base_dir, "NDS_Landkreise.shp"))
        landkreise.info()
        landkreise = landkreise.to_crs("EPSG:25832")
        
        # Load Germany eea grid
        grid = gpd.read_file(data_main_path+'/raw/eea_10_km_eea-ref-grid-de_p_2013_v02_r00')
        grid = grid.to_crs(landkreise.crs)
        # create index for grid in order to create grid_landkreise
        grid_ = grid.reset_index().rename(columns={'index': 'id'})
        
        # join grid and landkreise so that landkreise boundaries filters out grids that are outside of land boundary
        grid_landkreise = gpd.sjoin(grid_, landkreise, how='inner', predicate='intersects')
        grid_landkreise.info()
        grid_landkreise.plot()
        
        # compare bounding boxes of land and grid_landkreise to check if all of land is within grid_landkreise
        land = gpd.read_file(os.path.join(base_dir, "NDS_Landesflaeche.shp"))
        land = land.to_crs("EPSG:25832")
                
        # Calculate total bounding box for land
        land_total_bounds = land.total_bounds  # [minx, miny, maxx, maxy]
        print(land_total_bounds)

        # Calculate total bounding box for grid_land
        grid_landkreise_total_bounds = grid_landkreise.total_bounds  # [minx, miny, maxx, maxy]
        print(grid_landkreise_total_bounds)
        # Check if land's minx >= grid_land's minx and land's miny >= grid_land's miny
        # and land's maxx <= grid_land's maxx and land's maxy <= grid_land's maxy
        if (land_total_bounds[0] >= grid_landkreise_total_bounds[0] and
            land_total_bounds[1] >= grid_landkreise_total_bounds[1] and
            land_total_bounds[2] <= grid_landkreise_total_bounds[2] and
            land_total_bounds[3] <= grid_landkreise_total_bounds[3]):
            print("All of land is within grid_landkr.")
        else:
            print("Some parts of land are not within grid_landkr.")

        ##############################
        # maybe plot this regional map to see it but it is not necessary
        # Reproject to WGS84
        #if grid_landkreise.crs != "EPSG:4326":
        #    regions = grid_landkreise.to_crs("EPSG:4326")
        # Plot the regions map
        #fig, ax = plt.subplots(figsize=(10, 10))
        #regions.plot(ax=ax, color='lightgrey', edgecolor='black')
        # Annotate the map with regional district names
        #for idx, row in regions.iterrows():
        #    # Calculate the centroid of each district
        #    centroid = row['geometry'].centroid
        #    plt.text(centroid.x, centroid.y, row['LANDKREIS'], fontsize=8, ha='center')
        # Add title and adjust layout
        #plt.title('Lower Saxony Regional Districts')
        #plt.xlabel('Longitude')
        #plt.ylabel('Latitude')
        #plt.tight_layout()
        # Show the plot
        #plt.show()
        ##############################

        #count unique cellcodes in grid_landkreise
        unique_cellcodes = grid_landkreise['CELLCODE'].nunique()
        print(unique_cellcodes) # should be 602

        #check for duplicated grids in grid_landkreise
        duplicates = grid_landkreise.duplicated('id')
        # Print the number of duplicates
        print(duplicates.sum()) #455

        # Create a sample with all double assigned grids which are crossing landkreis borders 
        # and, therefore, are assigned to more than one  landkreise in grid_landkreise.
        double_assigned = grid_landkreise[grid_landkreise.index.isin(grid_landkreise[grid_landkreise.index.duplicated()].index)]

        # - Delete all double assigned from grid_landkreise
        grid_landkreise = grid_landkreise[~grid_landkreise.index.isin(grid_landkreise[grid_landkreise.index.duplicated()].index)]

        # --- Estimate the largest intersection for each duplicated cellcode with landkreise in the
        #     double assigned sample. Use the unit of ha
        double_assigned['intersection'] = [
            a.intersection(landkreise[landkreise.index == b].\
                geometry.values[0]).area/10000 for a, b in zip(
                double_assigned.geometry.values, double_assigned.index_right
            )]

        # --- Sort by intersection area and keep only the  row with the largest intersection
        double_assignedsorted = double_assigned.sort_values(by='intersection').\
            groupby('id').last().reset_index()
    
        #--- Add the data double_assigned to the grid_landkreise data
        grid_landkreise = pd.concat([grid_landkreise, double_assignedsorted])
        grid_landkreise.info()

        # keep only needed columns
        grid_landkreise = grid_landkreise.drop(columns=['id', 'EOFORIGIN', 'NOFORIGIN', 'index_right', 'LK', 'intersection'])

        # save to pickle
        grid_landkreise.to_pickle(os.path.join(intpath, 'grid_landkreise.pkl'))

    return grid_landkreise
# %%
grid_landkreise = join_gridregion(loadExistingData=True)