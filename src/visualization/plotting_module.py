# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from joypy import joyplot
import geoplot as gplt
import pickle


# %%
'''Plotting functions'''

##############################
# multiline plots for all data
##############################
def multiline_metrics(df, title, ylabel, metrics, save_path, color_mapping=None, format='png', dpi=300):
    """
    Function to plot multiple metrics over time with predefined colors and save the plot.

    Args:
        df (DataFrame): Dataframe containing the data to plot.
        title (str): Title of the plot.
        ylabel (str): Label for the Y-axis.
        metrics (dict): Dictionary mapping labels to column names.
        save_path (str): File path to save the plot.
        color_mapping (dict): Optional dictionary specifying colors for each metric label.
        format (str): Format to save the plot (e.g., 'png', 'pdf', 'svg').
        dpi (int): Resolution of the plot in dots per inch.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    # Use provided color mapping or default to seaborn's color cycle
    for label, column in metrics.items():
        color = color_mapping.get(label, None) if color_mapping else None
        sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)

    plt.title(title, fontsize=16, pad=12)
    plt.xlabel('Year', labelpad=12, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title='Metrics')

    # Save the plot
    plt.savefig(save_path, format=format, dpi=dpi)
    print(f"Plot saved to {save_path}")

    plt.show()
# %%
#############################################
# Correlation
#############################################
# 1. Correlation test
def test_correlation(df, target_columns, new_column_names):
    # Calculate correlation matrix
    correlation_matrix = df[target_columns].corr()
    
    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = new_column_names
    correlation_matrix.index = new_column_names
    
    return correlation_matrix

# 2. Single matrix plot
def plot_correlation_matrix(df, title, target_columns, new_column_names):
    # Get the correlation matrix
    corr_matrix = test_correlation(df, target_columns, new_column_names)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
# 3. Correlation matrix for metrics with and without outlier
''' requires both data with and without outlier to be loaded'''

def plot_correlation_matrices(df1, df2, title1, title2):
    # Get the correlation matrices
    corr_matrix1 = test_correlation(df1)
    corr_matrix2 = test_correlation(df2)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first correlation matrix
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[0])
    axes[0].set_title(title1)
    
    # Plot the second correlation matrix
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[1])
    axes[1].set_title(title2)
    
    plt.tight_layout()
    plt.show()

####################################################################
# multimetric extended for facet plot for grouped subsamples of data
# created directly from gld.
####################################################################
def multimetric_ss_plot(data_dict, title, ylabel, metrics):
    # Load the label_color_dict from a pickle file
    label_color_dict_path = 'reports/figures/ToF/label_color_dict.pkl'
    
    # Check if the file exists to prevent errors
    if os.path.exists(label_color_dict_path):
        with open(label_color_dict_path, 'rb') as f:
            label_color_dict = pickle.load(f)  # Load color dictionary
    else:
        print(f"Warning: {label_color_dict_path} not found. Using an empty dictionary.")
        label_color_dict = {}  # Fallback to empty dict if file is missing
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Determine the number of subplots based on the number of Gruppe values
    n_subplots = len(data_dict)
    n_cols = min(3, n_subplots)  # Maximum 3 columns
    n_rows = (n_subplots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    for idx, (gruppe, df) in enumerate(data_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot each metric for this Gruppe
        for label, column in metrics.items():
            if label in label_color_dict:
                color = label_color_dict[label]
                sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
            else:
                line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
                color = line.get_lines()[-1].get_color()
                label_color_dict[label] = color  # Update the dictionary with the new color
        
        ax.set_title(f'Gruppe: {gruppe}')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.legend(title='Metrics', loc='best')
    
    # Remove any unused subplots
    for idx in range(n_subplots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()

    # Save the updated color dictionary back to the file
    with open(label_color_dict_path, 'wb') as f:
        pickle.dump(label_color_dict, f)


####################################################################
# multimetric grid plot for categories of data e.g., created from
# gridgdf. (crop category plot with shared legend)
####################################################################
def plot_metrics_for_group(ax, df_group, metrics, color_mapping, ylabel):
    """
    Helper function to plot metrics for a specific group on a given axis without a legend.
    """
    for label, column in metrics.items():
        color = color_mapping.get(label, None) if color_mapping else None
        sns.lineplot(data=df_group, x='year', y=column, label=label, marker='o', color=color, ax=ax)
    ax.set_title(f"Category: {df_group['subsample'].iloc[0]}", fontsize=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    # Remove the legend from individual subplots
    ax.legend_.remove()

def multiline_metrics_with_shared_legend(df, title, ylabel, metrics, save_path, color_mapping=None, format='png', dpi=300):
    """
    Function to create a grid of plots for each subsample group with a shared legend positioned under the general title.
    """
    sns.set(style="whitegrid")
    
    # Get unique subsample groups
    subsample_groups = df['subsample'].unique()
    n_groups = len(subsample_groups)
    
    # Set up a grid of subplots (3 columns)
    n_cols = 3
    n_rows = (n_groups + n_cols - 1) // n_cols  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 7 * n_rows), dpi=dpi)
    
    # Flatten axes array for easier indexing
    axes = axes.flatten()
    
    # Loop through each group and create a plot
    for i, subsample in enumerate(subsample_groups):
        df_group = df[df['subsample'] == subsample]
        plot_metrics_for_group(axes[i], df_group, metrics, color_mapping, ylabel)
    
    # Hide unused subplots if there are fewer groups than grid slots
    for j in range(i + 1, len(axes)):  # Hide any extra subplots
        axes[j].axis('off')
    
    # Create a shared legend and place it under the general title
    handles = []
    labels = []
    
    # Collect handles and labels from one of the groups
    for label, column in metrics.items():
        color = color_mapping.get(label, None) if color_mapping else None
        handles.append(plt.Line2D([], [], color=color, marker='o', label=label))
        labels.append(label)
    
    # Add shared legend to the figure (under general title)
    fig.legend(
        handles=handles,
        labels=labels,
        loc='upper center',  # Position at the top center of the figure
        bbox_to_anchor=(0.5, 0.92),  # Fine-tune position (x=centered at 0.5; y just below title)
        fontsize=10,
        ncol=len(metrics)  # Arrange all legend items in one row
    )
    
    # Add a global title and adjust layout
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])  # Adjust layout to leave space for the title and legend
    
    # Save the plot
    plt.savefig(save_path, format=format, dpi=dpi)
    print(f"Plot saved to {save_path}")
    
    plt.show()


############################################
# Function to stack plots in a grid
############################################
# %%  
def stack_plots_in_grid(df, unique_values, plot_func, col1, col2, ncols=3, figsize=(18, 12), grid_title=None):
    nrows = (len(unique_values) + ncols - 1) // ncols  # Calculate number of rows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)  # Create grid
    axes = axes.flatten()  # Flatten axes for easy iteration

    for i, value in enumerate(unique_values):
        ax = axes[i]
        plot_func(df, value, col1, col2, ax)  # Call the user-provided function to generate a plot
        ax.set_title(f"Year: {value}", fontsize=16)
    
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a suptitle for the entire grid
    if grid_title:
        fig.suptitle(grid_title, fontsize=20, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

# scatterplot of PAR and area for a given year
def scatterplot_par_area(df, year, col1, col2, ax):
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create scatterplot
    sns.scatterplot(
        data=df_year,
        x=col1,
        y=col2,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel(col1, fontsize=14)
    ax.set_ylabel(col2, fontsize=14)

#unique_years = sorted(gld['year'].unique())
#stack_plots_in_grid(gld, unique_years, scatterplot_par_area, "area_ha", "par", ncols=4, figsize=(25, 15))

# scatterplot of grid average PAR and area for a given year
def scatterplot_mpar_marea(df, year, col1, col2, ax):
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create scatterplot
    sns.scatterplot(
        data=df_year,
        x=col1,
        y=col2,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel(col1, fontsize=14)
    ax.set_ylabel(col2, fontsize=14)


############################################
# Function to create a joyplot for each year
############################################
# %%
def create_yearly_joyplot(df, by_column, plot_column, title_template):
    """
    Create a joyplot for each year in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - by_column: Column name to group by (e.g., 'Gruppe').
    - plot_column: Column name to plot (e.g., 'area_ha').
    - title_template: Template for the plot title (e.g., "Area distribution in {year}").
    """
    unique_years = df['year'].unique()
    
    for year in unique_years:
        # Subset the DataFrame for the current year
        df_year = df[df['year'] == year]
        
        # Create labels for the current year
        labels = [y for y in list(df_year[by_column].unique())]
        
        # Create the joyplot for the current year
        fig, axes = joyplot(
            df_year, 
            by=by_column, 
            column=plot_column, 
            labels=labels, 
            range_style='own', 
            linewidth=1, 
            legend=True, 
            figsize=(6, 5),
            title=title_template.format(year=year),
            colormap=cm.autumn
        )
    
    plt.show()

# call
#create_yearly_joyplot(gld_no4, 'Gruppe', 'par', "PAR distribution in {year}")

########################
# choropleth map geoplot
########################
# %% facet grid of chloropleth maps: sequential
def plot_facet_choropleth_with_geoplot(gdf, column, cmap='viridis', year_col='year', ncols=4, title=""):
    # Get unique years
    unique_years = sorted(gdf[year_col].unique())
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate rows based on ncols

    # Create the figure and axes
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), 
        subplot_kw={'projection': gplt.crs.AlbersEqualArea()}  # Use an appropriate CRS
    )
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot each year's choropleth map
    for i, year in enumerate(unique_years):
        ax = axes[i]
        
        # Subset GeoDataFrame for the current year
        gdf_year = gdf[gdf[year_col] == year]
        
        # Plot the choropleth map
        gplt.choropleth(
            gdf_year,
            hue=column,
            cmap=cmap,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=True,
        )
        # Add title
        ax.set_title(f"Year: {year}", fontsize=12)
    
    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=18, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Example Usage
# Assuming `geoData` is your GeoDataFrame with 'year' and 'medpar' columns
#plot_facet_choropleth_with_geoplot(geoData, column='medpar', cmap='plasma', year_col='year', ncols=4)
