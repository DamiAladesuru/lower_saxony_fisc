# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # or two levels up if needed
print(project_root)

os.chdir(project_root)
print("Current working dir:", os.getcwd())

from src.data import dataload as dl

# %%
gld = dl.load_data(loadExistingData = True)
# 1. print info to learn about the total number of entries (len(gld)), data types and presence/ number of missing values
gld.info()

# %% get a feel of the data ###
############################
# 2. drop area columns and create a copy making kulturcode, CELLCODE and LANDKREIS categorical
def copy_data(data):
    data['kulturcode'] = data['kulturcode'].astype('category')
    data['CELLCODE'] = data['CELLCODE'].astype('category')
    data['LANDKREIS'] = data['LANDKREIS'].astype('category')
    return data
data = copy_data(gld)

# %%
# 3. examine data distribution (perhaps clean outliers)
# 3.1. take two representative years, 1st and last and create distribution plots for
# numeric columns
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
def distribution_plots(df):
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Group by year and create distribution plots for numeric columns in a FacetGrid
    for year, group in df.groupby('year'):
        plot_file = f'reports/distribution_plots_{year}.png'
        
        if os.path.exists(plot_file):
            # Load and display the plot from the file
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.title(f'Distribution of Numeric Columns in {year}')
            plt.show()
        else:
            # Create a FacetGrid for the current year
            g = sns.FacetGrid(pd.melt(group, id_vars=['year'], value_vars=numeric_columns), col='variable', col_wrap=3, sharex=False, sharey=False)
            g.map(sns.histplot, 'value', kde=True)
            
            # Set titles and labels
            g.fig.suptitle(f'Distribution of Numeric Columns in {year}', y=1.02)
            g.set_axis_labels('Value', 'Frequency')
            
            # Save the FacetGrid plot
            g.savefig(plot_file)
            plt.show()
# Filter the DataFrame to include only the years 2012 and 2023
filtered_data = data[data['year'].isin([2012, 2023])]
distribution_plots(filtered_data)
#  the data has a long right tail

# %% 
# check min and max values of numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns
min_max_values = data[numeric_columns].agg(['min', 'max']).T
min_max_values.columns = ['Min', 'Max']
print("Min and Max values of numeric columns:")
print(min_max_values)

''' we see min area_m2 is 0.0004 which is way below any reasonable agricultural field size
    Thus, we can set a threshold value to remove fields with area_m2 below 100m2
'''
# %% 3.2
data_trim = data[data['area_m2'] >= 100]  # filter out fields with area_m2 < 100

# %% create a bar plot of the percentage distribution of area_ha in specified ranges
# to get an understanding of the area distribution with and without outliers
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bar_plot(data, column, bins, labels):
    # Bin the data
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=labels, right=False)

    # Calculate percentage frequency using the dynamically named binned column
    percentage_frequency = data[f'{column}_binned'].value_counts(normalize=True) * 100
    percentage_frequency = percentage_frequency.reindex(labels)  # Ensure the order matches the labels
    
    # Drop empty bins if any
    percentage_frequency = percentage_frequency[percentage_frequency > 0]

    # Convert labels to strings to avoid the warning
    percentage_frequency.index = percentage_frequency.index.astype(str)

    # Create the percentage frequency bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=percentage_frequency.index, y=percentage_frequency.values)
    plt.title(f'Percentage Distribution of {column} in Specified Ranges')
    plt.xlabel('Range')
    plt.ylabel('Percentage Frequency')
    plt.grid(False)
    plt.xticks(rotation=45)

    # Remove the top and right spines
    sns.despine(left=True, bottom=True)

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.show()

# Usage for the field level data
bins1 = [0, 0.01, 1, 2, 4, 6, 8, 10, 20, float('inf')]
labels1 = ['0-0.01', '0.01-1', '1-2', '2-4', '4-6', '6-8', '8-10', '10-20', '>20']
bar_plot(data, 'area_ha', bins1, labels1)
bar_plot(data_trim, 'area_ha', bins1, labels1)
#data_trim.info()

# %% mean of data_trim 'area_ha' column
data['area_ha'].mean()

# %% 4.  yearly exploration of the data without outliers (i.e., use data_trim)
# 4.1. group data by year and count the unique values in the non-numeric columns
# Identify non-numeric columns
def count_unique(data):
    category_columns = data.select_dtypes(include=['category']).columns
    # Exclude 'area_ha_binned'
    category_columns = category_columns.drop('area_ha_binned', errors='ignore')
    # Group by year and count unique values in non-numeric columns
    unique_counts_by_year = gld.groupby('year')[category_columns].nunique()
    # Print the unique counts
    print("Unique counts in non-numeric columns by year:")
    print(unique_counts_by_year)
count_unique(data_trim)

# %% 4.2. for numeric columns, get the range of of values: the min, max, mean, median a standard deviation
def get_numstats(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    range_stats_by_year = gld.groupby('year')[numeric_columns].agg(['min', 'max', 'mean', 'median', 'std'])
    output_excel_path = 'reports/trim_numcolstats_by_year.xlsx'
    range_stats_by_year.to_excel(output_excel_path, sheet_name='NumericColumnStats')
    print(f"Range statistics saved to {output_excel_path}")
get_numstats(data_trim)

# %% 4.3. replot the distribution plots for the numeric columns
def distribution_plots(df):
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Group by year and create distribution plots for numeric columns in a FacetGrid
    for year, group in df.groupby('year'):
        plot_file = f'reports/trimmed_distribution_plots_{year}.png'
        
        if os.path.exists(plot_file):
            # Load and display the plot from the file
            img = plt.imread(plot_file)
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.title(f'Distribution of Numeric Columns in {year} without Outliers')
            plt.show()
        else:
            # Create a FacetGrid for the current year
            g = sns.FacetGrid(pd.melt(group, id_vars=['year'], value_vars=numeric_columns), col='variable', col_wrap=3, sharex=False, sharey=False)
            g.map(sns.histplot, 'value', kde=True)
            
            # Set titles and labels
            g.fig.suptitle(f'Distribution of Numeric Columns in {year} without Outliers', y=1.02)
            g.set_axis_labels('Value', 'Frequency')
            
            # Save the FacetGrid plot
            g.savefig(plot_file)
            plt.show()
# Filter the DataFrame to include only the years 2012 and 2023
filtered_data = data_trim[data_trim['year'].isin([2012, 2023])]
distribution_plots(filtered_data)

      
# %% 5. Test for correlation between numeric columns
def test_correlation(df):
    # Select numeric columns
    df = df.drop(columns=['area_m2', 'area_m2'], errors='ignore')
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    
    # Print the correlation matrix
    print("Correlation matrix:")
    print(correlation_matrix)
    
    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numeric Columns')
    plt.show()

# Example usage
test_correlation(data_trim)

''' Here, we see that perimeter and shape are positively correlated as
    we would expect that of two fields with thesame area, one with more perimter
    would have a more complex shape.
    
    Also, of two fields with same perimter, one with larger area is utilizing its
    perimeter more efficiently than the other with smaller area.
    Thus shape and area would be negatively correlated.
    
    This association is not observed between par and area,
    nullifying the validity of the formular compared to cpar/shape.'''

# %% 6
# general data descriptive statistics grouped by year
def yearly_gen_statistics(gld):
    yearlygen_stats = gld.groupby('year').agg(
        fields = ('geometry', 'count'),
        
        area_ha_sum=('area_ha', 'sum'),
        area_ha_mean=('area_ha', 'mean'),
        area_ha_median=('area_ha', 'median'),

        peri_m_sum=('peri_m', 'sum'),
        peri_m_mean=('peri_m', 'mean'),
        peri_m_median=('peri_m', 'median'),
                    
        shape_sum=('shape', 'sum'),
        shape_mean=('shape', 'mean'),
        shape_median=('shape', 'median'),
                                

    ).reset_index()
    
    yearlygen_stats['fields_ha'] = yearlygen_stats['fields'] / yearlygen_stats['area_ha_sum']

    return yearlygen_stats

ygs = yearly_gen_statistics(gld)
#ygs1 = yearly_gen_statistics(data_trim)

# %%
# Create line plot of yearly sum of all field areas or any other ygs/ygs1 column
sns.set_style("whitegrid")
sns.set_context("talk")
sns.set_palette("colorblind")

fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=ygs, x='year', y='area_ha_sum', color='purple', marker='o', ax=ax)
ax.set_title('Yearly sum of all field areas')
ax.set_ylabel('Area (ha)')
ax.set_xlabel('Year')
plt.show()

