# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging

# Set up the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))  # or two levels up if needed
print(project_root)

os.chdir(project_root)
print("Current working dir:", os.getcwd())


from src.data import dataload as dl

# Initialize logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Logging works!")

# %% 
# Group by year and get unique values of 'kulturcode' for each year
def get_uni_kulturcode(gld):
    output_dir='reports/Kulturcode/kulturcode_act_new.xlsx'
    # Group by year and get unique values of 'kulturcode' for each year
    kulturcode_act = gld.groupby('year')['kulturcode'].unique().to_dict()

    # Write the results to an Excel file with each year in a separate sheet
    with pd.ExcelWriter(output_dir, engine='openpyxl') as writer:
        for year, kulturcodes in kulturcode_act.items():
            # Convert the array of unique kulturcodes to a DataFrame
            df = pd.DataFrame(kulturcodes, columns=['kulturcode'])
            # Write the DataFrame to a sheet named after the year
            df.to_excel(writer, sheet_name=str(year), index=False)

    print(f"Unique kulturcodes by year have been written to '{output_dir}'")

    # Convert kulturcode_act and kulturart keys to numeric values
    kulturcode_act = {int(key): value for key, value in kulturcode_act.items()}
    # Print the keys
    print(kulturcode_act.keys())

    # Print the info for each sheet
    for key in kulturcode_act:
        print(f"{key}: {len(kulturcode_act[key])} unique kulturcodes")
        
    return kulturcode_act


def plot_unique_kulturcode_counts(kulturcode_act):
    output_image_path='reports/Kulturcode/unique_kulturcode_counts_by_year.png'
    # Create a DataFrame to store the year and the count of unique 'kulturcode' values
    df_year = {
        'year': list(kulturcode_act.keys()),
        'unique_kulturcode_count': [len(kulturcodes) for kulturcodes in kulturcode_act.values()]
    }
    df_unique_kulturcodes = pd.DataFrame(df_year)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df_unique_kulturcodes['year'], df_unique_kulturcodes['unique_kulturcode_count'], marker='o')
    plt.title('Unique Kulturcode Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Unique Kulturcode Count')
    plt.grid(False)
    plt.savefig(output_image_path)
    plt.show()


def load_kulturcode_description(file_path):
    # Load the multiple spreadsheets containing kulturart description and group
    kulturart = pd.read_excel(file_path, sheet_name=None)
    # Print the keys
    print(kulturart.keys())
      
    # keep only the columns 'Code', 'Kulturart' and 'Gruppe' for each year and rename 'Code'
    for key in kulturart:
        kulturart[key] = kulturart[key][['Code', 'Kulturart', 'Gruppe']]
        kulturart[key] = kulturart[key].rename(columns={'Code': 'kulturcode'})
        print(f"{key}: {kulturart[key].info()}")

    # Iterate over each year to check if all unique kulturcode values are numbers \
    # or if there are characters
    for key in kulturart:
    # Check for non-numeric kulturcode values
        non_numeric_kulturcodes = [code for code in kulturart[key]['kulturcode'] if not str(code).replace('.', '', 1).isdigit()]
        if non_numeric_kulturcodes:
            print(f"{key}: Non-numeric kulturcode values found: {non_numeric_kulturcodes}")
        else:
            print(f"{key}: All kulturcode values are numeric.")
        
    kulturart = {int(key): value for key, value in kulturart.items()}
    print(kulturart.keys())      
        
        
    return kulturart

def create_kulturcode_map(kulturcode_act, kulturart):
    # Initialize an empty dictionary to store the filtered datasets
    datacodewithart = {}

    # Convert all items in kulturcode_act to DataFrames if they are not already
    for key in kulturcode_act:
        if not isinstance(kulturcode_act[key], pd.DataFrame):
            # Convert to DataFrame and rename the column to 'kulturcode'
            kulturcode_act[key] = pd.DataFrame(kulturcode_act[key], columns=['kulturcode'])
        else:
            # Ensure the column is named 'kulturcode'
            kulturcode_act[key] = kulturcode_act[key].rename(columns={kulturcode_act[key].columns[0]: 'kulturcode'})

    # Convert all items in kulturart to DataFrames if they are not already
    for key in kulturart:
        if not isinstance(kulturart[key], pd.DataFrame):
            kulturart[key] = pd.DataFrame(kulturart[key])

    # Iterate through the keys (years) in the kulturcode_actual dictionary
    for key in kulturcode_act:
        # Check if the key is in the kulturart dictionary
        if key in kulturart:
            # Add the dataset to the new dictionary
            datacodewithart[key] = kulturcode_act[key]

    # For each key in the datacodewithart dictionary, merge data with kulturart data of same key on 'kulturcode' column
    for key in datacodewithart:
        datacodewithart[key] = pd.merge(datacodewithart[key], kulturart[key], on='kulturcode', how='left')
        print(f"{key}: {datacodewithart[key].info()}")

    # For each key in the datacodewithart dictionary, create a year column and set it to the key
    for key in datacodewithart:
        datacodewithart[key]['year'] = key
        print(f"{key}: {datacodewithart[key].info()}")

    # Append all dataframes in the datacodewithart dictionary to a single dataframe
    # this create a map of unique kulturcodes in data for years with available kulturart description
    kulturcode_map = pd.concat(datacodewithart.values(), ignore_index=True)
    kulturcode_map.info()

    return kulturcode_map

# manage years with missing kulturart description
# take the value of column 'gruppe' for each kulturcode in year 2020 and use it to fill empty \
    # values in 'gruppe' column for the same kulturcode in year 2021 and 2022
def map_2021_22(kulturcode_map):
    # Extract the rows for year 2020
    kulturcodemap_2020 = kulturcode_map[kulturcode_map['year'] == 2020]

    # Extract the rows for year 2021
    kulturcodemap_2021 = kulturcode_map[kulturcode_map['year'] == 2021]
    kulturcodemap_2021 = kulturcodemap_2021.drop(columns='Gruppe')

    # Extract the rows for year 2022
    kulturcodemap_2022 = kulturcode_map[kulturcode_map['year'] == 2022]
    kulturcodemap_2022 = kulturcodemap_2022.drop(columns='Gruppe')

    # Merge 2020 df to the dataframes
    kulturcodemap_2021 = pd.merge(kulturcodemap_2021, kulturcodemap_2020[['kulturcode', 'Gruppe']], on='kulturcode', how='left')
    kulturcodemap_2022 = pd.merge(kulturcodemap_2022, kulturcodemap_2020[['kulturcode', 'Gruppe']], on='kulturcode', how='left')

    # Print info for kulturcodemap_2020, kulturcodemap_2021 and kulturcodemap_2022
    print(kulturcodemap_2020.info())
    print(kulturcodemap_2021.info())
    print(kulturcodemap_2022.info())

    # Drop the rows with year 2021 and 2022 from the original dataframe
    kulturcode_map = kulturcode_map[kulturcode_map['year'] != 2021]
    kulturcode_map = kulturcode_map[kulturcode_map['year'] != 2022]

    # Concatenate the modified 2021 and 2022 dataframes back to the original dataframe
    kulturcode_map = pd.concat([kulturcode_map, kulturcodemap_2021, kulturcodemap_2022], ignore_index=True)

    # Rename 'Kulturart' column to 'kulturart'
    kulturcode_map = kulturcode_map.rename(columns={'Kulturart': 'kulturart'})

    kulturcode_map.info()
    
    
    return kulturcode_map

def fix_missingkulturart(kulturcode_map):
    # Check if there are rows with empty values in 'kulturart' column
    missing_kulturart = kulturcode_map[kulturcode_map['kulturart'].isnull()]
    logging.info(f"Number of rows with missing 'kulturart': {missing_kulturart.shape[0]}")

    # Remove rows with missing 'kulturart' from the original DataFrame
    kulturcode_map = kulturcode_map[~kulturcode_map['kulturart'].isnull()]
    logging.info(f"Number of rows after removing missing 'kulturart': {kulturcode_map.shape[0]}")

    # Merge kulturcode_map with missing_kulturart on 'kulturcode'
    kau_df = missing_kulturart.merge(kulturcode_map, on=['kulturcode'], how='left', suffixes=('_missing_kulturart', '_kulturcode_map'))
    # Calculate the difference in years between the missing 'kulturart' and the 'kulturcode' map
    kau_df['year_diff'] = kau_df['year_kulturcode_map'] - kau_df['year_missing_kulturart']
    kau_df = kau_df.dropna(subset=['year_diff'])

    # Split the DataFrame into positive and negative year_diff
    positive_year_diff_df = kau_df[kau_df['year_diff'] > 0]
    negative_year_diff_df = kau_df[kau_df['year_diff'] <= 0]

    # Find the nearest following year within 2 years for each kulturcode
    nearest_following_df = positive_year_diff_df[positive_year_diff_df['year_diff'] <= 2].sort_values('year_diff').drop_duplicates(['kulturcode', 'year_missing_kulturart'])

    # Find the closest previous year for each kulturcode if no nearest following year is found within 2 years
    closest_previous_df = negative_year_diff_df.sort_values('year_diff', ascending=False).drop_duplicates(['kulturcode', 'year_missing_kulturart'])

    # Combine the results
    combined_df = pd.concat([nearest_following_df, closest_previous_df]).drop_duplicates(['kulturcode', 'year_missing_kulturart'], keep='first')

    # Update the 'kulturart' in missing_kulturart based on the nearest year found
    updated_missing_kulturart = missing_kulturart.copy()
    updated_missing_kulturart = updated_missing_kulturart.merge(combined_df[['kulturcode', 'year_missing_kulturart', 'kulturart_kulturcode_map']], 
                                                                left_on=['kulturcode', 'year'], 
                                                                right_on=['kulturcode', 'year_missing_kulturart'], 
                                                                how='left')

    # Fill the 'kulturart' with the found values
    updated_missing_kulturart['kulturart'] = updated_missing_kulturart['kulturart_kulturcode_map'].combine_first(updated_missing_kulturart['kulturart'])

    # Drop the helper columns
    updated_missing_kulturart = updated_missing_kulturart.drop(columns=['year_missing_kulturart', 'kulturart_kulturcode_map'])

    # Save the updated DataFrame to a CSV file (commented out)
    # updated_missing_kulturart.to_csv('reports/Kulturcode/Fixingkulturart/updated_missing_kulturart.csv', encoding='windows-1252', index=False)

    # Rejoin the updated_missing_kulturart with the original DataFrame
    kulturcode_map = pd.concat([kulturcode_map, updated_missing_kulturart], ignore_index=True)
    logging.info(f"Number of rows after rejoining updated_missing_kulturart: {kulturcode_map.shape[0]}")
    logging.info(f"Number of rows with missing 'kulturart' after rejoining: {kulturcode_map[kulturcode_map['kulturart'].isnull()].shape[0]}")

    # print rows with missing 'kulturart'
    missing_kulturart = kulturcode_map[kulturcode_map['kulturart'].isnull()]
    logging.info(f"Missing kulturart:\n{missing_kulturart}")

    # Drop rows still with missing 'kulturart'
    kulturcode_map = kulturcode_map[~kulturcode_map['kulturart'].isnull()]


    return kulturcode_map

def fix_gruppe(kulturcode_map):
    # Identify rows with missing 'Gruppe'
    missing_gruppe = kulturcode_map[kulturcode_map['Gruppe'].isnull()]
    logging.info(f"Number of rows with missing 'Gruppe': {missing_gruppe.shape[0]}")
    # Remove rows with missing 'Gruppe' from the original DataFrame
    kulturcode_map = kulturcode_map[~kulturcode_map['Gruppe'].isnull()]
    logging.info(f"Number of rows after removing missing 'Gruppe': {kulturcode_map.shape[0]}")

    # Merge missing_gruppe with kulturcode_map on 'kulturcode' and 'kulturart'
    merged_df = missing_gruppe.merge(kulturcode_map, on=['kulturcode', 'kulturart'], how='left', suffixes=('_missing_gruppe', '_kulturcode_map'))
    merged_df['year_diff'] = merged_df['year_missing_gruppe'] - merged_df['year_kulturcode_map']
    merged_df = merged_df.dropna(subset=['year_diff'])

    # Function to retain the row with the smallest positive year_diff or the single row
    def retain_row(group):
        if len(group) > 1:
            positive_year_diff = group[group['year_diff'] > 0]
            if not positive_year_diff.empty:
                return positive_year_diff.loc[positive_year_diff['year_diff'].idxmin()]
        return group.iloc[0]

    # Group by 'kulturcode', 'kulturart', and 'year_missing_gruppe' and apply the function
    closest_year_df = merged_df.groupby(['kulturcode', 'kulturart', 'year_missing_gruppe']).apply(retain_row).reset_index(drop=True)

    # Create a dictionary for easy lookup
    closest_year_dict = closest_year_df.set_index(['kulturcode', 'kulturart', 'year_missing_gruppe'])['Gruppe_kulturcode_map'].to_dict()

    # Function to update the 'Gruppe' value in missing_gruppe based on the dictionary
    def update_Gruppe(row):
        return closest_year_dict.get((row['kulturcode'], row['kulturart'], row['year']), None)

    # Apply the function to update missing_gruppe
    missing_gruppe['Gruppe'] = missing_gruppe.apply(update_Gruppe, axis=1)

    # Rejoin the updated missing_gruppe with the original DataFrame
    kulturcode_map = pd.concat([kulturcode_map, missing_gruppe], ignore_index=True)

    # Drop rows still with missing 'Gruppe'
    missing_gruppe = kulturcode_map[kulturcode_map['Gruppe'].isnull()]
    kulturcode_map = kulturcode_map[~kulturcode_map['Gruppe'].isnull()]

    # Update 'Gruppe' based on specific 'kulturcode' values
    missing_gruppe.loc[missing_gruppe['kulturcode'].isin([48, 49, 866, 722, 764]), 'Gruppe'] = 'Sonstige Flächen'
    missing_gruppe.loc[missing_gruppe['kulturcode'].isin([513, 514, 669, 687]), 'Gruppe'] = 'Küchenkräuter/Heil-und Gewürzpflanzen'

    # Rejoin the updated missing_gruppe with the original DataFrame
    kulturcode_map = pd.concat([kulturcode_map, missing_gruppe], ignore_index=True)
    logging.info(f"Number of rows with missing 'Gruppe' after rejoining: {kulturcode_map[kulturcode_map['Gruppe'].isnull()].shape[0]}")

    return kulturcode_map

def review_groups(kulturcode_map):
    # get a list of all unique values in the 'Gruppe' column
    unique_gruppe = kulturcode_map['Gruppe'].unique()
    logging.info(unique_gruppe)

    # Define the mapping for Gruppe values
    gruppe_mapping = {
        'AUKM/GoG': 'AUKM',
        'AUKM/ unproduktive Fläche': 'AUKM',
        'unproduktive Fläche': 'AUKM',
        'Ackerfutter/GoG': 'Ackerfutter',
        'Sonstige  Flächen': 'Sonstige Flächen',
        'Sonstige LF auf AL': 'Sonstige Flächen',
        'Sonstige Flächen': 'Sonstige Flächen',
        'Aufforstung': 'Stilllegung/Aufforstung',
        'Stilllegung/Aufforstung': 'Stilllegung/Aufforstung',
        'Aus der Erzeugung genommen': 'Aus der Produktion genommen',
        'Aus der Produktion/Erzeugung genommen': 'Aus der Produktion genommen',
        'Leg-Mischung': 'Leguminosen'
    }
    
    #Replace the values in the 'Gruppe' column
    kulturcode_map['Gruppe'] = kulturcode_map['Gruppe'].replace(gruppe_mapping)

    # Filter rows where 'Gruppe' starts with 'Küchenkräuter'
    filtered_df = kulturcode_map[kulturcode_map['Gruppe'].str.startswith('Küchenkräuter', na=False)]

    # Remove the rows of the filtered DataFrame from the kulturcode_map
    kulturcode_map = kulturcode_map.drop(filtered_df.index)

    # For all of these rows in filtered_Df, let Gruppe value be 'Küchenkräuter'
    filtered_df['Gruppe'] = 'Kräuter'

    # Add the filtered DataFrame back to the kulturcode_map
    kulturcode_map = pd.concat([kulturcode_map, filtered_df], ignore_index=True)

    # get a list of all unique values in the 'Gruppe' column
    unique_gruppe = kulturcode_map['Gruppe'].unique()
    logging.info(unique_gruppe)
    
    return kulturcode_map

def retain_most_recent_kulturcode(kulturcode_map):
    # Sort by 'kulturcode' and 'year' in descending order
    sorted_df = kulturcode_map.sort_values(by=['kulturcode', 'year'], ascending=[True, False])
    
    # Drop duplicates based on 'kulturcode', keeping the first occurrence
    kulturcode_map = sorted_df.drop_duplicates(subset=['kulturcode'], keep='first')
    
    # rename year column to reference most_recent_year
    kulturcode_map = kulturcode_map.rename(columns={'year': 'sourceyear'})

    return kulturcode_map

def fix_string(kulturcode_map):
    # Correct multiple spaces in the 'text_column'
    kulturcode_map['kulturart'] = kulturcode_map['kulturart'].str.replace(r'\s+', ' ', regex=True)
    kulturcode_map['Gruppe'] = kulturcode_map['Gruppe'].str.replace(r'\s+', ' ', regex=True)
    # Remove leading and trailing whitespaces
    kulturcode_map['kulturart'] = kulturcode_map['kulturart'].str.strip()
    kulturcode_map['Gruppe'] = kulturcode_map['Gruppe'].str.strip()
    # Convert all text to lowercase
    kulturcode_map['kulturart'] = kulturcode_map['kulturart'].str.lower()
    kulturcode_map['Gruppe'] = kulturcode_map['Gruppe'].str.lower()
    
    return kulturcode_map


def check_kulturcode_presence(kulturcode_act, kulturcode_map):
    dropped_codes = {}  # Initialize the dictionary to store missing kulturcodes

    for year, df in kulturcode_act.items():
        if not isinstance(df, pd.DataFrame):
            logging.warning(f"The value for year {year} is not a DataFrame.")
            continue
        
        missing_kulturcodes = df[~df['kulturcode'].isin(kulturcode_map['kulturcode'])]
        
        if not missing_kulturcodes.empty:
            logging.info(f"Year {year}: The following kulturcodes are missing in kulturcode_map:\n{missing_kulturcodes}")
            # Add the year column to the missing_kulturcodes DataFrame
            missing_kulturcodes['year'] = year
            # Save the missing kulturcodes to the dictionary
            dropped_codes[year] = missing_kulturcodes
        else:
            logging.info(f"Year {year}: All kulturcodes are present in kulturcode_map.")

    return dropped_codes

def save_missing_kulturcodes_to_excel(dropped_codes):
   output_file='reports/Kulturcode/missing_kulturcodes.xlsx'
   with pd.ExcelWriter(output_file) as writer:
        for year, df in dropped_codes.items():
            # Save the missing kulturcodes to a sheet in the Excel file
            df.to_excel(writer, sheet_name=f'Missing_{year}', index=False)


def collect_unique_dropped_codes(dropped_codes):
    # Initialize an empty list to store rows
    rows = []

    # Iterate through each year and DataFrame in dropped_codes
    for year, df in dropped_codes.items():
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            # Append the row to the list, including the year
            rows.append(row)

    # Convert the list to a DataFrame
    unique_dropped_codes_df = pd.DataFrame(rows).drop_duplicates()

    return unique_dropped_codes_df


def assign_gruppe_and_kulturart(row):
    kulturcode = row['kulturcode']
    if kulturcode < 200:
        row['Gruppe'] = 'getreide'
        row['kulturart'] = 'getreide'
    elif 200 <= kulturcode < 500:
        row['Gruppe'] = 'ackerfutter'
        row['kulturart'] = 'ackerfutter'
    elif 500 <= kulturcode < 600:
        row['Gruppe'] = 'stilllegung/aufforstung'
        row['kulturart'] = 'stilllegung/aufforstung'
    elif 600 <= kulturcode < 700:
        row['Gruppe'] = 'kräuter'
        row['kulturart'] = 'kräuter'
    elif 700 <= kulturcode < 730:
        row['Gruppe'] = 'andere handelsgewächse'
        row['kulturart'] = 'andere handelsgewächse'
    elif 730 <= kulturcode < 800:
        row['Gruppe'] = 'zierpflanzen'
        row['kulturart'] = 'zierpflanzen'
    elif 800 <= kulturcode < 900:
        row['Gruppe'] = 'dauerkulturen'
        row['kulturart'] = 'dauerkulturen'
    elif kulturcode >= 900:
        row['Gruppe'] = 'sonstige flächen'
        row['kulturart'] = 'sonstige flächen'
        
    return row

def add_codes_fromb4_15(unique_dropped_codes_df, kulturcode_map1):
    # Assign 'Gruppe' and 'kulturart' values to dropped codes
    unique_dropped_codes_df = unique_dropped_codes_df.apply(assign_gruppe_and_kulturart, axis=1)

    #let 'sourceyear' value for all rows be 2015 and drop year column
    unique_dropped_codes_df['sourceyear'] = 2015
    unique_dropped_codes_df = unique_dropped_codes_df.drop(columns='year')
    
    # group by 'kulturcode' and drop duplicates
    unique_dropped_codes_df = unique_dropped_codes_df.drop_duplicates(subset='kulturcode')
    logging.info(f"Unique dropped codes have been assigned 'Gruppe', 'kulturart' and 'sourceyear' values.")

    # Merge the unique_dropped_codes_df with kulturcode_map1 on 'kulturcode' and 'year'
    kulturcode_map2 = pd.concat([kulturcode_map1, unique_dropped_codes_df], ignore_index=True)
    
    return kulturcode_map2

def category1(df, column_name):
    # Define the categories that should be labeled as 'environmental'
    environmental_categories = [
        'stilllegung/aufforstung', 
        'greening / landschaftselemente', 
        'aukm', 
        'aus der produktion genommen'
    ]

    # Create column 'category1' based on the conditions
    df['category1'] = df[column_name].apply(
        lambda x: 'environmental' if x in environmental_categories else 'others'
    )
    
    return df

def category2(df, column_name):
    # Define the categories that should be labeled as 'environmental'
    environmental_categories = [
        'stilllegung/aufforstung', 
        'greening / landschaftselemente', 
        'aukm', 
        'aus der produktion genommen'
    ]
    
    sonstige_flaechen = [
        'sonstige flächen',
        'andere handelsgewächse',
        'zierpflanzen',
        'kräuter',
        'energiepflanzen'
    ]

    leguminosen = [
        'leguminosen',
        'eiweißpflanzen'
    ]

    # Create column 'category2'
    df['category2'] = df[column_name].apply(
        lambda x: 'environmental' if x in environmental_categories \
            else 'sonstige flächen' if x in sonstige_flaechen \
            else 'leguminosen' if x in leguminosen \
            else x
    )
    
    return df

def category3(df, column_name):
    # categorization based on similar use of crops in reality
    environmental_categories = [
        'stilllegung/aufforstung', 
        'greening / landschaftselemente', 
        'aukm', 
        'aus der produktion genommen'
    ]
    
    ffc = [
        'getreide',
        'gemüse',
        'leguminosen',
        'eiweißpflanzen',
        'hackfrüchte',
        'ölsaaten',
        'kräuter',
        'ackerfutter'
    ]
        
    others = [
        'sonstige flächen',
        'andere handelsgewächse',
        'zierpflanzen',
        'mischkultur',
        'energiepflanzen'
    ]

    # Create column 'category2'
    df['category3'] = df[column_name].apply(
        lambda x: 'environmental' if x in environmental_categories \
            else 'ffc' if x in ffc \
            else 'others' if x in others \
            else x
    )
    
    return df

def process_kulturcode():
    # Define the path to the CSV file
    csv_path = 'data/interim/kulturcode_mastermap.csv'
    
    # Check if the CSV file already exists
    if os.path.exists(csv_path):
        print("Loading kulturcode_mastermap from CSV.")
        kulturcode_mastermap = pd.read_csv(csv_path, encoding='windows-1252')
    else:
        print("Processing kulturcode_mastermap.")
        # Load base data
        gld = dl.load_data(loadExistingData=True)
        
        kulturcode_act = get_uni_kulturcode(gld)
        plot_unique_kulturcode_counts(kulturcode_act)
        kulturart = load_kulturcode_description("N:/ds/data/Niedersachsen/kulturcode/kulturart_allyears.xlsx")
        kulturcode_map = create_kulturcode_map(kulturcode_act, kulturart)
        kulturcode_map = map_2021_22(kulturcode_map)
        kulturcode_map = fix_missingkulturart(kulturcode_map)
        kulturcode_map = fix_gruppe(kulturcode_map)
        kulturcode_map = review_groups(kulturcode_map)
        kulturcode_map1 = retain_most_recent_kulturcode(kulturcode_map)
        kulturcode_map1 = fix_string(kulturcode_map1)
        dropped_codes = check_kulturcode_presence(kulturcode_act, kulturcode_map1)
        unique_dropped_codes_df = collect_unique_dropped_codes(dropped_codes)
        kulturcode_mastermap = add_codes_fromb4_15(unique_dropped_codes_df, kulturcode_map1)
        check_kulturcode_presence(kulturcode_act, kulturcode_mastermap)
        kulturcode_mastermap = category1(kulturcode_mastermap, 'Gruppe')
        kulturcode_mastermap = category2(kulturcode_mastermap, 'Gruppe')
        kulturcode_mastermap = category3(kulturcode_mastermap, 'Gruppe')
        
        # Save to CSV
        kulturcode_mastermap.to_csv(csv_path, encoding='windows-1252', index=False)
    
    return kulturcode_mastermap

# %%
if __name__ == '__main__':
   kulturcode_mastermap = process_kulturcode()
# %%
