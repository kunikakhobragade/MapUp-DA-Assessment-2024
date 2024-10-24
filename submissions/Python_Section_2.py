#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Question 9

import pandas as pd
import numpy as np

def calculate_distance_matrix(df) -> pd.DataFrame:
   
    # Create a set of unique IDs to form the matrix
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    unique_ids.sort()  # Sort the IDs for better matrix readability

    # Initialize the distance matrix with infinite values (since not all routes are directly connected)
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

    # Fill the diagonal with 0 (distance from any location to itself is zero)
    np.fill_diagonal(distance_matrix.values, 0)

    # Fill the matrix with known distances from the dataset
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        dist = row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist  # Ensure symmetry

    # Implement the Floyd-Warshall algorithm to compute all pairs' shortest paths
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j],
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )

    return distance_matrix

# Example usage:
df = pd.read_csv('dataset-2.csv')
result = calculate_distance_matrix(df)
print(result)


# In[5]:


# Question 10

import pandas as pd

def unroll_distance_matrix(df) -> pd.DataFrame:
   
    # Unroll the matrix into three columns: id_start, id_end, and distance
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    # Remove rows where id_start is equal to id_end (no self-distances)
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    # Reset the index for a clean output
    unrolled_df.reset_index(drop=True, inplace=True)

    return unrolled_df

# Example usage:
df_matrix = calculate_distance_matrix(df)  # Using the function from Question 9
unrolled_result = unroll_distance_matrix(df_matrix)
print(unrolled_result)


# In[6]:


# Question 11

import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    
    # Calculate the average distance for the reference_id
    ref_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    # Calculate the 10% threshold
    lower_threshold = ref_avg_distance * 0.9
    upper_threshold = ref_avg_distance * 1.1

    # Group by 'id_start' and calculate the average distance for each ID
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()

    # Filter the IDs whose average distance lies within the 10% threshold
    result_df = avg_distances[
        (avg_distances['distance'] >= lower_threshold) & 
        (avg_distances['distance'] <= upper_threshold)
    ]

    # Sort the results by 'id_start'
    result_df = result_df.sort_values(by='id_start')

    return result_df

# Example usage:
df_unrolled = unroll_distance_matrix(df_matrix)  # From the previous question
reference_id = 1001406  # Example reference ID
result = find_ids_within_ten_percentage_threshold(df_unrolled, reference_id)
print(result)


# In[7]:


# Question 12

import pandas as pd

def calculate_toll_rate(df) -> pd.DataFrame:
    
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Create new columns by multiplying distance with each rate coefficient
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage:
df_unrolled = unroll_distance_matrix(df_matrix)  # From the previous question
result = calculate_toll_rate(df_unrolled)
print(result)


# In[11]:


# Question 13

import pandas as pd
import datetime

def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    
    # Initialize a list to hold the new rows
    new_rows = []

    # Define the days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    # Define time intervals for weekdays
    weekday_time_ranges = [
        (datetime.time(0, 0), datetime.time(10, 0), 0.8),  # 00:00:00 to 10:00:00
        (datetime.time(10, 0), datetime.time(18, 0), 1.2),  # 10:00:00 to 18:00:00
        (datetime.time(18, 0), datetime.time(23, 59, 59), 0.8)  # 18:00:00 to 23:59:59
    ]

    # Define the weekend discount factor
    weekend_discount_factor = 0.7

    # Loop through all unique (id_start, id_end) pairs in the DataFrame
    for _, row in df.iterrows():
        # Handle weekdays
        for day in weekdays:
            for start_time, end_time, discount_factor in weekday_time_ranges:
                # Create a copy of the current row
                row_copy = row.copy()
                row_copy['start_day'] = day
                row_copy['end_day'] = day
                row_copy['start_time'] = start_time
                row_copy['end_time'] = end_time
                
                # Apply the discount factor for each vehicle type
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    row_copy[vehicle] = row[vehicle] * discount_factor
                
                # Add the modified row to the new rows list
                new_rows.append(row_copy)

        # Handle weekends
        for day in weekends:
            row_copy = row.copy()
            row_copy['start_day'] = day
            row_copy['end_day'] = day
            row_copy['start_time'] = datetime.time(0, 0)  # Full day start
            row_copy['end_time'] = datetime.time(23, 59, 59)  # Full day end

            # Apply the weekend discount factor
            for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                row_copy[vehicle] = row[vehicle] * weekend_discount_factor
            
            # Add the modified row to the new rows list
            new_rows.append(row_copy)

    # Convert the list of new rows into a DataFrame
    time_based_df = pd.DataFrame(new_rows)

    return time_based_df

# Example usage:
df_unrolled = calculate_toll_rate(df)  # Replace this with your DataFrame from Question 12
result = calculate_time_based_toll_rates(df_unrolled)
print(result)

