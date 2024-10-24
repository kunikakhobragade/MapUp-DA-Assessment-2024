#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import Dict, List
import pandas as pd


# In[4]:


# Question 1

from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i + n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    return result

print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output should be: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))  # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output should be: [40, 30, 20, 10, 70, 60, 50]


# In[5]:


# Question 2

from typing import Dict, List

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result = {}
    for word in lst:
        length = len(word)
        if length not in result:
            result[length] = []
        result[length].append(word)
    return dict(sorted(result.items()))

print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output should be: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output should be: {3: ['one', 'two'], 4: ['four'], 5: ['three']}


# In[6]:


# Question 3

from typing import Dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    def flatten(current_dict, parent_key=""):
        items = []
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(flatten({f"{new_key}[{i}]": item}).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten(nested_dict)

nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dict(nested_dict))
# Output:
# {
#   'road.name': 'Highway 1',
#   'road.length': 350,
#   'road.sections[0].id': 1,
#   'road.sections[0].condition.pavement': 'good',
#   'road.sections[0].condition.traffic': 'moderate'
# }


# In[8]:


# Question 4

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(path, counter):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for num in counter:
            if counter[num] > 0:
                # Choose the current number and reduce its count
                path.append(num)
                counter[num] -= 1
                # Explore further with the updated path and counter
                backtrack(path, counter)
                # Backtrack and undo the choice
                path.pop()
                counter[num] += 1

    result = []
    # Use a counter to track frequency of each number
    counter = {num: nums.count(num) for num in set(nums)}
    # Start the backtracking process
    backtrack([], counter)
    return result

print(unique_permutations([1, 1, 2]))
# Output should be:
# [
#     [1, 1, 2],
#     [1, 2, 1],
#     [2, 1, 1]
# ]


# In[9]:


# Question 5

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))
# Output should be: ['23-08-1994', '08/23/1994', '1994.08.23']


# In[12]:


get_ipython().system('pip install polyline')


# In[14]:


# Question 6

import polyline
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coords = polyline.decode(polyline_str)
    data = []
    for i in range(len(coords)):
        lat, lon = coords[i]
        if i == 0:
            distance = 0
        else:
            lat_prev, lon_prev = coords[i - 1]
            distance = haversine(lat_prev, lon_prev, lat, lon)
        data.append([lat, lon, distance])
    return pd.DataFrame(data, columns=['latitude', 'longitude', 'distance'])

polyline_str = '_p~iF~ps|U_ulLnnqC_mqNvxq`@'
df = polyline_to_dataframe(polyline_str)
print(df)


# In[15]:


# Question 7


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    # Step 1: Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    
    # Transpose the matrix and reverse rows
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Step 2: Replace each element with the sum of all elements in its row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Row sum excluding current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            
            # Column sum excluding current element
            col_sum = sum(rotated_matrix[x][j] for x in range(n)) - rotated_matrix[i][j]
            
            # Set the value to row sum + column sum
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix

# Example usage
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)


# In[22]:


# Question 8

import pandas as pd

# Mapping of weekdays to a dummy week starting on Monday
week_map = {
    'Monday': '2024-01-01',
    'Tuesday': '2024-01-02',
    'Wednesday': '2024-01-03',
    'Thursday': '2024-01-04',
    'Friday': '2024-01-05',
    'Saturday': '2024-01-06',
    'Sunday': '2024-01-07'
}

def time_check(df: pd.DataFrame) -> pd.Series:
    # Define valid days and full time range
    valid_days = set(week_map.keys())
    
    # Create a list for results
    results = []

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    for (id_val, id_2_val), group in grouped:
        # Collect unique days
        days = set(group['startDay']).union(set(group['endDay']))
        
        # Generate a complete time range for the current group
        all_times = pd.concat([
            pd.Series(pd.date_range(
                start=f"{week_map[row['startDay']]} {row['startTime']}", 
                end=f"{week_map[row['endDay']]} {row['endTime']}", freq='S')) 
            for _, row in group.iterrows()
        ])
        
        # Check for required days and time coverage
        has_all_days = valid_days.issubset(days)
        covers_full_time = all_times.min().time() <= pd.to_datetime("00:00:00").time() and                            all_times.max().time() >= pd.to_datetime("23:59:59").time()
        
        # Append result as (id, id_2, is_incomplete)
        results.append((id_val, id_2_val, not (has_all_days and covers_full_time)))

    # Convert results to a MultiIndex Series
    multi_index_results = pd.Series(dict(((id_val, id_2_val), is_incomplete) for id_val, id_2_val, is_incomplete in results))
    
    return multi_index_results

# Example usage
df = pd.read_csv('dataset-1.csv')
result = time_check(df)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




