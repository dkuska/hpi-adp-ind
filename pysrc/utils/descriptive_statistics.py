import pandas as pd
import numpy as np

def file_column_statistics(file_path, delimiter=';', escapechar='\\'):
    descriptions = []
    df = pd.read_csv(file_path, delimiter=delimiter, escapechar=escapechar, dtype=str, on_bad_lines='skip')
    
    for column in df.columns:
        column_series = df[column].copy(deep=True)
        
        column_series.sort_values(inplace=True, ascending=True)
        count = column_series.count()
        unique_count = len(column_series.unique())
        min = column_series.iloc[0]
        max = column_series.iloc[count-1]
        
        column_series.index = column_series.str.len()
        column_series.sort_index(inplace=True)
        shortest = column_series.iloc[0]
        longest = column_series.iloc[count-1]
        
        descriptions.append({
            'count' : count,
            'unique_count': unique_count,
            'unique_ratio': round(unique_count / count, 5),
            'min': min,
            'max': max,
            'shortest': shortest,
            'longest': longest,
        })
        
    return descriptions    
