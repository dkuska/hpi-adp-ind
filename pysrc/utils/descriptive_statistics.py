import pandas as pd
import numpy as np

def file_column_statistics(file_path, delimiter=';'):
    descriptions = []
    df = pd.read_csv(file_path, delimiter=delimiter, dtype=str, on_bad_lines='skip')
    
    for column in df.columns:
        column_series = df[column]
        
        count = column_series.count()
        unique_count = len(column_series.unique())
        
        descriptions.append({
            'count' : count,
            'unique_count': unique_count,
            'unique_ratio': round(unique_count / count, 5)
        })
        
    return descriptions    
