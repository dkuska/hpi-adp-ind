import os, csv
import pandas as pd
import numpy as np

mode = ['sampling', 'describe'][1]
source_dir = 'src/'


def csv_description(file_path, seperator=';', header=None):
    descriptions = []
    print(file_path)
    df = pd.read_csv(file_path, sep=seperator, header=header, on_bad_lines='skip')
    
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()
    print(f'# columns    : {len(df.columns)}')
    print(f'# num columns: {len(numeric_columns)}')
    print(f'# cat columns: {len(categorical_columns)}')
    
    ### TODO: Frequency distributions
    for col in numeric_columns:
        column = df[col]
        
        descr = df[col].describe()
        descr['dtype'] = column.dtype
        descr['name '] = col
        num_uniques = len(column.unique())
        descr.at['unique'] = num_uniques
        descr.at['unique_ratio'] = num_uniques / column.size
        
        descr.sort_index()
        descriptions.append(descr)
        
    for col in categorical_columns:
        column = df[col]
        
        descr = df[col].describe()
        descr['dtype'] = column.dtype
        descr['name '] = col
        num_uniques = len(column.unique())
        descr.at['unique_ratio'] = num_uniques / column.size
        
        descr.sort_index()
        descriptions.append(descr)
    
    combined_stats = pd.concat(descriptions, axis=1)
    # combined_stats.sort_index(axis=0, inplace=True)
    combined_stats.sort_index(axis=1, inplace=True)

    print(combined_stats)
    
def describe_src_files():
    source_files = [os.path.join(os.getcwd(), source_dir, f) for f in os.listdir(os.path.join(os.getcwd(), source_dir)) if f.rsplit('.')[1] == 'csv']    
    for src_file in source_files:
        csv_description(src_file)
        
if __name__ == "__main__":
    if mode == 'describe':
        describe_src_files()