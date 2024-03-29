import pandas as pd

from pysrc.utils.is_non_zero_file import is_non_zero_file
from ..models.column_statistics import ColumnStatistic
from ..models.column_information import ColumnInformation


def file_column_statistics(file_path: str, *, header: bool, delimiter: str = ';', escapechar: str = '\\',
                           is_baseline: bool = False) -> list[ColumnStatistic]:
    descriptions: list[ColumnStatistic] = []
    df = pd.read_csv(file_path, delimiter=delimiter, escapechar=escapechar, dtype=str, on_bad_lines='skip',
                     header='infer' if header else None) if is_non_zero_file(file_path) else pd.DataFrame(dtype='str')
    
    for col_index, column in enumerate(df.columns):
        column_series: pd.Series = df[column].copy(deep=True)
        column_name: str = ''
        if header:
            column_name = column_series.iloc[0]
            column_series = column_series.iloc[1:]
        
        column_series.sort_values(inplace=True, ascending=True)
        count = column_series.count()
        unique_count = column_series.nunique()
        minimum = column_series.iloc[0]
        maximum = column_series.iloc[count-1]
        
        column_series.index = column_series.str.len()  # type: ignore
        column_series.sort_index(inplace=True)
        shortest = column_series.iloc[0]
        longest = column_series.iloc[count-1]
        
        file_name = file_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        table_name = file_name.rsplit('__', 1)[0]
        if is_baseline and not header:
            column_name = 'column' + str(col_index + 1)
        elif not header:
            column_name = 'column' + file_name.rsplit('_', 1)[-1]

        col_stats = ColumnStatistic(
            column_information=ColumnInformation(
                table_name=table_name,
                column_name=column_name),
            count=int(count), 
            unique_count=int(unique_count), 
            unique_ratio=round(unique_count / count, 5),
            min=minimum,
            max=maximum,
            shortest=shortest,
            longest=longest)
        descriptions.append(col_stats)
        
    return descriptions    
