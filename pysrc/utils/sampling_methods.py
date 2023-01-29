import math
import random
import pandas as pd
from typing import Callable

def preProcessData(series: pd.Series) -> pd.Series:
    tmp = series.replace(r'\^s*$', float('NaN'), regex=True)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]
    return tmp

def random_sample(data: list[list[str]],
                  num_samples: int,
                  num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    if len(tmp) >= num_samples:
        return tmp.sample(n=num_samples)
    else:
        return tmp

def first_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    if len(tmp) >= num_samples:
        return tmp.iloc[:num_samples]
    else:
        return tmp

def smallest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    grouped = tmp.groupby(tmp.str.len(), group_keys=False)

    out = grouped.apply(lambda x: x.sort_values(ascending=True))
    if len(out) >= num_samples:
        return out.iloc[:num_samples]
    else:
        return out
def biggest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    grouped = tmp.groupby(tmp.str.len(), group_keys=False)

    out = grouped.apply(lambda x: x.sort_values(ascending=False))
    if len(out) >= num_samples:
        return out.iloc[:num_samples]
    else:
        return out
def longest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    tmp.index = tmp.str.len()
    df = tmp.sort_index(ascending=False).reset_index(drop=True)
    if len(df) >= num_samples:
        return df.iloc[:num_samples]
    else:
        return df

def shortest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)

    tmp.index = tmp[0].str.len()
    df = tmp.sort_index(ascending=True).reset_index(drop=True)
    if len(df) >= num_samples:
        return df.iloc[:num_samples]
    else:
        return df


def evenly_spaced_sample(data: list[list[str]],
                         num_samples: int,
                         num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = preProcessData(tmp)
    space_width = math.ceil(len(tmp) / num_samples)
    starting_index = random.randint(0, space_width-1)
    out_indices = [i % len(tmp) for i in range(starting_index, len(tmp), space_width)]

    out = tmp.iloc[out_indices]
    return out


sampling_methods_dict: dict[str,
                            Callable[[list[list[str]], int, int],
                                     pd.Series]] = {
    'random': random_sample,
    'first': first_sample,
    'evenly-spaced': evenly_spaced_sample,
    'smallest-value': smallest_value_sample,
    'biggest-value': biggest_value_sample,
    'longest-value': longest_value_sample
}
