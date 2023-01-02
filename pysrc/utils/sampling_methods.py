import math
import random
import pandas as pd
from typing import Callable

def replaceEmptyField(series: pd.Series) -> pd.Series:
    return series.replace(r'\^s*$', float('NaN'), regex=True)


def random_sample(data: list[list[str]],
                  num_samples: int,
                  num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    return tmp.sample(n=num_samples)


def first_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    return tmp.iloc[:num_samples]

def smallest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    out = tmp.sort_values()
    return out.iloc[:num_samples]

def biggest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    out = tmp.sort_values(ascending=False)

    return out.iloc[:num_samples]

def longest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp.index = tmp.str.len()
    df = tmp.sort_index(ascending=False).reset_index(drop=True)

    return df.iloc[:num_samples]

def shortest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp.index = tmp[0].str.len()
    df = tmp.sort_index(ascending=True).reset_index(drop=True)

    return df.iloc[:num_samples]

def all_distinct_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    df = tmp.loc[tmp.astype(str).drop_duplicates().index]


    return df.iloc[:num_samples]

def evenly_spaced_sample(data: list[list[str]],
                         num_samples: int,
                         num_entries: int) -> pd.Series:
    space_width = math.ceil(num_entries / num_samples)
    starting_index = random.randint(0, space_width)
    tmp = pd.Series(data)
    out_indices = [i % num_entries for i in range(starting_index, num_entries + space_width, space_width)]
    out = tmp.iloc[out_indices]
    return out


sampling_methods_dict: dict[str,
                            Callable[[list[list[str]], int, int],
                                     list[str]]] = {
    'random': random_sample,
    'first': first_sample,
    'evenly-spaced': evenly_spaced_sample,
    'smallest-value': smallest_value_sample,
    'biggest-value': biggest_value_sample,
    'longest-value': longest_value_sample,
    'all-distinct': all_distinct_sample
}
