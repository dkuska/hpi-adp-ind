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
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    if len(tmp)-1 > num_samples:
        return tmp.sample(n=num_samples)
    else:
        return tmp

def first_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    if len(tmp) - 1 > num_samples:
        return tmp.iloc[:num_samples]
    else:
        return tmp

def smallest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    out = tmp.sort_values()
    if len(out) - 1 > num_samples:
        return out.iloc[:num_samples]
    else:
        return out
def biggest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    out = tmp.sort_values(ascending=False)
    if len(out) - 1 > num_samples:
        return out.iloc[:num_samples]
    else:
        return out
def longest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    tmp.index = tmp.str.len()
    df = tmp.sort_index(ascending=False).reset_index(drop=True)
    if len(df) - 1 > num_samples:
        return df.iloc[:num_samples]
    else:
        return df

def shortest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]

    tmp.index = tmp[0].str.len()
    df = tmp.sort_index(ascending=True).reset_index(drop=True)
    if len(df) - 1 > num_samples:
        return df.iloc[:num_samples]
    else:
        return df

#Probably the same as first now and therefore obsolete?
def all_distinct_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    df = tmp.loc[tmp.astype(str).drop_duplicates().index]
    return df.iloc[:num_samples]

#only method were I had to make it dependent on sample size after deduplicating
#no idea how to keep it there because at some points it tries to sample from spaces that doesn't exist
#doesnt't work?
def evenly_spaced_sample(data: list[list[str]],
                         num_samples: int,
                         num_entries: int) -> pd.Series:
    tmp = pd.Series(data)
    tmp = replaceEmptyField(tmp)
    tmp.dropna(inplace=True)
    tmp = tmp.loc[tmp.astype(str).drop_duplicates().index]
    rate = num_samples / num_entries
    num_samples_here = math.ceil(len(tmp) * rate)
    space_width = math.ceil(len(tmp) / num_samples_here)
    starting_index = random.randint(0, space_width)
    out_indices = [i % num_entries for i in range(starting_index, num_samples_here + space_width - 1, space_width)]

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
    'longest-value': longest_value_sample,
    'all-distinct': all_distinct_sample
}
