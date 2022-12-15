import math
import random
import pandas as pd
from typing import Callable


def random_sample(data: list[list[str]],
                  num_samples: int,
                  num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp = tmp.replace(r'^s*$', float('NaN'), regex=True)
    tmp = tmp.dropna(inplace=True)

    return tmp.sample(n=num_samples)


def first_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])

    return tmp.iloc[:num_samples]

#TODO Alle funktionen auf DataFrames umstellen sortieren und danach Empty Strings löschen führt zu mglw leereb Samples
def smallest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp = tmp.replace(r'^s*$', float('NaN'), regex=True)
    tmp.dropna(inplace=True)
    for col in tmp.columns:
        out = tmp.sort_values(by=col)

    return out.iloc[:num_samples]

def biggest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp = tmp.replace(r'^s*$', float('NaN'), regex=True)
    tmp.dropna(inplace=True)
    for col in tmp.columns:
        out = tmp.sort_values(by=col, ascending=False)

    return out.iloc[:num_samples]

def longest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp = tmp.replace(r'^s*$', float('NaN'), regex=True)
    tmp.dropna(inplace=True)
    tmp.index = tmp[0].str.len()
    df = tmp.sort_index(ascending=False).reset_index(drop=True)

    return df.iloc[:num_samples]

def shortest_value_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp = tmp.replace(r'^s*$', float('NaN'), regex=True)
    tmp.dropna(inplace=True)
    tmp.index = tmp[0].str.len()
    df = tmp.sort_index(ascending=True).reset_index(drop=True)

    return df.iloc[:num_samples]

#TODO rework that
def all_distinct_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    tmp = pd.DataFrame(data[0])
    tmp.loc[tmp.astype(str).drop_duplicates().index]
    return tmp
def evenly_spaced_sample(data: list[list[str]],
                         num_samples: int,
                         num_entries: int) -> list[str]:
    space_width = math.ceil(num_entries / num_samples)
    starting_index = random.randint(0, space_width)
    tmp = pd.DataFrame(data[0])
    return [
        tmp.iloc[i % num_entries]
        for i
        in range(starting_index, num_entries+space_width, space_width)]


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
