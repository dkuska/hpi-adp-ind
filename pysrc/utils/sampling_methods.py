import math
import random
import pandas as pd
from typing import Callable


def random_sample(data: list[list[str]],
                  num_samples: int,
                  num_entries: int) -> list[str]:
    return random.sample(data[0], k=num_samples)


def first_sample(data: list[list[str]],
                 num_samples: int,
                 num_entries: int) -> list[str]:
    return data[0][:num_samples]

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


def evenly_spaced_sample(data: list[list[str]],
                         num_samples: int,
                         num_entries: int) -> list[str]:
    space_width = math.ceil(num_entries / num_samples)
    starting_index = random.randint(0, space_width)
    return [
        data[0][i % num_entries]
        for i
        in range(starting_index, num_entries+space_width, space_width)]


sampling_methods_dict: dict[str,
                            Callable[[list[list[str]], int, int],
                                     list[str]]] = {
    'random': random_sample,
    'first': first_sample,
    'evenly-spaced': evenly_spaced_sample,
    'smallest-value': smallest_value_sample,
}
