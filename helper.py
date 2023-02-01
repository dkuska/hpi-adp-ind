from dataclasses import dataclass
import os
import sys
import pandas as pd
from typing import Literal
import math
import sys
from typing import Literal
from matplotlib import axes, pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class Data:
    threshold: float
    precision: float
    recall: float


@dataclass
class Config:
    dataset: str
    approach: str
    budget: int
    knowledge: str


@dataclass
class File:
    data: list[Data]
    config: Config


@dataclass
class ByApproach:
    logreg: File
    forest: File
    naive: File
    tree: File


def parse_title(title: str) -> Config:
    by_underscore = title.split('_')
    dataset = by_underscore[0].split(os.sep)[-1]
    approach = by_underscore[1]
    budget = int(by_underscore[2])
    knowledge = by_underscore[3].split('.')[0]
    config = Config(dataset=dataset, approach=approach, budget=budget, knowledge=knowledge)
    return config


def parse_contents(raw_string: str) -> list[Data]:
    entries = raw_string.split(' For')
    data: list[Data] = []
    for entry in entries:
        try:
            by_equal_sign = entry.split('=')
            assert len(by_equal_sign) == 9
            with_threshold = by_equal_sign[1]
            threshold = float(with_threshold.split(':', 1)[0])
            with_precision = by_equal_sign[7]
            precision = float(with_precision.split(',', 1)[0])
            with_recall = by_equal_sign[8]
            recall = float(with_recall)
            data_point = Data(threshold=threshold, precision=precision, recall=recall)
            data.append(data_point)
        except Exception as ex:
            print(f'{entry=}')
            raise ex
    return data


def group_by_approach(data: list[File]) -> list[ByApproach]:
    mapping: dict[tuple[str, int, str], list[File]] = {}
    for entry in data:
        key = (entry.config.dataset, entry.config.budget, entry.config.knowledge)
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(entry)
    by_approach: list[ByApproach] = []
    for value in mapping.values():
        try:
            logreg = next(entry for entry in value if entry.config.approach == 'LOGREG')
            forest = next(entry for entry in value if entry.config.approach == 'FOREST')
            naive = next(entry for entry in value if entry.config.approach == 'NAIVE')
            tree = next(entry for entry in value if entry.config.approach == 'TREE')
            this = ByApproach(logreg=logreg, forest=forest, naive=naive, tree=tree)
            by_approach.append(this)
        except Exception as ex:
            for val in value:
                print(val.config)
            raise ex
    return by_approach


def make_plot(by_approach: ByApproach, dir: str) -> str:
    thresholds = [data.threshold for data in by_approach.naive.data]
    logreg_precision = [data.precision for data in by_approach.logreg.data]
    forest_precision = [data.precision for data in by_approach.forest.data]
    naive_precision = [data.precision for data in by_approach.naive.data]
    tree_precision = [data.precision for data in by_approach.tree.data]
    logreg_recall = [data.recall for data in by_approach.logreg.data]
    forest_recall = [data.recall for data in by_approach.forest.data]
    naive_recall = [data.recall for data in by_approach.naive.data]
    tree_recall = [data.recall for data in by_approach.tree.data]
    if by_approach.forest.config.dataset == 'TCPH' and by_approach.forest.config.budget == 30_000 and by_approach.forest.config.knowledge == 'HIGHKNOWLEDGE':
        print(forest_precision)
    # print(f'{len(logreg_precision)=}, {len(forest_precision)=}, {len(naive_precision)=}, {len(tree_precision)=}, {len(logreg_recall)=}, {len(forest_recall)=}, {len(naive_recall)=}, {len(tree_recall)=}')
    data_precision: dict[Literal['Threshold'] | Literal['LOGREG'] | Literal['FOREST'] | Literal['NAIVE'] | Literal['TREE'], list[float]]= {
        'Threshold': thresholds,
        'LOGREG': logreg_precision,
        'FOREST': forest_precision,
        'NAIVE': naive_precision,
        'TREE': tree_precision
    }
    data_recall: dict[Literal['Threshold'] | Literal['LOGREG'] | Literal['FOREST'] | Literal['NAIVE'] | Literal['TREE'], list[float]] = {
        'Threshold': thresholds,
        'LOGREG': logreg_recall,
        'FOREST': forest_recall,
        'NAIVE': naive_recall,
        'TREE': tree_recall
    }
    data_f1: dict[Literal['Threshold'] | Literal['LOGREG'] | Literal['FOREST'] | Literal['NAIVE'] | Literal['TREE'], list[float]] = {
        'Threshold': thresholds,
        'LOGREG': [],
        'FOREST': [],
        'NAIVE': [],
        'TREE': [],
    }
    methods: list[Literal['LOGREG'] | Literal['FOREST'] | Literal['NAIVE'] | Literal['TREE']] = ['LOGREG', 'FOREST', 'NAIVE', 'TREE']
    for i in range(len(thresholds)):
        for method in methods:
            precision = data_precision[method][i]
            recall = data_recall[method][i]
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
            data_f1[method].append(f1)
    frame_precision = pd.DataFrame(data=data_precision)
    frame_recall = pd.DataFrame(data=data_recall)
    frame_f1 = pd.DataFrame(data=data_f1)
    f: Figure
    ax_precision: axes.Axes
    ax_recall: axes.Axes
    ax_f1: axes.Axes
    f, axs = plt.subplots(1, 3, figsize=(30, 20))
    ax_precision = axs[0]
    ax_recall = axs[1]
    ax_f1 = axs[2]
    plot_recall = sns.lineplot(data=pd.melt(frame_precision, ['Threshold'], var_name='Method', value_name='Precision'), x='Threshold', y='Precision', hue='Method', ax=ax_precision)
    plot_recall.set_title('Precision')
    plot_precision = sns.lineplot(data=pd.melt(frame_recall, ['Threshold'], var_name='Method', value_name='Recall'), x='Threshold', y='Recall', hue='Method', ax=ax_recall)
    plot_precision.set_title('Recall')
    plot_f1 = sns.lineplot(data=pd.melt(frame_f1, ['Threshold'], var_name='Method', value_name='F1-Score'), x='Threshold', y='F1-Score', hue='Method', ax=ax_f1)
    plot_f1.set_title('F1-Score')
    f.suptitle(f'Dataset: {by_approach.tree.config.dataset}. Budget: {by_approach.tree.config.budget}. {by_approach.tree.config.knowledge}.', fontsize=36)
    # sample_count = np.around(np.logspace(math.log10(1),math.log10(10),6))
    # plot.set(yscale='log')
    # plot.set(yticks=sample_count)
    # plot.set(yticklabels=sample_count)
    plot_path = os.path.join(dir, f'{by_approach.tree.config.dataset}_{by_approach.tree.config.budget}_{by_approach.tree.config.knowledge}.jpg')
    f.savefig(plot_path)
    return plot_path


def main() -> None:
    plt.rcParams.update({'font.size': 22})
    directory_with_files = sys.argv[1]
    files = [os.path.join(directory_with_files, file) for file in os.listdir(directory_with_files) if len(splitted := file.rsplit('.', 1)) > 1 and splitted[1] == 'txt']
    data: list[File] = []
    for file in files:
        with open(file, 'r') as f:
            try:
                config = parse_title(file)
                data_entry = parse_contents(f.read().strip().rstrip())
                file_data = File(data=data_entry, config=config)
                data.append(file_data)
            except Exception as ex:
                print(f'{file=}')
                raise ex
    by_approach = group_by_approach(data)
    plot_paths = [make_plot(approach, directory_with_files) for approach in by_approach]
    print(plot_paths)


if __name__ == '__main__':
    main()
