import math
import sys
from typing import Literal
from matplotlib import axes, pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns


def main() -> None:
    thresholds = [1.0, 0.995, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.01, 0.005, 0.001, 0.0]
    # Dataset: ENSEMBL. Budget: 300000. Low Knowledge
    # logreg_precision = [0.0, 0.09052, 0.09052, 0.09052, 0.09052, 0.09052, 0.09052, 0.09101, 0.09101, 0.09101, 0.09101, 0.09101, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199]
    # forest_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 0.30551, 0.14377, 0.12459, 0.09957, 0.09261, 0.09027, 0.09052, 0.09101, 0.09101, 0.09156, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199, 0.09199]
    # naive_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96096, 0.94767, 0.94051, 0.80144, 0.78372, 0.65058, 0.62177, 0.59752, 0.58813]
    # tree_precision = [0.09199] * len(thresholds)
    # logreg_recall = [0.0, 0.96562, 0.96562, 0.96562, 0.96562, 0.96562, 0.96562, 0.97135, 0.97135, 0.97135, 0.97135, 0.97135, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427]
    # forest_recall = [0.29513, 0.29513, 0.29799, 0.30086, 0.4384, 0.52436, 0.77364, 0.87393, 0.91977, 0.94842, 0.96275, 0.96562, 0.97135, 0.97135, 0.98854, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427, 0.99427]
    # naive_recall = [0.65903, 0.65903, 0.65903, 0.65903, 0.65903, 0.65903, 0.65903, 0.67335, 0.68195, 0.68195, 0.69341, 0.7106, 0.91691, 0.9341, 0.95129, 0.95989, 0.96562, 0.96562, 0.96562, 0.96562, 0.96562]
    # tree_recall = [0.99427] * len(thresholds)
    # Dataset: SCOP. Budget: 30000. High Knowledge
    # logreg_precision = [0.0, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414]
    # forest_precision = [0.0, 0.0, 0.0, 0.375, 0.6, 0.66667, 0.50794, 0.43023, 0.42268, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414, 0.41414]
    # naive_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.88889, 0.88462, 0.82609, 0.82609, 0.82609, 0.82609]
    # tree_precision = [0.33784] * len(thresholds)
    # logreg_recall = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # forest_recall = [0.0, 0.0, 0.0, 0.07317, 0.29268, 0.39024, 0.78049, 0.90244, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # naive_recall = [0.04878, 0.04878, 0.04878, 0.09756, 0.09756, 0.09756, 0.31707, 0.31707, 0.31707, 0.34146, 0.34146, 0.36585, 0.39024, 0.39024, 0.39024, 0.39024, 0.56098, 0.92683, 0.92683, 0.92683, 0.92683]
    # tree_recall = [0.60976] * len(thresholds)
    # Dataset: TCP-H. Budget: 150000. Low Knowledge
    logreg_precision = [0.0, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.08114, 0.8503, 0.8503, 0.8503, 0.8503, 0.8503, 0.8503, 0.8503, 0.8503, 0.8503]
    forest_precision = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02139, 0.08966, 0.07711, 0.06175, 0.0727, 0.08095, 0.0818, 0.08276, 0.08219, 0.08428, 0.08503, 0.08503, 0.08503, 0.08503, 0.08503]
    naive_precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.875, 0.81707, 0.79762, 0.79762, 0.79762, 0.73626, 0.62712, 0.62185, 0.59677, 0.5873]
    tree_precision = [0.0] * len(thresholds)
    logreg_recall = [0.0, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 0.94667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    forest_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05333, 0.34667, 0.42667, 0.54667, 0.70667, 0.86667, 0.94667, 0.96, 0.96, 0.98667, 1.0, 1.0, 1.0, 1.0, 1.0]
    naive_recall = [0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.58667, 0.6, 0.84, 0.89333, 0.89333, 0.89333, 0.89333, 0.89333, 0.98667, 0.98667, 0.98667, 0.98667]
    tree_recall = [0.0] * len(thresholds)
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
    f, axs = plt.subplots(1, 3, figsize=(15, 10))
    ax_precision = axs[0]
    ax_recall = axs[1]
    ax_f1 = axs[2]
    plot_recall = sns.lineplot(data=pd.melt(frame_precision, ['Threshold'], var_name='Method', value_name='Precision'), x='Threshold', y='Precision', hue='Method', ax=ax_precision)
    plot_recall.set_title('Precision')
    plot_precision = sns.lineplot(data=pd.melt(frame_recall, ['Threshold'], var_name='Method', value_name='Recall'), x='Threshold', y='Recall', hue='Method', ax=ax_recall)
    plot_precision.set_title('Recall')
    plot_f1 = sns.lineplot(data=pd.melt(frame_f1, ['Threshold'], var_name='Method', value_name='F1-Score'), x='Threshold', y='F1-Score', hue='Method', ax=ax_f1)
    plot_f1.set_title('F1-Score')
    f.suptitle('Dataset: TCP-H. Budget: 150000. Low Knowledge')
    # sample_count = np.around(np.logspace(math.log10(1),math.log10(10),6))
    # plot.set(yscale='log')
    # plot.set(yticks=sample_count)
    # plot.set(yticklabels=sample_count)
    plot_path = sys.argv[1]
    f.savefig(plot_path)
    print(plot_path)


if __name__ == '__main__':
    main()
