import os
from typing import Optional, Protocol

import numpy as np
from matplotlib import axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SamplingMethodPlotMethod(Protocol):
    def __call__(self, *, axes: axes.Axes, dataframe: pd.DataFrame, method: str) -> axes.Axes: ...


def create_plot(dataframe: pd.DataFrame, methods: list[str], plot_method: SamplingMethodPlotMethod, plot_folder: str,
                plot_fname: str, figsize=(15, 10)) -> str:
    f: Figure
    axiis: list[axes.Axes]
    axiis_np: np.ndarray | axes.Axes
    f, axiis_np = plt.subplots(1, len(methods), figsize=figsize)  # type: ignore (typngs of subplots are incorrect)
    axiis = axiis_np.tolist() if len(methods) > 1 else [axiis_np]  # type: ignore (Way too complex for Python)
    sns.despine(f)  # Removed upper and right borders on plot

    # If there are multiple methods, we create multiple plots side by side
    if len(methods) > 1:
        for ax, method in zip(axiis, methods):
            plot_method(axes=ax, dataframe=dataframe, method=method)
    else:
        plot_method(axes=axiis[0], dataframe=dataframe, method=methods[0])

    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path


def plot_missing_values(dataframe: pd.DataFrame, *, plot_folder: str, plot_fname: str,
                        figsize: tuple[int, int] = (15, 10)) -> str:
    f: Figure
    axiis: axes.Axes
    f, axiis = plt.subplots(1, 1, figsize=figsize)
    sns.despine(f)

    data_dict: dict[str, list] = {
        'Config': [],
        'Value': [],
        'Kind': [],
    }
    for _, row in dataframe.iterrows():
        # Don't plot baseline
        if row['sampling_method'] == 'None':
            continue
        data_dict['Config'].append(f'{row["sampling_method"]}, {row["budgets"]}')
        data_dict['Value'].append(row['mean_tp_missing_values'])
        data_dict['Kind'].append('Mean TP Missing Values')
        data_dict['Config'].append(f'{row["sampling_method"]}, {row["budgets"]}')
        data_dict['Value'].append(row['mean_fp_missing_values'])
        data_dict['Kind'].append('Mean FP Missing Values')
        
    df_missing_values = pd.DataFrame(data=data_dict)

    sns.barplot(df_missing_values, y='Config', x='Value', hue='Kind', ax=axiis, orient='h')
        
    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path


def create_precision_recall_f1_lineplot(axes: axes.Axes, dataframe: pd.DataFrame,
                                        groupby_attr: str) -> axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = len(df_grouped) > 10

    d = []
    identifiers = []
    for group_identifier, frame in df_grouped:
        identifiers.append(group_identifier)
        # TODO: Examine if mean is the best way to plot this. Could also be done with error bars....
        d.append([group_identifier, frame['precision'].mean(), frame['recall'].mean(), frame['f1'].mean()])
    df_grouped = pd.DataFrame(data=d, columns=[groupby_attr, 'Precision', 'Recall', 'F1-Score']).set_index(groupby_attr)

    sns.lineplot(data=df_grouped, ax=axes)

    if isinstance(identifiers[0], float):
        axes.set_xscale('log')

    if many_plots:  # If there are too many entries, attempt to make it somewhat readable...
        axes.tick_params(axis='x', rotation=90)
        axes.tick_params(axis='x', labelsize=4)
    axes.set_xlabel(f"{groupby_attr}")
    axes.set_ylabel("Percentages")

    return axes


def create_tp_fp_fn_stacked_barplot_by_method(axes: axes.Axes, dataframe: pd.DataFrame, method: str) -> axes.Axes:
    df_grouped = dataframe.groupby('sampling_method')
    df_matching: Optional[pd.DataFrame] = None
    for sampling_method, frame in df_grouped:
        if sampling_method != method:
            continue
        df_matching = frame
        break
    if df_matching is None:
        raise ValueError(f'{dataframe=} produced no grouping results of the {method=}')
    return create_tp_fp_fn_stacked_barplot(axes, df_matching, 'budgets', method)


def create_tp_fp_fn_stacked_barplot(axiis: axes.Axes, dataframe: pd.DataFrame,
                                    groupby_attr: str, custom_title: Optional[str] = None) -> axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = len(df_grouped) > 10

    identifiers: list[str] = []
    data: list[list[str | int]] = []
    # We have to do this since we abuse a histplot, which are usually made for distributions
    # For sure this could be done in a better way, but it ain't stupid if it works, I guess...
    for group_identifier, frame in df_grouped:
        # TODO: Is this a safe assumption? Under which circumstances would it not work?
        if not isinstance(group_identifier, str):
            raise TypeError(f'{dataframe=} produced a group identifier of type {type(group_identifier)} (value: {group_identifier}) when a string was expected.')
        identifiers.append(group_identifier)
        for _ in range(int(frame['tp'].mean())):
            data.append([group_identifier, 'tp', 1])
        for _ in range(int(frame['fp'].mean())):
            data.append([group_identifier, 'fp', 1])
        for _ in range(int(frame['fn'].mean())):
            data.append([group_identifier, 'fn', 1])

    df_grouped = pd.DataFrame(data, columns=[groupby_attr, 'type', 'count'])

    # If the identifiers are ints (aka groupby_attr == 'budgets') we want log_scale enabled
    # Otherwise the histogram bars are squished together for small values
    plot: axes.Axes
    if len(identifiers) > 0 and isinstance(identifiers[0], int):
        plot = sns.histplot(
            df_grouped,
            x=groupby_attr,
            hue='type',
            hue_order=['tp', 'fp', 'fn'],
            multiple='stack',
            ax=axiis,
            discrete=True,
            linewidth=.3,
            log_scale=True,
        )
        axiis.set_xticks(identifiers)
    else:
        plot = sns.histplot(
            df_grouped,
            x=groupby_attr,
            hue='type',
            hue_order=['tp', 'fp', 'fn'],
            multiple='stack',
            ax=axiis,
            discrete=True,
            linewidth=.3,
            bins=identifiers  # type: ignore (bins accepts lists which is not represented by the typings)
        )
    if custom_title is not None:
        plot.set_title(custom_title)

    if many_plots:  # If there are too many entries, attempt to make it somewhat readable...
        axiis.tick_params(axis='x', rotation=90)
        axiis.tick_params(axis='x', labelsize=4)
    axiis.set_xlabel(f"{groupby_attr}")
    axiis.set_ylabel("IND Count")

    return axiis


# For n-ary INDs, create plots showing TP per arity
def create_onion_plot(axes: axes.Axes, dataframe: pd.DataFrame, groupby_attr: str) -> axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    d = []
    for group_identifier, frame in df_grouped:
        print(group_identifier)
        tp, fp, fn = frame['tp'].str.split('; '), frame['fp'].str.split('; '), frame['fn'].str.split('; ')

        tps: dict[int, list[int]] = {}
        fps: dict[int, list[int]] = {}
        fns: dict[int, list[int]] = {}
        avg_tps, avg_fps, avg_fns = {}, {}, {}
        # Iterate over experiments for the level
        for tp_i, fp_i, fn_i in zip(tp, fp, fn):
            tp_i, fp_i, fn_i = [int(x) for x in tp_i], [int(x) for x in fp_i], [int(x) for x in fn_i]
            # Iterate over the arity levels
            for arity, (tp_ii, fp_ii, fn_ii) in enumerate(zip(tp_i, fp_i, fn_i)):
                if arity not in tps:
                    tps[arity] = []
                if arity not in fps:
                    fps[arity] = []
                if arity not in fns:
                    fns[arity] = []
                tps[arity].append(tp_ii)
                fps[arity].append(fp_ii)
                fns[arity].append(fn_ii)

        # Keys for tps, fps and fns are the same, we can iterate over any of them    
        for arity in tps.keys():
            avg_tps[arity] = sum(tps[arity]) / len(tps[arity])
            avg_fps[arity] = sum(fps[arity]) / len(fps[arity])
            avg_fns[arity] = sum(fns[arity]) / len(fns[arity])

        for arity in avg_tps.keys():
            d.append([group_identifier, arity, avg_tps[arity], avg_fps[arity], avg_fns[arity]])

    df = pd.DataFrame(d, columns=[groupby_attr, 'arity', 'avg_tp', 'avg_fp', 'avg_fn']).set_index(groupby_attr)
    sns.barplot(data=df, axes=axes, x='arity', y='avg_tp', hue=df.index)

    return axes
