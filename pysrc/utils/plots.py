import os
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_plot(dataframe: pd.DataFrame, groupby_attrs: list[str], plot_method: Callable, plot_folder: str,
                plot_fname: str, figsize=(15, 10)):
    f, axes = plt.subplots(1, len(groupby_attrs), figsize=figsize)
    sns.despine(f) # Removed upper and right borders on plot

    ## If there are multiple groupby attributes, we create multiple plots side by side
    if len(groupby_attrs) > 1:
        for ax, groupby_attr in zip(axes, groupby_attrs):
            ax = plot_method(axes=ax, dataframe=dataframe, groupby_attr=groupby_attr)
    else:
        axes = plot_method(axes=axes, dataframe=dataframe, groupby_attr=groupby_attrs[0])

    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path


def create_PrecisionRecallF1_lineplot(axes: matplotlib.axes.Axes, dataframe: pd.DataFrame,
                                      groupby_attr: str) -> matplotlib.axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = len(df_grouped) > 10

    d = []
    identifiers = []
    for group_identifier, frame in df_grouped:
        identifiers.append(group_identifier)
        #TODO: Examine if mean is the best way to plot this. Could also be done with error bars....
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


def create_TpFpFn_stacked_barplot(axes: matplotlib.axes.Axes, dataframe: pd.DataFrame,
                                  groupby_attr: str) -> matplotlib.axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = len(df_grouped) > 10

    identifiers = []
    d = []
    # We have to do this since we abuse a histplot, which are usually made for distributions
    # For sure this could be done in a better way, but it ain't stupid if it works, I guess...
    for group_identifier, frame in df_grouped:
        identifiers.append(group_identifier)
        for _ in range(int(frame['tp'].mean())):
            d.append([group_identifier, 'tp', 1])
        for _ in range(int(frame['fp'].mean())):
            d.append([group_identifier, 'fp', 1])
        for _ in range(int(frame['fn'].mean())):
            d.append([group_identifier, 'fn', 1])

    df_grouped = pd.DataFrame(d, columns=[groupby_attr, 'type', 'count'])

    # If the identifiers are floats (aka groupby_attr == 'sampling_rate') we want log_scale enabled
    # Otherwise the histogram bars are squished together for small values
    if isinstance(identifiers[0], float):
        sns.histplot(
            df_grouped,
            x=groupby_attr,
            hue='type',
            hue_order=['tp', 'fp', 'fn'],
            multiple='stack',
            ax=axes,
            discrete=True,
            linewidth=.3,
            log_scale=True,
        )
        axes.set_xticks(identifiers)
    else:
        sns.histplot(
            df_grouped,
            x=groupby_attr,
            hue='type',
            hue_order=['tp', 'fp', 'fn'],
            multiple='stack',
            ax=axes,
            discrete=True,
            linewidth=.3,
            bins=identifiers
        )

    if many_plots:  # If there are too many entries, attempt to make it somewhat readable...
        axes.tick_params(axis='x', rotation=90)
        axes.tick_params(axis='x', labelsize=4)
    axes.set_xlabel(f"{groupby_attr}")
    axes.set_ylabel("IND Count")

    return axes


# For n-ary INDs, create plots showing TP per arity
def create_onion_plot(axes: matplotlib.axes.Axes, dataframe: pd.DataFrame, groupby_attr: str) -> str:
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
                if arity not in tps: tps[arity] = []
                if arity not in fps: fps[arity] = []
                if arity not in fns: fns[arity] = []
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
