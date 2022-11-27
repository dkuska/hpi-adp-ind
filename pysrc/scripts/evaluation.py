import argparse
import csv
import json
import os
from typing import Optional, Callable
from dacite import from_dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

from pysrc.utils.enhanced_json_encoder import EnhancedJSONDecoder

from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   run_as_compared_csv_line)


def load_experiment_information(json_file: str) -> MetanomeRunBatch:
    with open(json_file) as f:
        data = json.load(f, cls=EnhancedJSONDecoder)
        batch = from_dict(MetanomeRunBatch, data)
        return batch


def create_evaluation_csv(runs: MetanomeRunBatch, output_file: str, output_folder: str) -> str:
    output_path = os.path.join(os.getcwd(), output_folder, output_file)
    output_csv = f'{output_path}.csv'

    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['sampled_files', 'sampling_method', "sampling_rate", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

        baseline: MetanomeRun = runs.baseline

        for run in runs:
            writer.writerow(run_as_compared_csv_line(run, baseline.results))
    
    return output_csv


def create_TpFpFn_stacked_barplot(axes : matplotlib.axes.Axes, dataframe: pd.DataFrame, groupby_attr: str) -> matplotlib.axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = False
    if len(df_grouped) > 10:
        many_plots = True
    
    d = []
    for group_identifier, frame in df_grouped:
        for _ in range(int(frame['tp'].mean())):
            d.append([group_identifier,'tp', 1])
        for _ in range(int(frame['fp'].mean())):
            d.append([group_identifier,'fp', 1])
        for _ in range(int(frame['fn'].mean())):
            d.append([group_identifier,'fn', 1])
        
    df_grouped = pd.DataFrame(d, columns=[groupby_attr, 'type', 'count'])
    sns.histplot(
        df_grouped,
        x=groupby_attr,
        hue='type',
        hue_order=['tp', 'fp', 'fn'],
        multiple='stack',
        ax=axes,
        discrete=True,
        linewidth=.3
    )
    if many_plots: # If there are too many entries, attempt to make it somewhat readable...
        axes.tick_params(axis='x', rotation=90)
        axes.tick_params(axis='x', labelsize=4)
    axes.set_xlabel(f"{groupby_attr}")
    axes.set_ylabel("IND Count")
    
    return axes


def create_PrecisionRecallF1_lineplot(axes : matplotlib.axes.Axes, dataframe: pd.DataFrame, groupby_attr: str) -> matplotlib.axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
    many_plots = False
    if len(df_grouped) > 10:
        many_plots = True
    
    d = []
    for group_identifier, frame in df_grouped:
        d.append([group_identifier, frame['precision'].mean(), frame['recall'].mean(), frame['f1'].mean()])            
    df_grouped = pd.DataFrame(d, columns=[groupby_attr, 'Precision', 'Recall', 'F1-Score']).set_index(groupby_attr)

    sns.lineplot(data=df_grouped, ax=axes)
    
    if many_plots: # If there are too many entries, attempt to make it somewhat readable...
        axes.tick_params(axis='x', rotation=90)
        axes.tick_params(axis='x', labelsize=4)
    axes.set_xlabel(f"{groupby_attr}")
    axes.set_ylabel("Percentages")
    
    return axes


def create_plot(dataframe: pd.DataFrame, groupby_attrs: list[str], plot_method: Callable, plot_folder: str, plot_fname: str, figsize = (15,10)):
    f, axes = plt.subplots(1, len(groupby_attrs), figsize=figsize)
    sns.despine(f)
    
    if len(groupby_attrs) > 1:
        for ax, groupby_attr in zip(axes, groupby_attrs):
            ax = plot_method(axes = ax, dataframe = dataframe, groupby_attr = groupby_attr)
    else:
        axes = plot_method(axes = axes, dataframe = dataframe, groupby_attr = groupby_attrs[0])
    
    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path


## For n-ary INDs, create plots showing TP,FP,FN per arity
# TODO: IMPLEMENT
def create_onion_plot(df: pd.DataFrame, plot_folder: str, config: GlobalConfiguration, figsize=(10,10)) -> str:
    f, axes = plt.subplots(figsize=figsize)
    
    
    sns.barplot(data=df)
    
    
    return ''

def make_plots(output_file: str, plot_folder: str, config: GlobalConfiguration) -> list[str]:
    plot_paths = []
    
    arity = 'unary' if 'unary' in output_file else 'nary'
    
    df = pd.read_csv(os.path.join(os.getcwd(), config.output_folder, output_file + '.csv'))   
    # Count how many files were in the source and how many were sampled
    num_files = len(df['sampled_files'].tolist()[0].split(';'))
    df = df.assign(num_sampled_files= lambda x: num_files - (x['sampling_method'].str.count('None')))
    
    if arity == 'nary':
        df_copy = df.copy(deep=True)
        # TODO: is there a way to make this prettier?.....
        df = df.assign(tp = lambda x: x['tp'].str.split('; ').tolist(),
                                 fp = lambda x: x['fp'].str.split('; '),
                                 fn = lambda x: x['fn'].str.split('; '),
                                 precision = lambda x: x['precision'].str.split('; '),
                                 recall = lambda x: x['recall'].str.split('; '),
                                 f1 = lambda x: x['f1'].str.split('; '))
        df['tp']        = df['tp'].map(lambda x: sum([int(i) for i in x]))
        df['fp']        = df['fp'].map(lambda x: sum([int(i) for i in x]))
        df['fn']        = df['fn'].map(lambda x: sum([int(i) for i in x]))
        df['precision'] = df['precision'].map(lambda x: sum([float(i) for i in x])/len(x))
        df['recall']    = df['recall'].map(lambda x: sum([float(i) for i in x])/len(x))
        df['f1']        = df['f1'].map(lambda x: sum([float(i) for i in x])/len(x))

        onionplot_path = create_onion_plot(df_copy, plot_folder, config)
        
    groupby_attributes = ['sampling_rate', 'sampling_method']
    plot_fname = f'{output_file}_stackedBarPlots_detailed.jpg'
    plot_path = create_plot(df, groupby_attributes, create_TpFpFn_stacked_barplot, plot_folder, plot_fname)
    plot_paths.append(plot_path)
    
    plot_fname = f'{output_file}_linePlots_detailed.jpg' 
    plot_path = create_plot(df, groupby_attributes, create_PrecisionRecallF1_lineplot, plot_folder, plot_fname)
    plot_paths.append(plot_path)
    
    groupby_attributes = ['num_sampled_files']
    plot_fname = f'{output_file}_stackedBarPlots_simplified.jpg'
    plot_path = create_plot(df, groupby_attributes, create_TpFpFn_stacked_barplot, plot_folder, plot_fname)
    plot_paths.append(plot_path)
    
    plot_fname = f'{output_file}_linePlots_simplified.jpg' 
    plot_path = create_plot(df, groupby_attributes, create_PrecisionRecallF1_lineplot, plot_folder, plot_fname)
    plot_paths.append(plot_path)
        
    return plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='The JSON file containing the experiment information to be evaluated')
    parser.add_argument('--return-path', type=str, required=False, default=None, help='Whether to return no path (default), the path of the created csv file (`csv`) or of the plot (`plot`)')
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, args: argparse.Namespace) -> Optional[str]:
    experiments: MetanomeRunBatch = load_experiment_information(json_file=args.file)
    
    # The file-names of the evaluations should depend on the source file timestamp, not the current timestamp!
    output_file = args.file.rsplit('/',1)[-1].rsplit('.', 1)[0]
    
    csv_path = create_evaluation_csv(experiments, output_file, config.output_folder)
    if config.create_plots:
        plot_paths = make_plots(output_file, config.plot_folder, config)
    match args.return_path:
        case 'csv':
            return csv_path
        case 'plot':
            return ', '.join(plot_paths)
        case _:
            return None


def main():
    args = parse_args()
    config = GlobalConfiguration.default(vars(args))
    result_paths = run_evaluation(config, args)
    if result_paths:
        print(result_paths)


if __name__ == '__main__':
    main()
