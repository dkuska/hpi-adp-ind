import argparse
import csv
import json
import os
from typing import Optional
from dacite import from_dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pysrc.utils.enhanced_json_encoder import EnhancedJSONDecoder

from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   run_as_compared_csv_line)


def load_experiment_information(json_file: str) -> MetanomeRunBatch:
    with open(json_file) as f:
        data = json.load(f, cls=EnhancedJSONDecoder)
        batch = from_dict(MetanomeRunBatch, data)
        return batch


def create_evaluation_csv(runs: MetanomeRunBatch, output_file: str, config: GlobalConfiguration) -> str:
    output_path = os.path.join(os.getcwd(), config.output_folder, output_file)
    output_csv = f'{output_path}.csv'

    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['sampled_files', 'sampling_method', "sampling_rate", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

        baseline: MetanomeRun = runs.baseline

        for run in runs:
            writer.writerow(run_as_compared_csv_line(run, baseline.results))
    
    return output_csv


def make_plots(output_file: str, plot_folder: str, config: GlobalConfiguration) -> str:
    df = pd.read_csv(os.path.join(os.getcwd(), config.output_folder, output_file + '.csv'))   
    # Count how many files were in the source and how many were sampled
    num_files = len(df['sampled_files'].tolist()[0].split(';'))
    df = df.assign(num_sampled_files= lambda x: num_files - (x['sampling_method'].str.count('None')))
    
    plot_fname = f'stackedBarPlots_{output_file}_detailed.jpg'
    groupby_attributes = ['sampling_rate', 'sampling_method']
    barplot_path = create_TpFpFn_stacked_barplot(df, groupby_attributes, plot_folder, plot_fname)
    
    plot_fname = f'stackedBarPlots_{output_file}_simplified.jpg'
    groupby_attributes = ['num_sampled_files']
    barplot_path = create_TpFpFn_stacked_barplot(df, groupby_attributes, plot_folder, plot_fname)
    
    if config.arity == 'nary':
        onionplot_path = create_onion_plot(df, plot_folder, config)

    return barplot_path

def create_TpFpFn_stacked_barplot(df: pd.DataFrame, groupby_attrs: list[str], plot_folder: str, plot_fname: str, figsize = (10,10)) -> str:
    
    f, axes = plt.subplots(1, len(groupby_attrs), figsize=figsize)
    sns.despine(f)
    
    if len(groupby_attrs) > 1:
        for ax, groupby_attr in zip(axes, groupby_attrs):
            ax = create_TpFpFn_stacked_barplot_single_axis(axes = ax, dataframe = df, groupby_attr = groupby_attr)
    else:
        axes = create_TpFpFn_stacked_barplot_single_axis(axes = axes, dataframe = df, groupby_attr = groupby_attrs[0])
    
    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path

def create_TpFpFn_stacked_barplot_single_axis(axes : matplotlib.axes.Axes, dataframe: pd.DataFrame, groupby_attr: str) -> matplotlib.axes.Axes:
    df_grouped = dataframe.groupby(groupby_attr)
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
    axes.tick_params(axis='x', rotation=90)
    axes.tick_params(axis='x', labelsize=4)
    axes.set_xlabel(f"{groupby_attr}")
    axes.set_ylabel("IND Count")
    
    return axes


## For n-ary INDs, create plots
def create_onion_plot(df: pd.DataFrame, plot_folder: str, config: GlobalConfiguration) -> str:
    return ''


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
    
    csv_path = create_evaluation_csv(experiments, output_file, config)
    if config.create_plots:
        plot_path = make_plots(output_file, config.plot_folder, config)
    match args.return_path:
        case 'csv':
            return csv_path
        case 'plot':
            return plot_path
        case _:
            return None


def main():
    args = parse_args()
    config = GlobalConfiguration.default(vars(args))
    result_path = run_evaluation(config, args)
    if result_path:
        print(result_path)


if __name__ == '__main__':
    main()
