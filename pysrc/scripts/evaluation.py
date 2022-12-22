import argparse
import csv
import json
import os
import sys
from typing import Literal, Optional

import pandas as pd

from pysrc.utils.enhanced_json_encoder import (EnhancedJSONDecoder,
                                               EnhancedJSONEncoder)

from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   run_as_compared_csv_line)
from ..utils.plots import (create_onion_plot, create_plot,
                           create_PrecisionRecallF1_lineplot,
                           create_TpFpFn_stacked_barplot)


def load_experiment_information(json_file: str) -> MetanomeRunBatch:
    with open(json_file) as f:
        data = json.load(f, cls=EnhancedJSONDecoder)
        batch = MetanomeRunBatch.from_dict(data)
        return batch


def create_evaluation_csv(runs: MetanomeRunBatch, output_folder: str) -> str:
    output_path = os.path.join(os.getcwd(), output_folder)
    output_csv = f'{output_path}{os.sep}data.csv'

    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['file_names', 'sampling_method', "sampling_rate", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

        baseline: MetanomeRun = runs.baseline

        for run in runs:
            writer.writerow(run_as_compared_csv_line(run, baseline.results))
    
    return output_csv


def plotting_preprocessing_evaluation_dataframe(df: pd.DataFrame, arity: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """After loading the data from the evaluation csv, some preprocessing needs to be done, before we can create plots
    """
    # Count how many files were in the source and how many were sampled
    # num_files = len(df['sampled_files'].tolist()[0].split(';'))
    # df = df.assign(num_sampled_files= lambda x: num_files - (x['sampling_method'].str.count('None')))
    
    if arity == 'nary':
        df_nary = df.copy(deep=True)
        # TODO: is there a way to make this prettier?.....
        df_unary = df.assign(tp = lambda x: x['tp'].str.split('; ').tolist(),
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
        
        return df_unary, df_nary
    else:
        # NOTE: This is a temporary fix and only works if we use only a single sampling method and rate per experiment
        df['sampling_rate'] = df['sampling_rate'].map(lambda x: x.split('; ')[0])
        df['sampling_method'] = df['sampling_method'].map(lambda x: x.split('; ')[0])
        
        return df, None


def make_plots(output_file: str) -> list[str]:
    plot_paths = []

    plot_prefix = f'{output_file}{os.sep}plot'
    
    arity = 'unary' if 'unary' in output_file.rsplit(os.sep, 1)[1] else 'nary'
    
    df = pd.read_csv(os.path.join(output_file, 'data.csv'))
       
    # Preprocessing of DataFrames, df_nary is None if arity == 'unary'
    df_original, df_nary = plotting_preprocessing_evaluation_dataframe(df, arity)
    
    if arity == 'nary':
        plot_fname = f'{plot_prefix}_onionPlot.jpg'
        groupby_attributes = ['num_sampled_files']
        onionplot_path = create_plot(df_nary, groupby_attributes, create_onion_plot, plot_prefix, plot_fname)
        plot_paths.append(onionplot_path)
        
    groupby_attributes = ['sampling_method', 'sampling_rate']
    plot_fname = f'{plot_prefix}_stackedBarPlots_detailed.jpg'
    plot_path = create_plot(df_original, groupby_attributes, create_TpFpFn_stacked_barplot, plot_prefix, plot_fname)
    plot_paths.append(plot_path)
    
    plot_fname = f'{plot_prefix}_linePlots_detailed.jpg'
    plot_path = create_plot(df_original, groupby_attributes, create_PrecisionRecallF1_lineplot, plot_prefix, plot_fname)
    plot_paths.append(plot_path)
   
    # groupby_attributes = ['num_sampled_files']
    # plot_fname = f'{output_file}_stackedBarPlots_simplified.jpg'
    # plot_path = create_plot(df_original, groupby_attributes, create_TpFpFn_stacked_barplot, plot_folder, plot_fname)
    # plot_paths.append(plot_path)
    
    # plot_fname = f'{output_file}_linePlots_simplified.jpg' 
    # plot_path = create_plot(df_original, groupby_attributes, create_PrecisionRecallF1_lineplot, plot_folder, plot_fname)
    # plot_paths.append(plot_path)
    
    return plot_paths


def collect_error_metrics(experiments: MetanomeRunBatch, mode: Literal['interactive', 'file'], output_folder: str) -> str:
    if mode == 'interactive':
        print('Error metrics collection is (temporarily) disabled for performance reasons')
    return ''
    tuples_to_remove = experiments.tuples_to_remove()
    if mode == 'interactive':
        print('### Tuples To Remove ###')
        print('When looking into the shown file combinations, on average that many tuples have to be removed to make all found INDs TPs.')
        print('The data is presented as (absolute number of tuples to be removed, relative percentage of tuples to be removed, absolute number of distinct tuples to be removed, relative percentage of distinct tuples to be removed).')
        print({
            tuple([
                file.rsplit('/', 1)[1]
                for file
                in run.configuration.source_files
            ]): error
            for run, error
            in tuples_to_remove.items()
            if error[0] + error[1] > 0.0
        })
    # Always print results in detail to a file
    output_path = os.path.join(os.getcwd(), output_folder)
    output_json = f'{output_path}{os.sep}errors.json'
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(experiments, json_file,
                 ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)

    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False, default=None, help='The JSON file containing the experiment information to be evaluated')
    parser.add_argument('--return-path', type=str, required=False, default=None, help='Whether to return no path (default), the path of the created csv file (`csv`), of the plot (`plot`), or of the error metrics (`error`)')
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, required=False, default=False, help='Whether to print the error metrics in a human-readable way')
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, file: str, interactive: bool, return_path: str) -> Optional[str]:
    experiments: MetanomeRunBatch = load_experiment_information(json_file=file)
    
    # The file-names of the evaluations should depend on the source file timestamp, not the current timestamp!
    output_sub_directory = file.rsplit(os.sep, 1)[0]  # .rsplit('.', 1)[0]
    
    csv_path = create_evaluation_csv(experiments, output_sub_directory)
    if config.create_plots:
        plot_paths = make_plots(output_sub_directory)
    error_path = collect_error_metrics(experiments, 'interactive' if interactive else 'file', output_sub_directory)
    match return_path:
        case 'csv':
            return csv_path
        case 'plot':
            return ', '.join(plot_paths)
        case 'error':
            return error_path
        case _:
            return None


def run_evaluations(config: GlobalConfiguration, args: argparse.Namespace) -> list[Optional[str]]:
    if not config.pipe:
        return [run_evaluation(config, args.file, args.interactive, args.return_path)]
    return [run_evaluation(config, file.rstrip(), args.interactive, args.return_path) for file in sys.stdin.read().split('\0')]


def main():
    args = parse_args()
    config = GlobalConfiguration.default(vars(args))
    if not config.pipe and args.file is None or config.pipe and args.file is not None:
        print('Must be either in pipe mode or receive a file argument')
        exit(1)
    result_paths = run_evaluations(config, args)
    for result_path in result_paths:
        if result_path:
            print(result_path)


if __name__ == '__main__':
    main()
