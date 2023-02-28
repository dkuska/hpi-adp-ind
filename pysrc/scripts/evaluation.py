import argparse
import csv
import json
import os
import sys
from typing import Literal, Optional

import pandas as pd
from ..models.ind import RankedIND

from ..utils.enhanced_json_encoder import (EnhancedJSONDecoder,
                                           EnhancedJSONEncoder)

from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   run_as_compared_csv_line)
from ..utils.plots import (create_TpFpFn_stacked_barplot_by_method, create_plot, plot_missing_values)


def load_experiment_information(json_file: str) -> MetanomeRunBatch:
    with open(json_file) as f:
        data = json.load(f, cls=EnhancedJSONDecoder)
        batch: MetanomeRunBatch = MetanomeRunBatch.from_dict(data)
        return batch


def create_evaluation_csv(runs: MetanomeRunBatch, output_folder: str) -> str:
    output_path = os.path.join(os.getcwd(), output_folder)
    output_csv = f'{output_path}{os.sep}data.csv'

    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        if runs.baseline.configuration.arity == 'unary':
            writer.writerow(['file_names', 'sampling_method', "budgets", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_tp_missing_values', 'mean_fp_missing_values'])
        else:
            writer.writerow(['file_names', 'sampling_method', "budgets", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

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
        # NOTE: This is a temporary fix and only works if we use only a single sampling method and budget per experiment
        df['budgets'] = df['budgets'].map(lambda x: x.split('; ')[0])
        df['sampling_method'] = df['sampling_method'].map(lambda x: x.split('; ')[0])
        
        return df, None


def make_plots(output_file: str) -> list[str]:
    plot_paths = []

    plot_prefix = f'{output_file}{os.sep}plot'
    
    arity = 'unary' if 'unary' in output_file.rsplit(os.sep, 1)[1] else 'nary'
    
    df: pd.DataFrame = pd.read_csv(os.path.join(output_file, 'data.csv'))
       
    # Preprocessing of DataFrames, df_nary is None if arity == 'unary'
    df_original, df_nary = plotting_preprocessing_evaluation_dataframe(df, arity)
    
    # if arity == 'nary':
    #     plot_fname = f'{plot_prefix}_onionPlot.jpg'
    #     groupby_attributes = ['num_sampled_files']
    #     onionplot_path = create_plot(df_nary, groupby_attributes, create_onion_plot, plot_prefix, plot_fname)
    #     plot_paths.append(onionplot_path)

    sampling_methods: list[str] = []
    for sampling_method, _ in df_original.groupby('sampling_method'):
        sampling_methods.append(sampling_method)

    # for sampling_method, _ in df_original.groupby('sampling_method'):
    #     print(f'{sampling_method=}')
    #     plot_fname = f'{plot_prefix}_stackedBarPlots_detailed_{sampling_method}_test.jpg'
    #     plot_path = create_plot(df_original, [sampling_method], create_TpFpFn_stacked_barplot_by_method, plot_prefix, plot_fname)
    #     plot_paths.append(plot_path)
    plot_fname = f'{plot_prefix}_stackedBarPlots_detailed.jpg'
    plot_path = create_plot(df_original, sampling_methods, create_TpFpFn_stacked_barplot_by_method, plot_prefix, plot_fname)
    plot_paths.append(plot_path)

    plot_fname = f'{plot_prefix}_missing_values.jpg'
    plot_path = plot_missing_values(df_original, plot_folder=plot_prefix, plot_fname=plot_fname)
    plot_paths.append(plot_path)

    # groupby_attributes = ['sampling_method', 'budgets']
    # plot_fname = f'{plot_prefix}_stackedBarPlots_detailed.jpg'
    # plot_path = create_plot(df_original, groupby_attributes, create_TpFpFn_stacked_barplot, plot_prefix, plot_fname)
    # plot_paths.append(plot_path)
    
    # plot_fname = f'{plot_prefix}_linePlots_detailed.jpg'
    # plot_path = create_plot(df_original, groupby_attributes, create_PrecisionRecallF1_lineplot, plot_prefix, plot_fname)
    # plot_paths.append(plot_path)
   
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


def collect_ind_ranking(experiments: MetanomeRunBatch, mode: Literal['interactive', 'file'], output_folder: str, top_inds: int) -> str:
    ranked_inds = experiments.ranked_inds()
    baseline = experiments.baseline
    ranked_inds_object = [RankedIND(ind, credibility, baseline.results.has_ind(ind)) for ind, credibility in ranked_inds.items()]
    if mode == 'interactive':
        sorted_ranked_inds = sorted(ranked_inds_object, key=lambda ranked_ind : ranked_ind.credibility, reverse=True)
        # print(sorted_ranked_inds)
        n = 0
        for ranked_ind in sorted_ranked_inds:
            if n >= top_inds and top_inds >= 0:
                break
            n += 1
            print(f'{ranked_ind.credibility=} ({ranked_ind.is_tp=}): {ranked_ind.ind=}')
    output_path = os.path.join(os.getcwd(), output_folder)
    output_json = f'{output_path}{os.sep}ranked_inds.json'
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json_file.write(RankedIND.schema().dumps(ranked_inds_object, many=True))
        # json.dump(ranked_inds_object, json_file, ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)

    return output_json


def evaluate_ind_rankings(ranked_inds_path: str, maximum_threshold_percentage: float) -> None:
    ranked_inds: list[RankedIND] = []
    with open(ranked_inds_path, 'r', encoding='utf-8') as json_file:
        ranked_inds = RankedIND.schema().loads(json_file.read(), many=True)
    if len(ranked_inds) < 1:
        print('No INDs available to be ranked.')
        return
    max_credibility = max(ranked_inds, key=lambda ranked_ind : ranked_ind.credibility).credibility
    tps, fps, tns, fns = 0, 0, 0, 0
    for ranked_ind in ranked_inds:
        if maximum_threshold_percentage > 0.0 and ranked_ind.credibility >= maximum_threshold_percentage * max_credibility \
            or maximum_threshold_percentage == 0.0 and ranked_ind.credibility > 0.0:
            if ranked_ind.is_tp is None:
                continue
            if ranked_ind.is_tp: tps += 1
            else: fps += 1
        else:
            if ranked_ind.is_tp is None:
                continue
            if ranked_ind.is_tp: fns += 1
            else: tns += 1
    total_number_of_inds = len(ranked_inds)
    accuracy = round((tps + tns) / total_number_of_inds, 5) if total_number_of_inds > 0 else 0.0
    precision = round(tps / (tps + fps), 5) if tps + fps > 0 else 0.0
    recall = round(tps / (tps + fns), 5) if tps + fns > 0 else 0.0
    print(f'For {maximum_threshold_percentage=}: {tps=}, {fps=}, {fns=}, {tns=} -> {accuracy=}, {precision=}, {recall=}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=False, default=None, help='The JSON file containing the experiment information to be evaluated')
    parser.add_argument('--return-path', type=str, required=False, default=None, help='Whether to return no path (default), the path of the created csv file (`csv`), of the plot (`plot`), of the error metrics (`error`), or of the ranked inds (`ranked`)')
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, required=False, default=False, help='Whether to print the error metrics in a human-readable way')
    parser.add_argument('--top-inds', type=int, default=-1, help='The number of INDs (from the top ranking) that should be shown. A negative number shows all.')
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, file: str, interactive: bool, return_path: str, top_inds: int) -> Optional[str]:
    experiments: MetanomeRunBatch = load_experiment_information(json_file=file)
    
    # The file-names of the evaluations should depend on the source file timestamp, not the current timestamp!
    output_sub_directory = file.rsplit(os.sep, 1)[0]  # .rsplit('.', 1)[0]
    
    csv_path = create_evaluation_csv(experiments, output_sub_directory)
    if config.create_plots:
        plot_paths = make_plots(output_sub_directory)
    error_path = collect_error_metrics(experiments, 'interactive' if interactive else 'file', output_sub_directory)
    ranked_inds_path = collect_ind_ranking(experiments, 'interactive' if interactive else 'file', output_sub_directory, top_inds)
    if interactive:
        thresholds = [1.0, 0.995, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.01, 0.005, 0.001, 0.0]
        for threshold in thresholds:
            evaluate_ind_rankings(ranked_inds_path, threshold)
    match return_path:
        case 'csv':
            return csv_path
        case 'plot':
            return ', '.join(plot_paths)
        case 'error':
            return error_path
        case 'ranked':
            return ranked_inds_path
        case _:
            return None


def run_evaluations(config: GlobalConfiguration, args: argparse.Namespace) -> list[Optional[str]]:
    if not config.pipe:
        return [run_evaluation(config, args.file, args.interactive, args.return_path, args.top_inds)]
    return [run_evaluation(config, file.rstrip(), args.interactive, args.return_path, args.top_inds) for file in sys.stdin.read().split('\0')]


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
