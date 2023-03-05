import argparse
import csv
import os
import sys
from typing import Any, Callable, Literal, Optional

import pandas as pd
from pysrc.evaluation_configuration import EvaluationConfiguration
from pysrc.core.compare_csv_line import run_as_compared_csv_line

from pysrc.models.metanome_run_batch import MetanomeRunBatch
from pysrc.utils.eprint import eprint
from ..models.ind import RankedIND

from ..configuration import GlobalConfiguration
from ..utils.plots import (create_TpFpFn_stacked_barplot_by_method, create_plot, plot_missing_values)


def load_experiment_information(json_file: str) -> MetanomeRunBatch:
    with open(json_file) as f:
        json_text = f.read()
        batch: MetanomeRunBatch = MetanomeRunBatch.from_json(json_text)
        return batch


def create_evaluation_csv(runs: MetanomeRunBatch, output_folder: str) -> str:
    output_path = os.path.join(os.getcwd(), output_folder)
    output_csv = os.path.join(output_path, 'data.csv')

    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        if runs.baseline.configuration.arity == 'unary':
            writer.writerow(['file_names', 'sampling_method', "budgets", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_tp_missing_values', 'mean_fp_missing_values'])
        else:
            writer.writerow(['file_names', 'sampling_method', "budgets", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

        baseline = runs.baseline

        for run in runs:
            writer.writerow(run_as_compared_csv_line(run, baseline.results))
    
    return output_csv


def plotting_preprocessing_evaluation_dataframe(df: pd.DataFrame, arity: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """After loading the data from the evaluation csv, some preprocessing needs to be done, before we can create plots"""
    
    if arity == 'nary':
        df_nary = df.copy(deep=True)
        # TODO: At one point there was a  `tolist`. This is now gone. Is this correct?
        def get_assign(key: str) -> Callable[[Any], Any]:
            return lambda x: x[key].str.split('; ')
        
        def get_map(*, type: type, avg: bool = False) -> Callable[[Any], Any]:
            return lambda x: sum([type(i) for i in x]) / (len(x) if avg else 1)

        df_unary = df.assign(tp = get_assign('tp'),
                             fp = get_assign('fp'),
                             fn = get_assign('fn'),
                             precision = get_assign('precision'),
                             recall = get_assign('recall'),
                             f1 = get_assign('f1'))
        df['tp']        = df['tp'].map(get_map(type=int))
        df['fp']        = df['fp'].map(get_map(type=int))
        df['fn']        = df['fn'].map(get_map(type=int))
        df['precision'] = df['precision'].map(get_map(type=float, avg=True))
        df['recall']    = df['recall'].map(get_map(type=float, avg=True))
        df['f1']        = df['f1'].map(get_map(type=float, avg=True))

        return df_unary, df_nary
    else:
        df['budgets'] = df['budgets'].map(lambda x: x.split('; ')[0])
        df['sampling_method'] = df['sampling_method'].map(lambda x: x.split('; ')[0])
        
        return df, None


def make_plots(output_file: str) -> list[str]:
    plot_paths: list[str] = []

    plot_prefix = os.path.join(output_file, 'plot')
    
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
        sampling_methods.append(sampling_method) # type: ignore (sampling_method is str, not Scalar)

    plot_fname = f'{plot_prefix}_stackedBarPlots_detailed.jpg'
    plot_path = create_plot(df_original, sampling_methods, create_TpFpFn_stacked_barplot_by_method, plot_prefix, plot_fname)
    plot_paths.append(plot_path)

    plot_fname = f'{plot_prefix}_missing_values.jpg'
    plot_path = plot_missing_values(df_original, plot_folder=plot_prefix, plot_fname=plot_fname)
    plot_paths.append(plot_path)
    
    return plot_paths


def collect_ind_ranking(experiments: MetanomeRunBatch, mode: Literal['interactive', 'file'], output_folder: str, top_inds: int) -> str:
    ranked_inds = experiments.ranked_inds()
    baseline = experiments.baseline
    ranked_inds_object = [RankedIND(ind, credibility, baseline.results.has_ind(ind)) for ind, credibility in ranked_inds.items()]
    if mode == 'interactive':
        sorted_ranked_inds = sorted(ranked_inds_object, key=lambda ranked_ind : ranked_ind.credibility, reverse=True)
        n = 0
        for ranked_ind in sorted_ranked_inds:
            if n >= top_inds >= 0:
                break
            n += 1
            print(f'{ranked_ind.credibility=} ({ranked_ind.is_tp=}): {ranked_ind.ind=}')
    output_path = os.path.join(os.getcwd(), output_folder)
    output_json = os.path.join(output_path, 'ranked_inds.json')
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json_file.write(RankedIND.schema().dumps(ranked_inds_object, many=True))

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
            or maximum_threshold_percentage == 0.0 and ranked_ind.credibility > 0.0: # When threshold == 0, still require results to be > 0 for better results
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
    EvaluationConfiguration.argparse_arguments(parser)
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, file: str, interactive: bool, return_path: str, top_inds: int) -> Optional[str]:
    experiments = load_experiment_information(json_file=file)
    
    # The file-names of the evaluations should depend on the source file timestamp, not the current timestamp!
    output_sub_directory = file.rsplit(os.sep, 1)[0]
    
    csv_path = create_evaluation_csv(experiments, output_sub_directory)
    if config.create_plots:
        plot_paths: list[str] = make_plots(output_sub_directory)
    else:
        plot_paths = []
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
        case 'ranked':
            return ranked_inds_path
        case _:
            return None


def run_evaluations(general_config: GlobalConfiguration, evaluation_config: EvaluationConfiguration) -> list[Optional[str]]:
    if evaluation_config.file is not None:
        return [run_evaluation(general_config, evaluation_config.file, evaluation_config.interactive, evaluation_config.return_path, evaluation_config.top_inds)]
    return [run_evaluation(general_config, file.rstrip(), evaluation_config.interactive, evaluation_config.return_path, evaluation_config.top_inds) for file in sys.stdin.read().split('\0')]


def main() -> None:
    args = parse_args()
    general_config = GlobalConfiguration.default(vars(args))
    evaluation_config = EvaluationConfiguration.default(vars(args))
    if not general_config.pipe and evaluation_config.file is None or general_config.pipe and evaluation_config.file is not None:
        eprint('Must be either in pipe mode or receive a file argument')
        exit(1)
    result_paths = run_evaluations(general_config, evaluation_config)
    for result_path in result_paths:
        if result_path is not None:
            print(result_path)


if __name__ == '__main__':
    main()
