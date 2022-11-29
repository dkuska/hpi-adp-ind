import argparse
import csv
import json
import os
from typing import Literal, Optional
from dacite import from_dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pysrc.utils.enhanced_json_encoder import EnhancedJSONDecoder, EnhancedJSONEncoder

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


# TODO: Actually implement this
def make_plots(output_file: str, plot_folder: str, config: GlobalConfiguration) -> str:
    df = pd.read_csv(os.path.join(os.getcwd(), config.output_folder, output_file + '.csv'))   
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15))
    sns.despine(f)
    
    df_method = df.groupby('sampling_method')
    d = []
    for method, frame in df_method:
        for i in range(int(frame['tp'].mean())):
            d.append([method,'tp', 1])
        for i in range(int(frame['fp'].mean())):
            d.append([method,'fp', 1])
        # for i in range(int(frame['fn'].mean())):
        #     d.append([method,'fn', 1])
        
    df_method = pd.DataFrame(d, columns=['method', 'type', 'count'])
    sns.histplot(
        df_method,
        x='method',
        hue='type',
        hue_order=['tp', 'fp', 'fn'],
        multiple='stack',
        ax=ax1,
        linewidth=.3,
    )
    ax1.tick_params(axis='x', rotation=90)
    ax1.tick_params(axis='x', labelsize=4)
    ax1.set_xlabel("Sampling Mehods")
    ax1.set_ylabel("Count")
    
    df_rate = df.groupby('sampling_rate')
    d = []
    for rate, frame in df_rate:
        for i in range(int(frame['tp'].mean())):
            d.append([rate,'tp', 1])
        for i in range(int(frame['fp'].mean())):
            d.append([rate,'fp', 1])
        for i in range(int(frame['fn'].mean())):
            d.append([rate,'fn', 1])

    df_rate = pd.DataFrame(d, columns=['rate', 'type', 'count'])
    
    sns.histplot(
        df_rate,
        x='rate',
        hue='type',
        hue_order=['tp', 'fp', 'fn'],
        multiple='stack',
        ax=ax2,
        linewidth=.3,
    )
    
    ax2.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', labelsize=4)
    ax2.set_xlabel("Sampling Rates")
    ax2.set_ylabel("Count")
    
    plot_fname = f'plots_{config.arity}_{config.now.year}{config.now.month:02d}{config.now.day:02d}_{config.now.hour:02d}{config.now.minute:02d}{config.now.second:02d}.jpg' 
    
    plot_path = os.path.join(os.getcwd(), plot_folder, plot_fname)
    f.savefig(plot_path)
    return plot_path


def collect_error_metrics(experiments: MetanomeRunBatch, mode: Literal['interactive', 'file'], config: GlobalConfiguration, output_file: str) -> str:
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
    output_path = os.path.join(os.getcwd(), config.output_folder, output_file)
    output_json = f'{output_path}_with_errors.json'
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(experiments, json_file,
                 ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)
    
    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='The JSON file containing the experiment information to be evaluated')
    parser.add_argument('--return-path', type=str, required=False, default=None, help='Whether to return no path (default), the path of the created csv file (`csv`), of the plot (`plot`), or of the error metrics (`error`)')
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction, required=False, default=False, help='Whether to print the error metrics in a human-readable way')
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, args: argparse.Namespace) -> Optional[str]:
    experiments: MetanomeRunBatch = load_experiment_information(json_file=args.file)
    csv_path = create_evaluation_csv(experiments, config.output_file, config)
    if config.create_plots:
        plot_path = make_plots(config.output_file, config.plot_folder, config)
    error_path = collect_error_metrics(experiments, 'interactive' if args.interactive == True else 'file', config, config.output_file)
    match args.return_path:
        case 'csv':
            return csv_path
        case 'plot':
            return plot_path
        case 'error':
            return error_path
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
