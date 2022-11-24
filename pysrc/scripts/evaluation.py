import argparse
import csv
import json
import os
from typing import Optional
from dacite import from_dict

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='The JSON file containing the experiment information to be evaluated')
    parser.add_argument('--return-path', type=str, required=False, default=None, help='Whether to return no path (default), the path of the created csv file (`csv`) or of the plot (`plot`)')
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def run_evaluation(config: GlobalConfiguration, args: argparse.Namespace) -> Optional[str]:
    experiments: MetanomeRunBatch = load_experiment_information(json_file=args.file)
    csv_path = create_evaluation_csv(experiments, config.output_file, config)
    if config.create_plots:
        plot_path = make_plots(config.output_file, config.plot_folder, config)
    print(experiments.tuples_to_remove())
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
