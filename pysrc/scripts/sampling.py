import argparse
import csv
import itertools
import json
import math
import os
import uuid
import pandas as pd
import numpy as np

from collections import defaultdict
from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   MetanomeRunConfiguration, run_metanome)
from ..sampling_methods import sampling_methods_dict
from ..utils.enhanced_json_encoder import EnhancedJSONEncoder


def sample_csv(file_path: str,
               sampling_method: str,
               sampling_rate: float,
               config: GlobalConfiguration) -> list[list[tuple[str, str, float]]]:
    """Sample a single file with a certain method and rate
    and create a new tmp file. Returns the path to the sampled file.
    """

    samples: list[list[tuple[str, str, float]]] = []

    file_prefix = file_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    #Innitializes the dict with value for no key present
    aggregate_data_per_column: [int, list[str]] = defaultdict(list)

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=';', escapechar='\\')
        for row in reader:
            for i in range(len(row)):
                #aggregates the data per line while reading file line by line
                aggregate_data_per_column[i].append(row[i])

    for column in aggregate_data_per_column:

        if config.header:
            file_header = aggregate_data_per_column[column][0]
            data = aggregate_data_per_column[column][1:]

        num_entries = len(aggregate_data_per_column[column])
        num_samples = math.ceil(num_entries * sampling_rate)

        #rename files column specific
        new_file_name = f'{file_prefix}_{str(sampling_rate).replace(".", "")}_{sampling_method}_{column+1}.csv'
        new_file_path = os.path.join(os.getcwd(), config.tmp_folder, new_file_name)

        sampling_method_function = sampling_methods_dict[sampling_method]
        sampled_data = sampling_method_function([aggregate_data_per_column[column]], num_samples, num_entries)

        with open(new_file_path, 'w') as file:
            writer = csv.writer(file, delimiter=',', escapechar='\\')
            if config.header:
                writer.writerow([file_header])

            #Changed for better readability
            for out_row in sampled_data:
                writer.writerow([out_row])

        out_tuple = (new_file_path, sampling_method, sampling_rate)
        samples.append(out_tuple)

    return samples


def create_result_json(runs: MetanomeRunBatch,
                       output_file: str,
                       config: GlobalConfiguration) -> str:
    """Creates and writes to a JSON file containing information
    about the set of experiments. Returns the file name."""

    output_path = os.path.join(os.getcwd(), config.output_folder, output_file)
    output_json = f'{output_path}.json'

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(runs, json_file,
                  ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)
    
    return output_json


def clean_tmp_csv(tmp_folder: str) -> None:
    csv_files = [
        f
        for f in os.listdir(tmp_folder)
        if f.rsplit('.')[1] == 'csv']
    for tmp_file in csv_files:
        os.remove(os.path.join(os.getcwd(), tmp_folder, tmp_file))


def clean_results(results_folder: str) -> None:
    result_files = [f for f in os.listdir(results_folder)]
    for tmp_file in result_files:
        os.remove(os.path.join(os.getcwd(), results_folder, tmp_file))

def get_file_combinations(samples: list[list[tuple[str, str, float]]], config: GlobalConfiguration) \
        -> list[list[tuple[str, str, float]]]:
    data_type_dict = {}
    for num_files_index in range(0, len(samples)):
        for sample_file_index in range(0, len(samples[num_files_index])):

            current_tuple = samples[num_files_index][sample_file_index]
            path_to_data = current_tuple[0]
            #TODO add handling of headers in files
            df = pd.read_csv(path_to_data, sep=';', header=None, on_bad_lines='skip')
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            categorical_columns = df.select_dtypes(include='object').columns.tolist()

            if len(numeric_columns) == 1:
                dtype = "number"
            elif len(categorical_columns) == 1:
                dtype = "object"
            else:
                dtype = "other"

            if dtype in data_type_dict.keys():
                data_type_dict[dtype].append((num_files_index, sample_file_index))
            else:
                data_type_dict[dtype] = [(num_files_index, sample_file_index)]

    datatype_tuples = []
    for key in data_type_dict:
        temp_list = []
        for ele in range(0, len(data_type_dict[key])):
            temp_list.append(samples[data_type_dict[key][ele][0]][data_type_dict[key][ele][1]])
        #TODO create from temp_list a tuple of tuples
        datatype_tuples.append(temp_list)

    return datatype_tuples


def run_experiments(config: GlobalConfiguration) -> str:
    """Run experiments and return path to JSON"""
    clean_tmp_csv(config.tmp_folder)
    clean_results(config.results_folder)

    experiments: list[MetanomeRun] = []
    source_files = [
        os.path.join(os.getcwd(), config.source_dir, f)
        for f
        in os.listdir(os.path.join(os.getcwd(), config.source_dir))
        if f.rsplit('.')[1] == 'csv'
    ]
    baseline_identifier = ' '.join(source_files)
    #Find clever way for column based sampling
    baselineset: list[list[tuple[str, str, float]]] = [
        [(src_file, 'None', 1.0)]
        for src_file
        in source_files
    ]

    # TODO new execution arm for the sampled data with new sample list. Create Function to give valuable Combinations
    # Sample each source file
    samples = []
    for i, file_path in enumerate(source_files):
        for sampling_method in config.sampling_methods:
            for sampling_rate in config.sampling_rates:
                # Sample
                new_file_list = sample_csv(file_path, sampling_method, sampling_rate, config)
                samples.append(new_file_list)



    # Build cartesian product of all possible file combinations
    configurations: list[MetanomeRunConfiguration] = []
    for baseline_tuple in itertools.product(*baselineset):
        baseline: list[str]; used_sampling_methods: list[str]; used_sampling_rates: list[float]
        file_combination, used_sampling_methods, used_sampling_rates = zip(*baseline_tuple)
        configurations.append(MetanomeRunConfiguration(
            arity=config.arity,
            sampling_rates=used_sampling_rates,
            sampling_methods=used_sampling_methods,
            time=config.now,
            source_dir=config.source_dir,
            source_files=file_combination,
            tmp_folder=config.tmp_folder,
            results_folder=config.results_folder,
            result_suffix=config.results_suffix,
            output_folder=config.output_folder,
            output_file=config.output_file,
            clip_output=config.clip_output,
            header=config.header,
            print_inds=config.print_inds,
            create_plots=config.create_plots,
            is_baseline=' '.join(file_combination) == baseline_identifier

        ))

    #TODO change to clever sampling schema change the cartesian product
    file_combinations_to_test = get_file_combinations(samples, config)
    for file_combination_setup in file_combinations_to_test:
        file_combination: list[str];
        used_sampling_methods: list[str];
        used_sampling_rates: list[float]
        file_combination, used_sampling_methods, used_sampling_rates = zip(*file_combination_setup)
        configurations.append(MetanomeRunConfiguration(
            arity=config.arity,
            sampling_rates=used_sampling_rates,
            sampling_methods=used_sampling_methods,
            time=config.now,
            source_dir=config.source_dir,
            source_files=file_combination,
            tmp_folder=config.tmp_folder,
            results_folder=config.results_folder,
            result_suffix=config.results_suffix,
            output_folder=config.output_folder,
            output_file=config.output_file,
            clip_output=config.clip_output,
            header=config.header,
            print_inds=config.print_inds,
            create_plots=config.create_plots,
            is_baseline=' '.join(file_combination) == baseline_identifier
        ))
    
    # And run experiment for each
    for configuration in configurations:
        current_files_str = ' '.join(configuration.source_files)

        output_file_name = str(uuid.uuid4())
        if configuration.print_inds:
            print(f'{current_files_str=}')
            print(f'{output_file_name=}')
        # Execute
        result = run_metanome(configuration, output_file_name)
        experiments.append(result)

    experiment_batch = MetanomeRunBatch(runs=experiments)

    return create_result_json(experiment_batch, config.output_file, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = GlobalConfiguration.default(vars(args))
    json_file_path = run_experiments(config)
    print(f'JSON: {json_file_path}')


if __name__ == '__main__':
    main()
