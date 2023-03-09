import argparse
import csv
import itertools
import json
import math
import os
import uuid
import pandas as pd
import numpy as np

from pathlib import Path

from collections import defaultdict
from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   MetanomeRunConfiguration, run_metanome)
from ..utils.enhanced_json_encoder import EnhancedJSONEncoder
from ..utils.sampling_methods import sampling_methods_dict
from ..utils.descriptive_statistics import file_column_statistics

from ..models.column_statistics import ColumnStatistic
def aggregate_statistic(file_path: str) -> list[ColumnStatistic]:
    return file_column_statistics(file_path, ';', '\\', False)
def sample_csv(source_files: list[str],
               sampling_method: str,
               description: list[ColumnStatistic],
               config: GlobalConfiguration) -> list[tuple[str, str, float]]:
    """Sample every single column of file separately with a certain method and rate
    and create a new tmp file for every column. Returns a list of tuples including
    the path, method, rate of the  column of the sampled file.
    """

    samples: list[tuple[str, str, float]] = []

    # Initializes the dict with value for no key present

    sampled_data_per_column = [[] for _ in range(len(source_files))]

    #get the stats from the files
    column_count = 0
    for ele in description:
        column_count += len(ele)

    #move to Config
    #TODO isn't a hard cut so far need to see if theres a way to calculate how much i can add per column
    total_budget = 10000
    #TODO find a way to realize that the capacity is reached if every column is increased???
    budget_per_sample = math.floor(total_budget / column_count)
    current_frame = 0
    tupleperRound = 100
    first_sample = True
    budget_left = True

    while budget_left:
        current_file_index = 0
        for file_path in source_files:
            #NOTE to imporve Performance do that once out of this loop AND do it directly per File and Colum?
            aggregate_data_per_column: dict[int, list[str]] = defaultdict(list)
            # Read input file into dataframe and cast all columns into strings
            source_df = pd.read_csv(file_path, delimiter=';', escapechar='\\', dtype='str')
            # Cast each column into a list
            for column_index, column in enumerate(source_df.columns):
                aggregate_data_per_column[column_index] = source_df[column].to_list()

            for column_index in aggregate_data_per_column:

                if config.header and first_sample:
                    file_header = aggregate_data_per_column[column_index][0]

                if description[current_file_index][column_index].count <= current_frame + tupleperRound:
                    #TODO Find a way to return that budget back to the columns who contain more values
                    # --> Adapt the current frame for that might work
                    #Alternative create a list of sample_per_column with the max samples it can take
                    #but I think it is done implicit because all other grow proportional till budget is empty
                    continue

                sampling_method_function = sampling_methods_dict[sampling_method]
                sampled_data = sampling_method_function(aggregate_data_per_column[column_index], current_frame, tupleperRound)

                if first_sample:
                    sampled_data_per_column[current_file_index].insert(column_index, sampled_data)
                else:
                    sampled_data_per_column[current_file_index][column_index] = pd.concat([sampled_data_per_column[current_file_index][column_index], sampled_data])

            current_file_index += 1

        current_frame += tupleperRound
        first_sample = False

        sampled_tuples = 0
        for sam_file in range(len(sampled_data_per_column)):
            for sam_column in sampled_data_per_column[sam_file]:
                sampled_tuples += len(sam_column)


        if sampled_tuples >= total_budget:
            budget_left = False

        sampling_method_function = sampling_methods_dict[sampling_method]
        sampled_data = sampling_method_function(column_data, num_samples+1, num_entries)


    for file_index in range(len(source_files)):
        for column_index in range(len(sampled_data_per_column[file_index])):
            if config.header:
                file_header = sampled_data_per_column[file_index][column_index][0]

            file_prefix = source_files[file_index].rsplit('/', 1)[1].rsplit('.', 1)[0]

            # rename files column specific
            new_file_name = f'{file_prefix}__{str(total_budget).replace(".", "")}_{sampling_method}_{column_index + 1}.csv'
            new_file_path = os.path.join(os.getcwd(), config.tmp_folder, new_file_name)

            with open(new_file_path, 'a') as file:
                writer = csv.writer(file, delimiter=';', escapechar='\\')
                if config.header:
                    writer.writerow([file_header])

                empty_str = ''
                # Changed for better readability
                for current_row in sampled_data_per_column[file_index][column_index].values:
                    # TODO Create Testcases to check if this always works should avoid writing empty lines into the sampled data
                    if current_row == empty_str:
                        continue
                    writer.writerow([current_row])

            out_tuple = (new_file_path, sampling_method, total_budget)
            samples.append(out_tuple)

    return samples


def create_result_json(dataset: str, runs: MetanomeRunBatch,
                       config: GlobalConfiguration) -> str:
    """Creates and writes to a JSON file containing information
    about the set of experiments. Returns the file name."""

    output_path = os.path.join(os.getcwd(), config.output_folder, dataset, config.result_output_folder_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # output_path = os.path.join(os.getcwd(), config.output_folder, output_file)
    output_json = f'{output_path}{os.sep}data.json'
    # output_json = f'{output_path}.json'

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

# NOTE: NOT IN USE CURRENTLY
def get_file_combinations(samples: list[list[tuple[str, str, float]]], config: GlobalConfiguration) \
        -> list[list[tuple[str, str, float]]]:
    data_type_dict: dict[str, list[tuple[int, int]]] = {}
    for num_files_index in range(0, len(samples)):
        for sample_file_index in range(0, len(samples[num_files_index])):

            current_tuple = samples[num_files_index][sample_file_index]
            path_to_data = current_tuple[0]
            # TODO add handling of headers in files
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
    for columns_for_datatype in data_type_dict.values():
        temp_list = []
        for column_for_datatype in columns_for_datatype:
            temp_list.append(samples[column_for_datatype[0]][column_for_datatype[1]])

        datatype_tuples.append(temp_list)

    return datatype_tuples


def run_experiments(dataset: str, config: GlobalConfiguration) -> str:
    """Run experiments and return path to JSON for one dataset"""
    clean_tmp_csv(config.tmp_folder)
    clean_results(config.results_folder)

    experiments: list[MetanomeRun] = []
    source_dir = os.path.join(os.getcwd(), config.source_dir, dataset)
    source_files = [
        os.path.join(source_dir, f)
        for f
        in os.listdir(source_dir)
        if f.rsplit('.')[1] == 'csv'
    ]

    configurations: list[MetanomeRunConfiguration] = []

    
    
    # Baseline
    baseline_set: list[list[tuple[str, str, float]]] = [
        [(src_file, 'None', 1.0)]
        for src_file
        in source_files
    ]
    # Add Configuration for baseline
    for baseline_tuple in itertools.product(*baseline_set):
        file_combination: list[str]
        used_sampling_methods: list[str]
        used_sampling_rates: list[float]
        file_combination, used_sampling_methods, used_sampling_rates = zip(*baseline_tuple)
        configurations.append(MetanomeRunConfiguration(
            algorithm=config.algorithm,
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
            clip_output=config.clip_output,
            header=config.header,
            print_inds=config.print_inds,
            create_plots=config.create_plots,
            is_baseline=True,
        ))
    description = []
    for i, file_path in enumerate(source_files):
        description.append(aggregate_statistic(file_path))
    # Sampled runs
    # Sample each source file
    # Note: New approach: Group by sampling approach and rate already during sample creation
    # This replaces the need for get_file_combinations later on
    #TODO move this into the sample_csv Function
    samples = []
    for sampling_method in config.sampling_methods:
            new_file_list = []
            new_file_list.extend(sample_csv(source_files, sampling_method, description, config))
            samples.append(new_file_list)
    # Note: Old approach
    # for i, file_path in enumerate(source_files):
    #     for sampling_method in config.sampling_methods:
    #         for sampling_rate in config.sampling_rates:
    #             # Sample
    #             new_file_list = sample_csv(file_path, sampling_method, sampling_rate, config)
    #             samples.append(new_file_list)

    # TODO change to clever sampling schema
    # file_combinations_to_test = get_file_combinations(samples, config)
    # for file_combination_setup in file_combinations_to_test:
    for file_combination_setup in samples:
        # TODO: Split this also by column type
        file_combination, used_sampling_methods, used_sampling_rates = zip(*file_combination_setup)
        configurations.append(MetanomeRunConfiguration(
            algorithm=config.algorithm,
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
            clip_output=config.clip_output,
            header=config.header,
            print_inds=config.print_inds,
            create_plots=config.create_plots,
            is_baseline=False
        ))

    # And run experiment for each
    for configuration in configurations:
        current_files_str = ' '.join(configuration.source_files)

        output_file_name = str(uuid.uuid4())
        if configuration.print_inds:
            print(f'{current_files_str=}')
            print(f'{output_file_name=}')
        # Execute
        result = run_metanome(configuration, output_file_name, config.pipe)
        experiments.append(result)

    experiment_batch = MetanomeRunBatch(runs=experiments)

    return create_result_json(dataset, experiment_batch, config)


def run_dataset_experiments(config: GlobalConfiguration) -> list[str]:
    """Run experiments for each dataset in the source folder"""
    return [
        run_experiments(dataset, config)
        for dataset
        in os.listdir(os.path.join(os.getcwd(), config.source_dir))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    GlobalConfiguration.argparse_arguments(parser)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = GlobalConfiguration.default(vars(args))
    json_file_paths = run_dataset_experiments(config)
    json_file_paths_string = ('\0' if config.pipe else '\n').join(json_file_paths)
    if args.pipe:
        print(json_file_paths_string)
    else:
        print(f'JSON files:\n{json_file_paths_string}')


if __name__ == '__main__':
    main()
