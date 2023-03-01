import argparse
import csv
from dataclasses import dataclass
import itertools
import math
import os
import uuid
import pandas as pd
import numpy as np

from pathlib import Path

from collections import defaultdict

from ..utils.is_non_zero_file import is_non_zero_file
from ..configuration import GlobalConfiguration
from ..models.metanome_run import (MetanomeRun, MetanomeRunBatch,
                                   MetanomeRunConfiguration, run_metanome)
from ..utils.sampling_methods import sampling_methods_dict

from ..utils.descriptive_statistics import file_column_statistics
from ..models.column_statistics import ColumnStatistic

@dataclass(frozen=False)
class ColumnBudgetInfo:
    allowed_budget: int
    full_column_fits_in_budget: bool

def aggregate_statistic(file_path: str, header: bool) -> list[ColumnStatistic]:
    return file_column_statistics(file_path, header=header)

def assign_budget(size_per_column: list[list[ColumnBudgetInfo]], budget_to_share: int, basic_size: int, track_changes: int) -> list[list[ColumnBudgetInfo]]:

    # Amount  of columns that have not their final budget so far
    count_columns_not_full = sum(
        1
        for sizes_for_file
        in size_per_column
        for size_for_column
        in sizes_for_file
        if not size_for_column.full_column_fits_in_budget
    )

    # budget that was left after the initial phase, which is now distributed between the columns that require more budget
    budget_per_column = 0
    if count_columns_not_full != 0:
        budget_per_column = math.floor(budget_to_share / count_columns_not_full)



    # Probably not even required adds another break condition if all columns are at the final budget stage and in between
    # two function calls no more adaptions to the budget could be made
    if count_columns_not_full == track_changes or budget_per_column == 0:
        for sizes_for_file in size_per_column:
            for size_for_column in sizes_for_file:
                if not size_for_column.full_column_fits_in_budget:
                    size_for_column.allowed_budget = budget_per_column + basic_size
        return size_per_column


    for sizes_for_file in size_per_column:
        for size_for_column in sizes_for_file:
            if not size_for_column.full_column_fits_in_budget:
                # check if the amount of unique values fits now with the additional budget then the allowed budget parameter
                # it doesn't fit so far but the budget of this iteration is added to the basic size for the next iteration
                # the adding is performed in line 82
                # decrease the budget pool by the now used budget
                if size_for_column.allowed_budget > budget_per_column + basic_size:
                    #size_for_column.allowed_budget = budget_per_column + basic_size
                    budget_to_share -= budget_per_column

                # all uniques fit now calculate the excess amount and give it to the pool
                # needs no adaption because all uniques can be fitted into the budget
                else:
                    size_for_column.full_column_fits_in_budget = True
                    budget_to_share += basic_size-size_for_column.allowed_budget




    # calls the function again and checks if the now left budget can be even further distributed to columns that require more budget'
    return assign_budget(size_per_column, budget_to_share, basic_size + budget_per_column, count_columns_not_full)


def sample_csv(file_path: str,
               sampling_method: str,
               budget: int,
               size_per_column: list[ColumnBudgetInfo],
               config: GlobalConfiguration) -> list[str]:
    """Sample every single column of file separately with a certain method and budget
    and create a new tmp file for every column.
    Returns a list of the path names of the sampled columns.
    """

    sampled_file_names: list[str] = []

    file_prefix = file_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    # Initializes the dict with value for no key present
    aggregate_data_per_column: dict[int, list[str]] = defaultdict(list)

    # Read input file into dataframe and cast all columns into strings

    source_df = pd.read_csv(file_path, delimiter=';', escapechar='\\', dtype='str', header=None) if is_non_zero_file(file_path) else pd.DataFrame(dtype='str')

    # Cast each column into a list
    for column_index, column in enumerate(source_df.columns):
        aggregate_data_per_column[column_index] = source_df[column].to_list()

    for column in aggregate_data_per_column:
        column_data = aggregate_data_per_column[column]

        if config.header:
            file_header = column_data[0]
            column_data = column_data[1:]

        # Can be removed or doesn't needed for sampling anymore
        num_entries = len(aggregate_data_per_column[column])
        num_samples = size_per_column[column].allowed_budget

        # rename files column specific
        new_file_name = f'{file_prefix}__{str(budget)}_{sampling_method}_{column + 1}.csv'
        new_file_path = os.path.join(os.getcwd(), config.tmp_folder, new_file_name)

        sampling_method_function = sampling_methods_dict[sampling_method]
        sampled_data = sampling_method_function(column_data, num_samples, num_entries)

        with open(new_file_path, 'w') as file:
            writer = csv.writer(file, delimiter=';', escapechar='\\')
            if config.header:
                writer.writerow([file_header]) # pyright: ignore [reportUnboundVariable]

            empty_str = ''
            # Changed for better readability
            for row_index in range(0, len(sampled_data)):
                #TODO Create Testcases to check if this always works should avoid writing empty lines into the sampled data
                if sampled_data.iloc[row_index] == empty_str:
                    continue
                writer.writerow([sampled_data.iloc[row_index]])

        sampled_file_names.append(new_file_path)

    return sampled_file_names


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
        json_file.write(runs.to_json())
        # json.dump(runs, json_file,
        #           ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)

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
            df = pd.read_csv(path_to_data, sep=';', header='infer' if config.header else None, on_bad_lines='skip')
            numeric_columns = df.select_dtypes(include=np.number).columns.tolist() # type: ignore
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
        if f.rsplit('.')[-1] == 'csv'
    ]

    description = [aggregate_statistic(file_path, config.header) for file_path in source_files]

    # find the largest unique count of a column
    largest_unique_count = max(column_description.unique_count for file_description in description for column_description in file_description)
    allowed_missing_values = math.ceil(0.5*largest_unique_count)

    configurations: list[MetanomeRunConfiguration] = []

    
    
    # Baseline
    configurations.append(MetanomeRunConfiguration(
        algorithm=config.algorithm,
        arity=config.arity,
        total_budget=None,
        sampling_method='None',
        allowed_missing_values=allowed_missing_values,
        time=config.now,
        source_dir=config.source_dir,
        source_files=source_files,
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

    # Sampled runs
    # Sample each source file
    # Note: New approach: Group by sampling approach and budget already during sample creation
    # This replaces the need for get_file_combinations later on
    experiment_setups: list[tuple[list[str], str, int]] = []
    for sampling_method in config.sampling_methods:
        for budget in config.total_budget:
            experiment_setup: tuple[list[str], str, int] = ([], sampling_method, budget)
            # Variables for the fair sampling
            budget_to_share = 0
            size_per_column: list[list[ColumnBudgetInfo]] = [[] for _ in range(len(source_files))]
            # Calculates the budget per Column if all column would get the same budget
            column_count = sum(len(file) for file in description)

            basic_size = math.floor(budget/column_count)

            for file_index, file_description in enumerate(description):
                for column_index, column_description in enumerate(file_description):
                    # Checks if the basic size is enough to represent all unique values
                    # if it's not enough write the required size into the ColumnBudgetInfo
                    if column_description.unique_count > basic_size:
                        size_per_column[file_index].insert(column_index, ColumnBudgetInfo(column_description.unique_count, False))

                    # this case the basic budget is enough for all uniques then write the unique count ColumnBudgetInfo
                    # and set True to indicate that no further budget adaptions are required
                    # Calculate the Budget that isn't required and give the amount back to a budget pool
                    else:
                        size_per_column[file_index].insert(column_index, ColumnBudgetInfo(column_description.unique_count, True))
                        budget_to_share += basic_size - column_description.unique_count
            # After the initial budget per columns it's necessary to split the pool of budget that wasn't required by some
            # columns and split it evenly to the columns that need more budget
            size_per_column = assign_budget(size_per_column, budget_to_share, basic_size, track_changes=0)

            #budget_used = sum(
            #    size_for_column.allowed_budget
            #    for sizes_for_file
            #    in size_per_column
            #    for size_for_column
            #    in sizes_for_file)


            for i, file_path in enumerate(source_files):
                experiment_setup[0].extend(sample_csv(file_path, sampling_method, budget, size_per_column[i], config))
            experiment_setups.append(experiment_setup)

    # Note: Old approach
    # for i, file_path in enumerate(source_files):
    #     for sampling_method in config.sampling_methods:
    #         for budget in config.total_budget:
    #             # Sample
    #             new_file_list = sample_csv(file_path, sampling_method, budget, config)
    #             samples.append(new_file_list)

    # TODO change to clever sampling schema
    # file_combinations_to_test = get_file_combinations(samples, config)
    # for file_combination_setup in file_combinations_to_test:
    for experiment_setup in experiment_setups:
        # TODO: Split this also by column type
        file_combination, used_sampling_method, used_budget = experiment_setup
        configurations.append(MetanomeRunConfiguration(
            algorithm=config.algorithm,
            arity=config.arity,
            total_budget=used_budget,
            sampling_method=used_sampling_method,
            allowed_missing_values=allowed_missing_values,
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
