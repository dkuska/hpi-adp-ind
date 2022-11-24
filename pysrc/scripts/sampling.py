import argparse
import csv
import itertools
import json
import math
import os
import uuid

from pysrc.configuration import GlobalConfiguration
from pysrc.models.metanome_run import MetanomeRun, MetanomeRunBatch, MetanomeRunConfiguration, run_metanome
from pysrc.sampling_methods import sampling_methods_dict
from pysrc.utils.enhanced_json_encoder import EnhancedJSONEncoder
from collections import defaultdict

def sample_csv(file_path: str,
               sampling_method: str,
               sampling_rate: float,
               config: GlobalConfiguration) -> str:
    """Sample a single file with a certain method and rate
    and create a new tmp file. Returns the path to the sampled file.
    """

    samples: list[list[tuple[str, str, float]]] = []

    file_prefix = file_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    columns = defaultdict(list)

    with open(file_path, 'r') as f:
        #TODO Input abhängig machen, was passiert mit Headern
        reader = csv.reader(f, delimiter=';', quotechar='"', escapechar='\\')
        for row in reader:
            for i in range(len(row)):
                columns[i].append(row[i])

    for col in columns:

        if config.header:
            file_header = columns[col][0]
            data = columns[col][1:]

        num_entries = len(columns[col])
        num_samples = math.ceil(num_entries * sampling_rate)

        #rename files column specific
        new_file_name = f'{file_prefix}_{str(sampling_rate).replace(".", "")}_{sampling_method}_{col}.csv'
        new_file_path = os.path.join(os.getcwd(), config.tmp_folder, new_file_name)

        sampling_method_function = sampling_methods_dict[sampling_method]
        data = sampling_method_function(columns[col], num_samples, num_entries)

        with open(new_file_path, 'w') as file:
            writer = csv.writer(file)
            if config.header:
                writer.writerow()

            writer.writerows((item,) for item in data)
        out_tuple = [(new_file_path, sampling_method, sampling_rate)]
        samples.extend(out_tuple)

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
    print(tmp_folder)
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
    #TODO replace sample with baseline ind
    #Find clever way for column based sampling
    samples: list[list[tuple[str, str, float]]] = [
        [(src_file, 'None', 1.0)]
        for src_file
        in source_files
    ]

    # TODO new execution arm for the sampled data with new sample list. Create Function to give valuable Combinations
    # Sample each source file
    for i, file_path in enumerate(source_files):
        for sampling_method in config.sampling_methods:
            for sampling_rate in config.sampling_rates:
                # Sample
                new_file_name = sample_csv(file_path, sampling_method, sampling_rate, config)
                samples[i].extend((new_file_name, sampling_method, sampling_rate))


    # Build cartesian product of all possible file combinations
    configurations: list[MetanomeRunConfiguration] = []
    for file_combination_setup in itertools.product(*samples):
        file_combination: list[str]; used_sampling_methods: list[str]; used_sampling_rates: list[float]
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
