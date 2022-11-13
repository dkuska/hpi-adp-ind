import csv
import datetime
import itertools
import json
import math
import os
import random
import uuid

import pandas as pd

from enhanced_json_encoder import EnhancedJSONEncoder
from metanome_run import (MetanomeRun, MetanomeRunConfiguration,
                          run_as_compared_csv_line, run_metanome)

## GLOBAL CONFIGURATION PARAMETERS
# Sampling settings
arity               = ['unary', 'nary'][0]
# sampling_rates      = [0.1, 0.01, 0.001]
# sampling_methods    = ['random', 'first', 'evenly-spaced']
sampling_rates      = [0.1]
sampling_methods    = ['evenly-spaced']

header              = False
clip_output         = True
print_inds          = False
create_plots        = False

# Paths, these dirs are assumed to already exist
now = datetime.datetime.now()

source_dir          = 'src/'
tmp_folder          = 'tmp/'
results_folder      = 'results/'
result_suffix       = '_inds'
output_folder       = 'output/'
output_file         = f'output_{arity}_{now.year}{now.month:02d}{now.day}_{now.hour}{now.minute:02d}{now.second:02d}'
plot_folder         = 'plots/'

# # It does not really matter, how you set this parameter. Just needs to be globally defined for create_evaluation_result_csv to access it
# baseline_identifier = 'baseline_None_1'

# TODO: Add support for headers
# TODO: Add support for more sampling methods
def sample_csv(file_path: str, sampling_method: str, sampling_rate: float) -> str:
    """Sample a single file with a certain method and rate and create a new tmp file"""
    data: list[list[str]] = []
    file_prefix = file_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    # Read data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        
    if header:
        file_header = data[0]
        data = data[1:]
        
    num_entries = len(data)
    num_samples = math.ceil(num_entries * sampling_rate)      

    new_file_name = f'{file_prefix}_{str(sampling_rate).replace(".", "")}_{sampling_method}.csv'
    new_file_path = os.path.join(os.getcwd(), tmp_folder, new_file_name)
    
    if sampling_method == 'random':
        data = random.sample(data, k=num_samples)
    elif sampling_method == 'first':
        data = data[:num_samples]
    elif sampling_method == 'evenly-spaced':
        space_width = math.ceil(num_entries / num_samples)
        starting_index = random.randint(0, space_width)
        data = [data[i%num_entries] for i in range(starting_index, num_entries+space_width, space_width)]
    elif sampling_method == 'kmeans':
        pass # TODO: implement this
    else:
        pass
        
    with open(new_file_path, 'w') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(file_header)
        
        writer.writerows(data)

    return new_file_path


def create_evaluation_result(eval: list[MetanomeRun], output_file: str, write_json: bool=True) -> None:
    """Here the list of experiment results gets turned into a csv and (optionally) a json file"""
    
    output_path = os.path.join(os.getcwd(), output_folder, output_file)
    output_csv = f'{output_path}.csv'
    output_json = f'{output_path}.json'
    
    with open(output_csv, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['sampled_files', 'sampling_method', "sampling_rate", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])

        baseline: MetanomeRun = next(run for run in eval if run.configuration.is_baseline)
        
        for run in eval:
            writer.writerow(run_as_compared_csv_line(run, baseline.results))

    if write_json:
        with open(output_json, 'w', encoding='utf-8') as json_output:
            json.dump(eval, json_output, ensure_ascii=False, indent=4, cls=EnhancedJSONEncoder)


def clean_tmp_csv(tmp_folder):
    csv_files = [f for f in os.listdir(tmp_folder) if f.rsplit('.')[1] == 'csv']
    for tmp_file in csv_files:
        os.remove(os.path.join(os.getcwd(), tmp_folder, tmp_file))


def clean_results(results_folder: str) -> None:
    result_files = [f for f in os.listdir(results_folder)]
    for tmp_file in result_files:
        os.remove(os.path.join(os.getcwd(), results_folder, tmp_file))


# TODO: Actually implement this
def make_plots(output_file: str, plot_folder: str):
    df = pd.read_csv(os.path.join(os.getcwd(), output_folder, output_file))    
    pass


def run():
    clean_tmp_csv(tmp_folder)
    clean_results(results_folder)

    experiments: list[MetanomeRun] = []
    source_files = [os.path.join(os.getcwd(), source_dir, f) for f in os.listdir(os.path.join(os.getcwd(), source_dir)) if f.rsplit('.')[1] == 'csv']
    baseline_identifier = " ".join(source_files)

    samples: list[list[tuple[str, str, float]]] = [[(src_file, 'None', 1.0)] for src_file in source_files]
    ### Sample each source file with each sampling configuration
    for i, file_path in enumerate(source_files):
        for sampling_method in sampling_methods:
            for sampling_rate in sampling_rates:
                ### Sample
                new_file_name = sample_csv(file_path, sampling_method, sampling_rate)
                samples[i].append((new_file_name, sampling_method, sampling_rate))

    ### Build cartesian product of all possible file combinations
    configurations: list[MetanomeRunConfiguration] = []
    for file_combination_setup in itertools.product(*samples):
        file_combination: list[str]; used_sampling_methods: list[str]; used_sampling_rates: list[float]
        file_combination, used_sampling_methods, used_sampling_rates = zip(*file_combination_setup)
        configurations.append(MetanomeRunConfiguration(
            arity=arity,
            sampling_rates=used_sampling_rates,
            sampling_methods=used_sampling_methods,
            time=now,
            source_dir=source_dir,
            source_files=file_combination,
            tmp_folder=tmp_folder,
            results_folder=results_folder,
            result_suffix=result_suffix,
            output_folder=output_folder,
            output_file=output_file,
            clip_output=clip_output,
            header=header,
            print_inds=print_inds,
            create_plots=create_plots,
            is_baseline=' '.join(file_combination) == baseline_identifier))

    ### And run experiment for each
    for config in configurations:
        current_files_str = " ".join(config.source_files)

        output_fname = str(uuid.uuid4())
        if config.print_inds:
            print(f'current_files_str : {current_files_str}')
            print(f'output_fname   : {output_fname}')
        ### Execute
        result = run_metanome(config, output_fname)
        experiments.append(result)

    ### Persist experiment identifiers    
    create_evaluation_result(experiments, output_file)

    ### Clean up tmp and results for good measure
    clean_tmp_csv(tmp_folder)
    clean_results(results_folder)

    if create_plots:
        make_plots(output_file, plot_folder)

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ratio", type=float)
#     args = parser.parse_args()

if __name__ == "__main__":
    run()
