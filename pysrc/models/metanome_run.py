import datetime
from dataclasses import dataclass
import json
import os
from typing import Iterator, Optional
from models.column_information import ColumnInformation

from models.ind import IND


@dataclass(frozen=True)
class MetanomeRunConfiguration:
    """Contains configuration information about a Metanome run"""
    arity: str
    sampling_rates: list[float]
    sampling_methods: list[str]
    time: datetime.date

    source_dir: str
    source_files: list[str]
    tmp_folder: str
    results_folder: str
    result_suffix: str
    output_folder: str
    output_file: str
    clip_output: bool
    header: bool
    print_inds: bool
    create_plots: bool

    is_baseline: bool


@dataclass(frozen=True)
class MetanomeRunResults:
    inds: list[IND]

    def __len__(self) -> int:
        return len(self.inds)

    def __iter__(self) -> Iterator[IND]:
        return self.inds.__iter__()


@dataclass(frozen=True)
class MetanomeRun:
    configuration: MetanomeRunConfiguration
    results: MetanomeRunResults


def parse_results(result_file_name: str, arity: str, results_folder: str, print_inds: bool) -> MetanomeRunResults:
    """Parses result file and returns run results"""
    ind_list: list[IND] = []
    lines: list[str] = []
    try:
        with open(os.path.join(os.getcwd(), results_folder, result_file_name), 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        return MetanomeRunResults(ind_list)

    for line in lines:
        line_json = json.loads(line)
        if arity == 'unary':
            dependant_raw = line_json['dependant']['columnIdentifiers'][0]
            dependant_table = dependant_raw['tableIdentifier'].rsplit('.', 1)[0]
            dependant_column = dependant_raw['columnIdentifier']
            dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)

            referenced_raw = line_json['referenced']['columnIdentifiers'][0]
            referenced_table = referenced_raw['tableIdentifier'].rsplit('.', 1)[0]
            referenced_column = referenced_raw['columnIdentifier']
            referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)

            # TODO: Figure out better way to identify inds. Is this parsing even necessary?
            ind = IND(dependants=[dependant], referenced=[referenced])
            # ind = f'{dependant_table}.{dependant_column} [= {referenced_table}.{referenced_column}'
        elif arity == 'nary':
            dependant_list: list[ColumnInformation] = []
            dependant_raw = line_json['dependant']['columnIdentifiers']
            for dependant_entry in dependant_raw:
                dependant_table = dependant_entry['tableIdentifier'].rsplit('.', 1)[0]
                dependant_column = dependant_entry['columnIdentifier']
                dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)
                # dependant_list.append(f'{dependant_table}.{dependant_column}')
                dependant_list.append(dependant)

            referenced_list: list[ColumnInformation] = []
            referenced_raw = line_json['referenced']['columnIdentifiers']
            for referenced_entry in referenced_raw:
                referenced_table = referenced_entry['tableIdentifier'].rsplit('.', 1)[0]
                referenced_column = referenced_entry['columnIdentifier']
                referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)
                # referenced_list.append(f'{referenced_table}.{referenced_column}')
                referenced_list.append(referenced)

            # ind = f'{" & ".join(dependant_list)} [= {" & ".join(referenced_list)}'
            ind = IND(dependants=dependant_list, referenced=referenced_list)
        else:
            continue

        ind_list.append(ind)

    if print_inds:
        print(ind_list)

    return MetanomeRunResults(ind_list)


def run_metanome(configuration: MetanomeRunConfiguration, output_fname: str) -> MetanomeRun:
    # Make these configurable
    metanome_cli_path = 'metanome-cli.jar'
    algorithm_path = 'BINDER.jar'
    algorithm_class_name = 'de.metanome.algorithms.binder.BINDERFile'
    separator = '\\;'
    output_rule = f'file:{output_fname}'

    # Construct Command
    file_name_list = ' '.join([f'"{file_name}"' for file_name in configuration.source_files])

    execute_str = f'java -cp {metanome_cli_path}:{algorithm_path} de.metanome.cli.App \
                    --algorithm {algorithm_class_name} \
                    --files {file_name_list} \
                    --separator {separator} \
                    --file-key INPUT_FILES \
                    --skip-differing-lines \
                    -o {output_rule} \
                    --algorithm-config DETECT_NARY:{"true" if configuration.arity == "nary" else "false"}'
    if configuration.clip_output:
        execute_str += ' | tail -n 2'
    # Run
    os.system(execute_str)
    # Parse
    result = parse_results(output_fname + configuration.result_suffix, configuration.arity, configuration.results_folder, configuration.print_inds)
    return MetanomeRun(configuration=configuration, results=result)


def run_as_compared_csv_line(run: MetanomeRun, baseline: MetanomeRunResults) -> list[str]:
    sampled_file_paths = run.configuration.source_files
    sampled_file_names = [path.rsplit('/', 1)[-1].replace('.csv', '') for path in sampled_file_paths]

    file_names, methods, rates = [],[],[]
    for sampled_file in sampled_file_names:
        split_filename = sampled_file.split('_')
        if len(split_filename) == 3:
            fname, sampling_rate, sampling_method = split_filename
            sampling_rate = sampling_rate[0] + '.' + sampling_rate[1:]
        else:
            fname, sampling_rate, sampling_method  = sampled_file, '1.0', 'None'

        file_names.append(fname)
        methods.append(sampling_method)
        rates.append(sampling_rate)

    tp, fp = 0, 0
    inds = run.results.inds
    num_inds = len(inds)

    for ind in inds:
        if ind in baseline.inds:
            tp += 1
        else:
            fp += 1

    fn = len(baseline.inds) - tp

    if num_inds > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)    
        f1 = 2*(precision * recall)/(precision + recall)
    else:
        precision, recall, f1 = 0, 0, 0

    return ['; '.join(file_names), '; '.join(methods), '; '.join(rates), tp, fp, fn, f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}']
