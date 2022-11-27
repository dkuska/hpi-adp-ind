import datetime
import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional

from pysrc.models.column_information import ColumnInformation
from pysrc.models.ind import IND


@dataclass(frozen=True)
class MetanomeRunConfiguration:
    """Contains configuration information about a Metanome run"""
    arity: str
    sampling_rates: list[float]
    sampling_methods: list[str]
    time: datetime.datetime

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

    def has_ind(self, other_ind: IND) -> bool:
        """This checks whether this object has an IND that is identical to the passed-in one,
        i.e. whether they are from the same table and column, but may differ by name"""
        # Check whether it's directly contained
        if other_ind in self.inds:
            return True
        clean_other_ind = IND(
            dependents=[
                ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                for column
                in other_ind.dependents
            ], referenced=[
                ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                for column
                in other_ind.referenced
            ])
        # Check whether cleaned other is directly contained
        if clean_other_ind in self.inds:
            return True

        clean_inds = [
            IND(
                dependents=[
                    ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                    for column
                    in ind.dependents
                ], referenced=[
                    ColumnInformation(table_name=column.table_name.split('__')[0], column_name=column.column_name)
                    for column
                    in ind.referenced
                ])
            for ind
            in self.inds
        ]
        # Check whether cleaned version is in cleaned version
        return clean_other_ind in clean_inds

    def __len__(self) -> int:
        return len(self.inds)

    def __iter__(self) -> Iterator[IND]:
        return self.inds.__iter__()


@dataclass(frozen=True)
class MetanomeRun:
    configuration: MetanomeRunConfiguration
    results: MetanomeRunResults


@dataclass(frozen=True)
class MetanomeRunBatch:
    runs: list[MetanomeRun]

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[MetanomeRun]:
        return self.runs.__iter__()

    @property
    def baseline(self) -> MetanomeRun:
        return next(run for run in self.runs if run.configuration.is_baseline)


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
            ind = IND(dependents=[dependant], referenced=[referenced])
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
            ind = IND(dependents=dependant_list, referenced=referenced_list)
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
    allowed_gb: int = 6

    # Construct Command
    file_name_list = ' '.join([f'"{file_name}"' for file_name in configuration.source_files])

    execute_str = f'java -Xmx{allowed_gb}g -cp {metanome_cli_path}:{algorithm_path} de.metanome.cli.App \
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


# For unary INDs, this method returns absolute counts for TP, FP, FN, etc.
def compare_csv_line_unary(inds: list[IND], baseline: MetanomeRunResults):
    tp, fp = 0, 0
    num_inds = len(inds)

    for ind in inds:
        if baseline.has_ind(ind):
        # if ind in baseline.inds:
            tp += 1
        else:
            fp += 1

    fn = len(baseline.inds) - tp

    if num_inds > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)    
        f1 = 2*(precision * recall)/(precision + recall) if recall + precision != 0 else float('nan')
    else:
        precision, recall, f1 = 0, 0, 0
        
    return tp, fp, fn, precision, recall, f1


# For nary INDs, this returns lists with counts for each arity
def compare_csv_line_nary(inds: list[IND], baseline: MetanomeRunResults):
    num_inds = len(inds)
    
    max_arity = max([ind.arity() for ind in baseline.inds])
 
    tp, fp = [0 for _ in range(max_arity)], [0 for _ in range(max_arity)]
    inds_per_arity = [0 for _ in range(max_arity)]
    for ind in baseline.inds:
        inds_per_arity[ind.arity() - 1] += 1
    
    for ind in inds:
        arity = ind.arity() - 1 # -1 to match list indices  
        if baseline.has_ind(ind):
            tp[arity] += 1
        else:
            fp[arity] += 1
    
    fn = [inds_per_arity[arity] - tp[arity] for arity in range(max_arity)]
    
    precision, recall, f1 = [0.0 for _ in range(max_arity)], [0.0 for _ in range(max_arity)], [0.0 for _ in range(max_arity)]
    for i in range(max_arity):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
                                    
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
                                 
        if recall[i] + precision[i] > 0:
            f1[i] = 2*(precision[i] * recall[i])/(precision[i] + recall[i])
        else:
            f1[i] = float('nan')       
    
    return tp, fp, fn, precision, recall, f1


def run_as_compared_csv_line(run: MetanomeRun, baseline: MetanomeRunResults) -> list[str]:
    sampled_file_paths = run.configuration.source_files
    sampled_file_names = [path.rsplit('/', 1)[-1].replace('.csv', '') for path in sampled_file_paths]

    file_names, methods, rates = [],[],[]
    for sampled_file in sampled_file_names:
        split_filename = sampled_file.split('__')
        split_metadata = []
        if len(split_filename) == 2:
            split_metadata = split_filename[1].split('_')
        split_filename = [split_filename[0]]
        if len(split_metadata) == 2:
            split_filename.append(split_metadata[0])
            split_filename.append(split_metadata[1])
        if len(split_filename) == 3:
            fname, sampling_rate, sampling_method = split_filename
            sampling_rate = sampling_rate[0] + '.' + sampling_rate[1:]
        else:
            fname, sampling_rate, sampling_method  = sampled_file, '1.0', 'None'

        file_names.append(fname)
        methods.append(sampling_method)
        rates.append(sampling_rate)

    if run.configuration.arity == 'unary':
        tp, fp, fn, precision, recall, f1 = compare_csv_line_unary(run.results.inds, baseline)
        return ['; '.join(file_names), '; '.join(methods), '; '.join(rates), str(tp), str(fp), str(fn), f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}']
    else:
        tp, fp, fn, precision, recall, f1 = compare_csv_line_nary(run.results.inds, baseline)
        return ['; '.join(file_names), '; '.join(methods), '; '.join(rates), \
                '; '.join([str(tp_i) for tp_i in tp]), \
                '; '.join([str(fp_i) for fp_i in fp]), \
                '; '.join([str(fn_i) for fn_i in fn]), \
                '; '.join([f'{precision_i:.3f}' for precision_i in precision]), \
                '; '.join([f'{recall_i:.3f}' for recall_i in recall]), \
                '; '.join([f'{f1_i:.3f}' for f1_i in f1])]

