import csv
from models.ind import IND
from models.metanome_run import MetanomeRun, MetanomeRunConfiguration


def tuples_to_remove(*, baseline_config: MetanomeRunConfiguration, experiment: MetanomeRun) -> dict[IND, int]:
    result: dict[IND, int] = {}
    for ind in experiment.results.inds:
        # Format: (file_path, column_name)
        dependents_files: list[tuple[str, str]] = []
        for dependant in ind.dependents:
            dependant_name = dependant.table_name.split('_')[0]
            file_path = next(file for file in baseline_config.source_files if dependant_name in file)
            dependents_files.append((file_path, dependant.column_name))
        referenced_files: list[tuple[str, str]] = []
        for referenced in ind.referenced:
            referenced_name = referenced.table_name.split('_')[0]
            file_path = next(file for file in baseline_config.source_files if referenced_name in file)
            referenced_files.append((file_path, referenced.column_name))
        dependent_entries_to_remove = 0
        for pair in zip(dependents_files, referenced_files):
            dependent_data: list[str]
            with open(pair[0][0], 'r') as dependent_file:
                reader = csv.reader(dependent_file, delimiter=';')
                dependant_column = int(pair[0][1][6:]) - 1  # This is 1-based from BINDER
                data = list(reader)
                dependent_data = [line[dependant_column] for line in data]
            referenced_data: list[str]
            with open(pair[0][0], 'r') as referenced_file:
                reader = csv.reader(referenced_file, delimiter=';')
                dependant_column = int(pair[0][1][6:]) - 1  # This is 1-based from BINDER
                data = list(reader)
                referenced_data = [line[dependant_column] for line in data]
            dependent_entries_to_remove += check_tuple_pair(dependent_data=dependent_data, referenced_data=referenced_data)
        result[ind] = dependent_entries_to_remove
    return result   


def check_tuple_pair(*, dependent_data: list[str], referenced_data: list[str]) -> int:
    dependent_values_to_remove = 0
    for dependant_value in dependent_data:
        if dependant_value not in referenced_data:
            dependent_values_to_remove += 1
    return dependent_values_to_remove
            
