import csv
import itertools
from ..models.ind import IND
from ..models import metanome_run


def tuples_to_remove(*, baseline_config: 'metanome_run.MetanomeRunConfiguration', experiment: 'metanome_run.MetanomeRun') -> dict[IND, int]:
    result: dict[IND, int] = {}
    for ind in experiment.results.inds:
        # Format: (file_path, column_name)
        dependents_files: list[tuple[str, str]] = []
        for dependant in ind.dependents:
            dependant_name = dependant.table_name.split('__')[0]
            file_path = next(file for file in baseline_config.source_files if dependant_name in file)
            dependents_files.append((file_path, dependant.column_name))
        referenced_files: list[tuple[str, str]] = []
        for referenced in ind.referenced:
            referenced_name = referenced.table_name.split('__')[0]
            file_path = next(file for file in baseline_config.source_files if referenced_name in file)
            referenced_files.append((file_path, referenced.column_name))
        dependent_entries_to_remove = 0
        file_contents: dict[str, list[list[str]]] = {}
        for file in itertools.chain(dependents_files, referenced_files):
            if file[0] in file_contents:
                continue
            with open(file[0], 'r') as file_handle:
                print(f'Read from file {file[0]}')
                reader = csv.reader(file_handle, delimiter=';')
                data = list(reader)
                file_contents[file[0]] = data
        for pair in zip(dependents_files, referenced_files):
            dependant_column = int(pair[0][1][6:]) - 1  # This is 1-based from BINDER
            dependent_data: list[str] = [line[dependant_column] for line in file_contents[pair[0][0]]]
            referenced_column = int(pair[1][1][6:]) - 1  # This is 1-based from BINDER
            referenced_data: list[str] = [line[referenced_column] for line in file_contents[pair[1][0]]]
            dependent_entries_to_remove += check_tuple_pair(dependent_data=dependent_data, referenced_data=referenced_data)
        result[ind] = dependent_entries_to_remove
    return result


def check_tuple_pair(*, dependent_data: list[str], referenced_data: list[str]) -> int:
    dependent_set = set(dependent_data)
    referenced_set = set(referenced_data)
    difference = dependent_set - referenced_set
    return len([dependent for dependent in dependent_data if dependent != '' and dependent in difference])
            
