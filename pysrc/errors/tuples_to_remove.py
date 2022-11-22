import csv
import itertools

from pysrc.models.errors import TuplesToRemove
from ..models import metanome_run


def tuples_to_remove(*, baseline_config: 'metanome_run.MetanomeRunConfiguration', experiment: 'metanome_run.MetanomeRun') -> None:
    """Get, for every IND, the absolute and relative numbers of tuples that have to be removed from the baseline such that this IND is valid.
    This adds TuplesToRemove to the errors list of each IND"""
    for ind in experiment.results.inds:
        # Format: (file_path, column_name)
        dependent_files: list[tuple[str, str]] = []
        for dependant in ind.dependents:
            dependant_name = dependant.table_name.split('__')[0]
            file_path = next(file for file in baseline_config.source_files if dependant_name in file)
            dependent_files.append((file_path, dependant.column_name))
        referenced_files: list[tuple[str, str]] = []
        for referenced in ind.referenced:
            referenced_name = referenced.table_name.split('__')[0]
            file_path = next(file for file in baseline_config.source_files if referenced_name in file)
            referenced_files.append((file_path, referenced.column_name))
        file_contents: dict[str, list[list[str]]] = {}
        for file in itertools.chain(dependent_files, referenced_files):
            if file[0] in file_contents:
                continue
            with open(file[0], 'r') as file_handle:
                reader = csv.reader(file_handle, delimiter=';')
                data = list(reader)
                file_contents[file[0]] = data
        dependent_data = data_from_files(dependent_files, file_contents)
        referenced_data = data_from_files(referenced_files, file_contents)
        dependent_entries_to_remove = check_tuple_pair(dependent_data=dependent_data, referenced_data=referenced_data)
        result = TuplesToRemove(absolute_tuples_to_remove=dependent_entries_to_remove, relative_tuples_to_remove=dependent_entries_to_remove / len(dependent_data))
        ind.errors.append(result)


def data_from_files(files: list[tuple[str, str]], file_contents: dict[str, list[list[str]]]) -> list[tuple[str,  ...]]:
    data_list: list[list[str]] = [
        [
            line[int(file[1][6:]) - 1]
            for line
            in file_contents[file[0]]
        ]
        for file
        in files]
    data: list[tuple[str,  ...]] = [entry for entry in zip(*data_list)]
    return data


def check_tuple_pair(*, dependent_data: list[tuple[str, ...]], referenced_data: list[tuple[str, ...]]) -> int:
    dependent_set = set(dependent_data)
    referenced_set = set(referenced_data)
    difference = dependent_set - referenced_set
    return len([dependent for dependent in dependent_data if dependent != '' and dependent in difference])
            
