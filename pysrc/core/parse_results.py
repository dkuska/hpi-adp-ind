import json
import os
from pysrc.models.column_information import ColumnInformation
from pysrc.models.errors import ErrorMetric, MissingValues
from pysrc.models.ind import IND
from pysrc.models.metanome_run_results import MetanomeRunResults


def parse_results(result_file_name: str, *, algorithm: str, arity: str, results_folder: str, print_inds: bool,
                  is_baseline: bool, header: bool) -> MetanomeRunResults:
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
        errors: list[ErrorMetric] = []
        if arity == 'unary' and is_baseline == True:
            dependant_raw = line_json['dependant']['columnIdentifiers'][0]
            dependant_table = dependant_raw['tableIdentifier'].rsplit('.', 1)[0]
            dependant_column = dependant_raw['columnIdentifier']
            dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)

            referenced_raw = line_json['referenced']['columnIdentifiers'][0]
            referenced_table = referenced_raw['tableIdentifier'].rsplit('.', 1)[0]
            referenced_column = referenced_raw['columnIdentifier']
            referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)

            if algorithm == 'PartialSPIDER':
                missing_values = line_json["missingValues"]
                errors.append(MissingValues(missing_values))
            # TODO: Figure out better way to identify inds. Is this parsing even necessary?
            ind = IND(dependents=[dependant], referenced=[referenced], errors=errors)

        elif arity == 'unary' and is_baseline == False:
            dependant_raw = line_json['dependant']['columnIdentifiers'][0]
            dependant_table = dependant_raw['tableIdentifier'].rsplit('.', 1)[0].split('__', 1)[0]
            dependant_column = dependant_raw['columnIdentifier'] if header else 'column' + str(dependant_raw['tableIdentifier'].rsplit('.', 1)[0].rsplit('_')[-1])
            dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)

            referenced_raw = line_json['referenced']['columnIdentifiers'][0]
            referenced_table = referenced_raw['tableIdentifier'].rsplit('.', 1)[0].split('__', 1)[0]
            referenced_column = referenced_raw['columnIdentifier'] if header else 'column' + str(referenced_raw['tableIdentifier'].rsplit('.', 1)[0].rsplit('_')[-1])
            referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)

            if algorithm == 'PartialSPIDER':
                missing_values = line_json["missingValues"]
                errors.append(MissingValues(missing_values))
            # TODO: Figure out better way to identify inds. Is this parsing even necessary?
            ind = IND(dependents=[dependant], referenced=[referenced], errors=errors)

        elif arity == 'nary' and is_baseline == True:
            dependant_list: list[ColumnInformation] = []
            dependant_raw = line_json['dependant']['columnIdentifiers']
            for dependant_entry in dependant_raw:
                dependant_table = dependant_entry['tableIdentifier'].rsplit('.', 1)[0]
                dependant_column = dependant_entry['columnIdentifier']
                dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)
                dependant_list.append(dependant)

            referenced_list: list[ColumnInformation] = []
            referenced_raw = line_json['referenced']['columnIdentifiers']
            for referenced_entry in referenced_raw:
                referenced_table = referenced_entry['tableIdentifier'].rsplit('.', 1)[0]
                referenced_column = referenced_entry['columnIdentifier']
                referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)
                referenced_list.append(referenced)

            ind = IND(dependents=dependant_list, referenced=referenced_list)

        elif arity == 'nary' and is_baseline == False:
            dependant_list = []
            dependant_raw = line_json['dependant']['columnIdentifiers']
            for dependant_entry in dependant_raw:
                dependant_table = dependant_entry['tableIdentifier'].rsplit('.', 1)[0].split('__', 1)[0]
                dependant_column = dependant_entry['columnIdentifier'] if header else 'column' + str(dependant_entry['tableIdentifier'].rsplit('.', 1)[0].rsplit('_')[-1])
                dependant = ColumnInformation(table_name=dependant_table, column_name=dependant_column)
                dependant_list.append(dependant)

            referenced_list = []
            referenced_raw = line_json['referenced']['columnIdentifiers']
            for referenced_entry in referenced_raw:
                referenced_table = referenced_entry['tableIdentifier'].rsplit('.', 1)[0].split('_', 1)[0]
                referenced_column = referenced_entry['columnIdentifier'] if header else 'column' + str(referenced_entry['tableIdentifier'].rsplit('.', 1)[0].rsplit('_')[-1])

                referenced = ColumnInformation(table_name=referenced_table, column_name=referenced_column)
                referenced_list.append(referenced)

            ind = IND(dependents=dependant_list, referenced=referenced_list)
        else:
            continue

        ind_list.append(ind)

    return MetanomeRunResults(ind_list)
