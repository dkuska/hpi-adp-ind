import os
from pysrc.core.parse_results import parse_results
from pysrc.models.metanome_run import MetanomeRun
from pysrc.models.metanome_run_configuration import MetanomeRunConfiguration
from pysrc.utils.descriptive_statistics import file_column_statistics


def run_metanome(configuration: MetanomeRunConfiguration, output_fname: str, pipe: bool) -> MetanomeRun:
    # TODO: Make these configurable
    if configuration.algorithm == 'BINDER':
        algorithm_path = 'BINDER.jar'
        algorithm_class_name = 'de.metanome.algorithms.binder.BINDERFile'
    elif configuration.algorithm == 'PartialSPIDER':
        if configuration.arity == 'nary':
            raise ValueError('SPIDER does not support n-ary INDs')
        algorithm_path = 'PartialSPIDER.jar'
        algorithm_class_name = 'de.metanome.algorithms.spider.SPIDERFile'
    else:
        raise ValueError(configuration.algorithm)

    metanome_cli_path = 'metanome-cli.jar'
    separator = '\\;'
    escape = '\\\\'
    output_rule = f'file:{output_fname}'
    allowed_gb = 6

    # Calculate File Statistics
    source_files_column_statistics = [stats for f in configuration.source_files
                                      for stats
                                      in file_column_statistics(f, header=configuration.header,
                                                                is_baseline=configuration.is_baseline)]

    # Construct Command
    # TODO: Beware that this might allow unsanitized code to be appended to the command.
    file_name_list = ' '.join([f'"{file_name}"' for file_name in configuration.source_files])

    execute_str = f'java -Xmx{allowed_gb}g -cp {metanome_cli_path}:{algorithm_path} de.metanome.cli.App \
                    --algorithm {algorithm_class_name} \
                    --files {file_name_list} \
                    --separator {separator} \
                    --file-key INPUT_FILES \
                    --skip-differing-lines \
                    -o {output_rule} \
                    --escape {escape} '

    if configuration.header:
        execute_str += '--header '

    if configuration.algorithm == 'BINDER':
        execute_str += f'--algorithm-config DETECT_NARY:{"true" if configuration.arity == "nary" else "false"}'
    else:
        execute_str += '--algorithm-config TEMP_FOLDER_PATH:SPIDER_temp,\
                        MAX_MEMORY_USAGE_PERCENTAGE:60,\
                        INPUT_ROW_LIMIT:-1,\
                        CLEAN_TEMP:true,\
                        MEMORY_CHECK_FREQUENCY:100'
        if not configuration.is_baseline:
            execute_str += f',MAX_NUMBER_MISSING_VALUES:{configuration.allowed_missing_values}'
        else:
            execute_str += ',MAX_NUMBER_MISSING_VALUES:0'
        
    if pipe:
        execute_str += ' | tail -n 0'
    elif configuration.clip_output:
        execute_str += ' | tail -n 2'

    # Run
    os.system(execute_str)
    # Parse
    result = parse_results(result_file_name=output_fname + configuration.result_suffix,
                           algorithm=configuration.algorithm,
                           arity=configuration.arity,
                           results_folder=configuration.results_folder,
                           is_baseline=configuration.is_baseline,
                           header=configuration.header)
    return MetanomeRun(configuration=configuration, column_statistics=source_files_column_statistics, results=result)
