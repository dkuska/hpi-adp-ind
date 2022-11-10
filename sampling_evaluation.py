import os, csv, random, math, json, sys, datetime

## GLOBAL CONFIGURATION PARAMETERS
# Paths, these dirs are assumed to already exist
now = datetime.datetime.now()

source_dir = 'src/'
tmp_folder = 'tmp/'
results_folder = 'results/'
result_suffix = '_inds'
output_folder = 'output/'
output_file = f'output_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.csv'
clip_output = True

# Sampling settings
sampling_mode = ['single_file', 'all_files'][0] #TODO: Actually implement this...
sampling_rates = [0.1, 0.01, 0.001]
sampling_methods = ['random', 'first']

# sampling_rates = [0.1]
# sampling_methods = ['random']

# It does not really matter, how you set this parameter. Just needs to be globally defined for create_evaluation_result_csv to access it
baseline_identifier = 'baseline_None_1'



## Sample a single file with a certain method and rate and create a new tmp file
# TODO: Add support for headers
# TODO: Add support for more sampling methods
def sample_csv(file_path, sampling_method, sampling_rate):
    data = []
    file_prefix = file_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    # Read data
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
        
    num_entries = len(data)
    num_samples = math.ceil(num_entries * sampling_rate)      

    new_file_name = file_prefix + '_' + str(sampling_rate).replace('.', '') + '_' + sampling_method + '.csv'
    new_file_path = os.path.join(os.getcwd(), tmp_folder, new_file_name)
    
    if sampling_method == 'random':
        data = random.sample(data, k=num_samples)
    
    if sampling_method == 'first':
        data = data[:num_samples]
        
    with open(new_file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    return new_file_path

def call_metanome_cli(file_name_list, output_fname='', clip_output=clip_output):
    execute_str = f'java -cp metanome-cli.jar:BINDER.jar de.metanome.cli.App \
                    --algorithm de.metanome.algorithms.binder.BINDERFile \
                    --files {file_name_list} \
                    --separator \; \
                    --file-key INPUT_FILES \
                    --skip-differing-lines'
    if output_fname != '':
        execute_str += f' -o file:{output_fname}'                     
                    
    if clip_output:
        execute_str += '| tail -n 2'                
    os.system(execute_str)
    
## Parses result file and returns list with string of the form 'dependentTable.dependentColumn[=referencedTable.referencedColumn'
def parse_results(result_file):

    return_list = []
    lines = []
    with open(os.path.join(os.getcwd(), results_folder, result_file), 'r') as file:
        lines = file.readlines()

    for line in lines:
        line_json = json.loads(line)
        
        dependant = line_json['dependant']['columnIdentifiers'][0]
        dependant_table = dependant['tableIdentifier'].rsplit('.')[0]
        dependant_column = dependant['columnIdentifier']
        
        referenced = line_json['referenced']['columnIdentifiers'][0]
        referenced_table = referenced['tableIdentifier'].rsplit('.')[0]
        referenced_column = referenced['columnIdentifier']
        
        # TODO: Figure out better way to identify inds. Is this parsing even necessary?
        ind = f'{dependant_table}.{dependant_column}[={referenced_table}.{referenced_column}'
        return_list.append(ind)
    
    return return_list

### Here the dict of experiment results gets turned into a csv file
def create_evaluation_result_csv(eval, baseline_identifier, output_file):
    
    output_path = os.path.join(os.getcwd(), output_folder, output_file)
    
    baseline_entry = eval[baseline_identifier]
    
    with open(output_path, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['sampled_file', 'sampling_method', "sampling_rate", 'num_inds', 'tp', 'fp', 'fn'])
        
        for key, item in eval.items(): 
            sampled_file, sampling_method, sampling_rate = key.split('_')

            num_inds = len(item)
            tp = 0
            fp = 0
    
            for i in item:
                if i in baseline_entry: tp += 1
                else: fp =+ 1
            
            writer.writerow([sampled_file, sampling_method, sampling_rate, num_inds, tp, fp, len(baseline_entry) - tp])

def clean_tmp_csv(tmp_folder):
    csv_files = [f for f in os.listdir(tmp_folder) if f.rsplit('.')[1] == 'csv']
    for tmp_file in csv_files:
        os.remove(os.path.join(os.getcwd(), tmp_folder, tmp_file))

def clean_results(results_folder):
    result_files = [f for f in os.listdir(results_folder)]
    for tmp_file in result_files:
        os.remove(os.path.join(os.getcwd(), results_folder, tmp_file))


# TODO: Actually implement this
def create_plots():
    pass

def run():
    evaluation_results = {}
    
    source_files = [os.path.join(os.getcwd(), source_dir, f) for f in os.listdir(os.path.join(os.getcwd(), source_dir)) if f.rsplit('.')[1] == 'csv']
    ## No Sampling - Baseline
    current_files_str = ' '.join(source_files)
    call_metanome_cli(current_files_str, output_fname='baseline')
    evaluation_results[baseline_identifier] = parse_results('baseline' + result_suffix)
    
    clean_results(results_folder)

    ### Sample each file
    for file_path in source_files:
        for sampling_method in sampling_methods:
            for sampling_rate in sampling_rates:
                ### Sample
                new_file_name = sample_csv(file_path, sampling_method, sampling_rate)
                all_but_current_files = [os.path.join(source_dir, file) for file in source_files if file != file_path]
                current_files_str = ' '.join(all_but_current_files)
                current_files_str += ' ' + new_file_name
                
                output_fname = 'experiment_result'
                
                ### Execute
                call_metanome_cli(current_files_str, output_fname)
                ### Parse Results
                # print(f'file_path: {file_path}')
                # print(f'sampling_method: {sampling_method}')
                # print(f'sampling_rate: {sampling_rate}')
                experiment_identifier = f'{file_path.rsplit("/", 1)[1].rsplit(".", 1)[0]}_{sampling_method}_{sampling_rate}'
                evaluation_results[experiment_identifier] = parse_results(output_fname + result_suffix)
                
                ### Clean up results
                clean_results(results_folder)
                
    ### Persist experiment identifiers    
    create_evaluation_result_csv(evaluation_results, baseline_identifier, output_file)

    ### Clean up tmp and results for good measure
    clean_tmp_csv(tmp_folder)
    clean_results(results_folder)

if __name__ == "__main__":
    run()