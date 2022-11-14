import os, csv, random, math, json, sys, datetime, uuid
import argparse
import itertools
import pandas as pd

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
output_file         = f'output_{arity}_{now.year}{now.month:02d}{now.day}_{now.hour}{now.minute:02d}{now.second:02d}.csv'
plot_folder         = 'plots/'

# # It does not really matter, how you set this parameter. Just needs to be globally defined for create_evaluation_result_csv to access it
# baseline_identifier = 'baseline_None_1'

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
        
    if header:
        file_header = data[0]
        data = data[1:]
        
    num_entries = len(data)
    num_samples = math.ceil(num_entries * sampling_rate)      

    new_file_name = file_prefix + '_' + str(sampling_rate).replace('.', '') + '_' + sampling_method + '.csv'
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

def call_metanome_cli(file_name_list, output_fname='', clip_output=clip_output):
    execute_str = f'java -cp metanome-cli.jar:BINDER.jar de.metanome.cli.App \
                    --algorithm de.metanome.algorithms.binder.BINDERFile \
                    --files {file_name_list} \
                    --separator \; \
                    --file-key INPUT_FILES \
                    --skip-differing-lines '
    if output_fname != '':
        execute_str += f' -o file:{output_fname}'     
    if arity == 'nary':
        execute_str += f' --algorithm-config DETECT_NARY:true'            
    if clip_output:
        execute_str += ' | tail -n 2'                
    os.system(execute_str)
    
## Parses result file and returns list with string of the form 'dependentTable.dependentColumn[=referencedTable.referencedColumn'
def parse_results(result_file):
    ind_list = []
    lines = []
    try:
        with open(os.path.join(os.getcwd(), results_folder, result_file), 'r') as file:
            lines = file.readlines()
    except FileNotFoundError as e:
        return ind_list

    for line in lines:
        line_json = json.loads(line)
        if arity == 'unary':
            dependant = line_json['dependant']['columnIdentifiers'][0]
            dependant_table = dependant['tableIdentifier'].rsplit('.', 1)[0]
            dependant_column = dependant['columnIdentifier']
            
            referenced = line_json['referenced']['columnIdentifiers'][0]
            referenced_table = referenced['tableIdentifier'].rsplit('.', 1)[0]
            referenced_column = referenced['columnIdentifier']
            
            # TODO: Figure out better way to identify inds. Is this parsing even necessary?
            ind = f'{dependant_table}.{dependant_column} [= {referenced_table}.{referenced_column}'
        elif arity == 'nary':
            dependant_list = []
            dependant = line_json['dependant']['columnIdentifiers']
            for dependant_entry in dependant:
                dependant_table = dependant_entry['tableIdentifier'].rsplit('.', 1)[0]
                dependant_column = dependant_entry['columnIdentifier']
                dependant_list.append(f'{dependant_table}.{dependant_column}')
            
            referenced_list = []
            referenced = line_json['referenced']['columnIdentifiers']
            for referenced_entry in referenced:
                referenced_table = referenced_entry['tableIdentifier'].rsplit('.', 1)[0]
                referenced_column = referenced_entry['columnIdentifier']
                referenced_list.append(f'{referenced_table}.{referenced_column}')

            ind = f'{" & ".join(dependant_list)} [= {" & ".join(referenced_list)}'    
        else:
            pass

        ind_list.append(ind)
    
    if print_inds:
        print(ind_list)
    
    return ind_list

### Here the dict of experiment results gets turned into a csv file
def create_evaluation_result_csv(eval, baseline_identifier, output_file):
    
    output_path = os.path.join(os.getcwd(), output_folder, output_file)
    
    baseline_inds = eval[baseline_identifier]
    baseline_num_inds = len(baseline_inds)
    
    with open(output_path, 'w') as csv_output:
        writer = csv.writer(csv_output, quoting=csv.QUOTE_ALL)
        writer.writerow(['sampled_files', 'sampling_method', "sampling_rate", 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        
        for key, inds in eval.items(): 
            sampled_file_paths = key.split(' ')
            sampled_file_names = [path.rsplit('/', 1)[1].replace('.csv', '') for path in sampled_file_paths]
            
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
            num_inds = len(inds)
            
            for ind in inds:
                if ind in baseline_inds: tp += 1
                else: fp += 1
            
            fn = baseline_num_inds - tp
            
            if num_inds > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)    
                f1 = 2*(precision * recall)/(precision + recall)
            else:
                precision, recall, f1 = 0,0,0
            
            writer.writerow(['; '.join(file_names), '; '.join(methods), '; '.join(rates), tp, fp, fn, f'{precision:.3f}', f'{recall:.3f}', f'{f1:.3f}'])

def clean_tmp_csv(tmp_folder):
    csv_files = [f for f in os.listdir(tmp_folder) if f.rsplit('.')[1] == 'csv']
    for tmp_file in csv_files:
        os.remove(os.path.join(os.getcwd(), tmp_folder, tmp_file))

def clean_results(results_folder):
    result_files = [f for f in os.listdir(results_folder)]
    for tmp_file in result_files:
        os.remove(os.path.join(os.getcwd(), results_folder, tmp_file))


# TODO: Actually implement this
def make_plots(output_file, plot_folder):
    df = pd.read_csv(os.path.join(os.getcwd(), output_folder, output_file))    
    pass

def run():
    clean_tmp_csv(tmp_folder)
    clean_results(results_folder)
    
    experiments = {}
    source_files = [os.path.join(os.getcwd(), source_dir, f) for f in os.listdir(os.path.join(os.getcwd(), source_dir)) if f.rsplit('.')[1] == 'csv']
    baseline_identifier = " ".join(source_files)
    
    samples = [[src_file] for src_file in source_files]
    ### Sample each source file with each sampling configuration
    for i, file_path in enumerate(source_files):
        for sampling_method in sampling_methods:
            for sampling_rate in sampling_rates:
                ### Sample
                new_file_name = sample_csv(file_path, sampling_method, sampling_rate)
                samples[i].append(new_file_name)
                
    ### Build cartesian product of all possible file combinations
    ### And run experiment for each
    for file_combination in itertools.product(*samples):
        current_files_str = " ".join(file_combination)
        
        output_fname = str(uuid.uuid4())
        if print_inds:
            print(f'current_files_str : {current_files_str}')
            print(f'output_fname   : {output_fname}')
        ### Execute
        call_metanome_cli(current_files_str, output_fname)
        experiments[current_files_str] = output_fname
        
    ### Parse all results
    evaluation_results = {}
    for experiment_identifier, output_fname in experiments.items():        
        evaluation_results[experiment_identifier] = parse_results(output_fname + result_suffix)
                
    ### Persist experiment identifiers    
    create_evaluation_result_csv(evaluation_results, baseline_identifier, output_file)

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