DELIMITER_COUNT = 25

def get_checkpoints_name(output_directory, models_identifier):
    return f'{output_directory}save_{models_identifier}.pt'

def get_predictions_file_name(output_directory, models_identifier, target, fold, run_identificator):
    return f'{output_directory}predictions_{target}_{fold}_{models_identifier}_{run_identificator}'

def get_output_directory_name(location, models_name):
    return  f'{location}mpr_{models_name}/'

def get_results_file_name(location):
    return f'{location}results.csv'

def delimiter():
    print('-'*DELIMITER_COUNT)

def save_results(predictions, true_output, output_file):
    with open(output_file, 'w+') as f:
        for pre_l, true_l in zip(predictions, true_output):
            f.write(f'{pre_l.cpu()[0][0]}, {pre_l.cpu()[0][1]}, {true_l.cpu().tolist()[0]}\n')
                