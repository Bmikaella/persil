from collections import namedtuple
import pdb


DELIMITER_COUNT = 25

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
SINGLE_TARGET = 'singletarget'
MULTI_TARGET = 'multitarget'
ONE_TARGET = 1

RELATIVE_DIFFERENCE = 0.00001

def get_optional(parameters, name):
    value = parameters[name]
    return None if value == "None" else value

def get_checkpoints_name(output_directory, models_identifier):
    return f'{output_directory}save_{models_identifier}.pt'

def get_predictions_file_name(output_directory, models_identifier, target, fold, run_identificator):
    return f'{output_directory}predictions_{target}_{fold}_{models_identifier}_{run_identificator}'

def get_output_directory_name(location, models_name):
    return  f'{location}mpr_{models_name}/'

def get_results_file_name(location, run_identificator):
    return f'{location}results_{run_identificator}.csv'

def delimiter():
    print('-'*DELIMITER_COUNT)

def save_results(predictions, true_output, output_file, prediction_type):
    # TODO fix for multitask
    if(prediction_type == 'classification'):
        with open(output_file, 'w+') as f:
            for pre_l, true_l in zip(predictions, true_output):
                f.write(f'{pre_l.cpu()[0][0]}, {pre_l.cpu()[0][1]}, {true_l.cpu().tolist()[0]}\n')
    elif(prediction_type == 'regression'):
        with open(output_file, 'w+') as f:
            for pre_l, true_l in zip(predictions, true_output):
                f.write(f'{pre_l.cpu()[0][0]}, {true_l.cpu().tolist()[0][0]}\n')


# used to determine the type of task we are handling, so that we can know what kind of loss we will use 
def determine_target_type(targets):
    # we are only checking the first of n targets as we only support constatnt singletarget or multitarget runs in one experiment
    if len(targets) == ONE_TARGET:
        return SINGLE_TARGET
    else:
        return MULTI_TARGET
    
# used to determine the output of the model
def determine_classes(data_frame, targets, targets_type, prediction_type):
    if(prediction_type == REGRESSION):
        if targets_type == MULTI_TARGET:
            return [1] * len(targets[0])
        return 1
    if targets_type == SINGLE_TARGET:
        return determine_singletarget_classes(data_frame, targets[0])
    else: 
        return determine_multitarget_classes(data_frame, targets)

# these two are used to determine the arhitecture of the model, we need to know what is the ouput 
def determine_singletarget_classes(data_frame, targets):
    return len(data_frame.get_all_column_unique_values(targets[0]))

def determine_multitarget_classes(data_frame, targets):
    classes_count = []
    for target in targets: 
        classes_count.append(determine_singletarget_classes(data_frame, target))
    return classes_count
 