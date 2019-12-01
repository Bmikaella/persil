import sys, argparse
import parser
import json
from data_operators import ModelPerformanceSaver, InputOutputFrame
from helpers import * 
from experiment import *
import os
from debugger import Debugger


OUTPUT_DIRECTORYS_LOCATION = 'output_directorys_location'

INPUT_LOCATION = 'input_location'
FOLDS_LOCATION = 'folds_location'
OUTPUT_LOCATION = 'output_location'

CUDA_DEVICE = 'cuda_device'
RANDOM_STATE = 'random_state'
USE_GPU  ='use_GPU'
VALIDATION_SET_PERCENTAGE = 'validation_set_percentage'
MAX_SENTENCES_PER_AUTHOR = 'max_sentences_per_author'
MIN_PADDING_PER_AUTHOR = 'min_padding_per_author'
PRINT_BATCH_STATUS = 'print_status_batch'     
MAX_CONSTANT_F1 = 'max_constant_f1'
NUMBER_OF_EPOCHS = 'n_epochs'
NUMBER_OF_ROWS = 'nrows'

MODELS_NAME = 'models_name'
EXPERIMENTS_NAME = 'experiments_name'
TARGETS = 'targets'
PREDICTION_TYPE = 'prediction_type'

FOLDS = 'folds'
RUN_IDENTIFICATOR = 'run_identificator'
BALANCE_DATA = 'balance_data'

RESULTS_IMPORT_LOCATION = 'results_import_location'
  
OPTIMIZATION_PARAMETERS = 'optimization_params'

DECAY_RATE  = "decay_rate"
DECAY_EPOCH = "decay_epoch"

DEBUG_STATUS = 'debug'

def main(experiment_meta_data, data_info):
    print(f'Parsing the {experiment_meta_data} to obtain model and parameters.')
    experiments_data = json.load(open(experiment_meta_data, 'r'))
    dataset_meta = json.load(open(data_info, 'r'))
    debugger = Debugger(experiments_data[DEBUG_STATUS])

    print(experiments_data)
    print(dataset_meta)

    data_frame = InputOutputFrame(debugger, dataset_meta[INPUT_LOCATION], dataset_meta[OUTPUT_LOCATION], dataset_meta[FOLDS_LOCATION], \
        dataset_meta[MAX_SENTENCES_PER_AUTHOR], dataset_meta[MIN_PADDING_PER_AUTHOR], nrows=get_optional(dataset_meta, NUMBER_OF_ROWS))

    models_name = experiments_data[MODELS_NAME]
    output_directory = get_output_directory_name(experiments_data[OUTPUT_DIRECTORYS_LOCATION], models_name)    
    if(not os.path.isdir(output_directory)):
        os.makedirs(output_directory)

    print("Starting the experiments.")
    experiment = Experiment(debugger, output_directory,get_optional(experiments_data,RESULTS_IMPORT_LOCATION), data_frame, experiments_data[BALANCE_DATA], experiments_data[EXPERIMENTS_NAME], models_name,\
         experiments_data[RUN_IDENTIFICATOR], experiments_data[FOLDS], experiments_data[PREDICTION_TYPE], experiments_data[TARGETS], experiments_data[OPTIMIZATION_PARAMETERS],\
             experiments_data[PRINT_BATCH_STATUS], experiments_data[MAX_CONSTANT_F1], experiments_data[NUMBER_OF_EPOCHS], experiments_data[DECAY_RATE], experiments_data[DECAY_EPOCH],\
             experiments_data[VALIDATION_SET_PERCENTAGE], experiments_data[CUDA_DEVICE], experiments_data[USE_GPU], experiments_data[RANDOM_STATE])

    experiment.start()
    print("Experiment finished!")
    #parse the parameters
    #prepare data
    #prepara a performance saver
    #choose model
    #clean and prepare the data 
    #optimeze hyper parameters 
    #-> train model
    #choose the best model and apply
    

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-info', help='File in which is the models data is described')
    parser.add_argument('--experiment-meta-data', help='File in which is the models name and parameters that define this experiment')
    args=parser.parse_args()
    main(args.experiment_meta_data, args.data_info)
