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

CLASSIFICATION = 'classification'
REGRESSION = 'regression'

def main(location):
    print(f'Parsing the {location} to obtain model and parameters.')
    models_data = json.load(open(location, 'r'))
    debugger = Debugger(models_data[DEBUG_STATUS])

    print(models_data)
    models_name = models_data[MODELS_NAME]
    output_directory = get_output_directory_name(models_data[OUTPUT_DIRECTORYS_LOCATION], models_name)    
    if(not os.path.isdir(output_directory)):
        os.makedirs(output_directory)

    nrows = get_optional(models_data, NUMBER_OF_ROWS)
    data_frame = InputOutputFrame(debugger, models_data[INPUT_LOCATION], models_data[OUTPUT_LOCATION], models_data[FOLDS_LOCATION], \
        models_data[MAX_SENTENCES_PER_AUTHOR], models_data[MIN_PADDING_PER_AUTHOR], nrows=nrows)

    print("Starting the experiments.")
    experiment = Experiment(debugger, output_directory,get_optional(models_data,RESULTS_IMPORT_LOCATION), data_frame, models_data[BALANCE_DATA], models_data[EXPERIMENTS_NAME], models_name,\
         models_data[RUN_IDENTIFICATOR], models_data[FOLDS], models_data[PREDICTION_TYPE], models_data[TARGETS], models_data[OPTIMIZATION_PARAMETERS],\
             models_data[PRINT_BATCH_STATUS], models_data[MAX_CONSTANT_F1], models_data[NUMBER_OF_EPOCHS], models_data[DECAY_RATE], models_data[DECAY_EPOCH],\
             models_data[VALIDATION_SET_PERCENTAGE], models_data[CUDA_DEVICE], models_data[USE_GPU], models_data[RANDOM_STATE])

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
    parser.add_argument('--info-location', help='File in which is the models name and parameters that define this experiment')
    args=parser.parse_args()
    main(args.info_location)
