import sys, argparse
import parser
import json
from persil.data_operators import ModelPerformanceSaver, InputOutputFrame
from persil.helpers import * 
from persil.experiment import *
from persil.model_builder import * 

OUTPUT_DIRECTORYS_LOCATION = 'output_directorys_location'

INPUT_LOCATION = 'input_location'
FOLDS_LOCATION = 'folds_location'
OUTPUT_LOCATION = 'output_location'

NUMBER_OF_CLASSES = 'number_of_classes'
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
MBTI_TRAITS = 'mbti_traits'
FOLDS = 'folds'
RUN_IDENTIFICATOR = 'run_identificator'

RESULTS_IMPORT_LOCATION = '/home/mibo/results_FINAL_all_traits.csv'
  
OPTIMIZATION_PARAMETERS = 'optimization_params '

DECAY_RATE  = "decay_rate"
DECAY_EPOCH = "decay_epoch"

def main(location):
    print(f'Parsing the {location} to obtain model and parameters ...')
    models_data = json.load(open(location, 'r'))
    
    models_name = models_data[MODELS_NAME]
    output_directory = get_output_directory_name(models_data[OUTPUT_DIRECTORYS_LOCATION], models_name)    

    parameters = models_data[OPTIMIZATION_PARAMETERS]

    models_performance_saver = ModelPerformanceSaver(columns=parameters.keys(), id_columns=parameters.keys(), save_location=get_results_file_name(output_directory), \
        import_location=models_name[RESULTS_IMPORT_LOCATION])
    
    data_frame = InputOutputFrame(models_data[INPUT_LOCATION], models_data[OUTPUT_LOCATION], models_data[FOLDS_LOCATION], \
        models_data[MAX_SENTENCES_PER_AUTHOR], models_data[MIN_PADDING_PER_AUTHOR], nrows=models_data[NUMBER_OF_ROWS])

    experiment = Experiment(output_directory, models_performance_saver, data_frame, models_data[EXPERIMENTS_NAME], models_name,\
         models_data[RUN_IDENTIFICATOR], models_data[FOLDS], models_data[MBTI_TRAITS], models_data[OPTIMIZATION_PARAMETERS],\
             models_data[PRINT_BATCH_STATUS], models_data[MAX_CONSTANT_F1], models_data[NUMBER_OF_EPOCHS], models_data[DECAY_RATE], models_data[DECAY_EPOCH],\
                 models_data[NUMBER_OF_CLASSES], models_data[VALIDATION_SET_PERCENTAGE], models_data[CUDA_DEVICE], models_data[USE_GPU], models_data[RANDOM_STATE])

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
    parser.add_argument('--info-location', help='File in which the models name and parameters that define this experiment')
    args=parser.parse_args()
    main(sys.argv[1])