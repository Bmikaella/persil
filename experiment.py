import helpers
from data_operators import *
from model_operator import * 
from itertools import product
from models import *
from regularization import * 
import torch as to
from torch.optim.lr_scheduler import StepLR
from model_operator import *
import time
from metrics import *

class Experiment:

    LEARNING_RATE = 'learning_rate'
    BATCH_SIZE = 'batch_size'
    REGULARIZATION = 'regularization_type'
    ALPHA = 'alpha'  

    def __init__(self, debugger, output_directory, old_results_location, data_frame, balance_data, experiments_name, \
        models_name, run_identificator, folds, prediction_type, targets, optimization_parameters, print_status_batch, \
            max_constant_f1, number_of_epochs,\
        decay_rate, decay_epoch, validation_set_percentage, cuda_device_index=1, use_GPU=True, random_state=56):
        self.debugger = debugger 
        self.print_status_batch = print_status_batch 
        self.random_state = random_state
        self.max_constant_f1 = max_constant_f1 
        self.number_of_epochs = number_of_epochs
        self.validation_set_percentage = validation_set_percentage 
        self.output_directory = output_directory
        self.data_frame = data_frame
        self.balance_data = balance_data
        self.folds = folds 
        self.targets = targets
        self.targets_type = determine_target_type(targets[0])
    
        
        self.experiments_name = experiments_name
        self.models_name = models_name 
        self.run_identificator = run_identificator  
        
        self.optimization_parameters = optimization_parameters 
        self.number_of_classes = determine_classes(data_frame, targets, self.targets_type, prediction_type)

        self.prediction_type = prediction_type

        self.models_performance_saver = ModelPerformanceSaver(debugger, columns=list(optimization_parameters.keys()), \
            id_columns=list(optimization_parameters.keys()), save_location=get_results_file_name(output_directory, run_identificator), \
            number_of_classes=self.number_of_classes, prediction_type=prediction_type, import_location=old_results_location)
        
        if (prediction_type == CLASSIFICATION):
            self.debugger.print(f"Usli smo u klasifikaciju a trazili smo {prediction_type}")
            self.metrics_handler = ClassificationMetricsHandler(self.debugger, self.models_performance_saver, self.number_of_classes)
        elif (prediction_type == REGRESSION):
            self.debugger.print(f"Usli smo u regresiju a trazili smo {prediction_type}")
            self.metrics_handler = RegressionMetricsHandler(self.debugger, self.models_performance_saver)
        else: 
            raise Exception(f"Please check tag 'prediction_type' as value {prediction_type} is not supported")

        if (prediction_type == CLASSIFICATION):
            self.loss_function = nn.CrossEntropyLoss()
        elif (prediction_type == REGRESSION):
            self.loss_function = nn.MSELoss()
        
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        
        
        self.cuda_device = to.device(cuda_device_index) if use_GPU == 'True' else None
    
    def start(self):
        
        delimiter()
        print(f"Starting experiment {self.experiments_name}...")
        delimiter()

        for fold in self.folds:
            for target in self.targets:
            
                print(f'Processing fold: {fold} and target: {target}')
                delimiter()

                train_input_indices, train_output, val_input_indices, val_output, test_input_indices, test_output = self.data_frame.\
                    get_train_val_test_input_output(target, fold, self.validation_set_percentage, self.random_state, self.targets_type, self.prediction_type)
        

                model, models_identifier, parameters_dict = self.get_the_best_model(target, fold, train_input_indices, train_output, val_input_indices, val_output)
                
                print(f'Apply best model to test for {target} on fold {fold}.')
                test_loader = self.data_frame.create_minibatches(test_input_indices, test_output, 1, self.cuda_device, model.convert_input)

                regularization = get_regularization(parameters_dict[self.REGULARIZATION])
                alpha = parameters_dict[self.ALPHA]
                test_loss, test_logits, test_true = apply(self.debugger, model, self.loss_function, test_loader, regularization, alpha, self.cuda_device)
                
                results = model.models_metrics(test_logits, test_true)

                self.metrics_handler.update_test_results(models_identifier, results)
                self.metrics_handler.print_test_results(test_loss, results)

                helpers.save_results(test_logits, test_true, helpers.get_predictions_file_name(self.output_directory, models_identifier, target, fold, self.run_identificator), self.prediction_type)        
                self.models_performance_saver.flush_data()
                
                print(f"+++ Finished with training and testing model for {target} on fold {fold}. +++")
                
                delimiter()
                delimiter()
    
    def get_the_best_model(self, target, fold, train_indices, train_output, val_indices, val_output):
        if self.balance_data == 'True':
            self.debugger.print(train_indices)
            self.debugger.print(train_output)
            train_indices, train_output = get_balanced_data(train_indices, train_output)
    
        parameters_combinations = product(*[list(zip([label] * len(values), values)) for label, values in self.optimization_parameters.items()])
        
        best_models_properties = None
        best_models_results = None
        best_model = None
        best_models_parameters = None 

        for parameters in parameters_combinations:
            parameters_dict = dict(parameters)
            print(f'Starting training for: {parameters_dict}')
            delimiter()
            
            models_identifier = self.models_performance_saver.create_new_entry(self.experiments_name, self.models_name, target, fold, self.run_identificator, parameters_dict)
            
            model = get_model(self.debugger, self.models_name, self.targets_type, self.prediction_type, self.number_of_classes, parameters_dict)
            if self.cuda_device:
                model.to(self.cuda_device)

            optimizer = to.optim.Adam(model.parameters(), parameters_dict[self.LEARNING_RATE], amsgrad=True)
            exp_lr_scheduler = StepLR(optimizer, step_size=self.decay_epoch, gamma=self.decay_rate)
            regularization = get_regularization(parameters_dict[self.REGULARIZATION])

            model_operator = ModelOperator(self.debugger, model, self.metrics_handler, self.loss_function, optimizer, self.number_of_epochs, self.print_status_batch, self.cuda_device,\
                regularization, exp_lr_scheduler, self.data_frame, parameters_dict[self.BATCH_SIZE], train_indices, train_output, val_indices, val_output,\
                    parameters_dict[self.ALPHA], self.max_constant_f1, self.models_performance_saver, models_identifier)

            
            model_operator.train()
            self.models_performance_saver.flush_data()

            current_models_properties, current_models_results = model_operator.get_best_models_properties()
            if (best_models_properties is None or self.metrics_handler.compare_new_results(best_models_results, current_models_results)):
                best_models_properties = current_models_properties
                best_models_results = current_models_results
                best_models_parameters = parameters_dict
                best_model = model

        
        best_checkpoint_filename = helpers.get_checkpoints_name(self.output_directory, best_models_properties.identifier)
        to.save({
        "models_identifier" : best_models_properties.identifier, 
        "model_state_dict" : best_models_properties.state_dict,
        "optimizer_state_dict" : best_models_properties.optimizers_state_dict
        }, best_checkpoint_filename)
         
        if self.cuda_device:
            best_model.to(self.cuda_device)
        

        best_model.load_state_dict(best_models_properties.state_dict)
        return best_model, best_models_properties.identifier, best_models_parameters