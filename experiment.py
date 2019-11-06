from persil.helpers import *
from persil.data_operators import *
from persil.model_operator import * 
from itertools import product
from persil.models import *
from persil.regularization import * 
import torch as to
from torch.optim.lr_scheduler import StepLR
from persil.model_operator import apply

RELATIVE_DIFFERENCE = 0.00001

class Experiment:

    LEARNING_RATE = 'learning_rate'
    BATCH_SIZE = 'batch_size'
    REGULARIZATION = 'regularization_type'
    ALPHA = 'alpha'  

    def __init__(self, output_directory, models_performance_saver, data_frame, experiments_name, models_name, run_identificator, folds, mbti_traits, optimization_parameters, print_status_batch, max_constant_f1, number_of_epochs,\
        decay_rate, decay_epoch, number_of_classes, validation_set_percentage, cuda_device_index=1, use_GPU=True, random_state=56):
        self.output_directory = output_directory
        self.models_performance_saver = models_performance_saver
        self.data_frame = data_frame
        self.experiments_name = experiments_name
        self.models_name = models_name 
        self.run_identificator = run_identificator  
        self.folds = folds 
        self.mbti_traits = mbti_traits 
        self.optimization_parameters = optimization_parameters 
        self.print_status_batch = print_status_batch 
        self.max_constant_f1 = max_constant_f1 
        self.number_of_epochs = number_of_epochs
        
        self.number_of_classes = number_of_classes
        
        self.validation_set_percentage = validation_set_percentage 
        self.cuda_device = to.device(cuda_device_index) if use_GPU else None
        self.random_state = random_state
        self.loss_function = nn.CrossEntropyLoss()
    
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch


    def start(self):
        
        delimiter()
        print(f"Starting experiment {self.experiments_name}...")
        delimiter()

        for target in self.mbti_traits:
            for fold in self.folds:
                print(f'Processing fold: {fold} and target: {target}')
                delimiter()

                train_input_indices, train_output, val_input_indices, val_output, test_input_indices, test_output = self.data_frame.\
                    get_train_val_test_input_output(target, fold, self.validation_set_percentage, self.random_state)
            

                model, models_identifier = self.get_the_best_model(target, fold, train_input_indices, train_output, val_input_indices, val_output)
                
                print(f'Apply best model to test for {target} on fold {fold}.')
                test_loader = self.data_frame.create_minibatches(test_input_indices, test_output, 1, self.cuda_device)

                test_loss, test_logits, test_true = apply(model, self.loss_function, test_loader, self.cuda_device)
                test_f1_score, test_precision_0, test_precision_1, test_precision_m, test_recall_0, test_recall_1, test_recall_m = calcMeasures(test_logits, test_true, 0.5)
                print(f'Test loss: {test_loss:.5f} - F1 score: {test_f1_score:.7f} ')
                print(f'0 class -> precision: {test_precision_0:.7f} - recall: {test_recall_0:.7f}')
                print(f'1 class -> precision: {test_precision_1:.7f} - recall: {test_recall_1:.7f}')
                print(f'precision: {test_precision_m:.7f} - recall: {test_recall_m:.7f} - MACRO')

                self.data_frame.update_models_test_results(models_identifier, test_f1_score, test_precision_0, test_precision_1, test_precision_m, test_recall_0, test_recall_1, test_recall_m)
                save_results(test_logits, test_true, get_predictions_file_name(self.output_directory, models_identifier, target, fold, self.run_identificator))        
                self.data_frame.flush_data()
                
                
                print(f"+++ Finished with training and testing model for {target} on fold {fold}. +++")
                
                delimiter()
                delimiter()
    
    def get_the_best_model(self, target, fold, train_indices, train_output, val_indices, val_output):
        train_input_indices, train_output = get_balanced_data(train_indices, train_output)
    
        parameters_combinations = product(*[list(zip([label] * len(values), values)) for label, values in self.optimization_parameters.items()])
        
        best_models_properties = None
        best_model = None 

        for parameters in parameters_combinations:
            parameters_dict = dict(parameters)
            print(f'Starting training for: {parameters_dict}')
            delimiter()
            
            models_identifier = self.models_performance_saver.create_new_entry(self.experiments_name, self.models_name, target, fold, self.run_identificator, parameters_dict)
            
            model = get_model(self.models_name, self.number_of_classes, parameters_dict)
            if self.cuda_device:
                model.to(self.cuda_device)

            optimizer = to.optim.Adam(model.parameters(), parameters_dict[self.LEARNING_RATE], amsgrad=True)
            exp_lr_scheduler = StepLR(optimizer, step_size=self.decay_epoch, gamma=self.decay_rate)
            regularization = get_regularization(parameters_dict[self.REGULARIZATION])

            model_operator = ModelOperator(model, self.loss_function, optimizer, self.number_of_epochs, self.print_status_batch, self.cuda_device,\
                regularization, exp_lr_scheduler, self.data_frame, parameters_dict[self.BATCH_SIZE], train_input_indices, train_output, val_indices, val_output,\
                    parameters_dict[self.ALPHA], self.max_constant_f1, self.models_performance_saver, models_identifier)

            
            model_operator.train()
            self.data_frame.flush_data()

            current_models_properties = model_operator.get_best_models_properties()
            if (not best_models_properties or best_models_properties.val_f1_m < current_models_properties.val_f1_m):
                best_models_properties = current_models_properties
                best_model = model

        
        best_checkpoint_filename = get_checkpoints_name(self.output_directory, best_models_properties.identifier)
        to.save({
        "models_identifier" : best_models_properties.identifier, 
        "model_state_dict" : best_models_properties.state_dict,
        "optimizer_state_dict" : best_models_properties.optimizers_state_dict
        }, best_checkpoint_filename)
         
        if self.cuda_device:
            best_model.to(self.cuda_device)
        

        best_model.load_state_dict(best_models_properties.state_dict)
        return best_model, best_models_properties.identifier