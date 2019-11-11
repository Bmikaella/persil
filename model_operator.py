import torch.nn as nn 
import torch as to
import time 
from data_operators import InputOutputFrame
from helpers import * 
from itertools import tee
from metrics import *
import experiment
from collections import namedtuple

ModelsProperties = namedtuple('ModelsProperties', 'state_dict optimizers_state_dict val_f1_m identifier')
       
def apply(model, loss_function, data_loader, cuda_device):
    model.eval()
    total_loss = 0
    predicted = []
    true_output = []
    minibatch_loss = 0
    with to.no_grad():
        n_batch = 0
        for inputs, labels in data_loader:
            n_batch += 1
            minibatch_logits = model(inputs.to(device=cuda_device, dtype=to.float))
            minibatch_loss = loss_function(minibatch_logits, labels)
            
            total_loss += minibatch_loss
            predicted.append(minibatch_logits)
            true_output.append(labels)
            
    return total_loss, predicted, true_output


class ModelOperator():

    def __init__ (self, model, loss_function, optimizer, number_of_epochs, print_status_batch, cuda_device, regularization, exp_lr_scheduler,\
         data_frame, batch_size, train_input_indices, train_output, val_input_indices, val_output, alpha, max_constant_f1, model_performance_saver, models_identifier):
        self.loss_function = loss_function
        self.number_of_epochs = number_of_epochs
        self.print_status_batch = print_status_batch
        self.optimizer = optimizer
        self.cuda_device = cuda_device
        self.data_frame = data_frame
        self.train_input_indices = train_input_indices
        self.val_input_indices = val_input_indices 
        self.train_output = train_output
        self.val_output = val_output
        self.regularization = regularization
        self.model_performance_saver = model_performance_saver
        self.exp_lr_scheduler = exp_lr_scheduler
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_constant_f1 = max_constant_f1

        self.model = model
        self.models_identifier = models_identifier

        self.best_models_properties = None

    def train(self):
        best_f1_validation = 0

        last_f1 = 0
        f1_constant_counter = 0
        
        val_loader = self.data_frame.create_minibatches(self.val_input_indices, self.val_output, self.batch_size, self.cuda_device, self.model.convert_input)
            
        for epoch in range(0, self.number_of_epochs):
            epoch_start = time.time()
            self.model.train()
            total_loss = 0
            n_batch = 0
            
            predicted = []
            true_output = []
            
            train_loader = self.data_frame.create_minibatches(self.train_input_indices, self.train_output, self.batch_size, self.cuda_device, self.model.convert_input)
            for inputs, labels in train_loader:
                n_batch += 1
                self.model.zero_grad()

                logits = self.model(inputs.to(device=self.cuda_device, dtype=to.float))
                    
                minibatch_loss = self.loss_function(logits, labels) + self.regularization(self.model.parameters(), self.alpha)
                total_loss += minibatch_loss
                
                minibatch_loss.backward()
                self.optimizer.step()
                
                predicted.append(logits)
                true_output.append(labels)
                
                if(n_batch % self.print_status_batch == 0):
                                
                    print(f"Train - Epoch {epoch} - batch {n_batch}, batch loss is {minibatch_loss:.6f}")
                    delimiter()

            self.exp_lr_scheduler.step()
            val_loader, val_loader_backup = tee(val_loader)
            val_loss, val_logits, true_val = apply(self.model, self.loss_function, val_loader_backup, self.cuda_device)
            val_f1_score, val_precision_0, val_precision_1, val_precision_m, val_recall_0, val_recall_1, val_recall_m = calcMeasures(val_logits, true_val, 0.5)
            train_f1_score, _, _, _, _, _,_ = calcMeasures(predicted, true_output, 0.5)
            
            print(f'Epoch {epoch} end: {time.time()-epoch_start}, TRAIN F1 is: {train_f1_score}')
            print(f'Validation loss: {val_loss:.7f} - F1 score: {val_f1_score:.7f}')
            print(f'0 class -> precision: {val_precision_0:.7f} - recall: {val_recall_0:.7f}')
            print(f'1 class -> precision: {val_precision_1:.7f} - recall: {val_recall_1:.7f}')
            print(f'precision: {val_precision_m:.7f} - recall: {val_recall_m:.7f} - MACRO')
            delimiter()
            
            if (abs(val_f1_score - last_f1) <= experiment.RELATIVE_DIFFERENCE):
                f1_constant_counter += 1
            else:
                f1_constant_counter = 0
            
            last_f1 = val_f1_score
            
            if(f1_constant_counter >= self.max_constant_f1):
                return self.best_models_properties
        
            if(best_f1_validation < val_f1_score):
                best_f1_validation = val_f1_score
                self.model_performance_saver.update_models_val_results(self.models_identifier, val_f1_score, val_precision_0, val_precision_1, val_precision_m, val_recall_0, val_precision_1, val_recall_m, epoch)
                self.best_models_properties = ModelsProperties(self.model.state_dict(), self.optimizer.state_dict(), val_f1_score, self.models_identifier)
    
    def get_best_models_properties(self):
        return self.best_models_properties

 