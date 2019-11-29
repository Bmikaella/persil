import torch.nn as nn 
import torch as to
import time 
from data_operators import InputOutputFrame
from helpers import * 
from itertools import tee
from metrics import *
import experiment
from collections import namedtuple
from helpers import *
from abc import ABC, abstractmethod 
from statistics import mean

ModelsProperties = namedtuple('ModelsProperties', 'state_dict optimizers_state_dict identifier')
       
def apply(debugger, model, loss_function, data_loader, regularization, alpha, cuda_device):
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
            minibatch_loss = loss_calculator(debugger, minibatch_logits, labels, model.parameters(), loss_function, regularization, alpha)
            
            total_loss += minibatch_loss
            predicted.append(minibatch_logits)
            true_output.append(labels)
            
    return total_loss, predicted, true_output

def loss_calculator(debugger, logits, labels, models_parameters, loss_function, regularization, alpha):
    debugger.print(logits[0])
    debugger.print (labels[0])
    debugger.print(logits.cpu())
    debugger.print(labels.cpu())
    return loss_function(logits, labels) + regularization(models_parameters, alpha)

class ModelOperator():

    def __init__ (self, debugger, model, metrics_handler, loss_function, optimizer, number_of_epochs, print_status_batch, cuda_device, regularization, exp_lr_scheduler,\
         data_frame, batch_size, train_input_indices, train_output, val_input_indices, val_output, alpha, max_constant_f1, model_performance_saver, models_identifier):
        self.debugger = debugger
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
        self.metrics_handler = metrics_handler
        self.max_constant_f1 = max_constant_f1

        self.model = model
        self.models_identifier = models_identifier

        self.best_models_properties = None
        self.best_models_results = None

    def models_loss(self, logits, labels):
        self.debugger.print("Here we are")
        self.debugger.print(logits)
        self.debugger.print(labels)
        return loss_calculator(self.debugger, logits, labels, self.model.parameters(), self.loss_function, self.regularization, self.alpha)
   
    def train(self):
        self.best_models_results = None
        
        results_constant = 0
        
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
                    
                minibatch_loss = self.models_loss(logits, labels)
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
            val_loss, val_logits, true_val = apply(self.debugger, self.model, self.loss_function, val_loader_backup, self.regularization, self.alpha, self.cuda_device)

            results_val = self.model.models_metrics(val_logits, true_val)

            results_train = self.model.models_metrics(predicted, true_output)
            
            self.metrics_handler.print_val_results(epoch, time.time()-epoch_start, val_loss, results_train, results_val)

            if (self.best_models_results is not None and self.metrics_handler.check_for_stagnation(self.best_models_results, results_val)):
                results_constant += 1
            else:
                results_constant = 0
            
            if(results_constant >= self.max_constant_f1):
                return
        
            if(self.best_models_results is None or self.metrics_handler.compare_new_results(self.best_models_results, results_val)):
                self.best_models_results = results_val
                self.metrics_handler.update_val_results(self.models_identifier, epoch, results_val)
                self.best_models_properties = ModelsProperties(self.model.state_dict(), self.optimizer.state_dict(), self.models_identifier)
    
    def get_best_models_properties(self):
        return self.best_models_properties, self.best_models_results
        