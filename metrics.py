import torch as to
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from scipy.stats import pearsonr
from helpers import *

class ClassificationMetricsHandler():

    def __init__ (self, debugger, models_performance_saver, number_of_classes):
        self.debugger = debugger 
        self.models_performance_saver = models_performance_saver
        self.number_of_classes = number_of_classes

    def update_test_results(self, models_identifier, test_results):
        self.models_performance_saver.update_models_test_results(models_identifier, test_results)

    def update_val_results(self, models_identifier, epoch, val_results):
        self.models_performance_saver.update_models_val_results(models_identifier, epoch, val_results)

    def print_val_results(self, epoch, time, val_loss, train_results, val_results,):
        train_f1_score = train_results['f1']
        print(f'Epoch {epoch} end: {time}, TRAIN F1 is: {train_f1_score}')
        print(f'Validation loss: {val_loss:.7f} - F1 score: {val_results["f1"]:.7f}')
        if(self.number_of_classes == 2):
            print(f'0 class -> precision: {val_results["precision_0"]:.7f} - recall: {val_results["recall_0"]:.7f}')
            print(f'1 class -> precision: {val_results["precision_1"]:.7f} - recall: {val_results["recall_1"]:.7f}')
        print(f'precision: {val_results["precision_macro"]:.7f} - recall: {val_results["recall_macro"]:.7f} - MACRO')
        delimiter()

    def print_test_results(self, test_loss, test_results):
        print(f'Test loss: {test_loss:.5f} - F1 score: {test_results["f1"]:.7f} ')
        if(self.number_of_classes == 2):
            print(f'0 class -> precision: {test_results["precision_0"]:.7f} - recall: {test_results["recall_0"]:.7f}')
            print(f'1 class -> precision: {test_results["precision_1"]:.7f} - recall: {test_results["recall_1"]:.7f}')
        print(f'precision: {test_results["precision_macro"]:.7f} - recall: {test_results["recall_macro"]:.7f} - MACRO')
        delimiter()

    def compare_new_results(self, old_values, new_values):
        return old_values['f1'] < new_values['f1']

    def check_for_stagnation(self, old_values, new_values):
        return abs(old_values['f1'] - new_values['f1']) <= RELATIVE_DIFFERENCE

class RegressionMetricsHandler():

    def __init__(self, debugger, models_performance_saver):
        self.debugger = debugger
        self.models_performance_saver = models_performance_saver

    def update_test_results(self, models_identifier, test_results):
        self.models_performance_saver.update_models_test_results(models_identifier, test_results)

    def update_val_results(self, models_identifier, epoch, val_results):
        self.models_performance_saver.update_models_val_results(models_identifier, epoch, val_results)

    def print_val_results(self, epoch, time, val_loss, train_results, val_results,):
        self.debugger.print(train_results)
        print(f'Epoch {epoch} end: {time}, TRAIN MSE is: {train_results["mse"]}')
        print(f'Validation loss: {val_loss:.7f} - MSE score: {val_results["mse"]:.7f}')
        print(f'Validation pearson is: {val_results["pearson"]}')
        print(f'Validation r2_score is: {val_results["r2_score"]}')
        delimiter()

    def print_test_results(self, test_loss, test_results):
        print(f'Test loss: {test_loss:.5f} - MSE score: {test_results["mse"]:.7f} ')
        print(f'Test pearson is: {test_results["pearson"]}')
        print(f'Test r2_score is: {test_results["r2_score"]}')
        delimiter()

    def compare_new_results(self, old_values, new_values):
        return old_values['mse'] < new_values['mse']

    def check_for_stagnation(self, old_values, new_values):
        return abs(old_values['mse'] - new_values['mse']) <= RELATIVE_DIFFERENCE
 

def calculate_classification_metrics_singletarget(debugger, total_logits, total_true, number_of_classes, threshold):
    if(number_of_classes == 2):
        total_preds = []
        for minibatch_logits in total_logits:
            minibatch_preds = minibatch_logits[:,1] > threshold
            total_preds.append(minibatch_preds)
        total_preds = to.cat(total_preds)
        total_true = to.cat(total_true)
        debugger.print(total_preds)
        debugger.print(total_true)
        results = {}
        F1 = f1_score(total_true.cpu(), total_preds.cpu(), pos_label = 1, average='macro')
        results['f1'] = F1
        precision_m = precision_score(total_true.cpu(), total_preds.cpu(), average='macro')
        results['precision_macro'] = precision_m
        precision_1 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
        results['precision_1'] = precision_1
        precision_0 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
        results['precision_0'] = precision_0
        recall_m = recall_score(total_true.cpu(), total_preds.cpu(), average='macro')
        results['recall_macro'] = recall_m
        recall_1 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
        results['recall_1'] = recall_1
        recall_0 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
        results['recall_0'] = recall_0
        return results
    else:
        total_preds = []
        for minibatch_logits in total_logits:
            minibatch_preds = minibatch_logits.argmax(dim=1)
            total_preds.append(minibatch_preds)
        total_preds = to.cat(total_preds)
        total_true = to.cat(total_true)
        debugger.print(total_preds)
        debugger.print(total_true)
        results = {}
        F1 = f1_score(total_true.cpu(), total_preds.cpu(), average='macro')
        results['f1'] = F1
        precision_m = precision_score(total_true.cpu(), total_preds.cpu(), average='macro')
        results['precision_macro'] = precision_m
        recall_m = recall_score(total_true.cpu(), total_preds.cpu(), average='macro')
        results['recall_macro'] = recall_m
        debugger.print(F1)
        debugger.print(precision_m)
        debugger.print(recall_m)
        return results
      

def calculate_classification_metrics_multitarget(debugger, total_logits, total_true, number_of_classes):
    pass
# TODO 
    # total_preds = []
    # for minibatch_logits in total_logits:
    #     minibatch_preds = minibatch_logits[:,1] > threshold
    #     total_preds.append(minibatch_preds)
    # total_preds = to.cat(total_preds)
    # total_true = to.cat(total_true)
    # F1 = f1_score(total_true.cpu(), total_preds.cpu(), pos_label = 1, average='macro')
    # precision_m = precision_score(total_true.cpu(), total_preds.cpu(), average='macro')
    # precision_1 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
    # precision_0 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
    # recall_m = recall_score(total_true.cpu(), total_preds.cpu(), average='macro')
    # recall_1 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
    # recall_0 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
    # return F1, precision_0, precision_1, precision_m, recall_0, recall_1, recall_m

def calculate_regression_metrics(debugger, predicted, true_values):
    total_preds = to.squeeze(to.cat(predicted)).cpu().detach().numpy()
    total_true = to.squeeze(to.cat(true_values)).cpu().detach().numpy()
    debugger.print(total_preds)
    debugger.print(total_true)
    results = {}
    results['r2_score'] = r2_score(total_true, total_preds)
    results['mse'] = mean_squared_error(total_true, total_preds)
    results['pearson'] = pearsonr(total_true, total_preds)[0]
    return results

def calculate_regression_metrics_multitarget(debugger, predicted, true_values):
    pass