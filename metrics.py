import torch as to
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from scipy.stats import pearsonr

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

# def calculate_pearson(predicted, true_values):
#     r2_score_value = r2_score(true_values, predicted)
#     mse = mean_squared_error(true_values, predicted)
#     pearson = pearsonr(true_values, predicted)
#     return r2_score_value, mse, pearson