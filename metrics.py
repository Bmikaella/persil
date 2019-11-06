import torch as to
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score 

def calcMeasures(total_logits, total_true, threshold):
    total_preds = []
    for minibatch_logits in total_logits:
        minibatch_preds = minibatch_logits[:,1] > threshold
        total_preds.append(minibatch_preds)
    total_preds = to.cat(total_preds)
    total_true = to.cat(total_true)
    F1 = f1_score(total_true.cpu(), total_preds.cpu(), pos_label = 1, average='macro')
    precision_m = precision_score(total_true.cpu(), total_preds.cpu(), average='macro')
    precision_1 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
    precision_0 = precision_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
    recall_m = recall_score(total_true.cpu(), total_preds.cpu(), average='macro')
    recall_1 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 1)
    recall_0 = recall_score(total_true.cpu(), total_preds.cpu(), pos_label = 0)
    return F1, precision_0, precision_1, precision_m, recall_0, recall_1, recall_m
