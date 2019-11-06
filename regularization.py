import torch as to 

L2_TYPE = 'L2'
L1_TYPE = 'L1'

def calculate_L2_regularization(parameters, alpha):
    l2_regularization = 0
    for parameter in parameters:
        l2_regularization += alpha * to.pow(to.norm(parameter, 2), 2.0)
    return l2_regularization

def calculate_L1_regularization(parameters, alpha):
    l1_regularization = 0
    for parameter in parameters:
        l1_regularization += alpha * to.sum(to.abs(parameter))
    return l1_regularization

def no_regularization(parameters, alpha):
    return 0

def get_regularization(regularization_type):
    if(regularization_type == L1_TYPE):
        return calculate_L1_regularization
    elif(regularization_type == L2_TYPE):
        return calculate_L2_regularization
    return no_regularization 