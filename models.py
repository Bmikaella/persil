import torch.nn as nn
import torch.nn.functional as func
from helpers import *
from metrics import *
import pdb

CONVO_CARPET_NAME = 'convo_carpet'
CONVO_CARPET_DEEP_NAME = 'convo_carpet_deep'
CONVO_CARPET_DEEP5_NAME = 'convo_carpet_deep5'

KERNELS_COUNTS = 'kernels_count'
SENTENCES_COUTS = 'sentences_count'
HIDDEN_LAYER1 = 'hidden_layer_1'
HIDDEN_LAYERS = 'hidden_layers'
ACTIVATION_FUNCTION = 'act_func'
DROPOUT = 'dropout'

def get_model(debugger, models_name, targets_type, prediction_type, number_of_classes, parameters):
    if (models_name == CONVO_CARPET_NAME):
        if targets_type == SINGLE_TARGET:
            return ConvoCarpetSingleTarget(debugger, prediction_type, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], \
            sentences_count=parameters[SENTENCES_COUTS], hidden_layer_1=parameters[HIDDEN_LAYER1])
    elif(models_name == CONVO_CARPET_DEEP_NAME):
            return ConvoCarpetDeep(debugger, prediction_type, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], \
            sentences_count=parameters[SENTENCES_COUTS], hidden_layers=parameters[HIDDEN_LAYERS], act_func=parameters[ACTIVATION_FUNCTION],
            dropout=parameters[DROPOUT])
    elif(models_name == CONVO_CARPET_DEEP5_NAME)
            return ConvoCarpetDeepMultitarget55(debugger, prediction_type, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], \
            sentences_count=parameters[SENTENCES_COUTS], hidden_layers=parameters[HIDDEN_LAYERS], act_func=parameters[ACTIVATION_FUNCTION],
            dropout=parameters[DROPOUT])
    raise Exception(f'{models_name} model is not implemented ')

def get_activation_function(name_of_function):
    if(name_of_function == 'tanh'):
        return nn.Tanh()
    elif(name_of_function == 'sig'):
        return nn.Sigmoid()
    elif(name_of_function == 'relu'):
        return nn.ReLU()
    else:
        raise Exception(f"Function {name_of_function} is not supported as an activation function")


def get_linear_layer(inputs, outputs, activation_function, dropout):
    return nn.Sequential(nn.Linear(inputs, outputs), get_activation_function(activation_function),\
        nn.Dropout(p=dropout))

class ConvoCarpetDeep(nn.Module):

    def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, \
    hidden_layers=[10, 10, 10], act_func='sig', dropout=0.5):
        super(ConvoCarpetDeep, self).__init__()
        self.debugger = debugger

        self.prediction_type = prediction_type
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))

        self.fc_layer1 = get_linear_layer(kernels_count, hidden_layers[0], act_func, dropout)
        self.fc_layer2 = get_linear_layer(hidden_layers[0], hidden_layers[1], act_func, dropout)
        self.fc_layer3 = get_linear_layer(hidden_layers[1], hidden_layers[2], act_func, dropout)
        
        self.fc_layer4 = nn.Linear(hidden_layers[2], number_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        
        last_output = self.fc_layer1(maxpool_output)
        last_output = self.fc_layer2(last_output)
        last_output = self.fc_layer3(last_output)

        last_output = self.fc_layer4(last_output)

        if(self.prediction_type == CLASSIFICATION):
            return self.softmax(last_output)
        elif(self.prediction_type == REGRESSION):
            return last_output
        else: 
            raise Exception("No such type for prediction")

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        if(self.prediction_type == CLASSIFICATION):
            return calculate_classification_metrics_singletarget(self.debugger, test_logits, test_true, self.number_of_classes, 0.5)
        elif(self.prediction_type == REGRESSION):
            return calculate_regression_metrics(self.debugger, test_logits, test_true)
        else:
            raise Exception("Can't calculate metrics!")

class ConvoCarpetDeepMultitarget5(nn.Module):

    def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, \
    hidden_layers=[10, 10, 10], act_func='sig', dropout=0.5):
        super(ConvoCarpetDeepMultitarget5, self).__init__()
        self.debugger = debugger

        self.prediction_type = prediction_type
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))

        self.fc_layer1 = get_linear_layer(kernels_count, hidden_layers[0], act_func, dropout)
        self.fc_layer2 = get_linear_layer(hidden_layers[0], hidden_layers[1], act_func, dropout)
        self.fc_layer3 = get_linear_layer(hidden_layers[1], hidden_layers[2], act_func, dropout)

        
        self.fc_layer4 = nn.Linear(hidden_layers[2], number_of_classes[0])
        self.fc_layer5 = nn.Linear(hidden_layers[2], number_of_classes[1])
        self.fc_layer6 = nn.Linear(hidden_layers[2], number_of_classes[2])
        self.fc_layer7 = nn.Linear(hidden_layers[2], number_of_classes[3])
        self.fc_layer8 = nn.Linear(hidden_layers[2], number_of_classes[4])

        self.all_layers = [self.fc_layer4, self.fc_layer5, self.fc_layer6, self.fc_layer7, self.fc_layer8]

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        
        last_output = self.fc_layer1(maxpool_output)
        last_output = self.fc_layer2(last_output)
        last_output = self.fc_layer3(last_output)

        outputs = []
        for fc_layer in self.all_layers:
            linear_output2 = fc_layer(last_output)
            if(self.prediction_type == CLASSIFICATION):
                outputs.append(self.softmax(linear_output2))
            elif(self.prediction_type == REGRESSION):
                outputs.append(linear_output2)
            else:
                raise Exception("No such type for prediction")
        return outputs

        if(self.prediction_type == CLASSIFICATION):
            return self.softmax(last_output)
        elif(self.prediction_type == REGRESSION):
            return last_output
        else: 
            raise Exception("No such type for prediction")

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        if(self.prediction_type == CLASSIFICATION):
            return calculate_classification_metrics_multitarget(self.debugger, test_logits, test_true, self.number_of_classes, 0.5)
        elif(self.prediction_type == REGRESSION):
            return calculate_regression_metrics_multitarget(self.debugger, test_logits, test_true)
        else:
            raise Exception("Can't calculate metrics!")


class ConvoCarpetSingleTarget(nn.Module):

    def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
        super(ConvoCarpetSingleTarget, self).__init__()
        self.debugger = debugger
        self.prediction_type = prediction_type
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))
        self.fc_layer1 = nn.Linear(kernels_count, hidden_layer_1)
        self.fc_layer2 = nn.Linear(hidden_layer_1, number_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        linear_output1 = nn.Sigmoida(self.fc_layer1(maxpool_output))
        linear_output2 = self.fc_layer2(linear_output1)
        if(self.prediction_type == CLASSIFICATION):
            return self.softmax(linear_output2)
        elif(self.prediction_type == REGRESSION):
            return linear_output2
        else: 
            raise Exception("No such type for prediction")

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        if(self.prediction_type == CLASSIFICATION):
            return calculate_classification_metrics_singletarget(self.debugger, test_logits, test_true, self.number_of_classes, 0.5)
        elif(self.prediction_type == REGRESSION):
            return calculate_regression_metrics(self.debugger, test_logits, test_true)
        else:
            raise Exception("Can't calculate metrics!")
 
# class ConvoCarpetMultiTarget(nn.Module):

#     def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
#         super(ConvoCarpetMultiTarget, self).__init__()
#         self.debugger = debugger
#         self.prediction_type = prediction_type
#         self.number_of_classes = number_of_classes
#         self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
#         self.pool_layer = nn.AdaptiveMaxPool2d((1, None))
#         self.fc_layer1 = nn.Linear(kernels_count, hidden_layer_1)
#         self.non_linear = nn.Sigmoid()
#         self.fc_layers2 = []
#         self.softmax = nn.Softmax(dim=1)
#         for count in number_of_classes:
#             self.fc_layers2.append(nn.Linear(hidden_layer_1, count))

#     def forward(self, input_batch):
#         conv_output = func.relu(self.conv_layer(input_batch))
#         maxpool_output = self.pool_layer(conv_output)
#         maxpool_output = maxpool_output.flatten(start_dim=1)
#         linear_output1 = self.fc_layer1(maxpool_output)
#         linear_output1 = self.non_linear(linear_output1)
#         outputs = []
#         for fc_layer in self.fc_layers2:
#             linear_output2 = fc_layer(linear_output1)
#             if(self.prediction_type == CLASSIFICATION):
#                 outputs.append(self.softmax(linear_output2))
#             elif(self.prediction_type == REGRESSION):
#                 outputs.append(linear_output2)
#             else:
#                 raise Exception("No such type for prediction")
#         return outputs

#     def convert_input(self, batch_elements):
#         return batch_elements.unsqueeze(1)

#     def models_metrics(self, test_logits, test_true):
#         if(self.prediction_type == CLASSIFICATION):
#             return calculate_classification_metrics_multitarget(self.debugger, test_logits, test_true, self.number_of_classes, 0.5)
#         elif(self.prediction_type == REGRESSION):
#             return calculate_regression_metrics_multitarget(self.debugger, test_logits, test_true)
#         else:
#             raise Exception("Can't calculate metrics!")

# # TODO : check if this implementation is even okay
# class AttentiveCarpet(nn.Module):
#     MODELS_NAME = 'attention_model'

#     CARPET_LENGTH = 'carpet_length'

#     def __init__ (self, carpet_length, embedding_size=1024):
#         super(AttentiveCarpet, self).__init__()
#         self.encoder_attentive_pooling_projection = nn.Linear((carpet_length, embedding_size), 1)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, input_batch):
#         self_attentive_logits = self.encoder_attentive_pooling_projection(input_batch).squeeze(2)
#         self_weights = util.softmax(self_attentive_logits)
#         self_attentive_pool = util.weighted_sum(input_batch, self_weights)
#         return self_attentive_pool

#     def convert_input(self, batch_elements):
#         return batch_elements
