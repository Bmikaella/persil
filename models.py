import torch.nn as nn
import torch.nn.functional as func
from helpers import *
from metrics import *
from functools import reduce

CONVO_CARPET_NAME = 'convo_carpet'
CONVO_CARPET_DEEP_NAME = 'convo_carpet_deep'

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
        else: 
            return ConvoCarpetMultiTarget(debugger, prediction_type, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], \
            sentences_count=parameters[SENTENCES_COUTS], hidden_layer_1=parameters[HIDDEN_LAYER1])
    elif(models_name == CONVO_CARPET_DEEP_NAME):
        if targets_type == SINGLE_TARGET:
            return ConvoCarpetDeep(debugger, prediction_type, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], \
            sentences_count=parameters[SENTENCES_COUTS], hidden_layers=parameters[HIDDEN_LAYERS], act_func=parameters[ACTIVATION_FUNCTION],
            dropout=parameters[DROPOUT])
        else:
            raise Exception(f'{models_name} model is not implemented for multitarget')
    raise Exception(f'{models_name} model is not implemented ')

def get_activation_function(debugger, name_of_function):
    if(name_of_function == 'tanh'):
        return nn.Tanh()
    elif(name_of_function == 'sig'):
        return nn.Sigmoid()
    elif(name_of_function == 'relu'):
        return nn.ReLU()
    elif(name_of_function == 'mine'):
        return 
    else:
        raise Exception(f"Function {name_of_function} is not supported as an activation function")


class ConvoCarpetDeep(nn.Module):

    def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, \
    hidden_layers=[10], act_func='sig', dropout=0.5):
        super(ConvoCarpetDeep, self).__init__()
        self.debugger = debugger
        self.prediction_type = prediction_type
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))
        self.activation_function = act_func
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_layers = []
        for no_neurons in hidden_layers:
            self.hidden_layers.append(nn.Linear(kernels_count, hidden_layer_1))
        else:
            self.fc_layer2 = nn.Linear(no_neurons, number_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        nn_deep_path = [maxpool_output] + self.hidden_layers
        hidden_layers_output = reduce(lambda last, hidden_layer : self.dropout(self.activation_function(hidden_layer(last))), nn_deep_path)
        linear_output2 = self.fc_layer2(hidden_layers_output)
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
 
class ConvoCarpetMultiTarget(nn.Module):

    def __init__ (self, debugger, prediction_type, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
        super(ConvoCarpetMultiTarget, self).__init__()
        self.debugger = debugger
        self.prediction_type = prediction_type
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))
        self.fc_layer1 = nn.Linear(kernels_count, hidden_layer_1)
        self.non_linear = nn.Sigmoid()
        self.fc_layers2 = []
        self.softmax = nn.Softmax(dim=1)
        for count in number_of_classes:
            self.fc_layers2.append(nn.Linear(hidden_layer_1, count))

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        linear_output1 = self.fc_layer1(maxpool_output)
        linear_output1 = self.non_linear(linear_output1)
        outputs = []
        for fc_layer in self.fc_layers2:
            linear_output2 = fc_layer(linear_output1)
            if(self.prediction_type == CLASSIFICATION):
                outputs.append(self.softmax(linear_output2))
            elif(self.prediction_type == REGRESSION):
                outputs.append(linear_output2)
            else:
                raise Exception("No such type for prediction")
        return outputs

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        if(self.prediction_type == CLASSIFICATION):
            return calculate_classification_metrics_multitarget(self.debugger, test_logits, test_true, self.number_of_classes)
        elif(self.prediction_type == REGRESSION):
            return calculate_regression_metrics_multitarget(self.debugger, test_logits, test_true)
        else:
            raise Exception("Can't calculate metrics!")

# TODO : check if this implementation is even okay
class AttentiveCarpet(nn.Module):
    MODELS_NAME = 'attention_model'

    CARPET_LENGTH = 'carpet_length'

    def __init__ (self, carpet_length, embedding_size=1024):
        super(AttentiveCarpet, self).__init__()
        self.encoder_attentive_pooling_projection = nn.Linear((carpet_length, embedding_size), 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_batch):
        self_attentive_logits = self.encoder_attentive_pooling_projection(input_batch).squeeze(2)
        self_weights = util.softmax(self_attentive_logits)
        self_attentive_pool = util.weighted_sum(input_batch, self_weights)
        return self_attentive_pool

    def convert_input(self, batch_elements):
        return batch_elements
