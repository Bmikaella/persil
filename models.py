import torch.nn as nn
import torch.nn.functional as func
from helpers import *
from metrics import *

def get_model(debugger, models_name, targets_type, number_of_classes, parameters):
    if (models_name == CONVO_CARPET_NAME):
        if targets_type == SINGLE_TARGET:
            return ConvoCarpetSingleTarget(debugger, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], sentences_count=parameters[SENTENCES_COUTS], hidden_layer_1=parameters[HIDDEN_LAYER1])
        else: 
            return ConvoCarpetMultiTarget(debugger, number_of_classes, kernels_count=parameters[KERNELS_COUNTS], sentences_count=parameters[SENTENCES_COUTS], hidden_layer_1=parameters[HIDDEN_LAYER1])
    raise Exception(f'{models_name} model is not implemented ')

# RegressionResults = namedtuple('RegressionResults', 'r2_score_value mse pearson')

CONVO_CARPET_NAME = 'convo_carpet'

KERNELS_COUNTS = 'kernels_count'
SENTENCES_COUTS = 'sentences_count'
HIDDEN_LAYER1 = 'hidden_layer_1'

class ConvoCarpetSingleTarget(nn.Module):

    def __init__ (self, debugger, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
        super(ConvoCarpetSingleTarget, self).__init__()
        self.debugger = debugger
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
        linear_output1 = self.fc_layer1(maxpool_output)
        linear_output2 = self.fc_layer2(linear_output1)
        output = self.softmax(linear_output2)
        return output

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        return calculate_classification_metrics_singletarget(self.debugger, test_logits, test_true, self.number_of_classes, 0.5)
 
class ConvoCarpetMultiTarget(nn.Module):

    def __init__ (self, debugger, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
        super(ConvoCarpetMultiTarget, self).__init__()
        self.debugger = debugger
        self.number_of_classes = number_of_classes
        self.conv_layer = nn.Conv2d(1, kernels_count, [sentences_count, embedding_size])
        self.pool_layer = nn.AdaptiveMaxPool2d((1, None))
        self.fc_layer1 = nn.Linear(kernels_count, hidden_layer_1)
        self.fc_layers2 = []
        self.softmax = nn.Softmax(dim=1)
        for count in number_of_classes:
            self.fc_layers2.append(nn.Linear(hidden_layer_1, count))

    def forward(self, input_batch):
        conv_output = func.relu(self.conv_layer(input_batch))
        maxpool_output = self.pool_layer(conv_output)
        maxpool_output = maxpool_output.flatten(start_dim=1)
        linear_output1 = self.fc_layer1(maxpool_output)
        outputs = []
        for fc_layer in self.fc_layers2:
            linear_output2 = fc_layer(linear_output1)
            outputs.append(self.softmax(linear_output2))
        return outputs

    def convert_input(self, batch_elements):
        return batch_elements.unsqueeze(1)

    def models_metrics(self, test_logits, test_true):
        return calculate_classification_metrics_multitarget(self.debugger, test_logits, test_true, self.number_of_classes)

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
