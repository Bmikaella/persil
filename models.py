import torch.nn as nn
import torch.nn.functional as func

def get_model(models_name, number_of_classes, parameters):
    if (models_name == ConvoCarpet.MODELS_NAME):
        return ConvoCarpet(number_of_classes, kernels_count=parameters[ConvoCarpet.KERNELS_COUNTS], sentences_count=parameters[ConvoCarpet.SENTENCES_COUTS], hidden_layer_1=parameters[ConvoCarpet.HIDDEN_LAYER1])
    raise Exception(f'{models_name} model is not implemented ')

class ConvoCarpet(nn.Module):
    MODELS_NAME = 'convo_carpet'

    KERNELS_COUNTS = 'kernels_count'
    SENTENCES_COUTS = 'sentences_count'
    HIDDEN_LAYER1 = 'hidden_layer_1'

    def __init__ (self, number_of_classes, embedding_size=1024, kernels_count=64, sentences_count=2, hidden_layer_1=4):
        super(ConvoCarpet, self).__init__()
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

class AttentiveCarpet(nn.Module):
    MODELS_NAME = 'attention_model'

    KERNELS_COUNTS = 'kernels_count'
    SENTENCES_COUTS = 'sentences_count'
    HIDDEN_LAYER1 = 'hidden_layer_1'

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
