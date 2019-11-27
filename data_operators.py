import pandas as pd
import hashlib
import random
import torch as to
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnnutils
from metrics import * 
from helpers import * 

def unfold_comments(sentences, maximal_carpet_size, minimal_carpet_size=0):
    carpet = sentences[:maximal_carpet_size].as_matrix(columns=sentences.columns[1:1025])
    sentences_count = min(len(sentences), maximal_carpet_size) 
    
    padd_size = minimal_carpet_size - sentences_count
    if(padd_size > 0):
        npad = [(0, padd_size), (0, 0)]
        carpet = np.pad(carpet, pad_width=npad, mode='constant', constant_values=0)
    return (sentences['author'].values[0], to.tensor(carpet))

def get_balanced_data(input_df, output_df):
    positive_indices = [index for index, element in enumerate(output_df) if element == 1.0]
    negative_indices = [index for index, element in enumerate(output_df) if element == 0.0]
    negative_count = len(negative_indices)
    positive_count = len(positive_indices)
    
    if(positive_count > negative_count):
        return create_balanced_data(positive_indices, negative_indices, input_df, output_df)
    elif(negative_count > positive_count):
        return create_balanced_data(negative_indices, positive_indices, input_df, output_df)

def create_balanced_data(more_frequent, less_frequent, input_df, output_df):
    more_frequent_count = len(more_frequent)
    less_frequent_count = len(less_frequent)
    constant_multiplyer = more_frequent_count // less_frequent_count
    remaining_additions = more_frequent_count % less_frequent_count
    balanced_indices = less_frequent*constant_multiplyer + less_frequent[:remaining_additions] + more_frequent
    output_df_balanced = [output_df[index] for index in balanced_indices]
    input_df_balanced = [input_df[index] for index in balanced_indices]
    return input_df_balanced, output_df_balanced

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]  

def convert_to_one_hot_enocoding(output_targets, number_of_classes, all_values_sorted):
    positions = dict(zip(all_values_sorted, list(range(number_of_classes))))
    print(f"Class to one hot encoding mapping: {positions}")
    return [positions[element] for element in output_targets]

class InputOutputFrame:
    BINARY_CLASSIFICATION = 2

    def __init__(self, debugger, input_location, output_location, folds_location, sentences_per_author, minimal_input_size=0, nrows=None):
        self. debugger = debugger
        data_df = pd.read_csv(input_location, nrows=nrows)
        self.folds_df = pd.read_csv(folds_location, usecols =['author', 'fold'])
        self.targets_info_df = pd.read_csv(output_location)
        self.input_df = dict(data_df.sort_values(by=['author', 'Unnamed: 0']).groupby(['author'])\
            .apply(lambda sentences : unfold_comments(sentences, sentences_per_author, minimal_input_size)).tolist())
        self.present_authors = self.input_df.keys()
        del data_df

    def get_input_df(self):
        return self.input_df

    def get_all_column_unique_values(self, column):
        self.debugger.print(column)
        return self.targets_info_df[column].dropna().unique()    

    def get_input_output(self, referent_authors, target, number_of_classes, all_values_sorted, prediction_type):
        input_authors_data = self.targets_info_df[self.targets_info_df['author'].isin(referent_authors)]
        input_authors = input_authors_data['author'].values
        
        output_targets = input_authors_data[target].apply(lambda x : [float (x)] if prediction_type == REGRESSION else int(x)).values
        
        if(prediction_type == CLASSIFICATION and number_of_classes > self.BINARY_CLASSIFICATION):
            output_targets = convert_to_one_hot_enocoding(output_targets, number_of_classes, all_values_sorted)

        self.debugger.print(f"Outputs: {output_targets}")
        self.debugger.print(input_authors)
        return input_authors, output_targets

    def singletarget_output_support(self, target, fold, prediction_type, random_state):
        all_output_values = self.targets_info_df[target].dropna().unique()
        self.debugger.print(all_output_values)

        number_of_classes = len(all_output_values)
        self.debugger.print(f'Number of classes: {number_of_classes} for target {target}')
        
        valid_authors = sorted(list(self.targets_info_df[self.targets_info_df[target].notnull() & self.targets_info_df['author'].isin(self.present_authors)]['author']))
        
        self.debugger.print('Not null authors:')
        self.debugger.print(valid_authors)

        test_data_authors = self.folds_df[(self.folds_df['fold'] == fold) & (self.folds_df['author'].isin(valid_authors))]['author'].tolist()
        train_data_authors = self.folds_df[(self.folds_df['fold'] != fold) & (self.folds_df['author'].isin(valid_authors))]['author'].tolist()

        train_input_authors, train_output = self.get_input_output(train_data_authors, target, number_of_classes, all_output_values, prediction_type)
        test_input_authors, test_output = self.get_input_output(test_data_authors, target, number_of_classes, all_output_values, prediction_type)

        return train_input_authors, train_output, test_input_authors, test_output

    def multitarget_output_resolution(self, target, fold, prediction_type, random_state):
        train_input_authors, train_output, test_input_authors, test_output = set(), [], set(), []
        for target_unit in target: 
            train_input_authors_new, train_output_new, test_input_authors_new, test_output_new = self.singletarget_output_support(target_unit, fold, prediction_type, random_state)
            
            train_input_authors.update(train_input_authors_new)
            train_output = train_output.append(train_output_new)
            test_input_authors.update(test_input_authors_new)
            test_output = test_output.append(test_output_new)

        return train_input_authors, train_output, test_input_authors, test_output

    def get_train_val_test_input_output(self, target, fold, validation_split, random_state, targets_type, prediction_type):
        if targets_type == SINGLE_TARGET:
            train_input_authors, train_output, test_input_authors, test_output = self.singletarget_output_support(target[0], fold, prediction_type, random_state)
        else: 
            train_input_authors, train_output, test_input_authors, test_output = self.multitarget_output_resolution(target, fold, prediction_type, random_state)

        train_input_authors, val_input_authors, train_output, val_output = train_test_split(train_input_authors, train_output, test_size=validation_split, random_state=random_state)
        assert len(train_input_authors) == len(train_output)
        assert len(test_input_authors) == len(test_output)
        assert len(val_input_authors) == len(val_output)
        print(f"Train size: {len(train_input_authors)}")
        print(f"Validation size: {len(val_input_authors)}")
        print(f"Test size: {len(test_input_authors)}")
        delimiter()
        return train_input_authors, train_output, val_input_authors, val_output, test_input_authors, test_output

    def create_minibatches(self, data_X, data_y, minibatch_size, cuda_dev, batch_operator):
        idx = list(range(len(data_X)))
        random.shuffle(idx)
        for idx_list in chunks(idx, minibatch_size):
            data_X_authors = [data_X[index] for index in idx_list]
            data_y_idx = [data_y[index] for index in idx_list]
            self.debugger.print(data_y_idx)
            self.debugger.print(self.input_df.keys())
            self.debugger.print(data_X_authors)
            self.debugger.print(len(data_X_authors))
            self.debugger.print(len(data_y_idx))
            
            minibatch_X = [self.input_df[author] for author in data_X_authors]
            minibatch_X = rnnutils.pad_sequence(minibatch_X, batch_first=True, padding_value = 0) 
            minibatch_X = batch_operator(minibatch_X)
            # minibatch_X = minibatch_X.unsqueeze(1)
            minibatch_y = to.tensor(data_y_idx)
            if cuda_dev is not None:
                minibatch_X = minibatch_X
                minibatch_y = minibatch_y.to(device=cuda_dev)

            yield((minibatch_X, minibatch_y))   

class ModelPerformanceSaver:

    MODELS_PREFORMANCE_COLUMNS_REGRESSION = ['val_mse', 'val_pearson', 'val_r2_score', \
                              'test_mse','test_pearson', 'test_r2_score', 'epoch']

    MODELS_PREFORMANCE_COLUMNS_CLASSIFICATION_MULTICLASS = ['val_f1', 'val_precision_macro', 'val_recall_macro', \
                              'test_f1','test_precision_macro', 'test_recall_macro', 'epoch']

    MODELS_PREFORMANCE_COLUMNS_CLASSIFICATION_BINARY = ['val_f1', 'val_precision_0', 'val_precision_1', 'val_precision_macro' ,\
                              'val_recall_0', 'val_recall_1', 'val_recall_macro', \
                              'test_f1', 'test_precision_0', 'test_precision_1', 'test_precision_macro',\
                              'test_recall_0', 'test_recall_1', 'test_recall_macro',\
                              'epoch']
    MODELS_META_DATA = ['hash_id']
    MODELS_IDENTIFIER_DATA = ['models_name', 'experiments_name', 'trait', 'fold', 'run_identificator']
    
    def __init__(self, debugger, columns, id_columns, save_location, number_of_classes, prediction_type, import_location=None):
        self.debugger = debugger

        preformance_columns = None
        if(prediction_type == CLASSIFICATION):
            preformance_columns = self.MODELS_PREFORMANCE_COLUMNS_CLASSIFICATION_BINARY
            if(number_of_classes > 2):
                preformance_columns = self.MODELS_PREFORMANCE_COLUMNS_CLASSIFICATION_MULTICLASS
        elif(prediction_type == REGRESSION):
            preformance_columns = self.MODELS_PREFORMANCE_COLUMNS_REGRESSION

        self.df = pd.DataFrame(data=None, columns=self.MODELS_IDENTIFIER_DATA+columns+preformance_columns+self.MODELS_META_DATA)
        self.id_columns = id_columns+self.MODELS_IDENTIFIER_DATA
        self.save_location = save_location
        if(import_location != None):
            self.df = pd.read_csv(import_location)

    def get_hash(self, index):
        identifier = hashlib.md5(''.join([str(x) for x in self.df[self.id_columns].iloc[index]]).encode('utf-8')).hexdigest()
        return identifier

    def create_new_entry(self, experiments_name, models_name, trait, fold, run_identificator, column_values):
        new_entry_values = {}
        new_entry_values.update(column_values)
        new_entry_values.update({'hash_id': 'DUMMY_VALUE', 'experiments_name': experiments_name, 'models_name' : models_name, 'trait' : trait,\
             'fold' : fold, 'run_identificator' : run_identificator})
        self.df = self.df.append(new_entry_values, ignore_index=True)
        entries_position = len(self.df)-1
        identifier = self.get_hash(entries_position)
        self.df.at[entries_position, 'hash_id'] = identifier
        return identifier        
        
    def update_models_val_results(self, models_identifier, epoch, preformance_data):
        self.df.loc[self.df['hash_id'] == models_identifier, 'epoch'] = epoch
        for label, value in preformance_data.items():
            self.debugger.print(f"Updating validation for label {label} with value {value}")
            self.df.loc[self.df['hash_id'] == models_identifier, 'val_'+label] = value
   
    def update_models_test_results(self, models_identifier, preformance_data):
        for label, value in preformance_data.items():
            self.debugger.print(f"Updating test for label {label} with value {value}")
            self.df.loc[self.df['hash_id'] == models_identifier, 'test_'+label] = value

    def get_best_models_data(self, target, fold, run_identificator):
        return self.df.loc[self.df[(self.df.mbti_trait == target) & (self.df.fold == fold) & (self.df.run_identificator == run_identificator)]['val_f1'].idxmax()]

    def get_data(self):
        return self.df

    def flush_data(self):
        self.df.to_csv(self.save_location, index=False)
