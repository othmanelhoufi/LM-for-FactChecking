#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
from texttable import Texttable

# Encode labels: from string to int
def encode_labels(labels):
    encoded_labels = {}
    decoded_labels = {}
    i = 0
    for l in labels:
        encoded_labels[str(l)] = i
        decoded_labels[i] = str(l)
        i = i + 1
    return encoded_labels, decoded_labels

class Dataset:
    def __init__(self, name, split_dev=False, num_labels=2): # name can be : FEVER, SciFact, Liar...
        self.train_dataset = self.__init_train_data__(name, num_labels)
        self.val_dataset, self.test_dataset = self.__init_dev_data__(name, num_labels, split_dev)

    def __init_train_data__(self, name, num_labels):
        df = pd.read_csv ('../dataset/'+ name +'/train-' + str(num_labels) +'L.jsonl', dtype={'label': str})
        self.labels = df['label'].unique()
        self.encoded_labels, self.decoded_labels = encode_labels(self.labels)

        df['label'] = df.label.replace(self.encoded_labels)
        return df

    def __init_dev_data__(self,name, num_labels, split_dev):
        df = pd.read_csv ('../dataset/'+ name +'/dev-' + str(num_labels) +'L.jsonl', dtype={'label': str})
        df['label'] = df.label.replace(self.encoded_labels)
        if split_dev:
            half_df = len(df) // 2
            df_val = df[:half_df]
            df_test = df[half_df:]
            return df_val, df_test

        return df, df

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_encodings(self):
        return self.labels, self.encoded_labels, self.decoded_labels

    def get_describtion(self):
        print("\n**************** Partition counts of different datasets ****************\n")

        count_train = self.train_dataset.groupby('label').count()
        count_val = self.val_dataset.groupby('label').count()
        count_test = self.test_dataset.groupby('label').count()

        labels = self.val_dataset['label'].unique()

        count_obj = {}
        for l in labels:
            count_obj[str(l)+'_train'] = count_train['text'][l]
            count_obj[str(l)+'_dev'] = count_val['text'][l]
            count_obj[str(l)+'_test'] = count_test['text'][l]

        header = ['Split']
        t0 = ['Training']
        t1 = ['Validation']
        t2 = ['Testing']

        for l in labels:
            header.append(self.decoded_labels[l])
            t0.append(count_obj[str(l)+'_train'])
            t1.append(count_obj[str(l)+'_dev'])
            t2.append(count_obj[str(l)+'_test'])

        header.append('Total')
        t0.append(count_train['text'].sum())
        t1.append(count_val['text'].sum())
        t2.append(count_test['text'].sum())

        table = Texttable()
        table.set_max_width(0)
        table.header(header)
        table.add_row(t0)
        table.add_row(t1)
        table.add_row(t2)

        print(table.draw())

    def print_data_example(self):
        print("\n**************** Random examples from Training dataset ****************\n")
        df_sample = self.train_dataset.sample(5)
        df_sample = df_sample.reset_index()  # make sure indexes pair with number of rows
        table = Texttable()
        table.header(["ID", "CLAIM", "LABEL"])
        for index, row in df_sample.iterrows():
            label = self.decoded_labels.get(row['label'])
            table.add_row([row['index'], row['text'], label])

        print(table.draw())


def preprocess_fever(data_path, export_path):
    relevant_col = ['claim', 'label']
    df_raw = pd.read_json(data_path, lines=True)
    df = df_raw.filter(items=relevant_col)

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.columns = ['text', 'label']
    df.dropna(inplace=True)

    # print(df.head(10))
    # print(df['label'].unique())
    #
    # df = df[(df.label == 'SUPPORTS') | (df.label == 'REFUTES')]
    # print(df.head(10))
    # print(df['label'].unique())
    df.to_csv(export_path, index=False)


def preprocess_scifact(data_path, export_path):
    relevant_col = ['claim', 'evidence']
    df_raw = pd.read_json(data_path, lines=True)
    df = df_raw.filter(items=relevant_col)

    nan_value = float("NaN")
    df = df.reset_index()  # make sure indexes pair with number of rows
    for index, row in df.iterrows():
        row = str(row['evidence'])
        if 'SUPPORT' in row:
            df.at[index, 'evidence'] = 'SUPPORT'
        elif 'CONTRADICT' in row:
            df.at[index, 'evidence'] = 'CONTRADICT'
        else:
            df.at[index, 'evidence'] = nan_value

    df = df.filter(items=relevant_col)
    df.columns = ['text', 'label']
    df.dropna(inplace=True)
    df.to_csv(export_path, index=False)


def preprocess_multifc(data_path, export_path):

    ############## preprocessing dataset and maping labels to new space ##############
    df = pd.read_csv(data_path, sep='\t')
    df.columns = [str(i) for i in range(0, 13)]
    df = df[['1','2']]
    df.columns = ['claim', 'label']

    true_set = ['true', 'truth!', 'promise kept', 'true messages', 'determination: true', 'verdict: true', 'confirmed authorship!', 'verdict: true', 'authorship confirmed!',
                'in-the-green', 'truth! & outdated!', 'correct', 'factscan score: true', 'fact', 'accurate', 'correct attribution!', 'correct attribution', 'conclusion: accurate']
    false_set = ['false', 'full flop', 'pants on fire!', 'determination: misleading', 'bogus warning', 'determination: false', 'incorrect', 'compromise', 'a lot of baloney', 'facebook scams',
                 'promise broken', 'misleading', 'verdict: false', 'unsubstantiated messages', 'in-the-red', 'verdict: unsubstantiated', 'factscan score: false', 'miscaptioned', 'scam!',
                 'fake news', 'fake', 'scam', 'unsupported', 'factscan score: misleading', 'rating: false', 'determination: huckster propaganda', 'inaccurate attribution!', 'incorrect attribution!',
                 'virus!', 'distorts the facts', 'we rate this claim false', 'exaggerates']
    mostly_true = ['determination: mostly true', 'mostly true', 'mostly truth!', 'mostly-correct', 'mostly_true', 'a little baloney']
    mostly_false = ['mostly false', 'determination: barely true', 'mostly_false']
    mixture = ['half-true', 'mixture', 'in-between', 'not the whole story', 'half true', 'some baloney', 'half flip', 'partly true', 'truth! & fiction!']
    fiction = ['fiction!', 'mostly fiction!', 'fiction! & satire!', 'fiction']

    def map_label_values(v):
        if v in true_set : return 'TRUE'
        elif v in false_set : return 'FALSE'
        elif v in mostly_true : return 'MOSTLY-TRUE'
        elif v in mostly_false : return 'MOSTLY-FALSE'
        elif v in mixture : return 'IN-BETWEEN'
        elif v in fiction : return 'FALSE'
        else: return float("NaN")


    df['label'] = df['label'].map(lambda x: map_label_values(x))
    #####################################################################################

    nan_value = float("NaN")
    df.columns = ['text', 'label']
    df.dropna(inplace=True)

    # print(df.head(10))
    # print(df['label'].unique())

    # df = df[(df.label == 'TRUE') | (df.label == 'FALSE') | (df.label == 'MOSTLY-TRUE') | (df.label == 'MOSTLY-FALSE') | (df.label == 'IN-BETWEEN') ]
    # print(df.head(10))
    # print(df['label'].unique())

    df.to_csv(export_path, index=False)



def preprocess_liar(data_path, export_path):
    df = pd.read_csv(data_path, sep='\t')
    df.columns = [str(i) for i in range(0, 14)]
    df = df[['1','2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].str.upper()
    columns_titles = ['text','label']
    df=df.reindex(columns=columns_titles)

    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(inplace=True)

    #====================================================================#
    #====================Maping labels to new space======================#
    #====================================================================#
    def map_label_values(v):
        if v == 'MOSTLY-TRUE' : return 'MOSTLY-TRUE'
        elif v == 'HALF-TRUE' : return 'HALF-TRUE'
        elif v == 'BARELY-TRUE' : return 'BARELY-TRUE'
        elif v == 'PANTS-FIRE' : return 'PANTS-FIRE'
        else: return v


    df['label'] = df['label'].map(lambda x: map_label_values(x))
    #====================================================================#

    df.to_csv(export_path, index=False)

if __name__ == '__main__':

    # preprocess_scifact('../dataset/SciFact/train_raw.jsonl', '../dataset/SciFact/train.jsonl')
    # preprocess_scifact('../dataset/SciFact/dev_raw.jsonl', '../dataset/SciFact/dev.jsonl')

    # preprocess_fever('../dataset/FEVER/train_raw.jsonl', '../dataset/FEVER/train-2L.jsonl')
    # preprocess_fever('../dataset/FEVER/dev_raw.jsonl', '../dataset/FEVER/dev-2L.jsonl')

    # preprocess_multifc('../dataset/MultiFC/train_raw.tsv', '../dataset/MultiFC/train-5L.jsonl')
    # preprocess_multifc('../dataset/MultiFC/dev_raw.tsv', '../dataset/MultiFC/dev-5L.jsonl')

    # preprocess_liar('../dataset/Liar/train_raw.tsv', '../dataset/Liar/train-6L.jsonl')
    # preprocess_liar('../dataset/Liar/dev_raw.tsv', '../dataset/Liar/dev-6L.jsonl')

    # dataset = Dataset(name='FEVER', split_dev=True, num_labels=3)
    # dataset.get_describtion()
    # dataset.print_data_example()


    pass
