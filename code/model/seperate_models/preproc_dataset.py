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
    def __init__(self, name, split_dev=False): # name can be : FEVER, SciFact
        self.train_dataset = self.__init_train_data__(name)
        self.val_dataset, self.test_dataset = self.__init_dev_data__(name, split_dev)

    def __init_train_data__(self, name):
        df = pd.read_csv ('../dataset/'+ name +'/train.jsonl')
        self.labels = df['label'].unique()
        self.encoded_labels, self.decoded_labels = encode_labels(self.labels)

        df['label'] = df.label.replace(self.encoded_labels)
        return df

    def __init_dev_data__(self,name, split_dev):
        df = pd.read_csv ('../dataset/'+ name +'/dev.jsonl')
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
        encoded_labels = encode_labels(labels)

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
            header.append(self.decoded_labels.get(l))
            t0.append(count_obj[str(l)+'_train'])
            t1.append(count_obj[str(l)+'_dev'])
            t2.append(count_obj[str(l)+'_test'])

        header.append('Total')
        t0.append(count_train['text'].sum())
        t1.append(count_val['text'].sum())
        t2.append(count_test['text'].sum())

        table = Texttable()
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

if __name__ == '__main__':

    # fever = FeverDataset()
    # train_dataset = fever.get_train_dataset()
    # val_dataset = fever.get_val_dataset()
    # test_dataset = fever.get_test_dataset()
    # fever.get_describtion()
    # fever.print_data_example()

    dataset = Dataset(name='FEVER', split_dev=True)
    dataset.get_describtion()
    dataset.print_data_example()

    dataset = Dataset(name='SciFact', split_dev=True)
    dataset.get_describtion()
    dataset.print_data_example()

    # preprocess_scifact('../dataset/SciFact/train_raw.jsonl', '../dataset/SciFact/train.jsonl')
    # preprocess_scifact('../dataset/SciFact/dev_raw.jsonl', '../dataset/SciFact/dev.jsonl')
    # preprocess_fever('../dataset/FEVER/train_raw.jsonl', '../dataset/FEVER/train.jsonl')
    # preprocess_fever('../dataset/FEVER/dev_raw.jsonl', '../dataset/FEVER/dev.jsonl')
    #
    # df = pd.read_csv ('../dataset/FEVER/dev.jsonl')
    #
    # print(df.head(10))
    # print(df.describe())
