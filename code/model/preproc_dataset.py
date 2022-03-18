#!/usr/bin/env python3

import pandas as pd
from texttable import Texttable


class FeverDataset:
    def __init__(self, with_evidence=False):
        self.train_dataset = self.__init_train_data__()
        self.val_dataset, self.test_dataset = self.__init_dev_data__()

        if with_evidence:
            self.evidence_dataset = self.__init_evidence_dataset__()

    def __init_train_data__(self):
        relevant_col = ['claim', 'label']
        df_raw = pd.read_json('../dataset/train.jsonl', lines=True)
        df_rel = df_raw.filter(items=relevant_col)

        nan_value = float("NaN")
        df_rel.replace("", nan_value, inplace=True)
        df_rel.dropna(inplace=True)

        encoded_dict = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}
        df_rel['label'] = df_rel.label.replace(encoded_dict)
        df_rel.columns = ["text", "label"]
        return df_rel

    def __init_dev_data__(self):
        relevant_col = ['claim', 'label']
        df_raw = pd.read_json('../dataset/dev.jsonl', lines=True)
        df_rel = df_raw.filter(items=relevant_col)

        nan_value = float("NaN")
        df_rel.replace("", nan_value, inplace=True)
        df_rel.dropna(inplace=True)

        encoded_dict = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}
        df_rel['label'] = df_rel.label.replace(encoded_dict)
        df_rel.columns = ["text", "label"]

        half_df = len(df_rel) // 2
        df_val = df_rel[:half_df]
        df_test = df_rel[half_df:]
        return df_val, df_test

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_describtion(self):
        print("\n**************** Partition counts of different datasets ****************\n")

        count_train = self.train_dataset.groupby('label').count()
        count_val = self.val_dataset.groupby('label').count()
        count_test = self.test_dataset.groupby('label').count()

        supports_count_train = count_train['text'][1]
        refutes_count_train = count_train['text'][0]
        nei_count_train = count_train['text'][2]

        supports_count_val = count_val['text'][1]
        refutes_count_val = count_val['text'][0]
        nei_count_val = count_val['text'][2]

        supports_count_test = count_test['text'][1]
        refutes_count_test = count_test['text'][0]
        nei_count_test = count_test['text'][2]


        table = Texttable()
        table.header(["Split", "SUPPORTS", "REFUTES", "NEI"])
        table.add_row(["Training", supports_count_train, refutes_count_train, nei_count_train])
        table.add_row(["Validation", supports_count_val, refutes_count_val, nei_count_val])
        table.add_row(["Testing", supports_count_test, refutes_count_test, nei_count_test])

        print(table.draw())

    def print_data_example(self):
        print("\n**************** Random examples from Training dataset ****************\n")
        df_sample = self.train_dataset.sample(5)
        df_sample = df_sample.reset_index()  # make sure indexes pair with number of rows
        table = Texttable()
        table.header(["ID", "TEXT", "LABEL"])
        encoded_dict = {0:'REFUTES', 1:'SUPPORTS', 2:'NOT ENOUGH INFO'}
        for index, row in df_sample.iterrows():
            label = encoded_dict.get(row['label'])
            table.add_row([row['index'], row['text'], label])

        print(table.draw())


if __name__ == '__main__':

    fever = FeverDataset()
    train_dataset = fever.get_train_dataset()
    val_dataset = fever.get_val_dataset()
    test_dataset = fever.get_test_dataset()
    fever.get_describtion()
    fever.print_data_example()
