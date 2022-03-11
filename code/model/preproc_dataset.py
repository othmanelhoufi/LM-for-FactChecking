#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class FeverDataset:
    def __init__(self):
        self.dataset = self.__init_data__()
        self.claim_dataset = self.__init_claim_dataset__()
        self.evidence_dataset = self.__init_evidence_dataset__()

    def __init_data__(self):
        relevant_col = ['claim', 'label', 'evidence']
        df_raw = pd.read_json('../dataset/feverous_train_challenges.jsonl', lines=True)
        df_rel = df_raw.filter(items=relevant_col)

        nan_value = float("NaN")
        df_rel.replace("", nan_value, inplace=True)
        df_rel.dropna(inplace=True)

        encoded_dict = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}
        df_rel['label'] = df_rel.label.replace(encoded_dict)

        return df_rel

    def __init_claim_dataset__(self):
        return self.dataset.filter(items=['claim', 'label'])

    def __init_evidence_dataset__(self):
        return self.dataset.filter(items=['evidence', 'label'])

    def __get_dataset__(self):
        return self.dataset

    def get_train_val(self, df, balance=True):
        if balance is True:
            # supports_cnt = df[df['label'] == 1].count().label
            # refutes_cnt = df[df['label'] == 0].count().label
            nei_cnt = df[df['label'] == 2].count().label

            nei_subset = df.loc[df["label"] == 2, :]
            refutes_subset = df.loc[df["label"] == 0, :]
            supports_subset = df.loc[df["label"] == 1, :]
            sampled_supports = supports_subset.sample(n=nei_cnt, random_state=1)
            sampled_refutes = refutes_subset.sample(n=nei_cnt, random_state=1)
            df = pd.concat([nei_subset, sampled_supports, sampled_refutes], ignore_index=True)
            df = shuffle(df, random_state=0)

        X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.15,
                                                  random_state=42,
                                                  stratify=df.label.values)

        df['data_type'] = ['not_set']*df.shape[0]

        df.loc[X_train, 'data_type'] = 'train'
        df.loc[X_val, 'data_type'] = 'val'

        cnt = df.groupby(['label', 'data_type']).count()
        # print(cnt)

        return df

    def get_describtion(self):

        val_count = self.dataset['label'].value_counts()
        print(val_count)
