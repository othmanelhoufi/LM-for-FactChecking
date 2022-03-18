#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from sklearn.metrics import classification_report


import preproc_dataset

# disable WANDB
import os
os.environ['WANDB_DISABLED'] = 'true'

# specify GPU
# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer model
MODEL_NAME = 'bert-base-uncased'

# hyperparams
MAX_SEQ_LEN = 128
BATCH_SIZE = 50
LR = 3e-5
EPOCHS = 10

# FEVER dataset
fever = preproc_dataset.FeverDataset()
DATASET = fever.claim_dataset
DATASET.columns = ["text", "label"]
# DATASET = DATASET.sample(150)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Preparing train data and eval data
def split_dataset(dataset=DATASET):
    # Preprocess data
    X = list(dataset['text'])
    y = list(dataset['label'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    return X_train, y_train, X_val, y_val


# Import BERT Model and BERT Tokenizer
def init_bert_model(model_name=MODEL_NAME):
    # import BERT-base pretrained model
    bert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return bert, tokenizer



# Tokenization
def init_tokens(X_train, X_val, max_seq_len=MAX_SEQ_LEN):
    # tokenize and encode sequences in the training set
    tokens_train = tokenizer(X_train, padding=True, truncation=True, max_length=max_seq_len)

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer(X_val, padding=True, truncation=True, max_length=max_seq_len)

    return tokens_train, tokens_val


# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall_micro = recall_score(y_true=labels, y_pred=pred, average='micro')
    recall_macro = recall_score(y_true=labels, y_pred=pred, average='macro')
    recall_weighted = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision_micro = precision_score(y_true=labels, y_pred=pred, average='micro')
    precision_macro = precision_score(y_true=labels, y_pred=pred, average='macro')
    precision_weighted = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1_micro = f1_score(y_true=labels, y_pred=pred, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=pred, average='macro')
    f1_weighted = f1_score(y_true=labels, y_pred=pred, average='weighted')

    return {"accuracy": accuracy,
            "precision_micro": precision_micro, "precision_macro": precision_macro, "precision_weighted": precision_weighted,
            "recall_micro": recall_micro, "recall_macro": recall_macro, "recall_weighted": recall_weighted,
            "f1_micro": f1_micro, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


if __name__ == '__main__':

    train_text, train_labels, val_text, val_labels = split_dataset(DATASET)
    bert, tokenizer = init_bert_model(MODEL_NAME)
    tokens_train, tokens_val = init_tokens(train_text, val_text, MAX_SEQ_LEN)

    train_dataset = Dataset(tokens_train, train_labels)
    val_dataset = Dataset(tokens_val, val_labels)


    # Define Trainer
    args = TrainingArguments(
        output_dir="outputs/" + MODEL_NAME,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=0,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=bert,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train pre-trained model
    # trainer.train()

    # Load trained model
    model_path = "outputs/checkpoint-12000"
    bert = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    # Define test trainer
    test_trainer = Trainer(bert)
    # Make prediction
    raw_pred, _, _ = test_trainer.predict(val_dataset)
    # Preprocess raw predictions
    y_pred = np.argmax(raw_pred, axis=1)

    labels = {'REFUTES':0, 'SUPPORTS':1, 'NOT ENOUGH INFO':2}
    print(classification_report(val_labels, y_pred, target_names=labels))
